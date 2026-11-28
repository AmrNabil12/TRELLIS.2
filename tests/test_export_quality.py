import os
import tempfile
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from runpod.server import _prepare_export_source, _prefer_quad_diagonal_02


class ExportQualityTests(unittest.TestCase):
    def test_concave_quad_in_either_winding(self):
        quad = torch.tensor([[0., 0., 0.], [2., 0., 0.], [.5, .5, 0.], [0., 2., 0.]])
        points = torch.stack([quad, quad.roll(1, 0), quad.flip(0)])
        self.assertEqual(_prefer_quad_diagonal_02(points, torch).tolist(), [True, False, False])

    def test_full_source_unless_cap_explicit(self):
        mesh = SimpleNamespace(faces=range(169828610), simplify=Mock())
        with patch.dict(os.environ, {"TRELLIS2_PRE_REMESH_MAX_FACES": "0"}):
            _prepare_export_source(mesh, True, torch)
        mesh.simplify.assert_not_called()
        with patch.dict(os.environ, {"TRELLIS2_PRE_REMESH_MAX_FACES": "1000"}):
            _prepare_export_source(mesh, True, torch)
        mesh.simplify.assert_called_once_with(1000)

    def test_projection_ignores_invalid_bvh_face(self):
        from runpod.server import _project_bake_positions

        pts = torch.zeros((2, 3))
        vertices = torch.ones((3, 3))
        faces = torch.tensor([[0, 1, 2]])
        bvh = Mock()
        bvh.unsigned_distance.return_value = (None, torch.tensor([-1, 0]), torch.tensor([[1., 0., 0.]] * 2))
        projected = _project_bake_positions(pts, bvh, vertices, faces, 1., 2., torch)
        torch.testing.assert_close(projected[0], pts[0])
        torch.testing.assert_close(projected[1], vertices[0])

    def test_material_projection_rejects_sharp_surface_crossings(self):
        from runpod.server import _project_bake_positions

        vertices = torch.tensor([[0., 0., 1.], [1., 0., 1.], [0., 1., 1.]])
        faces = torch.tensor([[0, 1, 2]])
        points = torch.zeros((3, 3))
        normals = torch.tensor([[1., 0., 0.], [0., 0., 1.], [0., 0., -1.]])
        bvh = Mock()
        bvh.unsigned_distance.return_value = (None, torch.zeros(3, dtype=torch.long),
                                              torch.tensor([[1., 0., 0.]] * 3))
        projected = _project_bake_positions(points, bvh, vertices, faces, 1., 2., torch, normals)
        torch.testing.assert_close(projected[0], points[0])
        torch.testing.assert_close(projected[1:], vertices[0].expand(2, 3))

    def test_missing_material_samples_follow_surface(self):
        import cv2
        import numpy as np
        from runpod.server import _extend_material_atlas

        atlas = np.zeros((1, 9, 2), dtype=np.uint8)
        atlas[0, 0] = (240, 0)
        atlas[0, 5] = (0, 250)
        covered = np.array([[1, 1, 1, 1, 0, 1, 0, 0, 1]], dtype=bool)
        supported = np.array([[1, 0, 0, 0, 0, 1, 0, 0, 0]], dtype=bool)
        xyz = torch.tensor([[0., 0, 0], [1., 0, 0], [2., 0, 0], [3., 0, 0],
                            [20., 0, 0], [0.1, 0, 0]])
        result = _extend_material_atlas(atlas, covered, supported, xyz, cv2, np)
        for pixel in (0, 1, 2, 3, 8):
            np.testing.assert_array_equal(result[0, pixel], [240, 0])
        np.testing.assert_array_equal(result[0, 5], [0, 250])
        self.assertTrue(all(tuple(row) in {(240, 0), (0, 250)} for row in result[0]))

    def test_low_vram_to_glb_cleans_post_simplify_and_recovers_voxel_misses(self) -> None:
        import types
        import cv2
        import numpy as np
        import torch
        from PIL import Image
        import runpod.server as srv

        calls: list[str] = []
        created_meshes: list[object] = []
        sample_counts: list[int] = []

        class FakeCuMesh:
            def __init__(self) -> None:
                self.v = None
                self.f = None

            def init(self, v, f) -> None:
                self.v = v.clone()
                self.f = f.clone()
                created_meshes.append(self)

            @property
            def num_vertices(self) -> int:
                return int(self.v.shape[0])

            @property
            def num_faces(self) -> int:
                return int(self.f.shape[0])

            def fill_holes(self, max_hole_perimeter=3e-2) -> None:
                calls.append(("fill_holes", max_hole_perimeter))

            def read(self):
                return self.v, self.f

            def simplify(self, target: int, verbose: bool = False) -> None:
                calls.append("simplify")

            def remove_degenerate_faces(self) -> None:
                calls.append("remove_degenerate_faces")

            def remove_duplicate_faces(self) -> None:
                calls.append("remove_duplicate_faces")

            def remove_small_connected_components(self, min_area: float) -> None:
                calls.append(("remove_small_connected_components", min_area))

            def repair_non_manifold_edges(self) -> None:
                calls.append("repair_non_manifold_edges")

            def uv_unwrap(self, **kwargs):
                calls.append("uv_unwrap")
                uvs = torch.tensor([[0.25, 0.25], [0.75, 0.25], [0.25, 0.75]], dtype=torch.float32)
                vmaps = torch.tensor([0, 1, 2], dtype=torch.int64)
                return self.v, self.f, uvs, vmaps

            def compute_vertex_normals(self) -> None:
                pass

            def read_vertex_normals(self):
                return torch.tensor([[0.0, 0.0, 1.0]] * 3, dtype=torch.float32)

        projection_counts = []

        class FakeBvh:
            def __init__(self, v, f) -> None:
                pass

            def unsigned_distance(self, pts, return_uvw=False):
                n = int(pts.shape[0])
                projection_counts.append(n)
                d = torch.zeros((n,), dtype=torch.float32, device=pts.device)
                fid = torch.zeros((n,), dtype=torch.int64, device=pts.device)
                uvw = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32, device=pts.device).repeat(n, 1)
                return (d, fid, uvw) if return_uvw else (d, fid, None)

        class FakeRastCtx:
            pass

        def fake_rasterize(ctx, uvs, faces, resolution):
            h, w = resolution
            out = torch.zeros((1, h, w, 4), dtype=torch.float32, device=uvs.device)
            out[0, 1:4, 1:4, 3] = 1.0  # 9 covered texels
            return out, None

        def fake_interpolate(verts, rast, faces):
            h, w = rast.shape[1], rast.shape[2]
            pos = torch.zeros((1, h, w, 3), dtype=torch.float32, device=verts.device)
            pos[0, 1, 1, 0] = -0.01  # Farther than the projection limit
            return pos, None

        def fake_grid_sample_3d(attr_vol, coords, shape, grid, mode="trilinear"):
            n_q = int(grid.shape[1])
            sample_counts.append(n_q)
            self.assertTrue(bool((attr_vol[:, 6] == 1).all()))
            out = torch.zeros((1, n_q, 7), dtype=torch.float32, device=grid.device)
            if n_q == 9 and float(grid[0, 1, 0]) > 768.0:
                out[0, 1:, :3] = 0.95
                out[0, 1:, 5:] = 1.0
            elif n_q == 2:
                self.assertTrue(torch.allclose(grid[..., 0], torch.full((1, 2), 768.25, device=grid.device)))
                out[0, :, :3] = 0.95
                out[0, :, 5:] = 1.0
            else:
                self.assertAlmostEqual(float(grid[0, 0, 0].item()), 752.64, places=3)
                self.assertAlmostEqual(float(grid[0, 1, 0].item()), 768.0, places=3)
                out[0, :2, :3] = 0.2
                out[0, :2, 6] = 1.0  # Valid opacity-zero material
                out[0, 4:, 0] = 0.80
                out[0, 4:, 1] = 0.65
                out[0, 4:, 2] = 0.55
                out[0, 4:, 5:] = 1.0
            return out

        fake_dr = types.SimpleNamespace(
            RasterizeCudaContext=FakeRastCtx,
            rasterize=fake_rasterize,
            interpolate=fake_interpolate,
        )
        fake_remeshing = types.SimpleNamespace(
            remesh_narrow_band_dc=lambda v, f, **kw: (v, f),
        )
        fake_cumesh = types.SimpleNamespace(
            CuMesh=FakeCuMesh,
            cuBVH=FakeBvh,
            remeshing=fake_remeshing,
        )

        class FakePBRMaterial:
            def __init__(self, *a, **kw) -> None:
                self.__dict__.update(kw)

        class FakeTexVisuals:
            def __init__(self, *a, **kw) -> None:
                self.__dict__.update(kw)

        class FakeTrimesh:
            def __init__(self, *a, **kw) -> None:
                self.__dict__.update(kw)

        def _marker_to_glb(*a, **kw):
            # Presence of 'remesh_narrow_band_dc' lets patch_o_voxel_postprocess recognize upstream to_glb
            _ = "remesh_narrow_band_dc"
            return None

        fake_trimesh = types.SimpleNamespace(
            Trimesh=FakeTrimesh,
            visual=types.SimpleNamespace(
                material=types.SimpleNamespace(PBRMaterial=FakePBRMaterial),
                TextureVisuals=FakeTexVisuals,
            ),
        )
        fake_postprocess = types.SimpleNamespace(
            np=np,
            cv2=cv2,
            cumesh=fake_cumesh,
            dr=fake_dr,
            grid_sample_3d=fake_grid_sample_3d,
            trimesh=fake_trimesh,
            Image=Image,
            tqdm=None,
            to_glb=_marker_to_glb,
        )
        srv.patch_o_voxel_postprocess(fake_postprocess, torch_mod=torch)
        self.assertTrue(getattr(fake_postprocess, "_trellis2_postprocess_patched", False))
        v_in = torch.tensor([[0.25 / 1536, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]], dtype=torch.float32)
        f_in = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        attrs = torch.ones((2, 6), dtype=torch.float32)
        coords = torch.zeros((2, 3), dtype=torch.int32)
        layout = {"base_color": slice(0, 3), "metallic": slice(3, 4), "roughness": slice(4, 5), "alpha": slice(5, 6)}
        def bake(**options):
            with patch.object(torch.cuda, "is_available", return_value=False):
                return fake_postprocess.to_glb(
                    vertices=v_in,
                    faces=f_in,
                    attr_volume=attrs,
                    coords=coords,
                    attr_layout=layout,
                    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                    grid_size=1536,
                    decimation_target=100,
                    texture_size=8,
                    remesh=True,
                    verbose=False,
                    **options,
                )

        res_mesh = bake()

        # Preserve open rims and small components through remeshing and UV unwrap.
        simp_idx = calls.index("simplify")
        post_simp = calls[simp_idx + 1 : calls.index("uv_unwrap")]
        self.assertEqual(
            post_simp,
            [
                "remove_degenerate_faces",
                "remove_duplicate_faces",
                ("fill_holes", 12 / 1536),
            ],
        )
        self.assertEqual(len(created_meshes), 1)
        self.assertEqual(sample_counts, [9, 2])
        self.assertEqual(projection_counts, [2])
        tex_arr = np.asarray(res_mesh.visual.material.baseColorTexture)
        roi = tex_arr[1:4, 1:4]
        np.testing.assert_array_equal(roi[0, :2], [[51, 51, 51, 0], [51, 51, 51, 0]])
        np.testing.assert_array_equal(roi[0, 2], [242, 242, 242, 255])
        np.testing.assert_array_equal(roi[1, 0], [242, 242, 242, 255])

        with tempfile.TemporaryDirectory() as temp_dir:
            diagnostic = Path(temp_dir) / "material-audit.npz"
            bake(material_audit_path=diagnostic)
            with np.load(diagnostic) as audit:
                self.assertEqual(int(audit["coverage"].sum()), 9)
                self.assertEqual(int(audit["direct_valid"].sum()), 7)
                self.assertEqual(int(audit["source_valid"].sum()), 8)
                self.assertTrue(audit["was_projected"][2])
                np.testing.assert_array_equal(audit["source_rgb"][2], [242, 242, 242])
