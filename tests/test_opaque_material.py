import unittest

import numpy as np

from runpod.opaque_material import pad_material_edges, recover_opaque_samples


class OpaqueMaterialTests(unittest.TestCase):
    def test_repairs_across_uv_charts_and_preserves_opaque_dark_marks(self):
        y, x = np.indices((16, 16))
        xyz = np.stack((x, y, x * 0), axis=-1).astype(float)
        rgb = np.full((16, 16, 3), 150, dtype=np.uint8)
        alpha = np.full((16, 16), 255, dtype=np.uint8)
        metal = np.full((16, 16), 90, dtype=np.uint8)
        faces = np.zeros((16, 16), dtype=int)
        normals = np.array([[0., 0., 1.], [0., 0., -1.]])
        covered = np.ones_like(alpha, dtype=bool)
        for col in [1, 3, 5, 7]:
            rgb[1, col] = 0
            metal[1, col] = 2
        alpha[1, [1, 3, 5]] = 80
        faces[1, 3] = 1
        xyz[1, 5] += 100
        self.assertEqual(recover_opaque_samples((rgb, metal, alpha), covered, xyz, faces, normals, 1), 1)
        self.assertEqual(rgb[1, 1].tolist(), [150] * 3)
        self.assertEqual(metal[1, 1], 90)
        self.assertTrue((rgb[1, [3, 5, 7]] == 0).all())

    def test_padding_copies_complete_sample_and_preserves_covered_pixels(self):
        pixels = np.zeros((12, 12, 4), dtype=np.uint8)
        covered = np.zeros((12, 12), dtype=bool)
        covered[1, 1] = True
        pixels[1, 1] = [23, 15, 80, 255]
        pad_material_edges((pixels,), covered)
        self.assertEqual(pixels[1, 2].tolist(), [23, 15, 80, 255])
        self.assertEqual(pixels[1, 1].tolist(), [23, 15, 80, 255])
        self.assertFalse(pixels[-1, -1].any())
