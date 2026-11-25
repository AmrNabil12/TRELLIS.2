"""FastAPI inference server for TRELLIS.2 on RunPod.

Exposes asynchronous job submission (recommended behind RunPod's ~100s HTTP
proxy timeout) as well as a synchronous convenience endpoint and a /health check.
Torch and TRELLIS.2 imports are deferred so the API layer can be imported and
tested without CUDA installed.
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,garbage_collection_threshold:0.8"
    if os.name != "nt"
    else "garbage_collection_threshold:0.8",
)


def _configure_persistent_kernel_caches() -> None:
    """Point Triton and FlexGEMM autotune caches to persistent RunPod storage before imports."""
    storage_root: Path | None = None
    for candidate in ("/runpod-volume", "/workspace"):
        p = Path(candidate)
        if p.is_dir() and os.access(p, os.W_OK):
            storage_root = p
            break
    cache_base = storage_root if storage_root is not None else Path(tempfile.gettempdir())
    triton_dir = Path(os.environ.setdefault("TRITON_CACHE_DIR", str(cache_base / "triton_cache")))
    try:
        triton_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    os.environ.setdefault(
        "FLEX_GEMM_AUTOTUNE_CACHE_PATH",
        str(cache_base / "flex_gemm_autotune_cache.json"),
    )
    os.environ.setdefault("FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE", "1")


import gc
import inspect
import io
import itertools
import json
import logging
import re
import secrets
import shutil
import tempfile
import threading
import time
import uuid
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

_configure_persistent_kernel_caches()

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Response, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

_JOB_ID_RE = re.compile(r"^[0-9a-f]{12}$")

log = logging.getLogger("trellis2.server")
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

PipelineType = Literal["512", "1024", "1024_cascade", "1536_cascade"]

DEFAULT_MODEL_ID = "microsoft/TRELLIS.2-4B"
MAX_RETAINED_JOBS = int(os.environ.get("MAX_RETAINED_JOBS", "20"))
HIGH_VRAM_THRESHOLD_GB = 44.0
VRAM_SAFETY_FLOOR_GB = max(2.0, float(os.environ.get("TRELLIS2_VRAM_RESERVE_GB", "4.0")))
MAX_INPUT_DIMENSION = max(256, int(os.environ.get("TRELLIS2_MAX_INPUT_DIM", "1536")))
DEFAULT_VOXEL_CHUNK_SIZE = max(16_384, int(os.environ.get("TRELLIS2_VOXEL_CHUNK", "262144")))
FAST_VOXEL_CHUNK_MID = 524_288
FAST_VOXEL_CHUNK_HIGH = 1_048_576
FAST_VOXEL_CHUNK_ULTRA = 2_097_152
DEFAULT_TOKEN_CHUNK_SIZE = max(4_096, int(os.environ.get("TRELLIS2_TOKEN_CHUNK", "16384")))
FAST_TOKEN_CHUNK_HIGH = 262_144
OOM_FALLBACK_CHAIN: dict[str, str] = {
    "1536_cascade": "1024_cascade",
    "1024_cascade": "1024",
}


class GenerationParams(BaseModel):
    """Validated parameters for TRELLIS.2 generation and GLB post-processing."""

    pipeline_type: PipelineType = "1024_cascade"
    seed: int = 42
    decimation_target: int = Field(default=300_000, ge=1_000, le=2_000_000)
    texture_size: int = Field(default=2048, ge=512, le=4096)
    remesh: bool = True
    remesh_band: float = Field(default=1.0, ge=0.0)
    remesh_project: float = Field(default=0.0, ge=0.0, le=1.0)
    preprocess_image: bool = True
    extension_webp: bool = False


def parse_generation_params(
    pipeline_type: PipelineType = Form("1024_cascade"),
    seed: int = Form(42),
    decimation_target: int = Form(300_000, ge=1_000, le=2_000_000),
    texture_size: int = Form(2048, ge=512, le=4096),
    remesh: bool = Form(True),
    remesh_band: float = Form(1.0, ge=0.0),
    remesh_project: float = Form(0.0, ge=0.0, le=1.0),
    preprocess_image: bool = Form(True),
    extension_webp: bool = Form(False),
) -> GenerationParams:
    """FastAPI dependency parsing multipart form fields into GenerationParams."""
    return GenerationParams(
        pipeline_type=pipeline_type,
        seed=seed,
        decimation_target=decimation_target,
        texture_size=texture_size,
        remesh=remesh,
        remesh_band=remesh_band,
        remesh_project=remesh_project,
        preprocess_image=preprocess_image,
        extension_webp=extension_webp,
    )


class InferenceEngine(Protocol):
    """Engine interface used by endpoints and job workers."""

    status: str
    stage: str
    error: str | None
    model_id: str
    gpu_name: str | None
    vram_gb: float | None
    low_vram: bool | None

    def start_loading(self) -> None: ...
    def health_info(self) -> dict[str, Any]: ...
    def generate_glb(self, image: Image.Image, params: GenerationParams, out_path: Path) -> Path: ...


def _resolve_low_vram(setting: str, vram_gb: float | None) -> bool:
    normalized = setting.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    # "auto": disable offloading on 48GB+ GPUs (e.g. L40S, A100-80GB, H100), enable on 24GB cards
    if vram_gb is not None and vram_gb >= HIGH_VRAM_THRESHOLD_GB:
        return False
    return True


def _cuda_free_gb(torch_mod: Any) -> float | None:
    """Return currently free GPU memory in GiB (driver-level), or None if unavailable."""
    if torch_mod is None or not torch_mod.cuda.is_available():
        return None
    mem_get_info = getattr(torch_mod.cuda, "mem_get_info", None)
    if not callable(mem_get_info):
        return None
    try:
        free_bytes, _total_bytes = mem_get_info(0)
        return free_bytes / (1024**3)
    except Exception:
        return None


def _effective_voxel_chunk(torch_mod: Any, base_chunk: int) -> int:
    """Scale voxel chunk size up to 2M rows above the 4 GiB safety floor to maximize Tensor Core occupancy."""
    if base_chunk != DEFAULT_VOXEL_CHUNK_SIZE or "TRELLIS2_VOXEL_CHUNK" in os.environ:
        return base_chunk
    if os.environ.get("TRELLIS2_FAST_VRAM", "1") == "0":
        return base_chunk
    free_gb = _cuda_free_gb(torch_mod)
    if free_gb is None:
        return base_chunk
    if free_gb >= VRAM_SAFETY_FLOOR_GB + 4.0:
        return max(base_chunk, FAST_VOXEL_CHUNK_ULTRA)
    if free_gb >= VRAM_SAFETY_FLOOR_GB + 2.0:
        return max(base_chunk, FAST_VOXEL_CHUNK_HIGH)
    if free_gb >= VRAM_SAFETY_FLOOR_GB + 0.8:
        return max(base_chunk, FAST_VOXEL_CHUNK_MID)
    return base_chunk


def _effective_token_chunk(torch_mod: Any, base_chunk: int) -> int:
    """Run DiT SparseFeedForwardNet in a single unchunked cuBLAS GEMM (up to 262K tokens) above 4 GiB reserve."""
    if base_chunk != DEFAULT_TOKEN_CHUNK_SIZE or "TRELLIS2_TOKEN_CHUNK" in os.environ:
        return base_chunk
    if os.environ.get("TRELLIS2_FAST_VRAM", "1") == "0":
        return base_chunk
    free_gb = _cuda_free_gb(torch_mod)
    if free_gb is not None and free_gb >= VRAM_SAFETY_FLOOR_GB + 0.5:
        return max(base_chunk, FAST_TOKEN_CHUNK_HIGH)
    return base_chunk


def _move_cond_dict(cond: dict[str, Any], device: Any, non_blocking: bool = False) -> dict[str, Any]:
    """Move conditioning tensors/VarLenTensors between CPU and CUDA."""
    moved: dict[str, Any] = {}
    for key, val in cond.items():
        if hasattr(val, "to"):
            try:
                moved[key] = val.to(device, non_blocking=non_blocking)
            except TypeError:
                moved[key] = val.to(device)
        else:
            moved[key] = val
    return moved


def patch_layernorm32(
    layernorm_cls: Any,
    torch_mod: Any,
    chunk_size: int = DEFAULT_VOXEL_CHUNK_SIZE,
) -> None:
    """Chunk LayerNorm32 along voxel rows to prevent multi-GiB float32 upcast spikes."""
    if getattr(layernorm_cls, "_trellis2_vram_patched", False):
        return
    orig_forward = layernorm_cls.forward

    def _chunked_ln_forward(self: Any, x: Any) -> Any:
        step = _effective_voxel_chunk(torch_mod, chunk_size)
        if x.ndim != 2 or x.shape[0] <= step:
            return orig_forward(self, x)
        out = torch_mod.empty_like(x)
        n = x.shape[0]
        for s in range(0, n, step):
            e = min(s + step, n)
            out[s:e].copy_(orig_forward(self, x[s:e]))
        return out

    layernorm_cls.forward = _chunked_ln_forward
    layernorm_cls._trellis2_vram_patched = True


def _make_compiled_pointwise(fn: Callable[..., Any], torch_mod: Any) -> Callable[..., Any]:
    """Compile a pointwise CUDA kernel once with dynamic shapes and automatic eager fallback."""
    if (
        torch_mod is None
        or os.name == "nt"
        or os.environ.get("TRELLIS2_FAST_VRAM", "1") == "0"
        or os.environ.get("TRELLIS2_TORCH_COMPILE", "1") == "0"
        or not callable(getattr(torch_mod, "compile", None))
        or not torch_mod.cuda.is_available()
    ):
        return fn
    try:
        return torch_mod.compile(fn, mode="default", dynamic=True)
    except Exception as exc:
        log.debug("torch.compile unavailable for %s; using eager mode: %s", fn.__name__, exc)
        return fn


def patch_sparse_convnext_block(
    convnext_cls: Any,
    f_mod: Any,
    chunk_size: int = DEFAULT_VOXEL_CHUNK_SIZE,
    torch_mod: Any = None,
) -> None:
    """Chunk pointwise Norm + MLP + SiLU + residual add in SparseConvNeXtBlock3d with persistent Inductor fusion.

    At 768^3 (~8.5M voxels, C=128, hidden=512), unchunked MLP + out-of-place SiLU
    requires >18 GiB of temporary tensors; adaptive 256K..2M-voxel chunks eliminate
    OOM while maximizing Tensor Core GEMM throughput with bit-identical outputs.
    """
    if getattr(convnext_cls, "_trellis2_vram_patched", False):
        return

    def _convnext_pointwise_eager(
        sub_h: Any, sub_x: Any, ln_w: Any, ln_b: Any, ln_eps: float, w1: Any, b1: Any, w2: Any, b2: Any
    ) -> Any:
        ln_w_f = ln_w.float() if ln_w is not None else None
        ln_b_f = ln_b.float() if ln_b is not None else None
        y = f_mod.layer_norm(sub_h.float(), (sub_h.shape[-1],), ln_w_f, ln_b_f, ln_eps).to(sub_h.dtype)
        y = f_mod.linear(y, w1, b1)
        y = f_mod.silu(y)
        y = f_mod.linear(y, w2, b2)
        return y + sub_x

    compiled_pointwise = _make_compiled_pointwise(_convnext_pointwise_eager, torch_mod)
    compile_state = {"failed": False}

    def _chunked_convnext_forward(self: Any, x: Any) -> Any:
        h = self.conv(x)
        feats = h.feats
        x_feats = x.feats
        n = feats.shape[0]
        step = _effective_voxel_chunk(torch_mod, chunk_size)
        use_fused = (
            not compile_state["failed"]
            and compiled_pointwise is not _convnext_pointwise_eager
            and bool(getattr(feats, "is_cuda", False))
            and hasattr(self.norm, "eps")
            and hasattr(self.mlp[0], "weight")
            and hasattr(self.mlp[2], "weight")
        )
        if use_fused:
            ln_w = getattr(self.norm, "weight", None)
            ln_b = getattr(self.norm, "bias", None)
            ln_eps = float(getattr(self.norm, "eps", 1e-6))
            w1, b1 = self.mlp[0].weight, self.mlp[0].bias
            w2, b2 = self.mlp[2].weight, self.mlp[2].bias
            try:
                for s in range(0, n, step):
                    e = min(s + step, n)
                    feats[s:e].copy_(
                        compiled_pointwise(feats[s:e], x_feats[s:e], ln_w, ln_b, ln_eps, w1, b1, w2, b2)
                    )
                return h
            except Exception as exc:
                log.debug("ConvNeXt compiled kernel failed; falling back to eager: %s", exc)
                compile_state["failed"] = True

        for s in range(0, n, step):
            e = min(s + step, n)
            sub = self.norm(feats[s:e])
            sub = self.mlp[0](sub)
            f_mod.silu(sub, inplace=True)
            sub = self.mlp[2](sub)
            feats[s:e].copy_(sub).add_(x_feats[s:e])
            del sub
        return h

    convnext_cls._forward = _chunked_convnext_forward
    convnext_cls._trellis2_vram_patched = True


def patch_sparse_c2s_block(
    c2s_cls: Any,
    sparse_tensor_cls: Any,
    torch_mod: Any,
    f_mod: Any,
    chunk_size: int = DEFAULT_VOXEL_CHUNK_SIZE,
) -> None:
    """Optimize SparseResBlockC2S3d via vectorized single-nonzero C2S gather, cross-decoder gather cache, and fused skip add."""
    if getattr(c2s_cls, "_trellis2_vram_patched", False):
        return
    orig_forward = c2s_cls._forward

    def _chunked_c2s_forward(self: Any, x: Any, subdiv: Any = None) -> Any:
        if getattr(self, "training", False):
            return orig_forward(self, x, subdiv)

        step = _effective_voxel_chunk(torch_mod, chunk_size)
        is_cuda_x = bool(getattr(x.feats, "is_cuda", False))
        free_gb = _cuda_free_gb(torch_mod) if is_cuda_x else None
        keep_sub_on_gpu = bool(
            os.environ.get("TRELLIS2_FAST_VRAM", "1") != "0"
            and is_cuda_x
            and (free_gb is None or free_gb >= VRAM_SAFETY_FLOOR_GB + 1.0)
        )

        sub_ret = None
        subdiv_binarized = None
        cached_gather: tuple[Any, Any] | None = None
        if self.pred_subdiv:
            raw_sub = self.to_subdiv(x)
            subdiv_binarized = raw_sub.replace(raw_sub.feats > 0)
            del raw_sub
            sub_ret = (
                subdiv_binarized
                if keep_sub_on_gpu
                else (subdiv_binarized.cpu() if hasattr(subdiv_binarized, "cpu") else subdiv_binarized)
            )
            if hasattr(sub_ret, "clear_spatial_cache"):
                sub_ret.clear_spatial_cache()
        elif subdiv is not None:
            cached_gather = getattr(subdiv, "_c2s_gather_cache", None)
            if cached_gather is not None:
                subdiv._c2s_gather_cache = None
                if cached_gather[0].device != x.device:
                    cached_gather = (
                        cached_gather[0].to(x.device, non_blocking=True),
                        cached_gather[1].to(x.device, non_blocking=True),
                    )
            else:
                sub_dev = (
                    subdiv.to(x.device, non_blocking=True)
                    if hasattr(subdiv, "to") and subdiv.device != x.device
                    else subdiv
                )
                subdiv_binarized = (
                    sub_dev if sub_dev.feats.dtype == torch_mod.bool else sub_dev.replace(sub_dev.feats > 0)
                )
                del sub_dev

        n_in = x.feats.shape[0]
        normed = torch_mod.empty_like(x.feats)
        for s in range(0, n_in, step):
            e = min(s + step, n_in)
            tmp = self.norm1(x.feats[s:e])
            f_mod.silu(tmp, inplace=True)
            normed[s:e].copy_(tmp)
            del tmp

        x_coords = x.coords
        x_scale = getattr(x, "_scale", None)
        x_shape = getattr(x, "_shape", None)
        offload_skip = bool(
            is_cuda_x
            and n_in >= 500_000
            and (free_gb is not None and free_gb < VRAM_SAFETY_FLOOR_GB + 3.5)
        )
        x_feats_saved = x.feats.cpu() if offload_skip else x.feats

        h = x.replace(normed)
        del x, normed

        h = self.conv1(h)
        if hasattr(h, "clear_spatial_cache"):
            h.clear_spatial_cache()

        can_fast_c2s = (
            getattr(self.updown, "factor", None) == 2
            and x_coords.ndim == 2
            and x_coords.shape[1] == 4
            and (cached_gather is not None or subdiv_binarized is not None)
        )

        flat_idx = None
        if can_fast_c2s:
            if cached_gather is not None:
                new_coords, flat_idx = cached_gather
                cached_gather = None
            else:
                nz = subdiv_binarized.feats.nonzero()
                row_idx = nz[:, 0]
                subidx = nz[:, 1]
                flat_idx = row_idx * 8 + subidx
                new_coords = x_coords[row_idx].clone()
                new_coords[:, 1:] *= 2
                shifts = torch_mod.tensor([0, 1, 2], device=x_coords.device, dtype=subidx.dtype)
                new_coords[:, 1:] += ((subidx[:, None] >> shifts) & 1).to(new_coords.dtype)
                del nz, row_idx, subidx, shifts
                if (
                    self.pred_subdiv
                    and sub_ret is not None
                    and (not is_cuda_x or keep_sub_on_gpu)
                    and os.environ.get("TRELLIS2_CROSS_DECODER_CACHE", "1") != "0"
                ):
                    sub_ret._c2s_gather_cache = (new_coords, flat_idx)

            h_shape = getattr(h, "_shape", None)
            out_shape = (
                None if h_shape is None else torch_mod.Size([h_shape[0], h_shape[1] // 8])
            )
            h_up_feats = h.feats.reshape(n_in * 8, -1)[flat_idx]
            h_scale = getattr(h, "_scale", None)
            h = sparse_tensor_cls(h_up_feats, new_coords, out_shape)
            if h_scale is not None:
                h._scale = tuple(sc / 2 for sc in h_scale)
            del h_up_feats, new_coords, subdiv_binarized
        else:
            h = self.updown(h, subdiv_binarized)

        n_out = h.feats.shape[0]
        for s in range(0, n_out, step):
            e = min(s + step, n_out)
            tmp = self.norm2(h.feats[s:e])
            f_mod.silu(tmp, inplace=True)
            h.feats[s:e].copy_(tmp)
            del tmp

        h = self.conv2(h)

        x_skip_feats = x_feats_saved.to(h.device, non_blocking=True) if offload_skip else x_feats_saved
        del x_feats_saved
        rep = self.out_channels // (self.channels // 8)

        if flat_idx is not None:
            # Fused chunked skip gather + repeat_interleave + in-place residual add (zero 20M-voxel x_up allocation)
            x_skip_2d = x_skip_feats.reshape(n_in * 8, self.channels // 8)
            for s in range(0, n_out, step):
                e = min(s + step, n_out)
                skip_sub = x_skip_2d[flat_idx[s:e]]
                if rep == 1:
                    h.feats[s:e].add_(skip_sub)
                else:
                    h.feats[s:e].add_(skip_sub.repeat_interleave(rep, dim=1))
                del skip_sub
            del x_skip_2d, x_skip_feats, flat_idx
        else:
            x_skip = sparse_tensor_cls(x_skip_feats, x_coords, x_shape)
            if x_scale is not None:
                x_skip._scale = x_scale
            del x_skip_feats
            x_up = self.updown(x_skip, subdiv_binarized)
            del x_skip, subdiv_binarized
            for s in range(0, n_out, step):
                e = min(s + step, n_out)
                h.feats[s:e].add_(x_up.feats[s:e].repeat_interleave(rep, dim=1))
            del x_up

        if self.pred_subdiv:
            return h, sub_ret
        return h

    c2s_cls._forward = _chunked_c2s_forward
    c2s_cls._trellis2_vram_patched = True


def patch_sparse_unet_decoder(
    decoder_cls: Any,
    torch_mod: Any,
    f_mod: Any,
    chunk_size: int = DEFAULT_VOXEL_CHUNK_SIZE,
) -> None:
    """Enable stage streaming (when needed), cross-decoder FlexGEMM/Swin cache reuse, and chunked output head."""
    if getattr(decoder_cls, "_trellis2_vram_patched", False):
        return
    orig_forward = decoder_cls.forward
    orig_upsample = decoder_cls.upsample
    orig_to = decoder_cls.to

    def _streaming_to(self: Any, *args: Any, **kwargs: Any) -> Any:
        if getattr(self, "_stream_active_blocks", False) and not getattr(self, "training", False):
            target_dev = args[0] if args else kwargs.get("device")
            if target_dev is not None and str(target_dev).startswith("cuda"):
                # Keep inactive blocks on CPU; forward/upsample stream 1 stage at a time to CUDA
                return self
        return orig_to(self, *args, **kwargs)

    def _streaming_forward(
        self: Any,
        x: Any,
        guide_subs: Any = None,
        return_subs: bool = False,
    ) -> Any:
        if getattr(self, "training", False):
            return orig_forward(self, x, guide_subs=guide_subs, return_subs=return_subs)

        assert guide_subs is None or not self.pred_subdiv, "Only decoders with pred_subdiv=False can be used with guide_subs"
        assert not return_subs or self.pred_subdiv, "Only decoders with pred_subdiv=True can be used with return_subs"

        dev = x.device
        is_cuda_x = str(dev).startswith("cuda")
        stream = bool(getattr(self, "low_vram", False) or getattr(self, "_stream_active_blocks", False))
        share_cross_cache = bool(
            not stream
            and os.environ.get("TRELLIS2_FAST_VRAM", "1") != "0"
            and os.environ.get("TRELLIS2_CROSS_DECODER_CACHE", "1") != "0"
        )

        cross_caches: dict[int, dict[str, Any]] | None = None
        if guide_subs and len(guide_subs) > 0:
            cross_caches = getattr(guide_subs[0], "_cross_decoder_stage_caches", None)
            if cross_caches is not None:
                guide_subs[0]._cross_decoder_stage_caches = None

        if stream:
            self.from_latent.to(dev)
        h = self.from_latent(x)
        if stream:
            self.from_latent.cpu()
        h = h.type(self.dtype)

        subs = []
        saved_stage_caches: dict[int, dict[str, Any]] = {}
        num_stages = len(self.blocks)
        try:
            for i, res in enumerate(self.blocks):
                num_res = len(res)
                if cross_caches is not None and i in cross_caches and not stream and hasattr(h, "_spatial_cache"):
                    expected_n, stage_dict = cross_caches.pop(i)
                    if h.coords.shape[0] == expected_n:
                        scale_k = str(getattr(h, "_scale", ""))
                        h._spatial_cache[scale_k] = stage_dict

                if stream and num_res > 0:
                    res.to(dev)
                try:
                    for j, block in enumerate(res):
                        if i < num_stages - 1 and j == num_res - 1:
                            if return_subs and self.pred_subdiv and share_cross_cache and hasattr(h, "_spatial_cache"):
                                free_gb = _cuda_free_gb(torch_mod) if is_cuda_x else None
                                if free_gb is None or free_gb >= VRAM_SAFETY_FLOOR_GB + 1.5:
                                    cur_sc = h._spatial_cache.get(str(getattr(h, "_scale", "")))
                                    if cur_sc:
                                        filtered = {k: v for k, v in cur_sc.items() if k != "subdivision"}
                                        if filtered:
                                            saved_stage_caches[i] = (h.coords.shape[0], filtered)
                            if self.pred_subdiv:
                                h, sub = block(h)
                                if stream and hasattr(sub, "cpu"):
                                    if hasattr(sub, "_c2s_gather_cache"):
                                        sub._c2s_gather_cache = None
                                    sub = sub.cpu()
                                if hasattr(sub, "clear_spatial_cache"):
                                    sub.clear_spatial_cache()
                                subs.append(sub)
                            else:
                                g_sub = guide_subs[i] if guide_subs is not None else None
                                h = block(h, subdiv=g_sub)
                        else:
                            h = block(h)
                finally:
                    if stream and num_res > 0:
                        res.cpu()

                if (
                    i == num_stages - 1
                    and return_subs
                    and self.pred_subdiv
                    and share_cross_cache
                    and hasattr(h, "_spatial_cache")
                ):
                    free_gb = _cuda_free_gb(torch_mod) if is_cuda_x else None
                    if free_gb is None or free_gb >= VRAM_SAFETY_FLOOR_GB + 2.5:
                        cur_sc = h._spatial_cache.get(str(getattr(h, "_scale", "")))
                        if cur_sc:
                            filtered = {k: v for k, v in cur_sc.items() if k != "subdivision"}
                            if filtered:
                                saved_stage_caches[i] = (h.coords.shape[0], filtered)

                if hasattr(h, "clear_spatial_cache"):
                    h.clear_spatial_cache()
                if stream and i >= 1:
                    free_gb = _cuda_free_gb(torch_mod)
                    if free_gb is not None and free_gb < VRAM_SAFETY_FLOOR_GB:
                        torch_mod.cuda.empty_cache()
        finally:
            if cross_caches is not None:
                cross_caches.clear()

        if stream:
            self.output_layer.to(dev)
        try:
            step = _effective_voxel_chunk(torch_mod, chunk_size)
            n_final = h.feats.shape[0]
            c_final = h.feats.shape[-1]
            out_feats = torch_mod.empty((n_final, self.out_channels), dtype=x.dtype, device=dev)
            w_out = self.output_layer.weight
            b_out = self.output_layer.bias
            for s in range(0, n_final, step):
                e = min(s + step, n_final)
                chunk_f32 = h.feats[s:e].to(torch_mod.float32)
                chunk_norm = f_mod.layer_norm(chunk_f32, (c_final,))
                del chunk_f32
                if chunk_norm.dtype != w_out.dtype:
                    chunk_norm = chunk_norm.to(w_out.dtype)
                chunk_out = f_mod.linear(chunk_norm, w_out, b_out)
                del chunk_norm
                out_feats[s:e].copy_(chunk_out.to(x.dtype))
                del chunk_out
        finally:
            if stream:
                self.output_layer.cpu()

        h = h.replace(out_feats)
        if hasattr(h, "clear_spatial_cache"):
            h.clear_spatial_cache()
        if return_subs and saved_stage_caches and len(subs) > 0:
            subs[0]._cross_decoder_stage_caches = saved_stage_caches
        return (h, subs) if return_subs else h

    def _streaming_upsample(self: Any, x: Any, upsample_times: int) -> Any:
        if getattr(self, "training", False):
            return orig_upsample(self, x, upsample_times)
        assert self.pred_subdiv, "Only decoders with pred_subdiv=True can be used with upsampling"

        dev = x.device
        stream = bool(getattr(self, "low_vram", False) or getattr(self, "_stream_active_blocks", False))
        if stream:
            self.from_latent.to(dev)
        h = self.from_latent(x)
        if stream:
            self.from_latent.cpu()
        h = h.type(self.dtype)

        num_stages = len(self.blocks)
        for i, res in enumerate(self.blocks):
            if i == upsample_times:
                coords = h.coords
                if hasattr(h, "clear_spatial_cache"):
                    h.clear_spatial_cache()
                del h
                return coords
            num_res = len(res)
            if stream and num_res > 0:
                res.to(dev)
            try:
                for j, block in enumerate(res):
                    if i < num_stages - 1 and j == num_res - 1:
                        h, _sub = block(h)
                        del _sub
                    else:
                        h = block(h)
            finally:
                if stream and num_res > 0:
                    res.cpu()
            if hasattr(h, "clear_spatial_cache"):
                h.clear_spatial_cache()
            if stream and i >= 1:
                free_gb = _cuda_free_gb(torch_mod)
                if free_gb is not None and free_gb < VRAM_SAFETY_FLOOR_GB:
                    torch_mod.cuda.empty_cache()

        return h.coords

    decoder_cls.to = _streaming_to
    decoder_cls.forward = _streaming_forward
    decoder_cls.upsample = _streaming_upsample
    decoder_cls._trellis2_vram_patched = True


def patch_fdg_decoder(
    fdg_cls: Any,
    base_decoder_cls: Any,
    torch_mod: Any,
    f_mod: Any,
    dual_grid_fn: Callable[..., Any],
    mesh_cls: Any,
    chunk_size: int = DEFAULT_VOXEL_CHUNK_SIZE,
) -> None:
    """Free decoded voxel tensor & caches before running Dual-Grid mesh extraction at 1536^3."""
    if getattr(fdg_cls, "_trellis2_vram_patched", False):
        return
    orig_forward = fdg_cls.forward

    def _low_mem_fdg_forward(self: Any, x: Any, gt_intersected: Any = None, **kwargs: Any) -> Any:
        if getattr(self, "training", False):
            return orig_forward(self, x, gt_intersected=gt_intersected, **kwargs)

        decoded = base_decoder_cls.forward(self, x, **kwargs)
        out_list = list(decoded) if isinstance(decoded, tuple) else [decoded]
        h = out_list[0]
        if hasattr(h, "clear_spatial_cache"):
            h.clear_spatial_cache()
        layout = list(h.layout)
        coords_all = h.coords
        feats_all = h.feats
        del h, decoded
        out_list[0] = None

        meshes = []
        step = _effective_voxel_chunk(torch_mod, chunk_size)
        margin = self.voxel_margin
        scale = 1.0 + 2.0 * margin
        for sl in layout:
            c_b = coords_all[sl, 1:].contiguous()
            f_b = feats_all[sl]
            n_b = f_b.shape[0]
            v_f = torch_mod.empty((n_b, 3), dtype=f_b.dtype, device=f_b.device)
            i_f = torch_mod.empty((n_b, 3), dtype=torch_mod.bool, device=f_b.device)
            q_f = torch_mod.empty((n_b, 1), dtype=f_b.dtype, device=f_b.device)
            for s in range(0, n_b, step):
                e = min(s + step, n_b)
                v_chunk = torch_mod.sigmoid(f_b[s:e, 0:3])
                v_chunk.mul_(scale).sub_(margin)
                v_f[s:e].copy_(v_chunk)
                i_f[s:e].copy_(f_b[s:e, 3:6] > 0)
                q_f[s:e].copy_(f_mod.softplus(f_b[s:e, 6:7]))
                del v_chunk
            del f_b
            if len(layout) == 1:
                del coords_all, feats_all
                free_gb = _cuda_free_gb(torch_mod)
                if torch_mod.cuda.is_available() and (free_gb is None or free_gb < VRAM_SAFETY_FLOOR_GB + 1.0):
                    torch_mod.cuda.empty_cache()
            meshes.append(
                mesh_cls(
                    *dual_grid_fn(
                        c_b,
                        v_f,
                        i_f,
                        q_f,
                        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                        grid_size=self.resolution,
                        train=False,
                    )
                )
            )
            del c_b, v_f, i_f, q_f

        out_list[0] = meshes
        return out_list[0] if len(out_list) == 1 else tuple(out_list)

    fdg_cls.forward = _low_mem_fdg_forward
    fdg_cls._trellis2_vram_patched = True


def patch_sparse_ffn(
    ffn_cls: Any,
    torch_mod: Any,
    f_mod: Any,
    chunk_size: int = DEFAULT_TOKEN_CHUNK_SIZE,
) -> None:
    """Chunk SparseFeedForwardNet (1536 -> 8192 -> 1536) when VRAM is tight; fuse with Inductor when VRAM is ample."""
    if getattr(ffn_cls, "_trellis2_vram_patched", False):
        return
    orig_forward = ffn_cls.forward

    def _ffn_tanh_eager(sub_x: Any, w1: Any, b1: Any, w2: Any, b2: Any) -> Any:
        return f_mod.linear(f_mod.gelu(f_mod.linear(sub_x, w1, b1), approximate="tanh"), w2, b2)

    def _ffn_none_eager(sub_x: Any, w1: Any, b1: Any, w2: Any, b2: Any) -> Any:
        return f_mod.linear(f_mod.gelu(f_mod.linear(sub_x, w1, b1), approximate="none"), w2, b2)

    compiled_ffn_tanh = _make_compiled_pointwise(_ffn_tanh_eager, torch_mod)
    compiled_ffn_none = _make_compiled_pointwise(_ffn_none_eager, torch_mod)
    compile_state = {"failed": False}

    def _chunked_ffn_forward(self: Any, x: Any) -> Any:
        if getattr(self, "training", False):
            return orig_forward(self, x)
        n = x.feats.shape[0]
        step = _effective_token_chunk(torch_mod, chunk_size)
        fc1, act, fc2 = self.mlp[0], self.mlp[1], self.mlp[2]
        approx = getattr(act, "approximate", "tanh")
        compiled_fn = compiled_ffn_tanh if approx == "tanh" else compiled_ffn_none
        eager_ref = _ffn_tanh_eager if approx == "tanh" else _ffn_none_eager
        use_compiled = (
            not compile_state["failed"]
            and compiled_fn is not eager_ref
            and bool(getattr(x.feats, "is_cuda", False))
            and hasattr(fc1, "weight")
            and hasattr(fc2, "weight")
        )
        if n <= step:
            if use_compiled:
                try:
                    return x.replace(compiled_fn(x.feats, fc1.weight, fc1.bias, fc2.weight, fc2.bias))
                except Exception as exc:
                    log.debug("SparseFFN compiled kernel failed; falling back to eager: %s", exc)
                    compile_state["failed"] = True
            return orig_forward(self, x)

        out = torch_mod.empty_like(x.feats)
        for s in range(0, n, step):
            e = min(s + step, n)
            if use_compiled and not compile_state["failed"]:
                try:
                    out[s:e].copy_(compiled_fn(x.feats[s:e], fc1.weight, fc1.bias, fc2.weight, fc2.bias))
                    continue
                except Exception as exc:
                    log.debug("SparseFFN chunked compiled kernel failed; falling back to eager: %s", exc)
                    compile_state["failed"] = True
            sub = f_mod.linear(x.feats[s:e], fc1.weight, fc1.bias)
            sub = f_mod.gelu(sub, approximate=approx)
            sub = f_mod.linear(sub, fc2.weight, fc2.bias)
            out[s:e].copy_(sub)
            del sub
        return x.replace(out)

    ffn_cls.forward = _chunked_ffn_forward
    ffn_cls._trellis2_vram_patched = True


def patch_sparse_attention_seqlens_cache(
    full_attn_mod: Any,
    torch_mod: Any,
    *alias_mods: Any,
) -> None:
    """Cache GPU int32 cu_seqlens tensors in sparse FlashAttention to eliminate ~6,000 CPU->GPU copies/asset."""
    if getattr(full_attn_mod, "_trellis2_seqlens_patched", False):
        return
    orig_sdpa = full_attn_mod.sparse_scaled_dot_product_attention
    seqlens_cache: dict[tuple[str, tuple[int, ...]], tuple[Any, int]] = {}

    def _get_cu_seqlens(seqlen_list: list[int], device: Any) -> tuple[Any, int]:
        key = (str(device), tuple(seqlen_list))
        hit = seqlens_cache.get(key)
        if hit is not None:
            return hit
        if len(seqlens_cache) >= 64:
            seqlens_cache.clear()
        cu = torch_mod.tensor(
            [0, *itertools.accumulate(seqlen_list)],
            dtype=torch_mod.int32,
            device=device,
        )
        res = (cu, max(seqlen_list))
        seqlens_cache[key] = res
        return res

    def _fast_sparse_sdpa(*args: Any, **kwargs: Any) -> Any:
        cfg = getattr(full_attn_mod, "config", None)
        if getattr(cfg, "ATTN", None) != "flash_attn":
            return orig_sdpa(*args, **kwargs)

        num_all_args = len(args) + len(kwargs)
        if num_all_args not in (1, 2, 3):
            return orig_sdpa(*args, **kwargs)

        if "flash_attn" not in full_attn_mod.__dict__:
            import flash_attn as _fa

            full_attn_mod.flash_attn = _fa
        fa = full_attn_mod.flash_attn
        varlen_cls = full_attn_mod.VarLenTensor

        if num_all_args == 1:
            qkv = args[0] if args else kwargs["qkv"]
            if not isinstance(qkv, varlen_cls):
                return orig_sdpa(*args, **kwargs)
            q_seqlen = [qkv.layout[i].stop - qkv.layout[i].start for i in range(qkv.shape[0])]
            cu_q, max_q = _get_cu_seqlens(q_seqlen, qkv.device)
            out = fa.flash_attn_varlen_qkvpacked_func(qkv.feats, cu_q, max_q)
            return qkv.replace(out)

        if num_all_args == 2:
            q = args[0] if len(args) > 0 else kwargs["q"]
            kv = args[1] if len(args) > 1 else kwargs["kv"]
            device = q.device
            if isinstance(q, varlen_cls):
                s = q
                q_seqlen = [q.layout[i].stop - q.layout[i].start for i in range(q.shape[0])]
                q_t = q.feats
                n_b = l_q = h_q = 0
            else:
                s = None
                n_b, l_q, h_q, c_q = q.shape
                q_seqlen = [l_q] * n_b
                q_t = q.reshape(n_b * l_q, h_q, c_q)

            if isinstance(kv, varlen_cls):
                kv_seqlen = [kv.layout[i].stop - kv.layout[i].start for i in range(kv.shape[0])]
                kv_t = kv.feats
            else:
                n_kv, l_kv, _, h_kv, c_kv = kv.shape
                kv_seqlen = [l_kv] * n_kv
                kv_t = kv.reshape(n_kv * l_kv, 2, h_kv, c_kv)

            cu_q, max_q = _get_cu_seqlens(q_seqlen, device)
            cu_kv, max_kv = _get_cu_seqlens(kv_seqlen, device)
            out = fa.flash_attn_varlen_kvpacked_func(q_t, kv_t, cu_q, cu_kv, max_q, max_kv)
            return s.replace(out) if s is not None else out.reshape(n_b, l_q, h_q, -1)

        q = args[0] if len(args) > 0 else kwargs["q"]
        k = args[1] if len(args) > 1 else kwargs["k"]
        v = args[2] if len(args) > 2 else kwargs["v"]
        device = q.device
        if isinstance(q, varlen_cls):
            s = q
            q_seqlen = [q.layout[i].stop - q.layout[i].start for i in range(q.shape[0])]
            q_t = q.feats
            n_b = l_q = h_q = 0
        else:
            s = None
            n_b, l_q, h_q, c_q = q.shape
            q_seqlen = [l_q] * n_b
            q_t = q.reshape(n_b * l_q, h_q, c_q)

        if isinstance(k, varlen_cls):
            kv_seqlen = [k.layout[i].stop - k.layout[i].start for i in range(k.shape[0])]
            k_t = k.feats
            v_t = v.feats
        else:
            n_kv, l_kv, h_kv, ci_kv = k.shape
            co_kv = v.shape[-1]
            kv_seqlen = [l_kv] * n_kv
            k_t = k.reshape(n_kv * l_kv, h_kv, ci_kv)
            v_t = v.reshape(n_kv * l_kv, h_kv, co_kv)

        cu_q, max_q = _get_cu_seqlens(q_seqlen, device)
        cu_kv, max_kv = _get_cu_seqlens(kv_seqlen, device)
        out = fa.flash_attn_varlen_func(q_t, k_t, v_t, cu_q, cu_kv, max_q, max_kv)
        return s.replace(out) if s is not None else out.reshape(n_b, l_q, h_q, -1)

    full_attn_mod.sparse_scaled_dot_product_attention = _fast_sparse_sdpa
    for mod in alias_mods:
        if mod is not None and hasattr(mod, "sparse_scaled_dot_product_attention"):
            mod.sparse_scaled_dot_product_attention = _fast_sparse_sdpa
    full_attn_mod._trellis2_seqlens_patched = True


def patch_cross_attention_kv_cache(
    sparse_mha_cls: Any,
    dense_mha_cls: Any | None = None,
    sparse_attn_fn_holder: Any | None = None,
    dense_attn_fn: Callable[..., Any] | None = None,
) -> None:
    """Cache static DINOv3 cross-attention (K, V) projections in VRAM across all Euler diffusion steps."""
    if sparse_mha_cls is not None and not getattr(sparse_mha_cls, "_trellis2_kv_cache_patched", False):
        orig_sparse_fwd = sparse_mha_cls.forward

        def _cached_sparse_mha_forward(self: Any, x: Any, context: Any = None) -> Any:
            if (
                getattr(self, "_type", None) != "cross"
                or not getattr(self, "_kv_cache_enabled", False)
                or getattr(self, "training", False)
                or context is None
            ):
                return orig_sparse_fwd(self, x, context=context)

            cache: dict[Any, Any] | None = getattr(self, "_kv_cache", None)
            if cache is None:
                cache = {}
                self._kv_cache = cache

            if hasattr(context, "feats") and hasattr(context, "layout"):
                ctx_f = context.feats
                layout_sig = tuple((s.start, s.stop) for s in context.layout)
                key = ("varlen", ctx_f.data_ptr(), tuple(ctx_f.shape), ctx_f.dtype, layout_sig)
            else:
                key = ("dense", context.data_ptr(), tuple(context.shape), context.dtype)

            q = self._linear(self.to_q, x)
            q = self._reshape_chs(q, (self.num_heads, -1))
            if self.qk_rms_norm:
                q = self.q_rms_norm(q)

            entry = cache.get(key)
            if entry is None:
                kv = self._linear(self.to_kv, context)
                kv = self._fused_pre(kv, num_fused=2)
                if self.qk_rms_norm:
                    k, v = kv.unbind(dim=-3)
                    k = self.k_rms_norm(k)
                    v = v.replace(v.feats.contiguous()) if hasattr(v, "feats") else v.contiguous()
                    entry = ("split", k, v)
                else:
                    entry = ("packed", kv)
                cache[key] = entry

            s_attn = (
                sparse_attn_fn_holder.sparse_scaled_dot_product_attention
                if sparse_attn_fn_holder is not None
                else getattr(self, "_sparse_sdpa_fn", None)
            )
            if s_attn is None:
                return orig_sparse_fwd(self, x, context=context)

            if entry[0] == "split":
                h = s_attn(q, entry[1], entry[2])
            else:
                h = s_attn(q, entry[1])
            h = self._reshape_chs(h, (-1,))
            return self._linear(self.to_out, h)

        sparse_mha_cls.forward = _cached_sparse_mha_forward
        sparse_mha_cls._trellis2_kv_cache_patched = True

    if dense_mha_cls is not None and not getattr(dense_mha_cls, "_trellis2_kv_cache_patched", False):
        orig_dense_fwd = dense_mha_cls.forward

        def _cached_dense_mha_forward(
            self: Any, x: Any, context: Any = None, phases: Any = None
        ) -> Any:
            if (
                getattr(self, "_type", None) != "cross"
                or not getattr(self, "_kv_cache_enabled", False)
                or getattr(self, "training", False)
                or context is None
                or dense_attn_fn is None
            ):
                return orig_dense_fwd(self, x, context=context, phases=phases)

            cache: dict[Any, Any] | None = getattr(self, "_kv_cache", None)
            if cache is None:
                cache = {}
                self._kv_cache = cache

            b_sz, l_q, _ = x.shape
            l_kv = context.shape[1]
            key = (context.data_ptr(), tuple(context.shape), context.dtype)

            q = self.to_q(x).reshape(b_sz, l_q, self.num_heads, -1)
            if self.qk_rms_norm:
                q = self.q_rms_norm(q)

            entry = cache.get(key)
            if entry is None:
                kv = self.to_kv(context).reshape(b_sz, l_kv, 2, self.num_heads, -1)
                if self.qk_rms_norm:
                    k, v = kv.unbind(dim=2)
                    k = self.k_rms_norm(k)
                    v = v.contiguous()
                    entry = ("split", k, v)
                else:
                    entry = ("packed", kv)
                cache[key] = entry

            if entry[0] == "split":
                h = dense_attn_fn(q, entry[1], entry[2])
            else:
                h = dense_attn_fn(q, entry[1])
            return self.to_out(h.reshape(b_sz, l_q, -1))

        dense_mha_cls.forward = _cached_dense_mha_forward
        dense_mha_cls._trellis2_kv_cache_patched = True


def _configure_model_kv_cache(model: Any, enabled: bool) -> None:
    """Enable or clear cross-attention KV caches across all attention blocks of a DiT model."""
    modules_fn = getattr(model, "modules", None)
    if not callable(modules_fn):
        return
    for mod in modules_fn():
        if getattr(mod, "_type", None) == "cross":
            mod._kv_cache_enabled = enabled
            kv_cache = getattr(mod, "_kv_cache", None)
            if isinstance(kv_cache, dict):
                kv_cache.clear()


def patch_cfg_sampler_batching(
    flow_euler_cls: Any,
    cfg_mixin_cls: Any,
    sparse_tensor_cls: Any,
    varlen_cat_fn: Callable[..., Any] | None,
    torch_mod: Any,
) -> None:
    """Batch positive+negative Classifier-Free Guidance into one batch=2 pass & manage KV/RoPE caches."""
    if not getattr(cfg_mixin_cls, "_trellis2_batch_cfg_patched", False):
        orig_cfg_inference = cfg_mixin_cls._inference_model

        def _batched_cfg_inference(
            self: Any,
            model: Any,
            x_t: Any,
            t: float,
            cond: Any,
            neg_cond: Any,
            guidance_strength: float,
            guidance_rescale: float = 0.0,
            **kwargs: Any,
        ) -> Any:
            if guidance_strength == 1 or guidance_strength == 0:
                return orig_cfg_inference(
                    self,
                    model,
                    x_t,
                    t,
                    cond,
                    neg_cond,
                    guidance_strength,
                    guidance_rescale=guidance_rescale,
                    **kwargs,
                )

            batch_enabled = bool(getattr(self, "_batch_cfg_enabled", False))
            x_shape = getattr(x_t, "shape", None)
            if not batch_enabled or x_shape is None or len(x_shape) == 0 or x_shape[0] != 1:
                return orig_cfg_inference(
                    self,
                    model,
                    x_t,
                    t,
                    cond,
                    neg_cond,
                    guidance_strength,
                    guidance_rescale=guidance_rescale,
                    **kwargs,
                )

            ctx_state: dict[str, Any] = getattr(self, "_active_sample_ctx", None) or {}
            base_infer = super(cfg_mixin_cls, self)._inference_model

            try:
                pred_pos = None
                pred_neg = None
                if (
                    isinstance(x_t, torch_mod.Tensor)
                    and isinstance(cond, torch_mod.Tensor)
                    and isinstance(neg_cond, torch_mod.Tensor)
                ):
                    cond_2 = ctx_state.get("cond_2")
                    if cond_2 is None:
                        cond_2 = torch_mod.cat([cond, neg_cond], dim=0)
                        ctx_state["cond_2"] = cond_2
                    kw = dict(kwargs)
                    cc = kw.get("concat_cond")
                    if isinstance(cc, torch_mod.Tensor):
                        cc_2 = ctx_state.get("concat_cond_2")
                        if cc_2 is None:
                            cc_2 = torch_mod.cat([cc, cc], dim=0)
                            ctx_state["concat_cond_2"] = cc_2
                        kw["concat_cond"] = cc_2
                    x_2 = torch_mod.cat([x_t, x_t], dim=0)
                    pred_2 = base_infer(model, x_2, t, cond_2, **kw)
                    pred_pos = pred_2[0:1]
                    pred_neg = pred_2[1:2]
                elif sparse_tensor_cls is not None and isinstance(x_t, sparse_tensor_cls):
                    m = x_t.feats.shape[0]
                    c_in = x_t.feats.shape[1]
                    proto_2 = ctx_state.get("proto_2")
                    if proto_2 is None or ctx_state.get("proto_m") != m:
                        coords_b1 = x_t.coords.clone()
                        coords_b1[:, 0] = 1
                        coords_2 = torch_mod.cat([x_t.coords, coords_b1], dim=0)
                        del coords_b1
                        feats_init = torch_mod.cat([x_t.feats, x_t.feats], dim=0)
                        proto_2 = sparse_tensor_cls(
                            feats=feats_init,
                            coords=coords_2,
                            shape=torch_mod.Size([2, c_in]),
                        )
                        if hasattr(proto_2, "register_spatial_cache"):
                            proto_2.register_spatial_cache("layout", [slice(0, m), slice(m, 2 * m)])
                            proto_2.register_spatial_cache(
                                "seqlen", torch_mod.tensor([m, m], dtype=torch_mod.long, device=x_t.device)
                            )
                            proto_2.register_spatial_cache(
                                "cum_seqlen",
                                torch_mod.tensor([0, m, 2 * m], dtype=torch_mod.long, device=x_t.device),
                            )
                            bmap = torch_mod.empty(2 * m, dtype=torch_mod.long, device=x_t.device)
                            bmap[:m] = 0
                            bmap[m:] = 1
                            proto_2.register_spatial_cache("batch_boardcast_map", bmap)
                        ctx_state["proto_2"] = proto_2
                        ctx_state["proto_m"] = m
                        x_2 = proto_2
                    else:
                        x_2 = proto_2.replace(torch_mod.cat([x_t.feats, x_t.feats], dim=0))

                    cond_2 = ctx_state.get("cond_2")
                    if cond_2 is None:
                        if isinstance(cond, torch_mod.Tensor) and isinstance(neg_cond, torch_mod.Tensor):
                            cond_2 = torch_mod.cat([cond, neg_cond], dim=0)
                        elif varlen_cat_fn is not None and hasattr(cond, "feats"):
                            cond_2 = varlen_cat_fn([cond, neg_cond], dim=0)
                        else:
                            return orig_cfg_inference(
                                self,
                                model,
                                x_t,
                                t,
                                cond,
                                neg_cond,
                                guidance_strength,
                                guidance_rescale=guidance_rescale,
                                **kwargs,
                            )
                        ctx_state["cond_2"] = cond_2

                    kw = dict(kwargs)
                    cc = kw.get("concat_cond")
                    if cc is not None and isinstance(cc, sparse_tensor_cls):
                        cc_2 = ctx_state.get("concat_cond_2")
                        if cc_2 is None:
                            cc_2 = proto_2.replace(torch_mod.cat([cc.feats, cc.feats], dim=0))
                            ctx_state["concat_cond_2"] = cc_2
                        kw["concat_cond"] = cc_2

                    pred_2 = base_infer(model, x_2, t, cond_2, **kw)
                    pred_pos = x_t.replace(pred_2.feats[:m])
                    pred_neg = x_t.replace(pred_2.feats[m:])
                    del pred_2
                else:
                    return orig_cfg_inference(
                        self,
                        model,
                        x_t,
                        t,
                        cond,
                        neg_cond,
                        guidance_strength,
                        guidance_rescale=guidance_rescale,
                        **kwargs,
                    )

                pred = guidance_strength * pred_pos + (1 - guidance_strength) * pred_neg
                if guidance_rescale > 0:
                    x_0_pos = self._pred_to_xstart(x_t, t, pred_pos)
                    x_0_cfg = self._pred_to_xstart(x_t, t, pred)
                    std_pos = x_0_pos.std(dim=list(range(1, x_0_pos.ndim)), keepdim=True)
                    std_cfg = x_0_cfg.std(dim=list(range(1, x_0_cfg.ndim)), keepdim=True)
                    x_0_rescaled = x_0_cfg * (std_pos / std_cfg)
                    x_0 = guidance_rescale * x_0_rescaled + (1 - guidance_rescale) * x_0_cfg
                    pred = self._xstart_to_pred(x_t, t, x_0)
                return pred
            except Exception as exc:
                if "out of memory" in str(exc).lower() or type(exc).__name__ == "OutOfMemoryError":
                    log.warning("CFG batch=2 hit VRAM limit; falling back to sequential CFG for remaining steps.")
                    self._batch_cfg_enabled = False
                    ctx_state.clear()
                    if torch_mod is not None and torch_mod.cuda.is_available():
                        torch_mod.cuda.empty_cache()
                    return orig_cfg_inference(
                        self,
                        model,
                        x_t,
                        t,
                        cond,
                        neg_cond,
                        guidance_strength,
                        guidance_rescale=guidance_rescale,
                        **kwargs,
                    )
                raise

        cfg_mixin_cls._inference_model = _batched_cfg_inference
        cfg_mixin_cls._trellis2_batch_cfg_patched = True

    if flow_euler_cls is not None and not getattr(flow_euler_cls, "_trellis2_fast_euler_patched", False):
        orig_sample = flow_euler_cls.sample

        def _fast_euler_sample(
            self: Any,
            model: Any,
            noise: Any,
            cond: Any = None,
            steps: int = 50,
            rescale_t: float = 1.0,
            verbose: bool = True,
            tqdm_desc: str = "Sampling",
            **kwargs: Any,
        ) -> Any:
            if kwargs.pop("return_trajectory", False):
                return orig_sample(
                    self, model, noise, cond, steps, rescale_t, verbose, tqdm_desc=tqdm_desc, **kwargs
                )

            import numpy as np

            try:
                from easydict import EasyDict as _edict
            except ImportError:
                class _edict(dict):  # type: ignore[no-redef]
                    __getattr__ = dict.__getitem__
                    __setattr__ = dict.__setitem__

            try:
                from tqdm import tqdm as _tqdm
            except ImportError:
                def _tqdm(it: Any, **_kw: Any) -> Any:
                    return it

            noise_dev = getattr(noise, "device", None)
            is_cuda_run = noise_dev is not None and str(noise_dev).startswith("cuda")
            free_gb = _cuda_free_gb(torch_mod) if is_cuda_run else None
            fast_vram = os.environ.get("TRELLIS2_FAST_VRAM", "1") != "0"
            use_kv_cache = (
                fast_vram
                and os.environ.get("TRELLIS2_KV_CACHE", "1") != "0"
                and (free_gb is None or free_gb >= VRAM_SAFETY_FLOOR_GB + 0.3)
            )
            use_batch_cfg = (
                fast_vram
                and os.environ.get("TRELLIS2_BATCH_CFG", "1") != "0"
                and not getattr(model, "_stream_active_blocks", False)
                and (free_gb is None or free_gb >= VRAM_SAFETY_FLOOR_GB + 0.8)
            )

            model_dtype = getattr(model, "dtype", None)
            autocast_on = bool(
                torch_mod is not None
                and hasattr(torch_mod, "is_autocast_enabled")
                and torch_mod.is_autocast_enabled()
            )
            call_kwargs = dict(kwargs)
            if not autocast_on and isinstance(model_dtype, getattr(torch_mod, "dtype", ())):
                if cond is not None and hasattr(cond, "type") and getattr(cond, "dtype", None) != model_dtype:
                    cond = cond.type(model_dtype)
                neg_c = call_kwargs.get("neg_cond")
                if neg_c is not None and hasattr(neg_c, "type") and getattr(neg_c, "dtype", None) != model_dtype:
                    call_kwargs["neg_cond"] = neg_c.type(model_dtype)

            # Hold strong refs to cast conditioning tensors until KV cache is cleared
            self._active_sample_ctx = {
                "cond_ref": cond,
                "neg_cond_ref": call_kwargs.get("neg_cond"),
            }
            self._batch_cfg_enabled = use_batch_cfg
            if use_kv_cache:
                _configure_model_kv_cache(model, True)

            try:
                sample = noise
                t_seq = np.linspace(1, 0, steps + 1)
                t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
                t_list = t_seq.tolist()
                t_pairs = [(t_list[i], t_list[i + 1]) for i in range(steps)]
                with torch_mod.no_grad():
                    for t_cur, t_prev in _tqdm(t_pairs, desc=tqdm_desc, disable=not verbose):
                        pred_v = self._inference_model(model, sample, t_cur, cond, **call_kwargs)
                        sample = sample - (t_cur - t_prev) * pred_v
                        del pred_v
                return _edict({"samples": sample, "pred_x_t": [], "pred_x_0": []})
            finally:
                self._batch_cfg_enabled = False
                self._active_sample_ctx = None
                if use_kv_cache:
                    _configure_model_kv_cache(model, False)

        flow_euler_cls.sample = _fast_euler_sample
        flow_euler_cls._trellis2_fast_euler_patched = True


def patch_timestep_embedder_cache(timestep_embedder_cls: Any, torch_mod: Any) -> None:
    """Cache sinusoidal frequency vectors directly on GPU to remove per-step CPU->GPU copies & enable CUDA Graphs."""
    if timestep_embedder_cls is None or getattr(timestep_embedder_cls, "_trellis2_freq_cached", False):
        return
    import math

    freq_cache: dict[tuple[str, int, int], Any] = {}

    @staticmethod  # type: ignore[misc]
    def _cached_timestep_embedding(t: Any, dim: int, max_period: int = 10000) -> Any:
        half = dim // 2
        key = (str(t.device), half, int(max_period))
        freqs = freq_cache.get(key)
        if freqs is None:
            if len(freq_cache) >= 32:
                freq_cache.clear()
            freqs = torch_mod.exp(
                -math.log(max_period)
                * torch_mod.arange(start=0, end=half, dtype=torch_mod.float32, device=t.device)
                / half
            )
            freq_cache[key] = freqs
        args = t[:, None].float() * freqs[None]
        embedding = torch_mod.cat([torch_mod.cos(args), torch_mod.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch_mod.cat([embedding, torch_mod.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    timestep_embedder_cls.timestep_embedding = _cached_timestep_embedding
    timestep_embedder_cls._trellis2_freq_cached = True


def patch_sparse_structure_cudagraph(ss_flow_cls: Any, torch_mod: Any) -> None:
    """Capture static-shape 30-block SparseStructureFlowModel into CUDA Graphs (~960 kernels -> 1 replay)."""
    if ss_flow_cls is None or getattr(ss_flow_cls, "_trellis2_cudagraph_patched", False):
        return
    orig_forward = ss_flow_cls.forward

    def _cudagraph_ss_forward(self: Any, x: Any, t: Any, cond: Any) -> Any:
        if (
            getattr(self, "training", False)
            or getattr(self, "_cudagraph_disabled", False)
            or os.environ.get("TRELLIS2_FAST_VRAM", "1") == "0"
            or os.environ.get("TRELLIS2_CUDAGRAPH", "1") == "0"
            or not getattr(x, "is_cuda", False)
            or not getattr(t, "is_cuda", False)
            or not getattr(cond, "is_cuda", False)
            or not hasattr(torch_mod.cuda, "CUDAGraph")
        ):
            return orig_forward(self, x, t, cond)

        graph_cache: dict[Any, tuple[Any, Any, Any, Any, Any]] | None = getattr(
            self, "_cudagraph_cache", None
        )
        if graph_cache is None:
            graph_cache = {}
            self._cudagraph_cache = graph_cache

        is_infer_mode = bool(
            hasattr(torch_mod, "is_inference_mode_enabled")
            and torch_mod.is_inference_mode_enabled()
        )
        key = (
            is_infer_mode,
            tuple(x.shape),
            x.dtype,
            tuple(t.shape),
            t.dtype,
            tuple(cond.shape),
            cond.dtype,
        )
        entry = graph_cache.get(key)
        if entry is None:
            free_gb = _cuda_free_gb(torch_mod)
            if (free_gb is not None and free_gb < VRAM_SAFETY_FLOOR_GB + 1.0) or len(graph_cache) >= 4:
                return orig_forward(self, x, t, cond)
            try:
                # Disable external data_ptr KV cache inside captured graph so static_cond.copy_(cond)
                # recomputes fresh cross-attention K/V for every new input image deterministically.
                saved_kv_flags: list[tuple[Any, bool]] = []
                for mod in self.modules():
                    if getattr(mod, "_type", None) == "cross":
                        saved_kv_flags.append((mod, bool(getattr(mod, "_kv_cache_enabled", False))))
                        mod._kv_cache_enabled = False
                try:
                    static_x = x.clone()
                    static_t = t.clone()
                    static_cond = cond.clone()

                    cur_stream = torch_mod.cuda.current_stream()
                    warm_stream = torch_mod.cuda.Stream()
                    warm_stream.wait_stream(cur_stream)
                    with torch_mod.cuda.stream(warm_stream):
                        for _ in range(2):
                            _ = orig_forward(self, static_x, static_t, static_cond)
                    cur_stream.wait_stream(warm_stream)

                    graph = torch_mod.cuda.CUDAGraph()
                    with torch_mod.cuda.graph(graph):
                        static_out = orig_forward(self, static_x, static_t, static_cond)
                    entry = (graph, static_x, static_t, static_cond, static_out)
                    graph_cache[key] = entry
                finally:
                    for mod, prev_flag in saved_kv_flags:
                        mod._kv_cache_enabled = prev_flag
            except Exception as exc:
                log.debug("Skipping SparseStructureFlowModel CUDAGraph capture: %s", exc)
                self._cudagraph_disabled = True
                graph_cache.clear()
                return orig_forward(self, x, t, cond)

        graph, static_x, static_t, static_cond, static_out = entry
        static_x.copy_(x)
        static_t.copy_(t)
        static_cond.copy_(cond)
        graph.replay()
        return static_out.clone()

    ss_flow_cls.forward = _cudagraph_ss_forward
    ss_flow_cls._trellis2_cudagraph_patched = True


def patch_mesh_simplify_noop(mesh_cls: Any) -> None:
    """Skip redundant CuMesh 12M-triangle build+readback when face count is already <= target (e.g. 16,777,216)."""
    if mesh_cls is None or getattr(mesh_cls, "_trellis2_simplify_patched", False):
        return
    orig_simplify = mesh_cls.simplify

    def _fast_simplify(
        self: Any, target: int = 1000000, verbose: bool = False, options: dict[str, Any] = {}
    ) -> None:
        faces = getattr(self, "faces", None)
        shape = getattr(faces, "shape", None)
        if shape is not None and len(shape) >= 1 and shape[0] <= target:
            return
        orig_simplify(self, target=target, verbose=verbose, options=options)

    mesh_cls.simplify = _fast_simplify
    mesh_cls._trellis2_simplify_patched = True


def gpu_dilate_uv_gutters(
    base_color: Any,
    metallic: Any,
    roughness: Any,
    alpha: Any,
    mask_inv: Any,
    torch_mod: Any,
    f_mod: Any,
    device: Any = None,
    num_iters: int = 6,
) -> tuple[Any, Any, Any, Any]:
    """Dilate UV island borders into empty UV gutters on GPU (<5ms) while keeping 100% of valid texels bit-exact."""
    import numpy as np

    dev = device if device is not None else ("cuda" if torch_mod.cuda.is_available() else "cpu")
    h, w = mask_inv.shape[:2]

    def _as_hwc(arr: Any, ch: int) -> Any:
        if arr.ndim == 2:
            return arr[..., None]
        if arr.ndim == 3 and arr.shape[2] == ch:
            return arr
        return arr.reshape(h, w, ch)

    bc_hwc = _as_hwc(base_color, 3)
    mt_hwc = _as_hwc(metallic, 1)
    rg_hwc = _as_hwc(roughness, 1)
    al_hwc = _as_hwc(alpha, 1)

    packed_np = np.concatenate([bc_hwc, mt_hwc, rg_hwc, al_hwc], axis=-1)  # (H, W, 6) uint8
    orig_u8 = torch_mod.from_numpy(packed_np).to(device=dev, non_blocking=True)
    valid_orig = (torch_mod.from_numpy(mask_inv).to(device=dev, non_blocking=True) == 0)

    if not bool(valid_orig.any()):
        return bc_hwc.copy(), mt_hwc.copy(), rg_hwc.copy(), al_hwc.copy()

    # Check if alpha is uniform across all valid UV island pixels (e.g. opaque 255)
    al_valid = orig_u8[..., 5][valid_orig]
    al_min = al_valid.min()
    al_uniform = bool((al_min == al_valid.max()).item())

    buf = orig_u8.permute(2, 0, 1).unsqueeze(0).to(torch_mod.float32)  # (1, 6, H, W)
    valid = valid_orig.unsqueeze(0).unsqueeze(0).to(torch_mod.float32)  # (1, 1, H, W)
    buf.mul_(valid)

    for _ in range(max(1, int(num_iters))):
        nbr_valid = f_mod.avg_pool2d(valid, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        need_fill = (valid == 0) & (nbr_valid > 0)
        if not bool(need_fill.any()):
            break
        nbr_sum = f_mod.avg_pool2d(buf, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        filled = nbr_sum / nbr_valid.clamp_min(1e-6)
        buf = torch_mod.where(need_fill, filled, buf)
        valid = torch_mod.where(need_fill, torch_mod.ones_like(valid), valid)

    out_u8 = buf.squeeze(0).permute(1, 2, 0).round().clamp_(0, 255).to(torch_mod.uint8)
    # Guarantee 100% bit-exact preservation of every valid UV island pixel
    out_u8 = torch_mod.where(valid_orig.unsqueeze(-1), orig_u8, out_u8)
    if al_uniform:
        out_u8[..., 5].fill_(int(al_min.item()))

    out_np = out_u8.cpu().numpy()
    return (
        np.ascontiguousarray(out_np[..., 0:3]),
        np.ascontiguousarray(out_np[..., 3:4]),
        np.ascontiguousarray(out_np[..., 4:5]),
        np.ascontiguousarray(out_np[..., 5:6]),
    )


def patch_o_voxel_postprocess(postprocess_mod: Any, torch_mod: Any = None) -> None:
    """Accelerate o_voxel.postprocess.to_glb via GPU UV-gutter dilation (<5ms) with 4-thread CPU cv2 fallback."""
    if postprocess_mod is None or getattr(postprocess_mod, "_trellis2_postprocess_patched", False):
        return
    try:
        src = inspect.getsource(postprocess_mod.to_glb)
    except Exception:
        return

    old_inpaint_block = (
        "    base_color = cv2.inpaint(base_color, mask_inv, 3, cv2.INPAINT_TELEA)\n"
        "    metallic = cv2.inpaint(metallic, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]\n"
        "    roughness = cv2.inpaint(roughness, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]\n"
        "    alpha = cv2.inpaint(alpha, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]"
    )
    if old_inpaint_block not in src:
        return

    inpaint_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="trellis2-inpaint")

    def _parallel_pbr_inpaint(
        base_color: Any, metallic: Any, roughness: Any, alpha: Any, mask_inv: Any
    ) -> tuple[Any, Any, Any, Any]:
        if (
            torch_mod is not None
            and os.environ.get("TRELLIS2_FAST_VRAM", "1") != "0"
            and os.environ.get("TRELLIS2_GPU_UV_DILATE", "1") != "0"
            and torch_mod.cuda.is_available()
        ):
            try:
                import torch.nn.functional as f_mod

                return gpu_dilate_uv_gutters(
                    base_color, metallic, roughness, alpha, mask_inv, torch_mod, f_mod
                )
            except Exception as exc:
                log.debug("GPU UV-gutter dilation failed; falling back to 4-thread cv2.inpaint: %s", exc)

        cv2_local = postprocess_mod.cv2
        flag = cv2_local.INPAINT_TELEA
        f_bc = inpaint_pool.submit(cv2_local.inpaint, base_color, mask_inv, 3, flag)
        f_mt = inpaint_pool.submit(cv2_local.inpaint, metallic, mask_inv, 1, flag)
        f_rg = inpaint_pool.submit(cv2_local.inpaint, roughness, mask_inv, 1, flag)
        f_al = inpaint_pool.submit(cv2_local.inpaint, alpha, mask_inv, 1, flag)
        return (
            f_bc.result(),
            f_mt.result()[..., None],
            f_rg.result()[..., None],
            f_al.result()[..., None],
        )

    patched_src = src.replace(
        old_inpaint_block,
        "    base_color, metallic, roughness, alpha = _trellis2_parallel_pbr_inpaint(base_color, metallic, roughness, alpha, mask_inv)",
    )

    try:
        postprocess_mod._trellis2_parallel_pbr_inpaint = _parallel_pbr_inpaint
        ns: dict[str, Any] = dict(postprocess_mod.__dict__)
        exec(compile(patched_src, "<trellis2_fast_postprocess>", "exec"), ns)
        postprocess_mod.to_glb = ns["to_glb"]
        postprocess_mod._trellis2_postprocess_patched = True
    except Exception as exc:
        log.debug("Keeping original o_voxel.postprocess.to_glb: %s", exc)


class _FastLosslessPngContext:
    """Use lossless zlib compress_level=1 when serializing 2K/4K PBR PNG textures into GLB (3-4x faster, 100% lossless)."""

    _lock = threading.Lock()

    def __enter__(self) -> None:
        self._orig_save = None
        if os.environ.get("TRELLIS2_FAST_PNG", "1") == "0":
            return
        self._lock.acquire()
        self._orig_save = Image.Image.save
        orig = self._orig_save

        def _fast_png_save(img_self: Image.Image, fp: Any, format: str | None = None, **params: Any) -> Any:
            fmt = format
            if fmt is None and isinstance(fp, (str, Path)):
                ext = str(fp).rsplit(".", 1)[-1].lower()
                if ext == "png":
                    fmt = "PNG"
            if isinstance(fmt, str) and fmt.upper() == "PNG" and "compress_level" not in params:
                params["compress_level"] = 1
            return orig(img_self, fp, format=format, **params)

        Image.Image.save = _fast_png_save  # type: ignore[method-assign]

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._orig_save is not None:
            try:
                Image.Image.save = self._orig_save  # type: ignore[method-assign]
            finally:
                self._orig_save = None
                self._lock.release()


def patch_cv2_inpaint_fastpath(cv2_mod: Any, np_mod: Any) -> None:
    """Skip slow CPU INPAINT_TELEA Fast Marching on uniform PBR channels (e.g. alpha=255) with exact output."""
    if getattr(cv2_mod, "_trellis2_inpaint_patched", False):
        return
    try:
        set_threads = getattr(cv2_mod, "setNumThreads", None)
        if callable(set_threads):
            set_threads(max(4, os.cpu_count() or 4))
    except Exception:
        pass

    orig_inpaint = cv2_mod.inpaint

    def _fast_inpaint(src: Any, inpaint_mask: Any, inpaint_radius: float, flags: int) -> Any:
        try:
            if isinstance(src, np_mod.ndarray) and isinstance(inpaint_mask, np_mod.ndarray):
                known = inpaint_mask == 0
                if np_mod.any(known):
                    valid = src[known]
                    if src.ndim == 2:
                        vmin = valid.min()
                        if vmin == valid.max():
                            return np_mod.full_like(src, vmin)
                    elif src.ndim == 3 and src.shape[2] == 1:
                        vmin = valid.min()
                        if vmin == valid.max():
                            # OpenCV's Python binding squeezes (H, W, 1) -> (H, W)
                            return np_mod.full(src.shape[:2], vmin, dtype=src.dtype)
                    elif src.ndim == 3 and src.shape[2] in (3, 4):
                        vmin = valid.min(axis=0)
                        if np_mod.array_equal(vmin, valid.max(axis=0)):
                            out = np_mod.empty_like(src)
                            out[...] = vmin
                            return out
        except Exception:
            pass
        return orig_inpaint(src, inpaint_mask, inpaint_radius, flags)

    cv2_mod.inpaint = _fast_inpaint
    cv2_mod._trellis2_inpaint_patched = True


def patch_torch_compile_modules(pipeline: Any, torch_mod: Any) -> list[str]:
    """Optionally apply persistent torch.compile to dense backbone modules when TRELLIS2_COMPILE_BACKBONES=1.

    By default (TRELLIS2_COMPILE_BACKBONES=0), whole-backbone compilation is skipped because:
    - BiRefNet (rembg_model.model) has dynamic Swin/deformable control flow that stalls TorchInductor
      for 7+ minutes while running in ~0.25s eagerly (never compiled even when opted in).
    - DINOv3 (image_cond_model) and sparse_structure_decoder run <0.5s total eagerly while adding
      ~55s+ to worker cold-boot and recompiling at 1024px.
    Hot-loop pointwise Triton kernels (SparseConvNeXtBlock3d MLP & SparseFeedForwardNet MLP) remain
    compiled separately via _make_compiled_pointwise under TRELLIS2_TORCH_COMPILE=1.
    """
    compiled: list[str] = []
    if (
        pipeline is None
        or torch_mod is None
        or os.name == "nt"
        or os.environ.get("TRELLIS2_FAST_VRAM", "1") == "0"
        or os.environ.get("TRELLIS2_TORCH_COMPILE", "1") == "0"
        or os.environ.get("TRELLIS2_COMPILE_BACKBONES", "0") != "1"
        or not callable(getattr(torch_mod, "compile", None))
        or not torch_mod.cuda.is_available()
    ):
        return compiled

    models_dict = getattr(pipeline, "models", None)
    if isinstance(models_dict, dict):
        ss_dec = models_dict.get("sparse_structure_decoder")
        if ss_dec is not None and not getattr(ss_dec, "_trellis2_compiled", False):
            try:
                compiled_dec = torch_mod.compile(ss_dec, mode="default", dynamic=True)
                compiled_dec._trellis2_compiled = True
                models_dict["sparse_structure_decoder"] = compiled_dec
                compiled.append("sparse_structure_decoder")
            except Exception as exc:
                log.debug("Skipping torch.compile for sparse_structure_decoder: %s", exc)

    img_cond = getattr(pipeline, "image_cond_model", None)
    if img_cond is not None and not getattr(img_cond, "_trellis2_compiled", False):
        extract_fn = getattr(img_cond, "extract_features", None)
        if callable(extract_fn):
            try:
                img_cond.extract_features = torch_mod.compile(extract_fn, mode="default", dynamic=True)
                img_cond._trellis2_compiled = True
                compiled.append("image_cond_model.extract_features")
            except Exception as exc:
                log.debug("Skipping torch.compile for image_cond_model.extract_features: %s", exc)

    if compiled:
        log.info("Enabled persistent torch.compile (Inductor) for: %s", ", ".join(compiled))
    return compiled


def warmup_pipeline_kernels(
    pipeline: Any,
    torch_mod: Any,
    model_mgr: VRAMModelManager | None = None,
) -> list[str]:
    """Pre-warm Stage-1 CUDA Graphs & cuDNN/cuBLAS kernels at worker boot before /ping reports ready."""
    warmed: list[str] = []
    if (
        pipeline is None
        or torch_mod is None
        or os.environ.get("TRELLIS2_FAST_VRAM", "1") == "0"
        or os.environ.get("TRELLIS2_BOOT_WARMUP", "1") == "0"
        or not torch_mod.cuda.is_available()
    ):
        return warmed

    free_gb = _cuda_free_gb(torch_mod)
    if free_gb is not None and free_gb < VRAM_SAFETY_FLOOR_GB + 1.5:
        log.info("Skipping boot kernel warmup (free VRAM %.2f GiB < %.2f GiB)", free_gb, VRAM_SAFETY_FLOOR_GB + 1.5)
        return warmed

    t0 = time.monotonic()
    dev = getattr(pipeline, "device", torch_mod.device("cuda"))
    models_dict = getattr(pipeline, "models", {}) or {}
    infer_ctx = torch_mod.inference_mode() if hasattr(torch_mod, "inference_mode") else nullcontext()

    with infer_ctx:
        rembg = getattr(pipeline, "rembg_model", None)
        if rembg is not None and callable(rembg):
            try:
                if model_mgr is not None:
                    model_mgr.ensure_on_gpu(
                        ["rembg_model"], dev, required_free_gb=VRAM_SAFETY_FLOOR_GB + 1.0
                    )
                dummy_rgb = Image.new("RGB", (64, 64), (128, 128, 128))
                _ = rembg(dummy_rgb)
                warmed.append("rembg_model")
            except Exception as exc:
                log.debug("Boot warmup skipped for rembg_model: %s", exc)

        cond_1 = None
        img_cond = getattr(pipeline, "image_cond_model", None)
        if img_cond is not None and callable(img_cond):
            try:
                if model_mgr is not None:
                    model_mgr.ensure_on_gpu(
                        ["image_cond_model"], dev, required_free_gb=VRAM_SAFETY_FLOOR_GB + 1.0
                    )
                prev_sz = getattr(img_cond, "image_size", None)
                if prev_sz is not None:
                    img_cond.image_size = 512
                try:
                    dummy_img = Image.new("RGB", (64, 64), (128, 128, 128))
                    cond_1 = img_cond([dummy_img])
                    warmed.append("image_cond_model")
                finally:
                    if prev_sz is not None:
                        img_cond.image_size = prev_sz
            except Exception as exc:
                log.debug("Boot warmup skipped for image_cond_model: %s", exc)
                cond_1 = None

        ss_flow = models_dict.get("sparse_structure_flow_model")
        if ss_flow is not None:
            try:
                if model_mgr is not None:
                    model_mgr.ensure_on_gpu(
                        ["sparse_structure_flow_model"],
                        dev,
                        required_free_gb=VRAM_SAFETY_FLOOR_GB + 1.0,
                    )
                reso = int(getattr(ss_flow, "resolution", 16))
                in_ch = int(getattr(ss_flow, "in_channels", 8))
                cond_ch = int(getattr(ss_flow, "cond_channels", 1024))
                flow_dtype = getattr(ss_flow, "dtype", torch_mod.float16)
                if cond_1 is None:
                    cond_1 = torch_mod.zeros((1, 1029, cond_ch), dtype=flow_dtype, device=dev)
                else:
                    cond_1 = cond_1.to(device=dev, dtype=flow_dtype)

                for bsz in (2, 1):
                    x_w = torch_mod.zeros((bsz, in_ch, reso, reso, reso), dtype=torch_mod.float32, device=dev)
                    t_w = torch_mod.full((bsz,), 500.0, dtype=torch_mod.float32, device=dev)
                    c_w = cond_1.expand(bsz, -1, -1).contiguous()
                    _ = ss_flow(x_w, t_w, c_w)
                    del x_w, t_w, c_w
                warmed.append("sparse_structure_flow_model(cudagraph)")
            except Exception as exc:
                log.debug("Boot warmup skipped for sparse_structure_flow_model: %s", exc)
        del cond_1

        ss_dec = models_dict.get("sparse_structure_decoder")
        if ss_dec is not None:
            try:
                if model_mgr is not None:
                    model_mgr.ensure_on_gpu(
                        ["sparse_structure_decoder"],
                        dev,
                        required_free_gb=VRAM_SAFETY_FLOOR_GB + 1.0,
                    )
                reso = int(getattr(ss_flow, "resolution", 16)) if ss_flow is not None else 16
                out_ch = int(getattr(ss_flow, "out_channels", 8)) if ss_flow is not None else 8
                z_w = torch_mod.zeros((1, out_ch, reso, reso, reso), dtype=torch_mod.float32, device=dev)
                _ = ss_dec(z_w)
                del z_w
                warmed.append("sparse_structure_decoder")
            except Exception as exc:
                log.debug("Boot warmup skipped for sparse_structure_decoder: %s", exc)

    if torch_mod.cuda.is_available():
        try:
            torch_mod.cuda.synchronize()
        except Exception:
            pass
    if warmed:
        log.info("Boot kernel warmup completed in %.2fs: %s", time.monotonic() - t0, ", ".join(warmed))
    return warmed


def _persist_flex_gemm_autotune_cache() -> None:
    """Flush FlexGEMM's Triton autotune cache to persistent disk if exposed by installed version."""
    import sys as _sys

    fg = _sys.modules.get("flex_gemm")
    if fg is None:
        return
    for attr_chain in (
        ("autotuner", "save_cache"),
        ("autotune", "save_cache"),
        ("ops", "autotuner", "save_cache"),
    ):
        obj: Any = fg
        for part in attr_chain:
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if callable(obj):
            try:
                obj()
                return
            except Exception:
                pass


class VRAMModelManager:
    """VRAM-budgeted LRU GPU residency + pinned-memory async CUDA stream prefetching."""

    def __init__(
        self,
        pipeline: Any,
        torch_mod: Any,
        vram_gb: float | None = None,
        enable_pin_memory: bool = True,
    ) -> None:
        self._pipeline = pipeline
        self._torch = torch_mod
        self._vram_gb = vram_gb
        self._pin_enabled = bool(
            enable_pin_memory
            and os.name != "nt"
            and os.environ.get("TRELLIS2_PIN_MEMORY", "1") != "0"
            and torch_mod is not None
            and torch_mod.cuda.is_available()
        )
        self._models: dict[str, Any] = {}
        for name, mod in (getattr(pipeline, "models", {}) or {}).items():
            if mod is not None:
                self._models[name] = mod
        for extra_name in ("image_cond_model", "rembg_model"):
            extra_mod = getattr(pipeline, extra_name, None)
            if extra_mod is not None:
                self._models[extra_name] = extra_mod

        self._sizes_gb: dict[str, float] = {
            name: self._estimate_size_gb(mod) for name, mod in self._models.items()
        }
        self._resident_lru: dict[str, int] = {}
        self._pinned: set[str] = set()
        self._tick = 0
        self._prefetch_stream: Any = None
        self._inflight_events: dict[str, Any] = {}
        if torch_mod is not None and torch_mod.cuda.is_available() and hasattr(torch_mod.cuda, "Stream"):
            try:
                self._prefetch_stream = torch_mod.cuda.Stream()
            except Exception:
                self._prefetch_stream = None

    @staticmethod
    def _unwrap_module(obj: Any) -> Any:
        inner = getattr(obj, "model", None)
        if inner is not None and hasattr(inner, "parameters") and not hasattr(obj, "parameters"):
            return inner
        return obj

    @classmethod
    def _estimate_size_gb(cls, obj: Any) -> float:
        mod = cls._unwrap_module(obj)
        total = 0
        seen: set[int] = set()
        for getter in ("parameters", "buffers"):
            fn = getattr(mod, getter, None)
            if callable(fn):
                try:
                    for t in fn():
                        ptr = t.data_ptr() if hasattr(t, "data_ptr") else id(t)
                        if ptr not in seen:
                            seen.add(ptr)
                            total += t.numel() * t.element_size()
                except Exception:
                    pass
        return max(0.05, total / (1024**3)) if total > 0 else 0.5

    def _pin_on_cpu(self, name: str) -> None:
        if not self._pin_enabled or name in self._pinned:
            return
        mod = self._unwrap_module(self._models.get(name))
        if mod is None or not hasattr(mod, "parameters"):
            return
        try:
            for p in mod.parameters():
                if not p.is_cuda and not p.is_pinned():
                    p.data = p.data.pin_memory()
            for b in mod.buffers():
                if not b.is_cuda and not b.is_pinned():
                    b.data = b.data.pin_memory()
            self._pinned.add(name)
        except Exception:
            # Respect container memlock ulimits cleanly without failing
            self._pin_enabled = False

    def _move_to_device(self, name: str, device: Any, non_blocking: bool = False) -> None:
        obj = self._models.get(name)
        if obj is None:
            return
        inner = self._unwrap_module(obj)
        if getattr(obj, "_stream_active_blocks", False) and str(device).startswith("cuda"):
            obj.to(device)
            return
        if inner is not obj and hasattr(inner, "to"):
            try:
                inner.to(device, non_blocking=non_blocking)
                return
            except TypeError:
                inner.to(device)
                return
        to_fn = getattr(obj, "to", None)
        if callable(to_fn):
            try:
                to_fn(device, non_blocking=non_blocking)
            except TypeError:
                to_fn(device)

    def _move_to_cpu(self, name: str) -> None:
        obj = self._models.get(name)
        if obj is None:
            return
        ev = self._inflight_events.pop(name, None)
        if ev is not None and hasattr(ev, "synchronize"):
            ev.synchronize()
        cg_cache = getattr(obj, "_cudagraph_cache", None)
        if isinstance(cg_cache, dict):
            cg_cache.clear()
        cpu_fn = getattr(obj, "cpu", None)
        if callable(cpu_fn):
            cpu_fn()
        self._resident_lru.pop(name, None)
        self._pin_on_cpu(name)

    def warm_initial_models(self, device: Any) -> None:
        """Pre-stage all active cascade models into VRAM up to the 4 GiB safety floor + diffusion buffer."""
        if self._torch is None or not self._torch.cuda.is_available():
            return
        # Pin unused 512 texture flow model on CPU so cascade runs never pay for it
        if "tex_slat_flow_model_512" in self._models:
            self._pin_on_cpu("tex_slat_flow_model_512")

        priority = (
            "rembg_model",
            "image_cond_model",
            "sparse_structure_flow_model",
            "sparse_structure_decoder",
            "shape_slat_flow_model_512",
            "shape_slat_decoder",
            "shape_slat_flow_model_1024",
            "tex_slat_flow_model_1024",
            "tex_slat_decoder",
        )
        reserve_gb = VRAM_SAFETY_FLOOR_GB + 1.5
        for name in priority:
            if name not in self._models:
                continue
            if getattr(self._models[name], "_stream_active_blocks", False):
                continue
            free_gb = _cuda_free_gb(self._torch)
            need_gb = self._sizes_gb.get(name, 1.0)
            if free_gb is not None and (free_gb - need_gb) < reserve_gb:
                self._pin_on_cpu(name)
                continue
            self._move_to_device(name, device, non_blocking=False)
            self._tick += 1
            self._resident_lru[name] = self._tick

        for name in self._models:
            if name not in self._resident_lru:
                self._pin_on_cpu(name)

    def ensure_on_gpu(
        self,
        names: Sequence[str],
        device: Any,
        required_free_gb: float = 6.0,
        protected: Sequence[str] = (),
    ) -> None:
        """Ensure `names` are on GPU and at least `required_free_gb` of activation headroom is available."""
        present_names = [n for n in names if n in self._models]
        keep = set(present_names) | set(protected)

        # Wait on any background DMA prefetch for requested models
        cur_stream = (
            self._torch.cuda.current_stream()
            if self._torch is not None and self._torch.cuda.is_available() and hasattr(self._torch.cuda, "current_stream")
            else None
        )
        for n in present_names:
            ev = self._inflight_events.pop(n, None)
            if ev is not None and cur_stream is not None:
                cur_stream.wait_event(ev)

        load_gb = sum(
            self._sizes_gb.get(n, 0.5)
            for n in present_names
            if n not in self._resident_lru and not getattr(self._models[n], "_stream_active_blocks", False)
        )
        free_gb = _cuda_free_gb(self._torch)
        if free_gb is not None and (free_gb - load_gb) < required_free_gb:
            if self._prefetch_stream is not None:
                self._prefetch_stream.synchronize()
            evict_order = sorted(
                (k for k in list(self._resident_lru.keys()) if k not in keep),
                key=lambda k: self._resident_lru[k],
            )
            freed_any = False
            projected = free_gb - load_gb
            for victim in evict_order:
                if projected >= required_free_gb:
                    break
                self._move_to_cpu(victim)
                projected += self._sizes_gb.get(victim, 0.5)
                freed_any = True
            if freed_any and self._torch is not None and self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()

        for n in present_names:
            if n not in self._resident_lru:
                self._move_to_device(n, device, non_blocking=bool( n in self._pinned ))
            self._tick += 1
            self._resident_lru[n] = self._tick

    def prefetch_async(
        self,
        names: Sequence[str],
        device: Any,
        min_free_after_gb: float = 8.0,
    ) -> None:
        """Asynchronously DMA-prefetch upcoming models on a separate CUDA stream when VRAM headroom allows."""
        if self._prefetch_stream is None or self._torch is None or not self._torch.cuda.is_available():
            return
        free_gb = _cuda_free_gb(self._torch)
        if free_gb is None:
            return
        for n in names:
            if n not in self._models or n in self._resident_lru or n in self._inflight_events:
                continue
            if getattr(self._models[n], "_stream_active_blocks", False):
                continue
            need_gb = self._sizes_gb.get(n, 1.0)
            if (free_gb - need_gb) < min_free_after_gb:
                continue
            free_gb -= need_gb
            self._pin_on_cpu(n)
            try:
                cur_stream = self._torch.cuda.current_stream()
                self._prefetch_stream.wait_stream(cur_stream)
                with self._torch.cuda.stream(self._prefetch_stream):
                    self._move_to_device(n, device, non_blocking=True)
                    ev = self._torch.cuda.Event()
                    ev.record(self._prefetch_stream)
                self._inflight_events[n] = ev
                self._tick += 1
                self._resident_lru[n] = self._tick
            except Exception:
                pass

    def evict_all(self) -> None:
        """Force-evict all models to CPU (used only during emergency OOM recovery)."""
        if self._prefetch_stream is not None:
            try:
                self._prefetch_stream.synchronize()
            except Exception:
                pass
        self._inflight_events.clear()
        for name in list(self._models.keys()):
            try:
                self._move_to_cpu(name)
            except Exception:
                pass
        self._resident_lru.clear()


def patch_dit_block_streaming(slat_flow_cls: Any) -> None:
    """Optional Active-Parameter Block Streaming for 1.3B DiT SLatFlowModel (1 block on GPU at a time)."""
    if getattr(slat_flow_cls, "_trellis2_dit_stream_patched", False):
        return
    orig_to = slat_flow_cls.to
    orig_forward = slat_flow_cls.forward

    def _dit_to(self: Any, *args: Any, **kwargs: Any) -> Any:
        if getattr(self, "_stream_active_blocks", False) and not getattr(self, "training", False):
            target_dev = args[0] if args else kwargs.get("device")
            if target_dev is not None and str(target_dev).startswith("cuda"):
                # Move lightweight stem/head layers, keep heavy 30-block torso on CPU
                self.t_embedder.to(target_dev)
                if getattr(self, "share_mod", False):
                    self.adaLN_modulation.to(target_dev)
                if hasattr(self, "pos_embedder"):
                    self.pos_embedder.to(target_dev)
                self.input_layer.to(target_dev)
                self.out_layer.to(target_dev)
                return self
        return orig_to(self, *args, **kwargs)

    def _dit_forward(self: Any, x: Any, t: Any, cond: Any, concat_cond: Any = None, **kwargs: Any) -> Any:
        if not getattr(self, "_stream_active_blocks", False) or getattr(self, "training", False):
            return orig_forward(self, x, t, cond, concat_cond=concat_cond, **kwargs)

        from trellis2.modules import sparse as sp
        from trellis2.modules.utils import manual_cast
        import torch.nn.functional as f_mod

        dev = x.device
        if concat_cond is not None:
            x = sp.sparse_cat([x, concat_cond], dim=-1)
        if isinstance(cond, list):
            cond = sp.VarLenTensor.from_tensor_list(cond)

        h = self.input_layer(x)
        h = manual_cast(h, self.dtype)
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = manual_cast(t_emb, self.dtype)
        cond = manual_cast(cond, self.dtype)

        if self.pe_mode == "ape":
            pe = self.pos_embedder(h.coords[:, 1:])
            h = h + manual_cast(pe, self.dtype)

        for block in self.blocks:
            block.to(dev)
            try:
                h = block(h, t_emb, cond)
            finally:
                block.cpu()

        h = manual_cast(h, x.dtype)
        h = h.replace(f_mod.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h)
        return h

    slat_flow_cls.to = _dit_to
    slat_flow_cls.forward = _dit_forward
    slat_flow_cls._trellis2_dit_stream_patched = True


def install_trellis2_vram_optimizations(
    pipeline: Any,
    torch_mod: Any,
    low_vram: bool = True,
    vram_gb: float | None = None,
) -> None:
    """Install zero-quality-loss VRAM optimizations and VRAM-backed compute/caching accelerations."""
    import cv2
    import numpy as np
    import o_voxel.postprocess as o_voxel_postprocess_mod
    import torch.nn.functional as f_mod
    from o_voxel.convert import flexible_dual_grid_to_mesh
    from trellis2.models.sc_vaes.fdg_vae import FlexiDualGridVaeDecoder
    from trellis2.models.sc_vaes.sparse_unet_vae import (
        SparseConvNeXtBlock3d,
        SparseResBlockC2S3d,
        SparseUnetVaeDecoder,
    )
    from trellis2.models.sparse_structure_flow import (
        SparseStructureFlowModel,
        TimestepEmbedder,
    )
    from trellis2.models.structured_latent_flow import SLatFlowModel
    from trellis2.modules.attention.full_attn import scaled_dot_product_attention
    from trellis2.modules.attention.modules import MultiHeadAttention
    from trellis2.modules.norm import LayerNorm32
    from trellis2.modules.sparse import SparseTensor, varlen_cat
    from trellis2.modules.sparse import attention as sparse_attn_pkg
    from trellis2.modules.sparse.attention import full_attn as sparse_full_attn_mod
    from trellis2.modules.sparse.attention import modules as sparse_attn_modules_mod
    from trellis2.modules.sparse.attention.modules import SparseMultiHeadAttention
    from trellis2.modules.sparse.conv import config as conv_config
    from trellis2.modules.sparse.transformer.blocks import SparseFeedForwardNet
    from trellis2.pipelines.samplers.classifier_free_guidance_mixin import (
        ClassifierFreeGuidanceSamplerMixin,
    )
    from trellis2.pipelines.samplers.flow_euler import FlowEulerSampler
    from trellis2.representations import Mesh

    # Keep FLEX_GEMM_HASHMAP_RATIO at upstream default (2.0) for full sparse-conv lookup speed;
    # only override if explicitly requested via TRELLIS2_HASHMAP_RATIO.
    if "TRELLIS2_HASHMAP_RATIO" in os.environ:
        conv_config.FLEX_GEMM_HASHMAP_RATIO = float(os.environ["TRELLIS2_HASHMAP_RATIO"])

    patch_layernorm32(LayerNorm32, torch_mod, DEFAULT_VOXEL_CHUNK_SIZE)
    patch_sparse_convnext_block(
        SparseConvNeXtBlock3d, f_mod, DEFAULT_VOXEL_CHUNK_SIZE, torch_mod=torch_mod
    )
    patch_sparse_c2s_block(
        SparseResBlockC2S3d, SparseTensor, torch_mod, f_mod, DEFAULT_VOXEL_CHUNK_SIZE
    )
    patch_sparse_unet_decoder(SparseUnetVaeDecoder, torch_mod, f_mod, DEFAULT_VOXEL_CHUNK_SIZE)
    patch_fdg_decoder(
        FlexiDualGridVaeDecoder,
        SparseUnetVaeDecoder,
        torch_mod,
        f_mod,
        flexible_dual_grid_to_mesh,
        Mesh,
        DEFAULT_VOXEL_CHUNK_SIZE,
    )
    patch_sparse_ffn(SparseFeedForwardNet, torch_mod, f_mod, DEFAULT_TOKEN_CHUNK_SIZE)

    # VRAM-backed lossless compute & caching accelerations
    patch_timestep_embedder_cache(TimestepEmbedder, torch_mod)
    patch_sparse_attention_seqlens_cache(
        sparse_full_attn_mod, torch_mod, sparse_attn_pkg, sparse_attn_modules_mod
    )
    patch_cross_attention_kv_cache(
        SparseMultiHeadAttention,
        MultiHeadAttention,
        sparse_attn_modules_mod,
        scaled_dot_product_attention,
    )
    patch_cfg_sampler_batching(
        FlowEulerSampler,
        ClassifierFreeGuidanceSamplerMixin,
        SparseTensor,
        varlen_cat,
        torch_mod,
    )
    patch_sparse_structure_cudagraph(SparseStructureFlowModel, torch_mod)
    patch_mesh_simplify_noop(Mesh)
    patch_cv2_inpaint_fastpath(cv2, np)
    patch_o_voxel_postprocess(o_voxel_postprocess_mod, torch_mod)

    models_dict = getattr(pipeline, "models", {}) or {}
    # On >=20 GiB GPUs (e.g. 24 GB RTX 4090 / A10G / L4), keep the 0.42 GiB decoders 100% GPU-resident
    # during decode; only force stage-by-stage CPU streaming on <20 GiB cards.
    force_decoder_stream = bool(low_vram and vram_gb is not None and vram_gb < 20.0)
    if low_vram:
        for dec_name in ("shape_slat_decoder", "tex_slat_decoder"):
            dec = models_dict.get(dec_name)
            if dec is not None:
                dec.low_vram = force_decoder_stream
                dec._stream_active_blocks = force_decoder_stream
                dec.cpu()

    stream_dit = os.environ.get("TRELLIS2_STREAM_DIT_BLOCKS", "auto").strip().lower()
    enable_dit_stream = stream_dit in {"1", "true", "yes", "on"} or (
        stream_dit == "auto" and vram_gb is not None and vram_gb < 16.0
    )
    if enable_dit_stream:
        patch_dit_block_streaming(SLatFlowModel)
        for flow_name in (
            "shape_slat_flow_model_512",
            "shape_slat_flow_model_1024",
            "tex_slat_flow_model_512",
            "tex_slat_flow_model_1024",
        ):
            flow_mod = models_dict.get(flow_name)
            if flow_mod is not None:
                flow_mod._stream_active_blocks = True
                flow_mod.cpu()

    log.info(
        "Installed TRELLIS.2 VRAM + speed optimizations (reserve=%.1fGiB, voxel_chunk=%d..%d, token_chunk=%d..%d, batch_cfg=%s, kv_cache=%s, cudagraph=%s, cross_dec_cache=%s)",
        VRAM_SAFETY_FLOOR_GB,
        DEFAULT_VOXEL_CHUNK_SIZE,
        FAST_VOXEL_CHUNK_ULTRA,
        DEFAULT_TOKEN_CHUNK_SIZE,
        FAST_TOKEN_CHUNK_HIGH,
        os.environ.get("TRELLIS2_BATCH_CFG", "1") != "0",
        os.environ.get("TRELLIS2_KV_CACHE", "1") != "0",
        os.environ.get("TRELLIS2_CUDAGRAPH", "1") != "0",
        os.environ.get("TRELLIS2_CROSS_DECODER_CACHE", "1") != "0",
    )


class Trellis2Engine:
    """Manages lifecycle and serialized GPU execution of Trellis2ImageTo3DPipeline."""

    def __init__(
        self,
        model_id: str | None = None,
        low_vram_setting: str | None = None,
    ) -> None:
        self.model_id = model_id or os.environ.get("TRELLIS2_MODEL_ID", DEFAULT_MODEL_ID)
        self._low_vram_setting = low_vram_setting or os.environ.get("TRELLIS2_LOW_VRAM", "auto")
        self.status: str = "loading"
        self.stage: str = "initializing"
        self.error: str | None = None
        self.gpu_name: str | None = None
        self.vram_gb: float | None = None
        self.low_vram: bool | None = None
        self.load_time_s: float | None = None

        self._pipeline: Any = None
        self._o_voxel: Any = None
        self._torch: Any = None
        self._model_mgr: VRAMModelManager | None = None
        self._gpu_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._load_thread: threading.Thread | None = None
        self._load_started_at: float | None = None

    def start_loading(self) -> None:
        """Load weights on a background thread so /health responds immediately."""
        with self._state_lock:
            if self._load_thread is not None and self._load_thread.is_alive():
                return
            self.status = "loading"
            self.stage = "starting"
            self.error = None
            self.load_time_s = None
            self._load_started_at = time.monotonic()
            self._load_thread = threading.Thread(
                target=self._load_worker, name="trellis2-loader", daemon=True
            )
            self._load_thread.start()

    def _load_worker(self) -> None:
        try:
            import torch
            import o_voxel
            from trellis2.pipelines import Trellis2ImageTo3DPipeline

            self._torch = torch
            self._o_voxel = o_voxel

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA GPU is not available inside the container.")

            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            for sdp_fn_name in ("enable_flash_sdp", "enable_mem_efficient_sdp"):
                sdp_fn = getattr(torch.backends.cuda, sdp_fn_name, None)
                if callable(sdp_fn):
                    try:
                        sdp_fn(True)
                    except Exception:
                        pass

            props = torch.cuda.get_device_properties(0)
            self.gpu_name = props.name
            self.vram_gb = round(props.total_memory / (1024**3), 2)
            self.low_vram = _resolve_low_vram(self._low_vram_setting, self.vram_gb)

            log.info(
                "Detected GPU: %s (%.2f GB VRAM), low_vram=%s",
                self.gpu_name,
                self.vram_gb,
                self.low_vram,
            )

            self.stage = "downloading_or_loading_weights"
            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            if token:
                os.environ.setdefault("HF_TOKEN", token)

            pipeline = Trellis2ImageTo3DPipeline.from_pretrained(self.model_id)
            # low_vram MUST be set before pipeline.cuda() because Pipeline.to() checks self.low_vram
            pipeline.low_vram = self.low_vram
            install_trellis2_vram_optimizations(
                pipeline,
                torch,
                low_vram=bool(self.low_vram),
                vram_gb=self.vram_gb,
            )
            self.stage = "moving_to_cuda"
            pipeline.cuda()
            patch_torch_compile_modules(pipeline, torch)
            model_mgr: VRAMModelManager | None = None
            if self.low_vram and os.environ.get("TRELLIS2_FAST_VRAM", "1") != "0":
                model_mgr = VRAMModelManager(pipeline, torch, vram_gb=self.vram_gb)
                model_mgr.warm_initial_models(pipeline.device)

            self.stage = "warming_up_kernels"
            warmup_pipeline_kernels(pipeline, torch, model_mgr=model_mgr)

            with self._state_lock:
                self._pipeline = pipeline
                self._model_mgr = model_mgr
                self.status = "ready"
                self.stage = "ready"
                self.error = None
                if self._load_started_at is not None:
                    self.load_time_s = round(max(0.0, time.monotonic() - self._load_started_at), 2)
            log.info("TRELLIS.2 pipeline ready (%s) in %.2fs", self.model_id, self.load_time_s or 0.0)
        except Exception as exc:
            log.exception("Failed to load TRELLIS.2 pipeline")
            with self._state_lock:
                self.status = "error"
                self.stage = "failed"
                self.error = f"{type(exc).__name__}: {exc}"

    def health_info(self) -> dict[str, Any]:
        with self._state_lock:
            info: dict[str, Any] = {
                "status": self.status,
                "stage": self.stage,
                "model_id": self.model_id,
                "gpu": self.gpu_name,
                "vram_gb": self.vram_gb,
                "low_vram": self.low_vram,
                "load_time_s": self.load_time_s,
                "error": self.error,
            }
        if self._torch is not None and self._torch.cuda.is_available():
            try:
                info["vram_allocated_gb"] = round(self._torch.cuda.memory_allocated(0) / (1024**3), 2)
                info["vram_reserved_gb"] = round(self._torch.cuda.memory_reserved(0) / (1024**3), 2)
                if hasattr(self._torch.cuda, "mem_get_info"):
                    free_bytes, _total_bytes = self._torch.cuda.mem_get_info(0)
                    info["vram_free_gb"] = round(free_bytes / (1024**3), 2)
            except Exception:
                pass
        return info

    def _release_vram(
        self,
        offload_models: bool = False,
        min_free_gb: float | None = None,
    ) -> None:
        """Free unused neural sub-models, Python garbage, and PyTorch CUDA cache when needed."""
        if min_free_gb is not None and not offload_models and self._model_mgr is not None:
            free_gb = _cuda_free_gb(self._torch)
            if free_gb is not None and free_gb >= min_free_gb:
                return
            gc.collect()
            if self._torch is not None and self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
            free_gb = _cuda_free_gb(self._torch)
            if free_gb is not None and free_gb < min_free_gb and self._pipeline is not None:
                self._model_mgr.ensure_on_gpu([], self._pipeline.device, required_free_gb=min_free_gb)
            return

        if offload_models and self.low_vram and self._pipeline is not None:
            if self._model_mgr is not None:
                self._model_mgr.evict_all()
            else:
                candidates = list(getattr(self._pipeline, "models", {}).values())
                for attr in ("image_cond_model", "rembg_model"):
                    extra = getattr(self._pipeline, attr, None)
                    if extra is not None:
                        candidates.append(extra)
                for model in candidates:
                    cpu_fn = getattr(model, "cpu", None)
                    if callable(cpu_fn):
                        try:
                            cpu_fn()
                        except Exception:
                            pass

        gc.collect()
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
            ipc_collect = getattr(self._torch.cuda, "ipc_collect", None)
            if callable(ipc_collect):
                ipc_collect()

    @staticmethod
    def _is_cuda_oom(exc: BaseException) -> bool:
        return "out of memory" in str(exc).lower() or type(exc).__name__ == "OutOfMemoryError"

    def _can_run_staged(self) -> bool:
        """True when self._pipeline exposes full Trellis2ImageTo3DPipeline methods (vs simple test mocks)."""
        pipe = self._pipeline
        return bool(
            self.low_vram
            and pipe is not None
            and callable(getattr(pipe, "get_cond", None))
            and callable(getattr(pipe, "sample_sparse_structure", None))
            and callable(getattr(pipe, "decode_shape_slat", None))
            and callable(getattr(pipe, "decode_tex_slat", None))
        )

    def _configure_decoder_residency(self, dec: Any) -> None:
        """Run 0.42 GiB 3D VAE decoder 100% GPU-resident on >=20 GiB cards above the 4 GiB reserve."""
        if dec is None:
            return
        free_gb = _cuda_free_gb(self._torch)
        should_stream = bool(
            (self.vram_gb is not None and self.vram_gb < 20.0)
            or (free_gb is not None and free_gb < VRAM_SAFETY_FLOOR_GB + 4.5)
        )
        dec.low_vram = should_stream
        dec._stream_active_blocks = should_stream

    def _run_pipeline_staged(
        self,
        image: Image.Image,
        params: GenerationParams,
        pipeline_type: str,
        on_stage: Callable[[str], None] | None = None,
    ) -> Any:
        """Execute Trellis2ImageTo3DPipeline with 4 GiB-floor GPU residency & async DMA prefetching."""
        def _emit(stage_name: str) -> None:
            if on_stage is not None:
                on_stage(stage_name)

        torch = self._torch
        pipe = self._pipeline
        mgr = self._model_mgr
        floor = VRAM_SAFETY_FLOOR_GB
        saved_pipe_low_vram = getattr(pipe, "low_vram", True)
        if mgr is not None:
            # Let VRAMModelManager control residency instead of upstream's unconditional .cpu() per call
            pipe.low_vram = False

        def _log_stage(label: str, started: float) -> None:
            free_now = _cuda_free_gb(torch)
            if free_now is not None:
                log.info(
                    "Stage %-26s done in %.2fs (free VRAM: %.2f GiB)",
                    label,
                    time.monotonic() - started,
                    free_now,
                )
            else:
                log.info("Stage %-26s done in %.2fs", label, time.monotonic() - started)

        try:
            if params.preprocess_image:
                _emit("preprocessing_image")
                t_st = time.monotonic()
                if mgr is not None:
                    mgr.ensure_on_gpu(
                        ["rembg_model"],
                        pipe.device,
                        required_free_gb=floor + 0.5,
                        protected=["image_cond_model"],
                    )
                    mgr.prefetch_async(
                        ["image_cond_model", "sparse_structure_flow_model"],
                        pipe.device,
                        min_free_after_gb=floor + 1.0,
                    )
                image = pipe.preprocess_image(image)
                if mgr is None:
                    self._release_vram(offload_models=True)
                _log_stage("preprocess_image", t_st)

            torch.manual_seed(params.seed)
            _emit("extracting_features")
            t_st = time.monotonic()
            if mgr is not None:
                mgr.ensure_on_gpu(
                    ["image_cond_model"],
                    pipe.device,
                    required_free_gb=floor + 0.5,
                    protected=["sparse_structure_flow_model", "sparse_structure_decoder"],
                )
                mgr.prefetch_async(
                    ["sparse_structure_flow_model", "sparse_structure_decoder"],
                    pipe.device,
                    min_free_after_gb=floor + 1.0,
                )
            cond_512 = pipe.get_cond([image], 512)
            cond_1024_raw = pipe.get_cond([image], 1024) if pipeline_type != "512" else None
            free_after_cond = _cuda_free_gb(torch)
            keep_cond_gpu = bool(
                mgr is not None and free_after_cond is not None and free_after_cond >= (floor + 1.0)
            )
            cond_1024_saved = (
                (cond_1024_raw if keep_cond_gpu else _move_cond_dict(cond_1024_raw, "cpu"))
                if cond_1024_raw is not None
                else None
            )
            del cond_1024_raw
            if mgr is None:
                self._release_vram(offload_models=True)
            _log_stage("get_cond", t_st)

            ss_res = {"512": 32, "1024": 64, "1024_cascade": 32, "1536_cascade": 32}[pipeline_type]
            next_flow = "shape_slat_flow_model_1024" if pipeline_type == "1024" else "shape_slat_flow_model_512"
            _emit("sampling_sparse_structure")
            t_st = time.monotonic()
            if mgr is not None:
                mgr.ensure_on_gpu(
                    ["sparse_structure_flow_model", "sparse_structure_decoder"],
                    pipe.device,
                    required_free_gb=floor + 0.8,
                )
                mgr.prefetch_async([next_flow], pipe.device, min_free_after_gb=floor + 1.5)
            coords = pipe.sample_sparse_structure(cond_512, ss_res, 1, {})
            if mgr is None:
                self._release_vram(offload_models=True)
            _log_stage("sample_sparse_structure", t_st)

            if pipeline_type == "512":
                if mgr is not None:
                    mgr.ensure_on_gpu(
                        ["shape_slat_flow_model_512"], pipe.device, required_free_gb=floor + 1.2
                    )
                    mgr.prefetch_async(
                        ["tex_slat_flow_model_512"], pipe.device, min_free_after_gb=floor + 1.5
                    )
                shape_slat = pipe.sample_shape_slat(
                    cond_512, pipe.models["shape_slat_flow_model_512"], coords, {}
                )
                del coords
                if mgr is None:
                    self._release_vram(offload_models=True)

                if mgr is not None:
                    mgr.ensure_on_gpu(
                        ["tex_slat_flow_model_512"], pipe.device, required_free_gb=floor + 1.2
                    )
                    mgr.prefetch_async(
                        ["shape_slat_decoder"], pipe.device, min_free_after_gb=floor + 1.5
                    )
                tex_slat = pipe.sample_tex_slat(
                    cond_512, pipe.models["tex_slat_flow_model_512"], shape_slat, {}
                )
                del cond_512
                if mgr is None:
                    self._release_vram(offload_models=True)
                res = 512
            elif pipeline_type == "1024":
                del cond_512
                assert cond_1024_saved is not None
                cond_1024 = (
                    cond_1024_saved
                    if keep_cond_gpu
                    else _move_cond_dict(cond_1024_saved, pipe.device, non_blocking=True)
                )
                del cond_1024_saved
                if mgr is not None:
                    mgr.ensure_on_gpu(
                        ["shape_slat_flow_model_1024"], pipe.device, required_free_gb=floor + 2.0
                    )
                    mgr.prefetch_async(
                        ["tex_slat_flow_model_1024"], pipe.device, min_free_after_gb=floor + 2.5
                    )
                shape_slat = pipe.sample_shape_slat(
                    cond_1024, pipe.models["shape_slat_flow_model_1024"], coords, {}
                )
                del coords
                if mgr is None:
                    self._release_vram(offload_models=True)

                if mgr is not None:
                    mgr.ensure_on_gpu(
                        ["tex_slat_flow_model_1024"], pipe.device, required_free_gb=floor + 2.0
                    )
                    mgr.prefetch_async(
                        ["shape_slat_decoder"], pipe.device, min_free_after_gb=floor + 2.5
                    )
                tex_slat = pipe.sample_tex_slat(
                    cond_1024, pipe.models["tex_slat_flow_model_1024"], shape_slat, {}
                )
                del cond_1024
                if mgr is None:
                    self._release_vram(offload_models=True)
                res = 1024
            elif pipeline_type in ("1024_cascade", "1536_cascade"):
                target_res = 1536 if pipeline_type == "1536_cascade" else 1024
                max_num_tokens = int(os.environ.get("TRELLIS2_MAX_NUM_TOKENS", "49152"))

                _emit("sampling_shape_512")
                t_st = time.monotonic()
                if mgr is not None:
                    mgr.ensure_on_gpu(
                        ["shape_slat_flow_model_512"],
                        pipe.device,
                        required_free_gb=floor + 1.2,
                        protected=["shape_slat_decoder", "shape_slat_flow_model_1024"],
                    )
                    mgr.prefetch_async(
                        ["shape_slat_decoder", "shape_slat_flow_model_1024"],
                        pipe.device,
                        min_free_after_gb=floor + 1.5,
                    )
                slat_lr = pipe.sample_shape_slat(
                    cond_512, pipe.models["shape_slat_flow_model_512"], coords, {}
                )
                del cond_512, coords
                if mgr is None:
                    self._release_vram(offload_models=True)
                _log_stage("sample_shape_slat_512", t_st)

                t_st = time.monotonic()
                shape_dec = pipe.models["shape_slat_decoder"]
                if mgr is not None:
                    self._configure_decoder_residency(shape_dec)
                    mgr.ensure_on_gpu(
                        ["shape_slat_decoder"],
                        pipe.device,
                        required_free_gb=floor + 2.0,
                        protected=["shape_slat_flow_model_1024"],
                    )
                    mgr.prefetch_async(
                        ["shape_slat_flow_model_1024", "tex_slat_flow_model_1024"],
                        pipe.device,
                        min_free_after_gb=floor + 2.0,
                    )
                else:
                    shape_dec.low_vram = True
                hr_coords = shape_dec.upsample(slat_lr, upsample_times=4)
                del slat_lr
                if mgr is None:
                    if hasattr(shape_dec, "cpu"):
                        shape_dec.cpu()
                    self._release_vram(offload_models=False)

                hr_resolution = target_res
                while True:
                    quant_coords = torch.cat(
                        [
                            hr_coords[:, :1],
                            ((hr_coords[:, 1:] + 0.5) / 512 * (hr_resolution // 16)).int(),
                        ],
                        dim=1,
                    )
                    hr_unique = quant_coords.unique(dim=0)
                    del quant_coords
                    if hr_unique.shape[0] < max_num_tokens or hr_resolution == 1024:
                        if hr_resolution != target_res:
                            log.info(
                                "Reduced cascade resolution %d -> %d to fit max_num_tokens=%d (tokens=%d)",
                                target_res,
                                hr_resolution,
                                max_num_tokens,
                                hr_unique.shape[0],
                            )
                        break
                    hr_resolution -= 128
                del hr_coords
                res = hr_resolution
                _log_stage(f"upsample_coords({res})", t_st)

                assert cond_1024_saved is not None
                cond_1024 = (
                    cond_1024_saved
                    if keep_cond_gpu
                    else _move_cond_dict(cond_1024_saved, pipe.device, non_blocking=True)
                )
                del cond_1024_saved
                _emit("sampling_shape_hr")
                t_st = time.monotonic()
                if mgr is not None:
                    mgr.ensure_on_gpu(
                        ["shape_slat_flow_model_1024"],
                        pipe.device,
                        required_free_gb=floor + 2.0,
                        protected=["tex_slat_flow_model_1024"],
                    )
                    mgr.prefetch_async(
                        ["tex_slat_flow_model_1024"], pipe.device, min_free_after_gb=floor + 2.5
                    )
                shape_slat = pipe.sample_shape_slat(
                    cond_1024, pipe.models["shape_slat_flow_model_1024"], hr_unique, {}
                )
                del hr_unique
                if mgr is None:
                    self._release_vram(offload_models=True)
                _log_stage(f"sample_shape_slat({res})", t_st)

                _emit("sampling_tex_hr")
                t_st = time.monotonic()
                if mgr is not None:
                    mgr.ensure_on_gpu(
                        ["tex_slat_flow_model_1024"],
                        pipe.device,
                        required_free_gb=floor + 2.0,
                        protected=["shape_slat_decoder", "tex_slat_decoder"],
                    )
                    mgr.prefetch_async(
                        ["shape_slat_decoder", "tex_slat_decoder"],
                        pipe.device,
                        min_free_after_gb=floor + 2.5,
                    )
                tex_slat = pipe.sample_tex_slat(
                    cond_1024, pipe.models["tex_slat_flow_model_1024"], shape_slat, {}
                )
                del cond_1024
                if mgr is None:
                    self._release_vram(offload_models=True)
                _log_stage(f"sample_tex_slat({res})", t_st)
            else:
                raise ValueError(f"Invalid pipeline type: {pipeline_type}")

            decode_headroom_gb = floor + (7.5 if res >= 1536 else 4.5)
            if hasattr(tex_slat, "clear_spatial_cache"):
                tex_slat.clear_spatial_cache()
            free_before_shape = _cuda_free_gb(torch)
            spill_tex_slat = bool(
                mgr is None
                or free_before_shape is None
                or free_before_shape < (decode_headroom_gb + 0.3)
            )
            tex_slat_staged = tex_slat.cpu() if spill_tex_slat else tex_slat
            del tex_slat
            if mgr is None:
                self._release_vram(offload_models=True)

            _emit("decoding_geometry")
            t_st = time.monotonic()
            shape_dec_mod = pipe.models.get("shape_slat_decoder")
            if mgr is not None:
                mgr.ensure_on_gpu(
                    ["shape_slat_decoder"],
                    pipe.device,
                    required_free_gb=decode_headroom_gb,
                    protected=["tex_slat_decoder"],
                )
                self._configure_decoder_residency(shape_dec_mod)
                mgr.prefetch_async(
                    ["tex_slat_decoder"], pipe.device, min_free_after_gb=decode_headroom_gb
                )
            meshes, subs = pipe.decode_shape_slat(shape_slat, res)
            del shape_slat
            if mgr is None:
                self._release_vram(offload_models=True)
            _log_stage(f"decode_shape_slat({res})", t_st)

            m = meshes[0]
            del meshes

            # Stage tex_slat_decoder before deciding whether to overlap CuMesh.fill_holes()
            # so any LRU eviction + empty_cache() finishes before concurrent CUDA work starts.
            tex_dec = pipe.models.get("tex_slat_decoder")
            if mgr is not None:
                mgr.ensure_on_gpu(
                    ["tex_slat_decoder"],
                    pipe.device,
                    required_free_gb=floor + (6.0 if res >= 1536 else 3.5),
                )
                self._configure_decoder_residency(tex_dec)
            elif tex_dec is not None:
                tex_dec.low_vram = True

            free_after_shape = _cuda_free_gb(torch)
            overlap_min_free = floor + (8.0 if res >= 1536 else 5.0)
            can_overlap_fill = bool(
                mgr is not None
                and os.environ.get("TRELLIS2_FAST_VRAM", "1") != "0"
                and os.environ.get("TRELLIS2_STREAM_OVERLAP", "1") != "0"
                and torch is not None
                and torch.cuda.is_available()
                and hasattr(torch.cuda, "Stream")
                and hasattr(torch.cuda, "Event")
                and free_after_shape is not None
                and free_after_shape >= overlap_min_free
            )
            fill_thread: threading.Thread | None = None
            fill_stream: Any = None
            fill_err: list[BaseException | None] = [None]
            verts_staged: Any = None
            faces_staged: Any = None
            spill_mesh_geom = False

            if can_overlap_fill:
                shape_done_evt = torch.cuda.Event()
                shape_done_evt.record(torch.cuda.current_stream())
                fill_stream = torch.cuda.Stream()

                def _fill_holes_async() -> None:
                    try:
                        with torch.cuda.stream(fill_stream):
                            fill_stream.wait_event(shape_done_evt)
                            m.fill_holes()
                    except BaseException as exc:
                        fill_err[0] = exc

                fill_thread = threading.Thread(
                    target=_fill_holes_async, name="trellis2-fill-holes", daemon=True
                )
                fill_thread.start()
            else:
                m.fill_holes()
                free_after_shape = _cuda_free_gb(torch)
                spill_mesh_geom = bool(
                    mgr is None
                    or free_after_shape is None
                    or free_after_shape < (floor + 3.0)
                )
                verts_staged = m.vertices.cpu() if spill_mesh_geom else m.vertices
                faces_staged = m.faces.cpu() if spill_mesh_geom else m.faces
                del m
                if mgr is None:
                    self._release_vram(offload_models=False)

            _emit("decoding_textures")
            t_st = time.monotonic()
            try:
                tex_slat_gpu = tex_slat_staged.to(pipe.device) if spill_tex_slat else tex_slat_staged
                del tex_slat_staged
                tex_voxels = pipe.decode_tex_slat(tex_slat_gpu, subs)
                del tex_slat_gpu, subs
                if mgr is None:
                    self._release_vram(offload_models=True)
            finally:
                if fill_thread is not None:
                    fill_thread.join()
                    if fill_stream is not None:
                        torch.cuda.current_stream().wait_stream(fill_stream)
                    if fill_err[0] is not None:
                        raise fill_err[0]
            _log_stage(f"decode_tex_slat({res})", t_st)

            if can_overlap_fill:
                verts_staged = m.vertices
                faces_staged = m.faces
                del m

            v = tex_voxels[0]
            del tex_voxels
            free_gb = _cuda_free_gb(torch)
            spill_voxels = bool(free_gb is not None and free_gb < (floor + 1.5))
            v_coords = v.coords[:, 1:].cpu() if spill_voxels else v.coords[:, 1:].contiguous()
            v_attrs = v.feats.cpu() if spill_voxels else v.feats.contiguous()
            v_shape = torch.Size([*v.shape, *v.spatial_shape])
            del v
            if spill_voxels:
                self._release_vram(offload_models=False)

            mesh_with_voxel_cls = getattr(pipe, "_mesh_with_voxel_cls", None)
            if mesh_with_voxel_cls is None:
                from trellis2.representations import MeshWithVoxel as mesh_with_voxel_cls

            out_verts = (
                verts_staged.to(pipe.device)
                if (spill_mesh_geom and not spill_voxels)
                else (verts_staged.cpu() if spill_voxels else verts_staged)
            )
            out_faces = (
                faces_staged.to(pipe.device)
                if (spill_mesh_geom and not spill_voxels)
                else (faces_staged.cpu() if spill_voxels else faces_staged)
            )
            return mesh_with_voxel_cls(
                out_verts,
                out_faces,
                origin=[-0.5, -0.5, -0.5],
                voxel_size=1.0 / res,
                coords=v_coords,
                attrs=v_attrs,
                voxel_shape=v_shape,
                layout=pipe.pbr_attr_layout,
            )
        finally:
            pipe.low_vram = saved_pipe_low_vram

    def _run_once(
        self,
        image: Image.Image,
        params: GenerationParams,
        out_path: Path,
        pipeline_type: str,
        on_stage: Callable[[str], None] | None = None,
    ) -> Path:
        def _emit(stage_name: str) -> None:
            if on_stage is not None:
                on_stage(stage_name)

        mesh = None
        glb = None
        succeeded = False
        infer_ctx = (
            self._torch.inference_mode()
            if self._torch is not None and hasattr(self._torch, "inference_mode")
            else nullcontext()
        )
        try:
            with infer_ctx:
                _emit("diffusion_inference")
                log.info(
                    "Running TRELLIS.2 pipeline_type=%s seed=%d decimation=%d texture=%d remesh=%s",
                    pipeline_type,
                    params.seed,
                    params.decimation_target,
                    params.texture_size,
                    params.remesh,
                )
                if self._can_run_staged():
                    mesh = self._run_pipeline_staged(
                        image, params, pipeline_type, on_stage=on_stage
                    )
                else:
                    mesh = self._pipeline.run(
                        image,
                        seed=params.seed,
                        preprocess_image=params.preprocess_image,
                        pipeline_type=pipeline_type,
                    )[0]

                # Ensure >= (4 GiB floor + 2.5 GiB = 6.5 GiB) free for cuBVH/CuMesh/nvdiffrast baking
                if self._model_mgr is not None:
                    self._release_vram(min_free_gb=VRAM_SAFETY_FLOOR_GB + 2.5)
                else:
                    self._release_vram(offload_models=True)

                _emit("simplifying_mesh")
                mesh.simplify(16777216)  # exact upstream nvdiffrast hardware face limit (no-op when <=16.77M)

                _emit("baking_textures")
                t_bake = time.monotonic()
                use_cuda = bool(self._torch is not None and self._torch.cuda.is_available())
                attr_vol = mesh.attrs.cuda() if use_cuda and hasattr(mesh.attrs, "cuda") else mesh.attrs
                vox_coords = mesh.coords.cuda() if use_cuda and hasattr(mesh.coords, "cuda") else mesh.coords
                glb = self._o_voxel.postprocess.to_glb(
                    vertices=mesh.vertices,
                    faces=mesh.faces,
                    attr_volume=attr_vol,
                    coords=vox_coords,
                    attr_layout=mesh.layout,
                    voxel_size=mesh.voxel_size,
                    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                    decimation_target=params.decimation_target,
                    texture_size=params.texture_size,
                    remesh=params.remesh,
                    remesh_band=params.remesh_band,
                    remesh_project=params.remesh_project,
                    verbose=True,
                )
                del attr_vol, vox_coords
                mesh = None
                log.info("Stage %-26s done in %.2fs", "baking_textures(to_glb)", time.monotonic() - t_bake)
                if self._model_mgr is not None:
                    self._release_vram(min_free_gb=VRAM_SAFETY_FLOOR_GB + 1.5)
                else:
                    self._release_vram(offload_models=False)

                _emit("exporting_glb")
                t_exp = time.monotonic()
                local_tmp = Path(tempfile.gettempdir()) / f"trellis2_{uuid.uuid4().hex[:10]}.glb"
                staging_tmp: Path | None = None
                try:
                    with _FastLosslessPngContext():
                        glb.export(str(local_tmp), extension_webp=params.extension_webp)
                    log.info("Stage %-26s done in %.2fs", "exporting_glb", time.monotonic() - t_exp)
                    try:
                        _atomic_replace(local_tmp, out_path)
                    except OSError:
                        staging_tmp = out_path.parent / f".{out_path.name}.{uuid.uuid4().hex[:6]}.tmp"
                        shutil.copyfile(local_tmp, staging_tmp)
                        _atomic_replace(staging_tmp, out_path)
                        staging_tmp = None
                finally:
                    local_tmp.unlink(missing_ok=True)
                    if staging_tmp is not None:
                        staging_tmp.unlink(missing_ok=True)
                _persist_flex_gemm_autotune_cache()
                succeeded = True
                return out_path
        finally:
            del mesh, glb
            if self._model_mgr is not None and succeeded:
                self._release_vram(min_free_gb=VRAM_SAFETY_FLOOR_GB + 1.5)
            else:
                self._release_vram(offload_models=True)

    def generate_glb(
        self,
        image: Image.Image,
        params: GenerationParams,
        out_path: Path,
        on_stage: Callable[[str], None] | None = None,
    ) -> Path:
        if self.status != "ready" or self._pipeline is None:
            raise RuntimeError(f"Model is not ready (status={self.status}, error={self.error})")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        allow_fallback = os.environ.get("TRELLIS2_OOM_FALLBACK", "1") != "0"
        current_type = params.pipeline_type
        with self._gpu_lock:
            while True:
                try:
                    return self._run_once(
                        image, params, out_path, current_type, on_stage=on_stage
                    )
                except Exception as exc:
                    next_type = OOM_FALLBACK_CHAIN.get(current_type)
                    if (
                        self._is_cuda_oom(exc)
                        and self.low_vram
                        and allow_fallback
                        and next_type is not None
                    ):
                        log.warning(
                            "CUDA OOM on pipeline_type=%s; clearing VRAM and retrying with %s.",
                            current_type,
                            next_type,
                        )
                        if on_stage is not None:
                            on_stage(f"retrying_{next_type}_after_oom")
                        self._release_vram(offload_models=True)
                        current_type = next_type
                        continue
                    raise


@dataclass
class JobRecord:
    job_id: str
    status: str  # queued | loading_weights | running | completed | failed
    stage: str
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    stage_started_at: float | None = None
    effective_pipeline_type: str | None = None
    glb_path: Path | None = None
    error: str | None = None

    def elapsed_s(self) -> float:
        end = self.finished_at if self.finished_at is not None else time.time()
        start = self.started_at if self.started_at is not None else self.created_at
        return round(max(0.0, end - start), 2)

    def stage_elapsed_s(self) -> float:
        end = self.finished_at if self.finished_at is not None else time.time()
        start = (
            self.stage_started_at
            if self.stage_started_at is not None
            else (self.started_at if self.started_at is not None else self.created_at)
        )
        return round(max(0.0, end - start), 2)

    def total_elapsed_s(self) -> float:
        end = self.finished_at if self.finished_at is not None else time.time()
        return round(max(0.0, end - self.created_at), 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "stage": self.stage,
            "elapsed_s": self.elapsed_s(),
            "stage_elapsed_s": self.stage_elapsed_s(),
            "total_elapsed_s": self.total_elapsed_s(),
            "effective_pipeline_type": self.effective_pipeline_type,
            "error": self.error,
            "download_url": f"/jobs/{self.job_id}/glb" if self.status == "completed" else None,
        }


def _atomic_replace(src: Path, dst: Path, retries: int = 8) -> None:
    """Replace dst with src, retrying brief Windows/NFS sharing violations."""
    for attempt in range(retries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(0.01 * (attempt + 1))


class JobManager:
    """Single-worker async job queue with disk-backed status and bounded GLB retention."""

    def __init__(
        self,
        engine: InferenceEngine,
        jobs_dir: Path,
        max_retained_jobs: int = MAX_RETAINED_JOBS,
    ) -> None:
        self._engine = engine
        self._jobs_dir = jobs_dir
        self._max_retained_jobs = max(1, max_retained_jobs)
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="trellis2-job")
        try:
            self._supports_on_stage = "on_stage" in inspect.signature(engine.generate_glb).parameters
        except (ValueError, TypeError):
            self._supports_on_stage = False

    def _write_status(self, record: JobRecord) -> None:
        job_dir = self._jobs_dir / record.job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "job_id": record.job_id,
            "status": record.status,
            "stage": record.stage,
            "created_at": record.created_at,
            "started_at": record.started_at,
            "finished_at": record.finished_at,
            "stage_started_at": record.stage_started_at,
            "effective_pipeline_type": record.effective_pipeline_type,
            "error": record.error,
        }
        tmp_path = job_dir / f"status.{uuid.uuid4().hex[:6]}.tmp"
        tmp_path.write_text(json.dumps(payload), encoding="utf-8")
        try:
            _atomic_replace(tmp_path, job_dir / "status.json")
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    def _read_status(self, job_id: str) -> JobRecord | None:
        if not _JOB_ID_RE.match(job_id):
            return None
        job_dir = self._jobs_dir / job_id
        status_file = job_dir / "status.json"
        if not status_file.is_file():
            return None
        try:
            data = json.loads(status_file.read_text(encoding="utf-8"))
            return JobRecord(
                job_id=data["job_id"],
                status=data["status"],
                stage=data["stage"],
                created_at=float(data["created_at"]),
                started_at=float(data["started_at"]) if data.get("started_at") is not None else None,
                finished_at=float(data["finished_at"]) if data.get("finished_at") is not None else None,
                stage_started_at=(
                    float(data["stage_started_at"]) if data.get("stage_started_at") is not None else None
                ),
                effective_pipeline_type=data.get("effective_pipeline_type"),
                glb_path=job_dir / "mesh.glb",
                error=data.get("error"),
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            log.warning("Failed to read job status for %s: %s", job_id, exc)
            return None

    def submit(self, image: Image.Image, params: GenerationParams) -> JobRecord:
        job_id = uuid.uuid4().hex[:12]
        job_dir = self._jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        out_path = job_dir / "mesh.glb"
        now = time.time()

        if self._engine.status == "loading":
            initial_status = "loading_weights"
            initial_stage = self._engine.stage or "loading_weights"
        else:
            initial_status = "queued"
            initial_stage = "queued"

        record = JobRecord(
            job_id=job_id,
            status=initial_status,
            stage=initial_stage,
            created_at=now,
            stage_started_at=now,
            effective_pipeline_type=params.pipeline_type,
            glb_path=out_path,
        )
        with self._lock:
            self._jobs[job_id] = record
            self._write_status(record)
            self._prune_locked()

        self._executor.submit(self._run_job, job_id, image, params, out_path)
        return record

    def _run_job(
        self, job_id: str, image: Image.Image, params: GenerationParams, out_path: Path
    ) -> None:
        last_stage: str | None = None
        while self._engine.status == "loading":
            if self._stop_event.is_set():
                return
            current_stage = self._engine.stage or "loading_weights"
            if current_stage != last_stage:
                with self._lock:
                    record = self._jobs.get(job_id)
                    if record is None:
                        return
                    record.status = "loading_weights"
                    record.stage = current_stage
                    record.stage_started_at = time.time()
                    self._write_status(record)
                last_stage = current_stage
            self._stop_event.wait(0.05)

        if self._engine.status == "error":
            now = time.time()
            with self._lock:
                record = self._jobs.get(job_id)
                if record is None:
                    return
                record.status = "failed"
                record.stage = "failed"
                record.stage_started_at = now
                record.error = f"TRELLIS.2 failed to initialize: {self._engine.error}"
                record.finished_at = now
                self._write_status(record)
            return

        now = time.time()
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return
            record.status = "running"
            record.stage = "generating"
            record.started_at = now
            record.stage_started_at = now
            self._write_status(record)

        def _update_stage(stage_name: str) -> None:
            with self._lock:
                rec = self._jobs.get(job_id)
                if rec is None:
                    return
                rec.stage = stage_name
                rec.stage_started_at = time.time()
                if stage_name.startswith("retrying_") and stage_name.endswith("_after_oom"):
                    rec.effective_pipeline_type = stage_name[len("retrying_") : -len("_after_oom")]
                self._write_status(rec)

        # Must end in .glb because trimesh.export() infers format from the file extension
        tmp_glb = out_path.parent / ".mesh.tmp.glb"
        try:
            if self._supports_on_stage:
                self._engine.generate_glb(image, params, tmp_glb, on_stage=_update_stage)
            else:
                self._engine.generate_glb(image, params, tmp_glb)
            _atomic_replace(tmp_glb, out_path)
            done_ts = time.time()
            with self._lock:
                record.status = "completed"
                record.stage = "completed"
                record.stage_started_at = done_ts
                record.finished_at = done_ts
                self._write_status(record)
        except Exception as exc:
            tmp_glb.unlink(missing_ok=True)
            log.exception("Job %s failed", job_id)
            fail_ts = time.time()
            with self._lock:
                record.status = "failed"
                record.stage = "failed"
                record.stage_started_at = fail_ts
                record.error = f"{type(exc).__name__}: {exc}"
                record.finished_at = fail_ts
                self._write_status(record)

    def queue_position(self, job_id: str) -> int:
        """Return 1-based queue position for queued/loading jobs, or 0 otherwise."""
        with self._lock:
            pending = sorted(
                (r for r in self._jobs.values() if r.status in {"queued", "loading_weights"}),
                key=lambda r: r.created_at,
            )
            for idx, rec in enumerate(pending, start=1):
                if rec.job_id == job_id:
                    return idx
        return 0

    def stats(self) -> dict[str, Any]:
        """Return current active job and queue depth."""
        with self._lock:
            active_id = next((r.job_id for r in self._jobs.values() if r.status == "running"), None)
            queued = sum(1 for r in self._jobs.values() if r.status in {"queued", "loading_weights"})
        return {"active_job_id": active_id, "queued_jobs": queued}

    def get(self, job_id: str) -> JobRecord | None:
        if not _JOB_ID_RE.match(job_id):
            return None
        with self._lock:
            record = self._jobs.get(job_id)
            if record is not None:
                return record
        return self._read_status(job_id)

    def _prune_locked(self) -> None:
        if not self._jobs_dir.is_dir():
            return
        entries: list[tuple[float, str, Path, str | None]] = []
        for child in self._jobs_dir.iterdir():
            if not child.is_dir() or not _JOB_ID_RE.match(child.name):
                continue
            rec = self._jobs.get(child.name) or self._read_status(child.name)
            status_val = rec.status if rec is not None else None
            created = rec.created_at if rec is not None else child.stat().st_mtime
            entries.append((created, child.name, child, status_val))

        if len(entries) <= self._max_retained_jobs:
            return
        entries.sort(key=lambda item: item[0])
        excess = len(entries) - self._max_retained_jobs
        for _, jid, jdir, status_val in entries:
            if excess <= 0:
                break
            if status_val in {"queued", "loading_weights", "running"}:
                continue
            self._jobs.pop(jid, None)
            shutil.rmtree(jdir, ignore_errors=True)
            excess -= 1

    def shutdown(self) -> None:
        self._stop_event.set()
        self._executor.shutdown(wait=True, cancel_futures=True)


def _default_jobs_dir() -> Path:
    env_dir = os.environ.get("TRELLIS2_JOBS_DIR")
    if env_dir:
        return Path(env_dir)
    for root in (Path("/runpod-volume"), Path("/workspace")):
        if root.is_dir() and os.access(root, os.W_OK):
            return root / "jobs"
    return Path(tempfile.gettempdir()) / "trellis2_jobs"


def _decode_uploaded_image(raw: bytes) -> Image.Image:
    if not raw:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded image is empty.")
    try:
        with Image.open(io.BytesIO(raw)) as probe:
            probe.verify()
        with Image.open(io.BytesIO(raw)) as img:
            img.load()
            has_alpha = img.mode in ("RGBA", "LA") or ("transparency" in img.info)
            converted = img.convert("RGBA" if has_alpha else "RGB")
            if max(converted.size) > MAX_INPUT_DIMENSION:
                converted.thumbnail((MAX_INPUT_DIMENSION, MAX_INPUT_DIMENSION), Image.Resampling.LANCZOS)
            return converted
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image upload: {exc}",
        ) from exc


def _require_ready_engine(engine: InferenceEngine) -> None:
    if engine.status == "ready":
        return
    if engine.status == "error":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"TRELLIS.2 failed to initialize: {engine.error}",
        )
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=f"TRELLIS.2 is still loading ({engine.stage}). Retry once /health is ready.",
    )


def create_app(
    engine: InferenceEngine | None = None,
    jobs_dir: Path | None = None,
    api_key: str | None = None,
    autoload: bool | None = None,
) -> FastAPI:
    """Create the FastAPI application with optional dependency overrides for testing."""
    resolved_engine = engine if engine is not None else Trellis2Engine()
    resolved_jobs_dir = jobs_dir if jobs_dir is not None else _default_jobs_dir()
    resolved_api_key = os.environ.get("API_KEY", "") if api_key is None else api_key
    should_autoload = (
        os.environ.get("TRELLIS2_SKIP_AUTOLOAD", "0") != "1" if autoload is None else autoload
    )

    manager = JobManager(resolved_engine, resolved_jobs_dir)

    def verify_api_key(
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> None:
        if not resolved_api_key:
            return
        candidate = x_api_key or ""
        if not candidate and authorization and authorization.lower().startswith("bearer "):
            candidate = authorization[7:].strip()
        if not candidate or not secrets.compare_digest(candidate, resolved_api_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid API key.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        resolved_jobs_dir.mkdir(parents=True, exist_ok=True)
        if should_autoload:
            resolved_engine.start_loading()
        try:
            yield
        finally:
            manager.shutdown()

    fastapi_app = FastAPI(
        title="TRELLIS.2 RunPod Inference Server",
        version="1.0.0",
        lifespan=lifespan,
    )

    @fastapi_app.get("/ping")
    def ping() -> Response:
        if resolved_engine.status in {"ready", "loading"}:
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={"status": "healthy", "model_status": resolved_engine.status},
            )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "error", "error": resolved_engine.error},
        )

    @fastapi_app.get("/health")
    def health() -> dict[str, Any]:
        info = resolved_engine.health_info()
        info.update(manager.stats())
        return info

    @fastapi_app.post("/jobs", status_code=status.HTTP_202_ACCEPTED, dependencies=[Depends(verify_api_key)])
    async def create_job(
        image: UploadFile = File(...),
        params: GenerationParams = Depends(parse_generation_params),
    ) -> dict[str, Any]:
        if resolved_engine.status == "error":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"TRELLIS.2 failed to initialize: {resolved_engine.error}",
            )
        pil_image = _decode_uploaded_image(await image.read())
        record = manager.submit(pil_image, params)
        return {
            "job_id": record.job_id,
            "status": record.status,
            "stage": record.stage,
            "model_status": resolved_engine.status,
            "status_url": f"/jobs/{record.job_id}",
            "download_url": f"/jobs/{record.job_id}/glb",
        }

    @fastapi_app.get("/jobs/{job_id}", dependencies=[Depends(verify_api_key)])
    def get_job(job_id: str) -> dict[str, Any]:
        record = manager.get(job_id)
        if record is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown job: {job_id}")
        payload = record.to_dict()
        payload["queue_position"] = manager.queue_position(job_id)
        payload["model_status"] = resolved_engine.status
        payload["model_stage"] = resolved_engine.stage
        if payload.get("status") == "loading_weights":
            payload["stage"] = resolved_engine.stage or payload.get("stage")
        return payload

    @fastapi_app.get("/jobs/{job_id}/glb", dependencies=[Depends(verify_api_key)])
    def download_job_glb(job_id: str) -> FileResponse:
        record = manager.get(job_id)
        if record is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown job: {job_id}")
        if record.status != "completed" or record.glb_path is None or not record.glb_path.is_file():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Job {job_id} is not completed (status={record.status}, error={record.error}).",
            )
        return FileResponse(
            path=record.glb_path,
            media_type="model/gltf-binary",
            filename=f"{job_id}.glb",
        )

    @fastapi_app.post("/generate", dependencies=[Depends(verify_api_key)])
    async def generate_sync(
        image: UploadFile = File(...),
        params: GenerationParams = Depends(parse_generation_params),
    ) -> Response:
        _require_ready_engine(resolved_engine)
        pil_image = _decode_uploaded_image(await image.read())
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "mesh.glb"
            resolved_engine.generate_glb(pil_image, params, out_path)
            payload = out_path.read_bytes()
        return Response(
            content=payload,
            media_type="model/gltf-binary",
            headers={"Content-Disposition": 'attachment; filename="mesh.glb"'},
        )

    return fastapi_app


app = create_app()
