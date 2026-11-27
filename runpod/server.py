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
    """Point Triton and FlexGEMM autotune caches to writable storage without starving 20 GB volumes."""
    storage_root: Path | None = None
    min_volume_free_bytes = int(1.5 * (1024**3))
    for candidate in ("/runpod-volume", "/workspace"):
        p = Path(candidate)
        if not (p.is_dir() and os.access(p, os.W_OK)):
            continue
        if candidate == "/runpod-volume":
            try:
                if shutil.disk_usage(p).free < min_volume_free_bytes:
                    continue
            except OSError:
                continue
        storage_root = p
        break
    if storage_root is not None:
        cache_base = storage_root
    elif Path("/var/tmp").is_dir() and os.access("/var/tmp", os.W_OK):
        cache_base = Path("/var/tmp") / "trellis2_cache"
    else:
        cache_base = Path(tempfile.gettempdir())
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


def _cleanup_stale_mmap_files() -> None:
    """Remove any leftover trellis2_stage_*.bin mmap scratch files from prior crashed runs."""
    candidates = [
        Path(os.environ.get("TRELLIS2_MMAP_DIR", "").strip()) if os.environ.get("TRELLIS2_MMAP_DIR", "").strip() else None,
        Path("/var/tmp"),
        Path("/tmp"),
        Path(tempfile.gettempdir()),
        Path("/workspace/.trellis2_mmap"),
        Path("/runpod-volume/.trellis2_mmap"),
    ]
    seen: set[str] = set()
    for base in candidates:
        if base is None or not base.is_dir():
            continue
        key = str(base)
        if key in seen:
            continue
        seen.add(key)
        try:
            for p in base.glob("trellis2_stage_*.bin"):
                try:
                    p.unlink()
                except OSError:
                    pass
        except OSError:
            pass


def _is_already_patched(target: Any, attr: str = "_trellis2_vram_patched") -> bool:
    """Check patching flag on target's own __dict__ so MRO inheritance never skips subclass patches."""
    try:
        return bool(vars(target).get(attr, False))
    except TypeError:
        return bool(getattr(target, attr, False))


def patch_flex_gemm_autotune_cache_safety() -> int:
    """Make FlexGEMM autotune cache saves non-fatal on Disk quota exceeded (EDQUOT) / ENOSPC."""
    mod_names = (
        "flex_gemm.utils.autotuner",
        "flex_gemm.ops.spconv",
        "flex_gemm.kernels.triton.spconv.config",
        "flex_gemm.autotuner",
        "flex_gemm.autotune",
    )
    fn_names = ("save_autotune_cache", "dump_autotune_cache", "save_cache")
    patched = 0
    for mod_name in mod_names:
        mod = sys.modules.get(mod_name)
        if mod is None:
            try:
                mod = importlib.import_module(mod_name)
            except Exception:
                continue
        for fn_name in fn_names:
            orig_fn = getattr(mod, fn_name, None)
            if not callable(orig_fn) or getattr(orig_fn, "_trellis2_nonfatal_cache_save", False):
                continue

            def _safe_save(*args: Any, _orig: Callable[..., Any] = orig_fn, _name: str = f"{mod_name}.{fn_name}", **kwargs: Any) -> Any:
                try:
                    return _orig(*args, **kwargs)
                except OSError as exc:
                    cache_path = os.environ.get("FLEX_GEMM_AUTOTUNE_CACHE_PATH", "")
                    if cache_path:
                        try:
                            tmp_p = Path(cache_path + ".tmp")
                            if tmp_p.exists():
                                tmp_p.unlink()
                        except OSError:
                            pass
                    logging.getLogger("trellis2.server").warning(
                        "Skipping FlexGEMM autotune cache save (%s): %s", _name, exc
                    )
                    return None
                except Exception as exc:
                    logging.getLogger("trellis2.server").warning(
                        "Skipping FlexGEMM autotune cache save (%s): %s", _name, exc
                    )
                    return None

            _safe_save._trellis2_nonfatal_cache_save = True  # type: ignore[attr-defined]
            setattr(mod, fn_name, _safe_save)
            patched += 1
    return patched


def _flush_flex_gemm_autotune_cache() -> None:
    """Best-effort immediate flush of FlexGEMM Triton autotune cache to persistent storage."""
    for mod_name in (
        "flex_gemm.utils.autotuner",
        "flex_gemm.ops.spconv",
        "flex_gemm.kernels.triton.spconv.config",
        "flex_gemm.autotuner",
        "flex_gemm.autotune",
    ):
        try:
            mod = sys.modules.get(mod_name) or importlib.import_module(mod_name)
        except Exception:
            continue
        for fn_name in ("save_autotune_cache", "dump_autotune_cache", "save_cache"):
            fn = getattr(mod, fn_name, None)
            if callable(fn):
                try:
                    fn()
                    return
                except Exception:
                    pass


import gc
import importlib
import inspect
import io
import itertools
import json
import logging
import re
import secrets
import shutil
import sys
import tempfile
import threading
import time
import traceback
import uuid
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

_configure_persistent_kernel_caches()
_cleanup_stale_mmap_files()

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
DEFAULT_MAX_NUM_TOKENS = max(16_384, int(os.environ.get("TRELLIS2_MAX_NUM_TOKENS", "81920")))
BATCH_CFG_MAX_SPARSE_TOKENS = max(4_096, int(os.environ.get("TRELLIS2_BATCH_CFG_MAX_TOKENS", "24576")))
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


_CUDART_LIB_CACHE: dict[str, Any] = {"loaded": False, "lib": None, "libs": []}


def _is_oom_like(exc: BaseException) -> bool:
    """Return True for PyTorch or C++ extension (e.g. CuMesh) CUDA out-of-memory errors."""
    return "out of memory" in str(exc).lower() or type(exc).__name__ == "OutOfMemoryError"


def _clear_sticky_cuda_error(torch_mod: Any, cudart_lib: Any = None) -> int:
    """Consume any pending recoverable CUDA runtime error (e.g. raw cudaMalloc OOM in CuMesh).

    Third-party C++ extensions such as CuMesh may link a different libcudart DSO than PyTorch's
    bundled wheel runtime and throw TORCH_CHECK on raw cudaMalloc() without clearing
    cudaGetLastError(), leaving cudaErrorMemoryAllocation armed across subsequent CUDA calls.
    """
    last_err = 0
    if cudart_lib is not None:
        libs = [cudart_lib]
    else:
        if not _CUDART_LIB_CACHE["loaded"]:
            _CUDART_LIB_CACHE["loaded"] = True
            import ctypes

            cands: list[str | None] = (
                [None] if os.name != "nt" else []
            ) + ["libcudart.so", "libcudart.so.12", "libcudart.so.11.0"]
            try:
                maps_p = Path("/proc/self/maps")
                if maps_p.is_file():
                    for line in maps_p.read_text(encoding="utf-8", errors="ignore").splitlines():
                        if "libcudart" in line and ".so" in line:
                            so_path = line.split()[-1]
                            if so_path not in cands:
                                cands.append(so_path)
            except Exception:
                pass
            loaded_libs: list[Any] = []
            seen_handles: set[int] = set()
            for cand in cands:
                try:
                    handle = ctypes.CDLL(cand)
                    h_id = getattr(handle, "_handle", id(handle))
                    if h_id not in seen_handles and callable(getattr(handle, "cudaGetLastError", None)):
                        seen_handles.add(h_id)
                        loaded_libs.append(handle)
                except Exception:
                    continue
            _CUDART_LIB_CACHE["libs"] = loaded_libs
            _CUDART_LIB_CACHE["lib"] = loaded_libs[0] if loaded_libs else None
        libs = list(_CUDART_LIB_CACHE.get("libs") or [])
        if not libs and _CUDART_LIB_CACHE.get("lib") is not None:
            libs = [_CUDART_LIB_CACHE["lib"]]

    for lib in libs:
        get_last = getattr(lib, "cudaGetLastError", None)
        if callable(get_last):
            try:
                err = int(get_last())
                if err != 0:
                    last_err = err
            except Exception:
                pass

    if torch_mod is not None and hasattr(torch_mod, "cuda") and torch_mod.cuda.is_available():
        sync_fn = getattr(torch_mod.cuda, "synchronize", None)
        if callable(sync_fn):
            try:
                sync_fn()
            except Exception:
                pass
        # Probe PyTorch's internal C10_CUDA_KERNEL_LAUNCH_CHECK slot so any queued async/malloc error
        # is consumed here rather than breaking the next OOM-retry model transfer.
        zeros_fn = getattr(torch_mod, "zeros", None)
        if callable(zeros_fn):
            try:
                probe = zeros_fn((1,), device="cuda")
                if hasattr(probe, "add_"):
                    probe.add_(1)
                del probe
                if callable(sync_fn):
                    sync_fn()
            except Exception:
                pass
    return last_err


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
    if _is_already_patched(layernorm_cls, "_trellis2_vram_patched"):
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
    """Compile a pointwise CUDA kernel when explicitly opted in (TRELLIS2_COMPILE_POINTWISE=1).

    Disabled by default on serverless workers because compiling multiple SparseConvNeXt channel
    widths (1024/512/256/128) on 6 vCPUs adds ~55s of cold Inductor overhead inside upsample_coords,
    whereas eager cuBLAS GEMMs + inplace SiLU already saturate Tensor Cores with 0s compile latency.
    """
    if (
        torch_mod is None
        or os.name == "nt"
        or os.environ.get("TRELLIS2_FAST_VRAM", "1") == "0"
        or os.environ.get("TRELLIS2_TORCH_COMPILE", "1") == "0"
        or os.environ.get("TRELLIS2_COMPILE_POINTWISE", "0") != "1"
        or not callable(getattr(torch_mod, "compile", None))
        or not torch_mod.cuda.is_available()
    ):
        return fn
    try:
        return torch_mod.compile(fn, mode="default", dynamic=True)
    except Exception as exc:
        log.debug("torch.compile unavailable for %s; using eager mode: %s", fn.__name__, exc)
        return fn


def _is_cuda_oom_msg(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cudaerrormemoryallocation" in msg


def _host_ram_stats_gb() -> tuple[float | None, float | None]:
    """Return ``(effective_limit_gb, effective_available_gb)`` respecting Linux cgroup v1/v2 limits."""
    mem_total_b: int | None = None
    mem_avail_b: int | None = None
    try:
        meminfo = Path("/proc/meminfo")
        if meminfo.is_file():
            for line in meminfo.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.startswith("MemTotal:"):
                    mem_total_b = int(line.split()[1]) * 1024
                elif line.startswith("MemAvailable:"):
                    mem_avail_b = int(line.split()[1]) * 1024
    except Exception:
        pass

    cg_limit_b: int | None = None
    cg_used_b: int | None = None
    cg_reclaim_b = 0
    max_sane = 1 << 56  # ignore "max" / unlimited sentinel (~2^63)
    try:
        cg2_max = Path("/sys/fs/cgroup/memory.max")
        cg2_cur = Path("/sys/fs/cgroup/memory.current")
        if cg2_max.is_file() and cg2_cur.is_file():
            raw_max = cg2_max.read_text(encoding="utf-8").strip()
            if raw_max.isdigit():
                val = int(raw_max)
                if 0 < val < max_sane:
                    cg_limit_b = val
            raw_cur = cg2_cur.read_text(encoding="utf-8").strip()
            if raw_cur.isdigit():
                cg_used_b = int(raw_cur)
            cg2_stat = Path("/sys/fs/cgroup/memory.stat")
            if cg2_stat.is_file():
                for line in cg2_stat.read_text(encoding="utf-8", errors="ignore").splitlines():
                    if line.startswith("inactive_file "):
                        cg_reclaim_b = int(line.split()[1])
                        break
        else:
            cg1_lim = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
            cg1_use = Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")
            if cg1_lim.is_file() and cg1_use.is_file():
                val = int(cg1_lim.read_text(encoding="utf-8").strip())
                if 0 < val < max_sane:
                    cg_limit_b = val
                cg_used_b = int(cg1_use.read_text(encoding="utf-8").strip())
                cg1_stat = Path("/sys/fs/cgroup/memory/memory.stat")
                if cg1_stat.is_file():
                    for line in cg1_stat.read_text(encoding="utf-8", errors="ignore").splitlines():
                        if line.startswith("total_inactive_file "):
                            cg_reclaim_b = int(line.split()[1])
                            break
    except Exception:
        pass

    eff_limit_b = mem_total_b
    if cg_limit_b is not None:
        eff_limit_b = cg_limit_b if eff_limit_b is None else min(eff_limit_b, cg_limit_b)

    eff_avail_b = mem_avail_b
    if cg_limit_b is not None and cg_used_b is not None:
        cg_free_b = max(0, cg_limit_b - max(0, cg_used_b - cg_reclaim_b))
        eff_avail_b = cg_free_b if eff_avail_b is None else min(eff_avail_b, cg_free_b)

    limit_gb = None if eff_limit_b is None else eff_limit_b / (1024**3)
    avail_gb = None if eff_avail_b is None else eff_avail_b / (1024**3)
    return limit_gb, avail_gb


def _choose_mmap_scratch_dir(min_free_bytes: int = 2 * (1024**3)) -> Path:
    """Pick a real disk-backed scratch directory (never RAM-backed tmpfs like /dev/shm)."""
    tmpfs_mounts: set[str] = {"/dev/shm", "/run"}
    try:
        mounts = Path("/proc/mounts")
        if mounts.is_file():
            for line in mounts.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.split()
                if len(parts) >= 3 and parts[2] in {"tmpfs", "ramfs"}:
                    tmpfs_mounts.add(parts[1].rstrip("/") or "/")
    except Exception:
        pass

    def _is_tmpfs_path(p: Path) -> bool:
        try:
            resolved = str(p.resolve())
        except Exception:
            resolved = str(p)
        best_mount = ""
        is_ram = False
        try:
            mounts = Path("/proc/mounts")
            if mounts.is_file():
                for line in mounts.read_text(encoding="utf-8", errors="ignore").splitlines():
                    parts = line.split()
                    if len(parts) < 3:
                        continue
                    mp = parts[1]
                    if resolved == mp or resolved.startswith(mp.rstrip("/") + "/") or mp == "/":
                        if len(mp) >= len(best_mount):
                            best_mount = mp
                            is_ram = parts[2] in {"tmpfs", "ramfs"}
                return is_ram
        except Exception:
            pass
        return any(resolved == m or resolved.startswith(m + "/") for m in tmpfs_mounts if m != "/")

    candidates: list[tuple[Path, int]] = []
    env_dir = os.environ.get("TRELLIS2_MMAP_DIR", "").strip()
    if env_dir:
        candidates.append((Path(env_dir), 0))
    # Prefer ephemeral container local disk (/var/tmp, /tmp when non-tmpfs) before persistent volumes
    # so large intermediate mmap staging never fills a tight 20 GB /runpod-volume network volume.
    for local_dir in (Path("/var/tmp"), Path("/tmp"), Path(tempfile.gettempdir())):
        if local_dir.is_dir():
            candidates.append((local_dir, 0))
    if Path("/workspace").is_dir():
        candidates.append((Path("/workspace/.trellis2_mmap"), 1 * (1024**3)))
    if Path("/runpod-volume").is_dir():
        candidates.append((Path("/runpod-volume/.trellis2_mmap"), 4 * (1024**3)))

    seen: set[str] = set()
    for cand, reserve_bytes in candidates:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        try:
            if _is_tmpfs_path(cand):
                continue
            cand.mkdir(parents=True, exist_ok=True)
            if os.access(cand, os.W_OK) and shutil.disk_usage(cand).free >= (min_free_bytes + reserve_bytes):
                return cand
        except Exception:
            continue
    fallback = Path("/var/tmp") if Path("/var/tmp").is_dir() else Path(tempfile.gettempdir())
    return fallback


class _DiskMmapTensor:
    """File-backed mmap tensor wrapper that uses <=128 MiB anonymous host RAM and self-unlinks."""

    def __init__(self, tensor: Any, path: str) -> None:
        self.tensor: Any = tensor
        self.path: str | None = path

    @property
    def shape(self) -> Any:
        return self.tensor.shape

    @property
    def dtype(self) -> Any:
        return self.tensor.dtype

    def is_contiguous(self) -> bool:
        return bool(self.tensor.is_contiguous())

    def __getitem__(self, item: Any) -> Any:
        return self.tensor[item]

    def to(self, device: Any, non_blocking: bool = False) -> Any:
        if str(device) == "cpu":
            return self.tensor
        n = int(self.tensor.shape[0])
        step = 4_194_304
        if n <= step:
            return self.tensor.to(device, non_blocking=non_blocking)
        out = self.tensor.new_empty(self.tensor.shape, device=device)
        for s in range(0, n, step):
            e = min(s + step, n)
            out[s:e].copy_(self.tensor[s:e].to(device, non_blocking=non_blocking))
        return out

    def close(self) -> None:
        self.tensor = None
        p = self.path
        self.path = None
        if p:
            try:
                os.unlink(p)
            except OSError:
                pass

    def __del__(self) -> None:
        self.close()


def _clear_oom_traceback_frames(exc: BaseException) -> None:
    """Release GPU tensors pinned in failed kernel/autotune stack frames before OOM retry."""
    tb = getattr(exc, "__traceback__", None)
    if tb is not None:
        try:
            traceback.clear_frames(tb)
        except Exception:
            pass


def _estimate_subm_conv_needed_bytes(conv_mod: Any, sp_tensor: Any, est_out_bytes: int) -> int:
    """Estimate peak additional VRAM needed for an unchunked SparseConv3d call (including neighbor_cache if unbuilt)."""
    n = int(sp_tensor.feats.shape[0])
    w = getattr(conv_mod, "weight", None)
    has_nc = False
    if w is not None and getattr(w, "ndim", 0) == 5 and hasattr(sp_tensor, "get_spatial_cache"):
        _co, kd, kh, kw, _ci = (int(d) for d in w.shape)
        dil = getattr(conv_mod, "dilation", (1, 1, 1))
        nc_key = f"SubMConv3d_neighbor_cache_{kw}x{kh}x{kd}_dilation{dil}"
        has_nc = sp_tensor.get_spatial_cache(nc_key) is not None
    # Unbuilt neighbor_cache needs ~108B/voxel (neighbor_map) + ~32B/voxel (hashmap/sort/valid_kernel)
    nc_bytes = 0 if has_nc else (n * 144)
    return est_out_bytes + nc_bytes + int(1.5 * (1024**3))


def _stage_tensor_off_gpu(
    src_gpu: Any,
    torch_mod: Any,
    min_free_ram_gb: float = 10.0,
    row_chunk: int = 4_194_304,
) -> Any:
    """Move ``src_gpu`` off GPU into disk-backed mmap (>=256 MiB) so container RAM limits are never exceeded."""
    if not bool(getattr(src_gpu, "is_cuda", False)):
        return src_gpu.clone() if hasattr(src_gpu, "clone") else src_gpu
    nbytes = int(src_gpu.numel()) * int(src_gpu.element_size())
    mode = os.environ.get("TRELLIS2_CONV_MMAP", "auto").strip().lower()
    use_mmap = False
    if mode in {"1", "true", "yes", "always"}:
        use_mmap = callable(getattr(torch_mod, "from_file", None))
    elif mode not in {"0", "false", "no", "off"} and nbytes >= 256 * (1024**2) and callable(getattr(torch_mod, "from_file", None)):
        # Container orchestrators (e.g. RunPod) may expose bare-metal /proc/meminfo while enforcing
        # a stricter container RAM cap externally; always mmap >=256 MiB tensors onto disk page cache.
        use_mmap = True

    if use_mmap:
        fd = -1
        mmap_path = ""
        try:
            scratch_dir = _choose_mmap_scratch_dir(min_free_bytes=max(nbytes * 2, 512 * (1024**2)))
            fd, mmap_path = tempfile.mkstemp(prefix="trellis2_stage_", suffix=".bin", dir=str(scratch_dir))
            os.close(fd)
            fd = -1
            numel = int(src_gpu.numel())
            flat_mmap = torch_mod.from_file(mmap_path, shared=True, size=numel, dtype=src_gpu.dtype)
            mmap_t = flat_mmap.reshape(tuple(src_gpu.shape))
            n = int(src_gpu.shape[0])
            step = max(1, int(row_chunk))
            for r0 in range(0, n, step):
                r1 = min(r0 + step, n)
                mmap_t[r0:r1].copy_(src_gpu[r0:r1].cpu())
            # Hint Linux kernel that file pages are cold/reclaimable so cgroup memory stays low
            posix_fadv = getattr(os, "posix_fadvise", None)
            dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
            if callable(posix_fadv) and dontneed is not None:
                try:
                    adv_fd = os.open(mmap_path, os.O_RDONLY)
                    try:
                        posix_fadv(adv_fd, 0, 0, dontneed)
                    finally:
                        os.close(adv_fd)
                except Exception:
                    pass
            return _DiskMmapTensor(mmap_t, mmap_path)
        except Exception as exc:
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
            if mmap_path:
                try:
                    os.unlink(mmap_path)
                except OSError:
                    pass
            log.debug("Disk mmap staging unavailable (%s); falling back to CPU tensor", exc)

    return src_gpu.cpu() if hasattr(src_gpu, "cpu") else src_gpu.clone()


def _release_staged_obj(obj: Any) -> None:
    if isinstance(obj, _DiskMmapTensor):
        obj.close()


def _release_staged_slices(slices: list[tuple[int, int, Any]] | None) -> None:
    if not slices:
        return
    for _c0, _c1, sl in slices:
        _release_staged_obj(sl)
    slices.clear()


def _chunked_subm_conv3d_slices_to_cpu(
    conv_mod: Any,
    x: Any,
    torch_mod: Any,
    max_slice_bytes: int | None = None,
) -> list[tuple[int, int, Any]] | None:
    """Run SparseConv3d in output-channel slices without exceeding GPU VRAM or container host RAM.

    - Earlier slices ``(c1 < Co)`` stream off GPU into disk-backed ``mmap`` (<=128 MiB anonymous
      host RAM, preventing Docker ``exit code 137`` cgroup OOM kills) or pageable CPU RAM when
      host RAM is ample.
    - The final slice ``(c1 == Co)`` stays directly on GPU (zero host memory, zero PCIe transfer).
    Bit-exact: reuses a single FlexGEMM ``neighbor_cache`` across all slices and halves ``c_step``
    automatically on CUDA OOM.
    """
    w_all = getattr(conv_mod, "weight", None)
    b_all = getattr(conv_mod, "bias", None)
    if w_all is None or getattr(w_all, "ndim", 0) != 5:
        return None
    n = int(x.feats.shape[0])
    co, kd, kh, kw, _ci = (int(d) for d in w_all.shape)
    if n <= 0 or co <= 0:
        return None

    is_cuda = bool(getattr(x.feats, "is_cuda", False))
    grouped_fn = getattr(conv_mod, "_grouped_conv_fn", None)
    dilation = getattr(conv_mod, "dilation", (1, 1, 1))
    if grouped_fn is None:
        if not is_cuda:
            return None
        try:
            import flex_gemm  # type: ignore[import-not-found]
            from flex_gemm.ops.spconv import sparse_submanifold_conv3d  # type: ignore[import-not-found]
            from trellis2.modules.sparse.conv import config as conv_cfg  # type: ignore[import-not-found]
        except Exception:
            return None
        if getattr(conv_cfg, "CONV", "flex_gemm") != "flex_gemm":
            return None
        flex_gemm.ops.spconv.set_algorithm(conv_cfg.FLEX_GEMM_ALGO)
        flex_gemm.ops.spconv.set_hashmap_ratio(conv_cfg.FLEX_GEMM_HASHMAP_RATIO)
        grouped_fn = sparse_submanifold_conv3d

    elem_size = int(x.feats.element_size())
    if max_slice_bytes is None:
        free_gb = _cuda_free_gb(torch_mod) if is_cuda else None
        if free_gb is not None:
            # Target <=40% of currently free VRAM (clamped to 256 MiB .. 1.5 GiB)
            max_slice_bytes = int(min(1.5 * (1024**3), max(256 * (1024**2), free_gb * 0.40 * (1024**3))))
        else:
            max_slice_bytes = 1024**3

    min_c_step = 16 if co >= 16 and co % 16 == 0 else 1
    raw_c_step = max(min_c_step, int(max_slice_bytes // max(1, n * elem_size)))
    if min_c_step == 16:
        c_step = max(16, (raw_c_step // 16) * 16)
    else:
        c_step = max(1, raw_c_step)
    c_step = min(co, c_step)
    if c_step >= co and co >= 32:
        c_step = max(min_c_step, co // 2)

    if is_cuda and hasattr(torch_mod, "cuda") and torch_mod.cuda.is_available():
        torch_mod.cuda.empty_cache()

    neighbor_cache_key = f"SubMConv3d_neighbor_cache_{kw}x{kh}x{kd}_dilation{dilation}"
    neighbor_cache = x.get_spatial_cache(neighbor_cache_key) if hasattr(x, "get_spatial_cache") else None
    spatial_shape = torch_mod.Size([*x.shape, *x.spatial_shape])
    staged_slices: list[tuple[int, int, Any]] = []

    spconv_mod_ref: Any = None
    saved_spconv_algo: Any = None
    if is_cuda:
        try:
            import flex_gemm.ops.spconv as spconv_mod_ref  # type: ignore[import-not-found]

            saved_spconv_algo = getattr(spconv_mod_ref, "ALGORITHM", None)
        except Exception:
            spconv_mod_ref = None

    try:
        c0 = 0
        while c0 < co:
            c1 = min(c0 + c_step, co)
            w_s = w_all[c0:c1].contiguous()
            b_s = None if b_all is None else b_all[c0:c1].contiguous()
            try:
                out_s, nc_ = grouped_fn(
                    x.feats,
                    x.coords,
                    spatial_shape,
                    w_s,
                    b_s,
                    neighbor_cache,
                    dilation,
                )
            except RuntimeError as exc:
                del w_s, b_s
                if not _is_cuda_oom_msg(exc) or c_step <= min_c_step:
                    raise
                _clear_oom_traceback_frames(exc)
                if hasattr(torch_mod, "cuda") and torch_mod.cuda.is_available():
                    torch_mod.cuda.empty_cache()
                next_step = max(min_c_step, (c_step // 2) // min_c_step * min_c_step)
                if next_step >= c_step:
                    raise
                log.warning(
                    "Chunked SubMConv3d slice OOM at c_step=%d (N=%d, Co=%d); retrying with c_step=%d",
                    c_step,
                    n,
                    co,
                    next_step,
                )
                c_step = next_step
                continue

            del w_s, b_s
            if neighbor_cache is None and nc_ is not None:
                neighbor_cache = nc_
                if hasattr(x, "register_spatial_cache"):
                    x.register_spatial_cache(neighbor_cache_key, neighbor_cache)

            if is_cuda and c1 == co:
                # Final slice stays directly on GPU — zero host RAM, zero disk IO, zero H2D transfer
                staged_slices.append((c0, c1, out_s))
            else:
                staged = _stage_tensor_off_gpu(out_s, torch_mod)
                del out_s
                staged_slices.append((c0, c1, staged))
            c0 = c1
    except BaseException:
        _release_staged_slices(staged_slices)
        raise
    finally:
        if spconv_mod_ref is not None and saved_spconv_algo is not None:
            spconv_mod_ref.ALGORITHM = saved_spconv_algo

    return staged_slices


def _chunked_subm_conv3d_to_cpu(
    conv_mod: Any,
    x: Any,
    torch_mod: Any,
    max_slice_bytes: int | None = None,
) -> Any | None:
    """Return a single ``(N, Co)`` CPU tensor from ``_chunked_subm_conv3d_slices_to_cpu``."""
    slices = _chunked_subm_conv3d_slices_to_cpu(conv_mod, x, torch_mod, max_slice_bytes=max_slice_bytes)
    if slices is None:
        return None
    try:
        cpu_parts = []
        for _c0, _c1, sl in slices:
            if isinstance(sl, _DiskMmapTensor):
                cpu_parts.append(sl.tensor.clone())
            elif hasattr(sl, "cpu"):
                cpu_parts.append(sl.cpu())
            else:
                cpu_parts.append(sl)
        return cpu_parts[0] if len(cpu_parts) == 1 else torch_mod.cat(cpu_parts, dim=1)
    finally:
        _release_staged_slices(slices)


def _copy_cpu_slices_into_gpu_feats(
    feats: Any,
    cpu_slices: list[tuple[int, int, Any]],
    step: int,
) -> None:
    """Stream staged channel slices back into ``feats[:, c0:c1]`` on GPU and immediately free each slice."""
    n = int(feats.shape[0])
    dev = feats.device
    try:
        # Write GPU-resident slices first so their GPU memory is released before streaming off-GPU slices
        order = sorted(
            range(len(cpu_slices)),
            key=lambda idx: 0 if bool(getattr(cpu_slices[idx][2], "is_cuda", False)) else 1,
        )
        for idx in order:
            c0, c1, sl_obj = cpu_slices[idx]
            if bool(getattr(sl_obj, "is_cuda", False)) and sl_obj.device == dev:
                feats[:, c0:c1].copy_(sl_obj)
            else:
                for s in range(0, n, step):
                    e = min(s + step, n)
                    chunk = sl_obj[s:e]
                    if getattr(chunk, "device", None) != dev:
                        chunk = chunk.to(dev, non_blocking=True)
                    feats[s:e, c0:c1].copy_(chunk)
                    del chunk
            _release_staged_obj(sl_obj)
            cpu_slices[idx] = (c0, c1, None)
    finally:
        _release_staged_slices(cpu_slices)


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
    if _is_already_patched(convnext_cls, "_trellis2_vram_patched"):
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
        x_feats = x.feats
        n = int(x_feats.shape[0])
        is_cuda = bool(getattr(x_feats, "is_cuda", False))
        step = _effective_voxel_chunk(torch_mod, chunk_size)

        cpu_slices = None
        h = None
        w_conv = getattr(self.conv, "weight", None)
        co = int(w_conv.shape[0]) if w_conv is not None and getattr(w_conv, "ndim", 0) == 5 else int(x_feats.shape[-1])
        est_out_bytes = n * co * int(x_feats.element_size())
        want_chunked_conv = False
        if is_cuda and torch_mod is not None and est_out_bytes >= 1024**3:
            free_gb = _cuda_free_gb(torch_mod)
            needed_b = _estimate_subm_conv_needed_bytes(self.conv, x, est_out_bytes)
            if free_gb is not None and (free_gb * (1024**3)) < needed_b:
                want_chunked_conv = True

        if want_chunked_conv:
            log.info(
                "Using output-channel chunked SubMConv3d in ConvNeXt block (N=%d, Co=%d, est_out=%.2f GiB)",
                n,
                co,
                est_out_bytes / (1024**3),
            )
            cpu_slices = _chunked_subm_conv3d_slices_to_cpu(self.conv, x, torch_mod)

        if cpu_slices is None:
            conv_oom = False
            try:
                h = self.conv(x)
            except RuntimeError as exc:
                if not (is_cuda and torch_mod is not None and _is_cuda_oom_msg(exc)):
                    raise
                _clear_oom_traceback_frames(exc)
                conv_oom = True
            if conv_oom:
                gc.collect()
                if hasattr(torch_mod, "cuda") and torch_mod.cuda.is_available():
                    torch_mod.cuda.empty_cache()
                log.warning(
                    "ConvNeXt SubMConv3d OOM (N=%d, Co=%d, est_out=%.2f GiB); retrying with channel-chunked staging",
                    n,
                    co,
                    est_out_bytes / (1024**3),
                )
                cpu_slices = _chunked_subm_conv3d_slices_to_cpu(self.conv, x, torch_mod)
                if cpu_slices is None:
                    raise RuntimeError("ConvNeXt SubMConv3d OOM and channel-chunked fallback unavailable")

        if cpu_slices is not None:
            dev = x_feats.device
            try:
                for s in range(0, n, step):
                    e = min(s + step, n)
                    if len(cpu_slices) == 1:
                        sl0 = cpu_slices[0][2]
                        c_part = sl0[s:e]
                        sub = c_part if getattr(c_part, "device", None) == dev else c_part.to(dev, non_blocking=True)
                    else:
                        parts = []
                        for _, _, sl_obj in cpu_slices:
                            c_part = sl_obj[s:e]
                            parts.append(
                                c_part if getattr(c_part, "device", None) == dev else c_part.to(dev, non_blocking=True)
                            )
                        sub = torch_mod.cat(parts, dim=1)
                        del parts
                    sub = self.norm(sub)
                    sub = self.mlp[0](sub)
                    f_mod.silu(sub, inplace=True)
                    sub = self.mlp[2](sub)
                    x_feats[s:e].add_(sub)
                    del sub
            finally:
                _release_staged_slices(cpu_slices)
            del cpu_slices
            return x

        feats = h.feats
        use_fused = (
            not compile_state["failed"]
            and compiled_pointwise is not _convnext_pointwise_eager
            and is_cuda
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


def patch_sparse_res_block(
    res_cls: Any,
    torch_mod: Any,
    f_mod: Any,
    chunk_size: int = DEFAULT_VOXEL_CHUNK_SIZE,
) -> None:
    """Chunk pointwise Norm + SiLU + residual add and add channel-chunked SubMConv3d fallback in SparseResBlock3d."""
    if res_cls is None or _is_already_patched(res_cls, "_trellis2_vram_patched"):
        return
    orig_forward = res_cls._forward

    def _run_conv_low_vram(conv_mod: Any, sp_in: Any, step: int) -> Any:
        n = int(sp_in.feats.shape[0])
        is_cuda = bool(getattr(sp_in.feats, "is_cuda", False))
        w_conv = getattr(conv_mod, "weight", None)
        co = int(w_conv.shape[0]) if w_conv is not None and getattr(w_conv, "ndim", 0) == 5 else int(sp_in.feats.shape[-1])
        est_bytes = n * co * int(sp_in.feats.element_size())
        cpu_slices = None
        if is_cuda and est_bytes >= 1024**3:
            free_gb = _cuda_free_gb(torch_mod)
            needed_b = _estimate_subm_conv_needed_bytes(conv_mod, sp_in, est_bytes)
            if free_gb is not None and (free_gb * (1024**3)) < needed_b:
                cpu_slices = _chunked_subm_conv3d_slices_to_cpu(conv_mod, sp_in, torch_mod)
        if cpu_slices is None:
            res_oom = False
            try:
                return conv_mod(sp_in)
            except RuntimeError as exc:
                if not (is_cuda and _is_cuda_oom_msg(exc)):
                    raise
                _clear_oom_traceback_frames(exc)
                res_oom = True
            if res_oom:
                gc.collect()
                if hasattr(torch_mod, "cuda") and torch_mod.cuda.is_available():
                    torch_mod.cuda.empty_cache()
                cpu_slices = _chunked_subm_conv3d_slices_to_cpu(conv_mod, sp_in, torch_mod)
                if cpu_slices is None:
                    raise RuntimeError("SparseResBlock3d SubMConv3d OOM and channel-chunked fallback unavailable")
        if co == int(sp_in.feats.shape[-1]):
            _copy_cpu_slices_into_gpu_feats(sp_in.feats, cpu_slices, step)
            del cpu_slices
            return sp_in
        dev = sp_in.feats.device
        dtype = sp_in.feats.dtype
        out_sp = sp_in.replace(torch_mod.empty((0, co), dtype=dtype, device=dev))
        del sp_in
        if hasattr(torch_mod, "cuda") and torch_mod.cuda.is_available():
            torch_mod.cuda.empty_cache()
        out_feats = torch_mod.empty((n, co), dtype=dtype, device=dev)
        _copy_cpu_slices_into_gpu_feats(out_feats, cpu_slices, step)
        del cpu_slices
        return out_sp.replace(out_feats)

    def _chunked_res_forward(self: Any, x: Any) -> Any:
        if (
            getattr(self, "training", False)
            or getattr(self, "downsample", False)
            or getattr(self, "upsample", False)
            or getattr(self, "resample_mode", "nearest") != "nearest"
        ):
            return orig_forward(self, x)

        n = int(x.feats.shape[0])
        is_cuda = bool(getattr(x.feats, "is_cuda", False))
        step = _effective_voxel_chunk(torch_mod, chunk_size)
        offload_skip = bool(is_cuda and n >= 1_000_000)
        x_skip_saved = x.feats.cpu() if offload_skip else x.feats

        normed = torch_mod.empty_like(x.feats)
        for s in range(0, n, step):
            e = min(s + step, n)
            tmp = self.norm1(x.feats[s:e])
            f_mod.silu(tmp, inplace=True)
            normed[s:e].copy_(tmp)
            del tmp
        h = x.replace(normed)
        del x, normed

        h = _run_conv_low_vram(self.conv1, h, step)
        for s in range(0, n, step):
            e = min(s + step, n)
            tmp = self.norm2(h.feats[s:e])
            f_mod.silu(tmp, inplace=True)
            h.feats[s:e].copy_(tmp)
            del tmp
        h = _run_conv_low_vram(self.conv2, h, step)

        dev = h.feats.device
        skip_mod = self.skip_connection
        is_identity = type(skip_mod).__name__ == "Identity"
        skip_linear = getattr(skip_mod, "weight", None) is not None and hasattr(skip_mod, "bias")
        if is_identity or skip_linear:
            w_sk = getattr(skip_mod, "weight", None)
            b_sk = getattr(skip_mod, "bias", None)
            for s in range(0, n, step):
                e = min(s + step, n)
                sk = x_skip_saved[s:e].to(dev, non_blocking=True) if offload_skip else x_skip_saved[s:e]
                if skip_linear:
                    sk = f_mod.linear(sk, w_sk, b_sk)
                h.feats[s:e].add_(sk)
                del sk
            del x_skip_saved
        else:
            sk_gpu = x_skip_saved.to(dev, non_blocking=True) if offload_skip else x_skip_saved
            del x_skip_saved
            x_tmp = h.replace(sk_gpu)
            del sk_gpu
            sk_out = skip_mod(x_tmp)
            del x_tmp
            for s in range(0, n, step):
                e = min(s + step, n)
                h.feats[s:e].add_(sk_out.feats[s:e])
            del sk_out
        return h

    res_cls._forward = _chunked_res_forward
    res_cls._trellis2_vram_patched = True


def _try_octant_grouped_c2s_conv1(
    conv1_mod: Any,
    h: Any,
    row_idx: Any,
    subidx: Any,
    out_ch: int,
    torch_mod: Any,
) -> Any | None:
    """Run SparseResBlockC2S3d.conv1 (C -> 8*out_ch) one octant at a time to cut 7+ GiB spike by 8x.

    Mathematically bit-exact: each octant k in 0..7 uses an independent contiguous slice
    weight[k*out_ch:(k+1)*out_ch] and bias[k*out_ch:(k+1)*out_ch] while reusing the exact
    same FlexGEMM SubMConv3d neighbor_cache across all 8 slices.
    """
    n_in = int(h.feats.shape[0])
    min_bytes = int(os.environ.get("TRELLIS2_C2S_GROUPED_CONV1_BYTES", str(1024**3)))
    est_bytes = n_in * out_ch * 8 * int(h.feats.element_size())
    if est_bytes < min_bytes:
        return None

    w_all = getattr(conv1_mod, "weight", None)
    b_all = getattr(conv1_mod, "bias", None)
    if w_all is None or getattr(w_all, "ndim", 0) != 5 or int(w_all.shape[0]) != out_ch * 8:
        return None
    if b_all is not None and int(b_all.shape[0]) != out_ch * 8:
        return None

    grouped_fn = getattr(conv1_mod, "_grouped_conv_fn", None)
    dilation = getattr(conv1_mod, "dilation", (1, 1, 1))
    if grouped_fn is None:
        if not bool(getattr(h.feats, "is_cuda", False)):
            return None
        try:
            import flex_gemm  # type: ignore[import-not-found]
            from flex_gemm.ops.spconv import sparse_submanifold_conv3d  # type: ignore[import-not-found]
            from trellis2.modules.sparse.conv import config as conv_cfg  # type: ignore[import-not-found]
        except Exception:
            return None
        if getattr(conv_cfg, "CONV", "flex_gemm") != "flex_gemm":
            return None
        flex_gemm.ops.spconv.set_algorithm(conv_cfg.FLEX_GEMM_ALGO)
        flex_gemm.ops.spconv.set_hashmap_ratio(conv_cfg.FLEX_GEMM_HASHMAP_RATIO)
        grouped_fn = sparse_submanifold_conv3d

    _co, kd, kh, kw, _ci = w_all.shape
    neighbor_cache_key = f"SubMConv3d_neighbor_cache_{kw}x{kh}x{kd}_dilation{dilation}"
    neighbor_cache = h.get_spatial_cache(neighbor_cache_key) if hasattr(h, "get_spatial_cache") else None
    spatial_shape = torch_mod.Size([*h.shape, *h.spatial_shape])
    n_out = int(row_idx.shape[0])
    h_up_feats = None

    for k in range(8):
        mask_k = subidx == k
        if not bool(mask_k.any()):
            del mask_k
            continue
        w_k = w_all[k * out_ch : (k + 1) * out_ch]
        b_k = None if b_all is None else b_all[k * out_ch : (k + 1) * out_ch]
        out_k, nc_ = grouped_fn(
            h.feats,
            h.coords,
            spatial_shape,
            w_k,
            b_k,
            neighbor_cache,
            dilation,
        )
        if neighbor_cache is None and nc_ is not None:
            neighbor_cache = nc_
            if hasattr(h, "register_spatial_cache"):
                h.register_spatial_cache(neighbor_cache_key, neighbor_cache)
        if h_up_feats is None:
            h_up_feats = torch_mod.empty((n_out, out_ch), dtype=out_k.dtype, device=out_k.device)
        idx_k = row_idx[mask_k]
        if idx_k.dtype != torch_mod.long:
            idx_k = idx_k.long()
        h_up_feats[mask_k] = out_k[idx_k]
        del out_k, mask_k, idx_k

    if h_up_feats is None:
        h_up_feats = torch_mod.empty((0, out_ch), dtype=h.feats.dtype, device=h.feats.device)
    return h_up_feats


def patch_flex_gemm_neighbor_cache(
    subm_mod: Any,
    torch_mod: Any = None,
) -> None:
    """Release FlexGEMM build-time hashmap tensors before CUB radix-sort post-processing.

    Upstream ``SubMConv3dFunction._compute_neighbor_cache`` keeps ``hashmap_keys`` and
    ``hashmap_vals`` (0.5–1.2 GiB at 1536^3) alive in local scope while calling
    ``neighbor_map_post_process_for_masked_implicit_gemm_1(_no_bwd)``, causing avoidable
    ~250 MiB–1 GiB OOMs at peak decoder resolution. Deleting them immediately after
    ``hashmap_build_submanifold_conv_neighbour_map`` is 100% bit-exact with zero speed cost.
    Also retries once with ``empty_cache()`` (and a tighter 1.5x hashmap load factor if
    the initial hashmap/neighbor build OOMs) to absorb allocator fragmentation.
    """
    subm_fn_cls = getattr(subm_mod, "SubMConv3dFunction", None)
    cache_cls = getattr(subm_mod, "SubMConv3dNeighborCache", None)
    if subm_fn_cls is None or cache_cls is None:
        return
    if _is_already_patched(subm_fn_cls, "_trellis2_vram_patched"):
        return

    alg_cls = getattr(subm_mod, "Algorithm", None)
    spconv_pkg = getattr(subm_mod, "spconv", None)
    utils_mod = getattr(subm_mod, "utils", None)
    kernels_mod = getattr(subm_mod, "kernels", None)
    if any(x is None for x in (alg_cls, spconv_pkg, utils_mod, kernels_mod)):
        return

    explicit_algs = {
        getattr(alg_cls, name, None)
        for name in ("EXPLICIT_GEMM", "IMPLICIT_GEMM", "IMPLICIT_GEMM_SPLITK")
    }
    masked_algs = {
        getattr(alg_cls, name, None)
        for name in ("MASKED_IMPLICIT_GEMM", "MASKED_IMPLICIT_GEMM_SPLITK")
    }

    def _empty_cuda_cache() -> None:
        if torch_mod is not None and hasattr(torch_mod, "cuda") and torch_mod.cuda.is_available():
            torch_mod.cuda.empty_cache()

    def _is_oom(exc: BaseException) -> bool:
        msg = str(exc).lower()
        return "out of memory" in msg or "cudaerrormemoryallocation" in msg

    @staticmethod
    def _low_vram_compute_neighbor_cache(
        coords: Any,
        shape: Any,
        kernel_size: tuple[int, int, int],
        dilation: tuple[int, int, int],
        needs_grad: bool,
    ) -> Any:
        assert coords.is_contiguous(), "Coords should be contiguous"
        if torch_mod is not None:
            assert coords.dtype in [torch_mod.int32], "Unsupported coords dtype. Expect int32"
        _, _, w, h, d = shape
        alg = spconv_pkg.ALGORITHM
        if alg not in explicit_algs and alg not in masked_algs:
            raise ValueError(f"Unsupported algorithm {alg}")
        if not coords.is_cuda:
            raise NotImplementedError("CPU version of hashmap is not implemented")

        ratio = float(getattr(spconv_pkg, "HASHMAP_RATIO", 2.0))
        if int(coords.shape[0]) >= 35_000_000 and ratio > 1.5:
            ratio = 1.5
        ratios_to_try = [ratio]
        if ratio > 1.5:
            ratios_to_try.append(1.5)

        neighbor_map = None
        last_exc: Exception | None = None
        for attempt_idx, cur_ratio in enumerate(ratios_to_try):
            hashmap_keys = None
            hashmap_vals = None
            try:
                hashmap_keys, hashmap_vals = utils_mod.init_hashmap(
                    shape, int(cur_ratio * coords.shape[0]), coords.device
                )
                neighbor_map = kernels_mod.cuda.hashmap_build_submanifold_conv_neighbour_map(
                    hashmap_keys,
                    hashmap_vals,
                    coords,
                    w,
                    h,
                    d,
                    kernel_size[0],
                    kernel_size[1],
                    kernel_size[2],
                    dilation[0],
                    dilation[1],
                    dilation[2],
                )
                del hashmap_keys, hashmap_vals
                break
            except RuntimeError as exc:
                del hashmap_keys, hashmap_vals
                if not _is_oom(exc):
                    raise
                _clear_oom_traceback_frames(exc)
                if attempt_idx + 1 >= len(ratios_to_try):
                    raise
                _empty_cuda_cache()
                log.warning(
                    "FlexGEMM neighbor_map build OOM at HASHMAP_RATIO=%.2f (N=%d); retrying at %.2f",
                    cur_ratio,
                    int(coords.shape[0]),
                    ratios_to_try[attempt_idx + 1],
                )

        if neighbor_map is None and last_exc is not None:
            raise last_exc

        if alg in explicit_algs:
            return cache_cls(neighbor_map=neighbor_map)

        vol = kernel_size[0] * kernel_size[1] * kernel_size[2]
        assert vol <= 32, "Currently, the max kernel volume is 32 because kernel mask is encoded as uint32"

        if needs_grad:
            try:
                gray_code, sorted_idx, valid_signal_i, valid_signal_o, valid_signal_seg = (
                    kernels_mod.cuda.neighbor_map_post_process_for_masked_implicit_gemm_1(
                        neighbor_map
                    )
                )
            except RuntimeError as exc:
                if not _is_oom(exc):
                    raise
                _empty_cuda_cache()
                gray_code, sorted_idx, valid_signal_i, valid_signal_o, valid_signal_seg = (
                    kernels_mod.cuda.neighbor_map_post_process_for_masked_implicit_gemm_1(
                        neighbor_map
                    )
                )
            return cache_cls(
                neighbor_map=neighbor_map,
                gray_code=gray_code,
                sorted_idx=sorted_idx,
                valid_signal_seg=valid_signal_seg,
                valid_signal_i=valid_signal_i,
                valid_signal_o=valid_signal_o,
            )

        post_oom = False
        try:
            gray_code, sorted_idx = (
                kernels_mod.cuda.neighbor_map_post_process_for_masked_implicit_gemm_1_no_bwd(
                    neighbor_map
                )
            )
        except RuntimeError as exc:
            if not _is_oom(exc):
                raise
            _clear_oom_traceback_frames(exc)
            post_oom = True
        if post_oom:
            _empty_cuda_cache()
            try:
                gray_code, sorted_idx = (
                    kernels_mod.cuda.neighbor_map_post_process_for_masked_implicit_gemm_1_no_bwd(
                        neighbor_map
                    )
                )
            except RuntimeError as exc2:
                implicit_alg = getattr(alg_cls, "IMPLICIT_GEMM", None)
                if not _is_oom(exc2) or implicit_alg is None:
                    raise
                _clear_oom_traceback_frames(exc2)
                _empty_cuda_cache()
                log.warning(
                    "FlexGEMM masked post-process OOM at N=%d; switching current conv to IMPLICIT_GEMM (saves ~1.8 GiB sort/mask buffers)",
                    int(coords.shape[0]),
                )
                spconv_pkg.ALGORITHM = implicit_alg
                return cache_cls(neighbor_map=neighbor_map)
        return cache_cls(
            neighbor_map=neighbor_map,
            gray_code=gray_code,
            sorted_idx=sorted_idx,
        )

    subm_fn_cls._compute_neighbor_cache = _low_vram_compute_neighbor_cache
    subm_fn_cls._trellis2_vram_patched = True


def patch_sparse_c2s_block(
    c2s_cls: Any,
    sparse_tensor_cls: Any,
    torch_mod: Any,
    f_mod: Any,
    chunk_size: int = DEFAULT_VOXEL_CHUNK_SIZE,
) -> None:
    """Optimize SparseResBlockC2S3d via vectorized single-nonzero C2S gather, cross-decoder gather cache, and fused skip add."""
    if _is_already_patched(c2s_cls, "_trellis2_vram_patched"):
        return
    orig_forward = c2s_cls._forward

    def _chunked_c2s_forward(self: Any, x: Any, subdiv: Any = None) -> Any:
        if getattr(self, "training", False):
            return orig_forward(self, x, subdiv)

        step = _effective_voxel_chunk(torch_mod, chunk_size)
        n_in = x.feats.shape[0]
        is_cuda_x = bool(getattr(x.feats, "is_cuda", False))
        free_gb = _cuda_free_gb(torch_mod) if is_cuda_x else None
        keep_sub_on_gpu = bool(
            os.environ.get("TRELLIS2_FAST_VRAM", "1") != "0"
            and is_cuda_x
            and n_in < 1_000_000
            and (free_gb is None or free_gb >= VRAM_SAFETY_FLOOR_GB + 6.0)
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
            and (
                n_in >= 1_000_000
                or (n_in >= 500_000 and free_gb is not None and free_gb < VRAM_SAFETY_FLOOR_GB + 6.0)
            )
        )
        x_feats_saved = _stage_tensor_off_gpu(x.feats, torch_mod) if offload_skip else x.feats
        x_shell = x

        h = x.replace(normed)
        if offload_skip:
            # Drop GPU buffer reference inside caller's SparseTensor shell (`h, sub = block(h)` keeps
            # caller `h` bound until `_chunked_c2s_forward` returns — ~3.1 GiB at 768^3 -> 1536^3!).
            try:
                x_shell.feats = None
            except Exception:
                pass
        del x, normed

        can_fast_c2s = (
            getattr(self.updown, "factor", None) == 2
            and x_coords.ndim == 2
            and x_coords.shape[1] == 4
            and (cached_gather is not None or subdiv_binarized is not None)
        )

        flat_idx = None
        flat_idx_saved = None
        offload_flat_idx = False
        use_i32_idx = bool(n_in * 8 < 2_147_483_647)
        if can_fast_c2s:
            if cached_gather is not None:
                new_coords, flat_idx = cached_gather
                cached_gather = None
                row_idx = torch_mod.div(flat_idx, 8, rounding_mode="floor")
                subidx = (flat_idx - row_idx * 8).to(torch_mod.uint8)
                if use_i32_idx and row_idx.dtype != torch_mod.int32:
                    row_idx = row_idx.int()
            else:
                nz = subdiv_binarized.feats.nonzero()
                row_idx_l = nz[:, 0]
                subidx_l = nz[:, 1]
                new_coords = x_coords[row_idx_l].clone()
                new_coords[:, 1:] *= 2
                shifts = torch_mod.tensor([0, 1, 2], device=x_coords.device, dtype=subidx_l.dtype)
                new_coords[:, 1:] += ((subidx_l[:, None] >> shifts) & 1).to(new_coords.dtype)
                del nz, shifts
                if use_i32_idx:
                    row_idx = row_idx_l.int()
                    subidx = subidx_l.to(torch_mod.uint8)
                    del row_idx_l, subidx_l
                    flat_idx = row_idx * 8 + subidx.int()
                else:
                    row_idx = row_idx_l
                    subidx = subidx_l
                    flat_idx = row_idx * 8 + subidx
                if (
                    self.pred_subdiv
                    and sub_ret is not None
                    and (not is_cuda_x or keep_sub_on_gpu)
                    and os.environ.get("TRELLIS2_CROSS_DECODER_CACHE", "1") != "0"
                ):
                    sub_ret._c2s_gather_cache = (new_coords, flat_idx)

            x_coords = None
            subdiv_binarized = None
            h_shape = getattr(h, "_shape", None)
            out_shape = (
                None if h_shape is None else torch_mod.Size([h_shape[0], self.out_channels])
            )
            h_scale = getattr(h, "_scale", None)

            # Spill flat_idx to CPU BEFORE conv1 on large upsamples so conv1 gets the extra ~220-430 MiB too
            if (
                is_cuda_x
                and flat_idx is not None
                and int(flat_idx.shape[0]) >= 1_500_000
                and hasattr(flat_idx, "cpu")
            ):
                if sub_ret is not None and getattr(sub_ret, "_c2s_gather_cache", None) is not None:
                    sub_ret._c2s_gather_cache = None
                flat_idx_saved = flat_idx.cpu()
                del flat_idx
                flat_idx = None
                offload_flat_idx = True

            h_up_feats = _try_octant_grouped_c2s_conv1(
                self.conv1, h, row_idx, subidx, int(self.out_channels), torch_mod
            )
            del row_idx, subidx
            if h_up_feats is not None:
                if hasattr(h, "clear_spatial_cache"):
                    h.clear_spatial_cache()
                del h
            else:
                h = self.conv1(h)
                if hasattr(h, "clear_spatial_cache"):
                    h.clear_spatial_cache()
                h_feats_2d = h.feats.reshape(n_in * 8, -1)
                del h
                gather_idx = (
                    flat_idx_saved.to(h_feats_2d.device, non_blocking=True)
                    if offload_flat_idx and flat_idx_saved is not None
                    else flat_idx
                )
                if gather_idx.dtype != torch_mod.long:
                    gather_idx = gather_idx.long()
                h_up_feats = h_feats_2d[gather_idx]
                del h_feats_2d, gather_idx

            h = sparse_tensor_cls(h_up_feats, new_coords, out_shape)
            if h_scale is not None:
                h._scale = tuple(sc / 2 for sc in h_scale)
            del h_up_feats, new_coords
            try:
                if hasattr(x_shell, "clear_spatial_cache"):
                    x_shell.clear_spatial_cache()
                x_shell.coords = None
            except Exception:
                pass
        else:
            h = self.conv1(h)
            if hasattr(h, "clear_spatial_cache"):
                h.clear_spatial_cache()
            h = self.updown(h, subdiv_binarized)

        n_out = h.feats.shape[0]
        for s in range(0, n_out, step):
            e = min(s + step, n_out)
            tmp = self.norm2(h.feats[s:e])
            f_mod.silu(tmp, inplace=True)
            h.feats[s:e].copy_(tmp)
            del tmp

        if flat_idx_saved is None:
            flat_idx_saved = flat_idx
        if is_cuda_x and n_out >= 1_500_000:
            if not offload_skip:
                x_feats_saved = _stage_tensor_off_gpu(x_feats_saved, torch_mod)
                offload_skip = True
                try:
                    x_shell.feats = None
                except Exception:
                    pass
            if not offload_flat_idx and flat_idx_saved is not None and hasattr(flat_idx_saved, "cpu"):
                if sub_ret is not None and getattr(sub_ret, "_c2s_gather_cache", None) is not None:
                    sub_ret._c2s_gather_cache = None
                flat_idx_saved = flat_idx_saved.cpu()
                flat_idx = None
                offload_flat_idx = True
            x_shell = None
            torch_mod.cuda.empty_cache()
        elif is_cuda_x and n_out >= 1_000_000:
            x_shell = None
            free_pre_conv2 = _cuda_free_gb(torch_mod)
            if free_pre_conv2 is not None and free_pre_conv2 < VRAM_SAFETY_FLOOR_GB + 6.0:
                torch_mod.cuda.empty_cache()
        else:
            x_shell = None

        conv2_slices_cpu = None
        est_conv2_bytes = n_out * int(self.out_channels) * int(h.feats.element_size())
        if is_cuda_x and est_conv2_bytes >= 1024**3:
            free_pre_conv2 = _cuda_free_gb(torch_mod)
            needed_conv2_b = _estimate_subm_conv_needed_bytes(self.conv2, h, est_conv2_bytes)
            if free_pre_conv2 is not None and (free_pre_conv2 * (1024**3)) < needed_conv2_b:
                log.info(
                    "Using output-channel chunked SubMConv3d in C2S conv2 (N=%d, Co=%d, est_out=%.2f GiB, free=%.2f GiB, needed=%.2f GiB)",
                    n_out,
                    int(self.out_channels),
                    est_conv2_bytes / (1024**3),
                    free_pre_conv2,
                    needed_conv2_b / (1024**3),
                )
                conv2_slices_cpu = _chunked_subm_conv3d_slices_to_cpu(self.conv2, h, torch_mod)

        if conv2_slices_cpu is None:
            conv2_oom = False
            try:
                h = self.conv2(h)
            except RuntimeError as exc:
                if not (is_cuda_x and _is_cuda_oom_msg(exc)):
                    raise
                _clear_oom_traceback_frames(exc)
                conv2_oom = True
            if conv2_oom:
                gc.collect()
                torch_mod.cuda.empty_cache()
                log.warning(
                    "C2S conv2 SubMConv3d OOM (N=%d, Co=%d, est_out=%.2f GiB); retrying with channel-chunked staging",
                    n_out,
                    int(self.out_channels),
                    est_conv2_bytes / (1024**3),
                )
                conv2_slices_cpu = _chunked_subm_conv3d_slices_to_cpu(self.conv2, h, torch_mod)
                if conv2_slices_cpu is None:
                    raise RuntimeError("C2S conv2 SubMConv3d OOM and channel-chunked fallback unavailable")

        if conv2_slices_cpu is not None:
            _copy_cpu_slices_into_gpu_feats(h.feats, conv2_slices_cpu, step)
            del conv2_slices_cpu

        # At 1536^3 (n_out >= 10M), stage 4 has 0 blocks so h's ~6.5 GiB neighbor_cache is dead immediately
        # after conv2; releasing it now gives ample VRAM before streaming x_skip_feats back onto GPU.
        if is_cuda_x and n_out >= 10_000_000 and hasattr(h, "clear_spatial_cache"):
            h.clear_spatial_cache()
            torch_mod.cuda.empty_cache()

        try:
            x_skip_feats = x_feats_saved.to(h.device, non_blocking=True) if offload_skip else x_feats_saved
        finally:
            if offload_skip:
                _release_staged_obj(x_feats_saved)
        del x_feats_saved
        rep = self.out_channels // (self.channels // 8)

        if flat_idx_saved is not None:
            # Fused chunked skip gather + repeat_interleave + in-place residual add (zero 20M-voxel x_up allocation)
            x_skip_2d = x_skip_feats.reshape(n_in * 8, self.channels // 8)
            for s in range(0, n_out, step):
                e = min(s + step, n_out)
                idx_chunk = (
                    flat_idx_saved[s:e].to(h.device, non_blocking=True)
                    if offload_flat_idx
                    else flat_idx_saved[s:e]
                )
                if idx_chunk.dtype != torch_mod.long:
                    idx_chunk = idx_chunk.long()
                skip_sub = x_skip_2d[idx_chunk]
                del idx_chunk
                if rep == 1:
                    h.feats[s:e].add_(skip_sub)
                else:
                    h.feats[s:e].add_(skip_sub.repeat_interleave(rep, dim=1))
                del skip_sub
            del x_skip_2d, x_skip_feats, flat_idx_saved, flat_idx
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
    if _is_already_patched(decoder_cls, "_trellis2_vram_patched"):
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
        cross_cache_max_voxels = 1_000_000
        num_stages = len(self.blocks)
        try:
            for i, res in enumerate(self.blocks):
                num_res = len(res)
                if cross_caches is not None and i in cross_caches and not stream and hasattr(h, "_spatial_cache"):
                    expected_n, stage_dict = cross_caches.pop(i)
                    if h.coords.shape[0] == expected_n:
                        scale_k = str(getattr(h, "_scale", ""))
                        h._spatial_cache[scale_k] = stage_dict

                if is_cuda_x:
                    n_stage_vox = int(h.coords.shape[0])
                    need_stage_evict = n_stage_vox >= 1_000_000 or i == num_stages - 1
                    if not need_stage_evict and saved_stage_caches and i >= 2:
                        free_stage = _cuda_free_gb(torch_mod)
                        if free_stage is not None and free_stage < VRAM_SAFETY_FLOOR_GB + 5.0:
                            need_stage_evict = True
                    if need_stage_evict:
                        evicted = False
                        if saved_stage_caches:
                            saved_stage_caches.clear()
                            evicted = True
                        for idx_s, s_tensor in enumerate(subs):
                            if getattr(s_tensor, "_c2s_gather_cache", None) is not None:
                                s_tensor._c2s_gather_cache = None
                                evicted = True
                            if hasattr(s_tensor, "cpu") and getattr(
                                getattr(s_tensor, "feats", None), "is_cuda", False
                            ):
                                subs[idx_s] = s_tensor.cpu()
                                evicted = True
                        if evicted:
                            torch_mod.cuda.empty_cache()

                if stream and num_res > 0:
                    res.to(dev)
                try:
                    for j, block in enumerate(res):
                        if i < num_stages - 1 and j == num_res - 1:
                            if (
                                return_subs
                                and self.pred_subdiv
                                and share_cross_cache
                                and hasattr(h, "_spatial_cache")
                                and h.coords.shape[0] <= cross_cache_max_voxels
                            ):
                                free_gb = _cuda_free_gb(torch_mod) if is_cuda_x else None
                                if free_gb is None or free_gb >= VRAM_SAFETY_FLOOR_GB + 7.0:
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
                    and h.coords.shape[0] <= cross_cache_max_voxels
                ):
                    free_gb = _cuda_free_gb(torch_mod) if is_cuda_x else None
                    if free_gb is None or free_gb >= VRAM_SAFETY_FLOOR_GB + 7.0:
                        cur_sc = h._spatial_cache.get(str(getattr(h, "_scale", "")))
                        if cur_sc:
                            filtered = {k: v for k, v in cur_sc.items() if k != "subdivision"}
                            if filtered:
                                saved_stage_caches[i] = (h.coords.shape[0], filtered)

                if hasattr(h, "clear_spatial_cache"):
                    h.clear_spatial_cache()
                if is_cuda_x and i >= 1:
                    free_gb = _cuda_free_gb(torch_mod)
                    flush_threshold = VRAM_SAFETY_FLOOR_GB if stream else VRAM_SAFETY_FLOOR_GB + 2.5
                    if free_gb is not None and free_gb < flush_threshold:
                        if saved_stage_caches:
                            saved_stage_caches.clear()
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


def _chunked_fdg_to_mesh(
    coords: Any,
    dual_vertices: Any,
    intersected_flag: Any,
    split_weight: Any,
    aabb: list[list[float]],
    grid_size: int,
    torch_mod: Any,
    o_voxel_c: Any | None = None,
    edge_chunk: int = 2_000_000,
) -> tuple[Any, Any] | None:
    """Bit-exact chunked Dual-Grid mesh extraction avoiding multi-GiB (N,3,4,3) dense edge expansion."""
    n = int(dual_vertices.shape[0])
    min_vox = int(os.environ.get("TRELLIS2_CHUNKED_FDG_MIN_VOXELS", "2000000"))
    if n < min_vox:
        return None
    if split_weight is None:
        log.warning("Chunked Dual-Grid extraction skipped (split_weight=None, N=%d)", n)
        return None
    if o_voxel_c is None:
        if not bool(getattr(coords, "is_cuda", False)):
            log.warning("Chunked Dual-Grid extraction skipped (non-CUDA coords, N=%d)", n)
            return None
        for mod_name in ("o_voxel._C", "o_voxel"):
            try:
                mod_obj = sys.modules.get(mod_name) or importlib.import_module(mod_name)
                cand_c = mod_obj if mod_name.endswith("._C") else getattr(mod_obj, "_C", None)
                if cand_c is not None:
                    o_voxel_c = cand_c
                    break
            except Exception:
                continue
        if o_voxel_c is None:
            fdg_mod = sys.modules.get("o_voxel.convert.flexible_dual_grid")
            o_voxel_c = getattr(fdg_mod, "_C", None) if fdg_mod is not None else None
    if not (
        callable(getattr(o_voxel_c, "hashmap_insert_3d_idx_as_val_cuda", None))
        and callable(getattr(o_voxel_c, "hashmap_lookup_3d_cuda", None))
    ):
        log.warning("Chunked Dual-Grid extraction skipped (o_voxel._C hash ops unavailable, N=%d)", n)
        return None

    log.info("Running chunked Dual-Grid mesh extraction (N=%d, grid=%d)", n, int(grid_size))
    dev = coords.device
    gs = [int(grid_size), int(grid_size), int(grid_size)]
    aabb_t = torch_mod.tensor(aabb, dtype=torch_mod.float32, device=dev)
    grid_size_t = torch_mod.tensor(gs, dtype=torch_mod.int32, device=dev)
    voxel_size = (aabb_t[1] - aabb_t[0]) / grid_size_t

    vol = gs[0] * gs[1] * gs[2]
    key_dtype = torch_mod.uint32 if vol < 2**32 else torch_mod.uint64
    hashmap_keys = torch_mod.full(
        (2 * n,), torch_mod.iinfo(key_dtype).max, dtype=key_dtype, device=dev
    )
    hashmap_vals = torch_mod.empty((2 * n,), dtype=torch_mod.uint32, device=dev)
    insert_keys = torch_mod.cat(
        [torch_mod.zeros((n, 1), dtype=coords.dtype, device=dev), coords], dim=-1
    )
    o_voxel_c.hashmap_insert_3d_idx_as_val_cuda(hashmap_keys, hashmap_vals, insert_keys, *gs)
    del insert_keys

    edge_offsets = torch_mod.tensor(
        [
            [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
            [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
            [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
        ],
        dtype=torch_mod.int32,
        device=dev,
    )

    vox_chunk = max(4_000_000, edge_chunk * 2)
    quad_chunks: list[Any] = []
    for vs in range(0, n, vox_chunk):
        ve = min(vs + vox_chunk, n)
        nz = intersected_flag[vs:ve].nonzero()
        m_sub = int(nz.shape[0])
        if m_sub == 0:
            del nz
            continue
        edge_vox = nz[:, 0] + vs
        edge_ax = nz[:, 1]
        del nz
        for s in range(0, m_sub, edge_chunk):
            e = min(s + edge_chunk, m_sub)
            c_len = e - s
            c_sub = coords[edge_vox[s:e]].unsqueeze(1) + edge_offsets[edge_ax[s:e]]
            q_keys = torch_mod.cat(
                [
                    torch_mod.zeros((c_len * 4, 1), dtype=torch_mod.int32, device=dev),
                    c_sub.reshape(-1, 3).int(),
                ],
                dim=1,
            )
            del c_sub
            q_idx = o_voxel_c.hashmap_lookup_3d_cuda(hashmap_keys, hashmap_vals, q_keys, *gs).reshape(
                c_len, 4
            ).int()
            del q_keys
            valid = (q_idx != 0xFFFFFFFF).all(dim=1)
            if bool(valid.any()):
                quad_chunks.append(q_idx[valid].int())
            del q_idx, valid
        del edge_vox, edge_ax

    del hashmap_keys, hashmap_vals, edge_offsets
    quad_indices = (
        torch_mod.cat(quad_chunks, dim=0)
        if quad_chunks
        else torch_mod.empty((0, 4), dtype=torch_mod.int32, device=dev)
    )
    del quad_chunks

    mesh_vertices = torch_mod.empty((n, 3), dtype=torch_mod.float32, device=dev)
    aabb_min = aabb_t[0].reshape(1, 3)
    for s in range(0, n, edge_chunk):
        e = min(s + edge_chunk, n)
        mv = coords[s:e].to(torch_mod.float32)
        mv.add_(dual_vertices[s:e]).mul_(voxel_size).add_(aabb_min)
        mesh_vertices[s:e].copy_(mv)
        del mv

    num_quads = int(quad_indices.shape[0])
    mesh_triangles = torch_mod.empty((num_quads * 2, 3), dtype=quad_indices.dtype, device=dev)
    quad_split_1 = torch_mod.tensor([0, 1, 2, 0, 2, 3], dtype=torch_mod.long, device=dev)
    quad_split_2 = torch_mod.tensor([0, 1, 3, 3, 1, 2], dtype=torch_mod.long, device=dev)
    for s in range(0, num_quads, edge_chunk):
        e = min(s + edge_chunk, num_quads)
        q_sub = quad_indices[s:e]
        sw = split_weight[q_sub]
        cond = (sw[:, 0] * sw[:, 2]) > (sw[:, 1] * sw[:, 3])
        tris = torch_mod.where(cond, q_sub[:, quad_split_1], q_sub[:, quad_split_2]).reshape(-1, 3)
        mesh_triangles[2 * s : 2 * e].copy_(tris)
        del q_sub, sw, cond, tris
    del quad_indices
    return mesh_vertices, mesh_triangles


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
    if _is_already_patched(fdg_cls, "_trellis2_fdg_patched"):
        return
    orig_forward = fdg_cls.forward
    dual_grid_mod = sys.modules.get(getattr(dual_grid_fn, "__module__", ""))
    dual_grid_c = getattr(dual_grid_mod, "_C", None) if dual_grid_mod is not None else None

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
            if len(layout) == 1:
                del coords_all
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
                del feats_all
                free_gb = _cuda_free_gb(torch_mod)
                if torch_mod.cuda.is_available() and (free_gb is None or free_gb < VRAM_SAFETY_FLOOR_GB + 1.0):
                    torch_mod.cuda.empty_cache()
            aabb_box = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
            extracted = _chunked_fdg_to_mesh(
                c_b,
                v_f,
                i_f,
                q_f,
                aabb=aabb_box,
                grid_size=self.resolution,
                torch_mod=torch_mod,
                o_voxel_c=dual_grid_c,
            )
            if extracted is None:
                extracted = dual_grid_fn(
                    c_b,
                    v_f,
                    i_f,
                    q_f,
                    aabb=aabb_box,
                    grid_size=self.resolution,
                    train=False,
                )
            meshes.append(mesh_cls(*extracted))
            del extracted, c_b, v_f, i_f, q_f

        out_list[0] = meshes
        return out_list[0] if len(out_list) == 1 else tuple(out_list)

    fdg_cls.forward = _low_mem_fdg_forward
    fdg_cls._trellis2_fdg_patched = True


def patch_sparse_ffn(
    ffn_cls: Any,
    torch_mod: Any,
    f_mod: Any,
    chunk_size: int = DEFAULT_TOKEN_CHUNK_SIZE,
) -> None:
    """Chunk SparseFeedForwardNet (1536 -> 8192 -> 1536) when VRAM is tight; fuse with Inductor when VRAM is ample."""
    if _is_already_patched(ffn_cls, "_trellis2_vram_patched"):
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
    if sparse_mha_cls is not None and not _is_already_patched(sparse_mha_cls, "_trellis2_kv_cache_patched"):
        orig_sparse_fwd = sparse_mha_cls.forward

        def _cached_sparse_mha_forward(self: Any, x: Any, context: Any = None) -> Any:
            s_attn = (
                sparse_attn_fn_holder.sparse_scaled_dot_product_attention
                if sparse_attn_fn_holder is not None
                else getattr(self, "_sparse_sdpa_fn", None)
            )
            if (
                getattr(self, "_type", None) == "self"
                and getattr(self, "attn_mode", None) == "full"
                and (getattr(self, "qk_rms_norm", False) or getattr(self, "use_rope", False))
                and not getattr(self, "training", False)
                and s_attn is not None
                and os.environ.get("TRELLIS2_FAST_VRAM", "1") != "0"
            ):
                # Avoid repacking [q, k, v] via torch.stack (~400-471 MB per block x 30 blocks x 21 passes)
                # by invoking flash_attn_varlen_func(q, k, v) directly (bit-identical FlashAttention-2 math).
                qkv = self._linear(self.to_qkv, x)
                qkv = self._fused_pre(qkv, num_fused=3)
                q, k, v = qkv.unbind(dim=-3)
                del qkv
                if self.qk_rms_norm:
                    q = self.q_rms_norm(q)
                    k = self.k_rms_norm(k)
                if self.use_rope:
                    q, k = self.rope(q, k)
                h = s_attn(q, k, v)
                del q, k, v
                h = self._reshape_chs(h, (-1,))
                return self._linear(self.to_out, h)

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

    if dense_mha_cls is not None and not _is_already_patched(dense_mha_cls, "_trellis2_kv_cache_patched"):
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
    if not _is_already_patched(cfg_mixin_cls, "_trellis2_batch_cfg_patched"):
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
                    max_batch_tokens = max(
                        1,
                        int(
                            os.environ.get(
                                "TRELLIS2_BATCH_CFG_MAX_TOKENS",
                                str(BATCH_CFG_MAX_SPARSE_TOKENS),
                            )
                        ),
                    )
                    if m > max_batch_tokens:
                        # Above ~24K tokens (1408^3/1536^3), B=1 already saturates RTX 4090 Tensor Cores;
                        # sequential CFG matches wall-clock speed while saving ~3.5-4.0 GiB activation VRAM.
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

    if flow_euler_cls is not None and not _is_already_patched(flow_euler_cls, "_trellis2_fast_euler_patched"):
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
    if ss_flow_cls is None or _is_already_patched(ss_flow_cls, "_trellis2_cudagraph_patched"):
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


def _compact_mesh_vertices(
    vertices: Any,
    faces: Any,
    torch_mod: Any,
    face_chunk: int = 4_000_000,
    return_keep_idx: bool = False,
) -> tuple[Any, ...]:
    """Strip unreferenced dual-grid vertices before CuMesh to shrink vertex/boundary buffers ~5-7x."""
    if vertices is None or faces is None or torch_mod is None:
        return (vertices, faces, None) if return_keep_idx else (vertices, faces)
    try:
        v_count = int(vertices.shape[0])
        f_count = int(faces.shape[0])
    except Exception:
        return (vertices, faces, None) if return_keep_idx else (vertices, faces)
    if v_count <= 0 or f_count <= 0:
        if return_keep_idx:
            return vertices, faces, torch_mod.arange(v_count, dtype=torch_mod.long, device=vertices.device)
        return vertices, faces

    dev = vertices.device
    used = torch_mod.zeros((v_count,), dtype=torch_mod.bool, device=dev)
    for s in range(0, f_count, face_chunk):
        e = min(s + face_chunk, f_count)
        f_slice = faces[s:e]
        if f_slice.device != dev:
            f_slice = f_slice.to(dev)
        used[f_slice.reshape(-1).long()] = True

    keep_idx = used.nonzero(as_tuple=True)[0]
    del used
    new_v_count = int(keep_idx.shape[0])
    if new_v_count >= v_count:
        if return_keep_idx:
            return vertices, faces, keep_idx
        del keep_idx
        return vertices, faces

    remap = torch_mod.empty((v_count,), dtype=faces.dtype, device=dev)
    remap[keep_idx] = torch_mod.arange(new_v_count, dtype=faces.dtype, device=dev)
    new_vertices = vertices[keep_idx].contiguous()
    if not return_keep_idx:
        del keep_idx
    new_faces = torch_mod.empty_like(faces, device=dev)
    for s in range(0, f_count, face_chunk):
        e = min(s + face_chunk, f_count)
        f_slice = faces[s:e]
        if f_slice.device != dev:
            f_slice = f_slice.to(dev)
        new_faces[s:e].copy_(remap[f_slice.long()])
    del remap
    if return_keep_idx:
        return new_vertices, new_faces, keep_idx
    return new_vertices, new_faces


def _run_cumesh_simplify_once(
    v_dev: Any,
    f_dev: Any,
    target: int,
    torch_mod: Any,
    cumesh_mod: Any,
    verbose: bool = False,
    opts: dict[str, Any] | None = None,
    compact_out: bool = True,
) -> tuple[Any, Any]:
    """Run a single CuMesh simplify pass after dropping caller input buffers on GPU."""
    use_cuda = bool(getattr(v_dev, "is_cuda", False) and torch_mod.cuda.is_available())
    cm = cumesh_mod.CuMesh()
    try:
        cm.init(v_dev, f_dev)
        del v_dev, f_dev
        if use_cuda:
            gc.collect()
            torch_mod.cuda.empty_cache()
        cm.simplify(target, verbose=verbose, options=opts or {})
        out_v, out_f = cm.read()
        del cm
        cm = None
        if use_cuda:
            gc.collect()
            torch_mod.cuda.empty_cache()
        if compact_out:
            out_v, out_f = _compact_mesh_vertices(out_v, out_f, torch_mod)
        return out_v, out_f
    except Exception as exc:
        if exc.__traceback__ is not None:
            traceback.clear_frames(exc.__traceback__)
        cm = None
        if use_cuda or _is_oom_like(exc):
            _clear_sticky_cuda_error(torch_mod)
        if use_cuda:
            gc.collect()
            torch_mod.cuda.empty_cache()
        raise


def _choose_min_seam_split_axis(
    v_work: Any,
    f_work: Any,
    num_slabs: int,
    torch_mod: Any,
    phase_shift: float = 0.0,
) -> tuple[int, Any]:
    """Select the spatial axis with the smallest cross-slab seam perimeter for quantile binning."""
    dev = v_work.device
    v_count = int(v_work.shape[0])
    f_count = int(f_work.shape[0])
    v_stride = max(1, v_count // 500_000)
    f_stride = max(1, f_count // 500_000)
    f_sample = f_work[::f_stride].long()

    # Quantile interior cut positions (shifted optionally on recursive retry to move cut lines off old seams)
    if phase_shift > 0.0:
        qs = torch_mod.linspace(0.0, 1.0, num_slabs + 1, dtype=torch_mod.float32, device=dev)[1:-1]
        step_q = 1.0 / float(num_slabs)
        qs = torch_mod.clamp(qs + (phase_shift * step_q), 0.02, 0.98)
    else:
        qs = torch_mod.linspace(0.0, 1.0, num_slabs + 1, dtype=torch_mod.float32, device=dev)[1:-1]

    span = v_work.max(dim=0).values - v_work.min(dim=0).values
    max_span = max(float(span.max().item()), 1e-8)

    best_axis = int(span.argmax().item())
    best_cuts = None
    best_seam = None
    for ax in range(3):
        v_ax_c = v_work[:, ax].contiguous()
        cuts_ax = torch_mod.quantile(v_ax_c[::v_stride].float(), qs).to(v_ax_c.dtype)
        # Skip severely squashed axes relative to longest span unless nothing better
        if float(span[ax].item()) < 0.25 * max_span and best_cuts is not None:
            continue
        vb_ax = torch_mod.bucketize(v_ax_c, cuts_ax)
        fb0 = vb_ax[f_sample[:, 0]]
        fb1 = vb_ax[f_sample[:, 1]]
        fb2 = vb_ax[f_sample[:, 2]]
        seam_est = int((~((fb0 == fb1) & (fb1 == fb2))).sum().item())
        if best_seam is None or seam_est < best_seam:
            best_seam = seam_est
            best_axis = ax
            best_cuts = cuts_ax
        del v_ax_c, vb_ax, fb0, fb1, fb2

    assert best_cuts is not None
    return best_axis, best_cuts


def _eval_qem_points(eq: Any, pt: Any) -> Any:
    """Evaluate packed symmetric 4x4 QEM error v^T Q v (clamped >= 0) for edge endpoints."""
    x, y, z = pt[:, 0], pt[:, 1], pt[:, 2]
    return (
        eq[:, 0] * x * x
        + 2.0 * eq[:, 1] * x * y
        + 2.0 * eq[:, 2] * x * z
        + eq[:, 3] * y * y
        + 2.0 * eq[:, 4] * y * z
        + eq[:, 5] * z * z
        + 2.0 * eq[:, 6] * x
        + 2.0 * eq[:, 7] * y
        + 2.0 * eq[:, 8] * z
        + eq[:, 9]
    ).clamp_min(0.0)


def _qem_coarsen_slab_locked(
    v_sub: Any,
    f_sub: Any,
    is_locked: Any,
    target_faces: int,
    torch_mod: Any,
    lambda_el: float = 1e-3,
    max_iters: int = 24,
) -> tuple[Any, Any]:
    """Vectorized GPU/CPU QEM edge-collapse that keeps locked/seam/boundary vertices fixed in place."""
    dev = v_sub.device
    v_cnt = int(v_sub.shape[0])
    f_cur = int(f_sub.shape[0])
    if f_cur <= target_faces or v_cnt <= 3:
        return v_sub, f_sub

    f = f_sub.to(torch_mod.int32).clone() if f_sub.dtype != torch_mod.int32 else f_sub.clone()
    p_min = v_sub.min(dim=0).values
    p_max = v_sub.max(dim=0).values
    center = 0.5 * (p_min + p_max)
    inv_scale = 2.0 / max(float((p_max - p_min).max().item()), 1e-8)
    p = (v_sub - center) * inv_scale

    # Chunked initial per-vertex quadrics (keeps peak VRAM small even on 40M+ face slabs)
    qem = torch_mod.zeros((v_cnt, 10), dtype=torch_mod.float32, device=dev)
    face_chunk = 4_000_000
    for s in range(0, f_cur, face_chunk):
        e = min(s + face_chunk, f_cur)
        fc = f[s:e].long()
        p0, p1, p2 = p[fc[:, 0]], p[fc[:, 1]], p[fc[:, 2]]
        n = torch_mod.cross(p1 - p0, p2 - p0, dim=-1)
        nl = n.norm(dim=-1, keepdim=True)
        n = torch_mod.where(nl > 1e-12, n / nl.clamp_min(1e-12), torch_mod.zeros_like(n))
        d = -(n * p0).sum(dim=-1)
        nx, ny, nz = n[:, 0], n[:, 1], n[:, 2]
        fq = torch_mod.stack(
            [nx * nx, nx * ny, nx * nz, ny * ny, ny * nz, nz * nz, nx * d, ny * d, nz * d, d * d],
            dim=-1,
        )
        for col in range(3):
            qem.index_add_(0, fc[:, col], fq)
        del fc, p0, p1, p2, n, nl, d, nx, ny, nz, fq

    sentinel = 1 << 62
    edge_chunk = 4_000_000

    for _it in range(max_iters):
        f_cur = int(f.shape[0])
        if f_cur <= target_faces:
            break

        # Build undirected edge keys in one pre-allocated int64 buffer (avoids (3F,2) intermediate)
        e_pack = torch_mod.empty((3 * f_cur,), dtype=torch_mod.int64, device=dev)
        for ei, (ia, ib) in enumerate(((0, 1), (1, 2), (2, 0))):
            off = ei * f_cur
            for s in range(0, f_cur, face_chunk):
                e = min(s + face_chunk, f_cur)
                va = f[s:e, ia].long()
                vb = f[s:e, ib].long()
                lo = torch_mod.minimum(va, vb)
                hi = torch_mod.maximum(va, vb)
                e_pack[off + s : off + e] = (lo << 32) | hi
        u_edges, e_cnts = torch_mod.unique(e_pack, return_counts=True)
        del e_pack

        e0 = u_edges >> 32
        e1 = u_edges & 0xFFFFFFFF
        del u_edges

        # Protect seam vertices, natural open boundaries, and non-manifold vertices from any collapse
        prot_nm = is_locked.clone()
        nm_e = e_cnts > 2
        if bool(nm_e.any()):
            prot_nm[e0[nm_e]] = True
            prot_nm[e1[nm_e]] = True

        prot = prot_nm.clone()
        bnd_e = e_cnts == 1
        if bool(bnd_e.any()):
            prot[e0[bnd_e]] = True
            prot[e1[bnd_e]] = True
        del e_cnts, bnd_e, nm_e

        free_mask = (~prot[e0]) & (~prot[e1])
        if not bool(free_mask.any()):
            # Fallback for thin open ribbons (e.g. single-strip hair tips) where every edge touches a natural boundary;
            # seam vertices (`is_locked`) and non-manifold vertices (`prot_nm`) remain strictly immovable.
            free_mask = (~prot_nm[e0]) & (~prot_nm[e1])
        del prot, prot_nm
        c0 = e0[free_mask]
        c1 = e1[free_mask]
        del e0, e1, free_mask

        m = int(c0.shape[0])
        if m == 0:
            break

        # Chunked edge QEM evaluation -> stores only float32 cost + uint8 endpoint choice (0=mid, 1=v0, 2=v1)
        costs = torch_mod.empty((m,), dtype=torch_mod.float32, device=dev)
        choices = torch_mod.empty((m,), dtype=torch_mod.uint8, device=dev)
        for s in range(0, m, edge_chunk):
            e = min(s + edge_chunk, m)
            s0, s1 = c0[s:e], c1[s:e]
            p0_c, p1_c = p[s0], p[s1]
            pm_c = 0.5 * (p0_c + p1_c)
            eq_c = qem[s0] + qem[s1]
            qm_c = _eval_qem_points(eq_c, pm_c)
            q0_c = _eval_qem_points(eq_c, p0_c)
            q1_c = _eval_qem_points(eq_c, p1_c)
            best_q, best_ch = torch_mod.stack([qm_c, q0_c, q1_c], dim=-1).min(dim=-1)
            l2_c = ((p1_c - p0_c) ** 2).sum(dim=-1)
            costs[s:e] = best_q + lambda_el * l2_c
            choices[s:e] = best_ch.to(torch_mod.uint8)
            del s0, s1, p0_c, p1_c, pm_c, eq_c, qm_c, q0_c, q1_c, best_q, best_ch, l2_c

        remove_need = f_cur - target_faces
        frac = min(1.0, max(0.20, (remove_need / max(1, 2 * m)) * 5.0))
        if frac < 0.99 and m > 256:
            # Sample <=500k elements before torch.quantile (PyTorch limits quantile() to <= 2^24 elements)
            c_samp = costs[:: max(1, m // 500_000)].float()
            cutoff = torch_mod.quantile(c_samp, frac)
            sel = costs <= cutoff
            c0, c1, choices, costs = c0[sel], c1[sel], choices[sel], costs[sel]
            del c_samp, cutoff, sel
            m = int(c0.shape[0])
            if m == 0:
                del costs, choices
                break

        # Quantize QEM cost into 16 log-scale tiers (bits 58..61) + 31-bit avalanche hash + 27-bit edge id
        log_c = torch_mod.log10(costs.float().clamp_min(1e-12))
        log_lo = float(log_c.min().item())
        log_hi = max(log_lo + 0.5, float(log_c.max().item()))
        tier = torch_mod.clamp(((log_c - log_lo) / (log_hi - log_lo) * 15.99).to(torch_mod.int64), 0, 15)
        z = ((c0 + 1) * 2654435761) ^ ((c1 + 1) * 2246822519) ^ ((_it + 1) * 3266489917)
        z = (z ^ (z >> 16)) * 3266489917
        z = (z ^ (z >> 15)) * 2654435761
        h = (z ^ (z >> 17)) & ((1 << 31) - 1)
        eid_low = torch_mod.arange(m, dtype=torch_mod.int64, device=dev) & ((1 << 27) - 1)
        keys = (tier << 58) | (h << 27) | eid_low
        del costs, log_c, tier, z, h, eid_low

        # Strictly 2-hop / 1-ring disjoint independent set (no two winners touch the same face)
        v_best = torch_mod.full((v_cnt,), sentinel, dtype=torch_mod.int64, device=dev)
        v_best.scatter_reduce_(0, c0, keys, reduce="amin", include_self=True)
        v_best.scatter_reduce_(0, c1, keys, reduce="amin", include_self=True)

        v_win = torch_mod.full((v_cnt,), sentinel, dtype=torch_mod.int64, device=dev)
        for s in range(0, f_cur, face_chunk):
            e = min(s + face_chunk, f_cur)
            fc = f[s:e].long()
            fb_c = torch_mod.minimum(
                torch_mod.minimum(v_best[fc[:, 0]], v_best[fc[:, 1]]), v_best[fc[:, 2]]
            )
            for col in range(3):
                v_win.scatter_reduce_(0, fc[:, col], fb_c, reduce="amin", include_self=True)
            del fc, fb_c
        del v_best

        win_mask = (v_win[c0] == keys) & (v_win[c1] == keys)
        del v_win, keys
        wc0 = c0[win_mask]
        wc1 = c1[win_mask]
        wch = choices[win_mask]
        del c0, c1, choices, win_mask

        w_cnt = int(wc0.shape[0])
        if w_cnt == 0:
            break

        max_collapses = max(1, (f_cur - target_faces + 1) // 2)
        if w_cnt > max_collapses:
            wc0 = wc0[:max_collapses]
            wc1 = wc1[:max_collapses]
            wch = wch[:max_collapses]
            w_cnt = max_collapses

        p0_w, p1_w = p[wc0], p[wc1]
        pm_w = 0.5 * (p0_w + p1_w)
        w_p = torch_mod.where(
            (wch == 1).unsqueeze(-1),
            p0_w,
            torch_mod.where((wch == 2).unsqueeze(-1), p1_w, pm_w),
        )
        del p0_w, p1_w, pm_w, wch

        # Tentative collapse validation: reject normal flips, zero-area slivers, and non-manifold pinches
        v_to = torch_mod.arange(v_cnt, dtype=torch_mod.int32, device=dev)
        v_to[wc1] = wc0.to(torch_mod.int32)
        p_tent = p.clone()
        p_tent[wc0] = w_p

        w_tag = torch_mod.full((v_cnt,), -1, dtype=torch_mod.int32, device=dev)
        w_ids = torch_mod.arange(w_cnt, dtype=torch_mod.int32, device=dev)
        w_tag[wc0] = w_ids
        w_tag[wc1] = w_ids
        del w_ids

        orig_chunks: list[Any] = []
        new_chunks: list[Any] = []
        tag_chunks: list[Any] = []
        for s in range(0, f_cur, face_chunk):
            e = min(s + face_chunk, f_cur)
            f_sub_c = f[s:e].long()
            ft_c = torch_mod.maximum(
                torch_mod.maximum(w_tag[f_sub_c[:, 0]], w_tag[f_sub_c[:, 1]]), w_tag[f_sub_c[:, 2]]
            )
            chg = ft_c >= 0
            if bool(chg.any()):
                fo_c = f_sub_c[chg]
                fn_c = v_to[fo_c].long()
                ftg_c = ft_c[chg]
                sv_c = (fn_c[:, 0] != fn_c[:, 1]) & (fn_c[:, 1] != fn_c[:, 2]) & (fn_c[:, 0] != fn_c[:, 2])
                if bool(sv_c.any()):
                    orig_chunks.append(fo_c[sv_c])
                    new_chunks.append(fn_c[sv_c])
                    tag_chunks.append(ftg_c[sv_c])
        del w_tag, v_to

        win_ok = torch_mod.ones((w_cnt,), dtype=torch_mod.bool, device=dev)
        if orig_chunks:
            f_s_orig = torch_mod.cat(orig_chunks, dim=0)
            f_s_new = torch_mod.cat(new_chunks, dim=0)
            f_s_tag = torch_mod.cat(tag_chunks, dim=0)
            del orig_chunks, new_chunks, tag_chunks

            old_n = torch_mod.cross(
                p[f_s_orig[:, 1]] - p[f_s_orig[:, 0]],
                p[f_s_orig[:, 2]] - p[f_s_orig[:, 0]],
                dim=-1,
            )
            new_n = torch_mod.cross(
                p_tent[f_s_new[:, 1]] - p_tent[f_s_new[:, 0]],
                p_tent[f_s_new[:, 2]] - p_tent[f_s_new[:, 0]],
                dim=-1,
            )
            on_l = old_n.norm(dim=-1)
            nn_l = new_n.norm(dim=-1)
            bad_flip = ((old_n * new_n).sum(dim=-1) <= 1e-6 * on_l * nn_l) | (nn_l <= 1e-12)
            if bool(bad_flip.any()):
                win_ok[f_s_tag[bad_flip].long()] = False
            del old_n, new_n, on_l, nn_l, bad_flip, f_s_orig

            is_keep_col = f_s_new == wc0[f_s_tag.long()].unsqueeze(-1)
            others = f_s_new[~is_keep_col].reshape(-1, 2)
            del is_keep_col, f_s_new
            e_ku = torch_mod.cat(
                [
                    torch_mod.stack([f_s_tag.long(), others[:, 0]], dim=-1),
                    torch_mod.stack([f_s_tag.long(), others[:, 1]], dim=-1),
                ],
                dim=0,
            )
            u_ku, ku_cnts = torch_mod.unique((e_ku[:, 0] << 32) | e_ku[:, 1], return_counts=True)
            del e_ku
            bad_nm = ku_cnts > 2
            if bool(bad_nm.any()):
                win_ok[u_ku[bad_nm] >> 32] = False
            del u_ku, ku_cnts, bad_nm

            olo = torch_mod.minimum(others[:, 0], others[:, 1])
            ohi = torch_mod.maximum(others[:, 0], others[:, 1])
            dup_key = torch_mod.stack([f_s_tag.long(), olo, ohi], dim=-1)
            u_dk, u_dc = torch_mod.unique(dup_key, dim=0, return_counts=True)
            bad_dup = u_dc > 1
            if bool(bad_dup.any()):
                win_ok[u_dk[bad_dup][:, 0]] = False
            del others, olo, ohi, dup_key, u_dk, u_dc, bad_dup, f_s_tag
        else:
            del orig_chunks, new_chunks, tag_chunks

        del p_tent
        wc0 = wc0[win_ok]
        wc1 = wc1[win_ok]
        w_p = w_p[win_ok]
        del win_ok
        if int(wc0.shape[0]) == 0:
            break

        # Commit verified collapses
        v_to = torch_mod.arange(v_cnt, dtype=torch_mod.int32, device=dev)
        v_to[wc1] = wc0.to(torch_mod.int32)
        p[wc0] = w_p
        qem[wc0] += qem[wc1]
        del wc0, wc1, w_p

        surv_list: list[Any] = []
        for s in range(0, f_cur, face_chunk):
            e = min(s + face_chunk, f_cur)
            fn_c = v_to[f[s:e].long()]
            keep_c = (fn_c[:, 0] != fn_c[:, 1]) & (fn_c[:, 1] != fn_c[:, 2]) & (fn_c[:, 0] != fn_c[:, 2])
            if bool(keep_c.any()):
                surv_list.append(fn_c[keep_c])
        del v_to
        f = torch_mod.cat(surv_list, dim=0) if surv_list else torch_mod.empty((0, 3), dtype=torch_mod.int32, device=dev)
        del surv_list

    del qem
    v_out = (p / inv_scale) + center
    return v_out.to(v_sub.dtype), f


def _weld_and_coarsen_slabs(
    v_work: Any,
    f_work: Any,
    mid_target: int,
    num_slabs: int,
    torch_mod: Any,
    use_cuda: bool,
    phase_shift: float = 0.0,
) -> tuple[Any, Any]:
    """Partition mesh cleanly along min-seam axis, lock & weld seam vertices globally, and QEM-coarsen slab interiors."""
    dev = v_work.device
    v_cnt = int(v_work.shape[0])
    f_cnt = int(f_work.shape[0])
    axis, cuts = _choose_min_seam_split_axis(v_work, f_work, num_slabs, torch_mod, phase_shift=phase_shift)
    v_ax = v_work[:, axis].contiguous()
    v_bin = torch_mod.bucketize(v_ax, cuts).to(torch_mod.int8)
    del v_ax, cuts

    f_slab = torch_mod.empty((f_cnt,), dtype=torch_mod.int8, device=dev)
    seam_verts_mask = torch_mod.zeros((v_cnt,), dtype=torch_mod.bool, device=dev)
    chunk_sz = 8_000_000
    for s in range(0, f_cnt, chunk_sz):
        e = min(s + chunk_sz, f_cnt)
        f_sub = f_work[s:e].long()
        b0 = v_bin[f_sub[:, 0]]
        b1 = v_bin[f_sub[:, 1]]
        b2 = v_bin[f_sub[:, 2]]
        same = (b0 == b1) & (b1 == b2)
        f_slab[s:e] = torch_mod.where(same, b0, torch_mod.tensor(-1, dtype=torch_mod.int8, device=dev))
        if not bool(same.all()):
            seam_verts_mask[f_sub[~same].reshape(-1)] = True
        del f_sub, b0, b1, b2, same
    del v_bin

    seam_faces = f_work[f_slab < 0]
    seam_count = int(seam_faces.shape[0])
    int_total = max(1, f_cnt - seam_count)
    int_mid_budget = max(1, mid_target - seam_count)

    global_to_merged = torch_mod.full((v_cnt,), -1, dtype=torch_mod.int32, device=dev)
    seam_gvids = torch_mod.nonzero(seam_verts_mask, as_tuple=False).reshape(-1)
    n_seam_v = int(seam_gvids.shape[0])

    merged_verts: list[Any] = []
    merged_faces: list[Any] = []
    next_v_off = 0
    if n_seam_v > 0:
        global_to_merged[seam_gvids] = torch_mod.arange(n_seam_v, dtype=torch_mod.int32, device=dev)
        sv = v_work[seam_gvids]
        merged_verts.append(sv.cpu() if use_cuda else sv)
        del sv
        next_v_off = n_seam_v
    del seam_gvids

    if seam_count > 0:
        sf_merged = torch_mod.empty((seam_count, 3), dtype=torch_mod.int32, device=dev)
        for s in range(0, seam_count, chunk_sz):
            e = min(s + chunk_sz, seam_count)
            sf_merged[s:e].copy_(global_to_merged[seam_faces[s:e].long()])
        merged_faces.append(sf_merged.cpu() if use_cuda else sf_merged)
        del sf_merged
    del seam_faces

    for k in range(num_slabs):
        slab_faces = f_work[f_slab == k]
        k_cnt = int(slab_faces.shape[0])
        if k_cnt == 0:
            del slab_faces
            continue
        k_target = max(1, int(round(int_mid_budget * (k_cnt / int_total))))
        vk, fk, keep_gids = _compact_mesh_vertices(v_work, slab_faces, torch_mod, return_keep_idx=True)
        del slab_faces
        assert keep_gids is not None
        lk = seam_verts_mask[keep_gids]
        vk_out, fk_out = _qem_coarsen_slab_locked(vk, fk, lk, min(k_target, k_cnt), torch_mod)
        del vk, fk

        used_mask = torch_mod.zeros((int(vk_out.shape[0]),), dtype=torch_mod.bool, device=dev)
        fk_out_cnt = int(fk_out.shape[0])
        for s in range(0, fk_out_cnt, chunk_sz):
            e = min(s + chunk_sz, fk_out_cnt)
            used_mask[fk_out[s:e].reshape(-1).long()] = True

        new_int_mask = used_mask & (~lk)
        loc_to_merged = torch_mod.full((int(vk_out.shape[0]),), -1, dtype=torch_mod.int32, device=dev)
        if bool(lk.any()):
            loc_to_merged[lk] = global_to_merged[keep_gids[lk]]
        n_new = int(new_int_mask.sum().item())
        if n_new > 0:
            loc_to_merged[new_int_mask] = torch_mod.arange(
                next_v_off, next_v_off + n_new, dtype=torch_mod.int32, device=dev
            )
            v_new_part = vk_out[new_int_mask]
            merged_verts.append(v_new_part.cpu() if use_cuda else v_new_part)
            del v_new_part
            next_v_off += n_new
        del used_mask, new_int_mask, lk, keep_gids, vk_out

        fk_merged = torch_mod.empty((fk_out_cnt, 3), dtype=torch_mod.int32, device=dev)
        for s in range(0, fk_out_cnt, chunk_sz):
            e = min(s + chunk_sz, fk_out_cnt)
            fk_merged[s:e].copy_(loc_to_merged[fk_out[s:e].long()])
        merged_faces.append(fk_merged.cpu() if use_cuda else fk_merged)
        del loc_to_merged, fk_out, fk_merged

    del v_work, f_work, f_slab, seam_verts_mask, global_to_merged
    if use_cuda:
        gc.collect()
        torch_mod.cuda.empty_cache()

    all_v = torch_mod.cat(merged_verts, dim=0) if merged_verts else torch_mod.empty((0, 3), dtype=torch_mod.float32)
    all_f = torch_mod.cat(merged_faces, dim=0) if merged_faces else torch_mod.empty((0, 3), dtype=torch_mod.int32)
    return all_v, all_f


def _simplify_mesh_oom_safe(
    mesh_obj: Any,
    target: int,
    torch_mod: Any,
    cumesh_mod: Any,
    verbose: bool = False,
    options: dict[str, Any] | None = None,
) -> None:
    """Simplify Mesh via CuMesh using welded seam-locked Stage-1 QEM + global Stage-2 CuMesh without OOM."""
    verts = mesh_obj.vertices
    faces = mesh_obj.faces
    if verts is None or faces is None:
        return
    f_count = int(faces.shape[0])
    if f_count <= target:
        return

    opts = options or {}
    use_cuda = bool(torch_mod.cuda.is_available())
    orig_dev = verts.device
    work_dev = torch_mod.device("cuda") if use_cuda else orig_dev

    # CuMesh.simplify allocates ~185 bytes/face across edges, CUB sort temp, vert2face, and QEMs.
    # Proactively bypass doomed single-shot simplify calls when F exceeds ~85% of free VRAM.
    single_shot_cap: int | None = None
    if use_cuda and hasattr(torch_mod.cuda, "mem_get_info"):
        try:
            free_bytes, _ = torch_mod.cuda.mem_get_info(0)
            single_shot_cap = max(25_000_000, int((free_bytes * 0.85) // 185))
        except Exception:
            single_shot_cap = 105_000_000
    env_cap = os.environ.get("TRELLIS2_SINGLE_SHOT_SIMPLIFY_MAX_FACES", "").strip()
    if env_cap:
        single_shot_cap = int(env_cap)

    skip_single_shot = bool(single_shot_cap is not None and f_count > single_shot_cap)

    v_work = verts.to(work_dev).float().contiguous() if verts.device != work_dev else verts.float().contiguous()
    f_work = faces.to(work_dev).int().contiguous() if faces.device != work_dev else faces.int().contiguous()
    # Release Mesh references so vertices+faces aren't held twice during simplify (~2.7 GiB saved)
    mesh_obj.vertices = None
    mesh_obj.faces = None
    del verts, faces
    if use_cuda:
        gc.collect()
        torch_mod.cuda.empty_cache()

    if not skip_single_shot:
        # Keep lightweight CPU backup for oversized meshes (>16.77M) in case single-shot still OOMs
        keep_backup = bool(orig_dev.type != "cuda" or f_count > 16_777_216)
        v_bak = v_work.cpu() if keep_backup else None
        f_bak = f_work.cpu() if keep_backup else None
        try:
            out_v, out_f = _run_cumesh_simplify_once(
                v_work, f_work, target, torch_mod, cumesh_mod, verbose=verbose, opts=opts
            )
            del v_bak, f_bak
            mesh_obj.vertices = out_v.to(orig_dev) if out_v.device != orig_dev else out_v
            mesh_obj.faces = out_f.to(orig_dev) if out_f.device != orig_dev else out_f
            return
        except Exception as exc:
            if not _is_oom_like(exc) or v_bak is None or f_bak is None:
                raise
            log.warning(
                "Single-shot CuMesh.simplify(%d) hit VRAM limit at %d faces; switching to seam-locked 2-stage simplify (%s)",
                target,
                f_count,
                " ".join(s.strip() for s in str(exc).splitlines() if s.strip())[:160],
            )
            v_work = v_bak.to(work_dev).contiguous()
            f_work = f_bak.to(work_dev).contiguous()
            del v_bak, f_bak
            eff_cap = max(target, int(f_count * 0.72))
    else:
        eff_cap = int(single_shot_cap)  # type: ignore[arg-type]
        log.info(
            "Using seam-locked 2-stage GPU simplify for %d -> %d faces (single-shot cap=%d)",
            f_count,
            target,
            eff_cap,
        )

    # Stage 1: Seam-locked, globally welded QEM coarsening down to ~48M-60M (well below eff_cap & >target so Stage 2 runs globally).
    safe_s2_cap = max(target, int(eff_cap * 0.60))
    if safe_s2_cap <= target:
        safe_s2_cap = max(target, int(eff_cap * 0.90))
    desired_mid = max(
        target * 3 if eff_cap >= target * 4 else max(target + 1, int(target * 1.5)),
        48_000_000 if f_count > 60_000_000 else min(f_count, target * 2),
    )
    mid_target = min(f_count, desired_mid, safe_s2_cap)
    num_slabs = max(2, min(8, (f_count + 34_999_999) // 35_000_000))

    all_v, all_f = _weld_and_coarsen_slabs(
        v_work, f_work, mid_target, num_slabs, torch_mod, use_cuda, phase_shift=0.0
    )
    del v_work, f_work

    # Stage 2: Single-shot global CuMesh.simplify(target) across the 100% welded intermediate mesh.
    # Unifies density across former seam bands and eliminates slab boundary artifacts completely.
    cur_f_cnt = int(all_f.shape[0])
    if cur_f_cnt > target:
        for attempt in range(2):
            v_bak_s2 = all_v.cpu() if attempt == 0 else None
            f_bak_s2 = all_f.cpu() if attempt == 0 else None
            all_v_dev = all_v.to(work_dev).contiguous() if all_v.device != work_dev else all_v.contiguous()
            all_f_dev = all_f.to(work_dev).contiguous() if all_f.device != work_dev else all_f.contiguous()
            del all_v, all_f
            try:
                all_v, all_f = _run_cumesh_simplify_once(
                    all_v_dev, all_f_dev, target, torch_mod, cumesh_mod, verbose=verbose, opts=opts
                )
                del v_bak_s2, f_bak_s2
                break
            except Exception as exc:
                if not _is_oom_like(exc) or attempt >= 1 or v_bak_s2 is None or f_bak_s2 is None:
                    raise
                tighter_mid = max(target, int(cur_f_cnt * 0.65))
                log.warning(
                    "Stage-2 CuMesh.simplify(%d) hit VRAM limit at %d faces; coarsening shifted slabs -> %d and retrying",
                    target,
                    cur_f_cnt,
                    tighter_mid,
                )
                v_retry = v_bak_s2.to(work_dev).contiguous()
                f_retry = f_bak_s2.to(work_dev).contiguous()
                del v_bak_s2, f_bak_s2
                all_v, all_f = _weld_and_coarsen_slabs(
                    v_retry, f_retry, tighter_mid, num_slabs, torch_mod, use_cuda, phase_shift=0.5
                )
                cur_f_cnt = int(all_f.shape[0])
                if cur_f_cnt <= target:
                    break

    mesh_obj.vertices = all_v.to(orig_dev) if all_v.device != orig_dev else all_v
    mesh_obj.faces = all_f.to(orig_dev) if all_f.device != orig_dev else all_f


def patch_mesh_simplify_noop(mesh_cls: Any, torch_mod: Any = None, cumesh_mod: Any = None) -> None:
    """Skip redundant CuMesh build+readback when <= target and avoid duplicate GPU buffers on simplify."""
    if mesh_cls is None or _is_already_patched(mesh_cls, "_trellis2_simplify_patched"):
        return
    orig_simplify = mesh_cls.simplify

    def _fast_simplify(
        self: Any, target: int = 1000000, verbose: bool = False, options: dict[str, Any] = {}
    ) -> None:
        faces = getattr(self, "faces", None)
        shape = getattr(faces, "shape", None)
        if shape is not None and len(shape) >= 1 and shape[0] <= target:
            return
        if (
            cumesh_mod is not None
            and torch_mod is not None
            and hasattr(cumesh_mod, "CuMesh")
            and getattr(self, "vertices", None) is not None
        ):
            _simplify_mesh_oom_safe(
                self,
                target=int(target),
                torch_mod=torch_mod,
                cumesh_mod=cumesh_mod,
                verbose=verbose,
                options=options,
            )
            return
        if torch_mod is not None:
            gc.collect()
            if hasattr(torch_mod, "cuda") and torch_mod.cuda.is_available():
                torch_mod.cuda.empty_cache()
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


def _chunked_remesh_narrow_band_dc(
    vertices: Any,
    faces: Any,
    center: Any,
    scale: float,
    resolution: int,
    band: float = 1,
    project_back: float = 0,
    verbose: bool = False,
    bvh: Any = None,
    *,
    remeshing_mod: Any,
    torch_mod: Any,
    udf_chunk: int = 8_000_000,
    vox_chunk: int = 2_000_000,
    quad_chunk: int = 2_000_000,
    proj_chunk: int = 4_000_000,
) -> tuple[Any, Any]:
    """Bit-exact memory-optimal replacement for cumesh.remeshing.remesh_narrow_band_dc.

    Upstream keeps >10 GiB of dead intermediates (edge_neighbor_voxel, connected_voxel_hash_key,
    connected_voxel, hashmap_vert, hashmap_vox, pts_vert, grid_verts) alive simultaneously with
    unchunked 21M-quad torch.cross splits (~22 GiB peak at 1408-1536). Releasing intermediates at
    last use and chunking UDF/hashmap/quad-split/projection cuts peak VRAM to ~2.6 GiB with
    100% identical output vertices and faces.
    """
    assert vertices.ndim == 2 and vertices.shape[1] == 3 and vertices.dtype == torch_mod.float32
    assert faces.ndim == 2 and faces.shape[1] == 3 and faces.dtype == torch_mod.int32
    assert center.ndim == 1 and center.shape[0] == 3

    device = vertices.device
    center = center.to(device=device, dtype=torch_mod.float32)
    c_ext = remeshing_mod._C
    init_hashmap = remeshing_mod._init_hashmap

    edge_neighbor_offset = torch_mod.tensor(
        [
            [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
            [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
            [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
        ],
        dtype=torch_mod.int32,
        device=device,
    )  # (3, 4, 3)
    quad_split_1_n = torch_mod.tensor([0, 1, 2, 0, 2, 3], dtype=torch_mod.long, device=device)
    quad_split_1_p = torch_mod.tensor([0, 2, 1, 0, 3, 2], dtype=torch_mod.long, device=device)
    quad_split_2_n = torch_mod.tensor([0, 1, 3, 3, 1, 2], dtype=torch_mod.long, device=device)
    quad_split_2_p = torch_mod.tensor([0, 3, 1, 3, 2, 1], dtype=torch_mod.long, device=device)
    offsets = torch_mod.tensor(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        dtype=torch_mod.int32,
        device=device,
    )

    own_bvh = bvh is None
    if own_bvh:
        if verbose:
            print("Building BVH...")
        bvh = remeshing_mod.cuBVH(vertices, faces)

    eps = band * scale / resolution
    base_resolution = int(resolution)
    while base_resolution > 32:
        assert base_resolution % 2 == 0, "Failed to find a base resolution that is a multiple of 2"
        base_resolution //= 2

    coords = (
        torch_mod.stack(
            torch_mod.meshgrid(
                torch_mod.arange(base_resolution, device=device),
                torch_mod.arange(base_resolution, device=device),
                torch_mod.arange(base_resolution, device=device),
                indexing="ij",
            ),
            dim=-1,
        )
        .int()
        .reshape(-1, 3)
    )

    tqdm_cls = getattr(remeshing_mod, "tqdm", None)
    total_steps = int(torch_mod.log2(torch_mod.tensor(resolution // base_resolution)).item()) + 1
    pbar = (
        tqdm_cls(total=total_steps, desc="Building Sparse Grid", disable=not verbose)
        if callable(tqdm_cls)
        else None
    )

    while True:
        cell_size = scale / base_resolution
        thresh = 0.87 * cell_size
        n_c = int(coords.shape[0])
        if n_c <= udf_chunk:
            pts = ((coords.float() + 0.5) / base_resolution - 0.5) * scale + center
            distances = bvh.unsigned_distance(pts)[0]
            del pts
            distances = torch_mod.abs(distances - eps)
            subdiv_mask = distances < thresh
            del distances
        else:
            subdiv_mask = torch_mod.empty((n_c,), dtype=torch_mod.bool, device=device)
            for s in range(0, n_c, udf_chunk):
                e = min(n_c, s + udf_chunk)
                pts_c = ((coords[s:e].float() + 0.5) / base_resolution - 0.5) * scale + center
                d_c = bvh.unsigned_distance(pts_c)[0]
                del pts_c
                d_c = torch_mod.abs(d_c - eps)
                subdiv_mask[s:e] = d_c < thresh
                del d_c
        coords = coords[subdiv_mask]
        del subdiv_mask

        if base_resolution >= resolution:
            break

        base_resolution *= 2
        coords = (coords * 2).unsqueeze(1) + offsets.unsqueeze(0)
        coords = coords.reshape(-1, 3)
        if pbar is not None:
            pbar.update(1)

    nvox = int(coords.shape[0])
    hashmap_vox = init_hashmap(resolution, 2 * nvox, device)
    c_ext.hashmap_insert_3d_idx_as_val_cuda(
        *hashmap_vox,
        torch_mod.cat([torch_mod.zeros_like(coords[:, :1]), coords], dim=1),
        resolution,
        resolution,
        resolution,
    )

    coords = coords.contiguous()
    grid_verts = c_ext.get_sparse_voxel_grid_active_vertices(
        *hashmap_vox, coords, resolution, resolution, resolution
    )
    nvert = int(grid_verts.shape[0])

    if nvert <= udf_chunk:
        pts_vert = (grid_verts.float() / resolution - 0.5) * scale + center
        distances_vert = bvh.unsigned_distance(pts_vert)[0]
        del pts_vert
        distances_vert = distances_vert - eps
    else:
        distances_vert = torch_mod.empty((nvert,), dtype=torch_mod.float32, device=device)
        for s in range(0, nvert, udf_chunk):
            e = min(nvert, s + udf_chunk)
            pts_c = (grid_verts[s:e].float() / resolution - 0.5) * scale + center
            d_c = bvh.unsigned_distance(pts_c)[0]
            del pts_c
            distances_vert[s:e] = d_c - eps
            del d_c

    if pbar is not None:
        pbar.update(1)
        pbar.close()

    if verbose:
        print("Running Dual Contouring...")

    hashmap_vert = init_hashmap(resolution + 1, 2 * nvert, device)
    c_ext.hashmap_insert_3d_idx_as_val_cuda(
        *hashmap_vert,
        torch_mod.cat([torch_mod.zeros_like(grid_verts[:, :1]), grid_verts], dim=1),
        resolution + 1,
        resolution + 1,
        resolution + 1,
    )
    del grid_verts

    dual_verts, intersected = c_ext.simple_dual_contour(
        *hashmap_vert, coords, distances_vert, resolution + 1, resolution + 1, resolution + 1
    )
    del hashmap_vert, distances_vert

    quad_chunks: list[Any] = []
    dir_chunks: list[Any] = []
    step_vox = max(1, int(vox_chunk))
    for s in range(0, nvox, step_vox):
        e = min(nvox, s + step_vox)
        i_sub = intersected[s:e]
        nz_v, nz_ax = torch_mod.nonzero(i_sub != 0, as_tuple=True)
        if int(nz_v.numel()) == 0:
            del i_sub, nz_v, nz_ax
            continue
        conn_v = coords[s:e][nz_v].unsqueeze(1) + edge_neighbor_offset[nz_ax]  # (m_c, 4, 3)
        i_vals = i_sub[nz_v, nz_ax]
        del i_sub, nz_v, nz_ax
        m_c = int(conn_v.shape[0])
        h_key = torch_mod.cat(
            [
                torch_mod.zeros((m_c * 4, 1), dtype=torch_mod.int32, device=device),
                conn_v.reshape(-1, 3),
            ],
            dim=1,
        )
        del conn_v
        c_idx = (
            c_ext.hashmap_lookup_3d_cuda(
                *hashmap_vox, h_key, resolution, resolution, resolution
            )
            .reshape(m_c, 4)
            .int()
        )
        del h_key
        c_valid = (c_idx != 0xFFFFFFFF).all(dim=1)
        if bool(c_valid.any()):
            quad_chunks.append(c_idx[c_valid].int())
            dir_chunks.append(i_vals[c_valid].int())
        del c_idx, c_valid, i_vals

    del coords, intersected, hashmap_vox

    if quad_chunks:
        quad_indices = quad_chunks[0] if len(quad_chunks) == 1 else torch_mod.cat(quad_chunks, dim=0)
        intersected_dir = dir_chunks[0] if len(dir_chunks) == 1 else torch_mod.cat(dir_chunks, dim=0)
    else:
        quad_indices = torch_mod.empty((0, 4), dtype=torch_mod.int32, device=device)
        intersected_dir = torch_mod.empty((0,), dtype=torch_mod.int32, device=device)
    del quad_chunks, dir_chunks

    unique_verts = torch_mod.unique(quad_indices.reshape(-1))
    dual_verts = dual_verts[unique_verts]
    vert_map = torch_mod.zeros((nvox,), dtype=torch_mod.int32, device=device)
    vert_map[unique_verts] = torch_mod.arange(
        unique_verts.shape[0], dtype=torch_mod.int32, device=device
    )
    del unique_verts
    quad_indices = vert_map[quad_indices]
    del vert_map

    mesh_vertices = (dual_verts / resolution - 0.5) * scale + center
    del dual_verts

    num_quads = int(quad_indices.shape[0])
    mesh_triangles = torch_mod.empty((num_quads * 2, 3), dtype=torch_mod.int32, device=device)
    step_quad = max(1, int(quad_chunk))
    for s in range(0, num_quads, step_quad):
        e = min(num_quads, s + step_quad)
        q_c = quad_indices[s:e]
        d_pos = (intersected_dir[s:e] == 1).unsqueeze(1)
        t0 = torch_mod.where(d_pos, q_c[:, quad_split_1_p], q_c[:, quad_split_1_n])
        n0 = torch_mod.linalg.cross(
            mesh_vertices[t0[:, 1]] - mesh_vertices[t0[:, 0]],
            mesh_vertices[t0[:, 2]] - mesh_vertices[t0[:, 0]],
            dim=-1,
        )
        n1 = torch_mod.linalg.cross(
            mesh_vertices[t0[:, 2]] - mesh_vertices[t0[:, 1]],
            mesh_vertices[t0[:, 3]] - mesh_vertices[t0[:, 1]],
            dim=-1,
        )
        align0 = (n0 * n1).sum(dim=1).abs()
        del n0, n1

        t1 = torch_mod.where(d_pos, q_c[:, quad_split_2_p], q_c[:, quad_split_2_n])
        n0 = torch_mod.linalg.cross(
            mesh_vertices[t1[:, 1]] - mesh_vertices[t1[:, 0]],
            mesh_vertices[t1[:, 2]] - mesh_vertices[t1[:, 0]],
            dim=-1,
        )
        n1 = torch_mod.linalg.cross(
            mesh_vertices[t1[:, 2]] - mesh_vertices[t1[:, 1]],
            mesh_vertices[t1[:, 3]] - mesh_vertices[t1[:, 1]],
            dim=-1,
        )
        align1 = (n0 * n1).sum(dim=1).abs()
        del n0, n1, d_pos, q_c

        chosen = torch_mod.where((align0 > align1).unsqueeze(1), t0, t1).reshape(-1, 3).int()
        mesh_triangles[2 * s : 2 * e] = chosen
        del t0, t1, align0, align1, chosen

    del quad_indices, intersected_dir

    if project_back > 0 and int(mesh_vertices.shape[0]) > 0:
        if verbose:
            print("Projecting back to original mesh...")
        n_mv = int(mesh_vertices.shape[0])
        step_proj = max(1, int(proj_chunk))
        for s in range(0, n_mv, step_proj):
            e = min(n_mv, s + step_proj)
            mv_c = mesh_vertices[s:e]
            _, face_id, uvw = bvh.unsigned_distance(mv_c, return_uvw=True)
            orig_tri_verts = vertices[faces[face_id.long()]]
            projected_verts = (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)
            mv_c.sub_(project_back * (mv_c - projected_verts))
            del face_id, uvw, orig_tri_verts, projected_verts

    if own_bvh:
        del bvh

    return mesh_vertices, mesh_triangles


def patch_cumesh_remeshing(remeshing_mod: Any, torch_mod: Any = None) -> None:
    """Patch cumesh.remeshing.remesh_narrow_band_dc to eliminate >12 GiB of temporary VRAM spikes."""
    if remeshing_mod is None or getattr(remeshing_mod, "_trellis2_remesh_patched", False):
        return
    orig_remesh = getattr(remeshing_mod, "remesh_narrow_band_dc", None)
    if not callable(orig_remesh):
        return
    t_mod = torch_mod if torch_mod is not None else getattr(remeshing_mod, "torch", None)
    if t_mod is None:
        return

    def _patched_remesh_narrow_band_dc(
        vertices: Any,
        faces: Any,
        center: Any,
        scale: float,
        resolution: int,
        band: float = 1,
        project_back: float = 0,
        verbose: bool = False,
        bvh: Any = None,
    ) -> tuple[Any, Any]:
        if os.environ.get("TRELLIS2_CHUNKED_REMESH", "1") == "0":
            return orig_remesh(
                vertices,
                faces,
                center=center,
                scale=scale,
                resolution=resolution,
                band=band,
                project_back=project_back,
                verbose=verbose,
                bvh=bvh,
            )
        return _chunked_remesh_narrow_band_dc(
            vertices,
            faces,
            center=center,
            scale=scale,
            resolution=resolution,
            band=band,
            project_back=project_back,
            verbose=verbose,
            bvh=bvh,
            remeshing_mod=remeshing_mod,
            torch_mod=t_mod,
        )

    remeshing_mod.remesh_narrow_band_dc = _patched_remesh_narrow_band_dc
    remeshing_mod._trellis2_remesh_patched = True


def patch_o_voxel_postprocess(postprocess_mod: Any, torch_mod: Any = None) -> None:
    """Optimize o_voxel.postprocess.to_glb for 24 GB VRAM at 1536^3/4K + <5ms GPU UV-gutter dilation."""
    if postprocess_mod is None or getattr(postprocess_mod, "_trellis2_postprocess_patched", False):
        return
    try:
        src = inspect.getsource(postprocess_mod.to_glb)
    except Exception:
        src = ""

    old_inpaint_block = (
        "    base_color = cv2.inpaint(base_color, mask_inv, 3, cv2.INPAINT_TELEA)\n"
        "    metallic = cv2.inpaint(metallic, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]\n"
        "    roughness = cv2.inpaint(roughness, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]\n"
        "    alpha = cv2.inpaint(alpha, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]"
    )

    inpaint_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="trellis2-inpaint")

    def _parallel_pbr_inpaint(
        base_color: Any, metallic: Any, roughness: Any, alpha: Any, mask_inv: Any
    ) -> tuple[Any, Any, Any, Any]:
        t_local = torch_mod if torch_mod is not None else getattr(postprocess_mod, "torch", None)
        if (
            t_local is not None
            and os.environ.get("TRELLIS2_FAST_VRAM", "1") != "0"
            and os.environ.get("TRELLIS2_GPU_UV_DILATE", "1") != "0"
            and t_local.cuda.is_available()
        ):
            try:
                import torch.nn.functional as f_mod

                return gpu_dilate_uv_gutters(
                    base_color, metallic, roughness, alpha, mask_inv, t_local, f_mod
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

    is_full_to_glb = (
        hasattr(postprocess_mod, "cumesh")
        and hasattr(postprocess_mod, "dr")
        and hasattr(postprocess_mod, "grid_sample_3d")
        and hasattr(postprocess_mod, "trimesh")
        and ("remesh_narrow_band_dc" in src or src == "")
    )
    if is_full_to_glb:
        t_mod = torch_mod if torch_mod is not None else postprocess_mod.torch
        np_mod = postprocess_mod.np

        def _low_vram_to_glb(
            vertices: Any,
            faces: Any,
            attr_volume: Any,
            coords: Any,
            attr_layout: dict[str, slice],
            aabb: Any,
            voxel_size: Any = None,
            grid_size: Any = None,
            decimation_target: int = 1000000,
            texture_size: int = 2048,
            remesh: bool = False,
            remesh_band: float = 1,
            remesh_project: float = 0.9,
            mesh_cluster_threshold_cone_half_angle_rad: float = float(np_mod.radians(90.0)),
            mesh_cluster_refine_iterations: int = 0,
            mesh_cluster_global_iterations: int = 1,
            mesh_cluster_smooth_strength: float = 1,
            verbose: bool = False,
            use_tqdm: bool = False,
        ) -> Any:
            cumesh_local = postprocess_mod.cumesh
            dr_local = postprocess_mod.dr
            grid_sample_fn = postprocess_mod.grid_sample_3d
            trimesh_local = postprocess_mod.trimesh
            pil_image = postprocess_mod.Image
            tqdm_local = postprocess_mod.tqdm

            use_cuda = bool(t_mod.cuda.is_available())
            target_dev = t_mod.device("cuda") if use_cuda else vertices.device

            def _to_dev(t: Any) -> Any:
                if use_cuda and hasattr(t, "cuda") and getattr(t.device, "type", "") != "cuda":
                    return t.cuda()
                if hasattr(t, "to") and t.device != target_dev:
                    return t.to(target_dev)
                return t

            def _flush_allocator_for_cpp() -> None:
                """Release PyTorch cached blocks back to the CUDA driver before raw C++ cudaMalloc calls."""
                gc.collect()
                if use_cuda:
                    t_mod.cuda.empty_cache()

            stage_vol_cpu = bool(
                use_cuda and os.environ.get("TRELLIS2_STAGE_BAKE_VOLUME_CPU", "1") != "0"
            )
            attr_vol_staged = attr_volume.cpu() if (stage_vol_cpu and hasattr(attr_volume, "cpu")) else attr_volume
            coords_staged = coords.cpu() if (stage_vol_cpu and hasattr(coords, "cpu")) else coords
            attr_channels = int(attr_vol_staged.shape[1])
            del attr_volume, coords

            if isinstance(aabb, (list, tuple)):
                aabb = np_mod.array(aabb)
            if isinstance(aabb, np_mod.ndarray):
                aabb = t_mod.tensor(aabb, dtype=t_mod.float32, device=target_dev)
            else:
                aabb = _to_dev(aabb).to(dtype=t_mod.float32)
            assert isinstance(aabb, t_mod.Tensor) and aabb.dim() == 2 and aabb.size(0) == 2 and aabb.size(1) == 3

            if voxel_size is not None:
                if isinstance(voxel_size, (int, float)):
                    voxel_size = [voxel_size, voxel_size, voxel_size]
                if isinstance(voxel_size, (list, tuple)):
                    voxel_size = np_mod.array(voxel_size)
                if isinstance(voxel_size, np_mod.ndarray):
                    voxel_size = t_mod.tensor(voxel_size, dtype=t_mod.float32, device=target_dev)
                else:
                    voxel_size = _to_dev(voxel_size).to(dtype=t_mod.float32)
                grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()
            else:
                assert grid_size is not None, "Either voxel_size or grid_size must be provided"
                if isinstance(grid_size, int):
                    grid_size = [grid_size, grid_size, grid_size]
                if isinstance(grid_size, (list, tuple)):
                    grid_size = np_mod.array(grid_size)
                if isinstance(grid_size, np_mod.ndarray):
                    grid_size = t_mod.tensor(grid_size, dtype=t_mod.int32, device=target_dev)
                else:
                    grid_size = _to_dev(grid_size).to(dtype=t_mod.int32)
                voxel_size = (aabb[1] - aabb[0]) / grid_size

            pbar = tqdm_local(total=6, desc="Extracting GLB") if use_tqdm else None
            if verbose:
                print(f"Original mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces")

            _flush_allocator_for_cpp()
            vertices = _to_dev(vertices)
            faces = _to_dev(faces)
            vertices, faces = _compact_mesh_vertices(vertices, faces, t_mod)
            _flush_allocator_for_cpp()

            mesh = cumesh_local.CuMesh()
            mesh.init(vertices, faces)
            mesh.fill_holes(max_hole_perimeter=3e-2)
            if verbose:
                print(f"After filling holes: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
            vertices, faces = mesh.read()
            if pbar is not None:
                pbar.update(1)

            if remesh:
                # Destroy initial 16M+ face C++ CuMesh buffers before building cuBVH & Dual Contouring
                del mesh
                mesh = None
                _flush_allocator_for_cpp()

            if pbar is not None:
                pbar.set_description("Building BVH")
            if verbose:
                print("Building BVH for current mesh...", end="", flush=True)
            bvh = cumesh_local.cuBVH(vertices, faces)
            if pbar is not None:
                pbar.update(1)
            if verbose:
                print("Done")

            if pbar is not None:
                pbar.set_description("Cleaning mesh")
            if verbose:
                print("Cleaning mesh...")

            orig_verts_cpu = None
            orig_faces_cpu = None

            if not remesh:
                assert mesh is not None
                _flush_allocator_for_cpp()
                mesh.simplify(decimation_target * 3, verbose=verbose)
                if verbose:
                    print(f"After inital simplification: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
                mesh.remove_duplicate_faces()
                mesh.repair_non_manifold_edges()
                mesh.remove_small_connected_components(1e-5)
                mesh.fill_holes(max_hole_perimeter=3e-2)
                if verbose:
                    print(f"After initial cleanup: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
                mesh.simplify(decimation_target, verbose=verbose)
                if verbose:
                    print(f"After final simplification: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
                mesh.remove_duplicate_faces()
                mesh.repair_non_manifold_edges()
                mesh.remove_small_connected_components(1e-5)
                mesh.fill_holes(max_hole_perimeter=3e-2)
                if verbose:
                    print(f"After final cleanup: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
                mesh.unify_face_orientations()
            else:
                center = aabb.mean(dim=0)
                scale = float((aabb[1] - aabb[0]).max().item())
                full_res = int(grid_size.max().item())
                dc_candidates = [full_res]
                if full_res > 1024 and os.environ.get("TRELLIS2_DC_OOM_FALLBACK", "1") != "0":
                    dc_candidates.append(1024)

                for idx_cand, dc_res in enumerate(dc_candidates):
                    dc_verts = None
                    dc_faces = None
                    try:
                        if vertices is None:
                            vertices = _to_dev(orig_verts_cpu)
                            faces = _to_dev(orig_faces_cpu)
                        _flush_allocator_for_cpp()
                        dc_verts, dc_faces = cumesh_local.remeshing.remesh_narrow_band_dc(
                            vertices,
                            faces,
                            center=center,
                            scale=(dc_res + 3 * remesh_band) / dc_res * scale,
                            resolution=dc_res,
                            band=remesh_band,
                            project_back=remesh_project,
                            verbose=verbose,
                            bvh=bvh,
                        )
                        if stage_vol_cpu and vertices is not None:
                            orig_verts_cpu = vertices.cpu()
                            orig_faces_cpu = faces.cpu()
                            vertices = None
                            faces = None
                        mesh = cumesh_local.CuMesh()
                        mesh.init(dc_verts, dc_faces)
                        del dc_verts, dc_faces
                        dc_verts = None
                        dc_faces = None
                        # Release all PyTorch cached segments BEFORE CuMesh C++ simplify calls raw cudaMalloc()
                        _flush_allocator_for_cpp()
                        if verbose:
                            print(f"After remeshing: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
                        mesh.simplify(decimation_target, verbose=verbose)
                        if verbose:
                            print(f"After simplifying: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
                        break
                    except Exception as remesh_exc:
                        if remesh_exc.__traceback__ is not None:
                            traceback.clear_frames(remesh_exc.__traceback__)
                        mesh = None
                        del dc_verts, dc_faces
                        _clear_sticky_cuda_error(t_mod)
                        _flush_allocator_for_cpp()
                        if idx_cand + 1 < len(dc_candidates) and _is_oom_like(remesh_exc):
                            next_dc = dc_candidates[idx_cand + 1]
                            log.warning(
                                "CuMesh remesh/simplify hit VRAM limit at dc_resolution=%d (%s); "
                                "retrying DC cage at %d while preserving %d^3 BVH & PBR volume.",
                                dc_res,
                                " ".join(s.strip() for s in str(remesh_exc).splitlines() if s.strip())[:200],
                                next_dc,
                                full_res,
                            )
                            continue
                        raise

            if pbar is not None:
                pbar.update(1)
            if verbose:
                print("Done")

            if pbar is not None:
                pbar.set_description("Parameterizing new mesh")
            if verbose:
                print("Parameterizing new mesh...")

            assert mesh is not None
            _flush_allocator_for_cpp()
            out_vertices, out_faces, out_uvs, out_vmaps = mesh.uv_unwrap(
                compute_charts_kwargs={
                    "threshold_cone_half_angle_rad": mesh_cluster_threshold_cone_half_angle_rad,
                    "refine_iterations": mesh_cluster_refine_iterations,
                    "global_iterations": mesh_cluster_global_iterations,
                    "smooth_strength": mesh_cluster_smooth_strength,
                },
                return_vmaps=True,
                verbose=verbose,
            )
            out_vertices = _to_dev(out_vertices)
            out_faces = _to_dev(out_faces)
            out_uvs = _to_dev(out_uvs)
            out_vmaps = _to_dev(out_vmaps)
            mesh.compute_vertex_normals()
            out_normals = _to_dev(mesh.read_vertex_normals())[out_vmaps]
            del out_vmaps, mesh
            _flush_allocator_for_cpp()

            if pbar is not None:
                pbar.update(1)
            if verbose:
                print("Done")

            if pbar is not None:
                pbar.set_description("Sampling attributes")
            if verbose:
                print("Sampling attributes...", end="", flush=True)

            ctx = dr_local.RasterizeCudaContext()
            uvs_rast = t_mod.cat(
                [out_uvs * 2 - 1, t_mod.zeros_like(out_uvs[:, :1]), t_mod.ones_like(out_uvs[:, :1])],
                dim=-1,
            ).unsqueeze(0)
            rast = t_mod.zeros((1, texture_size, texture_size, 4), device=target_dev, dtype=t_mod.float32)

            for i in range(0, int(out_faces.shape[0]), 100000):
                rast_chunk, _ = dr_local.rasterize(
                    ctx,
                    uvs_rast,
                    out_faces[i : i + 100000],
                    resolution=[texture_size, texture_size],
                )
                mask_chunk = rast_chunk[..., 3:4] > 0
                rast_chunk[..., 3:4] += i
                rast = t_mod.where(mask_chunk, rast_chunk, rast)
                del rast_chunk, mask_chunk

            mask = rast[0, ..., 3] > 0
            pos = dr_local.interpolate(out_vertices.unsqueeze(0), rast, out_faces)[0][0]
            valid_pos = pos[mask]
            del pos, rast, uvs_rast, ctx

            if vertices is None:
                vertices = _to_dev(orig_verts_cpu)
                faces = _to_dev(orig_faces_cpu)
            del orig_verts_cpu, orig_faces_cpu

            n_valid = int(valid_pos.shape[0])
            proj_step = 4_000_000
            if n_valid <= proj_step:
                _, face_id, uvw = bvh.unsigned_distance(valid_pos, return_uvw=True)
                orig_tri_verts = vertices[faces[face_id.long()]]
                valid_pos = (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)
                del face_id, uvw, orig_tri_verts
            else:
                projected_pos = t_mod.empty_like(valid_pos)
                for s in range(0, n_valid, proj_step):
                    e = min(n_valid, s + proj_step)
                    _, face_id, uvw = bvh.unsigned_distance(valid_pos[s:e], return_uvw=True)
                    orig_tri_verts = vertices[faces[face_id.long()]]
                    projected_pos[s:e] = (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)
                    del face_id, uvw, orig_tri_verts
                valid_pos = projected_pos
                del projected_pos

            # Free cuBVH + original 16M-face Dual-Grid mesh BEFORE moving PBR voxel volume onto GPU
            del bvh, vertices, faces
            _flush_allocator_for_cpp()

            attr_vol_dev = _to_dev(attr_vol_staged)
            coords_dev = _to_dev(coords_staged)
            del attr_vol_staged, coords_staged

            attrs = t_mod.zeros(
                texture_size, texture_size, attr_channels, device=target_dev, dtype=attr_vol_dev.dtype
            )
            if n_valid > 0:
                attrs[mask] = grid_sample_fn(
                    attr_vol_dev,
                    t_mod.cat([t_mod.zeros_like(coords_dev[:, :1]), coords_dev], dim=-1),
                    shape=t_mod.Size([1, attr_channels, *grid_size.tolist()]),
                    grid=((valid_pos - aabb[0]) / voxel_size).reshape(1, -1, 3),
                    mode="trilinear",
                )
            del attr_vol_dev, coords_dev, valid_pos
            _flush_allocator_for_cpp()

            if pbar is not None:
                pbar.update(1)
            if verbose:
                print("Done")

            if pbar is not None:
                pbar.set_description("Finalizing mesh")
            if verbose:
                print("Finalizing mesh...", end="", flush=True)

            mask_np = mask.cpu().numpy()
            del mask

            base_color = np_mod.clip(attrs[..., attr_layout["base_color"]].float().cpu().numpy() * 255, 0, 255).astype(np_mod.uint8)
            metallic = np_mod.clip(attrs[..., attr_layout["metallic"]].float().cpu().numpy() * 255, 0, 255).astype(np_mod.uint8)
            roughness = np_mod.clip(attrs[..., attr_layout["roughness"]].float().cpu().numpy() * 255, 0, 255).astype(np_mod.uint8)
            alpha = np_mod.clip(attrs[..., attr_layout["alpha"]].float().cpu().numpy() * 255, 0, 255).astype(np_mod.uint8)
            del attrs
            alpha_mode = "OPAQUE"

            mask_inv = (~mask_np).astype(np_mod.uint8)
            base_color, metallic, roughness, alpha = _parallel_pbr_inpaint(
                base_color, metallic, roughness, alpha, mask_inv
            )

            material = trimesh_local.visual.material.PBRMaterial(
                baseColorTexture=pil_image.fromarray(np_mod.concatenate([base_color, alpha], axis=-1)),
                baseColorFactor=np_mod.array([255, 255, 255, 255], dtype=np_mod.uint8),
                metallicRoughnessTexture=pil_image.fromarray(
                    np_mod.concatenate([np_mod.zeros_like(metallic), roughness, metallic], axis=-1)
                ),
                metallicFactor=1.0,
                roughnessFactor=1.0,
                alphaMode=alpha_mode,
                doubleSided=True if not remesh else False,
            )

            vertices_np = out_vertices.cpu().numpy()
            faces_np = out_faces.cpu().numpy()
            uvs_np = out_uvs.cpu().numpy()
            normals_np = out_normals.cpu().numpy()
            del out_vertices, out_faces, out_uvs, out_normals

            vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2].copy(), -vertices_np[:, 1].copy()
            normals_np[:, 1], normals_np[:, 2] = normals_np[:, 2].copy(), -normals_np[:, 1].copy()
            uvs_np[:, 1] = 1 - uvs_np[:, 1]

            textured_mesh = trimesh_local.Trimesh(
                vertices=vertices_np,
                faces=faces_np,
                vertex_normals=normals_np,
                process=False,
                visual=trimesh_local.visual.TextureVisuals(uv=uvs_np, material=material),
            )

            if pbar is not None:
                pbar.update(1)
                pbar.close()
            if verbose:
                print("Done")

            return textured_mesh

        postprocess_mod._trellis2_parallel_pbr_inpaint = _parallel_pbr_inpaint
        postprocess_mod.to_glb = _low_vram_to_glb
        postprocess_mod._trellis2_postprocess_patched = True
        return

    if old_inpaint_block not in src:
        return

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
        host_lim_gb, host_avail_gb = _host_ram_stats_gb()
        pin_mode = os.environ.get("TRELLIS2_PIN_MEMORY", "0").strip().lower()
        # Pinning 15+ GiB of offloaded DiT weights in PyTorch's CachingHostAllocator permanently locks
        # unswappable pages and triggers container OOM kills (exit 137) even when bare-metal /proc/meminfo
        # reports >64 GiB. Keep disabled by default unless explicitly opted in via TRELLIS2_PIN_MEMORY=1.
        want_pin = pin_mode in {"1", "true", "yes", "on"}
        self._pin_enabled = bool(
            enable_pin_memory
            and os.name != "nt"
            and want_pin
            and torch_mod is not None
            and torch_mod.cuda.is_available()
        )
        if host_lim_gb is not None or host_avail_gb is not None:
            log.info(
                "Detected Host/cgroup RAM: limit=%s GiB, available=%s GiB (pin_memory=%s)",
                f"{host_lim_gb:.2f}" if host_lim_gb is not None else "unlimited",
                f"{host_avail_gb:.2f}" if host_avail_gb is not None else "unknown",
                self._pin_enabled,
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
        else:
            inner = self._unwrap_module(obj)
            if inner is not obj and callable(getattr(inner, "cpu", None)):
                inner.cpu()
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

    def evict_except(self, keep: Sequence[str] = (), clear_cache: bool = True) -> int:
        """Evict all GPU-resident models not in `keep` back to pinned CPU RAM."""
        keep_set = set(keep)
        if self._prefetch_stream is not None:
            try:
                self._prefetch_stream.synchronize()
            except Exception:
                pass
        evicted = 0
        for name in list(self._resident_lru.keys()):
            if name not in keep_set:
                self._move_to_cpu(name)
                evicted += 1
        for name in list(self._inflight_events.keys()):
            if name not in keep_set:
                self._move_to_cpu(name)
                evicted += 1
        if clear_cache and self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
        return evicted

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
    if _is_already_patched(slat_flow_cls, "_trellis2_dit_stream_patched"):
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


def _freeze_pipeline_parameters(pipeline: Any) -> int:
    """Set requires_grad_(False) on all pipeline sub-models for pure inference.

    PyTorch's autograd.Function ctx.needs_input_grad inspects tensor.requires_grad even inside
    torch.inference_mode(). Freezing model parameters forces FlexGEMM's SubMConv3dFunction onto
    its *_no_bwd neighbor-map path, saving ~7-9 GiB of backward-only buffers at 1536^3.
    """
    frozen = 0
    seen: set[int] = set()
    candidates: list[Any] = list((getattr(pipeline, "models", None) or {}).values())
    for attr in ("image_cond_model", "rembg_model"):
        extra = getattr(pipeline, attr, None)
        if extra is not None:
            candidates.append(extra)
    for obj in candidates:
        for mod in (obj, getattr(obj, "model", None)):
            if mod is None or id(mod) in seen:
                continue
            seen.add(id(mod))
            req_fn = getattr(mod, "requires_grad_", None)
            if callable(req_fn):
                try:
                    req_fn(False)
                    frozen += 1
                except Exception:
                    pass
    return frozen


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
    import trellis2.models.sc_vaes.sparse_unet_vae as sparse_unet_vae_mod
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

    patch_flex_gemm_autotune_cache_safety()
    patch_layernorm32(LayerNorm32, torch_mod, DEFAULT_VOXEL_CHUNK_SIZE)
    try:
        import flex_gemm.ops.spconv.submanifold_conv3d as flex_subm_mod  # type: ignore[import-not-found]

        patch_flex_gemm_neighbor_cache(flex_subm_mod, torch_mod)
    except Exception as exc:
        log.debug("Skipping FlexGEMM neighbor cache patch: %s", exc)
    patch_sparse_convnext_block(
        SparseConvNeXtBlock3d, f_mod, DEFAULT_VOXEL_CHUNK_SIZE, torch_mod=torch_mod
    )
    patch_sparse_res_block(
        getattr(sparse_unet_vae_mod, "SparseResBlock3d", None),
        torch_mod,
        f_mod,
        DEFAULT_VOXEL_CHUNK_SIZE,
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
    try:
        import cumesh as _cumesh_top  # type: ignore[import-not-found]
    except Exception:
        _cumesh_top = None
    patch_mesh_simplify_noop(Mesh, torch_mod, cumesh_mod=_cumesh_top)
    patch_cv2_inpaint_fastpath(cv2, np)
    try:
        import cumesh.remeshing as cumesh_remeshing_mod  # type: ignore[import-not-found]

        patch_cumesh_remeshing(cumesh_remeshing_mod, torch_mod)
    except Exception as exc:
        log.debug("Skipping CuMesh remeshing patch: %s", exc)
    patch_o_voxel_postprocess(o_voxel_postprocess_mod, torch_mod)

    models_dict = getattr(pipeline, "models", {}) or {}
    frozen_mods = _freeze_pipeline_parameters(pipeline)
    log.info("Froze inference parameters (requires_grad=False) on %d pipeline modules", frozen_mods)
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
        if offload_models:
            _clear_sticky_cuda_error(self._torch)

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
        return _is_oom_like(exc)

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
                max_num_tokens = int(
                    os.environ.get("TRELLIS2_MAX_NUM_TOKENS", str(DEFAULT_MAX_NUM_TOKENS))
                )

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
                    # Evict stale 512/conditioning models FIRST so _configure_decoder_residency sees
                    # post-eviction free VRAM (>= floor + 4.5 GiB) and keeps shape_dec 100% GPU-resident.
                    mgr.ensure_on_gpu(
                        ["shape_slat_decoder"],
                        pipe.device,
                        required_free_gb=floor + 4.5,
                        protected=["shape_slat_flow_model_1024"],
                    )
                    self._configure_decoder_residency(shape_dec)
                    mgr.prefetch_async(
                        ["shape_slat_flow_model_1024", "tex_slat_flow_model_1024"],
                        pipe.device,
                        min_free_after_gb=floor + 4.5,
                    )
                else:
                    shape_dec.low_vram = True
                hr_coords = shape_dec.upsample(slat_lr, upsample_times=4)
                _flush_flex_gemm_autotune_cache()
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
                        else:
                            log.info(
                                "Cascade target resolution %d achieved (tokens=%d <= max_num_tokens=%d)",
                                hr_resolution,
                                hr_unique.shape[0],
                                max_num_tokens,
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
                num_hr_tokens = int(hr_unique.shape[0])
                dense_hr_tokens = bool(
                    (self.vram_gb is None or self.vram_gb < 32.0) and num_hr_tokens > 49_152
                )
                hr_flow_headroom_gb = floor + (4.5 if res > 1024 else 2.0)
                _emit("sampling_shape_hr")
                t_st = time.monotonic()
                if mgr is not None:
                    if dense_hr_tokens:
                        mgr.evict_except(["shape_slat_flow_model_1024"], clear_cache=True)
                    mgr.ensure_on_gpu(
                        ["shape_slat_flow_model_1024"],
                        pipe.device,
                        required_free_gb=hr_flow_headroom_gb,
                        protected=[] if dense_hr_tokens else ["tex_slat_flow_model_1024"],
                    )
                    if not dense_hr_tokens:
                        mgr.prefetch_async(
                            ["tex_slat_flow_model_1024"],
                            pipe.device,
                            min_free_after_gb=hr_flow_headroom_gb + 0.5,
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
                    if dense_hr_tokens:
                        mgr.evict_except(["tex_slat_flow_model_1024"], clear_cache=True)
                    mgr.ensure_on_gpu(
                        ["tex_slat_flow_model_1024"],
                        pipe.device,
                        required_free_gb=hr_flow_headroom_gb,
                        protected=["shape_slat_decoder", "tex_slat_decoder"],
                    )
                    mgr.prefetch_async(
                        ["shape_slat_decoder", "tex_slat_decoder"],
                        pipe.device,
                        min_free_after_gb=hr_flow_headroom_gb + 0.5,
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

            # On <=32 GiB GPUs (e.g., 24 GiB RTX 4090), all DiT flow models are finished for this job;
            # evicting them before 3D sparse VAE decoding frees ~10-14 GiB of VRAM for FlexGEMM/C2S at 1024^3-1536^3.
            tight_gpu = bool(self.vram_gb is None or self.vram_gb < 32.0)
            decode_headroom_gb = floor + (7.5 if res > 1024 else 4.5)
            if hasattr(tex_slat, "clear_spatial_cache"):
                tex_slat.clear_spatial_cache()
            free_before_shape = _cuda_free_gb(torch)
            spill_tex_slat = bool(
                mgr is None
                or tight_gpu
                or res > 1024
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
                if tight_gpu or res > 1024:
                    mgr.evict_except(["shape_slat_decoder"], clear_cache=True)
                    mgr.ensure_on_gpu(
                        ["shape_slat_decoder"],
                        pipe.device,
                        required_free_gb=decode_headroom_gb,
                    )
                else:
                    mgr.ensure_on_gpu(
                        ["shape_slat_decoder"],
                        pipe.device,
                        required_free_gb=decode_headroom_gb,
                        protected=["tex_slat_decoder"],
                    )
                self._configure_decoder_residency(shape_dec_mod)
                if not tight_gpu and res <= 1024:
                    mgr.prefetch_async(
                        ["tex_slat_decoder"], pipe.device, min_free_after_gb=decode_headroom_gb
                    )
            free_pre_decode = _cuda_free_gb(torch)
            if free_pre_decode is not None:
                log.info(
                    "Starting decode_shape_slat(%d) (free VRAM: %.2f GiB, streaming=%s)",
                    res,
                    free_pre_decode,
                    bool(getattr(shape_dec_mod, "low_vram", False)),
                )
            meshes, subs = pipe.decode_shape_slat(shape_slat, res)
            _flush_flex_gemm_autotune_cache()
            del shape_slat
            if mgr is None:
                self._release_vram(offload_models=True)
            _log_stage(f"decode_shape_slat({res})", t_st)

            m = meshes[0]
            del meshes
            max_rast_faces = 16_777_216
            if (
                getattr(getattr(m, "vertices", None), "is_cuda", False)
                and getattr(getattr(m, "faces", None), "is_cuda", False)
            ):
                orig_v_cnt = int(m.vertices.shape[0])
                m.vertices, m.faces = _compact_mesh_vertices(m.vertices, m.faces, torch)
                if int(m.vertices.shape[0]) < orig_v_cnt:
                    log.info(
                        "Compacted mesh vertices before CuMesh cleanup: %d -> %d (faces=%d)",
                        orig_v_cnt,
                        int(m.vertices.shape[0]),
                        int(m.faces.shape[0]),
                    )

            is_oversized_mesh = int(getattr(getattr(m, "faces", None), "shape", (0,))[0]) > max_rast_faces
            tex_dec = pipe.models.get("tex_slat_decoder")
            defer_tex_load = bool(tight_gpu or res > 1024 or is_oversized_mesh)
            if mgr is not None:
                if defer_tex_load:
                    # Free ALL neural models & PyTorch cached blocks before raw-cudaMalloc CuMesh calls
                    mgr.evict_except([], clear_cache=True)
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    # Stage tex_slat_decoder before deciding whether to overlap CuMesh.fill_holes()
                    mgr.ensure_on_gpu(
                        ["tex_slat_decoder"],
                        pipe.device,
                        required_free_gb=floor + 3.5,
                    )
                    self._configure_decoder_residency(tex_dec)
            elif tex_dec is not None:
                tex_dec.low_vram = True

            free_after_shape = _cuda_free_gb(torch)
            overlap_min_free = floor + (8.0 if res > 1024 else 5.0)
            can_overlap_fill = bool(
                mgr is not None
                and not defer_tex_load
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

            def _run_fill_holes_safe(mesh_obj: Any) -> None:
                try:
                    mesh_obj.fill_holes()
                except Exception as fh_exc:
                    if not _is_oom_like(fh_exc):
                        raise
                    if fh_exc.__traceback__ is not None:
                        traceback.clear_frames(fh_exc.__traceback__)
                    _clear_sticky_cuda_error(torch)
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    log.warning(
                        "Deferring CuMesh.fill_holes to postprocess.to_glb after VRAM OOM (%s)",
                        fh_exc,
                    )

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
                if is_oversized_mesh:
                    # Skip OOM-prone fill_holes on >16.77M-face raw dual-grid meshes; simplify to the
                    # nvdiffrast cap first so fill_holes runs fast on <=16.77M faces with ~20+ GiB free.
                    log.info(
                        "Simplifying oversized raw mesh (%d > %d faces) before CuMesh.fill_holes",
                        int(m.faces.shape[0]),
                        max_rast_faces,
                    )
                    m.simplify(max_rast_faces)
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                _run_fill_holes_safe(m)
                free_after_shape = _cuda_free_gb(torch)
                spill_mesh_geom = bool(
                    mgr is None
                    or defer_tex_load
                    or free_after_shape is None
                    or free_after_shape < (floor + 5.0)
                )
                verts_staged = m.vertices.cpu() if spill_mesh_geom else m.vertices
                faces_staged = m.faces.cpu() if spill_mesh_geom else m.faces
                del m
                if mgr is None or spill_mesh_geom:
                    self._release_vram(offload_models=False)
                if defer_tex_load and mgr is not None:
                    mgr.ensure_on_gpu(
                        ["tex_slat_decoder"],
                        pipe.device,
                        required_free_gb=floor + (6.0 if res > 1024 else 3.5),
                    )
                    self._configure_decoder_residency(tex_dec)

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
            stage_bake_cpu = os.environ.get("TRELLIS2_STAGE_BAKE_VOLUME_CPU", "1") != "0"
            spill_voxels = bool(
                (res > 1024 and stage_bake_cpu) or (free_gb is not None and free_gb < (floor + 1.5))
            )
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
                    run_kwargs: dict[str, Any] = {
                        "seed": params.seed,
                        "preprocess_image": params.preprocess_image,
                        "pipeline_type": pipeline_type,
                    }
                    try:
                        if "max_num_tokens" in inspect.signature(self._pipeline.run).parameters:
                            run_kwargs["max_num_tokens"] = int(
                                os.environ.get("TRELLIS2_MAX_NUM_TOKENS", str(DEFAULT_MAX_NUM_TOKENS))
                            )
                    except (TypeError, ValueError):
                        pass
                    mesh = self._pipeline.run(image, **run_kwargs)[0]

                # Free neural models before CuMesh/cuBVH/nvdiffrast high-res baking (no models needed after decode)
                if self._model_mgr is not None:
                    if (
                        self.vram_gb is None
                        or self.vram_gb < 32.0
                        or params.texture_size >= 4096
                        or params.decimation_target >= 1_000_000
                    ):
                        self._model_mgr.evict_except([], clear_cache=True)
                    else:
                        self._release_vram(min_free_gb=VRAM_SAFETY_FLOOR_GB + 2.5)
                else:
                    self._release_vram(offload_models=True)

                _emit("simplifying_mesh")
                mesh.simplify(16777216)  # exact upstream nvdiffrast hardware face limit (no-op when <=16.77M)

                _emit("baking_textures")
                t_bake = time.monotonic()
                use_cuda = bool(self._torch is not None and self._torch.cuda.is_available())
                postprocess_patched = bool(
                    getattr(getattr(self._o_voxel, "postprocess", None), "_trellis2_postprocess_patched", False)
                )
                stage_bake_cpu = bool(
                    use_cuda
                    and postprocess_patched
                    and (self.low_vram or self.vram_gb is None or self.vram_gb < 32.0)
                    and os.environ.get("TRELLIS2_STAGE_BAKE_VOLUME_CPU", "1") != "0"
                )
                bake_verts = mesh.vertices
                bake_faces = mesh.faces
                bake_layout = mesh.layout
                bake_voxel_size = mesh.voxel_size
                if stage_bake_cpu:
                    attr_vol = mesh.attrs.cpu() if hasattr(mesh.attrs, "cpu") else mesh.attrs
                    vox_coords = mesh.coords.cpu() if hasattr(mesh.coords, "cpu") else mesh.coords
                else:
                    attr_vol = mesh.attrs.cuda() if use_cuda and hasattr(mesh.attrs, "cuda") else mesh.attrs
                    vox_coords = mesh.coords.cuda() if use_cuda and hasattr(mesh.coords, "cuda") else mesh.coords
                # Drop wrapper references before to_glb so intermediate CuMesh steps don't pin duplicate tensors
                mesh = None
                if stage_bake_cpu and self._torch is not None and self._torch.cuda.is_available():
                    gc.collect()
                    self._torch.cuda.empty_cache()

                glb = self._o_voxel.postprocess.to_glb(
                    vertices=bake_verts,
                    faces=bake_faces,
                    attr_volume=attr_vol,
                    coords=vox_coords,
                    attr_layout=bake_layout,
                    voxel_size=bake_voxel_size,
                    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                    decimation_target=params.decimation_target,
                    texture_size=params.texture_size,
                    remesh=params.remesh,
                    remesh_band=params.remesh_band,
                    remesh_project=params.remesh_project,
                    verbose=True,
                )
                del bake_verts, bake_faces, attr_vol, vox_coords
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
                _flush_flex_gemm_autotune_cache()
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
                retry_next: str | None = None
                oom_detail = ""
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
                        tb_loc = ""
                        if exc.__traceback__ is not None:
                            tb_frames = traceback.extract_tb(exc.__traceback__)
                            if tb_frames:
                                last_f = tb_frames[-1]
                                tb_loc = f" at {Path(last_f.filename).name}:{last_f.lineno} in {last_f.name}"
                            traceback.clear_frames(exc.__traceback__)
                        flat_msg = " ".join(s.strip() for s in str(exc).splitlines() if s.strip())
                        oom_detail = f"{flat_msg[:260]}{tb_loc}"
                        retry_next = next_type
                    else:
                        raise
                if retry_next is not None:
                    log.warning(
                        "CUDA OOM on pipeline_type=%s (%s); clearing VRAM and retrying with %s.",
                        current_type,
                        oom_detail,
                        retry_next,
                    )
                    if on_stage is not None:
                        on_stage(f"retrying_{retry_next}_after_oom")
                    self._release_vram(offload_models=True)
                    free_after = _cuda_free_gb(self._torch)
                    if free_after is not None:
                        log.info("VRAM recovered after OOM cleanup: %.2f GiB free", free_after)
                    current_type = retry_next


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
        if not self._jobs_dir.is_dir():
            return None
        try:
            job_dir: Path | None = None
            for entry in self._jobs_dir.iterdir():
                if entry.name == job_id and entry.is_dir():
                    job_dir = entry
                    break
            if job_dir is None:
                return None
            status_file = job_dir / "status.json"
            if not status_file.is_file():
                return None
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
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
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
