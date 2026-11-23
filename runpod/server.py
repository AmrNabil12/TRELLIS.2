"""FastAPI inference server for TRELLIS.2 on RunPod.

Exposes asynchronous job submission (recommended behind RunPod's ~100s HTTP
proxy timeout) as well as a synchronous convenience endpoint and a /health check.
Torch and TRELLIS.2 imports are deferred so the API layer can be imported and
tested without CUDA installed.
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc
import io
import json
import logging
import re
import secrets
import shutil
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

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

        self._pipeline: Any = None
        self._o_voxel: Any = None
        self._torch: Any = None
        self._gpu_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._load_thread: threading.Thread | None = None

    def start_loading(self) -> None:
        """Load weights on a background thread so /health responds immediately."""
        with self._state_lock:
            if self._load_thread is not None and self._load_thread.is_alive():
                return
            self.status = "loading"
            self.stage = "starting"
            self.error = None
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
            self.stage = "moving_to_cuda"
            pipeline.cuda()

            with self._state_lock:
                self._pipeline = pipeline
                self.status = "ready"
                self.stage = "ready"
                self.error = None
            log.info("TRELLIS.2 pipeline ready (%s)", self.model_id)
        except Exception as exc:
            log.exception("Failed to load TRELLIS.2 pipeline")
            with self._state_lock:
                self.status = "error"
                self.stage = "failed"
                self.error = f"{type(exc).__name__}: {exc}"

    def health_info(self) -> dict[str, Any]:
        with self._state_lock:
            return {
                "status": self.status,
                "stage": self.stage,
                "model_id": self.model_id,
                "gpu": self.gpu_name,
                "vram_gb": self.vram_gb,
                "low_vram": self.low_vram,
                "error": self.error,
            }

    def _release_vram(self, offload_models: bool = False) -> None:
        """Free unused neural sub-models, Python garbage, and PyTorch CUDA cache."""
        if offload_models and self.low_vram and self._pipeline is not None:
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

    def _run_once(
        self,
        image: Image.Image,
        params: GenerationParams,
        out_path: Path,
        pipeline_type: str,
    ) -> Path:
        mesh = None
        glb = None
        infer_ctx = (
            self._torch.inference_mode()
            if self._torch is not None and hasattr(self._torch, "inference_mode")
            else nullcontext()
        )
        try:
            with infer_ctx:
                log.info(
                    "Running TRELLIS.2 pipeline_type=%s seed=%d decimation=%d texture=%d remesh=%s",
                    pipeline_type,
                    params.seed,
                    params.decimation_target,
                    params.texture_size,
                    params.remesh,
                )
                mesh = self._pipeline.run(
                    image,
                    seed=params.seed,
                    preprocess_image=params.preprocess_image,
                    pipeline_type=pipeline_type,
                )[0]
                # Free diffusion/encoder models & intermediate activations before mesh/texture baking
                self._release_vram(offload_models=True)

                mesh.simplify(16777216)  # nvdiffrast face limit
                glb = self._o_voxel.postprocess.to_glb(
                    vertices=mesh.vertices,
                    faces=mesh.faces,
                    attr_volume=mesh.attrs,
                    coords=mesh.coords,
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
                mesh = None
                self._release_vram(offload_models=False)

                glb.export(str(out_path), extension_webp=params.extension_webp)
                return out_path
        finally:
            del mesh, glb
            self._release_vram(offload_models=True)

    def generate_glb(self, image: Image.Image, params: GenerationParams, out_path: Path) -> Path:
        if self.status != "ready" or self._pipeline is None:
            raise RuntimeError(f"Model is not ready (status={self.status}, error={self.error})")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with self._gpu_lock:
            try:
                return self._run_once(image, params, out_path, params.pipeline_type)
            except Exception as exc:
                allow_fallback = os.environ.get("TRELLIS2_OOM_FALLBACK", "1") != "0"
                if (
                    self._is_cuda_oom(exc)
                    and self.low_vram
                    and params.pipeline_type == "1024_cascade"
                    and allow_fallback
                ):
                    log.warning(
                        "CUDA OOM on pipeline_type=1024_cascade; clearing VRAM and retrying with 1024."
                    )
                    self._release_vram(offload_models=True)
                    return self._run_once(image, params, out_path, "1024")
                raise


@dataclass
class JobRecord:
    job_id: str
    status: str  # queued | running | completed | failed
    stage: str
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    glb_path: Path | None = None
    error: str | None = None

    def elapsed_s(self) -> float:
        end = self.finished_at if self.finished_at is not None else time.time()
        start = self.started_at if self.started_at is not None else self.created_at
        return round(max(0.0, end - start), 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "stage": self.stage,
            "elapsed_s": self.elapsed_s(),
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
            created_at=time.time(),
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
                    self._write_status(record)
                last_stage = current_stage
            self._stop_event.wait(0.05)

        if self._engine.status == "error":
            with self._lock:
                record = self._jobs.get(job_id)
                if record is None:
                    return
                record.status = "failed"
                record.stage = "failed"
                record.error = f"TRELLIS.2 failed to initialize: {self._engine.error}"
                record.finished_at = time.time()
                self._write_status(record)
            return

        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return
            record.status = "running"
            record.stage = "generating"
            record.started_at = time.time()
            self._write_status(record)

        # Must end in .glb because trimesh.export() infers format from the file extension
        tmp_glb = out_path.parent / ".mesh.tmp.glb"
        try:
            self._engine.generate_glb(image, params, tmp_glb)
            _atomic_replace(tmp_glb, out_path)
            with self._lock:
                record.status = "completed"
                record.stage = "completed"
                record.finished_at = time.time()
                self._write_status(record)
        except Exception as exc:
            tmp_glb.unlink(missing_ok=True)
            log.exception("Job %s failed", job_id)
            with self._lock:
                record.status = "failed"
                record.stage = "failed"
                record.error = f"{type(exc).__name__}: {exc}"
                record.finished_at = time.time()
                self._write_status(record)

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
        self._executor.shutdown(wait=False)


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
            return img.convert("RGBA" if has_alpha else "RGB")
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
        return resolved_engine.health_info()

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
