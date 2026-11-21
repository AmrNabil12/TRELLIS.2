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
import logging
import secrets
import shutil
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Response, UploadFile, status
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

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

    def generate_glb(self, image: Image.Image, params: GenerationParams, out_path: Path) -> Path:
        if self.status != "ready" or self._pipeline is None:
            raise RuntimeError(f"Model is not ready (status={self.status}, error={self.error})")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        mesh = None
        glb = None
        with self._gpu_lock:
            try:
                log.info(
                    "Running TRELLIS.2 pipeline_type=%s seed=%d decimation=%d texture=%d remesh=%s",
                    params.pipeline_type,
                    params.seed,
                    params.decimation_target,
                    params.texture_size,
                    params.remesh,
                )
                mesh = self._pipeline.run(
                    image,
                    seed=params.seed,
                    preprocess_image=params.preprocess_image,
                    pipeline_type=params.pipeline_type,
                )[0]
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
                glb.export(str(out_path), extension_webp=params.extension_webp)
                return out_path
            finally:
                del mesh, glb
                gc.collect()
                if self._torch is not None and self._torch.cuda.is_available():
                    self._torch.cuda.empty_cache()


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
        end = self.finished_at if self.finished_at is not None else time.monotonic()
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


class JobManager:
    """Single-worker async job queue with bounded on-disk GLB retention."""

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
        self._job_order: list[str] = []
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="trellis2-job")

    def submit(self, image: Image.Image, params: GenerationParams) -> JobRecord:
        job_id = uuid.uuid4().hex[:12]
        job_dir = self._jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        out_path = job_dir / "mesh.glb"

        record = JobRecord(
            job_id=job_id,
            status="queued",
            stage="queued",
            created_at=time.monotonic(),
            glb_path=out_path,
        )
        with self._lock:
            self._jobs[job_id] = record
            self._job_order.append(job_id)
            self._prune_locked()

        self._executor.submit(self._run_job, job_id, image, params, out_path)
        return record

    def _run_job(
        self, job_id: str, image: Image.Image, params: GenerationParams, out_path: Path
    ) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return
            record.status = "running"
            record.stage = "generating"
            record.started_at = time.monotonic()

        try:
            self._engine.generate_glb(image, params, out_path)
            with self._lock:
                record.status = "completed"
                record.stage = "completed"
                record.finished_at = time.monotonic()
        except Exception as exc:
            log.exception("Job %s failed", job_id)
            with self._lock:
                record.status = "failed"
                record.stage = "failed"
                record.error = f"{type(exc).__name__}: {exc}"
                record.finished_at = time.monotonic()

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def _prune_locked(self) -> None:
        while len(self._job_order) > self._max_retained_jobs:
            oldest_id = self._job_order[0]
            oldest = self._jobs.get(oldest_id)
            # Never prune a job currently queued or running
            if oldest is not None and oldest.status in {"queued", "running"}:
                break
            self._job_order.pop(0)
            self._jobs.pop(oldest_id, None)
            if oldest is not None and oldest.glb_path is not None:
                shutil.rmtree(oldest.glb_path.parent, ignore_errors=True)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)


def _default_jobs_dir() -> Path:
    env_dir = os.environ.get("TRELLIS2_JOBS_DIR")
    if env_dir:
        return Path(env_dir)
    workspace = Path("/workspace")
    if workspace.is_dir() and os.access(workspace, os.W_OK):
        return workspace / "jobs"
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

    @fastapi_app.get("/health")
    def health() -> dict[str, Any]:
        return resolved_engine.health_info()

    @fastapi_app.post("/jobs", status_code=status.HTTP_202_ACCEPTED, dependencies=[Depends(verify_api_key)])
    async def create_job(
        image: UploadFile = File(...),
        params: GenerationParams = Depends(parse_generation_params),
    ) -> dict[str, Any]:
        _require_ready_engine(resolved_engine)
        pil_image = _decode_uploaded_image(await image.read())
        record = manager.submit(pil_image, params)
        return {
            "job_id": record.job_id,
            "status": record.status,
            "status_url": f"/jobs/{record.job_id}",
            "download_url": f"/jobs/{record.job_id}/glb",
        }

    @fastapi_app.get("/jobs/{job_id}", dependencies=[Depends(verify_api_key)])
    def get_job(job_id: str) -> dict[str, Any]:
        record = manager.get(job_id)
        if record is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown job: {job_id}")
        return record.to_dict()

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
