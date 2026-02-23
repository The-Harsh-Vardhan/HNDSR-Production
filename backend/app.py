"""
backend/app.py
================
Production FastAPI application for HNDSR super-resolution inference.

What  : Serves the HNDSR model via HTTP endpoints with health checks,
        Prometheus metrics, version info, and real model inference.
Why   : The Jupyter notebook cannot serve production traffic. This app
        provides async request handling, backpressure, and monitoring.
How   : FastAPI + HNDSRInferenceEngine with DDIM sampling, thread-pool
        GPU offloading, semaphore concurrency, and prometheus_client metrics.

Endpoints:
  GET  /health    — Liveness probe (Kubernetes)
  GET  /ready     — Readiness probe (model loaded?)
  POST /infer     — Super-resolution inference (real HNDSR pipeline)
  GET  /metrics   — Prometheus metrics (histogram-based)
  GET  /version   — API and model version info

Post-Audit Fixes Applied:
  1. Prometheus histogram metrics (P50/P95/P99)
  2. Rate limiter memory leak fixed
  3. Image dimension guard (16M pixels)
  4. GPU OOM handling with torch.cuda.empty_cache()
  5. Graceful shutdown (30s drain)
  6. CORS spec-compliant
  7. Thread-safe active_requests counter

Model Integration:
  - Real HNDSR 3-stage pipeline (Autoencoder + FNO + Diffusion UNet)
  - DDIM 50-step sampling
  - FP16 mixed precision
  - Singleton model loader with warm-start
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import threading
import time
import traceback
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import re

import torch
import torchvision.transforms.functional as TF
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

# Register AVIF / HEIF support (iPhones, modern browsers)
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    pillow_heif.register_avif_opener()
except ImportError:
    pass  # graceful degradation — only JPEG/PNG/WebP supported
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def _env_bool(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

class ServerConfig:
    """Server configuration from environment variables."""
    max_concurrent_inferences: int = int(os.getenv("MAX_CONCURRENT", "4"))
    request_timeout_s: float = float(os.getenv("REQUEST_TIMEOUT_S", "300"))
    max_payload_mb: float = float(os.getenv("MAX_PAYLOAD_MB", "20"))
    max_queue_depth: int = int(os.getenv("MAX_QUEUE_DEPTH", "20"))
    model_dir: str = os.getenv("MODEL_DIR", "./checkpoints")
    device: str = os.getenv("DEVICE", "auto")
    rate_limit_per_hour: int = int(os.getenv("RATE_LIMIT_PER_HOUR", "100"))
    max_image_pixels: int = int(os.getenv("MAX_IMAGE_PIXELS", "16000000"))
    use_fp16: bool = os.getenv("USE_FP16", "false").lower() == "true"
    ddim_steps: int = int(os.getenv("DDIM_STEPS", "10"))
    tile_size: int = int(os.getenv("TILE_SIZE", "128"))
    tile_overlap: int = int(os.getenv("TILE_OVERLAP", "16"))
    # Max longest-side dimension for input images (0 = no limit).
    # On CPU-only deployments, large inputs cause timeouts; server
    # will auto-downscale to this before inference.
    max_input_dim: int = int(os.getenv("MAX_INPUT_DIM", "512"))
    checkpoint_manifest_path: str = os.getenv("CHECKPOINT_MANIFEST_PATH", "./checkpoints/manifest.json")
    enforce_checkpoint_manifest: bool = _env_bool("ENFORCE_CHECKPOINT_MANIFEST", "true")
    allow_fallback_on_invalid_ckpt: bool = _env_bool("ALLOW_FALLBACK_ON_INVALID_CKPT", "true")
    enable_quality_probe: bool = _env_bool("ENABLE_QUALITY_PROBE", "true")
    quality_probe_ddim_steps: int = int(os.getenv("QUALITY_PROBE_DDIM_STEPS", "10"))
    quality_probe_input_size: int = int(os.getenv("QUALITY_PROBE_INPUT_SIZE", "64"))
    quality_probe_min_std: float = float(os.getenv("QUALITY_PROBE_MIN_STD", "0.05"))


CONFIG = ServerConfig()


# ─────────────────────────────────────────────────────────────────────────────
# Prometheus metrics
# ─────────────────────────────────────────────────────────────────────────────

REQUEST_COUNT = Counter("hndsr_requests_total", "Total inference requests", ["method", "endpoint", "status"])
ERROR_COUNT = Counter("hndsr_errors_total", "Total inference errors", ["error_type"])
INFERENCE_LATENCY = Histogram("hndsr_inference_seconds", "Inference latency in seconds",
                               buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0])
ACTIVE_REQUESTS = Gauge("hndsr_active_requests", "Currently active inference requests")
GPU_MEMORY_USED = Gauge("hndsr_gpu_memory_used_bytes", "GPU memory used in bytes")
UPTIME = Gauge("hndsr_uptime_seconds", "Server uptime in seconds")
RATE_LIMITED = Counter("hndsr_rate_limited_total", "Total requests rejected by rate limiter")
BACKPRESSURE_REJECTED = Counter("hndsr_backpressure_rejected_total", "Total requests rejected by backpressure")
FALLBACK_INFERENCES = Counter("hndsr_fallback_inferences_total", "Total bicubic fallback inferences")
CKPT_VALIDATION_FAILURES = Counter(
    "hndsr_checkpoint_validation_failures_total",
    "Total checkpoint manifest/quality validation failures",
)


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────────────────────────────────────────

class InferenceRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded input image (PNG/JPEG)")
    scale_factor: int = Field(default=4, ge=2, le=8, description="Upscaling factor")
    ddim_steps: Optional[int] = Field(default=None, ge=10, le=200, description="DDIM sampling steps (default: server config)")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    return_metadata: bool = Field(default=True)

    @field_validator("scale_factor", mode="before")
    @classmethod
    def normalize_scale_factor(cls, value):
        if value is None:
            return 4
        if isinstance(value, str) and value.strip() == "":
            return 4
        return value

    @field_validator("ddim_steps", mode="before")
    @classmethod
    def normalize_ddim_steps(cls, value):
        if value is None:
            return None
        if isinstance(value, str) and value.strip() == "":
            return None
        return value


class InferenceResponse(BaseModel):
    image: str = Field(..., description="Base64-encoded output image")
    width: int
    height: int
    scale_factor: int
    metadata: Optional[dict] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    gpu_available: bool
    gpu_name: Optional[str] = None
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    uptime_seconds: float
    active_requests: int
    inference_engine: str = "HNDSRInferenceEngine"
    inference_mode: str = "hndsr"
    checkpoint_validated: bool = False


class VersionResponse(BaseModel):
    api_version: str = "1.0.0"
    model_version: str = "unknown"
    python_version: str
    torch_version: str
    cuda_available: bool
    cuda_version: Optional[str] = None
    model_architecture: str = "HNDSR (Autoencoder + FNO + Diffusion UNet)"
    ddim_steps: int = 50
    checkpoint_hashes: dict[str, str] = Field(default_factory=dict)
    checkpoint_manifest_match: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Application State
# ─────────────────────────────────────────────────────────────────────────────

class AppState:
    """Mutable application state with thread-safe counters."""

    def __init__(self):
        self.model_loaded: bool = False
        self.start_time: float = time.time()
        self.engine = None  # HNDSRInferenceEngine
        self.tile_processor = None  # SatelliteTileProcessor
        self.device = None
        self.inference_mode: str = "hndsr"
        self.checkpoint_validated: bool = False
        self.checkpoint_manifest_match: bool = False
        self.fallback_reason: Optional[str] = None
        self.checkpoint_hashes: dict[str, str] = {}

        self._active_requests: int = 0
        self._active_lock: threading.Lock = threading.Lock()

        self.request_counts: dict = {}
        self._request_counter: int = 0

    @property
    def active_requests(self) -> int:
        with self._active_lock:
            return self._active_requests

    def increment_active(self) -> int:
        with self._active_lock:
            self._active_requests += 1
            ACTIVE_REQUESTS.set(self._active_requests)
            return self._active_requests

    def decrement_active(self) -> int:
        with self._active_lock:
            self._active_requests -= 1
            ACTIVE_REQUESTS.set(self._active_requests)
            return self._active_requests

    def check_rate_limit(self, client_ip: str) -> bool:
        now = time.time()
        current_hour = int(now // 3600)
        hour_key = f"{client_ip}:{current_hour}"
        self.request_counts[hour_key] = self.request_counts.get(hour_key, 0) + 1

        self._request_counter += 1
        if self._request_counter % 100 == 0:
            stale_keys = [k for k in self.request_counts if not k.endswith(f":{current_hour}")]
            for k in stale_keys:
                del self.request_counts[k]

        return self.request_counts[hour_key] <= CONFIG.rate_limit_per_hour


state: AppState = None


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan (startup / shutdown)
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load real HNDSR model, warm up GPU, graceful shutdown."""
    global state
    state = AppState()

    logger.info("=" * 60)
    logger.info("Starting HNDSR API with REAL model pipeline")
    logger.info("=" * 60)

    # Resolve device
    from backend.inference.model_loader import resolve_device, get_model_loader
    from backend.inference.engine import HNDSRInferenceEngine
    from backend.inference.tile_processor import SatelliteTileProcessor
    from backend.inference.quality_probe import run_quality_probe

    device = resolve_device(CONFIG.device)
    state.device = device

    # Initialize model loader and checkpoint manifest validation
    loader = get_model_loader()
    model_dir = Path(CONFIG.model_dir)
    manifest_path = Path(CONFIG.checkpoint_manifest_path)

    try:
        loader.initialize(
            model_dir=model_dir,
            autoencoder_ckpt="autoencoder_best.pth",
            neural_operator_ckpt="neural_operator_best.pth",
            diffusion_ckpt="diffusion_best.pth",
            device=device,
            use_fp16=CONFIG.use_fp16,
            manifest_path=manifest_path,
        )
    except Exception as exc:
        logger.error("ModelLoader initialization failed: %s", exc)
        raise SystemExit(1)

    manifest_ok = loader.validate_checkpoint_manifest()
    state.checkpoint_manifest_match = manifest_ok
    state.checkpoint_hashes = dict(loader.checkpoint_hashes)

    if CONFIG.enforce_checkpoint_manifest and not manifest_ok:
        CKPT_VALIDATION_FAILURES.inc()
        manifest_err = loader.manifest_validation_error or "unknown manifest validation error"
        if CONFIG.allow_fallback_on_invalid_ckpt:
            state.inference_mode = "bicubic_fallback"
            state.fallback_reason = f"checkpoint_manifest_validation_failed: {manifest_err}"
            logger.warning("Switching to fallback mode: %s", state.fallback_reason)
        else:
            logger.error("Checkpoint manifest validation failed and fallback is disabled.")
            raise SystemExit(1)

    if state.inference_mode == "hndsr":
        try:
            loader.warm_start()
        except FileNotFoundError as exc:
            CKPT_VALIDATION_FAILURES.inc()
            if CONFIG.allow_fallback_on_invalid_ckpt:
                state.inference_mode = "bicubic_fallback"
                state.fallback_reason = f"model_load_failed: {exc}"
                logger.warning("Switching to fallback mode: %s", state.fallback_reason)
            else:
                logger.error("=" * 60)
                logger.error("CHECKPOINT FILES NOT FOUND!")
                logger.error("Run: python -m backend.inference.generate_checkpoints")
                logger.error("Error: %s", exc)
                logger.error("=" * 60)
                raise SystemExit(1)

    if state.inference_mode == "hndsr":
        # Create inference engine (uses composite HNDSR model)
        engine = HNDSRInferenceEngine(
            model=loader.model,
            device=device,
            ddim_steps=CONFIG.ddim_steps,
            max_concurrent=CONFIG.max_concurrent_inferences,
        )

        # Warmup
        engine.warmup(tile_size=64, n_runs=2)

        # Create tile processor
        tile_proc = SatelliteTileProcessor(
            inference_engine=engine,
            tile_size=CONFIG.tile_size,
            overlap=CONFIG.tile_overlap,
            batch_size=1,
            max_pixels=CONFIG.max_image_pixels,
        )

        state.engine = engine
        state.tile_processor = tile_proc

        probe_passed = True
        if CONFIG.enable_quality_probe:
            probe = run_quality_probe(
                model=loader.model,
                device=device,
                scale_factor=2,
                ddim_steps=CONFIG.quality_probe_ddim_steps,
                input_size=CONFIG.quality_probe_input_size,
                min_std=CONFIG.quality_probe_min_std,
            )
            logger.info("Startup quality probe: %s", probe.as_dict())
            probe_passed = probe.passed

            if not probe_passed:
                CKPT_VALIDATION_FAILURES.inc()
                if CONFIG.allow_fallback_on_invalid_ckpt:
                    state.inference_mode = "bicubic_fallback"
                    state.fallback_reason = f"quality_probe_failed: {probe.reason}"
                    state.engine = None
                    state.tile_processor = None
                    loader.unload_all()
                    logger.warning("Switching to fallback mode: %s", state.fallback_reason)
                else:
                    logger.error("Quality probe failed and fallback is disabled: %s", probe.reason)
                    raise SystemExit(1)

        state.checkpoint_validated = bool(
            state.checkpoint_manifest_match and probe_passed and state.inference_mode == "hndsr"
        )
    else:
        state.checkpoint_validated = False

    state.model_loaded = True
    state.start_time = time.time()

    logger.info("=" * 60)
    logger.info(
        "HNDSR API ready! mode=%s device=%s DDIM=%d FP16=%s manifest_match=%s validated=%s",
        state.inference_mode,
        device,
        CONFIG.ddim_steps,
        CONFIG.use_fp16,
        state.checkpoint_manifest_match,
        state.checkpoint_validated,
    )
    if state.inference_mode == "bicubic_fallback":
        logger.warning("Fallback reason: %s", state.fallback_reason)
    if state.checkpoint_hashes:
        logger.info("Checkpoint hashes: %s", state.checkpoint_hashes)
    logger.info("=" * 60)
    yield

    # Graceful shutdown
    logger.info("Shutting down HNDSR API...")
    state.model_loaded = False

    drain_timeout = 30.0
    drain_start = time.time()
    while state.active_requests > 0 and (time.time() - drain_start) < drain_timeout:
        logger.info("Draining %d active requests...", state.active_requests)
        await asyncio.sleep(1.0)

    loader.unload_all()
    logger.info("Shutdown complete.")


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="HNDSR Super-Resolution API",
    description="Production API for Hybrid Neural Operator-Diffusion Super-Resolution",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def request_validation_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    logger.warning(
        "Request validation failed: %s %s (%d error(s))",
        request.method,
        request.url.path,
        len(errors),
    )
    if errors:
        first = errors[0]
        logger.warning(
            "First validation error loc=%s msg=%s",
            first.get("loc"),
            first.get("msg"),
        )
    return JSONResponse(
        status_code=422,
        content={
            "message": "Request validation failed. Check payload fields and types.",
            "detail": errors,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """Liveness probe."""
    gpu_name, gpu_mem_used, gpu_mem_total = None, None, None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info(0)
        gpu_mem_used = (total - free) / 1e6
        gpu_mem_total = total / 1e6
        GPU_MEMORY_USED.set(total - free)

    UPTIME.set(time.time() - state.start_time)

    return HealthResponse(
        status="healthy" if state.model_loaded else "loading",
        model_loaded=state.model_loaded,
        gpu_available=torch.cuda.is_available(),
        gpu_name=gpu_name,
        gpu_memory_used_mb=gpu_mem_used,
        gpu_memory_total_mb=gpu_mem_total,
        uptime_seconds=time.time() - state.start_time,
        active_requests=state.active_requests,
        inference_mode=state.inference_mode,
        checkpoint_validated=state.checkpoint_validated,
    )


@app.get("/ready")
async def ready():
    """Readiness probe."""
    if not state.model_loaded:
        raise HTTPException(503, detail="Model not loaded")
    if state.active_requests >= CONFIG.max_queue_depth:
        raise HTTPException(503, detail="Server overloaded", headers={"Retry-After": "5"})
    return {"status": "ready", "engine": "HNDSRInferenceEngine"}


@app.post("/infer", response_model=InferenceResponse)
async def infer(request: InferenceRequest, req: Request):
    """
    Real HNDSR super-resolution inference.

    Pipeline:
      1. Decode base64 → PIL Image → tensor [-1, 1]
      2. Neural Operator: LR → structural prior (FNO)
      3. DDIM reverse diffusion: 50 steps, noise → latent
      4. Autoencoder decoder: latent → HR image
      5. Encode tensor → base64

    NOT a placeholder. Uses real HNDSR model stages on GPU.
    """
    # Backpressure
    if state.active_requests >= CONFIG.max_queue_depth:
        BACKPRESSURE_REJECTED.inc()
        raise HTTPException(503, detail="Server overloaded", headers={"Retry-After": "5"})

    # Rate limiting
    client_ip = req.client.host if req.client else "unknown"
    if not state.check_rate_limit(client_ip):
        RATE_LIMITED.inc()
        raise HTTPException(429, detail="Rate limit exceeded")

    # Payload size
    payload_mb = len(request.image) * 3 / 4 / 1e6
    if payload_mb > CONFIG.max_payload_mb:
        raise HTTPException(413, detail=f"Payload {payload_mb:.1f} MB exceeds limit {CONFIG.max_payload_mb} MB")

    # Decode image
    try:
        raw_b64 = request.image
        # Strip data-URL prefix if the client accidentally left it
        if raw_b64.startswith("data:"):
            raw_b64 = raw_b64.split(",", 1)[-1]
        # Fix missing base64 padding
        raw_b64 = raw_b64.strip()
        padding = 4 - len(raw_b64) % 4
        if padding != 4:
            raw_b64 += "=" * padding
        img_bytes = base64.b64decode(raw_b64, validate=True)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        w, h = img.size
    except base64.binascii.Error as exc:
        ERROR_COUNT.labels(error_type="decode").inc()
        raise HTTPException(422, detail=f"Invalid base64 encoding: {exc}")
    except Image.UnidentifiedImageError:
        ERROR_COUNT.labels(error_type="decode").inc()
        raise HTTPException(
            422,
            detail="Unsupported image format. Please upload a JPEG or PNG file.",
        )
    except Exception as exc:
        ERROR_COUNT.labels(error_type="decode").inc()
        raise HTTPException(422, detail=f"Could not decode image: {exc}")

    # Dimension guard
    total_pixels = w * h
    if total_pixels > CONFIG.max_image_pixels:
        raise HTTPException(413, detail=f"Image too large: {w}x{h} = {total_pixels:,} px. Max: {CONFIG.max_image_pixels:,}")

    # Auto-downscale large inputs for CPU-only deployments to avoid timeouts
    max_dim = CONFIG.max_input_dim
    if max_dim > 0 and max(w, h) > max_dim:
        ratio = max_dim / max(w, h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        logger.info("Auto-downscaling input %dx%d → %dx%d (max_input_dim=%d)",
                    w, h, new_w, new_h, max_dim)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        w, h = img.size

    # Convert to tensor [-1, 1]
    lr_tensor = TF.to_tensor(img) * 2.0 - 1.0  # (3, H, W) in [-1, 1]

    # Run inference
    state.increment_active()
    start_time = time.perf_counter()
    output_img = None

    try:
        if state.inference_mode == "bicubic_fallback":
            target_w = w * request.scale_factor
            target_h = h * request.scale_factor
            output_img = img.resize((target_w, target_h), Image.BICUBIC)
            FALLBACK_INFERENCES.inc()
        else:
            async with asyncio.timeout(CONFIG.request_timeout_s):
                # Run real model inference in thread pool
                loop = asyncio.get_event_loop()
                hr_tensor = await loop.run_in_executor(
                    None,
                    _run_real_inference,
                    lr_tensor,
                    request.scale_factor,
                    request.ddim_steps,
                    request.seed,
                )
            # Encode output tensor -> PIL image
            hr_tensor_cpu = ((hr_tensor.clamp(-1, 1) + 1.0) / 2.0).cpu()
            output_img = TF.to_pil_image(hr_tensor_cpu)

    except asyncio.TimeoutError:
        ERROR_COUNT.labels(error_type="timeout").inc()
        raise HTTPException(504, detail=f"Inference timeout ({CONFIG.request_timeout_s}s)")
    except HTTPException:
        raise
    except torch.cuda.OutOfMemoryError:
        ERROR_COUNT.labels(error_type="gpu_oom").inc()
        logger.error("GPU OOM during inference for %dx%d image", w, h)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise HTTPException(503, detail="GPU out of memory. Try smaller image.", headers={"Retry-After": "10"})
    except Exception as exc:
        ERROR_COUNT.labels(error_type="internal").inc()
        logger.error("Inference error: %s", traceback.format_exc())
        raise HTTPException(500, detail=f"Inference failed: {exc}")
    finally:
        state.decrement_active()

    latency_s = time.perf_counter() - start_time
    INFERENCE_LATENCY.observe(latency_s)
    REQUEST_COUNT.labels(method="POST", endpoint="/infer", status="200").inc()

    if output_img is None:
        ERROR_COUNT.labels(error_type="internal").inc()
        raise HTTPException(500, detail="Inference produced no output")

    buf = io.BytesIO()
    output_img.save(buf, format="PNG")
    output_b64 = base64.b64encode(buf.getvalue()).decode()

    out_w, out_h = output_img.size

    metadata = None
    if request.return_metadata:
        metadata = {
            "input_size": f"{w}x{h}",
            "output_size": f"{out_w}x{out_h}",
            "scale_factor": request.scale_factor,
            "ddim_steps": request.ddim_steps or CONFIG.ddim_steps,
            "latency_ms": round(latency_s * 1000, 1),
            "device": str(state.device),
            "model": "HNDSR (Autoencoder + FNO + Diffusion UNet)",
            "fp16": CONFIG.use_fp16,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "inference_mode": state.inference_mode,
        }
        if state.inference_mode == "bicubic_fallback":
            metadata["fallback_reason"] = state.fallback_reason or "checkpoint_validation_failed"

    return InferenceResponse(
        image=output_b64,
        width=out_w,
        height=out_h,
        scale_factor=request.scale_factor,
        metadata=metadata,
    )


def _run_real_inference(
    lr_tensor: torch.Tensor,
    scale_factor: int,
    ddim_steps: Optional[int],
    seed: Optional[int],
) -> torch.Tensor:
    """
    Run REAL HNDSR inference synchronously (called from thread pool).

    Pipeline (inside HNDSR.super_resolve):
      1. Bicubic upscale LR → target size
      2. Neural Operator (FNO) → structural prior
      3. Resize prior to latent spatial dimensions
      4. ImplicitAmplification (scale-conditioned gain)
      5. GAP pool → 1-D context vector
      6. DDIM reverse diffusion
      7. Autoencoder decode latent → HR image

    For large images, uses the tile processor with Hann-window stitching
    to avoid GPU OOM.  The output is in [-1, 1] range, (3, H, W) tensor.
    """
    if seed is not None:
        torch.manual_seed(seed)

    _, h_in, w_in = lr_tensor.shape

    # Use tile processor for large images (prevents GPU OOM)
    tile_proc = state.tile_processor
    hr_tensor = tile_proc.process(lr_tensor, scale=float(scale_factor), seed=seed)

    # Crop back to exact target resolution (tile processor may pad the input)
    _, h_cur, w_cur = hr_tensor.shape
    target_h = h_in * scale_factor
    target_w = w_in * scale_factor

    if h_cur != target_h or w_cur != target_w:
        hr_tensor = hr_tensor[:, :target_h, :target_w]

    return hr_tensor


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        GPU_MEMORY_USED.set(total - free)
    UPTIME.set(time.time() - state.start_time)

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/version", response_model=VersionResponse)
async def version():
    """API and model version info."""
    import platform
    return VersionResponse(
        api_version="1.0.0",
        model_version=os.getenv("MODEL_VERSION", "1.0.0"),
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        cuda_version=torch.version.cuda if torch.cuda.is_available() else None,
        ddim_steps=CONFIG.ddim_steps,
        checkpoint_hashes=state.checkpoint_hashes if state else {},
        checkpoint_manifest_match=bool(state.checkpoint_manifest_match) if state else False,
    )
