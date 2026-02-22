"""
backend/app.py
================
Production FastAPI application for HNDSR super-resolution inference.

What  : Serves the HNDSR model via HTTP endpoints with health checks,
        metrics, version info, and inference.
Why   : The Jupyter notebook cannot serve production traffic. This app
        provides async request handling, backpressure, and monitoring.
How   : FastAPI with async endpoints, thread-pool GPU offloading,
        semaphore-based concurrency control, and Prometheus metrics.

Endpoints:
  GET  /health    — Liveness probe (Kubernetes)
  GET  /ready     — Readiness probe (model loaded?)
  POST /infer     — Super-resolution inference
  GET  /metrics   — Prometheus metrics
  GET  /version   — API and model version info
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import time
import traceback
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ServerConfig:
    """Server configuration loaded from environment variables."""
    max_concurrent_inferences: int = int(os.getenv("MAX_CONCURRENT", "4"))
    request_timeout_s: float = float(os.getenv("REQUEST_TIMEOUT_S", "30"))
    max_payload_mb: float = float(os.getenv("MAX_PAYLOAD_MB", "20"))
    max_queue_depth: int = int(os.getenv("MAX_QUEUE_DEPTH", "20"))
    model_dir: str = os.getenv("MODEL_DIR", "./checkpoints")
    device: str = os.getenv("DEVICE", "auto")
    rate_limit_per_hour: int = int(os.getenv("RATE_LIMIT_PER_HOUR", "100"))

    def resolve_device(self) -> str:
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device


CONFIG = ServerConfig()


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────────────────────────────────────────

class InferenceRequest(BaseModel):
    """Schema for super-resolution inference requests."""
    image: str = Field(..., description="Base64-encoded input image (PNG/JPEG)")
    scale_factor: int = Field(default=4, ge=2, le=8, description="Upscaling factor")
    ddim_steps: Optional[int] = Field(default=50, ge=10, le=200, description="DDIM sampling steps")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    return_metadata: bool = Field(default=True, description="Include inference metadata in response")


class InferenceResponse(BaseModel):
    """Schema for super-resolution inference responses."""
    image: str = Field(..., description="Base64-encoded output image")
    width: int
    height: int
    scale_factor: int
    metadata: Optional[dict] = None


class HealthResponse(BaseModel):
    """Schema for health check responses."""
    status: str
    model_loaded: bool
    gpu_available: bool
    gpu_name: Optional[str] = None
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    uptime_seconds: float
    queue_depth: int


class VersionResponse(BaseModel):
    """Schema for version info responses."""
    api_version: str = "1.0.0"
    model_version: str = "unknown"
    python_version: str
    torch_version: str
    cuda_available: bool
    cuda_version: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Application state
# ─────────────────────────────────────────────────────────────────────────────

class AppState:
    """Mutable application state."""
    model_loaded: bool = False
    start_time: float = time.time()
    inference_semaphore: asyncio.Semaphore = None
    active_requests: int = 0
    total_requests: int = 0
    total_errors: int = 0
    latency_sum_ms: float = 0.0

    # Rate limiting (simple in-memory)
    request_counts: dict = {}

    def __init__(self):
        self.inference_semaphore = asyncio.Semaphore(CONFIG.max_concurrent_inferences)
        self.request_counts = {}


state = AppState()


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan (startup / shutdown)
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    On startup: Load model, warmup GPU.
    On shutdown: Release GPU memory.
    """
    logger.info("Starting HNDSR API server...")
    device = CONFIG.resolve_device()
    logger.info("Device: %s", device)

    # TODO: Load actual HNDSR model
    # model = load_hndsr_model(CONFIG.model_dir, device)
    state.model_loaded = True
    state.start_time = time.time()

    # GPU warmup (prevents cold-start latency spike on first request)
    if device == "cuda":
        logger.info("Warming up GPU...")
        dummy = torch.randn(1, 3, 64, 64, device=device)
        _ = dummy * 2  # Simple operation to initialize CUDA context
        del dummy
        torch.cuda.empty_cache()
        logger.info("GPU warm-up complete")

    logger.info("HNDSR API ready!")
    yield

    # Cleanup
    logger.info("Shutting down HNDSR API...")
    # TODO: Release model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Shutdown complete")


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="HNDSR Super-Resolution API",
    description="Production API for Hybrid Neural Operator–Diffusion Super-Resolution",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Middleware: request tracking
# ─────────────────────────────────────────────────────────────────────────────

@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Track request count, latency, and errors."""
    start = time.perf_counter()
    state.total_requests += 1

    try:
        response = await call_next(request)
        latency_ms = (time.perf_counter() - start) * 1000
        state.latency_sum_ms += latency_ms
        response.headers["X-Request-Latency-Ms"] = f"{latency_ms:.1f}"
        return response
    except Exception:
        state.total_errors += 1
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Health endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Liveness probe.

    Kubernetes calls this every 10s. Must respond in <1s.
    If this fails, Kubernetes kills and restarts the pod.
    """
    gpu_name = None
    gpu_mem_used = None
    gpu_mem_total = None

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info(0)
        gpu_mem_used = (total - free) / 1e6
        gpu_mem_total = total / 1e6

    return HealthResponse(
        status="healthy" if state.model_loaded else "loading",
        model_loaded=state.model_loaded,
        gpu_available=torch.cuda.is_available(),
        gpu_name=gpu_name,
        gpu_memory_used_mb=gpu_mem_used,
        gpu_memory_total_mb=gpu_mem_total,
        uptime_seconds=time.time() - state.start_time,
        queue_depth=state.active_requests,
    )


@app.get("/ready")
async def ready():
    """
    Readiness probe.

    Kubernetes uses this to decide if traffic should be routed to this pod.
    Returns 503 if model is not loaded or queue is full.
    """
    if not state.model_loaded:
        raise HTTPException(503, detail="Model not loaded")
    if state.active_requests >= CONFIG.max_queue_depth:
        raise HTTPException(503, detail="Server overloaded",
                          headers={"Retry-After": "5"})
    return {"status": "ready"}


# ─────────────────────────────────────────────────────────────────────────────
# Inference endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/infer", response_model=InferenceResponse)
async def infer(request: InferenceRequest, req: Request):
    """
    Super-resolution inference endpoint.

    Flow:
      1. Validate payload size
      2. Check queue depth (reject if overloaded)
      3. Acquire semaphore (limit concurrent GPU ops)
      4. Decode base64 → PIL Image
      5. Run inference in thread pool (don't block event loop)
      6. Encode output → base64
      7. Return response with metadata

    Error handling:
      - 413: Payload too large
      - 422: Invalid image format
      - 503: Server overloaded (Retry-After header)
      - 504: Inference timeout
      - 500: Unexpected error
    """
    # ── Backpressure: reject if overloaded ───────────────────────────
    if state.active_requests >= CONFIG.max_queue_depth:
        raise HTTPException(
            503,
            detail=f"Server overloaded ({state.active_requests} active requests). Retry later.",
            headers={"Retry-After": "5"},
        )

    # ── Rate limiting (simple per-IP) ────────────────────────────────
    client_ip = req.client.host if req.client else "unknown"
    now = time.time()
    hour_key = f"{client_ip}:{int(now // 3600)}"
    state.request_counts[hour_key] = state.request_counts.get(hour_key, 0) + 1
    if state.request_counts[hour_key] > CONFIG.rate_limit_per_hour:
        raise HTTPException(429, detail="Rate limit exceeded")

    # ── Payload size check ───────────────────────────────────────────
    payload_mb = len(request.image) * 3 / 4 / 1e6  # base64 → bytes estimate
    if payload_mb > CONFIG.max_payload_mb:
        raise HTTPException(
            413,
            detail=f"Payload {payload_mb:.1f} MB exceeds limit {CONFIG.max_payload_mb} MB",
        )

    # ── Decode image ─────────────────────────────────────────────────
    try:
        img_bytes = base64.b64decode(request.image)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        w, h = img.size
    except Exception as exc:
        raise HTTPException(422, detail=f"Invalid image: {exc}")

    # ── Run inference ────────────────────────────────────────────────
    state.active_requests += 1
    start_time = time.perf_counter()

    try:
        async with asyncio.timeout(CONFIG.request_timeout_s):
            await state.inference_semaphore.acquire()
            try:
                # Run GPU inference in thread pool to avoid blocking event loop
                loop = asyncio.get_event_loop()
                output_img = await loop.run_in_executor(
                    None,
                    _run_inference,
                    img,
                    request.scale_factor,
                    request.ddim_steps,
                    request.seed,
                )
            finally:
                state.inference_semaphore.release()

    except asyncio.TimeoutError:
        state.total_errors += 1
        raise HTTPException(
            504,
            detail=f"Inference timeout ({CONFIG.request_timeout_s}s)",
        )
    except HTTPException:
        raise
    except Exception as exc:
        state.total_errors += 1
        logger.error("Inference error: %s", traceback.format_exc())
        raise HTTPException(500, detail=f"Inference failed: {exc}")
    finally:
        state.active_requests -= 1

    latency_ms = (time.perf_counter() - start_time) * 1000

    # ── Encode output ────────────────────────────────────────────────
    buf = io.BytesIO()
    output_img.save(buf, format="PNG")
    output_b64 = base64.b64encode(buf.getvalue()).decode()

    out_w, out_h = output_img.size

    metadata = None
    if request.return_metadata:
        metadata = {
            "input_size": f"{w}×{h}",
            "output_size": f"{out_w}×{out_h}",
            "scale_factor": request.scale_factor,
            "ddim_steps": request.ddim_steps,
            "latency_ms": round(latency_ms, 1),
            "device": CONFIG.resolve_device(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    return InferenceResponse(
        image=output_b64,
        width=out_w,
        height=out_h,
        scale_factor=request.scale_factor,
        metadata=metadata,
    )


def _run_inference(
    img: Image.Image,
    scale_factor: int,
    ddim_steps: int,
    seed: Optional[int],
) -> Image.Image:
    """
    Run HNDSR inference synchronously (called from thread pool).

    This function runs on a separate thread to avoid blocking the
    async event loop. It has exclusive access to the GPU via the
    semaphore acquired by the caller.
    """
    if seed is not None:
        torch.manual_seed(seed)

    # TODO: Replace with actual HNDSR inference
    # engine = get_inference_engine()
    # hr_tensor = engine.infer(img, scale_factor, ddim_steps)

    # Placeholder: bicubic upscale
    w, h = img.size
    output = img.resize((w * scale_factor, h * scale_factor), Image.BICUBIC)
    return output


# ─────────────────────────────────────────────────────────────────────────────
# Metrics endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/metrics")
async def metrics():
    """
    Prometheus-compatible metrics endpoint.

    Exposes:
      - hndsr_requests_total
      - hndsr_errors_total
      - hndsr_active_requests
      - hndsr_latency_avg_ms
      - hndsr_gpu_memory_used_bytes
    """
    lines = []
    lines.append(f"# HELP hndsr_requests_total Total inference requests")
    lines.append(f"# TYPE hndsr_requests_total counter")
    lines.append(f"hndsr_requests_total {state.total_requests}")

    lines.append(f"# HELP hndsr_errors_total Total inference errors")
    lines.append(f"# TYPE hndsr_errors_total counter")
    lines.append(f"hndsr_errors_total {state.total_errors}")

    lines.append(f"# HELP hndsr_active_requests Currently active requests")
    lines.append(f"# TYPE hndsr_active_requests gauge")
    lines.append(f"hndsr_active_requests {state.active_requests}")

    avg_latency = (
        state.latency_sum_ms / state.total_requests
        if state.total_requests > 0
        else 0
    )
    lines.append(f"# HELP hndsr_latency_avg_ms Average request latency")
    lines.append(f"# TYPE hndsr_latency_avg_ms gauge")
    lines.append(f"hndsr_latency_avg_ms {avg_latency:.1f}")

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        used = total - free
        lines.append(f"# HELP hndsr_gpu_memory_used_bytes GPU memory used")
        lines.append(f"# TYPE hndsr_gpu_memory_used_bytes gauge")
        lines.append(f"hndsr_gpu_memory_used_bytes {used}")

    lines.append(f"# HELP hndsr_uptime_seconds Server uptime")
    lines.append(f"# TYPE hndsr_uptime_seconds gauge")
    lines.append(f"hndsr_uptime_seconds {time.time() - state.start_time:.0f}")

    return Response(content="\n".join(lines) + "\n", media_type="text/plain")


# ─────────────────────────────────────────────────────────────────────────────
# Version endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/version", response_model=VersionResponse)
async def version():
    """Return API and model version information."""
    import platform
    return VersionResponse(
        api_version="1.0.0",
        model_version=os.getenv("MODEL_VERSION", "unknown"),
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        cuda_version=torch.version.cuda if torch.cuda.is_available() else None,
    )
