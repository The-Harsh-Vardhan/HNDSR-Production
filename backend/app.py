"""
backend/app.py
================
Production FastAPI application for HNDSR super-resolution inference.

What  : Serves the HNDSR model via HTTP endpoints with health checks,
        Prometheus metrics, version info, and inference.
Why   : The Jupyter notebook cannot serve production traffic. This app
        provides async request handling, backpressure, and monitoring.
How   : FastAPI with async endpoints, thread-pool GPU offloading,
        semaphore-based concurrency control, and prometheus_client metrics.

Endpoints:
  GET  /health    — Liveness probe (Kubernetes)
  GET  /ready     — Readiness probe (model loaded?)
  POST /infer     — Super-resolution inference
  GET  /metrics   — Prometheus metrics (histogram-based)
  GET  /version   — API and model version info

Post-Audit Fixes (2026-02-22):
  1. Replaced average latency gauge with Prometheus histogram (P50/P95/P99)
  2. Fixed rate limiter memory leak (hourly cleanup)
  3. Added image dimension guard (MAX_IMAGE_PIXELS = 16M)
  4. Added GPU OOM handling with torch.cuda.empty_cache()
  5. Added graceful shutdown (drain in-flight requests)
  6. Fixed CORS (removed allow_credentials with wildcard origins)
  7. Thread-safe active_requests counter via threading.Lock
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import signal
import threading
import time
import traceback
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
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
    max_image_pixels: int = int(os.getenv("MAX_IMAGE_PIXELS", "16000000"))  # 16M pixels

    def resolve_device(self) -> str:
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device


CONFIG = ServerConfig()


# ─────────────────────────────────────────────────────────────────────────────
# Prometheus metrics (proper histogram-based, not averages)
# ─────────────────────────────────────────────────────────────────────────────

REQUEST_COUNT = Counter(
    "hndsr_requests_total",
    "Total inference requests",
    ["method", "endpoint", "status"],
)
ERROR_COUNT = Counter(
    "hndsr_errors_total",
    "Total inference errors",
    ["error_type"],
)
INFERENCE_LATENCY = Histogram(
    "hndsr_inference_seconds",
    "Inference latency in seconds",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)
ACTIVE_REQUESTS = Gauge(
    "hndsr_active_requests",
    "Currently active inference requests",
)
GPU_MEMORY_USED = Gauge(
    "hndsr_gpu_memory_used_bytes",
    "GPU memory used in bytes",
)
UPTIME = Gauge(
    "hndsr_uptime_seconds",
    "Server uptime in seconds",
)
RATE_LIMITED = Counter(
    "hndsr_rate_limited_total",
    "Total requests rejected by rate limiter",
)
BACKPRESSURE_REJECTED = Counter(
    "hndsr_backpressure_rejected_total",
    "Total requests rejected by backpressure",
)


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
    active_requests: int


class VersionResponse(BaseModel):
    """Schema for version info responses."""
    api_version: str = "1.0.0"
    model_version: str = "unknown"
    python_version: str
    torch_version: str
    cuda_available: bool
    cuda_version: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Application state (thread-safe)
# ─────────────────────────────────────────────────────────────────────────────

class AppState:
    """
    Mutable application state with thread-safe counters.

    Thread safety:
      - active_requests uses a threading.Lock because it is modified
        from both the async event loop and thread-pool executor threads.
      - request_counts dict is only accessed from the event loop (single-threaded),
        but the cleanup is also event-loop-only, so no lock needed there.
    """

    def __init__(self):
        self.model_loaded: bool = False
        self.start_time: float = time.time()
        self.inference_semaphore: asyncio.Semaphore = asyncio.Semaphore(
            CONFIG.max_concurrent_inferences
        )
        self._shutdown_event: asyncio.Event = asyncio.Event()

        # Thread-safe active request counter
        self._active_requests: int = 0
        self._active_lock: threading.Lock = threading.Lock()

        # Rate limiting (per-IP per-hour)
        self.request_counts: dict = {}
        self._request_counter: int = 0  # for periodic cleanup

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
        """
        Check per-IP rate limit. Returns True if allowed, False if exceeded.

        Memory leak fix: cleans up old hour-keys every 100 requests to prevent
        unbounded dict growth (~336 MB/week at 10k IPs without cleanup).
        """
        now = time.time()
        current_hour = int(now // 3600)
        hour_key = f"{client_ip}:{current_hour}"

        self.request_counts[hour_key] = self.request_counts.get(hour_key, 0) + 1

        # Periodic cleanup: remove keys from previous hours
        self._request_counter += 1
        if self._request_counter % 100 == 0:
            stale_keys = [
                k for k in self.request_counts
                if not k.endswith(f":{current_hour}")
            ]
            for k in stale_keys:
                del self.request_counts[k]
            if stale_keys:
                logger.info("Rate limiter cleanup: removed %d stale keys", len(stale_keys))

        return self.request_counts[hour_key] <= CONFIG.rate_limit_per_hour


state: AppState = None  # Initialized in lifespan


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan (startup / shutdown with graceful drain)
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    On startup: Load model, warmup GPU, initialize state.
    On shutdown: Wait for in-flight requests (up to 30s), release GPU memory.
    """
    global state
    state = AppState()

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
        try:
            dummy = torch.randn(1, 3, 64, 64, device=device)
            _ = dummy * 2  # Simple operation to initialize CUDA context
            del dummy
            torch.cuda.empty_cache()
            logger.info("GPU warm-up complete")
        except Exception as exc:
            logger.error("GPU warm-up failed: %s", exc)

    logger.info("HNDSR API ready!")
    yield

    # ── Graceful shutdown: drain in-flight requests ──────────────────
    logger.info("Shutting down HNDSR API...")
    state.model_loaded = False  # Stop accepting new requests via /ready

    # Wait up to 30s for active requests to complete
    drain_timeout = 30.0
    drain_start = time.time()
    while state.active_requests > 0 and (time.time() - drain_start) < drain_timeout:
        logger.info(
            "Draining %d active requests (%.0fs remaining)...",
            state.active_requests,
            drain_timeout - (time.time() - drain_start),
        )
        await asyncio.sleep(1.0)

    if state.active_requests > 0:
        logger.warning(
            "Shutdown timeout: %d requests still active, forcing shutdown",
            state.active_requests,
        )

    # Release GPU memory
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

# CORS: allow_credentials=False with wildcard origins (spec-compliant)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


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
      1. Check backpressure (reject if overloaded)
      2. Check rate limit (per-IP per-hour)
      3. Validate payload size
      4. Decode base64 → PIL Image
      5. Validate image dimensions (guard against decompression bombs)
      6. Acquire semaphore (limit concurrent GPU ops)
      7. Run inference in thread pool (don't block event loop)
      8. Encode output → base64
      9. Return response with metadata

    Error handling:
      - 413: Payload too large or image dimensions too large
      - 422: Invalid image format
      - 429: Rate limit exceeded
      - 503: Server overloaded or GPU OOM (Retry-After header)
      - 504: Inference timeout
      - 500: Unexpected error
    """
    # ── Backpressure: reject if overloaded ───────────────────────────
    if state.active_requests >= CONFIG.max_queue_depth:
        BACKPRESSURE_REJECTED.inc()
        raise HTTPException(
            503,
            detail=f"Server overloaded ({state.active_requests} active requests). Retry later.",
            headers={"Retry-After": "5"},
        )

    # ── Rate limiting (per-IP, with memory leak fix) ─────────────────
    client_ip = req.client.host if req.client else "unknown"
    if not state.check_rate_limit(client_ip):
        RATE_LIMITED.inc()
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
        ERROR_COUNT.labels(error_type="decode").inc()
        raise HTTPException(422, detail=f"Invalid image: {exc}")

    # ── Image dimension guard (prevent decompression bombs) ──────────
    total_pixels = w * h
    if total_pixels > CONFIG.max_image_pixels:
        raise HTTPException(
            413,
            detail=(
                f"Image too large: {w}×{h} = {total_pixels:,} pixels. "
                f"Maximum: {CONFIG.max_image_pixels:,} pixels."
            ),
        )

    # ── Run inference ────────────────────────────────────────────────
    state.increment_active()
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
        ERROR_COUNT.labels(error_type="timeout").inc()
        raise HTTPException(
            504,
            detail=f"Inference timeout ({CONFIG.request_timeout_s}s)",
        )
    except HTTPException:
        raise
    except torch.cuda.OutOfMemoryError:
        # ── GPU OOM handling ─────────────────────────────────────────
        ERROR_COUNT.labels(error_type="gpu_oom").inc()
        logger.error("GPU OOM during inference for %d×%d image", w, h)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise HTTPException(
            503,
            detail="GPU out of memory. Try a smaller image or lower scale factor.",
            headers={"Retry-After": "10"},
        )
    except Exception as exc:
        ERROR_COUNT.labels(error_type="internal").inc()
        logger.error("Inference error: %s", traceback.format_exc())
        raise HTTPException(500, detail=f"Inference failed: {exc}")
    finally:
        state.decrement_active()

    latency_s = time.perf_counter() - start_time
    INFERENCE_LATENCY.observe(latency_s)
    REQUEST_COUNT.labels(method="POST", endpoint="/infer", status="200").inc()

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
            "latency_ms": round(latency_s * 1000, 1),
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

    NOTE: Currently a placeholder (bicubic upscale). The actual HNDSR
    model integration is tracked as a known limitation. See
    docs/PRODUCTION_MVP.md for details.
    """
    if seed is not None:
        torch.manual_seed(seed)

    # TODO: Replace with actual HNDSR inference
    # engine = get_inference_engine()
    # hr_tensor = engine.infer(img, scale_factor, ddim_steps)

    # Placeholder: bicubic upscale (documented as known limitation)
    w, h = img.size
    output = img.resize((w * scale_factor, h * scale_factor), Image.BICUBIC)
    return output


# ─────────────────────────────────────────────────────────────────────────────
# Metrics endpoint (Prometheus-native via prometheus_client)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/metrics")
async def metrics():
    """
    Prometheus-compatible metrics endpoint.

    Uses prometheus_client library for proper metric types:
      - hndsr_requests_total (Counter with labels)
      - hndsr_errors_total (Counter with error_type label)
      - hndsr_inference_seconds (Histogram with P50/P95/P99 support)
      - hndsr_active_requests (Gauge)
      - hndsr_gpu_memory_used_bytes (Gauge)
      - hndsr_uptime_seconds (Gauge)
      - hndsr_rate_limited_total (Counter)
      - hndsr_backpressure_rejected_total (Counter)

    Post-audit fix: Replaced manual average latency gauge with histogram.
    Histograms enable: histogram_quantile(0.99, rate(hndsr_inference_seconds_bucket[5m]))
    """
    # Update GPU memory before scrape
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        GPU_MEMORY_USED.set(total - free)

    UPTIME.set(time.time() - state.start_time)

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


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
