"""
backend/inference_worker.py
=============================
Mini-batch GPU inference worker for asynchronous processing.

What  : Redis-backed worker that processes inference jobs from a queue.
Why   : Decouples request handling from GPU computation. Enables:
        - Horizontal scaling (add more workers for more throughput)
        - Mini-batching (combine multiple requests for GPU efficiency)
        - Job persistence (requests survive pod restarts)
How   : Worker BRPOP from Redis, groups jobs into mini-batches,
        runs inference, stores results back in Redis.

Architecture:
  Client → FastAPI (LPUSH job) → Redis Queue → Worker (BRPOP) → GPU → Redis (result)
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import signal
import time
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# ─────────────────────────────────────────────────────────────────────────────
# Worker configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WorkerConfig:
    """Configuration for the inference worker."""
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    job_queue: str = "hndsr:inference:queue"
    result_prefix: str = "hndsr:inference:result:"
    dead_letter_queue: str = "hndsr:inference:dlq"

    # Batching
    max_batch_size: int = int(os.getenv("MAX_BATCH_SIZE", "4"))
    batch_timeout_s: float = float(os.getenv("BATCH_TIMEOUT_S", "0.5"))

    # Processing
    device: str = os.getenv("DEVICE", "auto")
    model_dir: str = os.getenv("MODEL_DIR", "./checkpoints")
    result_ttl_s: int = int(os.getenv("RESULT_TTL_S", "3600"))  # 1 hour
    max_retries: int = int(os.getenv("MAX_RETRIES", "3"))

    def resolve_device(self) -> str:
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device


# ─────────────────────────────────────────────────────────────────────────────
# Job data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class InferenceJob:
    """A single inference job from the queue."""
    job_id: str
    image_b64: str
    scale_factor: int = 4
    ddim_steps: int = 50
    seed: Optional[int] = None
    attempt: int = 0
    created_at: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str) -> "InferenceJob":
        return cls(**json.loads(data))


@dataclass
class InferenceResult:
    """Result of an inference job."""
    job_id: str
    status: str  # "completed" | "failed"
    image_b64: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    latency_ms: Optional[float] = None
    error: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self))


# ─────────────────────────────────────────────────────────────────────────────
# Mini-batching inference worker
# ─────────────────────────────────────────────────────────────────────────────

class InferenceWorker:
    """
    GPU inference worker with mini-batching.

    Mini-batching strategy:
      1. BRPOP one job from Redis (blocking, waits for work)
      2. Non-blocking LPOP up to (max_batch_size - 1) additional jobs
      3. If batch_timeout reached or max_batch_size filled, process batch
      4. Store results in Redis with TTL

    Why mini-batching:
      - GPU is most efficient with batch_size > 1 (better utilization)
      - Individual requests arrive at different times
      - Without batching: GPU utilization ~20-30%
      - With mini-batching: GPU utilization ~60-80%

    Why dead-letter queue:
      - Jobs that fail max_retries times are moved to DLQ
      - Prevents infinite retry loops
      - Enables manual inspection and replay
    """

    def __init__(self, config: Optional[WorkerConfig] = None):
        self.config = config or WorkerConfig()
        self._running = False
        self._redis = None

        # Graceful shutdown
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info("Received signal %d, shutting down...", signum)
        self._running = False

    def _connect_redis(self):
        """Lazy Redis connection."""
        if self._redis is None:
            import redis
            self._redis = redis.from_url(
                self.config.redis_url,
                decode_responses=True,
            )
            logger.info("Connected to Redis: %s", self.config.redis_url)
        return self._redis

    def run(self):
        """
        Main worker loop.

        Blocks indefinitely, processing jobs from the Redis queue.
        Exits cleanly on SIGTERM/SIGINT.
        """
        r = self._connect_redis()
        device = self.config.resolve_device()
        logger.info("Worker started on device: %s", device)

        # TODO: Load actual model
        # model = load_hndsr_model(self.config.model_dir, device)

        self._running = True
        batch_count = 0

        while self._running:
            try:
                # ── Step 1: Blocking pop for first job ────────────────
                result = r.brpop(self.config.job_queue, timeout=5)
                if result is None:
                    continue  # Timeout, loop and check _running

                _, job_data = result
                jobs = [InferenceJob.from_json(job_data)]

                # ── Step 2: Non-blocking collect more jobs for batch ──
                batch_start = time.time()
                while (
                    len(jobs) < self.config.max_batch_size
                    and (time.time() - batch_start) < self.config.batch_timeout_s
                ):
                    more = r.lpop(self.config.job_queue)
                    if more is None:
                        break
                    jobs.append(InferenceJob.from_json(more))

                # ── Step 3: Process batch ─────────────────────────────
                batch_count += 1
                logger.info(
                    "Processing batch #%d: %d jobs", batch_count, len(jobs)
                )

                results = self._process_batch(jobs, device)

                # ── Step 4: Store results ─────────────────────────────
                for result in results:
                    result_key = f"{self.config.result_prefix}{result.job_id}"
                    r.setex(
                        result_key,
                        self.config.result_ttl_s,
                        result.to_json(),
                    )

                    if result.status == "failed":
                        # Check retry count
                        job = next(j for j in jobs if j.job_id == result.job_id)
                        if job.attempt < self.config.max_retries:
                            job.attempt += 1
                            r.lpush(self.config.job_queue, job.to_json())
                            logger.warning(
                                "Retrying job %s (attempt %d/%d)",
                                job.job_id, job.attempt, self.config.max_retries,
                            )
                        else:
                            # Move to dead-letter queue
                            r.lpush(self.config.dead_letter_queue, job.to_json())
                            logger.error(
                                "Job %s moved to DLQ after %d retries",
                                job.job_id, self.config.max_retries,
                            )

            except KeyboardInterrupt:
                break
            except Exception as exc:
                logger.error("Worker error: %s", exc)
                time.sleep(1)  # Backoff on unexpected errors

        logger.info("Worker stopped after %d batches", batch_count)

    def _process_batch(
        self, jobs: List[InferenceJob], device: str
    ) -> List[InferenceResult]:
        """Process a batch of inference jobs."""
        results = []

        for job in jobs:
            start = time.perf_counter()
            try:
                # Decode image
                img_bytes = base64.b64decode(job.image_b64)
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

                # Set seed if provided
                if job.seed is not None:
                    torch.manual_seed(job.seed)

                # TODO: Replace with actual HNDSR inference
                w, h = img.size
                output = img.resize(
                    (w * job.scale_factor, h * job.scale_factor),
                    Image.BICUBIC,
                )

                # Encode output
                buf = io.BytesIO()
                output.save(buf, format="PNG")
                out_b64 = base64.b64encode(buf.getvalue()).decode()

                out_w, out_h = output.size
                latency_ms = (time.perf_counter() - start) * 1000

                results.append(InferenceResult(
                    job_id=job.job_id,
                    status="completed",
                    image_b64=out_b64,
                    width=out_w,
                    height=out_h,
                    latency_ms=round(latency_ms, 1),
                ))

            except Exception as exc:
                logger.error("Job %s failed: %s", job.job_id, exc)
                results.append(InferenceResult(
                    job_id=job.job_id,
                    status="failed",
                    error=str(exc),
                ))

        return results


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    worker = InferenceWorker()
    worker.run()
