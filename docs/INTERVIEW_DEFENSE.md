# HNDSR Interview Defense Preparation

**Prep Date:** 2026-02-22 (48 hours before interview)
**Context:** Post-audit system simplification

---

## Part 1: 5-Minute System Explanation

### Opening (30 seconds)

> "HNDSR is a Hybrid Neural Operator–Diffusion model for satellite image super-resolution. I built the production inference serving infrastructure — not just the model, but the system around it: containerized deployment, GPU-aware concurrency control, Prometheus monitoring, and CI/CD. After an internal audit, I simplified the architecture to remove premature complexity and focus on what actually matters for a <5 req/s system."

### What Changed After Audit (60 seconds)

> "The audit found that I had built infrastructure for problems I didn't have yet — canary deployments with no users, a Redis queue worker that wasn't even connected to the API, shadow testing with no traffic. Classic over-engineering. I cut ~650 lines of dead code and focused on fixing the real bugs: a memory leak in the rate limiter, missing GPU OOM handling, average-only latency metrics that hid P99 spikes, and an invalid CORS configuration."

### Architecture (60 seconds)

> "The simplified system is three containers: FastAPI inference server with GPU access, Prometheus for metrics collection, and Grafana for dashboards. The inference endpoint uses an asyncio semaphore to limit GPU concurrency to 4, backpressure at 20 queued requests (returns 503 with Retry-After), and runs model inference in a thread pool to keep the event loop responsive. Metrics use prometheus_client histograms, not hand-rolled averages — so I can query P50, P95, P99 latency."

### Key Technical Decisions (60 seconds)

> "Three decisions I'd defend:
>
> 1. **Sync inference over Redis queue**: For <5 req/s, the overhead of Redis serialization, job polling, and result retrieval adds complexity without benefit. Direct inference is simpler and has lower latency.
>
> 2. **Semaphore over thread pool sizing**: I use asyncio.Semaphore(4) to limit concurrent GPU operations rather than limiting the thread pool, because the semaphore provides backpressure visibility (I can count waiting requests) and decouples concurrency control from Python's executor implementation.
>
> 3. **Prometheus histograms over averages**: An average latency of 400ms could mean all requests take 400ms, or 99% take 100ms and 1% take 30 seconds. Only histograms expose the tail. I chose buckets [0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30] because they bracket the expected DDIM sampling range."

### What's Deferred and Why (60 seconds)

> "The actual HNDSR model architecture is pending — the inference endpoint currently returns bicubic upscaling as a placeholder. This is documented, not hidden. The infrastructure is designed to receive the real model: the same thread-pool pattern, GPU memory management, and monitoring will work. When traffic justifies it, I'd add: a Redis-backed async path for large images, horizontal scaling via Kubernetes, and TensorRT optimization for 2-4× inference speedup."

### Closing (30 seconds)

> "The key lesson: production readiness isn't about having the most components — it's about every component actually working. A system with 3 working services and honest documentation is more production-ready than one with 6 services where half are scaffolding."

---

## Part 2: 5 Tough System Design Questions

### Q1: "Your inference endpoint runs in a thread pool. The GIL should make this pointless — why not use multiprocessing?"

**Answer:**

> "The GIL concern is valid but misapplied here. The inference thread spends >95% of its time in PyTorch operations, which release the GIL during CUDA kernel execution. The GIL is only held during Python-level operations: base64 decode (~5ms), PIL image open (~2ms), result encoding (~5ms). That's ~12ms of GIL contention per request out of a ~1000ms total.
>
> Multiprocessing would require model duplication in each process — that's 800MB× of GPU memory per worker. On a 16GB T4 with a semaphore of 4, that's 3.2GB just for model copies, leaving only 12.8GB for actual inference tensors. The thread pool shares one model copy.
>
> If CPU-bound preprocessing became the bottleneck (it won't at <5 req/s), I'd move base64 decoding into a preprocessing service, not add multiprocessing to the inference server."

---

### Q2: "Your semaphore allows 4 concurrent requests, but asyncio.timeout only cancels the coroutine — the thread keeps running. How do you handle this?"

**Answer:**

> "You're right — this is a documented known limitation. When asyncio.timeout fires, the coroutine is cancelled and the client gets a 504, but the thread continues consuming GPU resources until the inference completes. The semaphore is released by the `finally` block, which means a new request can start while the timed-out thread is still running.
>
> For the current placeholder (bicubic resize, <10ms), this is harmless. For the real model (1-3s), the mitigation path is:
>
> 1. Pass a `threading.Event` to `_run_inference`
> 2. Check the event between DDIM sampling steps (50 natural checkpoints)
> 3. If the event is set, abort early and call `torch.cuda.empty_cache()`
>
> I didn't implement this yet because the placeholder doesn't need it, and cooperative cancellation requires model-level integration that I'll add when the model is integrated."

---

### Q3: "How would you scale this beyond a single GPU?"

**Answer:**

> "Three phases:
>
> **Phase 1 (5-20 req/s):** Add a load balancer (Nginx or cloud ALB) in front of 2-4 identical Docker Compose stacks on separate GPU machines. Sticky sessions not needed — inference is stateless.
>
> **Phase 2 (20-100 req/s):** Move to Kubernetes with GPU node pools. HPA based on custom metric (active_requests / max_queue_depth). Pre-warm pods to avoid cold start latency. Use KEDA for scale-to-zero during off-hours.
>
> **Phase 3 (100+ req/s):** Replace direct PyTorch serving with NVIDIA Triton Inference Server. Triton provides: dynamic batching (stack multiple requests into one GPU forward pass), model format optimization (TensorRT INT8 — 2-4× speedup), multi-model serving on shared GPUs, and gRPC for lower serialization overhead.
>
> The asyncio.Semaphore pattern scales well because each instance independently manages its own GPU. The Prometheus metrics aggregate cleanly across instances. The key bottleneck at scale isn't the serving code — it's the diffusion model's sequential 50-step DDIM sampling, which can be reduced to 20 steps with ~1-2 dB quality tradeoff."

---

### Q4: "Your rate limiter is in-memory and per-process. How do you rate-limit across multiple instances?"

**Answer:**

> "For the current single-instance deployment, per-process rate limiting works. The cleanup every 100 requests prevents the memory leak that the audit identified.
>
> For multi-instance, I'd migrate to Redis-based rate limiting using a sliding window algorithm:
>
> ```
> key = f'ratelimit:{client_ip}'
> current = redis.incr(key)
> if current == 1:
>     redis.expire(key, 3600)  # 1 hour TTL
> if current > limit:
>     return 429
> ```
>
> Redis gives: atomic counters across instances, automatic TTL expiry (no cleanup code), and the ability to inspect rate limit state for debugging. I deliberately removed Redis from the current stack because a single running service that I understand is better than three services where one is misconfigured. When the second instance is added, Redis comes back — but purposefully, not as a premature dependency."

---

### Q5: "The audit says your model inference is a placeholder. How is this system production-ready?"

**Answer:**

> "It's not production-ready for end users — and I don't claim it is. What IS production-ready is the inference infrastructure: the container builds, the monitoring pipeline works, the backpressure prevents overload, the GPU OOM handler prevents cascading failures, and the CI validates configuration files.
>
> This is intentional. ML systems have two independent failure modes: the model can be wrong, and the serving can be broken. By separating them, I can:
>
> 1. Validate all serving infrastructure independently (health checks, metrics, error handling, deployment)
> 2. Swap in the real model as a single code change in `_run_inference()`
> 3. Test the model integration against a serving stack that's already proven stable
>
> If I'd tried to do both simultaneously — implement the model AND build the serving infrastructure — I'd have two untested systems composed together. Instead, I have one tested system and one clear integration point. The placeholder is a feature of the development process, not a bug."

---

## Quick Reference: Key Numbers

| Metric | Value | Source |
|--------|-------|--------|
| Semaphore concurrency | 4 | `MAX_CONCURRENT` env var |
| Backpressure threshold | 20 active requests | `MAX_QUEUE_DEPTH` env var |
| Rate limit | 100 req/hr/IP | `RATE_LIMIT_PER_HOUR` env var |
| Request timeout | 30 seconds | `REQUEST_TIMEOUT_S` env var |
| Max image size | 16M pixels (4000×4000) | `MAX_IMAGE_PIXELS` env var |
| Max payload | 20 MB | `MAX_PAYLOAD_MB` env var |
| Histogram buckets | [0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30]s | Hardcoded in app.py |
| Cleanup frequency | Every 100 requests | Rate limiter hygiene |
| Graceful shutdown drain | 30 seconds max | Lifespan handler |
| Expected real-model latency | 520–1540ms per 256×256 tile | Architecture analysis |
| T4 GPU memory | 16 GB | Hardware spec |
