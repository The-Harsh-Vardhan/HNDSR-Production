# HNDSR Production MVP — Simplified Architecture

**Date:** 2026-02-22 (Post-Audit Revision)
**Status:** MVP — Inference infrastructure ready, model integration pending
**Target Load:** <5 req/s, single GPU, burst-safe

---

## System Overview

After a production-readiness audit revealed HIGH RISK from over-engineering and missing implementations, the system was stripped to a minimal viable production (MVP) stack focused on stable inference serving.

### What Was Removed (and Why)

| Component | Lines Removed | Reason |
|-----------|:---:|--------|
| Canary deployment (`canary_deploy.py`) | 354 | No users, no real model — premature |
| Shadow deployment (in canary_deploy.py) | ~70 | No traffic to shadow-test against |
| Redis queue worker (`inference_worker.py`) | 294 | Orphaned — never connected to API, fake batching |
| Redis service | — | No longer needed (sync inference only) |
| Nginx service | — | No config existed, placeholder |

### What Was Fixed

| Fix | Bug | Impact |
|-----|-----|--------|
| Prometheus histogram | Average latency hid P99 spikes | Can now query P50/P95/P99 |
| Rate limiter cleanup | Dict grew ~336 MB/week at 10k IPs | Memory leak eliminated |
| Image dimension guard | 256M pixel bomb → server OOM | Rejects >16M pixels |
| GPU OOM handling | OOM corrupted CUDA context | Catches OOM, clears cache, returns 503 |
| Graceful shutdown | Hard kill dropped in-flight requests | 30s drain on SIGTERM |
| CORS fix | `allow_credentials=True` with `*` is invalid | Spec-compliant now |
| Thread-safe counter | `active_requests += 1` was not atomic | Uses `threading.Lock` |

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    DEPLOYMENT HOST                       │
│                  (Single GPU Machine)                    │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │              Docker Compose Stack                  │  │
│  │                                                   │  │
│  │  ┌─────────────────────────────────────────────┐  │  │
│  │  │  hndsr-api (FastAPI + Uvicorn)              │  │  │
│  │  │  Port 8000                                  │  │  │
│  │  │                                             │  │  │
│  │  │  Endpoints:                                 │  │  │
│  │  │    GET  /health  → Liveness probe           │  │  │
│  │  │    GET  /ready   → Readiness probe          │  │  │
│  │  │    POST /infer   → Inference (GPU)          │  │  │
│  │  │    GET  /metrics → Prometheus metrics        │  │  │
│  │  │    GET  /version → Version info             │  │  │
│  │  │                                             │  │  │
│  │  │  Concurrency: Semaphore(4) + backpressure   │  │  │
│  │  │  GPU: NVIDIA T4/A10 (single)                │  │  │
│  │  └─────────────┬───────────────────────────────┘  │  │
│  │                │ scrape /metrics every 15s         │  │
│  │  ┌─────────────▼───────────────────────────────┐  │  │
│  │  │  Prometheus (v2.48.0)                       │  │  │
│  │  │  Port 9090                                  │  │  │
│  │  │  Alerting rules: 4 critical + 3 warning     │  │  │
│  │  └─────────────┬───────────────────────────────┘  │  │
│  │                │ data source                      │  │
│  │  ┌─────────────▼───────────────────────────────┐  │  │
│  │  │  Grafana (v10.2.0)                          │  │  │
│  │  │  Port 3001                                  │  │  │
│  │  │  Dashboard: 7 panels                        │  │  │
│  │  └─────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## Request Flow

```
Client POST /infer (base64 image, scale_factor)
    │
    ├─ [1] Backpressure check: active_requests >= 20?
    │      YES → 503 + Retry-After: 5
    │
    ├─ [2] Rate limit check: >100 req/hr/IP?
    │      YES → 429 Rate Limit Exceeded
    │
    ├─ [3] Payload size check: >20 MB?
    │      YES → 413 Payload Too Large
    │
    ├─ [4] Base64 decode → PIL Image
    │      FAIL → 422 Invalid Image
    │
    ├─ [5] Dimension guard: w×h > 16M pixels?
    │      YES → 413 Image Too Large
    │
    ├─ [6] Acquire semaphore (max 4 concurrent)
    │      TIMEOUT (30s) → 504 Timeout
    │
    ├─ [7] Run inference in ThreadPoolExecutor
    │      GPU OOM → torch.cuda.empty_cache() → 503
    │      Other error → 500
    │
    ├─ [8] Encode output → base64 PNG
    │
    └─ [9] Return InferenceResponse + metadata
           Record latency in Prometheus histogram
```

---

## Failure Handling

| Failure | Detection | Response | Recovery |
|---------|-----------|----------|----------|
| Traffic spike (5× normal) | `hndsr_active_requests` hits 20 | 503 with Retry-After | Automatic — backpressure sheds load |
| GPU OOM | `torch.cuda.OutOfMemoryError` | 503 + `empty_cache()` | Automatic — CUDA cache cleared |
| Large image attack | Dimension check before GPU | 413 rejection | Automatic — never reaches GPU |
| Rate limit abuse | Per-IP hourly counter | 429 rejection | Automatic — counter resets hourly |
| Memory leak (old) | **Fixed** — periodic cleanup | N/A | Cleanup every 100 requests |
| Inference timeout | `asyncio.timeout(30s)` | 504 timeout | Thread continues (known limitation) |
| Service crash | Docker `restart: unless-stopped` | Auto-restart | ~60s cold start |

---

## Backpressure Logic

```
Incoming request
    │
    ▼
active_requests >= MAX_QUEUE_DEPTH (20)?
    │ YES: Return 503 + Retry-After: 5
    │ NO:  Continue
    ▼
Acquire semaphore (capacity: 4)
    │ Blocks if all 4 slots busy
    │ Timeout: 30s → 504
    ▼
Run inference (thread pool)
    │
    ▼
Release semaphore
```

**Why this works for <5 req/s:**
- At 1 req/s with ~1s inference: 1 concurrent request on average
- At 5 req/s with ~1s inference: ~5 concurrent, semaphore queues 1
- At 20 req/s: backpressure kicks in, 503s shed excess load
- No queue explosion, no OOM cascade

---

## Known Limitations

> [!CAUTION]
> These are intentionally documented, not hidden. Each has a rationale for deferral.

| Limitation | Impact | Why Deferred |
|------------|--------|-------------|
| **Placeholder inference** (bicubic upscale, not real HNDSR model) | Core functionality incomplete | Model architecture requires multi-week implementation; infrastructure is ready to receive it |
| **In-memory model registry** | Registry state lost on restart | MLflow backend provides persistence; in-memory is a cache layer |
| **Single worker process** | No CPU parallelism (GIL) | Adequate for <5 req/s; GPU is the bottleneck, not CPU |
| **No Kubernetes** | Manual scaling only | Docker Compose sufficient for single GPU; K8s adds operational overhead |
| **asyncio.timeout doesn't cancel thread** | Timed-out inference continues consuming GPU | Cooperative cancellation requires model-level changes |
| **No authentication** | API is open | Not needed for initial deployment; add API key middleware when users exist |

---

## Current Scaling Capacity

| Metric | Value | Basis |
|--------|-------|-------|
| Max sustained throughput | ~4 req/s (small images) | Semaphore=4, ~1s per request |
| Max burst capacity | 20 requests queued | MAX_QUEUE_DEPTH before 503 |
| Cold start time | 8–30s (model dependent) | Docker healthcheck allows 60s |
| Peak GPU memory | ~8.8 GB (4 concurrent, 256×256) | Fits on T4 (16 GB) |
| Single request latency | 520–1540ms (real model estimate) | Currently ~10ms (placeholder) |

---

## Monitoring Coverage

| What's Monitored | Metric | Alert |
|------------------|--------|-------|
| Request throughput | `hndsr_requests_total` | — |
| Error rate by type | `hndsr_errors_total{error_type}` | >5% → Critical |
| Latency P50/P95/P99 | `hndsr_inference_seconds` histogram | P99>10s → Critical |
| GPU memory | `hndsr_gpu_memory_used_bytes` | >95% → Critical |
| Active requests | `hndsr_active_requests` | — |
| Rate limiting | `hndsr_rate_limited_total` | >10/s → Warning |
| Backpressure | `hndsr_backpressure_rejected_total` | >1/s → Warning |
| Service health | `up{job="hndsr-api"}` | Down → Critical |
| Uptime | `hndsr_uptime_seconds` | — |

---

## Deployment

### Quick Start

```bash
# Start the stack
cd docker/
docker compose up -d

# Verify health
curl http://localhost:8000/health

# View metrics
curl http://localhost:8000/metrics

# Grafana dashboard
open http://localhost:3001  # admin / hndsr-admin

# Prometheus
open http://localhost:9090
```

### GPU Requirement

The API container requires an NVIDIA GPU with the NVIDIA Container Toolkit installed.
For CPU-only testing, set `DEVICE=cpu` in docker-compose.yml.

---

## Risk Disclosure

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Model integration changes inference contract | High (when real model is added) | Shape contract tests in CI |
| Single GPU failure = total outage | Medium | Docker auto-restart; future: multi-replica |
| Rate limiter bypassable via IP rotation | Medium | Acceptable at <5 req/s scale |
| No data encryption in transit | Low (internal network) | Add TLS termination when exposed publicly |
