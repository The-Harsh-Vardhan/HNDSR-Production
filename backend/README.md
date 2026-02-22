# Backend API

## What
Production FastAPI application serving HNDSR super-resolution inference with async processing, backpressure, retry logic, and health monitoring.

## Why

### Async Pipeline Design
GPU inference takes 500ms–3s. Without async:
- Server handles 1 request at a time per worker
- 10 concurrent users → 30s wait for the last user
- Health checks time out → Kubernetes restarts pod

With async:
- Inference runs in a thread pool via `run_in_executor()`
- Event loop stays responsive for health checks, metrics
- Semaphore limits concurrent GPU operations (prevents OOM)

### Collapse Prevention
Without backpressure, a traffic spike can:
1. Queue 100 requests → GPU OOM → Pod crash → More traffic to other pods
2. Other pods cascade-fail → Complete service outage

Our approach:
- **Queue depth check**: Reject requests when queue > 20 (HTTP 503)
- **Semaphore**: Max 4 concurrent GPU inferences
- **Request timeout**: Cancel after 30s
- **Rate limiting**: Per-IP token bucket (100 requests/hour)

### GPU Starvation Protection
If all inference slots are occupied:
1. New requests receive 503 with `Retry-After` header
2. Client implements exponential backoff (1s → 2s → 4s → 8s)
3. Circuit breaker opens after 3 consecutive GPU failures
4. Automatic fallback to CPU inference (slower but available)

## How
```bash
# Run the API server
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --workers 1

# Test the API
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{"image": "<base64-encoded-image>", "scale_factor": 4}'
```
