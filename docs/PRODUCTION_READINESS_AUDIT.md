# HNDSR Production-Readiness & Feasibility Audit

**Auditor Role:** Principal Engineer  
**Date:** 2026-02-22  
**Scope:** Full end-to-end architecture review before real-world deployment  
**Verdict:** HIGH RISK — see §10 for details  

---

## Table of Contents

1. [Architecture Stress Test](#1-architecture-stress-test)
2. [Performance Feasibility Analysis](#2-performance-feasibility-analysis)
3. [Cost Feasibility & Economic Realism](#3-cost-feasibility--economic-realism)
4. [Failure Mode & Stability Audit](#4-failure-mode--stability-audit)
5. [Scalability Limits & Hard Constraints](#5-scalability-limits--hard-constraints)
6. [Security & Hardening Review](#6-security--hardening-review)
7. [Observability & Monitoring Audit](#7-observability--monitoring-audit)
8. [CI/CD & GitHub Actions Audit](#8-cicd--github-actions-audit)
9. [Top 5 Most Likely Production Failures](#9-top-5-most-likely-production-failures)
10. [Final Production Readiness Verdict](#10-final-production-readiness-verdict)

---

## 1. Architecture Stress Test

### 1.1 Data Layer

#### What Is Implemented
`data_pipeline/etl_pipeline.py` provides a single-process ETL script that validates images (corruption, decompression-bomb guard), downsamples HR→LR at multiple scale factors via PIL `BICUBIC`, stratified-splits into train/val/test, computes per-file SHA-256 hashes, and writes a JSON manifest plus optional Parquet metadata. `storage_config.py` centralises S3/MinIO/local paths via Pydantic-settings.

#### Why It Exists
Reproducible data preparation is the foundation of any ML system. Without deterministic splits and hash-verified data, training runs are non-comparable, and data corruption can silently degrade model quality by 0.5–2 dB PSNR.

#### How It Works Technically
1. Scan `input_dir` for images with extensions `.png/.tif/.tiff/.jpg/.jpeg`.
2. For each image: open with PIL, force full decode (`.load()`) to detect corruption, record dimensions/channels/filesize/SHA-256.
3. Stratified split via `numpy.random.RandomState(seed).permutation()` — **note: this is not truly stratified** (see below).
4. Copy HR originals into `output_dir/{split}/HR/`, then downsample at each scale into `output_dir/{split}/LR_{scale}x/`.
5. Aggregate all hashes, sort alphabetically, concatenate, SHA-256 the result → dataset fingerprint.
6. Write `manifest.json` and `metadata.parquet`.

#### Assumptions It Depends On
- All training images fit in local memory (PIL opens one-at-a-time but copies entire files via `shutil.copy2`).
- `numpy.random.RandomState` is deterministic across numpy versions — **true only if the version is fixed** (RandomState is legacy API; numpy ≥1.17 recommends `default_rng`).
- The input directory is flat (no recursion into subdirectories).
- Network access to S3/MinIO is not implemented in the ETL script — `storage_config.py` defines boto3 settings but `etl_pipeline.py` reads from local paths only.

#### Realistic Failure Modes
| Failure | Likelihood | Impact |
|---------|-----------|--------|
| S3 download not implemented — pipeline cannot ingest from cloud | **Confirmed** | **BLOCKING** — claimed architecture shows S3 as data source but code reads local only |
| "Stratified split" is actually random permutation — no geographic stratification | **Confirmed** | Medium — validation metrics will be optimistically biased if tiles cluster geographically |
| numpy RandomState version drift breaks split determinism | Medium | High — different numpy versions may yield different splits, invalidating comparison across runs |
| Single-threaded — O(N) sequential image I/O becomes bottleneck at >10k images | High | Training stalled on data prep |
| No resumability — crash mid-ETL requires full re-run | Medium | Wasted compute |
| Parquet write fails silently if pyarrow not installed | **Confirmed** (logged as warning) | Medium — downstream tools expecting Parquet will fail |

#### Hidden Bottlenecks
- Full file copy via `shutil.copy2` doubles storage requirements (raw + output). At satellite-scale datasets (100k+ tiles × 4 scales), this means 5× the raw storage.
- SHA-256 hashing reads every file twice (once for validation, once during copy verification) — double I/O.
- No parallel processing — `multiprocessing.Pool` or `concurrent.futures` would give 4–8× speedup on multi-core machines.

#### What Is Over-Engineered
- SHA-256 per-file hashing is correct but the combined "dataset hash" (sorting all hashes and re-hashing) adds negligible value over DVC's own content hashing.

#### What Is Under-Engineered
- **No cloud storage integration** despite `storage_config.py` defining S3 settings — the ETL never calls `get_s3_client()`.
- **No incremental processing** — adding 10 new images forces re-processing the entire dataset.
- **No geographic metadata** — `sensor`, `capture_date`, and `geographic_region` fields exist in schema but are never populated. Claimed stratification is a misnomer.
- **No data validation schema enforcement** — the Parquet schema in `storage_config.py` (METADATA_SCHEMA, CATALOG_SCHEMA) is never used for validation.

---

### 1.2 Training Pipeline

#### What Is Implemented
`training/train_pipeline.py` defines a three-stage sequential trainer (Autoencoder → Neural Operator → Diffusion UNet) with MLflow integration, CosineAnnealing LR scheduler, and early stopping. `experiment_tracking.py` wraps MLflow with auto-logging of system info, SHA-256 checkpoint hashing, and HPO via Optuna.

#### Why It Exists
The three stages must execute sequentially because each stage's frozen outputs become the next stage's input. MLflow enables run comparison, and HPO automates the hyperparameter search space.

#### How It Works Technically
Stage 1 trains the autoencoder on L1 reconstruction loss. Stage 2 freezes the autoencoder, encodes LR/HR into latent space, and trains the Neural Operator on MSE loss between `z_lr→z_pred` and `z_hr`. Stage 3 freezes both prior stages, implements DDPM noise scheduling (`betas = linspace(1e-4, 0.02, T)`), samples random timesteps, adds noise `z_noisy = sqrt(ᾱ_t)·z_hr + sqrt(1-ᾱ_t)·ε`, and trains the UNet to predict `ε_θ(z_noisy, t, z_no)` with gradient clipping at `max_norm=1.0`.

#### Assumptions It Depends On
- The three model architectures (`HNDSRAutoencoder`, `HNDSRNeuralOperator`, `HNDSRDiffusionUNet`) exist and provide `.encode()`, `.decode()`, and `forward(z_noisy, t, condition)` interfaces. **These are not implemented** — the pipeline has `# TODO` placeholders.
- DataLoaders return dicts with keys `"hr"` and `"lr"`. **No dataset class is implemented.**
- MLflow server is accessible. Falls back gracefully if not, but **no metrics are persisted in that case**.
- Single-GPU training only — no `DistributedDataParallel` or multi-GPU support.

#### Realistic Failure Modes
| Failure | Likelihood | Impact |
|---------|-----------|--------|
| **Entire pipeline is a skeleton** — all model/data code is TODO stubs | **Confirmed** | **CRITICAL — cannot train any model** |
| No mixed-precision training (AMP) — VRAM usage will be 2× higher than necessary | High | OOM on 16 GB GPUs with batch_size=16 at 256×256 |
| CosineAnnealing restarts from 0 after early stopping — wasted schedule budget | Medium | Suboptimal LR when stopped early at epoch 12/20 |
| No gradient accumulation — batch_size is hard-limited by VRAM | High | Cannot simulate large effective batch on small GPUs |

#### Hidden Coupling
- The diffusion training computes `alpha_bar` on-device at pipeline init, but `config.device` is resolved AFTER `seed_everything()`, creating a potential timing issue if device resolution modifies torch state.
- MLflow's `sqlite:///mlflow.db` uses a local SQLite file — concurrent training runs will cause database locks.

#### Technical Debt
- `train_pipeline.py` accepts CLI `--stage` arg in DVC pipeline but parses `--ae-lr`, `--no-lr`, `--diff-lr` etc. — the `run_training_pipeline()` function ignores `--stage` and always runs all three stages.
- No checkpoint resumption — crash during Stage 3 (epoch 20/30) requires re-training all three stages from scratch.

---

### 1.3 DVC Pipeline

#### What Is Implemented
`dvc_pipeline/dvc.yaml` defines a 5-stage DAG: `preprocess → train_autoencoder → train_neural_operator → train_diffusion → evaluate`. Each stage declares `deps`, `params`, `outs`, `metrics`, and `plots`. `params.yaml` contains all hyperparameters.

#### Assumptions That Break
- The DVC pipeline calls `train_pipeline.py --stage autoencoder` etc., but `train_pipeline.py` has no `--stage` CLI argument. **The DVC stages will fail to execute.**
- Stages 2-4 reference `--ae-checkpoint`, `--no-checkpoint` flags that don't exist in the argparse definition.
- DVC outputs include `./checkpoints/autoencoder_metrics.json` etc. — these files are never generated by the training code.
- DVC `outs` paths use relative `./` from the `dvc_pipeline/` directory but `cmd` uses `../training/` — this CWD dependency is fragile and breaks if DVC is invoked from the repo root.

#### What Fails First Under Load
DVC itself is not the bottleneck. The bottleneck is the training code it calls (which doesn't work). If the training code were implemented, DVC would add ~2–5s overhead per stage for hashing and metadata tracking — negligible.

---

### 1.4 Model Registry

#### What Is Implemented
`model_registry/registry_integration.py` implements `ModelRegistry` with version registration, stage promotion (dev → staging → canary → production → archived), quality gates (PSNR ≥ 26.0, SSIM ≥ 0.75, LPIPS ≤ 0.30), rollback, and hash verification. `canary_deploy.py` implements `CanaryDeployment` with progressive traffic shifting (10% → 30% → 50% → 100%) and auto-rollback on quality regression, plus a `ShadowDeployment` skeleton.

#### Assumptions That Break
- Model versions are stored **in-memory** (`self._versions: List[ModelVersion]`). A process restart loses all registry state. This is fundamentally incompatible with production use.
- The registry checks for duplicate checkpoint hashes in memory but not in MLflow — if MLflow already has the version, a duplicate is created on MLflow's side.
- `_mlflow_stage()` maps both "canary" and "staging" to MLflow's "Staging" — there is no distinction between canary and staging in the actual MLflow backend, which means querying "what is in canary?" via MLflow is impossible.
- Canary deployment `evaluate()` requires external metric ingestion — no integration with Prometheus, Grafana, or any real monitoring. The caller must manually construct `CanaryMetrics` objects.
- The 30-minute increment interval and 60-minute observation window are hardcoded defaults. At scale, these should be auto-adjusted based on traffic volume.

#### Hidden Coupling
- The registry assumes exactly three checkpoint files (`autoencoder_best.pth`, `neural_operator_best.pth`, `diffusion_unet_best.pth`). If model architecture changes, the registry code must be manually updated.
- Rollback finds "previous production version" by scanning `self._versions` in reverse for `stage == "archived" and version < current`. If multiple archived versions exist at different quality levels, this picks the most recent archived one — which may itself have been rolled back for quality issues.

---

### 1.5 Backend Inference

#### What Is Implemented
`backend/app.py`: FastAPI app with `/health`, `/ready`, `/infer`, `/metrics`, `/version` endpoints. Async inference with `asyncio.Semaphore(4)` for GPU concurrency control, `asyncio.timeout(30s)` per request, base64 image I/O, per-IP in-memory rate limiting, and backpressure (reject at `max_queue_depth=20` active requests). CORS enabled with wildcard `*`.

#### How It Works Technically
1. Client sends POST `/infer` with base64-encoded image.
2. Middleware increments `total_requests`, starts perf counter.
3. Backpressure check: if `active_requests >= 20`, return 503 with `Retry-After: 5`.
4. Rate limiter: per-IP per-hour counter in a dict. If > 100/hr, return 429.
5. Payload size estimation: `len(base64_string) * 3/4 ÷ 1e6`. If > 20 MB, return 413.
6. Decode base64 → PIL Image. If fails, return 422.
7. Increment `active_requests`, acquire semaphore (max 4).
8. `run_in_executor(None, _run_inference, ...)` — runs in **default ThreadPoolExecutor**.
9. Release semaphore, decrement `active_requests`.
10. Encode output to PNG → base64, return response with metadata.

#### Assumptions That Break

| Assumption | Reality |
|-----------|---------|
| `_run_inference` is placeholder `Image.BICUBIC` resize | **No actual model inference** — system returns bicubic upscale |
| `loop.run_in_executor(None, ...)` uses Python's default ThreadPoolExecutor | Default executor has only `min(32, os.cpu_count() + 4)` threads. On a 2-CPU pod, that's 6 threads — the semaphore allows 4 concurrent inferences, so this barely fits |
| `state.request_counts` dict grows unbounded | Rate limiting dict never cleans up old hour-keys. After 30 days of operation: ~720 keys per active IP. With 10k unique IPs: ~7.2M dict entries, ~500 MB memory leak |
| `active_requests` counter is not thread-safe | `state.active_requests += 1` is not atomic in Python. Under concurrent load, the counter can skip increments, making the backpressure check unreliable |
| Uvicorn runs with `--workers 1` | Single-process deployment means a single GIL and single event loop. No CPU parallelism. |

#### What Fails First Under 1000 Concurrent Users
1. **Rate limiter rejects most requests** (100/hr/IP). If 1000 users each send >100 requests/hour, all get 429s.
2. **Backpressure rejects at 20 active requests.** Even if rate limits pass, request 21 onward gets 503.
3. **Base64 encoding/decoding in Python is CPU-bound.** A 20 MB image takes ~200ms to base64-decode on Python. With 20 concurrent decodes, the event loop is saturated.
4. **No connection pooling on Uvicorn.** The single worker handles all connections; at 1000 concurrent, TCP accept queue overflows.

---

### 1.6 Queue System

#### What Is Implemented
`backend/inference_worker.py`: Redis-backed inference worker using `BRPOP`/`LPUSH` for job queuing, `SETEX` for result storage (1hr TTL), a dead-letter queue for jobs exceeding `max_retries=3`, and signal handling for graceful shutdown.

#### How It Works Technically
1. Worker calls `BRPOP hndsr:inference:queue 5` (5s timeout).
2. On first job arrival, enters a tight loop calling `LPOP` to collect up to `max_batch_size=4` additional jobs within `batch_timeout_s=0.5s`.
3. Processes each job **sequentially** (not as a true GPU batch).
4. Stores results in `hndsr:inference:result:{job_id}` with 1hr TTL.
5. On failure: increment `attempt`, re-push to queue if `attempt < 3`, else push to DLQ.

#### Critical Finding: Mini-Batching Is Not Batched
Despite the architecture claiming "mini-batching for GPU efficiency" and documenting "GPU utilization ~60-80%", the actual `_process_batch()` method processes jobs in a **sequential for-loop**:
```python
for job in jobs:
    # ... decode one image
    # ... run inference on one image
    # ... encode one result
```
There is no tensor stacking, no batched GPU forward pass. Each image is processed independently. The "batch" is merely an I/O optimization (fewer Redis round-trips), not a compute optimization. **GPU utilization remains ~20-30% — exactly what the docs claim batching solves.**

#### Assumptions That Break
- The FastAPI app does NOT enqueue jobs to Redis. The `/infer` endpoint runs inference directly via `run_in_executor`. **The queue worker is orphaned** — there is no producer. The sync and async paths described in `architecture.md` are not connected in the code.
- `redis.from_url(..., decode_responses=True)` decodes everything as UTF-8 strings. Large base64 image payloads (~27 MB for a 20 MB image) will be held in Python strings, not bytes. This doubles memory usage (Python strings are UCS-2/4 internally).
- Redis `maxmemory 256mb` (from `docker-compose.yml`) with `allkeys-lru` eviction means that under load, job results will be evicted before clients retrieve them.

---

### 1.7 Kubernetes Deployment

#### What Is Implemented
The `kubernetes/README.md` references `deployment.yaml`, `service.yaml`, `hpa.yaml`, `pdb.yaml` — **none of these files exist in the repository.** The README describes them, and `architecture.md` shows topology diagrams, but the actual YAML manifests are absent.

#### Impact
- **No Kubernetes deployment is possible.** There are no manifests to apply.
- The claimed HPA (min:2, max:8, target GPU:70%), PDB (`minAvailable: 1`), and GPU scheduling (`nvidia.com/gpu: 1`) are design documents, not implementations.
- All Kubernetes-related analysis in `architecture.md` is aspirational, not actual.

---

### 1.8 Monitoring

#### What Is Implemented
- `/metrics` endpoint in `app.py` exposes 5 metrics in Prometheus text format: `hndsr_requests_total`, `hndsr_errors_total`, `hndsr_active_requests`, `hndsr_latency_avg_ms`, `hndsr_gpu_memory_used_bytes`, `hndsr_uptime_seconds`.
- `docker-compose.yml` runs Prometheus (v2.48.0) and Grafana (v10.2.0).
- References to `observability/prometheus.yml`, `observability/alerting_rules.yml`, `observability/grafana_dashboard.json` — **none of these files exist.**

#### What Fails First
- Without `prometheus.yml`, Prometheus has no scrape targets. It will start but collect nothing.
- Without `alerting_rules.yml`, there are no alerts. All the alarm thresholds described in `architecture.md` (P95 > 5s, Error > 5%, GPU OOM, Circuit open) do not exist.
- Without the Grafana dashboard JSON, Grafana shows a blank default screen.

---

### 1.9 CI/CD

#### What Is Implemented
A single GitHub Actions workflow: `.github/workflows/code_quality.yml` that:
1. Checks out code
2. Installs Python 3.11 + `requirements.txt`
3. Runs `flake8` lint (errors-only + all-warnings)
4. Runs `pytest tests/`

#### What Is Missing
The README claims 5 workflows:
- `code_quality.yml` ✅ exists
- `model_validation.yml` ❌ does not exist
- `docker_build.yml` ❌ does not exist
- `deploy.yml` ❌ does not exist
- `dvc_validation.yml` ❌ does not exist

**4 out of 5 claimed CI/CD workflows are missing.** There is no Docker build, no ECR push, no staging/production deployment, no canary automation, no DVC validation.

---

## 2. Performance Feasibility Analysis

### 2.1 Expected Inference Latency

The HNDSR pipeline consists of three stages: Autoencoder encode, Neural Operator forward pass (with FFT), and DDIM diffusion denoising (50 steps).

**Realistic estimate per 256×256 tile on T4 GPU:**

| Stage | Operation | Estimated Time |
|-------|-----------|---------------|
| Autoencoder Encode | Conv-based encode, 64→128 channels | 5–10 ms |
| Neural Operator | 4-layer FNO with FFT at 16 modes | 10–20 ms |
| Diffusion (50 DDIM steps) | 50 × UNet forward pass | 500–1500 ms |
| Autoencoder Decode | ConvTranspose, 128→3 channels | 5–10 ms |
| **Total per tile** | | **520–1540 ms** |

For a full image requiring 16 tiles (1024×1024 input): **8–24 seconds** without batching.

**Key insight:** Diffusion dominates. The 50 DDIM steps each require a full UNet forward pass. Reducing DDIM steps to 20 would cut latency by ~60% with 1–2 dB quality loss. This tradeoff is not configurable per-request without significant refactoring.

**Metric to watch:** `hndsr_inference_latency_ms` P95 and P99. Currently, only the **average** is tracked — averages hide tail latency spikes.

### 2.2 GPU Utilization Behavior

**Current state:** Placeholder bicubic resize runs entirely on CPU. Even when the real model is integrated, the single-request sequential processing (no true batching in the worker) means:

- GPU is idle during: image decode (CPU), base64 encode (CPU), network I/O, PIL resize.
- GPU is busy during: model forward pass only.
- **Expected utilization: 20–35%** based on the ratio of GPU compute time to total request time.

True mini-batching (stacking multiple images into a single tensor batch) would increase utilization to 60–80% but **is not implemented** despite being claimed.

### 2.3 Queue Growth Dynamics

The queue system is disconnected from the API. If connected:

- **Arrival rate:** At P requests/second, queue grows at rate `P - D` where D is the drain rate.
- **Drain rate per worker:** `batch_size / (batch_timeout + inference_time)` ≈ `4 / (0.5 + 6)` ≈ 0.6 batches/s ≈ 2.4 requests/s.
- **At 10 req/s:** Queue grows at 7.6 req/s. Redis fills 256 MB in ~4 minutes with 20 MB image payloads.
- **Mitigation:** Pre-validate and reject oversized images before enqueueing. Compress/thumbnail before queue insertion.

### 2.4 Cold Start Latency

Model loading for a three-stage pipeline:
- Autoencoder: ~200 MB → 1–3s from SSD, 5–10s from network
- Neural Operator: ~100 MB → 0.5–2s
- Diffusion UNet: ~500 MB → 3–8s
- CUDA context initialization: 2–5s
- First inference warmup: 1–3s
- **Total cold start: 8–30 seconds**

Kubernetes health check `start_period` is 60s, which provides adequate buffer. However, HPA scale-up + scheduling + image pull + cold start = **3–7 minutes** before a new pod handles traffic. During this window, existing pods absorb all load.

### 2.5 Autoscaling Reaction Delay

**Sequence under traffic spike:**
1. T+0s: Traffic jumps from 5 to 50 req/s
2. T+15s: Prometheus scrapes new metric values
3. T+30s: HPA evaluates custom metric (GPU utilization)
4. T+60s: HPA decides to scale (after stabilization window)
5. T+90s: Kubernetes schedules new pod (GPU node may need provisioning)
6. T+120-300s: New node joins cluster (if node pool was empty)
7. T+300-420s: Pod starts, pulls image, loads model, warms up
8. T+420s+: New pod ready to serve

**Total: 2–7 minutes.** During this window, the existing 2 pods handle 50 req/s. With max 4 concurrent each, queue depth = 50 - 8 = 42 rejected requests per second. **~85% of requests fail with 503.**

### 2.6 IO vs Compute Bottlenecks

| Component | IO-bound or Compute-bound | Bottleneck |
|-----------|--------------------------|------------|
| Base64 decode | CPU-bound | Python GIL |
| Image resize (PIL) | CPU-bound | Python GIL |
| Tensor to GPU transfer | IO-bound (PCIe) | ~2 GB/s for T4 |
| Model forward pass | Compute-bound | GPU FLOPS |
| DDIM sampling loop | Compute-bound | Sequential GPU ops |
| Base64 encode response | CPU-bound | Python GIL |
| Redis BRPOP/LPUSH | IO-bound (network) | Redis single-threaded |

**Key finding:** The pipeline is GIL-limited. Base64 encoding a 4× upscaled 1024×1024 PNG image (12 MB+ raw) takes ~500ms in Python. With Uvicorn's single worker, this blocks the event loop. **Solution:** Use `io.BytesIO` in the executor thread (already done) but also move base64 encoding into the executor.

### 2.7 Retry Storm Risk

The worker retries failed jobs up to 3 times. If failure is systematic (GPU OOM, model corruption):
- 100 jobs arrive → all fail → 100 retries → all fail → 100 retries → all fail → 100 DLQ entries
- **Total GPU time wasted:** 300 failed inference attempts × 6s each = 30 min of GPU compute burned on guaranteed failures
- **Mitigation:** Implement circuit breaker in the worker. After N consecutive failures, pause for M seconds before retrying. The architecture diagram shows a circuit breaker but **no code implements it**.

---

## 3. Cost Feasibility & Economic Realism

### 3.1 GPU Hourly Cost

| GPU Type | Cloud Provider | On-Demand $/hr | Spot $/hr | Monthly (24/7) |
|----------|---------------|----------------|-----------|----------------|
| T4 (16 GB) | AWS (g4dn.xlarge) | $0.526 | $0.16 | $378 |
| A10 (24 GB) | AWS (g5.xlarge) | $1.006 | $0.30 | $724 |
| A100 (40 GB) | AWS (p4d.24xlarge) | $32.77 | $9.83 | $23,594 |

**HNDSR baseline (2 pods, T4, on-demand):** $1.05/hr = **$756/month**

**Peak (8 pods, T4, on-demand):** $4.21/hr = **$3,031/month**

**Average with autoscaling (estimate 3 pods avg):** $1.58/hr = **$1,134/month**

### 3.2 Autoscaling Waste Patterns

HPA scales based on GPU utilization. With real model inference:
- A burst of 100 requests triggers scale to 8 pods
- Burst clears in 5 minutes; HPA cooldown is typically 5 minutes
- 6 extra pods idle for 5 min = 0.5 pod-hours wasted = $0.26 per spike
- **At 20 spikes/day: $5.20/day = $156/month in waste**

The larger waste is **cold GPU idle time.** Even with 0 traffic, `minReplicas: 2` means 2 GPUs running 24/7: **$756/month floor cost.**

### 3.3 Idle GPU Inefficiency

The architecture deploys one GPU per pod running one model. But:
- Between requests, the GPU is 100% idle (no time-sharing)
- Kubernetes does not support GPU fractional scheduling natively
- NVIDIA MPS (Multi-Process Service) could allow sharing but adds complexity
- **NVIDIA Triton Inference Server** with dynamic batching would dramatically improve utilization but requires model format conversion (ONNX/TensorRT)

### 3.4 Data Storage Costs

| Storage Component | Estimated Size | Monthly Cost (S3 Standard) |
|-------------------|---------------|---------------------------|
| Raw HR images (10k tiles) | ~50 GB | $1.15 |
| Processed (3 scales × 3 splits) | ~200 GB | $4.60 |
| DVC cache | ~200 GB (mirrors processed) | $4.60 |
| Model checkpoints (10 versions) | ~8 GB | $0.18 |
| MLflow artifacts | ~5 GB | $0.12 |
| ECR Docker images (20 × 3 GB) | ~60 GB | $6.00 |
| **Total** | **~523 GB** | **~$16.65/month** |

Storage cost is negligible. **Compute dominates by 50×.**

### 3.5 Cost Explosion Scenarios

1. **Runaway HPO:** 20 Optuna trials × 8 hours each = 160 GPU-hours = $84 on T4. If someone sets `n_trials=200`, that's $840.
2. **Oversized diffusion timesteps:** Setting `timesteps=2000` (2× default) doubles training time and inference latency.
3. **Uncontrolled autoscaling:** Malicious load test triggers 8 pods for an extended period: $4.21/hr. A 24-hour attack costs $101 just in GPU.
4. **DVC artifact duplication:** Each `dvc repro` that changes parameters creates a new copy of all outputs. Without aggressive `dvc gc`, cache grows linearly with experiment count. At 100 experiments: 20 TB.

### 3.6 Cost-Conscious Mitigations

| Action | Savings | Effort |
|--------|---------|--------|
| Use spot instances for worker pods | 70% GPU cost reduction | Medium (need spot interruption handling) |
| Scale-to-zero with KEDA | Eliminate idle cost ($756/mo) | Medium |
| NVIDIA Triton for batching/sharing | 2-3× utilization improvement | High |
| TensorRT INT8 quantization | 2-4× inference speedup → fewer GPUs needed | Medium |
| DVC garbage collection cron job | Prevent storage explosion | Low |
| Enforce HPO trial limits in CI | Prevent runaway compute | Low |

---

## 4. Failure Mode & Stability Audit

### 4.1 Sudden Traffic Spike (5× normal)

**Scenario:** Normal: 10 req/s. Spike: 50 req/s.

**Step-by-step failure progression:**
1. T+0s: 50 req/s hit Nginx (if deployed — currently no Nginx config exists)
2. T+0.1s: Requests reach FastAPI. First 4 acquire semaphore, 16 wait in queue, remaining 30 get 503.
3. T+1s: 4 requests complete (assuming ~1s latency). Next 4 acquire semaphore. Queue: 16 → 12 + 50 new = 62 pending. 42 get 503.
4. T+5s: Sustained 503 rate of ~84%. Health check still passing (it doesn't check queue depth for liveness).
5. T+15s: Prometheus scrapes. `hndsr_active_requests` reported.
6. T+60s: HPA would react (if manifests existed).
7. T+420s: New pods ready (if K8s manifests existed).

**Collapse or graceful degradation?** GRACEFUL DEGRADATION — the 503 backpressure prevents OOM, but 84% failure rate is not acceptable. No retry mechanism exists in the API layer (503 with `Retry-After: 5` relies on clients obeying the header).

**First metric to change:** `hndsr_active_requests` hits `MAX_QUEUE_DEPTH`.

**Prevention:**
- Implement request queuing (Redis) so requests wait rather than fail
- Add priority tiers (paid users get guaranteed capacity)
- Pre-provision more pods or use KEDA for faster scaling

### 4.2 Heavy-Tailed Request Sizes

**Scenario:** Most images are 256×256 (100 KB), but 5% are 4096×4096 (20 MB).

**Progression:**
1. Large image arrives. Base64 decode: ~200ms. Tiling into 256×256 tiles: 256 tiles.
2. Each tile takes ~1s → total: ~256s for one request.
3. This request holds the semaphore for 256 seconds (timeout is 30s → request times out at 30s).
4. BUT: the `_run_inference` call in the executor thread **is not cancelled when asyncio.timeout fires**. The thread continues running, consuming GPU for 256s.
5. The semaphore is released when the thread eventually completes, not when the timeout occurs. **Bug:** `asyncio.timeout` cancels the awaiting coroutine, but the thread pool task continues running. The `finally: semaphore.release()` fires immediately on timeout, but the thread still holds the GPU.
6. Result: GPU is occupied by a "cancelled" request, and the semaphore is released prematurely, allowing a new request to start GPU work concurrently with the timed-out one.

**Impact:** Two concurrent GPU operations on a single T4 → OOM.

**Fix:** Implement cooperative cancellation in `_run_inference` using a `threading.Event` check between tiles.

### 4.3 Corrupt Image Uploads

**Current handling:** `base64.b64decode()` followed by `Image.open().convert("RGB")`. If either fails, HTTP 422.

**Edge cases NOT handled:**
- **Valid JPEG with EXIF rotation:** PIL opens it in raw orientation. Output will be rotated relative to input.
- **16-bit TIFF images:** `.convert("RGB")` truncates to 8-bit. Silent quality loss.
- **Animated GIF:** `Image.open()` returns first frame only. No error raised, but user expectation violated.
- **Zip bomb:** A 1 KB PNG that decompresses to 100 GB. **PIL has `MAX_IMAGE_PIXELS` but the code overrides it with `max_pixels=16000*16000 = 256M pixels` in ETL only. The API endpoint has NO pixel limit.**
- **Base64 with embedded newlines/whitespace:** `base64.b64decode()` is permissive and ignores whitespace by default, but some clients may send URL-safe base64 (`-_` instead of `+/`). This fails silently.

### 4.4 GPU OOM

**Scenario:** A 512×512 image at scale factor 8× requires generating a 4096×4096 output. With 128 latent channels and 50 DDIM steps:
- Latent tensor: 1×128×128×128 = 8 MB
- UNet intermediate activations: ~500 MB per forward pass
- 50 DDIM steps: ~500 MB peak (not cumulative — same memory reused)
- Autoencoder decode of 128→3 at 4096×4096: ~1 GB
- **Peak: ~2 GB** per request on top of model weights (~800 MB)

With 4 concurrent requests (semaphore limit): 4 × 2 GB + 0.8 GB model = **8.8 GB** — fits on T4 (16 GB) but leaves only 7.2 GB for CUDA overhead and fragmentation.

**When it OOMs:**
- Scale factor 8× with large input (>512×512)
- Multiple concurrent max-size requests
- CUDA memory fragmentation after many requests

**Detection:** `torch.cuda.OutOfMemoryError` → caught by generic `except Exception` → HTTP 500. No specific handling, no metric change visible (it's counted in `total_errors` but not distinguished from other errors).

**After OOM:** CUDA context may be corrupted. Subsequent requests may fail until pod restart. There is no `torch.cuda.empty_cache()` in the error handler.

### 4.5 Network Timeout

**Scenario:** Redis becomes unreachable for 10 seconds.

**Worker impact:** `BRPOP` with 5s timeout returns `None`. Worker loops, retries `BRPOP`. After Redis timeout, `redis.ConnectionError` raised → caught by `except Exception` → 1s backoff → retry.

**API impact:** The API does NOT use Redis, so API is unaffected. This is actually good isolation.

**If the API DID use Redis (as the architecture claims):** Every `/infer` request would fail for 10s. With retry-after headers, clients retry at T+5s — during the outage. At T+10s when Redis recovers, a wave of retried requests hits simultaneously → secondary overload.

### 4.6 Retry Amplification

**Scenario:** GPU OOM causes all inferences to fail. Worker retries each job 3 times.

**Progression:**
1. 100 jobs arrive during GPU-troubled period
2. All 100 fail on first attempt → 100 retries enqueued (attempt=1)
3. All 100 fail again → 100 retries (attempt=2)
4. All 100 fail again → 100 jobs go to DLQ
5. **Total GPU attempts: 300, all wasted.**
6. During this cascade (duration: 300 × 1s ≈ 5 min), new arriving jobs also fail and enter the retry cycle.

**Amplification factor:** 3× (max_retries). With exponential backoff instead of immediate retry, the burst would be spread over time.

**Missing:** No exponential backoff. No circuit breaker. No failure rate threshold that pauses processing.

---

## 5. Scalability Limits & Hard Constraints

### 5.1 GPU Scheduling in Kubernetes

- GPUs are non-preemptible, non-shareable resources in Kubernetes.
- A pod requesting `nvidia.com/gpu: 1` exclusively claims that GPU.
- If a node has 1 GPU and 1 pod is running, no more GPU pods can schedule on that node.
- **Node affinity requirement:** GPU pods must schedule on nodes in a GPU node pool. If the node pool has 8 nodes with 1 GPU each, maximum pods = 8.
- **Scheduling latency:** If all GPU nodes are occupied and a new node must be provisioned (cloud autoscaling), expect 2–5 minutes for a new node to join.

### 5.2 HPA Lag Behavior

HPA checks metrics every 15s (default). Decisions are made using a rolling window (default 5 min) to prevent flapping. Combined with pod startup time:

| Event | Elapsed Time |
|-------|-------------|
| Load spike begins | T+0 |
| Metric reflects spike | T+15s |
| HPA triggers scale-up | T+60-300s (depends on stabilization window) |
| Pod scheduled | T+60-600s (depends on GPU node availability) |
| Pod ready | T+120-600s (image pull + model load) |
| **Total to handle load** | **T+2 min to T+10 min** |

For burst-tolerant workloads, the minimum safe approach is over-provisioning: set `minReplicas` high enough to handle expected peaks without scaling.

### 5.3 Maximum Safe Throughput

Per pod (T4 GPU, 1 worker, ~1s/tile latency):
- 1 req/s sustained (256×256 → 1 tile requests)
- 0.25 req/s for 512×512 (4 tiles per request)
- With semaphore=4: ~4 req/s peak (queued, not parallel)

With 8 pods (max HPA): **~32 req/s** peak for small images, **~8 req/s** for medium images.

**Realistic throughput with DDIM 50 steps:**
- Per-request latency: 1–3s (small tiles), 10–30s (large images requiring many tiles)
- Per-pod throughput: 0.3–1.0 req/s
- 8-pod cluster: **2.4–8.0 req/s**

### 5.4 Queue Size vs Memory Tradeoff

Redis is configured with `maxmemory 256mb` and `allkeys-lru` eviction.

| Job Payload Size | Max Queue Depth | Time to Fill (at 10 req/s) |
|-----------------|----------------|---------------------------|
| 100 KB (small image) | ~2,560 jobs | ~256 seconds |
| 1 MB (medium image) | ~256 jobs | ~25 seconds |
| 20 MB (large image) | ~12 jobs | ~1.2 seconds |

**At 20 MB payloads, Redis fills in 1.2 seconds under moderate load.** LRU eviction will delete queued jobs and stored results, causing data loss.

**Fix:** Store images in S3/MinIO and enqueue only references (job ID + S3 key). This decouples payload size from queue capacity.

### 5.5 Vertical vs Horizontal Scaling Limits

| Scaling Dimension | Limit | Why |
|-------------------|-------|-----|
| Vertical (bigger GPU) | 80 GB (A100-80G) | Larger GPU allows bigger batches, reduces latency. Diminishing returns above batch_size=16 for diffusion |
| Horizontal (more pods) | ~20 pods practical | GPU node pools cost grows linearly. At 20 × $0.53/hr = $10.60/hr = $7,632/month |
| Model parallelism | Not applicable | HNDSR models are small enough for single GPU (<1 GB each) |
| Data parallelism | Not implemented | No distributed inference. Each pod is independent |

---

## 6. Security & Hardening Review

### 6.1 Container Security

**Implemented:**
- Multi-stage build separates build-time deps from runtime ✅
- Non-root user `hndsr` in production Dockerfile ✅
- No `--privileged` flag ✅

**Missing:**
| Issue | Attack Vector | Likelihood | Severity | Mitigation |
|-------|--------------|-----------|----------|------------|
| Base image `nvidia/cuda:12.1.1-runtime-ubuntu22.04` has known CVEs | Supply chain attack via base image vulnerability | Medium | High | Pin image digest, scan with Trivy/Snyk in CI |
| No `securityContext` in K8s manifests (manifests don't exist) | Container escape via writable root fs | Medium | High | `readOnlyRootFilesystem: true`, `allowPrivilegeEscalation: false` |
| `vim`, `htop`, `nvtop` in dev image | Reconnaissance tools available after container compromise | Low | Low | Acceptable for dev; ensure dev image never reaches prod |
| No `USER` in `Dockerfile.dev` | Dev container runs as root | High | Medium | Add `USER` to dev Dockerfile |

### 6.2 Dependency Vulnerabilities

`requirements.txt` pins nothing except minimum versions (`torch>=2.0.0`). This means:
- `pip install -r requirements.txt` installs latest versions, which may introduce breaking changes or vulnerabilities.
- No hash pinning — MITM supply-chain attacks can replace packages.
- `requirements-prod.txt` (referenced in Dockerfile) does not exist in the repository.

**Risk:** HIGH — a compromised PyPI package installed in production could exfiltrate model weights or training data.

**Mitigation:** Use `pip-compile` to generate locked `requirements.txt` with hashes. Scan with `pip-audit` in CI.

### 6.3 Model Artifact Integrity

**Implemented:** SHA-256 hashing of checkpoints in experiment tracker and registry ✅

**Missing:**
- No hash verification at model load time. The API loads from `MODEL_DIR` without checking any hash.
- No signed artifacts. Anyone with access to the checkpoint volume can replace `diffusion_unet_best.pth` with a backdoored model.
- `torch.load()` with default settings can execute arbitrary Python code via pickle deserialization. **If an attacker replaces a checkpoint, they get arbitrary code execution.**

**Severity:** CRITICAL

**Mitigation:** Always use `torch.load(path, weights_only=True)`. Verify checkpoint hash against registry before loading. Sign artifacts with GPG.

### 6.4 Input Validation

**Implemented:**
- Base64 decode validation ✅
- Payload size check ✅
- Scale factor range (2–8) ✅
- DDIM steps range (10–200) ✅

**Missing:**
| Vulnerability | Attack | Mitigation |
|-----------|--------|------------|
| No image dimension limit in API | Send 15999×15999 image (255M pixels) → PIL decompresses to ~768 MB → OOM | Add `MAX_IMAGE_PIXELS` check before `Image.open()` |
| No Content-Type validation | Send arbitrary binary as base64 → PIL `Image.open()` probes all formats → slow | Require `image/png` or `image/jpeg` header |
| Base64 decode of 20 MB payload allocates ~27 MB | Memory amplification | Already limited by `MAX_PAYLOAD_MB` but decode happens before limit check is fully effective |
| No request body size limit at FastAPI level | Uvicorn reads entire body into memory | Add `--limit-request-body` to uvicorn or use Nginx `client_max_body_size` |

### 6.5 Rate Limit Bypass

**Current rate limiter:** In-memory dict keyed on `req.client.host` (IP).

**Bypass methods:**
1. **Proxy rotation:** Using 100 different IPs → 100 × 100 = 10,000 req/hr
2. **IP spoofing behind load balancer:** If behind Nginx/LB, `req.client.host` is the proxy's IP, not the user's. All users share one rate limit. Conversely, adding `X-Forwarded-For` support enables spoofing.
3. **Clock wraparound:** The hour-key is `ip:int(now // 3600)`. Sending requests at hh:59:59 and hh+1:00:00 effectively doubles the limit to 200 requests in 2 seconds.

**Mitigation:** Use Redis-based distributed rate limiting with sliding window (e.g., `redis.incr` + `expire`). Use token bucket algorithm. Trust `X-Forwarded-For` only from known proxies.

### 6.6 CORS Configuration

```python
allow_origins=["*"]
allow_methods=["*"]
allow_headers=["*"]
allow_credentials=True
```

**`allow_origins=["*"]` with `allow_credentials=True` is an invalid combination** per CORS spec (browsers reject it). More critically, even if corrected:
- Any website can make requests to the API on behalf of any user
- Enables CSRF attacks if authentication is ever added
- Allows credential stuffing if rate limits are bypassable

**Severity:** Medium (no authentication currently, so limited impact, but blocks future security improvements)

**Fix:** Restrict `allow_origins` to specific domains. Remove `allow_credentials=True` if not using cookies.

### 6.7 Denial of Service

**Attack surface:**
1. **Image bomb:** Craft a PNG that's tiny on disk but decompresses to huge dimensions → server OOM. Currently no dimension limit in API.
2. **Slowloris:** Hold connections open by sending data very slowly. Uvicorn's single worker is blocked. Mitigation: Nginx (not configured yet) provides `client_header_timeout` and `client_body_timeout`.
3. **GPU exhaustion:** Send `scale_factor=8` with maximum image size → GPU OOM → pod restart → healthcheck fail → Kubernetes kills and restarts → cold start → more requests fail.
4. **Rate limit memory exhaustion:** Generate requests from millions of unique IPs → `state.request_counts` dict grows without bound → API process OOM.

### 6.8 Logging Exposure

**Risk:** The `/health` endpoint exposes GPU model name, VRAM usage, and memory details. The `/version` endpoint exposes Python version, Torch version, and CUDA version.

**Impact:** Attacker learns exact software stack → can target known CVEs.

**Likelihood:** Low (requires access to the API)

**Mitigation:** Restrict `/health` and `/version` to internal network (Kubernetes Service ClusterIP, not LoadBalancer).

---

## 7. Observability & Monitoring Audit

### 7.1 Metrics Analysis

**Currently collected (app.py `/metrics`):**

| Metric | Type | Adequate? | Issue |
|--------|------|-----------|-------|
| `hndsr_requests_total` | Counter | ✅ | Good, basic throughput |
| `hndsr_errors_total` | Counter | ⚠️ | Not labeled by error type (OOM vs timeout vs validation). Cannot distinguish failure modes. |
| `hndsr_active_requests` | Gauge | ✅ | Good for queue depth |
| `hndsr_latency_avg_ms` | Gauge | ❌ | **Averages are useless for latency.** A P99 of 30s with P50 of 100ms averages to ~500ms — hiding the 30s outliers. Need histograms. |
| `hndsr_gpu_memory_used_bytes` | Gauge | ✅ | Good, but only for GPU 0 |
| `hndsr_uptime_seconds` | Gauge | ✅ | Good for restart detection |

**Missing critical metrics:**

| Metric | Why Needed |
|--------|-----------|
| `hndsr_inference_latency_ms` histogram (P50/P95/P99) | SLA enforcement |
| `hndsr_inference_latency_ms` per-stage (encode/NO/diffusion/decode) | Bottleneck identification |
| `hndsr_queue_depth` (Redis queue length) | Queue growth detection |
| `hndsr_batch_size` histogram | Batching efficiency |
| `hndsr_gpu_utilization_pct` | HPA scaling signal |
| `hndsr_requests_by_status` (200/4xx/5xx) | Error rate decomposition |
| `hndsr_image_size_pixels` histogram | Request profile understanding |
| `hndsr_model_psnr_rolling` | Quality drift detection |
| `hndsr_circuit_breaker_state` | Operational awareness |
| `hndsr_rate_limited_total` | Rate limit effectiveness |
| `hndsr_retry_count` | Retry storm detection |

### 7.2 Tail Latency Measurement

**Not measured.** The `/metrics` endpoint exposes only the arithmetic mean of all latencies. This is the single most critical missing metric.

**Why it matters:** If P99 latency is 25s but average is 400ms, you won't know until users complain. Prometheus histograms with bucket boundaries `[100, 250, 500, 1000, 2500, 5000, 10000, 30000]` ms would enable `histogram_quantile(0.99, ...)` queries.

**Fix:** Replace the manual metrics with `prometheus_client` library (already in `requirements.txt` but unused):
```python
from prometheus_client import Histogram, Counter, Gauge
LATENCY = Histogram('hndsr_inference_seconds', 'Inference latency', buckets=[.1, .25, .5, 1, 2.5, 5, 10, 30])
```

### 7.3 GPU Starvation Detection

**Not detectable.** There is no metric for GPU utilization percentage. `hndsr_gpu_memory_used_bytes` shows VRAM used but not compute utilization.

**How to detect:** Use `nvidia-smi --query-gpu=utilization.gpu --format=csv` or NVIDIA DCGM exporter for Kubernetes. Expose as Prometheus metric.

### 7.4 Alert Definitions

**None exist.** `observability/alerting_rules.yml` is referenced but not created.

**Recommended alert definitions:**

#### Critical Alerts (page on-call)
| Alert | Condition | For | Severity |
|-------|-----------|-----|----------|
| HighErrorRate | `rate(hndsr_errors_total[5m]) / rate(hndsr_requests_total[5m]) > 0.05` | 2 min | Critical |
| InferenceLatencyP99High | `histogram_quantile(0.99, rate(hndsr_inference_seconds_bucket[5m])) > 10` | 5 min | Critical |
| GPUMemoryExhausted | `hndsr_gpu_memory_used_bytes / hndsr_gpu_memory_total_bytes > 0.95` | 1 min | Critical |
| PodCrashLooping | `rate(kube_pod_container_status_restarts_total[15m]) > 0` | 5 min | Critical |
| QueueDepthExplosion | `hndsr_queue_depth > 100` | 2 min | Critical |

#### Warning Alerts (Slack notification)
| Alert | Condition | For | Severity |
|-------|-----------|-----|----------|
| LatencyP95Elevated | `histogram_quantile(0.95, ...) > 5` | 10 min | Warning |
| GPUMemoryHigh | `gpu_memory_ratio > 0.80` | 5 min | Warning |
| RateLimitHigh | `rate(hndsr_rate_limited_total[5m]) > 10` | 5 min | Warning |
| LowThroughput | `rate(hndsr_requests_total[5m]) < 0.01` and uptime > 300s | 10 min | Warning |

#### SLA Definitions
| SLI | Target SLO | Measurement |
|-----|-----------|-------------|
| Availability | 99.9% (8.7 hr/year downtime) | `1 - (5xx_responses / total_responses)` |
| Latency P95 | < 5s for 256×256 input | `histogram_quantile(0.95, ...)` |
| Latency P99 | < 15s for 256×256 input | `histogram_quantile(0.99, ...)` |
| Error rate | < 1% (excluding 429s) | `rate(5xx) / rate(total)` |

---

## 8. CI/CD & GitHub Actions Audit

### 8.1 Current State

Only `.github/workflows/code_quality.yml` exists. It runs `flake8` and `pytest`.

#### Issues in existing workflow:

| Finding | Severity | Detail |
|---------|----------|--------|
| `flake8` with `--exit-zero` for all warnings | Medium | Warnings never block merge. Accumulated tech debt. |
| No `mypy` despite being in `requirements.txt` | Medium | Type errors undetected |
| No `black` formatting check | Low | Inconsistent formatting possible |
| Tests import `conftest` by name but `tests/` might not be in `sys.path` | Medium | `pytest tests/` should handle this, but `from conftest import THRESHOLDS` in test files uses implicit import |
| Tests import `from data_pipeline.etl_pipeline import ...` assuming package structure | Medium | Need `PYTHONPATH=.` or `pip install -e .` |
| No test coverage reporting | Low | Unknown code coverage |
| No caching of pip downloads between runs | Low | Slower CI (already uses `cache: 'pip'` in setup-python, which is good) |

### 8.2 Missing CI/CD Workflows

**4 of 5 claimed workflows are absent.** Here's what each should contain:

#### `model_validation.yml` (missing)
Should run: shape contract tests, inference consistency tests, benchmark tests (without GPU — skip GPU-only tests). Should gate merges.

#### `docker_build.yml` (missing)
Should: build Docker image, tag with git SHA, push to ECR, scan with Trivy for CVEs, fail on CRITICAL/HIGH vulnerabilities.

#### `deploy.yml` (missing)
Should: deploy to staging (auto), run smoke tests against staging, gate production with manual approval, execute canary deployment, monitor for 60 min, auto-rollback if error rate > 5%.

#### `dvc_validation.yml` (missing)
Should: on changes to `data_pipeline/` or `dvc_pipeline/`, verify `dvc status` shows no stale stages, `dvc repro --dry` confirms pipeline would succeed, check data hash consistency.

### 8.3 Code-Model Version Alignment

**Not enforced.** There is no mechanism to ensure that the Docker image in production contains the model version registered in MLflow. The `MODEL_VERSION` env var in `docker-compose.yml` is set to the static string `"1.0.0"` — it doesn't dynamically fetch from the registry.

**Fix:** The deploy workflow should:
1. Query MLflow for the current production model version
2. Pass it as a build arg into Docker
3. Embed the model version in the image label
4. Verify at startup that loaded model hash matches registry expectation

### 8.4 Reproducibility Enforcement

**Partially implemented.** DVC tracks data+params→checkpoints, and MLflow logs system info + git hash. However:
- `requirements.txt` is not locked — different pip installs yield different environments
- No `Pipfile.lock` or `poetry.lock`
- No Docker image digest pinning
- Base image `nvidia/cuda:12.1.1-runtime-ubuntu22.04` is a floating tag

### 8.5 Secrets Management

**Finding:** `storage_config.py` reads `HNDSR_STORAGE_S3_ACCESS_KEY` and `HNDSR_STORAGE_S3_SECRET_KEY` from environment variables or `.env` file. If `.env` is committed, credentials are leaked.

**Missing:** `.gitignore` entry for `.env` (not verified but likely absent). GitHub Actions secrets are not referenced in the workflow (no ECR login, no K8s kubeconfig).

### 8.6 Artifact Traceability

**Partial.** SHA-256 hashing of checkpoints exists. Git hash logged to MLflow. ECR tagging strategy documented (not implemented).

**Gap:** No end-to-end trace from "production model" → "MLflow run" → "git commit" → "training data version" → "DVC hash". Each link exists in isolation but the chain is not automated or queryable.

---

## 9. Top 5 Most Likely Production Failures

### #1: No Model Inference Implemented (Probability: 100%)

**Why:** Every inference code path is a `# TODO` placeholder returning `Image.BICUBIC` resize. The training pipeline is also a skeleton with `# TODO: Initialize actual model architectures` comments. **The system cannot produce any super-resolution output.**

**Early warning:** First real user tests return blurry bicubic output instead of SR output.

**Prevention:** Implement the actual HNDSR model architecture classes and integrate them into the loading/inference code paths.

---

### #2: Kubernetes Manifests Do Not Exist (Probability: 100%)

**Why:** `kubernetes/` directory contains only a README referencing files that were never created. No `deployment.yaml`, `service.yaml`, `hpa.yaml`, `pdb.yaml`. The system cannot be deployed to Kubernetes.

**Early warning:** `kubectl apply -f kubernetes/` fails with "no objects passed to apply."

**Prevention:** Create the Kubernetes manifests with proper GPU scheduling, resource limits, health probes, HPA configuration, and PDB settings.

---

### #3: Rate Limiter Memory Leak (Probability: 95% within 7 days)

**Why:** `state.request_counts` is a Python dict that grows by one key per unique `{IP}:{hour}` pair per hour. Keys are never cleaned up. At 100 unique IPs/hr: 2,400 keys/day. At 10,000 IPs/hr (moderate traffic): 240,000 keys/day = 1.68M keys/week. At ~200 bytes/key, that's ~336 MB/week. After 30 days: ~1.4 GB of dead rate limit entries.

**Early warning:** Process RSS memory grows linearly over time without corresponding traffic increase.

**Prevention:** Add a cleanup routine: `state.request_counts = {k: v for k, v in state.request_counts.items() if k.endswith(f":{current_hour}")}`
Or use Redis with TTL-based rate limiting.

---

### #4: API/Worker Queue Disconnection (Probability: 100%)

**Why:** The FastAPI `/infer` endpoint runs inference synchronously via `run_in_executor`. It never pushes jobs to Redis. The `inference_worker.py` reads from Redis `hndsr:inference:queue` which never receives jobs. The async path described in the architecture docs is not wired.

**Early warning:** Worker logs show "0 batches processed" with no jobs arriving.

**Prevention:** Either:
- (a) Route large requests from `/infer` to Redis queue + add a `/result/{job_id}` polling endpoint
- (b) Remove the worker module and rely on synchronous inference only
- (c) Build a separate `/infer-async` endpoint that uses the queue path

---

### #5: Observability Blindness (Probability: 100%)

**Why:** `prometheus.yml`, `alerting_rules.yml`, and `grafana_dashboard.json` do not exist despite being volume-mounted in `docker-compose.yml`. Prometheus starts but has no scrape targets. Grafana starts but has no dashboards. No alerts fire because no rules are defined.

**Early warning:** Grafana dashboard shows "No data" on all panels. Prometheus targets page is empty.

**Prevention:** Create the observability configuration files. At minimum: `prometheus.yml` with job targeting `hndsr-api:8000/metrics`, basic alert rules for error rate and latency, and a Grafana dashboard JSON with key panels.

---

## 10. Final Production Readiness Verdict

### Overall Risk Level: 🔴 HIGH

**This system is not ready for production deployment.**

### What Must Be Fixed Before Deployment

| Priority | Item | Effort | Impact |
|----------|------|--------|--------|
| **P0 (Blocking)** | Implement actual HNDSR model architecture & inference | ~2–4 weeks | System non-functional without this |
| **P0 (Blocking)** | Create Kubernetes manifests (deployment, service, HPA, PDB) | ~1–2 days | Cannot deploy to K8s |
| **P0 (Blocking)** | Create observability config files (prometheus.yml, alerting rules, dashboard) | ~1 day | Completely blind to failures |
| **P0 (Blocking)** | Fix rate limiter memory leak (add cleanup or use Redis) | ~2 hours | Process OOM within days |
| **P0 (Blocking)** | Wire API to worker queue (or remove worker) | ~1 day | Dead code / broken architecture |
| **P0 (Blocking)** | Create missing CI/CD workflows (Docker build, deploy, model validation, DVC) | ~2–3 days | No automated deployment |
| **P0 (Blocking)** | Add `requirements-prod.txt` (referenced in Dockerfile but missing) | ~1 hour | Docker build fails |
| **P1 (Before GA)** | Add image dimension limits to API endpoint | ~2 hours | DoS via decompression bombs |
| **P1 (Before GA)** | Implement true mini-batch GPU processing in worker | ~1–2 days | 2–3× throughput improvement |
| **P1 (Before GA)** | Replace average latency metric with histogram | ~2 hours | Can't measure SLA compliance |
| **P1 (Before GA)** | Lock dependency versions with hashes | ~2 hours | Supply chain security |
| **P1 (Before GA)** | Add `torch.load(..., weights_only=True)` | ~30 min | Arbitrary code execution risk |

### What Can Be Deferred

| Item | Why Deferrable |
|------|---------------|
| S3 cloud storage integration in ETL | Local filesystem sufficient for initial deployment |
| Distributed training support | Single-GPU training is adequate for current model size |
| NVIDIA Triton Inference Server | Direct PyTorch serving works, just less efficient |
| TensorRT optimization | Can ship FP32 initially, optimize later |
| Shadow deployments | Canary is sufficient for initial rollouts |
| Geographic stratification in data splits | Random split is acceptable with sufficient data volume |

### What Is Unnecessarily Complex

| Component | Why Over-Engineered |
|-----------|-------------------|
| Canary deployment with progressive traffic shifting | For a system that can't serve inference, a 300-line canary module with statistical comparison is premature |
| HPO configuration (120 lines of YAML) | Hyperparameters are undefined until the model exists |
| Three-tier shadow deployment pattern | No users exist yet to shadow-test against |
| ECR cleanup lifecycle policies | No ECR registry is configured |

### What Is Dangerously Underdeveloped

| Component | Risk Level | Why Dangerous |
|-----------|-----------|---------------|
| **Model inference** (100% placeholder) | 🔴 CRITICAL | The core functionality does not exist |
| **Kubernetes manifests** (0% implemented) | 🔴 CRITICAL | Cannot deploy |
| **Observability configs** (0% implemented) | 🔴 CRITICAL | Cannot monitor |
| **CI/CD** (20% implemented) | 🔴 HIGH | Cannot safely deploy changes |
| **Security hardening** (minimal) | 🟡 MEDIUM | Open CORS, no input dimension limit, unsigned artifacts |
| **Error handling** (generic catch-all) | 🟡 MEDIUM | GPU OOM not handled specifically, CUDA corruption not recovered |

### What Would Fail Under Real-World Load

Under 100 concurrent users (~10 req/s sustained):

1. **Rate limiter rejects 90%+** of repeat users after first 100 requests/hr
2. **Backpressure rejects at 20 active requests** — ~50% of non-rate-limited requests get 503
3. **No actual super-resolution** — users receive bicubic resize
4. **No monitoring alerts** — team is unaware of failures
5. **Memory leak** causes OOM within 1–2 weeks
6. **No autoscaling** — fixed 2 pods cannot handle peaks

### Honest Summary

The HNDSR project demonstrates strong **architectural thinking** and **documentation quality**. The architecture diagrams, tradeoff analyses (Flask vs FastAPI), and documented failure mode cascades show genuine understanding of production ML systems. The code that does exist (ETL, training skeleton, API structure, canary logic) follows good patterns (dataclasses, type hints, docstrings, separation of concerns).

However, the project is fundamentally a **design document implemented as scaffolding**, not a production system. The critical path — model architecture, model inference, Kubernetes manifests, observability configuration, and CI/CD pipelines — consists of placeholder code, missing files, and `# TODO` comments.

**To reach production readiness:** Focus 100% of effort on the P0 items. Defer all architectural elegance until the system can serve a single real inference request end-to-end. A working system with bicubic resize and basic monitoring is more production-ready than a non-working system with canary deployments and shadow testing.

---

*Audit conducted: 2026-02-22 | Auditor: Principal Engineer Review*
