# HNDSR Production Architecture — Detailed Diagrams

---

## 1. System Topology

```
                    ┌─────────────────────────────┐
                    │       Internet / CDN         │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │     Nginx Reverse Proxy      │
                    │   • TLS termination          │
                    │   • Rate limiting (L7)       │
                    │   • Request buffering        │
                    └──────────────┬──────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                     │
    ┌─────────▼──────┐  ┌─────────▼──────┐  ┌─────────▼──────┐
    │  FastAPI Pod 1  │  │  FastAPI Pod 2  │  │  FastAPI Pod N  │
    │  (GPU: T4/A10)  │  │  (GPU: T4/A10)  │  │  (GPU: T4/A10)  │
    │                 │  │                 │  │                 │
    │  ┌───────────┐  │  │  ┌───────────┐  │  │  ┌───────────┐  │
    │  │Autoencoder│  │  │  │Autoencoder│  │  │  │Autoencoder│  │
    │  │Neural Op. │  │  │  │Neural Op. │  │  │  │Neural Op. │  │
    │  │Diff. UNet │  │  │  │Diff. UNet │  │  │  │Diff. UNet │  │
    │  └───────────┘  │  │  └───────────┘  │  │  └───────────┘  │
    └────────┬────────┘  └────────┬────────┘  └────────┬────────┘
             │                    │                     │
             └────────────────────┼─────────────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │       Redis Cluster        │
                    │   • Job queue (FIFO)       │
                    │   • Result store (TTL)     │
                    │   • Dead-letter queue      │
                    └─────────────┬─────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                    │
    ┌─────────▼──────┐  ┌────────▼───────┐  ┌────────▼───────┐
    │  GPU Worker 1   │  │  GPU Worker 2   │  │  GPU Worker N   │
    │  (Async Queue)  │  │  (Async Queue)  │  │  (Async Queue)  │
    └────────────────┘  └────────────────┘  └────────────────┘
```

---

## 2. Inference Request Flow

```
Client Request
      │
      ▼
┌─────────────┐     ┌──────────────┐     ┌────────────────┐
│  Nginx      │────▶│  Rate        │────▶│  Payload       │
│  Ingress    │     │  Limiter     │     │  Validator     │
└─────────────┘     │  (Token      │     │  • Size check  │
                    │   Bucket)    │     │  • Format      │
                    └──────────────┘     │  • Dimensions  │
                                         └───────┬────────┘
                                                  │
                              ┌────────────────────┤
                              │                    │
                         Sync Path            Async Path
                         (≤4 tiles)           (>4 tiles)
                              │                    │
                    ┌─────────▼──────┐   ┌─────────▼──────┐
                    │ Queue Depth    │   │ Redis Queue    │
                    │ Check          │   │ LPUSH job      │
                    │ (reject >20)   │   │ Return job_id  │
                    └─────────┬──────┘   └─────────┬──────┘
                              │                    │
                    ┌─────────▼──────┐   ┌─────────▼──────┐
                    │ Semaphore      │   │ Worker BRPOP   │
                    │ Acquire        │   │ Process job    │
                    │ (max=4)        │   │ Store result   │
                    └─────────┬──────┘   └────────────────┘
                              │
                    ┌─────────▼──────────────────────────────┐
                    │           GPU Inference Pipeline        │
                    │                                         │
                    │  ┌─────────┐  ┌──────────┐  ┌────────┐ │
                    │  │  Tile   │─▶│  Stage 1  │─▶│Stage 2 │ │
                    │  │Splitter │  │Autoencoder│  │Neural  │ │
                    │  │(256×256)│  │  Encode   │  │Operator│ │
                    │  └─────────┘  └──────────┘  └───┬────┘ │
                    │                                  │      │
                    │  ┌─────────┐  ┌──────────┐  ┌───▼────┐ │
                    │  │  Tile   │◀─│  Stage 1  │◀─│Stage 3 │ │
                    │  │Stitcher │  │Autoencoder│  │DDIM    │ │
                    │  │(Hann)   │  │  Decode   │  │Denoise │ │
                    │  └────┬────┘  └──────────┘  └────────┘ │
                    │       │                                  │
                    └───────┼──────────────────────────────────┘
                            │
                    ┌───────▼────────┐
                    │  Output        │
                    │  Validation    │
                    │  • NaN check   │
                    │  • Range check │
                    │  • Sharpness   │
                    └───────┬────────┘
                            │
                    ┌───────▼────────┐
                    │  Base64 Encode │
                    │  + Response    │
                    └────────────────┘
```

---

## 3. Training Pipeline DAG (DVC)

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Raw HR     │     │  Preprocess  │     │  Autoencoder │
│   Images     │────▶│  • Downsample│────▶│  Training    │
│   (S3/MinIO) │     │  • Split     │     │  (~20 epochs)│
└──────────────┘     │  • Metadata  │     │  L1 loss     │
                     └──────────────┘     └──────┬───────┘
                                                  │
                                     Freeze encoder weights
                                                  │
                                         ┌────────▼───────┐
                                         │ Neural Operator │
                                         │ Training        │
                                         │ (~15 epochs)    │
                                         │ MSE loss        │
                                         └────────┬───────┘
                                                  │
                                     Freeze NO weights
                                                  │
                                         ┌────────▼───────┐
                                         │ Diffusion UNet  │
                                         │ Training        │
                                         │ (~30 epochs)    │
                                         │ Noise pred loss │
                                         └────────┬───────┘
                                                  │
                                         ┌────────▼───────┐
                                         │   Evaluation    │
                                         │   • PSNR        │
                                         │   • SSIM        │
                                         │   • LPIPS       │
                                         │   → MLflow log  │
                                         └────────────────┘

  DVC tracks: data hashes ← params.yaml ← model checkpoints ← metrics
```

---

## 4. CI/CD Pipeline Flow

```
┌──────────┐
│  git push│
│  to main │
└────┬─────┘
     │
     ▼
┌────────────────────────────────────────────────────────────────────┐
│  GitHub Actions                                                     │
│                                                                     │
│  ┌─────────────────┐                                               │
│  │ code_quality.yml │                                               │
│  │ • ruff lint      │                                               │
│  │ • pytest         │────┐                                          │
│  │ • mypy           │    │                                          │
│  └─────────────────┘    │                                          │
│                          │  All pass                                │
│  ┌─────────────────┐    │                                          │
│  │model_validation │    │                                          │
│  │ • Shape tests   │────┤                                          │
│  │ • Config check  │    │                                          │
│  └─────────────────┘    │                                          │
│                          ▼                                          │
│                 ┌────────────────┐    ┌───────────────────┐        │
│                 │docker_build.yml│───▶│   deploy.yml       │        │
│                 │ • Build image  │    │   • Staging auto   │        │
│                 │ • Tag :sha     │    │   • Prod manual ✋  │        │
│                 │ • Push ECR     │    │   • Canary 10%     │        │
│                 └────────────────┘    │   • Full rollout   │        │
│                                       └───────────────────┘        │
│  ┌─────────────────┐                                               │
│  │dvc_validation   │  (on data/ changes)                           │
│  │ • Hash check    │                                               │
│  │ • Artifact repro│                                               │
│  └─────────────────┘                                               │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

---

## 5. Deployment Topology (Kubernetes)

```
┌─────────────────────── Kubernetes Cluster ──────────────────────────┐
│                                                                      │
│  Namespace: hndsr-prod                                              │
│                                                                      │
│  ┌───────────────────────────────────────────────────┐              │
│  │  Deployment: hndsr-api (replicas: 2–8)            │              │
│  │  ┌─────────────────┐  ┌─────────────────┐        │              │
│  │  │  Pod (GPU: T4)   │  │  Pod (GPU: T4)   │  ...  │              │
│  │  │  ┌─────────────┐ │  │  ┌─────────────┐ │       │              │
│  │  │  │  FastAPI     │ │  │  │  FastAPI     │ │       │              │
│  │  │  │  + Models    │ │  │  │  + Models    │ │       │              │
│  │  │  └─────────────┘ │  │  └─────────────┘ │       │              │
│  │  │  Resources:      │  │  Resources:      │       │              │
│  │  │   GPU: 1         │  │   GPU: 1         │       │              │
│  │  │   Mem: 8Gi       │  │   Mem: 8Gi       │       │              │
│  │  │   CPU: 2         │  │   CPU: 2         │       │              │
│  │  └─────────────────┘  └─────────────────┘        │              │
│  └───────────────────────────────────────────────────┘              │
│                          │                                           │
│  ┌───────────────────────▼───────────────────────────┐              │
│  │  Service: hndsr-lb (LoadBalancer)                  │              │
│  │  Port: 80 → 8000                                  │              │
│  └───────────────────────────────────────────────────┘              │
│                                                                      │
│  ┌──────────────────┐  ┌──────────────────┐                        │
│  │  HPA             │  │  PDB             │                        │
│  │  min: 2          │  │  minAvailable: 1 │                        │
│  │  max: 8          │  │                  │                        │
│  │  target GPU: 70% │  │                  │                        │
│  └──────────────────┘  └──────────────────┘                        │
│                                                                      │
│  ┌─────────────────────────────────────────┐                        │
│  │  StatefulSet: redis (replicas: 1)       │                        │
│  │  PVC: 10Gi                              │                        │
│  └─────────────────────────────────────────┘                        │
│                                                                      │
│  ┌─────────────────────────────────────────┐                        │
│  │  DaemonSet: nvidia-device-plugin        │                        │
│  │  (GPU nodes only)                       │                        │
│  └─────────────────────────────────────────┘                        │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 6. Failure Mode Cascade

```
Normal Operation
      │
      ▼
┌─────────────┐  GPU OOM or     ┌──────────────┐  3 consecutive   ┌──────────────┐
│   CLOSED    │  driver crash   │   COUNTING    │  failures        │    OPEN      │
│  (healthy)  │────────────────▶│  (degrading)  │─────────────────▶│  (rejecting) │
└─────────────┘                 └──────────────┘                  └──────┬───────┘
      ▲                                                                  │
      │                           30s timeout                            │
      │                                                                  │
      │                         ┌──────────────┐                         │
      │     1 success           │  HALF_OPEN   │◀────────────────────────┘
      └─────────────────────────│  (probing)   │
                                └──────────────┘
                                  │
                                  │  1 failure
                                  └──────────────▶ back to OPEN

Cascade path:
  GPU OOM → Circuit Open → 503 responses → LB reroutes → Other pods overloaded
  → HPA scales up (2–5 min lag) → New pods cold start (60s model load)
  → Total recovery time: 3–7 minutes
```

---

## 7. Data Flow Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Data Lake (S3 / MinIO)                        │
│                                                                      │
│  raw/                    processed/              models/             │
│  ├── sentinel2/          ├── train/              ├── v1.0.0/         │
│  │   ├── 2024/           │   ├── HR/             │   ├── autoenc.pth │
│  │   │   ├── tile_001.   │   ├── LR_2x/         │   ├── neural.pth  │
│  │   │   ├── tile_002.   │   ├── LR_4x/         │   └── diff.pth    │
│  │   │   └── ...         │   └── metadata.parq   ├── v1.1.0/         │
│  │   └── 2025/           ├── val/                │   └── ...         │
│  └── metadata/           └── test/               └── staging/        │
│      └── catalog.parq                                                │
│                                                                      │
│  DVC tracks: SHA-256 hash of every file + directory                 │
│  Parquet stores: filename, dimensions, sensor, date, split, hash    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 8. Monitoring & Alerting Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│  HNDSR API Pod                                                     │
│                                                                     │
│  GET /metrics ──────────────────────────┐                          │
│    • hndsr_inference_latency_ms          │                          │
│    • hndsr_requests_total                │                          │
│    • hndsr_errors_total                  │                          │
│    • hndsr_gpu_vram_used_bytes           │  Prometheus scrape       │
│    • hndsr_queue_depth                   │  (every 15s)             │
│    • hndsr_psnr_rolling_mean             │                          │
│    • hndsr_circuit_breaker_state         │                          │
└──────────────────────────────────────────┼─────────────────────────┘
                                           │
                              ┌────────────▼────────────┐
                              │      Prometheus         │
                              │   • 15-day retention    │
                              │   • Alert rules         │
                              └────────────┬────────────┘
                                           │
                    ┌──────────────────────┼───────────────────────┐
                    │                      │                        │
          ┌─────────▼──────┐    ┌──────────▼───────┐    ┌──────────▼───────┐
          │  Grafana        │    │  Alertmanager     │    │  PagerDuty /     │
          │  Dashboard      │    │  • P95 > 5s      │    │  Slack           │
          │  • Latency      │    │  • Error > 5%    │    │  Notifications   │
          │  • Throughput   │    │  • GPU OOM       │    │                  │
          │  • GPU util     │    │  • Circuit open  │    │                  │
          │  • Queue depth  │    │  • Drift detect  │    │                  │
          └────────────────┘    └──────────────────┘    └──────────────────┘
```

---

**Each diagram above maps to a specific section in the codebase. See the corresponding README.md in each folder for detailed implementation notes.**
