# HNDSR in Production — A Complete ML System Architecture

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-24.0+-blue.svg)](https://docker.com)
[![Kubernetes](https://img.shields.io/badge/K8s-1.28+-purple.svg)](https://kubernetes.io)

> **HNDSR** (Hybrid Neural Operator–Diffusion Model for Continuous-Scale Satellite Image Super-Resolution) repackaged as a production-grade, scalable ML system.

---

## 📐 System Architecture (High Level)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           HNDSR Production System                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐    ┌──────────┐    ┌───────────┐    ┌──────────────────────┐  │
│  │ Frontend │───▶│  Nginx   │───▶│ FastAPI   │───▶│  GPU Inference       │  │
│  │ (Upload) │    │ (Proxy)  │    │ (API)     │    │  Engine              │  │
│  └──────────┘    └──────────┘    └─────┬─────┘    │  ┌────────────────┐  │  │
│                                        │          │  │ Autoencoder    │  │  │
│                                        │          │  │ Neural Operator│  │  │
│                                        │          │  │ Diffusion UNet │  │  │
│                                        │          │  └────────────────┘  │  │
│                                        │          └──────────────────────┘  │
│                                        │                                    │
│  ┌──────────┐    ┌──────────┐    ┌─────▼─────┐    ┌──────────────────────┐  │
│  │Prometheus│◀───│ /metrics │    │   Redis   │───▶│  Queue Workers       │  │
│  │ Grafana  │    │ endpoint │    │  (Queue)  │    │  (Horizontal Scale)  │  │
│  └──────────┘    └──────────┘    └───────────┘    └──────────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  ML Pipeline: DVC → MLflow → Model Registry → Canary Deploy        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  CI/CD: GitHub Actions → Lint/Test → Docker Build → K8s Deploy     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Folder Structure

```
HNDSR in Production/
├── README.md                          # This file
├── architecture.md                    # Detailed architecture diagrams
│
├── data_pipeline/                     # 🗄️  Layer 1: Data & ETL
│   ├── README.md                      # What/Why/How for data pipeline
│   ├── etl_pipeline.py                # HR → LR downsampling, splits, hashing
│   └── storage_config.py              # S3/MinIO config, Parquet schemas
│
├── training/                          # 🧠  Layer 1: Experiment Tracking & HPO
│   ├── README.md                      # What/Why/How for training
│   ├── experiment_tracking.py         # MLflow integration wrapper
│   ├── hpo_config.yaml                # Hyperparameter sweep definitions
│   └── train_pipeline.py              # 3-stage sequential training script
│
├── dvc_pipeline/                      # 🔁  Layer 1: Reproducible Pipeline
│   ├── README.md                      # What/Why/How for DVC
│   ├── dvc.yaml                       # 5-stage DVC pipeline
│   └── params.yaml                    # Versioned hyperparameters
│
├── model_registry/                    # 📦  Layer 1: Model Registry
│   ├── README.md                      # What/Why/How for registry
│   ├── registry_integration.py        # DVC → MLflow Registry connector
│   └── canary_deploy.py               # Canary + shadow deployment logic
│
├── tests/                             # 🧪  Layer 1: Model Testing
│   ├── conftest.py                    # Shared fixtures & thresholds
│   ├── test_preprocessing.py          # Data preprocessing unit tests
│   ├── test_shape_validation.py       # Shape contract tests
│   ├── test_inference_consistency.py  # Reproducibility tests
│   └── test_benchmarks.py             # Latency/memory profiling tests
│
├── backend/                           # ⚙️  Layer 2: Serving
│   ├── README.md                      # What/Why/How for backend
│   ├── api_comparison.md              # Flask vs FastAPI analysis
│   ├── app.py                         # Production FastAPI application
│   └── inference_worker.py            # Mini-batch GPU inference architecture
│
├── docker/                            # 🐳  Layer 2: Containers
│   ├── Dockerfile                     # Multi-stage production image
│   ├── Dockerfile.dev                 # Development image
│   ├── docker-compose.yml             # Full-stack compose
│   └── registry_strategy.md           # ECR tagging & rollback
│
├── kubernetes/                        # ☸️  Layer 2: Orchestration
│   ├── README.md                      # What/Why/How for K8s
│   ├── deployment.yaml                # GPU-scheduled deployment
│   ├── hpa.yaml                       # Horizontal Pod Autoscaler
│   ├── pdb.yaml                       # PodDisruptionBudget
│   └── service.yaml                   # LoadBalancer + probes
│
├── frontend/                          # 🖥️  Layer 3: User Interface
│   ├── index.html                     # Upload & display interface
│   ├── app.js                         # API integration & error handling
│   └── styles.css                     # Modern responsive design
│
├── observability/                     # 📊  Layer 4: Monitoring
│   ├── README.md                      # What/Why/How for observability
│   ├── prometheus.yml                 # Scrape config
│   ├── grafana_dashboard.json         # Pre-built dashboard
│   └── alerting_rules.yml             # Alert thresholds & SLAs
│
├── .github/workflows/                 # 🔁  Layer 5: CI/CD
│   ├── code_quality.yml               # Lint + test + static analysis
│   ├── model_validation.yml           # Shape contracts + inference tests
│   ├── docker_build.yml               # Docker build + ECR push
│   ├── deploy.yml                     # Staging → production pipeline
│   └── dvc_validation.yml             # Data hash + artifact checks
│
├── ci_cd/                             # 📘  CI/CD Documentation
│   └── README.md                      # What/Why/How for CI/CD in ML
│
├── performance/                       # 📈  Layer 6: Performance
│   ├── README.md                      # What/Why/How for perf engineering
│   ├── locustfile.py                  # Load testing script
│   └── benchmark.py                   # Latency/memory/GPU profiling
│
└── docs/                              # ⚠️  Layer 7: Risks & Limitations
    ├── limitations_and_risks.md       # Per-layer risk analysis
    ├── production_readiness_checklist.md # Pre-launch checklist
    └── tradeoffs.md                   # Architectural tradeoff analysis
```

---

## 🚀 Quick Start

### Development
```bash
# Start the full stack locally
cd docker/
docker compose up --build

# Access:
#   API:        http://localhost:8000
#   Frontend:   http://localhost:3000
#   Grafana:    http://localhost:3001
#   Prometheus: http://localhost:9090
```

### Training Pipeline
```bash
# Initialize DVC and run the full pipeline
cd dvc_pipeline/
dvc repro           # Runs all 5 stages
dvc push            # Push artifacts to remote storage
```

### CI/CD
Push to `main` branch triggers:
1. Code quality checks (lint, test, type check)
2. Model validation (shape contracts, inference tests)
3. Docker build + push to ECR
4. Auto-deploy to staging
5. Manual approval → production canary rollout

---

## 🏗️ Layer-by-Layer What / Why / How

| Layer | What | Why | How |
|-------|------|-----|-----|
| **Data Pipeline** | ETL from raw HR → train/val/test | Reproducibility, drift detection | S3 + Parquet + SHA-256 hashing |
| **Experiment Tracking** | Log all training runs | Compare architectures, prevent waste | MLflow + Optuna sweeps |
| **DVC Pipeline** | Reproducible training DAG | Re-run from scratch deterministically | `dvc.yaml` + `params.yaml` |
| **Model Registry** | Version + promote models | Safe rollback, canary testing | MLflow Registry + semantic versioning |
| **Backend API** | Serve HNDSR inference | Low-latency, high-throughput serving | FastAPI + Redis queue + GPU workers |
| **Docker** | Containerized deployment | Reproducible environments | Multi-stage CUDA images |
| **Kubernetes** | Orchestrated scaling | Handle traffic spikes, zero-downtime | HPA + PDB + GPU scheduling |
| **Frontend** | User-facing upload/display | Demonstrate the system | HTML/CSS/JS SPA |
| **Observability** | Real-time monitoring | Detect drift, latency spikes | Prometheus + Grafana |
| **CI/CD** | Automated quality gates | Prevent regressions, automate deploy | GitHub Actions × 5 workflows |
| **Performance** | Load & stress testing | Find bottlenecks before production | Locust + custom benchmarks |
| **Risk Analysis** | Failure mode documentation | Proactive incident prevention | Per-layer risk matrices |

---

## 📊 Architecture Details

See [architecture.md](architecture.md) for detailed diagrams of:
- Inference request flow
- Training pipeline DAG
- CI/CD pipeline
- Deployment topology
- Failure mode cascade

---

**Built for HNDSR v1.0.0 | Last Updated: February 2026**
