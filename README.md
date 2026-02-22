# HNDSR in Production — ML Inference System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-24.0+-blue.svg)](https://docker.com)

> **HNDSR** (Hybrid Neural Operator–Diffusion Model for Continuous-Scale Satellite Image Super-Resolution) deployed as a containerized GPU inference system with Prometheus monitoring.

**Status:** MVP — Inference infrastructure operational, model integration pending.
See [docs/PRODUCTION_MVP.md](docs/PRODUCTION_MVP.md) for honest system state.

---

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│              Docker Compose Stack                    │
│                                                     │
│  ┌───────────────────────────────────────────────┐  │
│  │  hndsr-api (FastAPI + Uvicorn, GPU)           │  │
│  │  /health /ready /infer /metrics /version      │  │
│  │  Semaphore(4) + backpressure(20) + rate limit │  │
│  └──────────────┬────────────────────────────────┘  │
│                 │ scrape every 15s                   │
│  ┌──────────────▼────────────────────────────────┐  │
│  │  Prometheus → Alerting Rules (7 alerts)       │  │
│  └──────────────┬────────────────────────────────┘  │
│                 │                                    │
│  ┌──────────────▼────────────────────────────────┐  │
│  │  Grafana Dashboard (7 panels)                 │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

---

## Folder Structure

```
HNDSR in Production/
├── backend/                           # Inference API
│   ├── app.py                         # FastAPI with GPU inference, monitoring
│   └── api_comparison.md              # Flask vs FastAPI analysis
│
├── model_registry/                    # Model lifecycle management
│   └── registry_integration.py        # DVC → MLflow connector + quality gates
│
├── docker/                            # Containerization
│   ├── Dockerfile                     # Multi-stage production image
│   ├── Dockerfile.dev                 # Development image
│   └── docker-compose.yml             # API + Prometheus + Grafana
│
├── observability/                     # Monitoring stack
│   ├── prometheus.yml                 # Scrape config
│   ├── alerting_rules.yml             # 4 critical + 3 warning alerts
│   └── grafana_dashboard.json         # 7-panel dashboard
│
├── tests/                             # Test suite
│   ├── conftest.py                    # Fixtures & acceptance thresholds
│   ├── test_preprocessing.py          # Data pipeline tests
│   ├── test_shape_validation.py       # Shape contract tests
│   ├── test_inference_consistency.py  # Reproducibility tests
│   └── test_benchmarks.py             # Performance tests
│
├── data_pipeline/                     # ETL pipeline
│   ├── etl_pipeline.py                # HR→LR downsampling, splits, hashing
│   └── storage_config.py              # Storage configuration
│
├── training/                          # Training pipeline
│   ├── train_pipeline.py              # 3-stage sequential trainer
│   ├── experiment_tracking.py         # MLflow integration
│   └── hpo_config.yaml                # Hyperparameter search config
│
├── dvc_pipeline/                      # Reproducible pipeline (DVC)
│
├── .github/workflows/                 # CI/CD
│   ├── code_quality.yml               # Lint + test + config validation
│   └── docker_build.yml               # Docker image build
│
├── docs/                              # Documentation
│   ├── PRODUCTION_MVP.md              # Current system architecture (honest)
│   ├── INTERVIEW_DEFENSE.md           # Interview preparation
│   └── PRODUCTION_READINESS_AUDIT.md  # Full audit report
│
├── requirements.txt                   # Dev dependencies (pinned)
├── requirements-prod.txt              # Production dependencies (pinned)
└── architecture.md                    # Original architecture design
```

---

## Quick Start

```bash
# Start the stack (requires NVIDIA GPU + Container Toolkit)
cd docker/
docker compose up -d

# Verify
curl http://localhost:8000/health

# View metrics
curl http://localhost:8000/metrics

# Grafana dashboard: http://localhost:3001 (admin / hndsr-admin)
# Prometheus:        http://localhost:9090
```

### CPU-only testing
Set `DEVICE=cpu` in `docker/docker-compose.yml` and remove the GPU reservation.

---

## CI/CD

Push to `main` triggers:
1. **Code quality** — flake8 lint + pytest + YAML/JSON config validation
2. **Docker build** — build image, tagged with git SHA (on backend/docker changes)

---

## Documentation

| Document | Purpose |
|----------|---------|
| [PRODUCTION_MVP.md](docs/PRODUCTION_MVP.md) | Current simplified architecture, request flow, failure handling, known limitations |
| [INTERVIEW_DEFENSE.md](docs/INTERVIEW_DEFENSE.md) | 5-minute system explanation + 5 tough Q&A |
| [PRODUCTION_READINESS_AUDIT.md](docs/PRODUCTION_READINESS_AUDIT.md) | Full audit report (pre-simplification) |

---

**Built for HNDSR v1.0.0 | Last Updated: February 2026**
