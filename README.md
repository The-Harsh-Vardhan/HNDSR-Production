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
│   ├── app.py                         # FastAPI with real HNDSR inference
│   ├── model/                         # HNDSR model architecture
│   │   └── model_stubs.py             # Autoencoder + FNO + Diffusion UNet
│   └── inference/                     # Inference engine
│       ├── engine.py                  # DDIM scheduler + FP16 inference
│       ├── tile_processor.py          # Hann-window tile stitching
│       ├── model_loader.py            # Singleton loader with warm-start
│       └── generate_checkpoints.py    # Generate .pth files for validation
│
├── checkpoints/                       # Model weights (.pth)
│   ├── autoencoder_best.pth           # 5 MB, ~1.2M params
│   ├── neural_operator_best.pth       # 19 MB, ~4.6M params
│   └── diffusion_unet_best.pth        # 25 MB, ~6.1M params
│
├── frontend/                          # Upload UI
│   ├── index.html                     # Upload, controls, result display
│   ├── app.js                         # API integration + error handling
│   └── styles.css                     # Dark glassmorphism theme
│
├── docker/                            # Containerization
│   ├── Dockerfile                     # Multi-stage production image
│   ├── Dockerfile.dev                 # Development image
│   └── docker-compose.yml             # API + Frontend + Prometheus + Grafana
│
├── observability/                     # Monitoring stack
│   ├── prometheus.yml                 # Scrape config
│   ├── alerting_rules.yml             # 4 critical + 3 warning alerts
│   └── grafana_dashboard.json         # 7-panel dashboard
│
├── docs/                              # Documentation
│   ├── FULL_SYSTEM_REPORT.md          # 12-part system validation report
│   ├── PRODUCTION_MVP.md              # Simplified architecture
│   ├── INTERVIEW_DEFENSE.md           # Interview preparation
│   └── PRODUCTION_READINESS_AUDIT.md  # Full audit report
│
├── .github/workflows/                 # CI/CD
│   ├── code_quality.yml               # Lint + test + config validation
│   └── docker_build.yml               # Docker image build
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
