# DVC Pipeline — Reproducible Training

## What
A 5-stage DVC (Data Version Control) pipeline that makes the entire HNDSR training process deterministic and reproducible from raw data to final evaluation metrics.

## Why
- **"Worked on my machine"**: Without DVC, recreating a training result requires manually remembering which data, code, and parameters were used
- **Pipeline DAG**: DVC creates a dependency graph — changing `params.yaml` automatically reruns only the affected stages
- **Artifact tracking**: Every checkpoint and evaluation result is hashed and tracked, enabling rollback to any prior state
- **CI/CD integration**: The pipeline can be validated in CI — ensuring that parameter changes don't break downstream stages

### Comparison: With vs Without DVC
| Without DVC | With DVC |
|---|---|
| "I think I used lr=1e-4 last time" | `git log params.yaml` shows exact parameters |
| "The data might have changed" | SHA-256 hash mismatch → pipeline won't run |
| "Let me retrain from scratch" | `dvc repro` only reruns changed stages |
| "Where's that checkpoint from March?" | `dvc pull` fetches exact artifact |
| "Did someone modify the test split?" | DVC lock file catches any change |

## How

### Pipeline Stages
```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ preprocess   │────▶│ train_ae     │────▶│ train_no     │
│ (etl_pipeline│     │ (autoencoder)│     │ (neural op.) │
│  .py)        │     │              │     │              │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                     ┌──────────────┐     ┌───────▼──────┐
                     │  evaluate    │◀────│ train_diff   │
                     │  (PSNR/SSIM/ │     │ (diffusion   │
                     │   LPIPS)     │     │  UNet)       │
                     └──────────────┘     └──────────────┘
```

### Usage
```bash
# Initialize DVC
dvc init

# Add remote storage
dvc remote add -d myremote s3://hndsr-data/dvc-store

# Run the full pipeline
dvc repro

# Push artifacts to remote
dvc push

# Check what changed
dvc status

# Reproduce only changed stages
dvc repro --downstream preprocess
```
