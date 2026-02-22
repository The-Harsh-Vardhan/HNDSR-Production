# Experiment Tracking & Hyperparameter Optimization

## What
MLflow-integrated experiment tracking and automated hyperparameter sweeps for the three-stage HNDSR training pipeline.

## Why

### Experiment Reproducibility
- Every training run must be traceable: hyperparameters → data version → model checkpoint → evaluation metrics
- Without tracking, teams waste GPU hours re-running experiments they've already tried
- MLflow provides a centralized UI for comparing runs across architectures, learning rates, and diffusion configurations

### Compute Cost Analysis
| Instance | Cost/hr | VRAM | Typical Use |
|----------|---------|------|-------------|
| AWS p3.2xlarge (V100) | $3.06 | 16 GB | Single-stage training |
| AWS p3.8xlarge (4×V100) | $12.24 | 64 GB | HPO sweeps |
| AWS p4d.24xlarge (8×A100) | $32.77 | 320 GB | Full pipeline + large sweep |
| Kaggle (P100) | Free | 16 GB | Development iteration |

A 10-trial HPO sweep on a p3.2xlarge costs ~$30. Without experiment tracking, you may repeat these runs 3–4× due to lost results.

### Leakage Prevention
- **Data leakage**: Validation set must never appear in training. The ETL pipeline enforces spatial separation. Experiment tracking logs the dataset hash to verify this.
- **Temporal leakage**: Early stopping on validation loss can overfit to the validation set if the patience is too low. Track the gap between train and val loss.
- **Hyperparameter leakage**: Using test set metrics to select hyperparameters. Track which metric was used for model selection.

## How

```bash
# Start MLflow tracking server
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

# Run a training experiment
python train_pipeline.py --stage autoencoder --epochs 20 --lr 1e-4

# Run HPO sweep
python experiment_tracking.py --sweep hpo_config.yaml --n-trials 20
```
