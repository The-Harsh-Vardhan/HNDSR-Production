# Model Registry

## What
MLflow Model Registry integration for versioning, promoting, and deploying HNDSR models through the dev → staging → production lifecycle.

## Why

### Safe Model Lifecycle
Without a registry, model deployment is ad-hoc:
- "I copied the checkpoint to the server" → No rollback capability
- "Which version is in production?" → Nobody knows
- "The new model seems worse" → Cannot compare versions or roll back

### Version Tagging Strategy
| Stage | Tag | Criteria | Auto/Manual |
|-------|-----|----------|-------------|
| **Development** | `dev` | Any training run | Auto |
| **Staging** | `staging` | Passes quality gates (PSNR ≥ 26, SSIM ≥ 0.75, LPIPS ≤ 0.30) | Auto |
| **Production** | `production` | Staging passed + manual approval | Manual |
| **Archived** | `archived` | Replaced by newer version | Auto |

### Rollback Strategy
```
Current: production/v1.3.0
Problem: PSNR drops 2 dB after deployment

Action:  1. Flag current version as "degraded"
         2. Promote previous version (v1.2.0) back to "production"
         3. Total rollback time: < 2 minutes (no model reload if cached)
```

## How
```python
from registry_integration import ModelRegistry

registry = ModelRegistry()

# Register a new model version
registry.register_model(
    checkpoint_dir="./checkpoints/",
    metrics={"psnr": 27.5, "ssim": 0.82},
    dataset_hash="abc123..."
)

# Promote through lifecycle
registry.promote("hndsr", version=5, stage="staging")
registry.promote("hndsr", version=5, stage="production")

# Rollback if needed
registry.rollback("hndsr")
```
