# Model Registry

## What
MLflow Model Registry integration for versioning, promoting, and deploying HNDSR models through the dev → staging → production lifecycle.

## Why

### Safe Model Lifecycle
Without a registry, model deployment is ad-hoc:
- "I copied the checkpoint to the server" → No rollback capability
- "Which version is in production?" → Nobody knows
- "The new model seems worse" → Cannot A/B test or compare

### Version Tagging Strategy
| Stage | Tag | Criteria | Auto/Manual |
|-------|-----|----------|-------------|
| **Development** | `dev` | Any training run | Auto |
| **Staging** | `staging` | Passes shape tests + PSNR > 26 dB | Auto |
| **Pre-Production** | `canary` | 10% traffic, no error spike for 1 hour | Manual trigger |
| **Production** | `production` | Canary passed + manual approval | Manual |
| **Archived** | `archived` | Replaced by newer version | Auto |

### Rollback Strategy
```
Current: production/v1.3.0
Problem: PSNR drops 2 dB after deployment

Action:  1. Flag current version as "degraded"
         2. Promote previous version (v1.2.0) back to "production"
         3. Kubernetes picks up new version via ConfigMap
         4. Total rollback time: ~2 minutes (no model reload if cached)
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

# Promote to staging
registry.promote("hndsr", version=5, stage="staging")

# Start canary deployment
registry.start_canary("hndsr", version=5, traffic_pct=10)
```
