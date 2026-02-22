# Container Registry Strategy

## What
Tagging, versioning, and rollback strategy for HNDSR Docker images in AWS ECR.

## Why
Without a tagging strategy:
- "Which image is in production?" → Unknown
- "Roll back to last Thursday" → Impossible
- "Did we test this exact image?" → Can't verify

## Tagging Convention

```
<ecr-registry>/hndsr:<tag>
```

| Tag | When Applied | Example | Purpose |
|-----|-------------|---------|---------|
| `sha-<git-sha>` | Every CI build | `sha-a1b2c3d` | Exact build traceability |
| `latest` | Every main branch merge | `latest` | Dev convenience (never use in prod) |
| `staging` | Staging deploy | `staging` | Current staging image |
| `production` | Production deploy | `production` | Current production image |
| `v<semver>` | Manual release | `v1.2.0` | Semantic version for releases |
| `canary` | Canary deploy | `canary` | Under A/B testing |

## Lifecycle

```
CI Build                    Staging                    Production
   │                          │                           │
   ▼                          ▼                           ▼
sha-a1b2c3d  ──────▶  sha-a1b2c3d     ──────▶    sha-a1b2c3d
                      + staging tag              + production tag
                                                 + v1.2.0 tag
```

## Rollback

```bash
# Instant rollback: re-tag previous production image
aws ecr batch-get-image --repository-name hndsr \
    --image-ids imageTag=v1.1.0 --output json |
    jq '.images[0].imageManifest' |
    xargs -I {} aws ecr put-image --repository-name hndsr \
        --image-tag production --image-manifest '{}'

# Verify
aws ecr describe-images --repository-name hndsr \
    --image-ids imageTag=production
```

## Cleanup Policy

```json
{
  "rules": [
    {
      "rulePriority": 1,
      "description": "Keep production and staging tags forever",
      "selection": {
        "tagStatus": "tagged",
        "tagPrefixList": ["production", "staging", "v"],
        "countType": "sinceImagePushed",
        "countUnit": "days",
        "countNumber": 9999
      },
      "action": { "type": "expire" }
    },
    {
      "rulePriority": 2,
      "description": "Keep last 20 SHA-tagged images",
      "selection": {
        "tagStatus": "tagged",
        "tagPrefixList": ["sha-"],
        "countType": "imageCountMoreThan",
        "countNumber": 20
      },
      "action": { "type": "expire" }
    },
    {
      "rulePriority": 3,
      "description": "Remove untagged images after 7 days",
      "selection": {
        "tagStatus": "untagged",
        "countType": "sinceImagePushed",
        "countUnit": "days",
        "countNumber": 7
      },
      "action": { "type": "expire" }
    }
  ]
}
```
