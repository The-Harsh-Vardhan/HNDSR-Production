# Data Pipeline & ETL

## What
An automated pipeline that transforms raw high-resolution (HR) satellite images into a reproducible, versioned dataset ready for training the HNDSR model.

## Why

### Data Versioning Matters
- **Reproducibility**: Without hashing and versioning, you cannot guarantee that two training runs used the same data. A single corrupted tile can silently degrade model quality by 1–2 dB PSNR.
- **Auditability**: Production ML requires proving which data produced which model. Regulatory and internal review demand this traceability.
- **Rollback**: If a new data batch introduces drift (e.g., a satellite sensor recalibration), you need to quickly revert to the previous known-good dataset.

### Data Drift Risks
| Drift Type | Cause | Impact |
|------------|-------|--------|
| Sensor drift | Satellite sensor degradation over time | Gradual PSNR decline |
| Distribution shift | New geographic region (e.g., desert → urban) | Model produces artifacts |
| Preprocessing drift | Change in downsampling kernel | LR/HR misalignment |
| Label drift | HR ground truth from different sensor | Inconsistent supervision |

### ETL Bottlenecks
- **I/O bound**: Reading thousands of large TIFF files from S3 is the primary bottleneck (~80% of ETL time)
- **CPU bound**: Bicubic downsampling is parallelizable but memory-intensive for large tiles
- **Storage**: Raw HR + 3 LR scales = 4× storage of original dataset

### Storage Cost Considerations
| Storage Tier | Cost (AWS S3) | Use Case |
|-------------|---------------|----------|
| S3 Standard | $0.023/GB/month | Active training data |
| S3 IA | $0.0125/GB/month | Previous data versions |
| S3 Glacier | $0.004/GB/month | Archived experiments |
| Local SSD | One-time cost | Development iteration |

## How

### Pipeline Stages
```
Raw HR Images (S3/MinIO)
        │
        ▼
┌───────────────┐
│  Validate     │  Check format, dimensions, corruption
│  & Catalog    │  Generate metadata (Parquet)
└───────┬───────┘
        │
        ▼
┌───────────────┐
│  Downsample   │  Bicubic interpolation at 2×, 4×, 6×
│  (LR Gen)     │  Anti-aliasing filter applied
└───────┬───────┘
        │
        ▼
┌───────────────┐
│  Split        │  Stratified by geographic region
│  Train/Val/   │  80/10/10 split
│  Test         │  No spatial overlap between splits
└───────┬───────┘
        │
        ▼
┌───────────────┐
│  Hash &       │  SHA-256 per file
│  Version      │  Dataset manifest → DVC tracking
└───────────────┘
```

### Usage
```bash
python etl_pipeline.py \
    --input-dir s3://hndsr-data/raw/ \
    --output-dir s3://hndsr-data/processed/ \
    --scales 2 4 6 \
    --split-ratio 0.8 0.1 0.1 \
    --seed 42
```
