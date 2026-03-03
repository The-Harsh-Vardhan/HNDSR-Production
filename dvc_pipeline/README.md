# HNDSR DVC Training Pipeline

Reproducible 5-stage training pipeline for the **Hybrid Neural Operator-Diffusion Super-Resolution** model, managed by [DVC](https://dvc.org/).

## Pipeline DAG

```
train_autoencoder --> train_neural_operator --> train_diffusion --> evaluate
                                                                --> visualize
```

| Stage | Component | Loss | Epochs | Checkpoint |
|-------|-----------|------|--------|------------|
| 1 | Latent Autoencoder | L1 | 50 | `autoencoder_best.pth` |
| 2 | FNO + ImplicitAmp | MSE (latent) | 30 | `neural_operator_best.pth` |
| 3 | Diffusion UNet | MSE (noise) | 100 | `diffusion_best.pth` |
| 4 | Evaluation | PSNR / SSIM / LPIPS | - | `metrics/eval_metrics.json` |
| 5 | Visualize | Before/after grids | - | `visualize_results/` |

## Prerequisites

```bash
pip install -r ../requirements.txt
pip install dvc
```

Ensure the Kaggle dataset is available at the paths in `params.yaml`.

## Quick Start

```bash
cd dvc_pipeline

# Run the full pipeline (all 4 stages)
dvc repro

# Run a single stage (automatically runs upstream deps first)
dvc repro train_autoencoder

# Check the pipeline DAG
dvc dag

# View metrics
dvc metrics show

# Compare parameters between experiments
dvc params diff
```

## File Structure

```
dvc_pipeline/
+-- dvc.yaml          # Pipeline definition (5 stages)
+-- params.yaml       # All hyperparameters (DVC-tracked)
+-- README.md
+-- src/
|   +-- __init__.py
|   +-- dataset.py    # SatelliteDataset (patch-based HR/LR pairs)
|   +-- models.py     # Full HNDSR architecture
|   +-- utils.py      # Seeding, metrics, YAML loading
|   +-- train_stage.py # CLI entry-point for per-stage training
|   +-- evaluate.py   # Final evaluation with PSNR/SSIM/LPIPS
|   +-- visualize.py  # Before/after comparison grid generator
+-- metrics/          # JSON metric files (auto-generated)
+-- evaluation_results/ # Evaluation PNGs (auto-generated)
+-- visualize_results/  # Comparison grids (auto-generated)
```

## Hyperparameter Experiments

Edit `params.yaml` and re-run:

```bash
# Change learning rate, for example
# Edit params.yaml -> autoencoder.lr: 5.0e-5

dvc repro                   # Only re-trains affected stages
dvc params diff             # See what changed
dvc metrics diff            # Compare old vs new metrics
```

## MLflow Integration (Optional)

Pass `--mlflow` to enable experiment tracking:

```bash
python src/train_stage.py --stage autoencoder --mlflow
```

This requires an MLflow tracking server or local SQLite backend.

## Outputs

- **Checkpoints** are written to `../checkpoints/` (shared with the inference backend).
- **Metrics** are written to `metrics/*.json` and tracked by `dvc metrics`.
- **Visual samples** (first 10 test images) are saved to `evaluation_results/`.
- **Comparison grids** are saved to `visualize_results/`.

## Visualize Stage

The `visualize` stage produces side-by-side comparison images for qualitative evaluation:

```
[ LR (nearest ↑) | Bicubic (↑) | SR (model) | HR (ground truth) ]
```

Each comparison strip shows 4 columns — the blocky nearest-upscale of LR, naive bicubic baseline, model's super-resolved output, and ground-truth HR. A full montage (`comparison_grid.png`) and per-sample strips are generated.

```bash
# Run only the visualize stage
dvc repro visualize

# Or run standalone
cd dvc_pipeline
python src/visualize.py --params params.yaml
```

### Output layout

```
visualize_results/
+-- comparison_grid.png           # Full montage (all samples stacked)
+-- sample_000_comparison.png     # Per-sample horizontal strips
+-- sample_001_comparison.png
+-- ...
```

Metrics are written to `metrics/visualize_metrics.json` with per-sample PSNR/SSIM for both SR and bicubic, plus aggregate improvement statistics.

## Testing

```bash
# Run the DVC pipeline test suite
pytest tests/test_dvc_pipeline.py -v

# Run all tests
pytest tests/ -v
```

The test suite covers:
- **Utils**: seed reproducibility, YAML parsing, denormalize, PSNR/SSIM metrics
- **Dataset**: HR/LR pair creation, shape contracts, normalisation range
- **Models**: autoencoder roundtrip, FNO shape, super-resolve, scheduler, checkpoint save/load
- **Smoke tests**: params.yaml/dvc.yaml integrity, model instantiation (6.3M params), checkpoint loading (skipped if not present)
