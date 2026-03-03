# Plan: DVC Pipeline for HNDSR Training

**TL;DR:** Build a new DVC pipeline from scratch by extracting the training logic from the MLFlow notebook into modular Python scripts, then wiring them with a `dvc.yaml` DAG. The pipeline has 4 stages (train AE → train FNO+ImplicitAmp → train Diffusion → evaluate), uses the notebook's hyperparameters (batch_size=2, patch_size=64, 180 total epochs), reads data from the Kaggle cache paths, and outputs checkpoints + JSON metrics tracked by DVC.

## Steps

### 1. Create `dvc_pipeline/src/dataset.py`

Extract `SatelliteDataset` from the notebook. Includes patch-based random crop (train) / center crop (eval), normalization to [-1, 1], HR/LR filename-stem matching with positional fallback. Data paths and `patch_size` configurable via arguments (defaulting to the notebook's Kaggle cache paths).

### 2. Create `dvc_pipeline/src/models.py`

Extract all model classes from the notebook: `ResidualBlock`, `LatentAutoencoder` (latent_dim=128, downsample_ratio=8), `SpectralConv2d`, `NeuralOperator` (modes=8, width=32), `ImplicitAmplification` (hidden_dim=256), `SinusoidalPositionEmbeddings`, `AttentionBlock`, `CrossAttentionBlock`, `ResidualBlockWithTime`, `DiffusionUNet` (model_channels=64, context_dim=128), `DDPMScheduler` (1000 timesteps, linear beta), and the composite `HNDSR` class with `get_no_prior()` and `super_resolve()`.

### 3. Create `dvc_pipeline/src/utils.py`

Utility functions: `set_seed(seed)` (Python + NumPy + PyTorch + CUDA deterministic), `get_device()`, `denormalize(tensor)` for [-1,1]→[0,1], PSNR/SSIM/LPIPS metric helpers, and a `load_params(path)` function to read `params.yaml`.

### 4. Create `dvc_pipeline/src/train_stage.py`

CLI-driven training entry point with `--stage {autoencoder, neural_operator, diffusion}` argument. Each stage:

- Reads hyperparameters from `params.yaml`
- Creates `SatelliteDataset` + 90/10 `random_split` + `DataLoader(batch_size=2, num_workers=0, drop_last=True)`
- Runs the per-stage training loop (matching the notebook exactly):
  - **autoencoder**: L1Loss, AdamW(lr=1e-4, wd=1e-4), CosineAnnealingLR, 50 epochs, saves `autoencoder_best.pth`
  - **neural_operator**: MSELoss on latent space, freezes AE, trains FNO + ImplicitAmp jointly, 30 epochs, saves `neural_operator_best.pth` (dict with both state_dicts)
  - **diffusion**: MSE noise prediction, freezes AE+FNO+ImplicitAmp, trains DiffusionUNet with gradient clipping at 1.0, 100 epochs, saves `diffusion_best.pth`
- After each stage, runs a sanity PSNR/SSIM check on a few validation samples
- Writes per-stage metrics to `metrics/stage{N}_metrics.json` (train_loss, val_loss, best_val_loss, check_psnr, check_ssim)
- Optionally integrates MLflow logging (gated behind `--mlflow` flag, so DVC works without MLflow running)

### 5. Create `dvc_pipeline/src/evaluate.py`

Evaluation entry point that:

- Loads all 3 checkpoints into the composite `HNDSR` model
- Creates a test dataset (SatelliteDataset with `training=False`, up to 50 random samples, batch_size=1)
- Computes PSNR, SSIM, LPIPS (AlexNet) with denormalization to [0,1]
- Saves results to `metrics/eval_metrics.json` (mean ± std for all 3 metrics)
- Saves first 10 visual comparison samples (LR, SR, HR) to `evaluation_results/`
- Reads `params.yaml` for data paths and model hyperparameters

### 6. Create `dvc_pipeline/params.yaml`

Replace existing file with notebook-matched hyperparameters:

- `data.hr_dir`, `data.lr_dir` → Kaggle cache paths from the notebook
- `data.patch_size: 64`, `data.batch_size: 2`, `data.num_workers: 0`, `data.val_split: 0.1`
- `seed: 42`
- `autoencoder`: lr=1e-4, wd=1e-4, epochs=50, latent_dim=128, downsample_ratio=8, num_res_blocks=4
- `neural_operator`: lr=1e-4, wd=1e-4, epochs=30, modes=8, width=32
- `implicit_amp`: hidden_dim=256
- `diffusion`: lr=1e-4, wd=1e-4, epochs=100, model_channels=64, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, num_inference_steps=50, grad_clip=1.0
- `evaluate`: num_test_samples=50, thresholds (psnr>29, ssim>0.85, lpips<0.20)
- `checkpoints.dir: ../checkpoints` (output directory)

### 7. Create `dvc_pipeline/dvc.yaml`

4-stage DAG:

- **`train_autoencoder`**: cmd=`python src/train_stage.py --stage autoencoder`, deps=[`src/`, `params.yaml`], params=[`seed`, `data`, `autoencoder`], outs=[`../checkpoints/autoencoder_best.pth`], metrics=[`metrics/stage1_metrics.json`]
- **`train_neural_operator`**: cmd=`python src/train_stage.py --stage neural_operator`, deps=[`src/`, `params.yaml`, `../checkpoints/autoencoder_best.pth`], params=[`seed`, `data`, `neural_operator`, `implicit_amp`], outs=[`../checkpoints/neural_operator_best.pth`], metrics=[`metrics/stage2_metrics.json`]
- **`train_diffusion`**: cmd=`python src/train_stage.py --stage diffusion`, deps=[`src/`, `params.yaml`, `../checkpoints/autoencoder_best.pth`, `../checkpoints/neural_operator_best.pth`], params=[`seed`, `data`, `diffusion`], outs=[`../checkpoints/diffusion_best.pth`], metrics=[`metrics/stage3_metrics.json`]
- **`evaluate`**: cmd=`python src/evaluate.py`, deps=[`src/`, `params.yaml`, all 3 checkpoints], params=[`seed`, `data`, `evaluate`], outs=[`evaluation_results/`], metrics=[`metrics/eval_metrics.json`]

### 8. Create `dvc_pipeline/README.md`

Replace existing with usage documentation: prerequisites, `dvc repro` workflow, `dvc params diff`, `dvc metrics show`, and how to run individual stages.

## File Tree

```
dvc_pipeline/
├── dvc.yaml
├── params.yaml
├── README.md
└── src/
    ├── dataset.py
    ├── models.py
    ├── train_stage.py
    ├── evaluate.py
    └── utils.py
```

## Verification

- `cd dvc_pipeline && dvc repro` runs the full 4-stage pipeline end-to-end
- `dvc metrics show` displays per-stage losses + final PSNR/SSIM/LPIPS
- `dvc params diff` shows hyperparameter changes between experiments
- `dvc dag` visualizes the 4-stage DAG: `train_autoencoder → train_neural_operator → train_diffusion → evaluate`
- Checkpoints at `checkpoints/` match the format expected by `backend/inference/model_loader.py` (`strict=True` loading)

## Decisions

- **Notebook model architecture over production `model_stubs.py`**: The DVC pipeline trains the notebook's architecture (which produced the original checkpoints). Production model_stubs.py was reverse-engineered separately and may differ — the DVC pipeline is for training, not inference compatibility.
- **No separate ETL stage**: The Kaggle dataset is already split into HR/LR. The `SatelliteDataset` handles patching and augmentation on the fly. An ETL stage would add complexity without value here.
- **MLflow logging is optional**: The `--mlflow` flag enables it, but the pipeline works standalone with just DVC metric JSON files. This avoids requiring an MLflow server for `dvc repro`.
- **Checkpoints output to `../checkpoints/`**: Shared with the rest of the project, matching the existing directory structure used by the inference backend.
