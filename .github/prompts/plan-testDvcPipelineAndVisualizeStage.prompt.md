## Plan: Test DVC Pipeline + Add Visualize Stage

**TL;DR** — Three deliverables: (A) fix 2 existing bugs, (B) add a new `visualize` DVC stage that creates side-by-side LR↑ | SR | HR comparison grids as PNGs, (C) create a pytest-based test suite covering `dataset.py`, `models.py`, `utils.py`, and a smoke test for the full pipeline. The visualize stage chains after `evaluate`, reusing the existing checkpoint-loading and inference patterns. Tests follow the mock-model and fixture patterns already established in [tests/conftest.py](tests/conftest.py).

**Steps**

### A. Bug Fixes

1. **Fix `total_mem` → `total_memory`** in [dvc_pipeline/src/utils.py](dvc_pipeline/src/utils.py#L39) — `torch.cuda.get_device_properties(0).total_memory` is the correct attribute.

2. **Fix "possibly unbound" warnings** in [dvc_pipeline/src/train_stage.py](dvc_pipeline/src/train_stage.py) — initialize `avg_train = 0.0` and `avg_val = 0.0` before the epoch loop in all 3 training functions (autoencoder ~L196, neural_operator ~L322, diffusion ~L469).

3. **Fix "possibly unbound" warnings** in [dvc_pipeline/src/evaluate.py](dvc_pipeline/src/evaluate.py) — initialize `lpips_fn = None`, `avg_lpips = 0.0`, `std_lpips = 0.0` before the LPIPS conditional block.

### B. New Visualize Stage

4. **Add `visualize` section to [params.yaml](dvc_pipeline/params.yaml)**:
   ```yaml
   visualize:
     num_samples: 8
     output_dir: "visualize_results"
     diffusion_strength: 0.0
     grid_cols: 4         # columns in the montage grid
     save_individual: true # also save separate LR/SR/HR
   ```

5. **Create [dvc_pipeline/src/visualize.py](dvc_pipeline/src/visualize.py)** — new script (~180 lines):
   - **CLI:** `python src/visualize.py [--params params.yaml]`
   - **Flow:**
     1. Load params → instantiate `HNDSR` → load all 3 checkpoints (reuse exact `_load_checkpoints()` pattern from `evaluate.py`)
     2. Create val dataset with `SatelliteDataset(train=False)`, take first `num_samples` images
     3. For each sample, run `model.super_resolve(lr, diffusion_strength=...)` to get SR
     4. Create bicubic-upscaled LR: `F.interpolate(lr, scale_factor=4, mode='bicubic')`
     5. Build per-sample horizontal strip: `[LR↑ | Bicubic↑ | SR | HR]` using `torch.cat(dim=-1)` (width-wise concat)
     6. Compute per-sample PSNR/SSIM overlaid as text (optional, using PIL `ImageDraw`)
     7. Build a full montage grid via `torchvision.utils.make_grid(strips, nrow=1)` → one tall composite
     8. Save outputs to `visualize_results/`:
        - `comparison_grid.png` — full montage
        - `sample_{idx:03d}_comparison.png` — individual strips
        - `visualize_metrics.json` — per-sample PSNR/SSIM
   - **Key imports:** `torch`, `torchvision.utils.save_image`, `torch.nn.functional`, plus local `dataset`, `models`, `utils`
   - **Checkpoint loading:** identical pattern to `evaluate.py` (`_load_checkpoints()`)

6. **Add `visualize` stage to [dvc.yaml](dvc_pipeline/dvc.yaml)**:
   ```yaml
   visualize:
     cmd: python src/visualize.py --params params.yaml
     deps:
       - src/dataset.py
       - src/models.py
       - src/utils.py
       - src/visualize.py
       - ../checkpoints/autoencoder_best.pth
       - ../checkpoints/neural_operator_best.pth
       - ../checkpoints/diffusion_best.pth
     params:
       - seed
       - data
       - checkpoints
       - visualize
     outs:
       - visualize_results
     metrics:
       - metrics/visualize_metrics.json:
           cache: false
   ```
   This stage depends on all 3 checkpoints (chains after training) but runs independently of `evaluate` — both can execute in parallel after training.

### C. Test Suite for the DVC Pipeline

7. **Create [tests/test_dvc_pipeline.py](tests/test_dvc_pipeline.py)** — pytest suite (~250 lines):

   **Class `TestUtils`:**
   - `test_set_seed_reproducibility` — call `set_seed(42)` twice, verify `torch.randn` produces identical results
   - `test_load_params` — create a temp YAML with `tmp_path`, verify parsed dict matches
   - `test_denormalize` — verify `denormalize(tensor(-1))` → 0, `denormalize(tensor(1))` → 1
   - `test_calculate_psnr_identical` — same image → PSNR ≥ 40 dB
   - `test_calculate_ssim_identical` — same image → SSIM ≈ 1.0
   - `test_calculate_psnr_different` — random noise images → PSNR < 20

   **Class `TestDataset`:**
   - `test_dataset_creates_pairs` — create temp dir with 5 dummy `.png` HR/LR images via PIL, verify `len(ds) == 5` and returned dict has keys `lr`, `hr`, `scale`
   - `test_dataset_shapes_train` — verify HR patch is `(3, patch_size, patch_size)` and LR patch is `(3, patch_size//4, patch_size//4)`
   - `test_dataset_shapes_eval` — same but for eval mode (center crop)
   - `test_dataset_normalization` — verify returned tensors are in `[-1, 1]` range

   **Class `TestModels`:**
   - Use a **tiny model** fixture: `HNDSR(ae_latent_dim=8, ae_downsample_ratio=8, no_width=4, no_modes=2, diffusion_channels=8, num_timesteps=10)` — fast to instantiate/run
   - `test_autoencoder_roundtrip` — encode → decode, verify output shape matches input
   - `test_neural_operator_shape` — verify FNO output shape `(B, latent_dim, H_latent, W_latent)`
   - `test_super_resolve_shape` — full pipeline: input `(1,3,16,16)` → output `(1,3,64,64)`
   - `test_super_resolve_no_nan` — verify no NaN in output
   - `test_ddpm_scheduler_noise` — add noise at t, verify output shape + dtype
   - `test_checkpoint_save_load_roundtrip` — save AE state dict → reload into fresh model → verify weights match

   **Class `TestPipelineSmoke`:**
   - `test_params_yaml_parse` — load actual `params.yaml`, verify all required keys exist (`seed`, `data`, `checkpoints`, `autoencoder`, `neural_operator`, `implicit_amp`, `diffusion`, `evaluate`, `visualize`)
   - `test_dvc_yaml_stages` — parse `dvc.yaml` and verify all 5 stages exist with correct `cmd` strings
   - `test_model_instantiation_from_params` — load `params.yaml`, build model with those params, verify param count is 6,290,371
   - `test_checkpoint_loading` — if checkpoints exist at `../checkpoints/`, load them into model, verify no errors (skip if checkpoints not found via `pytest.mark.skipif`)

8. **Update [dvc_pipeline/README.md](dvc_pipeline/README.md)** — add section documenting:
   - The new `visualize` stage and its purpose
   - How to run tests: `pytest tests/test_dvc_pipeline.py -v`
   - Example output structure of `visualize_results/`

**Verification**
- `cd dvc_pipeline && python -c "import yaml; yaml.safe_load(open('params.yaml'))"` — params parse with new `visualize` section
- `cd dvc_pipeline && dvc dag` — verify 5-stage DAG shows correctly
- `cd dvc_pipeline && python src/visualize.py --params params.yaml` — runs end-to-end, creates `visualize_results/comparison_grid.png`
- `pytest tests/test_dvc_pipeline.py -v` — all tests pass
- `dvc repro visualize` — full pipeline reproduction including visualize stage

**Decisions**
- Visualize stage runs **independently of evaluate** (parallel after training) rather than chaining after evaluate — both only need the 3 checkpoints, no dependency between them.
- Side-by-side layout is `[LR↑ | Bicubic↑ | SR | HR]` (4 columns) — includes bicubic baseline to highlight the model's improvement over naive upscaling.
- Tests use a **tiny model** (`latent_dim=8, width=4, modes=2, channels=8`) for speed, plus one conditional test that loads real checkpoints when available.
- `diffusion_strength: 0.0` default in visualize params — keeps it fast; user can bump to test diffusion refinement.
