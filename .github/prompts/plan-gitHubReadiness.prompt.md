# GitHub Readiness Plan

## Completed

- [x] **Copy autoencoder checkpoint** — Copied `autoencoder_best.pth` from `MLFlow/` to `checkpoints/`
- [x] **Version bump to 1.1.0** — Updated `backend/app.py` in 3 locations (`api_version`, FastAPI constructor, `/version` endpoint)
- [x] **Update `.gitignore`** — Added `!checkpoints/manifest.json` exception so manifest is tracked
- [x] **Rewrite `model_registry/README.md`** — Removed stale canary references, fixed API examples, updated lifecycle to 4 stages
- [x] **Create `.github/dependabot.yml`** — pip (weekly), docker (monthly), github-actions (weekly)
- [x] **Create `.github/CODEOWNERS`** — All directories assigned to @The-Harsh-Vardhan
- [x] **Update project structure in README** — Removed ghost files (`inference_worker.py`, `canary_deploy.py`), added `backend/inference/`, `backend/model/`, `dvc_pipeline/src/`, all 11 test files, all 5 docs, `manifest.json`, `.github/` entries
- [x] **Fix broken references in README** — Corrected `HNDSR_Kaggle.ipynb` → `HNDSR_Kaggle_Updated.ipynb`

## Remaining

- [ ] **Update README version badge/header** — Ensure version references are consistent with 1.1.0
- [ ] **Git commit and push to GitHub**
  - Remote: `https://github.com/The-Harsh-Vardhan/HNDSR-Production`
  - Stage all changes: `git add -A`
  - Commit message: `feat: add DVC visualize stage, test suite, GitHub readiness updates`
  - Push to `main` branch
  - Note: `.pth` files under `checkpoints/` are gitignored (except `manifest.json`). Consider Git LFS if weights need to be in the repo.

## Files Modified This Session

| File | Change |
|------|--------|
| `backend/app.py` | Version 1.0.0 → 1.1.0 (3 locations) |
| `.gitignore` | Added `!checkpoints/manifest.json` |
| `model_registry/README.md` | Full rewrite (canary removal, API fixes) |
| `README.md` | Project structure rewrite, notebook reference fix |
| `dvc_pipeline/src/visualize.py` | New — comparison grid generation |
| `dvc_pipeline/dvc.yaml` | Added `visualize` stage |
| `dvc_pipeline/params.yaml` | Added `visualize` section |
| `tests/test_dvc_pipeline.py` | New — 30 pytest tests |
| `dvc_pipeline/src/utils.py` | Bug fix: `total_mem` → `total_memory` |
| `dvc_pipeline/src/train_stage.py` | Bug fix: unbound `avg_train`/`avg_val` |
| `dvc_pipeline/src/evaluate.py` | Bug fix: unbound `lpips_fn`/`avg_lpips`/`std_lpips` |

## Files Created This Session

| File | Purpose |
|------|---------|
| `.github/dependabot.yml` | Automated dependency updates |
| `.github/CODEOWNERS` | PR auto-assignment |
| `dvc_pipeline/src/visualize.py` | Before/after SR comparison grids |
| `tests/test_dvc_pipeline.py` | DVC pipeline unit + smoke tests |
