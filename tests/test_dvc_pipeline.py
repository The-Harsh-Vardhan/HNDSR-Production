"""
tests/test_dvc_pipeline.py
===========================
Pytest suite for the HNDSR DVC training pipeline.

Covers:
    - utils  (seed, params, denormalize, PSNR, SSIM)
    - dataset (pair construction, shapes, normalisation)
    - models (autoencoder, FNO, super_resolve, scheduler, checkpoints)
    - smoke  (params.yaml / dvc.yaml integrity, model instantiation)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

# Allow imports from dvc_pipeline/src/ regardless of where pytest is invoked
_SRC_DIR = str(Path(__file__).resolve().parent.parent / "dvc_pipeline" / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from dataset import SatelliteDataset
from models import HNDSR, DDPMScheduler, LatentAutoencoder
from utils import set_seed, load_params, denormalize, calculate_psnr, calculate_ssim

# Paths relative to the project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DVC_DIR = _PROJECT_ROOT / "dvc_pipeline"
_CKPT_DIR = _PROJECT_ROOT / "checkpoints"


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tiny_model():
    """A very small HNDSR instance for fast unit tests.

    Uses ae_downsample_ratio=1 so that the encoder output channels
    equal ae_latent_dim (no channel-doubling from downsample steps),
    keeping all component dimensions consistent.
    """
    return HNDSR(
        ae_latent_dim=8,
        ae_downsample_ratio=1,
        no_width=4,
        no_modes=2,
        diffusion_channels=8,
        num_timesteps=10,
    )


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def dummy_image_dirs(tmp_path):
    """Create temp HR / LR directories with 5 matching 64×64 / 16×16 PNG pairs."""
    hr_dir = tmp_path / "HR"
    lr_dir = tmp_path / "LR"
    hr_dir.mkdir()
    lr_dir.mkdir()

    for i in range(5):
        hr_img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        lr_img = Image.fromarray(np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8))
        hr_img.save(hr_dir / f"img_{i:03d}.png")
        lr_img.save(lr_dir / f"img_{i:03d}.png")

    return str(hr_dir), str(lr_dir)


# ─────────────────────────────────────────────────────────────────────────────
# A. Utils
# ─────────────────────────────────────────────────────────────────────────────

class TestUtils:
    """Tests for dvc_pipeline/src/utils.py."""

    def test_set_seed_reproducibility(self):
        set_seed(42)
        a = torch.randn(5)
        set_seed(42)
        b = torch.randn(5)
        assert torch.equal(a, b), "set_seed should produce identical random sequences"

    def test_load_params(self, tmp_path):
        yaml_data = {"seed": 123, "data": {"batch_size": 4}}
        path = tmp_path / "test_params.yaml"
        with open(path, "w") as f:
            yaml.dump(yaml_data, f)

        result = load_params(str(path))
        assert result == yaml_data

    def test_denormalize_bounds(self):
        t_min = torch.tensor([-1.0])
        t_max = torch.tensor([1.0])
        assert denormalize(t_min).item() == pytest.approx(0.0)
        assert denormalize(t_max).item() == pytest.approx(1.0)

    def test_denormalize_midpoint(self):
        t_mid = torch.tensor([0.0])
        assert denormalize(t_mid).item() == pytest.approx(0.5)

    def test_calculate_psnr_identical(self):
        torch.manual_seed(0)
        img = torch.randn(2, 3, 32, 32).clamp(-1, 1)
        p = calculate_psnr(img, img)
        assert p >= 40.0, f"Identical images should have PSNR >= 40 dB, got {p:.2f}"

    def test_calculate_ssim_identical(self):
        torch.manual_seed(0)
        img = torch.randn(2, 3, 32, 32).clamp(-1, 1)
        s = calculate_ssim(img, img)
        assert s >= 0.99, f"Identical images should have SSIM ≈ 1.0, got {s:.4f}"

    def test_calculate_psnr_different(self):
        torch.manual_seed(0)
        a = torch.randn(2, 3, 32, 32).clamp(-1, 1)
        torch.manual_seed(99)
        b = torch.randn(2, 3, 32, 32).clamp(-1, 1)
        p = calculate_psnr(a, b)
        assert p < 20.0, f"Random noise images should have low PSNR, got {p:.2f}"


# ─────────────────────────────────────────────────────────────────────────────
# B. Dataset
# ─────────────────────────────────────────────────────────────────────────────

class TestDataset:
    """Tests for dvc_pipeline/src/dataset.py."""

    def test_dataset_creates_pairs(self, dummy_image_dirs):
        hr_dir, lr_dir = dummy_image_dirs
        ds = SatelliteDataset(hr_dir, lr_dir, patch_size=64, training=True)
        assert len(ds) == 5

    def test_dataset_returns_correct_keys(self, dummy_image_dirs):
        hr_dir, lr_dir = dummy_image_dirs
        ds = SatelliteDataset(hr_dir, lr_dir, patch_size=64, training=True)
        sample = ds[0]
        assert "lr" in sample
        assert "hr" in sample
        assert "scale" in sample

    def test_dataset_shapes_train(self, dummy_image_dirs):
        hr_dir, lr_dir = dummy_image_dirs
        patch_size = 64
        ds = SatelliteDataset(hr_dir, lr_dir, patch_size=patch_size, training=True)
        sample = ds[0]
        # HR should be (3, patch_size, patch_size)
        assert sample["hr"].shape == (3, patch_size, patch_size), \
            f"Expected HR shape (3, {patch_size}, {patch_size}), got {sample['hr'].shape}"
        # LR should be (3, patch_size//4, patch_size//4)
        lr_size = patch_size // 4
        assert sample["lr"].shape == (3, lr_size, lr_size), \
            f"Expected LR shape (3, {lr_size}, {lr_size}), got {sample['lr'].shape}"

    def test_dataset_shapes_eval(self, dummy_image_dirs):
        hr_dir, lr_dir = dummy_image_dirs
        patch_size = 64
        ds = SatelliteDataset(hr_dir, lr_dir, patch_size=patch_size, training=False)
        sample = ds[0]
        assert sample["hr"].shape == (3, patch_size, patch_size)
        lr_size = patch_size // 4
        assert sample["lr"].shape == (3, lr_size, lr_size)

    def test_dataset_normalisation_range(self, dummy_image_dirs):
        hr_dir, lr_dir = dummy_image_dirs
        ds = SatelliteDataset(hr_dir, lr_dir, patch_size=64, training=True)
        sample = ds[0]
        for key in ("lr", "hr"):
            assert sample[key].min() >= -1.0 - 1e-6, f"{key} min below -1"
            assert sample[key].max() <= 1.0 + 1e-6, f"{key} max above  1"

    def test_dataset_scale_is_four(self, dummy_image_dirs):
        hr_dir, lr_dir = dummy_image_dirs
        ds = SatelliteDataset(hr_dir, lr_dir, patch_size=64, training=True)
        assert ds[0]["scale"] == 4

    def test_empty_dir_raises(self, tmp_path):
        hr_dir = tmp_path / "empty_hr"
        lr_dir = tmp_path / "empty_lr"
        hr_dir.mkdir()
        lr_dir.mkdir()
        with pytest.raises(ValueError, match="No images found"):
            SatelliteDataset(str(hr_dir), str(lr_dir))


# ─────────────────────────────────────────────────────────────────────────────
# C. Models
# ─────────────────────────────────────────────────────────────────────────────

class TestModels:
    """Tests for dvc_pipeline/src/models.py (using a tiny HNDSR)."""

    def test_autoencoder_roundtrip_shape(self, tiny_model):
        x = torch.randn(1, 3, 64, 64)
        recon, z = tiny_model.autoencoder(x)
        assert recon.shape == x.shape, \
            f"AE roundtrip shape mismatch: {recon.shape} != {x.shape}"

    def test_autoencoder_latent_shape(self, tiny_model):
        x = torch.randn(1, 3, 64, 64)
        _, z = tiny_model.autoencoder(x)
        # downsample_ratio=1 → no spatial reduction, channels = latent_dim=8
        assert z.shape == (1, 8, 64, 64), f"Latent shape expected (1,8,64,64), got {z.shape}"

    def test_neural_operator_output_shape(self, tiny_model):
        lr = torch.randn(1, 3, 16, 16)
        lr_up = torch.nn.functional.interpolate(lr, scale_factor=4, mode="bicubic", align_corners=False)
        out = tiny_model.neural_operator(lr_up, 4)
        # FNO output: (B, latent_dim, H, W) same spatial as lr_up
        assert out.shape[0] == 1
        assert out.shape[1] == 8  # latent_dim
        assert out.shape[2] == lr_up.shape[2]
        assert out.shape[3] == lr_up.shape[3]

    def test_super_resolve_shape(self, tiny_model, device):
        tiny_model.to(device)
        lr = torch.randn(1, 3, 16, 16, device=device)
        sr = tiny_model.super_resolve(lr, scale_factor=4, diffusion_strength=0.0)
        assert sr.shape == (1, 3, 64, 64), f"SR shape expected (1,3,64,64), got {sr.shape}"

    def test_super_resolve_no_nan(self, tiny_model, device):
        tiny_model.to(device)
        lr = torch.randn(1, 3, 16, 16, device=device)
        sr = tiny_model.super_resolve(lr, scale_factor=4, diffusion_strength=0.0)
        assert not torch.isnan(sr).any(), "SR output contains NaN"

    def test_super_resolve_with_diffusion(self, tiny_model, device):
        tiny_model.to(device)
        lr = torch.randn(1, 3, 16, 16, device=device)
        sr = tiny_model.super_resolve(
            lr, scale_factor=4, num_inference_steps=3, diffusion_strength=0.2
        )
        assert sr.shape == (1, 3, 64, 64), f"SR with diffusion shape {sr.shape}"
        assert not torch.isnan(sr).any(), "SR (diffusion) output contains NaN"

    def test_ddpm_scheduler_add_noise(self):
        sched = DDPMScheduler(num_timesteps=10)
        x0 = torch.randn(2, 8, 8, 8)
        noise = torch.randn_like(x0)
        t = torch.tensor([3, 7])
        noisy = sched.add_noise(x0, noise, t)
        assert noisy.shape == x0.shape
        assert noisy.dtype == x0.dtype

    def test_ddpm_scheduler_ddim_sample(self):
        sched = DDPMScheduler(num_timesteps=10)
        noise_pred = torch.randn(2, 8, 8, 8)
        z_t = torch.randn(2, 8, 8, 8)
        # Signature: ddim_sample(model_output, timestep, sample)
        out = sched.ddim_sample(noise_pred, 5, z_t)
        assert out.shape == z_t.shape

    def test_checkpoint_save_load_roundtrip(self, tiny_model, tmp_path):
        """Save AE state dict and reload into a fresh model; weights must match."""
        ckpt = tmp_path / "ae.pth"
        torch.save(tiny_model.autoencoder.state_dict(), ckpt)

        fresh = HNDSR(
            ae_latent_dim=8, ae_downsample_ratio=1,
            no_width=4, no_modes=2, diffusion_channels=8, num_timesteps=10,
        )
        fresh.autoencoder.load_state_dict(torch.load(ckpt, weights_only=True))

        for (n1, p1), (n2, p2) in zip(
            tiny_model.autoencoder.named_parameters(),
            fresh.autoencoder.named_parameters(),
        ):
            assert n1 == n2
            assert torch.equal(p1, p2), f"Weight mismatch on {n1}"


# ─────────────────────────────────────────────────────────────────────────────
# D. Pipeline smoke tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineSmoke:
    """Validate params.yaml / dvc.yaml integrity and model construction."""

    @pytest.fixture(autouse=True)
    def _load_yamls(self):
        """Load both YAML files once per test class."""
        params_path = _DVC_DIR / "params.yaml"
        dvc_path = _DVC_DIR / "dvc.yaml"
        if params_path.exists():
            with open(params_path) as f:
                self.params = yaml.safe_load(f)
        else:
            self.params = None
        if dvc_path.exists():
            with open(dvc_path) as f:
                self.dvc = yaml.safe_load(f)
        else:
            self.dvc = None

    @pytest.mark.skipif(
        not (_DVC_DIR / "params.yaml").exists(),
        reason="params.yaml not found",
    )
    def test_params_yaml_required_keys(self):
        required = [
            "seed", "data", "checkpoints", "autoencoder",
            "neural_operator", "implicit_amp", "diffusion", "evaluate", "visualize",
        ]
        for key in required:
            assert key in self.params, f"Missing required key '{key}' in params.yaml"

    @pytest.mark.skipif(
        not (_DVC_DIR / "params.yaml").exists(),
        reason="params.yaml not found",
    )
    def test_params_data_section(self):
        data = self.params["data"]
        for field in ("hr_dir", "lr_dir", "patch_size", "batch_size"):
            assert field in data, f"Missing 'data.{field}' in params.yaml"

    @pytest.mark.skipif(
        not (_DVC_DIR / "params.yaml").exists(),
        reason="params.yaml not found",
    )
    def test_params_visualize_section(self):
        vis = self.params["visualize"]
        for field in ("num_samples", "output_dir", "diffusion_strength"):
            assert field in vis, f"Missing 'visualize.{field}' in params.yaml"

    @pytest.mark.skipif(
        not (_DVC_DIR / "dvc.yaml").exists(),
        reason="dvc.yaml not found",
    )
    def test_dvc_yaml_has_all_stages(self):
        expected = [
            "train_autoencoder",
            "train_neural_operator",
            "train_diffusion",
            "evaluate",
            "visualize",
        ]
        stages = list(self.dvc["stages"].keys())
        for name in expected:
            assert name in stages, f"Missing stage '{name}' in dvc.yaml"

    @pytest.mark.skipif(
        not (_DVC_DIR / "dvc.yaml").exists(),
        reason="dvc.yaml not found",
    )
    def test_dvc_yaml_visualize_stage_structure(self):
        vis = self.dvc["stages"]["visualize"]
        assert "cmd" in vis
        assert "visualize.py" in vis["cmd"]
        assert "deps" in vis
        assert "params" in vis
        assert "outs" in vis

    @pytest.mark.skipif(
        not (_DVC_DIR / "params.yaml").exists(),
        reason="params.yaml not found",
    )
    def test_model_instantiation_from_params(self):
        """Build the full-size model from params and verify param count."""
        ae = self.params["autoencoder"]
        no = self.params["neural_operator"]
        diff = self.params["diffusion"]
        model = HNDSR(
            ae_latent_dim=ae["latent_dim"],
            ae_downsample_ratio=ae["downsample_ratio"],
            no_width=no["width"],
            no_modes=no["modes"],
            diffusion_channels=diff["model_channels"],
            num_timesteps=diff["num_timesteps"],
        )
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params == 6_290_371, \
            f"Expected 6,290,371 params, got {n_params:,}"

    @pytest.mark.skipif(
        not all((_CKPT_DIR / f).exists() for f in [
            "autoencoder_best.pth",
            "neural_operator_best.pth",
            "diffusion_best.pth",
        ]),
        reason="Pre-trained checkpoints not found in ../checkpoints/",
    )
    def test_checkpoint_loading(self):
        """Load real checkpoints into a full-size model.

        Handles both old (raw state_dict) and new (dict-wrapped) checkpoint
        formats for neural_operator_best.pth.
        """
        ae = self.params["autoencoder"]
        no = self.params["neural_operator"]
        diff = self.params["diffusion"]
        model = HNDSR(
            ae_latent_dim=ae["latent_dim"],
            ae_downsample_ratio=ae["downsample_ratio"],
            no_width=no["width"],
            no_modes=no["modes"],
            diffusion_channels=diff["model_channels"],
            num_timesteps=diff["num_timesteps"],
        )
        device = torch.device("cpu")

        # Autoencoder (raw state_dict)
        model.autoencoder.load_state_dict(
            torch.load(_CKPT_DIR / "autoencoder_best.pth", map_location=device, weights_only=True)
        )

        # Neural operator — old checkpoints store raw state_dict,
        # DVC pipeline stores {"neural_operator": ..., "implicit_amp": ...}
        no_ckpt = torch.load(_CKPT_DIR / "neural_operator_best.pth", map_location=device, weights_only=True)
        if "neural_operator" in no_ckpt:
            model.neural_operator.load_state_dict(no_ckpt["neural_operator"])
            model.implicit_amp.load_state_dict(no_ckpt["implicit_amp"])
        else:
            # Old format: raw state_dict for neural_operator only
            model.neural_operator.load_state_dict(no_ckpt)

        # Diffusion (dict with 1 key)
        diff_ckpt = torch.load(_CKPT_DIR / "diffusion_best.pth", map_location=device, weights_only=True)
        if "diffusion_unet" in diff_ckpt:
            model.diffusion_unet.load_state_dict(diff_ckpt["diffusion_unet"])
        else:
            model.diffusion_unet.load_state_dict(diff_ckpt)

        # Quick forward pass
        model.eval()
        lr = torch.randn(1, 3, 16, 16)
        with torch.no_grad():
            sr = model.super_resolve(lr, scale_factor=4, diffusion_strength=0.0)
        assert sr.shape == (1, 3, 64, 64)
        assert not torch.isnan(sr).any()
