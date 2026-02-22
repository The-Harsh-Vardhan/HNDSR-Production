"""
tests/conftest.py
===================
Shared pytest fixtures and production acceptance thresholds for HNDSR tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Production acceptance thresholds
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AcceptanceThresholds:
    """
    Minimum quality and performance requirements for production deployment.

    These thresholds gate the staging → production promotion in the model
    registry. Any model failing these is rejected automatically.
    """
    # Quality
    MIN_PSNR_DB: float = 26.0
    MIN_SSIM: float = 0.75
    MAX_LPIPS: float = 0.30

    # Performance
    MAX_INFERENCE_LATENCY_MS: float = 3000.0  # Per tile, P95
    MAX_GPU_MEMORY_GB: float = 12.0           # Peak VRAM
    MAX_COLD_START_S: float = 60.0            # Model loading time

    # Shape contracts
    EXPECTED_SCALE_FACTOR: int = 4
    MIN_TILE_SIZE: int = 64
    MAX_TILE_SIZE: int = 512

    # Numerical stability
    MAX_NAN_RATIO: float = 0.0  # Zero tolerance for NaN outputs
    MAX_INF_RATIO: float = 0.0  # Zero tolerance for Inf outputs


THRESHOLDS = AcceptanceThresholds()


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def device():
    """Use GPU if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_lr_tensor():
    """Create a sample low-resolution input tensor (B, C, H, W)."""
    torch.manual_seed(42)
    return torch.randn(1, 3, 64, 64)


@pytest.fixture
def sample_hr_tensor():
    """Create a sample high-resolution target tensor (B, C, H, W)."""
    torch.manual_seed(42)
    return torch.randn(1, 3, 256, 256)


@pytest.fixture
def sample_latent_tensor():
    """Create a sample latent space tensor."""
    torch.manual_seed(42)
    return torch.randn(1, 128, 16, 16)


@pytest.fixture
def batch_lr_tensors():
    """Create a batch of 4 LR tensors for batch processing tests."""
    torch.manual_seed(42)
    return torch.randn(4, 3, 64, 64)


@pytest.fixture
def thresholds():
    """Return the acceptance thresholds."""
    return THRESHOLDS
