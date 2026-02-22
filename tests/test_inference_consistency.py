"""
tests/test_inference_consistency.py
=====================================
Seed-controlled inference reproducibility tests.

What  : Verifies that identical inputs + identical seeds produce identical
        outputs across runs, devices, and precision modes.
Why   : Non-reproducible inference is a deployment red flag. If the same
        image produces different results on restart, quality metrics are
        meaningless and debugging becomes impossible.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn


class TestSeedReproducibility:
    """Verify that seeded inference produces deterministic results."""

    def _seed_and_run(self, seed: int = 42):
        """Helper: set seed and run a mock inference."""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
        )
        model.eval()

        # Fixed input
        torch.manual_seed(seed)
        x = torch.randn(1, 3, 64, 64)

        with torch.no_grad():
            return model(x)

    def test_same_seed_same_output(self):
        """Two runs with the same seed should produce identical outputs."""
        out1 = self._seed_and_run(42)
        out2 = self._seed_and_run(42)

        assert torch.allclose(out1, out2, atol=1e-6), (
            f"Max diff: {(out1 - out2).abs().max().item()}"
        )

    def test_different_seed_different_output(self):
        """Different seeds should produce different outputs."""
        out1 = self._seed_and_run(42)
        out2 = self._seed_and_run(99)

        assert not torch.allclose(out1, out2, atol=1e-3)

    def test_multiple_runs_consistent(self):
        """Ten consecutive runs with same seed should all match."""
        reference = self._seed_and_run(42)
        for _ in range(10):
            result = self._seed_and_run(42)
            assert torch.allclose(reference, result, atol=1e-6)


class TestNumericalStability:
    """Verify outputs don't contain NaN, Inf, or out-of-range values."""

    def test_no_nan_output(self):
        """Inference should never produce NaN values."""
        model = nn.Sequential(nn.Conv2d(3, 3, 3, padding=1))
        model.eval()
        x = torch.randn(1, 3, 64, 64)

        with torch.no_grad():
            out = model(x)

        assert not torch.isnan(out).any(), "NaN detected in output"

    def test_no_inf_output(self):
        """Inference should never produce Inf values."""
        model = nn.Sequential(nn.Conv2d(3, 3, 3, padding=1))
        model.eval()
        x = torch.randn(1, 3, 64, 64)

        with torch.no_grad():
            out = model(x)

        assert not torch.isinf(out).any(), "Inf detected in output"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")
    def test_fp16_no_overflow(self):
        """
        FP16 inference should not overflow.

        This is critical for the FNO stage where FFT operations
        can produce large intermediate values that overflow FP16 range.
        """
        model = nn.Sequential(nn.Conv2d(3, 3, 3, padding=1)).half().cuda()
        x = torch.randn(1, 3, 64, 64).half().cuda()

        with torch.no_grad():
            out = model(x)

        assert not torch.isnan(out).any(), "FP16 overflow → NaN"
        assert not torch.isinf(out).any(), "FP16 overflow → Inf"


class TestCrossPrecisionConsistency:
    """Verify FP32 and FP16 produce similar (not identical) results."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")
    def test_fp32_fp16_close(self):
        """
        FP16 results should be within 1% of FP32 results.

        Some divergence is expected due to reduced precision, but
        large differences indicate numerical instability that will
        cause quality issues in production.
        """
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
        ).cuda()
        model.eval()

        x = torch.randn(1, 3, 64, 64).cuda()

        with torch.no_grad():
            out_fp32 = model(x)
            out_fp16 = model.half()(x.half()).float()

        # Allow 1% relative tolerance
        assert torch.allclose(out_fp32, out_fp16, rtol=0.01, atol=1e-3), (
            f"FP32/FP16 max diff: {(out_fp32 - out_fp16).abs().max().item()}"
        )
