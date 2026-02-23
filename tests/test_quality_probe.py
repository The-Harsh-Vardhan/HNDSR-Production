"""
tests/test_quality_probe.py
===========================
Runtime quality probe tests.
"""

from __future__ import annotations

import torch

from backend.inference.quality_probe import run_quality_probe


class GoodModel(torch.nn.Module):
    def super_resolve(self, lr, scale_factor=2, num_inference_steps=10):
        return torch.nn.functional.interpolate(
            lr,
            scale_factor=scale_factor,
            mode="bicubic",
            align_corners=False,
        ).clamp(-1, 1)


class FlatModel(torch.nn.Module):
    def super_resolve(self, lr, scale_factor=2, num_inference_steps=10):
        b, _, h, w = lr.shape
        return torch.zeros(b, 3, h * scale_factor, w * scale_factor, device=lr.device)


class NanModel(torch.nn.Module):
    def super_resolve(self, lr, scale_factor=2, num_inference_steps=10):
        b, _, h, w = lr.shape
        out = torch.zeros(b, 3, h * scale_factor, w * scale_factor, device=lr.device)
        out[:, :, 0, 0] = float("nan")
        return out


def test_quality_probe_passes_for_reasonable_output():
    result = run_quality_probe(
        model=GoodModel(),
        device=torch.device("cpu"),
        scale_factor=2,
        ddim_steps=5,
        input_size=32,
        min_std=0.01,
    )
    assert result.passed is True
    assert result.finite is True
    assert result.output_std >= 0.01


def test_quality_probe_rejects_low_contrast_output():
    result = run_quality_probe(
        model=FlatModel(),
        device=torch.device("cpu"),
        scale_factor=2,
        ddim_steps=5,
        input_size=32,
        min_std=0.01,
    )
    assert result.passed is False
    assert result.finite is True
    assert "low_output_std" in (result.reason or "")


def test_quality_probe_rejects_non_finite_output():
    result = run_quality_probe(
        model=NanModel(),
        device=torch.device("cpu"),
        scale_factor=2,
        ddim_steps=5,
        input_size=32,
        min_std=0.01,
    )
    assert result.passed is False
    assert result.finite is False
    assert result.reason == "non_finite_output"
