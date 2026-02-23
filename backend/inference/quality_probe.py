"""
backend/inference/quality_probe.py
==================================
Startup quality sanity checks for HNDSR inference.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class QualityProbeResult:
    passed: bool
    output_std: float
    finite: bool
    latency_ms: float
    reason: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "passed": self.passed,
            "output_std": self.output_std,
            "finite": self.finite,
            "latency_ms": self.latency_ms,
            "reason": self.reason,
        }


@torch.no_grad()
def run_quality_probe(
    model: torch.nn.Module,
    device: torch.device,
    scale_factor: int = 2,
    ddim_steps: int = 10,
    input_size: int = 64,
    seed: int = 1234,
    min_std: float = 0.05,
) -> QualityProbeResult:
    """
    Run a deterministic startup probe to detect pathological model outputs.

    The probe checks:
      1) output contains only finite values
      2) output has enough contrast (std in [0,1] space)
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    lr = torch.rand(1, 3, input_size, input_size, device=device) * 2.0 - 1.0
    t0 = time.perf_counter()

    try:
        sr = model.super_resolve(
            lr,
            scale_factor=scale_factor,
            num_inference_steps=ddim_steps,
        )
    except Exception as exc:
        return QualityProbeResult(
            passed=False,
            output_std=0.0,
            finite=False,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            reason=f"probe_exception: {exc}",
        )

    latency_ms = (time.perf_counter() - t0) * 1000.0
    finite = bool(torch.isfinite(sr).all().item())
    sr01 = ((sr.clamp(-1, 1) + 1.0) / 2.0).float()
    output_std = float(sr01.std().item())

    if not finite:
        return QualityProbeResult(
            passed=False,
            output_std=output_std,
            finite=False,
            latency_ms=latency_ms,
            reason="non_finite_output",
        )

    if output_std < min_std:
        return QualityProbeResult(
            passed=False,
            output_std=output_std,
            finite=True,
            latency_ms=latency_ms,
            reason=f"low_output_std<{min_std}",
        )

    return QualityProbeResult(
        passed=True,
        output_std=output_std,
        finite=True,
        latency_ms=latency_ms,
        reason=None,
    )
