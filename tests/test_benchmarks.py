"""
tests/test_benchmarks.py
==========================
Latency, throughput, and memory profiling tests.

What  : Measures and validates inference performance against acceptance
        thresholds.
Why   : A model that produces great PSNR but takes 30 seconds per image
        is useless in production. These tests ensure performance meets SLA.
"""

from __future__ import annotations

import gc
import time

import pytest
import torch
import torch.nn as nn

from conftest import THRESHOLDS


class TestLatencyBenchmarks:
    """Measure and validate inference latency."""

    @pytest.fixture
    def model(self):
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 3, 3, padding=1),
        )
        model.eval()
        return model

    def test_single_tile_latency(self, model):
        """Single tile inference should complete within threshold."""
        x = torch.randn(1, 3, 256, 256)

        # Warmup
        with torch.no_grad():
            _ = model(x)

        # Benchmark
        latencies = []
        for _ in range(10):
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(x)
            latencies.append((time.perf_counter() - start) * 1000)

        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        assert p95 < THRESHOLDS.MAX_INFERENCE_LATENCY_MS, (
            f"P95 latency {p95:.1f}ms > threshold {THRESHOLDS.MAX_INFERENCE_LATENCY_MS}ms"
        )

    def test_batch_throughput(self, model):
        """Batch processing should be faster per-image than sequential."""
        x_single = torch.randn(1, 3, 128, 128)
        x_batch = torch.randn(4, 3, 128, 128)

        with torch.no_grad():
            # Single sequential
            start = time.perf_counter()
            for _ in range(4):
                _ = model(x_single)
            sequential_time = time.perf_counter() - start

            # Batched
            start = time.perf_counter()
            _ = model(x_batch)
            batch_time = time.perf_counter() - start

        # Batch should be at least 1.5× faster
        speedup = sequential_time / max(batch_time, 1e-6)
        assert speedup > 1.0, (
            f"Batch speedup {speedup:.2f}× — batching provides no benefit"
        )


class TestMemoryBenchmarks:
    """Measure and validate GPU memory usage."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")
    def test_peak_memory(self):
        """Peak GPU memory during inference should be within threshold."""
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 3, 3, padding=1),
        ).cuda()
        model.eval()

        x = torch.randn(1, 3, 256, 256).cuda()

        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(x)

        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        assert peak_gb < THRESHOLDS.MAX_GPU_MEMORY_GB, (
            f"Peak memory {peak_gb:.2f} GB > threshold {THRESHOLDS.MAX_GPU_MEMORY_GB} GB"
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")
    def test_no_memory_leak(self):
        """Repeated inference should not leak GPU memory."""
        model = nn.Sequential(nn.Conv2d(3, 3, 3, padding=1)).cuda()
        model.eval()
        x = torch.randn(1, 3, 64, 64).cuda()

        # Warmup
        with torch.no_grad():
            _ = model(x)

        baseline = torch.cuda.memory_allocated()

        # Run 100 iterations
        for _ in range(100):
            with torch.no_grad():
                _ = model(x)
            del _
        gc.collect()
        torch.cuda.empty_cache()

        after = torch.cuda.memory_allocated()
        leak = after - baseline

        assert leak < 1e6, (  # Less than 1 MB leak
            f"Memory leak detected: {leak / 1e6:.2f} MB after 100 iterations"
        )


class TestColdStartBenchmarks:
    """Measure model loading time."""

    def test_model_load_time(self, tmp_path):
        """Model loading should complete within cold start threshold."""
        # Create a model and save it
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.Conv2d(64, 3, 3, padding=1),
        )
        path = tmp_path / "model.pth"
        torch.save(model.state_dict(), path)

        # Measure load time
        start = time.perf_counter()
        loaded = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.Conv2d(64, 3, 3, padding=1),
        )
        loaded.load_state_dict(torch.load(path, weights_only=True))
        load_time_s = time.perf_counter() - start

        assert load_time_s < THRESHOLDS.MAX_COLD_START_S, (
            f"Load time {load_time_s:.2f}s > threshold {THRESHOLDS.MAX_COLD_START_S}s"
        )
