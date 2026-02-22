"""
tests/test_shape_validation.py
================================
Shape contract tests for all three HNDSR model stages.

What  : Validates input/output tensor shapes for autoencoder, neural
        operator, and diffusion UNet.
Why   : Shape mismatches are the #1 cause of silent model failures.
        A model can pass all quality tests in development but crash
        in production with a different input resolution.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Mock models for shape testing (replace with actual model imports)
# ─────────────────────────────────────────────────────────────────────────────

class MockAutoencoder(nn.Module):
    """Mock autoencoder for shape testing."""
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Conv2d(3, latent_dim, 4, stride=4)
        self.decoder = nn.ConvTranspose2d(latent_dim, 3, 4, stride=4)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))


class MockNeuralOperator(nn.Module):
    """Mock neural operator for shape testing."""
    def __init__(self, in_channels=128, out_channels=128):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.conv(x)


class MockDiffusionUNet(nn.Module):
    """Mock diffusion UNet for shape testing."""
    def __init__(self, channels=128):
        super().__init__()
        self.conv = nn.Conv2d(channels * 2, channels, 1)  # Input + condition

    def forward(self, x_noisy, t, condition):
        combined = torch.cat([x_noisy, condition], dim=1)
        return self.conv(combined)


# ─────────────────────────────────────────────────────────────────────────────
# Shape contract tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAutoencoderShapes:
    """Validate autoencoder input/output shapes."""

    @pytest.fixture
    def model(self):
        return MockAutoencoder(latent_dim=128)

    def test_encode_shape(self, model, sample_lr_tensor):
        """Encoder should reduce spatial dims by 4× and output latent_dim channels."""
        z = model.encode(sample_lr_tensor)
        B, C, H, W = sample_lr_tensor.shape
        assert z.shape == (B, 128, H // 4, W // 4)

    def test_decode_shape(self, model, sample_latent_tensor):
        """Decoder should expand spatial dims by 4× and output 3 channels."""
        x = model.decode(sample_latent_tensor)
        B, C, H, W = sample_latent_tensor.shape
        assert x.shape == (B, 3, H * 4, W * 4)

    def test_roundtrip_shape(self, model, sample_lr_tensor):
        """Encode → decode should preserve input shape."""
        output = model(sample_lr_tensor)
        assert output.shape == sample_lr_tensor.shape

    def test_batch_independence(self, model):
        """Different batch sizes should produce consistent shapes."""
        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, 3, 64, 64)
            out = model(x)
            assert out.shape[0] == batch_size
            assert out.shape[1:] == (3, 64, 64)

    def test_various_spatial_sizes(self, model):
        """Model should handle different spatial dimensions."""
        for size in [64, 128, 256]:
            x = torch.randn(1, 3, size, size)
            out = model(x)
            assert out.shape == (1, 3, size, size)


class TestNeuralOperatorShapes:
    """Validate neural operator input/output shapes."""

    @pytest.fixture
    def model(self):
        return MockNeuralOperator(in_channels=128, out_channels=128)

    def test_same_shape_io(self, model, sample_latent_tensor):
        """NO should preserve spatial dimensions (continuous mapping)."""
        output = model(sample_latent_tensor)
        assert output.shape == sample_latent_tensor.shape

    def test_channel_preservation(self, model):
        """Output channels should match expected latent dim."""
        x = torch.randn(1, 128, 16, 16)
        output = model(x)
        assert output.shape[1] == 128


class TestDiffusionUNetShapes:
    """Validate diffusion UNet input/output shapes."""

    @pytest.fixture
    def model(self):
        return MockDiffusionUNet(channels=128)

    def test_noise_prediction_shape(self, model, sample_latent_tensor):
        """UNet should output same shape as noisy input (noise prediction)."""
        condition = torch.randn_like(sample_latent_tensor)
        t = torch.randint(0, 1000, (sample_latent_tensor.shape[0],))
        output = model(sample_latent_tensor, t, condition)
        assert output.shape == sample_latent_tensor.shape

    def test_various_timesteps(self, model, sample_latent_tensor):
        """Different timesteps should all produce valid outputs."""
        condition = torch.randn_like(sample_latent_tensor)
        for t_val in [0, 100, 500, 999]:
            t = torch.full((sample_latent_tensor.shape[0],), t_val)
            output = model(sample_latent_tensor, t, condition)
            assert output.shape == sample_latent_tensor.shape


class TestEndToEndShapes:
    """Validate the full pipeline shape contracts."""

    def test_full_pipeline_shape(self):
        """LR input → SR output should have correct scale factor."""
        ae = MockAutoencoder(128)
        no = MockNeuralOperator(128, 128)

        lr_input = torch.randn(1, 3, 64, 64)

        # Encode
        z_lr = ae.encode(lr_input)
        assert z_lr.shape == (1, 128, 16, 16)

        # Neural operator
        z_sr = no(z_lr)
        assert z_sr.shape == z_lr.shape

        # Decode
        sr_output = ae.decode(z_sr)
        assert sr_output.shape == (1, 3, 64, 64)  # Same as input for mock

    def test_no_nan_in_outputs(self):
        """No stage should produce NaN values."""
        ae = MockAutoencoder(128)
        no = MockNeuralOperator(128, 128)

        x = torch.randn(1, 3, 64, 64)
        z = ae.encode(x)
        z_no = no(z)
        output = ae.decode(z_no)

        assert not torch.isnan(z).any(), "NaN in encoder output"
        assert not torch.isnan(z_no).any(), "NaN in neural operator output"
        assert not torch.isnan(output).any(), "NaN in decoder output"
