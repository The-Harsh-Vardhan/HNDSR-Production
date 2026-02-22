"""
tests/test_preprocessing.py
==============================
Unit tests for the data preprocessing / ETL pipeline.

What  : Validates image validation, downsampling, splitting, and hashing.
Why   : A bug in preprocessing (e.g., wrong downsample kernel) silently
        degrades model quality by 0.5–2.0 dB PSNR. These tests catch
        such issues before training begins.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


class TestImageValidation:
    """Test suite for image validation."""

    def test_valid_png_accepted(self, tmp_path):
        """Valid PNG files should pass validation."""
        from data_pipeline.etl_pipeline import validate_image

        img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        path = tmp_path / "valid.png"
        img.save(path)

        meta = validate_image(path)
        assert meta is not None
        assert meta.width == 256
        assert meta.height == 256
        assert meta.channels == 3
        assert len(meta.sha256) == 64  # SHA-256 hex length

    def test_corrupt_file_rejected(self, tmp_path):
        """Corrupted files should return None."""
        from data_pipeline.etl_pipeline import validate_image

        path = tmp_path / "corrupt.png"
        path.write_bytes(b"not an image at all")

        meta = validate_image(path)
        assert meta is None

    def test_oversized_image_rejected(self, tmp_path):
        """Images exceeding max_pixels should be rejected."""
        from data_pipeline.etl_pipeline import validate_image

        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        path = tmp_path / "small.png"
        img.save(path)

        # Use absurdly low limit to trigger rejection
        meta = validate_image(path, max_pixels=50 * 50)
        assert meta is None

    def test_grayscale_image_accepted(self, tmp_path):
        """Grayscale images should be accepted with channels=1."""
        from data_pipeline.etl_pipeline import validate_image

        img = Image.fromarray(np.random.randint(0, 255, (128, 128), dtype=np.uint8))
        path = tmp_path / "gray.png"
        img.save(path)

        meta = validate_image(path)
        assert meta is not None
        assert meta.channels == 1


class TestDownsampling:
    """Test suite for image downsampling."""

    def test_correct_dimensions(self, tmp_path):
        """Downsampled image should have dimensions / scale_factor."""
        from data_pipeline.etl_pipeline import downsample_image

        img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        hr_path = tmp_path / "hr.png"
        img.save(hr_path)

        out_dir = tmp_path / "lr"
        out_dir.mkdir()

        lr_path = downsample_image(hr_path, out_dir, scale_factor=4.0)
        assert lr_path is not None

        lr_img = Image.open(lr_path)
        assert lr_img.size == (64, 64)

    def test_too_small_rejected(self, tmp_path):
        """Images that would be <16px after downsampling should be rejected."""
        from data_pipeline.etl_pipeline import downsample_image

        img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
        hr_path = tmp_path / "tiny.png"
        img.save(hr_path)

        out_dir = tmp_path / "lr"
        out_dir.mkdir()

        result = downsample_image(hr_path, out_dir, scale_factor=4.0)
        assert result is None  # 32/4 = 8 < 16 → rejected

    def test_multiple_scales(self, tmp_path):
        """Multiple scale factors should all produce valid outputs."""
        from data_pipeline.etl_pipeline import downsample_image

        img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        hr_path = tmp_path / "hr.png"
        img.save(hr_path)

        for scale in [2.0, 4.0, 6.0]:
            out_dir = tmp_path / f"lr_{scale}"
            out_dir.mkdir()
            lr_path = downsample_image(hr_path, out_dir, scale_factor=scale)
            assert lr_path is not None
            lr_img = Image.open(lr_path)
            expected_w = int(256 / scale)
            assert lr_img.size[0] == expected_w


class TestSplitting:
    """Test suite for train/val/test splitting."""

    def test_correct_proportions(self):
        """Split should respect the given ratios."""
        from data_pipeline.etl_pipeline import stratified_split

        paths = [Path(f"img_{i}.png") for i in range(100)]
        splits = stratified_split(paths, ratios=(0.8, 0.1, 0.1))

        assert len(splits["train"]) == 80
        assert len(splits["val"]) == 10
        assert len(splits["test"]) == 10

    def test_no_overlap(self):
        """There should be zero overlap between splits."""
        from data_pipeline.etl_pipeline import stratified_split

        paths = [Path(f"img_{i}.png") for i in range(100)]
        splits = stratified_split(paths, ratios=(0.8, 0.1, 0.1))

        all_files = set()
        for split_files in splits.values():
            for f in split_files:
                assert f not in all_files, f"Duplicate: {f}"
                all_files.add(f)

    def test_deterministic_with_seed(self):
        """Same seed should produce identical splits."""
        from data_pipeline.etl_pipeline import stratified_split

        paths = [Path(f"img_{i}.png") for i in range(50)]
        split1 = stratified_split(paths, seed=42)
        split2 = stratified_split(paths, seed=42)

        assert split1["train"] == split2["train"]
        assert split1["val"] == split2["val"]
        assert split1["test"] == split2["test"]


class TestHashing:
    """Test suite for dataset hashing."""

    def test_same_file_same_hash(self, tmp_path):
        """Identical files should produce identical hashes."""
        from data_pipeline.etl_pipeline import compute_sha256

        content = b"test content for hashing"
        f1 = tmp_path / "f1.txt"
        f2 = tmp_path / "f2.txt"
        f1.write_bytes(content)
        f2.write_bytes(content)

        assert compute_sha256(f1) == compute_sha256(f2)

    def test_different_files_different_hash(self, tmp_path):
        """Different files should produce different hashes."""
        from data_pipeline.etl_pipeline import compute_sha256

        f1 = tmp_path / "f1.txt"
        f2 = tmp_path / "f2.txt"
        f1.write_bytes(b"content A")
        f2.write_bytes(b"content B")

        assert compute_sha256(f1) != compute_sha256(f2)
