"""
tests/test_checkpoint_manifest_validation.py
===========================================
Checkpoint manifest hash validation tests.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from backend.inference.model_loader import HNDSRModelLoader


def _write_file(path: Path, payload: bytes) -> None:
    path.write_bytes(payload)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _fresh_loader(model_dir: Path) -> HNDSRModelLoader:
    loader = HNDSRModelLoader.__new__(HNDSRModelLoader)
    loader._initialized = False
    loader.initialize(
        model_dir=model_dir,
        device=torch.device("cpu"),
        use_fp16=False,
        manifest_path=model_dir / "manifest.json",
    )
    return loader


def _prepare_checkpoints(model_dir: Path) -> dict[str, str]:
    files = {
        "autoencoder_best.pth": b"ae-bytes",
        "neural_operator_best.pth": b"no-bytes",
        "diffusion_best.pth": b"diff-bytes",
    }
    hashes: dict[str, str] = {}
    for name, payload in files.items():
        path = model_dir / name
        _write_file(path, payload)
        hashes[name] = _sha256(path)
    return hashes


def test_manifest_validation_passes_when_hashes_match(tmp_path):
    hashes = _prepare_checkpoints(tmp_path)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"files": hashes}, indent=2),
        encoding="utf-8",
    )

    loader = _fresh_loader(tmp_path)
    ok = loader.validate_checkpoint_manifest()

    assert ok is True
    assert loader.checkpoint_manifest_match is True
    assert loader.manifest_validation_error is None
    assert loader.checkpoint_hashes == hashes


def test_manifest_validation_fails_on_hash_mismatch(tmp_path):
    _prepare_checkpoints(tmp_path)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "files": {
                    "autoencoder_best.pth": "0" * 64,
                    "neural_operator_best.pth": "1" * 64,
                    "diffusion_best.pth": "2" * 64,
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    loader = _fresh_loader(tmp_path)
    ok = loader.validate_checkpoint_manifest()

    assert ok is False
    assert loader.checkpoint_manifest_match is False
    assert loader.manifest_validation_error is not None


def test_manifest_validation_fails_when_manifest_missing(tmp_path):
    _prepare_checkpoints(tmp_path)

    loader = _fresh_loader(tmp_path)
    ok = loader.validate_checkpoint_manifest()

    assert ok is False
    assert loader.checkpoint_manifest_match is False
    assert "manifest" in (loader.manifest_validation_error or "").lower()
