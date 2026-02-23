"""
tests/test_infer_fallback_mode.py
=================================
API behavior tests for bicubic fallback mode.
"""

from __future__ import annotations

import base64
import io
import time
from contextlib import asynccontextmanager

from fastapi.testclient import TestClient
from PIL import Image

import backend.app as app_module


def _set_noop_lifespan(monkeypatch):
    @asynccontextmanager
    async def _noop_lifespan(_app):
        yield

    monkeypatch.setattr(app_module.app.router, "lifespan_context", _noop_lifespan)


def _set_fallback_state(monkeypatch):
    state = app_module.AppState()
    state.model_loaded = True
    state.start_time = time.time()
    state.inference_mode = "bicubic_fallback"
    state.checkpoint_validated = False
    state.checkpoint_manifest_match = False
    state.fallback_reason = "checkpoint_manifest_validation_failed: test"
    state.checkpoint_hashes = {
        "autoencoder_best.pth": "a" * 64,
        "neural_operator_best.pth": "b" * 64,
        "diffusion_best.pth": "c" * 64,
    }
    monkeypatch.setattr(app_module, "state", state)
    return state


def _sample_png_b64(width: int = 8, height: int = 8) -> str:
    img = Image.new("RGB", (width, height), (30, 80, 160))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_infer_uses_bicubic_fallback_and_returns_valid_png(monkeypatch):
    _set_noop_lifespan(monkeypatch)
    _set_fallback_state(monkeypatch)

    payload = {
        "image": _sample_png_b64(),
        "scale_factor": 2,
        "ddim_steps": 10,
        "return_metadata": True,
    }

    with TestClient(app_module.app) as client:
        res = client.post("/infer", json=payload)
        assert res.status_code == 200
        body = res.json()

    assert body["width"] == 16
    assert body["height"] == 16
    assert body["scale_factor"] == 2
    assert body["metadata"]["inference_mode"] == "bicubic_fallback"
    assert "fallback_reason" in body["metadata"]

    decoded = Image.open(io.BytesIO(base64.b64decode(body["image"])))
    assert decoded.size == (16, 16)


def test_health_and_version_include_fallback_and_checkpoint_fields(monkeypatch):
    _set_noop_lifespan(monkeypatch)
    state = _set_fallback_state(monkeypatch)

    with TestClient(app_module.app) as client:
        health = client.get("/health")
        version = client.get("/version")

    assert health.status_code == 200
    h = health.json()
    assert h["inference_mode"] == "bicubic_fallback"
    assert h["checkpoint_validated"] is False

    assert version.status_code == 200
    v = version.json()
    assert v["checkpoint_manifest_match"] is False
    assert v["checkpoint_hashes"] == state.checkpoint_hashes
