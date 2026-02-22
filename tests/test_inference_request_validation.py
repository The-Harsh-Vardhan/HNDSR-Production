"""
tests/test_inference_request_validation.py
=========================================
Validation and normalization tests for the /infer request schema.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.app import InferenceRequest


def _payload(**overrides):
    payload = {
        "image": "ZmFrZV9pbWFnZV9wYXlsb2Fk",  # schema only validates type here
        "return_metadata": True,
    }
    payload.update(overrides)
    return payload


def test_scale_factor_none_defaults_to_four():
    req = InferenceRequest.model_validate(_payload(scale_factor=None))
    assert req.scale_factor == 4


def test_scale_factor_empty_string_defaults_to_four():
    req = InferenceRequest.model_validate(_payload(scale_factor=""))
    assert req.scale_factor == 4


def test_ddim_steps_empty_string_defaults_to_none():
    req = InferenceRequest.model_validate(_payload(ddim_steps=""))
    assert req.ddim_steps is None


def test_scale_factor_out_of_range_raises_validation_error():
    with pytest.raises(ValidationError):
        InferenceRequest.model_validate(_payload(scale_factor=1))


def test_ddim_steps_out_of_range_raises_validation_error():
    with pytest.raises(ValidationError):
        InferenceRequest.model_validate(_payload(ddim_steps=5))
