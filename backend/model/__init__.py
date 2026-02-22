# HNDSR Model Architecture
from backend.model.model_stubs import (
    HNDSR,
    HNDSRModel,
    LatentAutoencoder,
    NeuralOperator,
    DiffusionUNet,
    DDPMScheduler,
    ImplicitAmplification,
    # Backward-compat aliases
    HNDSRAutoencoder,
    HNDSRNeuralOperator,
    HNDSRDiffusionUNet,
)

__all__ = [
    "HNDSR",
    "HNDSRModel",
    "LatentAutoencoder",
    "NeuralOperator",
    "DiffusionUNet",
    "DDPMScheduler",
    "ImplicitAmplification",
    "HNDSRAutoencoder",
    "HNDSRNeuralOperator",
    "HNDSRDiffusionUNet",
]
