"""
backend/inference/generate_checkpoints.py
==========================================
Generate random-initialized HNDSR checkpoint files for pipeline validation.

This script creates .pth files containing the model state_dict with random
weights. These checkpoints prove the full inference pipeline works end-to-end
(GPU ops, DDIM sampling, tile processing) WITHOUT trained weights.

Output will be noise-like (untrained model), which is DOCUMENTED:
"The inference pipeline is validated; training requires the Kaggle dataset."

Usage:
    cd "HNDSR in Production"
    python -m backend.inference.generate_checkpoints

    # Or specify output directory:
    python -m backend.inference.generate_checkpoints --output-dir ./checkpoints
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.model.model_stubs import (
    HNDSRAutoencoder,
    HNDSRNeuralOperator,
    HNDSRDiffusionUNet,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def generate_checkpoints(output_dir: Path) -> None:
    """Generate random-init checkpoint files for all 3 HNDSR stages."""
    output_dir.mkdir(parents=True, exist_ok=True)

    models = {
        "autoencoder_best.pth": HNDSRAutoencoder(in_ch=3, latent_ch=64),
        "neural_operator_best.pth": HNDSRNeuralOperator(in_ch=3, latent_ch=64, fno_layers=4, modes=12),
        "diffusion_best.pth": HNDSRDiffusionUNet(latent_ch=64, t_dim=128),
    }

    total_params = 0
    for filename, model in models.items():
        param_count = sum(p.numel() for p in model.parameters())
        total_params += param_count
        filepath = output_dir / filename

        # Save as state_dict (not full model — safe against arbitrary code execution)
        torch.save(
            {"model_state_dict": model.state_dict()},
            filepath,
        )

        size_mb = filepath.stat().st_size / 1e6
        logger.info(
            "Saved %s: %.1fM params, %.1f MB",
            filename, param_count / 1e6, size_mb,
        )

    logger.info(
        "Total: %.1fM params across 3 stages",
        total_params / 1e6,
    )
    logger.info("Checkpoints saved to: %s", output_dir.resolve())

    # Verification: try loading one back
    logger.info("Verifying checkpoint loading...")
    test_model = HNDSRAutoencoder()
    raw = torch.load(output_dir / "autoencoder_best.pth", weights_only=True)
    test_model.load_state_dict(raw["model_state_dict"])
    test_model.eval()

    # Quick forward pass test
    dummy = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        recon, latent = test_model(dummy)
    logger.info(
        "Verification PASSED: input=%s → latent=%s → recon=%s",
        list(dummy.shape), list(latent.shape), list(recon.shape),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate HNDSR checkpoint files")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent / "checkpoints",
        help="Directory to save checkpoint files",
    )
    args = parser.parse_args()
    generate_checkpoints(args.output_dir)
