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
    HNDSR,
    LatentAutoencoder,
    NeuralOperator,
    DiffusionUNet,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def generate_checkpoints(output_dir: Path) -> None:
    """Generate random-init checkpoint files matching training save format."""
    output_dir.mkdir(parents=True, exist_ok=True)

    model = HNDSR()

    # Save autoencoder as raw state_dict (matches training notebook)
    ae_path = output_dir / "autoencoder_best.pth"
    torch.save(model.autoencoder.state_dict(), ae_path)
    ae_params = sum(p.numel() for p in model.autoencoder.parameters())
    logger.info("Saved autoencoder_best.pth: %.1fM params, %.1f MB",
                ae_params / 1e6, ae_path.stat().st_size / 1e6)

    # Save neural operator as raw state_dict
    no_path = output_dir / "neural_operator_best.pth"
    torch.save(model.neural_operator.state_dict(), no_path)
    no_params = sum(p.numel() for p in model.neural_operator.parameters())
    logger.info("Saved neural_operator_best.pth: %.1fM params, %.1f MB",
                no_params / 1e6, no_path.stat().st_size / 1e6)

    # Save diffusion as nested dict (matches training notebook format)
    diff_path = output_dir / "diffusion_best.pth"
    torch.save({
        "diffusion_unet": model.diffusion_unet.state_dict(),
        "ema_shadow": model.diffusion_unet.state_dict(),  # use same for dummy
    }, diff_path)
    diff_params = sum(p.numel() for p in model.diffusion_unet.parameters())
    logger.info("Saved diffusion_best.pth: %.1fM params, %.1f MB",
                diff_params / 1e6, diff_path.stat().st_size / 1e6)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Total: %.1fM params across full HNDSR model", total_params / 1e6)
    logger.info("Checkpoints saved to: %s", output_dir.resolve())

    # Verification: load back and test forward pass
    logger.info("Verifying checkpoint loading...")
    from backend.inference.model_loader import HNDSRModelLoader
    loader = HNDSRModelLoader.__new__(HNDSRModelLoader)
    loader._initialized = False
    loader.initialize(
        model_dir=output_dir,
        device=torch.device("cpu"),
        use_fp16=False,
    )
    test_model = loader._load_full_model()
    dummy = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out = test_model.super_resolve(dummy, scale_factor=2, num_inference_steps=2)
    logger.info(
        "Verification PASSED: input=%s → output=%s",
        list(dummy.shape), list(out.shape),
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
