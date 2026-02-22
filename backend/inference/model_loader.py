"""
backend/inference/model_loader.py
===================================
Thread-safe singleton model loader for the composite HNDSR model.

Builds one HNDSR() model and loads individual sub-model checkpoints
(autoencoder, neural operator, diffusion UNet) into it.

Checkpoint formats (from training notebook):
  autoencoder_best.pth   → raw state_dict (OrderedDict of tensors)
  neural_operator_best.pth → raw state_dict
  diffusion_best.pth     → {'diffusion_unet': state_dict, 'ema_shadow': state_dict}
"""

from __future__ import annotations

import gc
import logging
import threading
import time
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Device helpers
# ─────────────────────────────────────────────────────────────────────────────

def resolve_device(preference: str = "auto") -> torch.device:
    """
    Resolve the compute device.
    "auto" → CUDA if available, else CPU.
    "cuda" → CUDA; raises RuntimeError if unavailable.
    "cpu"  → Always CPU.
    """
    if preference == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Auto-selected device: %s", device)
        return device
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but not available. "
                "Set DEVICE=cpu or ensure CUDA drivers are installed."
            )
        return torch.device("cuda")
    return torch.device("cpu")


def log_memory_stats(tag: str = "") -> None:
    """Log current GPU and CPU memory usage."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info("[%s] GPU memory: allocated=%.2f GB reserved=%.2f GB",
                     tag, alloc, reserved)


def free_gpu_memory() -> None:
    """Aggressively free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    logger.debug("GPU memory freed.")


# ─────────────────────────────────────────────────────────────────────────────
# Model loader singleton
# ─────────────────────────────────────────────────────────────────────────────

class HNDSRModelLoader:
    """
    Thread-safe singleton that builds one HNDSR model and loads
    all checkpoint weights into it.

    Usage::

        loader = get_model_loader()
        loader.initialize(model_dir, device=device)
        loader.warm_start()
        model = loader.model  # full HNDSR with all weights loaded
    """

    _instance: Optional["HNDSRModelLoader"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "HNDSRModelLoader":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def initialize(
        self,
        model_dir: Path,
        autoencoder_ckpt: str = "autoencoder_best.pth",
        neural_operator_ckpt: str = "neural_operator_best.pth",
        diffusion_ckpt: str = "diffusion_best.pth",
        device: torch.device = torch.device("cpu"),
        use_fp16: bool = True,
    ) -> None:
        """Configure the loader. Call once at application startup."""
        if self._initialized:
            logger.warning("ModelLoader already initialized; ignoring re-init.")
            return

        self.model_dir = Path(model_dir)
        self.autoencoder_ckpt = autoencoder_ckpt
        self.neural_operator_ckpt = neural_operator_ckpt
        self.diffusion_ckpt = diffusion_ckpt
        self.device = device
        self.use_fp16 = use_fp16

        self._model: Optional[torch.nn.Module] = None
        self._load_lock = threading.Lock()

        self._initialized = True
        logger.info("ModelLoader initialized. model_dir=%s device=%s",
                     model_dir, device)

    # ── Internal loader ───────────────────────────────────────────────────

    def _load_full_model(self) -> torch.nn.Module:
        """Build the composite HNDSR model and load all checkpoint files."""
        from backend.model.model_stubs import HNDSR

        t0 = time.perf_counter()
        model = HNDSR()

        # ── Autoencoder ──
        ae_path = self.model_dir / self.autoencoder_ckpt
        if not ae_path.exists():
            raise FileNotFoundError(
                f"Autoencoder checkpoint not found: {ae_path}")
        ae_raw = torch.load(ae_path, map_location="cpu", weights_only=False)
        ae_state = (ae_raw.get("model_state_dict", ae_raw)
                    if isinstance(ae_raw, dict) and "model_state_dict" in ae_raw
                    else ae_raw)
        model.autoencoder.load_state_dict(ae_state, strict=True)
        logger.info("Loaded autoencoder from %s", ae_path.name)

        # ── Neural Operator ──
        no_path = self.model_dir / self.neural_operator_ckpt
        if not no_path.exists():
            raise FileNotFoundError(
                f"Neural operator checkpoint not found: {no_path}")
        no_raw = torch.load(no_path, map_location="cpu", weights_only=False)
        no_state = (no_raw.get("model_state_dict", no_raw)
                    if isinstance(no_raw, dict) and "model_state_dict" in no_raw
                    else no_raw)
        model.neural_operator.load_state_dict(no_state, strict=True)
        logger.info("Loaded neural operator from %s", no_path.name)

        # ── Diffusion UNet ──
        diff_path = self.model_dir / self.diffusion_ckpt
        if not diff_path.exists():
            raise FileNotFoundError(
                f"Diffusion checkpoint not found: {diff_path}")
        diff_raw = torch.load(diff_path, map_location="cpu", weights_only=False)

        # Prefer EMA shadow weights (moving average → better quality)
        if isinstance(diff_raw, dict) and "ema_shadow" in diff_raw:
            diff_state = diff_raw["ema_shadow"]
            logger.info("Using EMA shadow weights for diffusion UNet")
        elif isinstance(diff_raw, dict) and "diffusion_unet" in diff_raw:
            diff_state = diff_raw["diffusion_unet"]
            logger.info("Using training weights for diffusion UNet")
        elif isinstance(diff_raw, dict) and "model_state_dict" in diff_raw:
            diff_state = diff_raw["model_state_dict"]
        else:
            diff_state = diff_raw
        model.diffusion_unet.load_state_dict(diff_state, strict=True)
        logger.info("Loaded diffusion UNet from %s", diff_path.name)

        # Note: ImplicitAmplification has no saved checkpoint because it was
        # never optimised independently during training. It uses fresh random
        # init. The diffusion model's cross-attention is robust to this since
        # the context vector primarily captures global structure.
        logger.warning(
            "ImplicitAmplification uses random init (no checkpoint). "
            "Quality may differ slightly from the training session."
        )

        model.eval()
        model = model.to(self.device)

        elapsed = time.perf_counter() - t0
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        logger.info(
            "Full HNDSR model loaded in %.2f s (%.1f M params) → %s",
            elapsed, total_params, self.device,
        )
        return model

    # ── Public API ────────────────────────────────────────────────────────

    @property
    def model(self) -> torch.nn.Module:
        """Return the full HNDSR model (lazy-loaded on first access)."""
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    self._model = self._load_full_model()
        return self._model

    # Backward-compat sub-model accessors
    @property
    def autoencoder(self) -> torch.nn.Module:
        return self.model.autoencoder

    @property
    def neural_operator(self) -> torch.nn.Module:
        return self.model.neural_operator

    @property
    def diffusion_unet(self) -> torch.nn.Module:
        return self.model.diffusion_unet

    def warm_start(self) -> None:
        """Pre-load the full model. Call at server startup."""
        logger.info("Warm-starting model loader...")
        log_memory_stats("before_warm_start")
        _ = self.model
        log_memory_stats("after_warm_start")
        logger.info("All model stages loaded and ready.")

    def unload_all(self) -> None:
        """Unload all models from memory."""
        self._model = None
        free_gpu_memory()
        logger.info("All models unloaded.")


# Module-level singleton accessor
_loader = HNDSRModelLoader()


def get_model_loader() -> HNDSRModelLoader:
    """Return the global singleton model loader."""
    return _loader
