"""Photo-real (SeedVR2-3B) engines: MLX on Apple Silicon, numz's PyTorch runner on NVIDIA."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from upscaler.seedvr2.base import SeedVR2Engine, SeedVR2Unavailable

if TYPE_CHECKING:
    from upscaler.hardware import Hardware

__all__ = ["SeedVR2Engine", "SeedVR2Unavailable", "get_seedvr2_engine"]


def get_seedvr2_engine(hw: Hardware) -> SeedVR2Engine | None:
    """The engine for this computer, or None when it has no supported GPU.

    Memory eligibility (16 GB Mac / 12 GB card) is decided by ``hardware.photoreal_support``.
    """
    if hw.apple_silicon:
        return _engine("mlx")
    if hw.device == "cuda":
        return _engine("torch")
    return None


@functools.cache
def _engine(name: str) -> SeedVR2Engine:
    # One instance per process, so a loaded model is reused across images and jobs.
    if name == "mlx":
        from upscaler.seedvr2.mlx_engine import MLXSeedVR2Engine

        return MLXSeedVR2Engine()
    from upscaler.seedvr2.torch_engine import TorchSeedVR2Engine

    return TorchSeedVR2Engine()
