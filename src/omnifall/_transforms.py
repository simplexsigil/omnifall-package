"""Optional training-ready transform presets for video datasets.

Provides ``VideoMAETransform`` as a ready-made transform compatible with
``OmniFallVideoDataset``. Uses VideoMAE image processor from transformers
for val/test, and pytorchvideo-style augmentations for training.

Requires: ``transformers``, ``torchvision``, ``torch``.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, RandomCrop


# ---------------------------------------------------------------------------
# Lightweight video transforms (ported from pytorchvideo / fall-da)
# ---------------------------------------------------------------------------


def _short_side_scale(x: torch.Tensor, size: int) -> torch.Tensor:
    """Scale shorter spatial side of a CTHW tensor to *size*."""
    c, t, h, w = x.shape
    if w < h:
        new_h = int(math.floor((float(h) / w) * size))
        new_w = size
    else:
        new_h = size
        new_w = int(math.floor((float(w) / h) * size))
    return F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)


class _ToTensorVideo:
    """Convert uint8 ``(T, H, W, C)`` tensor to float ``(C, T, H, W)``."""

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        return clip.float().permute(3, 0, 1, 2) / 255.0


class _RandomShortSideScale(torch.nn.Module):
    """Randomly scale shorter side to a size in ``[min_size, max_size]``."""

    def __init__(self, min_size: int, max_size: int) -> None:
        super().__init__()
        self._min_size = min_size
        self._max_size = max_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = torch.randint(self._min_size, self._max_size + 1, (1,)).item()
        return _short_side_scale(x, size)


class _NormalizeVideo(torch.nn.Module):
    """Normalize CTHW video clip by mean/std (applied per-frame)."""

    def __init__(
        self, mean: Sequence[float], std: Sequence[float]
    ) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(-1, 1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(-1, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device)) / self.std.to(x.device)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class VideoMAETransform:
    """Training-ready transform for VideoMAE-style models.

    In **train** mode applies random augmentations (short-side scale, random
    crop, normalization). In **val**/**test** mode uses the HuggingFace
    ``VideoMAEImageProcessor`` directly.

    This is designed to be passed as the ``transform`` argument to
    ``OmniFallVideoDataset``.

    Args:
        mode: One of ``"train"``, ``"val"``, ``"test"``.
        model_name: HuggingFace model id for the image processor.
        mean: Per-channel mean for normalization.
        std: Per-channel std for normalization.
        min_size: Min short-side for random scale (train only).
        max_size: Max short-side for random scale (train only).
        crop_size: Spatial crop size.

    Example::

        transform = VideoMAETransform(mode="train")
        vds = OmniFallVideoDataset(hf_ds, transform=transform)
    """

    def __init__(
        self,
        mode: str = "train",
        model_name: str = "MCG-NJU/videomae-small-finetuned-kinetics",
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
        min_size: int = 256,
        max_size: int = 320,
        crop_size: int = 224,
    ) -> None:
        self.mode = mode
        self.mean = mean
        self.std = std

        if mode == "train":
            self._transform = Compose(
                [
                    _ToTensorVideo(),
                    _RandomShortSideScale(min_size=min_size, max_size=max_size),
                    RandomCrop((crop_size, crop_size)),
                    _NormalizeVideo(mean=list(mean), std=list(std)),
                ]
            )
        else:
            from transformers import AutoImageProcessor

            self._processor = AutoImageProcessor.from_pretrained(model_name)
            self._processor.image_mean = list(mean)
            self._processor.image_std = list(std)

    def __call__(self, frames: list[np.ndarray]) -> dict:
        """Transform a list of ``[H, W, C]`` uint8 frames.

        Args:
            frames: List of T numpy arrays, each ``(H, W, 3)`` uint8.

        Returns:
            Dict with ``"pixel_values"`` tensor of shape ``(C, T, H, W)``.
        """
        if self.mode == "train":
            stacked = torch.tensor(np.stack(frames))  # (T, H, W, C)
            pixel_values = self._transform(stacked)  # (C, T, H, W)
            # Permute to (T, C, H, W) then back to (C, T, H, W) for consistency
            # The train transform already outputs (C, T, H, W)
            return {"pixel_values": pixel_values}
        else:
            inputs = self._processor(frames, return_tensors="pt")
            # processor returns (B, T, C, H, W), rearrange to (C, T, H, W)
            pv = inputs["pixel_values"]  # (1, T, C, H, W)
            pv = pv.squeeze(0)  # (T, C, H, W)
            pv = pv.permute(1, 0, 2, 3)  # (C, T, H, W)
            return {"pixel_values": pv}

    def __repr__(self) -> str:
        return (
            f"VideoMAETransform(mode={self.mode!r}, "
            f"mean={self.mean}, std={self.std})"
        )
