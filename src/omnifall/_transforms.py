"""Training-ready video transforms for :class:`~omnifall.OmniFallVideoDataset`.

:class:`VideoTransform` turns the raw ``(T, H, W, 3)`` uint8 clip produced by
:func:`omnifall._decode.decode_segment` into a normalized float tensor in either
of the two layouts used in practice:

* ``"TCHW"`` -- per sample ``(T, C, H, W)``, batching to ``(B, T, C, H, W)``.
  This is what HuggingFace video models (``VideoMAEForVideoClassification``,
  ``VideoMAEModel``, ``TimesformerForVideoClassification``, ...) require, and
  also what the ``fall-da`` training code feeds its models. It is the default.
* ``"CTHW"`` -- per sample ``(C, T, H, W)``, batching to ``(B, C, T, H, W)``.
  The pytorchvideo convention, offered because some third-party heads want it.

Requires the optional ``torch`` dependency; ``transformers`` is needed only when
*model_name* is given. This module is imported lazily by :mod:`omnifall`.

.. note::
   **Behaviour change vs. omnifall 0.1.0.** :class:`VideoMAETransform` used to
   emit ``(C, T, H, W)`` and used ``AutoImageProcessor`` in val/test mode. The
   ``(C, T, H, W)`` output was a porting mistake: ``fall-da`` permutes back to
   ``(T, C, H, W)`` right after its own transform, and that permute was dropped
   on the way into omnifall. The default is now ``output_format="TCHW"``, the
   layout HuggingFace models actually accept, with resize/centre-crop
   implemented directly. Pass ``output_format="CTHW"`` for the 0.1.0 layout.
"""

from __future__ import annotations

import math
from typing import Any, Literal, Sequence

import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "OUTPUT_FORMATS",
    "VideoMAETransform",
    "VideoTransform",
    "from_model",
]

#: ImageNet channel means, the default used by VideoMAE and Timesformer.
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
#: ImageNet channel standard deviations.
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

#: Layouts understood by the transforms and by ``OmniFallVideoDataset``.
OUTPUT_FORMATS = ("TCHW", "CTHW", "THWC")

OutputFormat = Literal["TCHW", "CTHW", "THWC"]

_PROCESSOR_CACHE: dict[str, dict[str, Any]] = {}


def _processor_config(model_name: str) -> dict[str, Any]:
    """Read image size, mean and std from a HuggingFace image processor.

    The processor is fetched once per model id and only the four constants we
    need are kept, so nothing heavy stays alive.

    Args:
        model_name: A HuggingFace model id, e.g.
            ``"MCG-NJU/videomae-small-finetuned-kinetics"``.

    Returns:
        Dict with ``image_size``, ``resize_size``, ``mean`` and ``std``.

    Raises:
        ImportError: If ``transformers`` is not installed.
    """
    if model_name in _PROCESSOR_CACHE:
        return _PROCESSOR_CACHE[model_name]

    try:
        from transformers import AutoImageProcessor
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "model_name=... requires the 'transformers' package. "
            "Install it with `pip install omnifall[video]`, or pass "
            "image_size/mean/std explicitly instead."
        ) from exc

    processor = AutoImageProcessor.from_pretrained(model_name)

    crop = _size_value(getattr(processor, "crop_size", None), ("height", "width"))
    resize = _size_value(
        getattr(processor, "size", None), ("shortest_edge", "height", "width")
    )
    if crop is None and resize is None:
        raise ValueError(
            f"Image processor for {model_name!r} declares neither a crop_size "
            "nor a size; pass image_size=... explicitly."
        )

    image_size = crop if crop is not None else resize
    config = {
        "image_size": int(image_size),
        "resize_size": int(resize if resize is not None else image_size),
        "mean": tuple(float(v) for v in processor.image_mean),
        "std": tuple(float(v) for v in processor.image_std),
    }
    _PROCESSOR_CACHE[model_name] = config
    return config


def _size_value(size: Any, keys: Sequence[str]) -> int | None:
    """Extract a single integer edge length from a processor size entry."""
    if size is None:
        return None
    for key in keys:
        value = size.get(key) if hasattr(size, "get") else getattr(size, key, None)
        if value is not None:
            return int(value)
    return None


# ---------------------------------------------------------------------------
# Spatial primitives -- all operate on float ``(T, C, H, W)`` tensors
# ---------------------------------------------------------------------------


def _short_side_scale(
    clip: torch.Tensor, size: int, antialias: bool = True
) -> torch.Tensor:
    """Bilinearly scale the shorter spatial side of a ``(T, C, H, W)`` clip.

    Args:
        clip: Float clip in ``(T, C, H, W)`` layout.
        size: Target length of the shorter spatial side.
        antialias: Whether to low-pass filter when downscaling. This is what
            makes the output match PIL -- and therefore HuggingFace's image
            processors -- rather than differing by a few percent.

    Returns:
        The rescaled clip.
    """
    _, _, height, width = clip.shape
    if height == width == size:
        return clip
    if width < height:
        new_h, new_w = int(math.floor((height / width) * size)), size
    else:
        new_h, new_w = size, int(math.floor((width / height) * size))
    return F.interpolate(
        clip,
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
        antialias=antialias,
    )


def _center_crop(clip: torch.Tensor, size: int) -> torch.Tensor:
    """Centre-crop a ``(T, C, H, W)`` clip to ``size x size``."""
    _, _, height, width = clip.shape
    top = max(0, (height - size) // 2)
    left = max(0, (width - size) // 2)
    return clip[..., top : top + size, left : left + size]


def _random_crop(
    clip: torch.Tensor, size: int, rng: np.random.Generator
) -> torch.Tensor:
    """Crop a ``(T, C, H, W)`` clip at one random position shared by all frames."""
    _, _, height, width = clip.shape
    top = int(rng.integers(0, max(1, height - size + 1)))
    left = int(rng.integers(0, max(1, width - size + 1)))
    return clip[..., top : top + size, left : left + size]


def _normalize(
    clip: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    """Normalize a ``(T, C, H, W)`` clip with per-channel *mean* / *std*."""
    return (clip - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)


def _to_tchw_float(frames: Any) -> torch.Tensor:
    """Convert decoded frames to a float ``(T, C, H, W)`` tensor scaled to [0, 1].

    Args:
        frames: ``(T, H, W, 3)`` uint8 array (what ``decode_segment`` returns), a
            list of ``(H, W, 3)`` uint8 arrays, or an equivalent tensor.

    Returns:
        Float tensor of shape ``(T, C, H, W)`` with values in ``[0, 1]``.

    Raises:
        ValueError: If the input is not a 4D clip in ``(T, H, W, 3)`` layout.
    """
    if isinstance(frames, torch.Tensor):
        clip = frames
    else:
        clip = torch.from_numpy(np.ascontiguousarray(np.stack(frames)))

    if clip.ndim != 4 or clip.shape[-1] != 3:
        raise ValueError(
            "Expected frames in (T, H, W, 3) layout as produced by "
            f"omnifall._decode.decode_segment, got shape {tuple(clip.shape)}."
        )

    clip = clip.permute(0, 3, 1, 2).contiguous()
    if clip.dtype == torch.uint8:
        return clip.float().div_(255.0)
    return clip.float()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class VideoTransform:
    """Resize / crop / normalize a decoded clip for a video classifier.

    In ``"train"`` mode: random short-side scale in ``[min_scale, max_scale]``,
    a random crop shared by all frames, a random horizontal flip shared by all
    frames, then normalization. In ``"val"`` / ``"test"`` mode: deterministic
    short-side resize followed by a centre crop, then normalization -- the same
    pipeline ``VideoMAEImageProcessor`` applies, implemented directly so that no
    checkpoint has to be downloaded just to read four constants. With the
    default ``antialias=True`` the result matches that processor to within uint8
    rounding.

    Args:
        mode: ``"train"`` for the randomized pipeline, ``"val"`` or ``"test"``
            for the deterministic one.
        model_name: Optional HuggingFace model id. When given, *image_size*,
            *mean* and *std* default to that model's image-processor config
            instead of the ImageNet defaults. Explicit arguments always win.
        image_size: Spatial crop size. Defaults to 224, or to the model's crop
            size when *model_name* is given.
        mean: Per-channel mean. Defaults to :data:`IMAGENET_MEAN`, or to the
            model's when *model_name* is given.
        std: Per-channel standard deviation. Defaults to :data:`IMAGENET_STD`,
            or to the model's when *model_name* is given.
        output_format: ``"TCHW"`` (default, what HuggingFace models expect) or
            ``"CTHW"`` (what ``fall-da`` expects). Exposed as
            :attr:`output_format` so ``OmniFallVideoDataset`` knows the layout
            without guessing.
        min_scale: Lower bound of the random short-side scale (train only).
        max_scale: Upper bound of the random short-side scale (train only).
        resize_size: Short-side target for the deterministic val/test resize.
            Defaults to *image_size* (which is what the VideoMAE processor
            does), or to the model's ``size.shortest_edge`` when *model_name*
            is given.
        antialias: Low-pass filter when downscaling. On by default, which makes
            val/test output match ``VideoMAEImageProcessor`` to within uint8
            rounding (measured max absolute difference 0.018 on a value range
            of about 4.7); turning it off reproduces the un-filtered
            ``F.interpolate`` behaviour and drifts by up to 1.9.

    Raises:
        ValueError: For an unknown *mode*, *output_format*, or an inverted
            scale range.

    Example::

        train_tf = VideoTransform("train")
        val_tf = VideoTransform("val")
        ds = OmniFallVideoDataset(hf_ds, transform=val_tf, sampling="uniform")
    """

    def __init__(
        self,
        mode: str = "train",
        *,
        model_name: str | None = None,
        image_size: int | None = None,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
        output_format: OutputFormat = "TCHW",
        min_scale: int | None = None,
        max_scale: int | None = None,
        resize_size: int | None = None,
        antialias: bool = True,
    ) -> None:
        if mode not in ("train", "val", "test"):
            raise ValueError(f"mode must be 'train', 'val' or 'test', got {mode!r}.")
        if output_format not in ("TCHW", "CTHW"):
            raise ValueError(
                f"output_format must be 'TCHW' or 'CTHW', got {output_format!r}. "
                "Use OmniFallVideoDataset(output_format='THWC', transform=None) "
                "for raw uint8 clips."
            )

        config = _processor_config(model_name) if model_name is not None else None

        self.mode = mode
        self.model_name = model_name
        self.image_size = int(
            image_size
            if image_size is not None
            else (config["image_size"] if config else 224)
        )
        self.resize_size = int(
            resize_size
            if resize_size is not None
            else (config["resize_size"] if config else self.image_size)
        )
        self.mean = tuple(
            float(v)
            for v in (
                mean
                if mean is not None
                else (config["mean"] if config else IMAGENET_MEAN)
            )
        )
        self.std = tuple(
            float(v)
            for v in (
                std if std is not None else (config["std"] if config else IMAGENET_STD)
            )
        )
        self.output_format: str = output_format
        # Scale bounds default to the VideoMAE ratios (224 -> 256..320) applied
        # to whatever crop size is in force. Hard-coding 256/320 meant a 384px
        # checkpoint scaled its short side *below* the crop, so every sample came
        # out a different, wrong shape and only failed later inside collate.
        self.min_scale = int(
            min_scale if min_scale is not None else round(self.image_size * 256 / 224)
        )
        self.max_scale = int(
            max_scale if max_scale is not None else round(self.image_size * 320 / 224)
        )
        self.antialias = bool(antialias)

        if self.min_scale > self.max_scale:
            raise ValueError(
                f"min_scale ({self.min_scale}) must not exceed max_scale "
                f"({self.max_scale})."
            )
        if self.min_scale < self.image_size:
            raise ValueError(
                f"min_scale ({self.min_scale}) is smaller than the crop size "
                f"({self.image_size}), so a random crop cannot be taken. Raise "
                "min_scale, or lower image_size."
            )
        if self.resize_size < self.image_size:
            raise ValueError(
                f"resize_size ({self.resize_size}) is smaller than the crop size "
                f"({self.image_size}); the centre crop would exceed the resized "
                "frame and silently return a non-square clip. Set resize_size >= "
                "image_size."
            )
        if len(self.mean) != 3 or len(self.std) != 3:
            raise ValueError(
                f"mean and std must have 3 entries, got {self.mean} / {self.std}."
            )
        if any(s == 0 for s in self.std):
            raise ValueError(f"std must not contain zeros, got {self.std}.")

        self._mean = torch.tensor(self.mean, dtype=torch.float32)
        self._std = torch.tensor(self.std, dtype=torch.float32)

    def __call__(
        self, frames: Any, *, rng: np.random.Generator | None = None
    ) -> dict[str, torch.Tensor]:
        """Apply the transform to one decoded clip.

        Args:
            frames: ``(T, H, W, 3)`` uint8 array (or a list of ``(H, W, 3)``
                frames) as returned by ``decode_segment``.
            rng: Random source for the train-mode augmentations.
                ``OmniFallVideoDataset`` passes its per-item generator here, so
                that ``seed=...`` makes augmentation reproducible too. When
                ``None``, a generator seeded from torch's global RNG is used,
                which ``DataLoader`` seeds per worker and per epoch.

        Returns:
            ``{"pixel_values": tensor}`` in :attr:`output_format`.
        """
        clip = _to_tchw_float(frames)

        if self.mode == "train":
            if rng is None:
                rng = np.random.default_rng(
                    int(torch.randint(0, 2**31 - 1, (1,)).item())
                )
            scale = int(rng.integers(self.min_scale, self.max_scale + 1))
            clip = _short_side_scale(clip, scale, self.antialias)
            clip = _random_crop(clip, self.image_size, rng)
            if rng.random() < 0.5:
                clip = torch.flip(clip, dims=[-1])
        else:
            clip = _short_side_scale(clip, self.resize_size, self.antialias)
            clip = _center_crop(clip, self.image_size)

        clip = _normalize(clip, self._mean, self._std)

        if self.output_format == "CTHW":
            clip = clip.permute(1, 0, 2, 3).contiguous()
        return {"pixel_values": clip}

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(mode={self.mode!r}, "
            f"image_size={self.image_size}, output_format={self.output_format!r}, "
            f"model_name={self.model_name!r})"
        )


class VideoMAETransform(VideoTransform):
    """Backwards-compatible alias of :class:`VideoTransform`.

    Kept so that code written against omnifall 0.1.0 keeps importing. The old
    keyword names ``min_size``, ``max_size`` and ``crop_size`` are still
    accepted.

    .. warning::
       The default layout changed from ``(C, T, H, W)`` to ``(T, C, H, W)``,
       because ``(C, T, H, W)`` is rejected by HuggingFace video models. Pass
       ``output_format="CTHW"`` to get the old behaviour. Val/test mode no
       longer instantiates ``AutoImageProcessor``; with the default ImageNet
       mean/std the produced values are unchanged, since the old code
       overwrote the processor's constants with exactly those.

    Args:
        mode: ``"train"``, ``"val"`` or ``"test"``.
        min_size: Deprecated name for ``min_scale``.
        max_size: Deprecated name for ``max_scale``.
        crop_size: Deprecated name for ``image_size``.
        **kwargs: Forwarded to :class:`VideoTransform`.
    """

    def __init__(
        self,
        mode: str = "train",
        *,
        min_size: int | None = None,
        max_size: int | None = None,
        crop_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        if min_size is not None:
            kwargs.setdefault("min_scale", min_size)
        if max_size is not None:
            kwargs.setdefault("max_scale", max_size)
        if crop_size is not None:
            kwargs.setdefault("image_size", crop_size)
        super().__init__(mode, **kwargs)


def _from_model_classmethod(
    cls: type[VideoTransform],
    model_name: str,
    mode: str = "val",
    **kwargs: Any,
) -> VideoTransform:
    """``VideoTransform.from_model(...)``, mirroring the module-level function.

    Both spellings exist because both read naturally: the function form matches
    ``omnifall.from_model(...)``, and the classmethod matches the
    ``from_pretrained`` convention people already have in their fingers.
    """
    return cls(mode, model_name=model_name, **kwargs)


VideoTransform.from_model = classmethod(_from_model_classmethod)  # type: ignore[attr-defined]


def from_model(model_name: str, mode: str = "val", **kwargs: Any) -> VideoTransform:
    """Build a :class:`VideoTransform` matching a HuggingFace checkpoint.

    Image size, mean and std are read from the model's image-processor config.

    Args:
        model_name: HuggingFace model id, e.g.
            ``"MCG-NJU/videomae-small-finetuned-kinetics"``.
        mode: ``"train"``, ``"val"`` or ``"test"``.
        **kwargs: Forwarded to :class:`VideoTransform`.

    Returns:
        A configured :class:`VideoTransform`.

    Example::

        tf = omnifall.from_model("MCG-NJU/videomae-small-finetuned-kinetics")
        ds = OmniFallVideoDataset(hf_ds, transform=tf, num_frames=16)
    """
    return VideoTransform(mode, model_name=model_name, **kwargs)
