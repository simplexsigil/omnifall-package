"""The entry point most users touch: :func:`load`.

``load`` is a thin wrapper around ``datasets.load_dataset`` that knows where the
video files are. It stays thin on purpose --- anything you can do with a
``datasets.Dataset`` you can still do with what comes back, because what comes
back *is* one.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from datasets import Dataset, DatasetDict, load_dataset

from ._configs import NON_SEGMENT_CONFIGS
from ._constants import HF_REPO_ID

if TYPE_CHECKING:  # pragma: no cover
    from ._video_dataset import OmniFallVideoDataset


def load(
    config: str = "labels",
    split: str | None = None,
    *,
    video: bool = False,
    video_dirs: Mapping[str, str | Path] | None = None,
    check: bool = True,
    strict: bool = False,
    download: bool = False,
    consent: bool = False,
    **kwargs: Any,
) -> Dataset | DatasetDict:
    """Load an OmniFall config, optionally resolving video file paths.

    Args:
        config: Any config served by the Hub repository, e.g. ``"of-syn"``,
            ``"of-itw"``, ``"cs"``, ``"le2i-cv"``. Run ``omnifall configs`` or
            :func:`omnifall.list_configs` for the current list.
        split: ``"train"``, ``"validation"``, ``"test"``, or ``None`` for a
            ``DatasetDict`` holding every split the config defines.
        video: Add a ``video`` column of absolute file paths.
        video_dirs: Per-dataset overrides, ``{"le2i": "/data/le2i/video"}``.
            Each value is the directory that ``path`` values are relative to.
        check: Stat each file and set ``None`` where it is absent. Turning this
            off is faster but you will only find out about a missing file when
            the dataloader tries to decode it.
        strict: Raise :class:`~omnifall.MissingVideosError` instead of warning
            when any referenced video is absent.
        download: Obtain missing component datasets before resolving. Only some
            components can be fetched automatically; see ``omnifall sources``.
        consent: Accept the OOPS license prompt non-interactively. Only
            meaningful together with ``download``.
        **kwargs: Forwarded to ``datasets.load_dataset``.

    Returns:
        A ``Dataset`` when *split* is given, otherwise a ``DatasetDict``.

    Raises:
        ValueError: If *video* is requested for a config that has no per-segment
            video counterpart (``metadata-syn``, ``framewise-syn``).

    Examples:
        >>> import omnifall
        >>> ds = omnifall.load("le2i-cs")                       # annotations only
        >>> ds = omnifall.load("of-itw", split="test", video=True)
    """
    if video and config in NON_SEGMENT_CONFIGS:
        raise ValueError(
            f"Config {config!r} is not a per-segment table, so there is nothing "
            f"to attach video paths to. Configs without video: "
            f"{sorted(NON_SEGMENT_CONFIGS)}."
        )

    ds = load_dataset(HF_REPO_ID, config, split=split, **kwargs)

    if video:
        from ._video import add_video

        ds = add_video(
            ds,
            config,
            video_dirs=video_dirs,
            check=check,
            strict=strict,
            download=download,
            consent=consent,
        )

    return ds


def load_video_dataset(
    config: str,
    split: str | None = None,
    *,
    num_frames: int = 16,
    target_fps: float = 15.0,
    sampling: str | Mapping[str, str] = "auto",
    transform: Callable | Mapping[str, Callable] | None = None,
    output_format: str = "TCHW",
    seed: int | None = None,
    on_error: str = "raise",
    **kwargs: Any,
) -> "OmniFallVideoDataset | dict[str, OmniFallVideoDataset]":
    """Load OmniFall as PyTorch dataset(s) that decode video on demand.

    Combines ``load(config, video=True)`` with
    :class:`~omnifall.OmniFallVideoDataset`.

    Args:
        config: Config name.
        split: Split name, or ``None`` for a dict of all splits.
        num_frames: Frames per clip.
        target_fps: Sampling rate used to lay out those frames in time.
        sampling: ``"random"``, ``"uniform"``, ``"center"``, or ``"auto"``
            (the default) which uses ``"random"`` for the train split and
            ``"uniform"`` everywhere else --- the usual choice, and the one that
            keeps evaluation reproducible. May also be a per-split mapping.
        transform: A callable, or a per-split mapping of them. See
            :class:`~omnifall.VideoTransform`.
        output_format: ``"TCHW"`` for HuggingFace ``transformers`` (the
            default), ``"CTHW"`` for the channels-first video convention, or
            ``"THWC"`` for raw uint8 frames.
        seed: Base seed for ``sampling="random"``. Given a seed, sampling is
            reproducible across processes and epochs.
        on_error: ``"raise"`` (default), ``"skip"`` or ``"retry"``.
        **kwargs: Forwarded to :func:`load` (``video_dirs``, ``download``, ...).

    Returns:
        One ``OmniFallVideoDataset`` when *split* is given, otherwise a dict
        keyed by split name.

    Example:
        >>> import omnifall
        >>> from torch.utils.data import DataLoader
        >>> parts = omnifall.load_video_dataset("le2i-cs", num_frames=16)
        >>> loader = DataLoader(parts["train"], batch_size=4,
        ...                     collate_fn=omnifall.collate_fn)
    """
    from ._video_dataset import OmniFallVideoDataset

    ds = load(config, split=split, video=True, **kwargs)

    def _for(name: str | None, part: Dataset) -> "OmniFallVideoDataset":
        return OmniFallVideoDataset(
            part,
            num_frames=num_frames,
            target_fps=target_fps,
            sampling=_pick(sampling, name),
            transform=_pick(transform, name),
            output_format=output_format,
            seed=seed,
            on_error=on_error,
        )

    if isinstance(ds, DatasetDict):
        return {name: _for(name, part) for name, part in ds.items()}
    return _for(split, ds)


def _pick(value: Any, split: str | None) -> Any:
    """Resolve a possibly per-split option for *split*.

    ``"auto"`` becomes ``"random"`` on the train split and ``"uniform"``
    elsewhere, which is what makes evaluation reproducible by default.
    """
    if isinstance(value, Mapping):
        if split is None:
            raise ValueError(
                "A per-split mapping was given but the split is unknown. "
                "Pass an explicit split= or a single value."
            )
        if split not in value:
            raise KeyError(
                f"No entry for split {split!r} in {sorted(value)}. "
                "Provide one for every split, or pass a single value."
            )
        return value[split]
    if value == "auto":
        return "random" if _is_train_split(split) else "uniform"
    return value


def _is_train_split(split: str | None) -> bool:
    """Whether *split* names the training split.

    ``datasets`` accepts slicing and arithmetic in split names, so the train
    split can arrive as ``"train[:80%]"`` or ``"train[:100]+train[-100:]"``. A
    bare ``== "train"`` silently turned those into deterministic sampling, which
    is the opposite of what ``"auto"`` promises and costs augmentation diversity
    without any error.
    """
    if not split:
        return False
    head = split.split("[", 1)[0].split("+", 1)[0].strip()
    return head == "train"
