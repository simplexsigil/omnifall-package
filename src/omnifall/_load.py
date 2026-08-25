"""Core load function wrapping datasets.load_dataset with video support."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset

from ._constants import HF_REPO_ID
from ._video import add_video


def load(
    config: str = "labels",
    split: str | None = None,
    *,
    video: bool = False,
    syn_video_dir: str | Path | None = None,
    oops_video_dir: str | Path | None = None,
    download_syn: bool = True,
    consent: bool = False,
    **kwargs: Any,
) -> Dataset | DatasetDict:
    """Load an OmniFall dataset config, optionally with video file paths.

    This is the primary entry point for loading OmniFall data. It wraps
    ``datasets.load_dataset()`` and optionally adds a ``video`` column
    containing absolute file paths to the video files.

    When the ``OMNIFALL_ROOT`` environment variable is set, video paths are
    resolved locally as ``{OMNIFALL_ROOT}/{dataset}/video/{path}.mp4`` and
    no downloads are performed.

    Args:
        config: Dataset config name (e.g., "of-syn", "of-itw", "of-sta-cs").
            Defaults to "labels" (all staged + OOPS labels).
        split: Optional split name ("train", "validation", "test").
            If None, returns a DatasetDict with all available splits.
        video: If True, add a ``video`` column with absolute file paths.
            OF-Syn videos are auto-downloaded; OOPS videos are auto-prepared
            with an interactive license consent prompt (skip with ``consent=True``).
            With ``OMNIFALL_ROOT``, all datasets are supported without downloads.
        syn_video_dir: Custom directory for OF-Syn videos.
            Ignored when ``OMNIFALL_ROOT`` is set.
        oops_video_dir: Custom directory for OOPS videos.
            Ignored when ``OMNIFALL_ROOT`` is set.
        download_syn: If True (default), auto-download OF-Syn videos.
            Ignored when ``OMNIFALL_ROOT`` is set.
        consent: If True, skip the interactive OOPS license consent prompt
            during auto-preparation.
        **kwargs: Additional keyword arguments passed to
            ``datasets.load_dataset()``.

    Returns:
        A Dataset (if split specified) or DatasetDict.

    Examples:
        >>> import omnifall
        >>> ds = omnifall.load("of-syn")
        >>> ds = omnifall.load("of-syn", video=True)
        >>> ds = omnifall.load("of-itw", split="test", video=True)
    """
    ds = load_dataset(HF_REPO_ID, config, split=split, **kwargs)

    if video:
        ds = add_video(
            ds,
            config,
            syn_video_dir=syn_video_dir,
            oops_video_dir=oops_video_dir,
            download_syn=download_syn,
            consent=consent,
        )

    return ds


def load_video_dataset(
    config: str,
    split: str | None = None,
    *,
    target_fps: float = 15.0,
    num_frames: int = 16,
    transform: Callable | None = None,
    fast: bool = True,
    **kwargs: Any,
) -> "OmniFallVideoDataset | dict[str, OmniFallVideoDataset]":
    """Load OmniFall as a PyTorch video dataset ready for ``DataLoader``.

    Convenience function combining ``load(config, video=True)`` with
    ``OmniFallVideoDataset`` construction.

    Args:
        config: Dataset config name (e.g., ``"cmdfall-cs"``, ``"of-syn"``).
        split: Optional split name. If None, returns a dict mapping split
            names to ``OmniFallVideoDataset`` instances.
        target_fps: Target FPS for frame sampling.
        num_frames: Number of frames per segment.
        transform: Optional transform callable.
        fast: Use fast PTS-based video loading.
        **kwargs: Passed through to ``omnifall.load()`` (e.g.,
            ``syn_video_dir``, ``consent``).

    Returns:
        An ``OmniFallVideoDataset`` (if *split* given) or a dict of them.

    Example::

        datasets = omnifall.load_video_dataset("cmdfall-cs", target_fps=15, num_frames=16)
        train_ds = datasets["train"]
        loader = DataLoader(train_ds, batch_size=4, collate_fn=omnifall.collate_fn)
    """
    from ._video_dataset import OmniFallVideoDataset

    ds = load(config, split=split, video=True, **kwargs)

    if isinstance(ds, DatasetDict):
        return {
            name: OmniFallVideoDataset(
                split_ds,
                target_fps=target_fps,
                num_frames=num_frames,
                transform=transform,
                fast=fast,
            )
            for name, split_ds in ds.items()
        }
    return OmniFallVideoDataset(
        ds,
        target_fps=target_fps,
        num_frames=num_frames,
        transform=transform,
        fast=fast,
    )
