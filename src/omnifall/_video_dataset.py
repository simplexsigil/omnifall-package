"""PyTorch video datasets over an OmniFall HuggingFace ``Dataset``.

Each row of an OmniFall dataset is one **temporal segment** of a video file
(``path``, ``label``, ``start``, ``end``, ``subject``, ``cam``, ``dataset``),
plus a ``video`` column with the absolute file path added by
:func:`omnifall.add_video`. Many segments share the same file.

:class:`OmniFallVideoDataset` decodes one segment per item;
:class:`MultiOmniFallDataset` concatenates several of them while keeping track
of which domain each item came from.

Requires the optional ``torch``, ``av`` and ``numpy`` dependencies. This module
is imported lazily by :mod:`omnifall`, never at package import time.
"""

from __future__ import annotations

import inspect
import logging
import warnings
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from ._decode import SAMPLING_STRATEGIES, VideoDecodeError, decode_segment, probe
from ._label_maps import IDX2LABEL

logger = logging.getLogger(__name__)

__all__ = [
    "ERROR_POLICIES",
    "OUTPUT_FORMATS",
    "VideoUnavailableError",
    "MultiOmniFallDataset",
    "OmniFallVideoDataset",
]

#: Per-sample layouts ``OmniFallVideoDataset`` can emit.
OUTPUT_FORMATS = ("TCHW", "CTHW", "THWC")

#: Accepted values of the ``on_error`` argument.
ERROR_POLICIES = ("raise", "skip", "retry")

OutputFormat = Literal["TCHW", "CTHW", "THWC"]
ErrorPolicy = Literal["raise", "skip", "retry"]


class VideoUnavailableError(FileNotFoundError):
    """Raised when a row has no video file at all (``video`` column is ``None``).

    This is a data-preparation problem, not a transient decoding failure, so it
    is raised regardless of the ``on_error`` policy: silently skipping rows
    whose media was never downloaded would misreport the size of an evaluation.
    """


def _infer_layout(shape: tuple[int, ...]) -> str | None:
    """Infer the layout of a 4D clip tensor from its shape.

    Args:
        shape: The tensor shape.

    Returns:
        ``"TCHW"``, ``"CTHW"``, ``"THWC"``, or ``None`` when more than one
        layout is consistent with *shape* (which happens for a 3-frame clip,
        where ``(3, 3, H, W)`` is genuinely ambiguous).
    """
    if len(shape) != 4:
        return None
    candidates = set()
    if shape[0] == 3:
        candidates.add("CTHW")
    if shape[1] == 3:
        candidates.add("TCHW")
    if shape[3] == 3:
        candidates.add("THWC")
    if len(candidates) == 1:
        return candidates.pop()
    return None


_PERMUTATIONS: dict[tuple[str, str], tuple[int, ...]] = {
    ("THWC", "TCHW"): (0, 3, 1, 2),
    ("THWC", "CTHW"): (3, 0, 1, 2),
    ("TCHW", "CTHW"): (1, 0, 2, 3),
    ("TCHW", "THWC"): (0, 2, 3, 1),
    ("CTHW", "TCHW"): (1, 0, 2, 3),
    ("CTHW", "THWC"): (1, 2, 3, 0),
}


def _convert_layout(clip: torch.Tensor, src: str, dst: str) -> torch.Tensor:
    """Permute *clip* from layout *src* to layout *dst*."""
    if src == dst:
        return clip
    try:
        perm = _PERMUTATIONS[(src, dst)]
    except KeyError as exc:  # pragma: no cover - guarded by validation
        raise ValueError(f"Cannot convert layout {src!r} -> {dst!r}.") from exc
    return clip.permute(*perm).contiguous()


class OmniFallVideoDataset(Dataset):
    """Decode OmniFall video segments into model-ready tensors.

    Args:
        hf_dataset: A ``datasets.Dataset`` from
            ``omnifall.load(config, video=True)``. Must have the columns
            ``video``, ``label``, ``start`` and ``end``.
        num_frames: Number of frames per segment.
        target_fps: Sampling rate of the extracted clip, in frames per second.
            Used by ``sampling="center"`` and ``"random"``; ignored by
            ``"uniform"``.
        sampling: Temporal sampling strategy. ``"uniform"`` (default) spreads
            the frames evenly over the whole annotated segment and is
            deterministic -- the right choice for validation and test.
            ``"center"`` takes a deterministic ``(num_frames - 1) / target_fps``
            second window from the middle of the segment. ``"random"`` takes the
            same window at a random offset and is the right choice for training.
        transform: Optional callable applied to the decoded
            ``(T, H, W, 3)`` uint8 clip. May return a tensor or a dict
            containing ``"pixel_values"``. If it exposes an ``output_format``
            attribute (as the transforms in this package do), that attribute
            decides the layout it produced; otherwise the layout is inferred
            from the tensor shape, and if that is ambiguous the tensor is
            assumed to already be in *output_format*. A transform whose
            ``__call__`` accepts an ``rng`` keyword is handed this dataset's
            per-item generator, which makes its augmentations reproducible too.
        seed: Base seed for ``sampling="random"`` and for random transforms.
            With a seed, item *i* always gets the same clip, in any process and
            with any number of workers. With ``None``, the per-item generator is
            seeded from torch's global RNG, which ``DataLoader`` reseeds per
            worker and per epoch.
        on_error: What to do when a segment cannot be decoded.
            ``"raise"`` (default) propagates the
            :class:`~omnifall._decode.VideoDecodeError`. ``"skip"`` returns a
            sample with ``pixel_values=None`` and an ``error`` string for
            :func:`omnifall.collate_fn` to drop -- **this silently shrinks
            batches and changes epoch statistics, so only use it for
            exploratory work.** ``"retry"`` is the omnifall 0.1.0 behaviour of
            substituting a random other index; every substitution is logged at
            WARNING level naming both indices.
        max_retries: Number of substitutions ``on_error="retry"`` may make
            before giving up and raising.
        output_format: Layout of ``pixel_values``. ``"TCHW"`` (default) gives
            ``(T, C, H, W)`` per sample and ``(B, T, C, H, W)`` per batch, which
            is what both HuggingFace video models and ``fall-da`` expect.
            ``"CTHW"`` gives ``(C, T, H, W)`` / ``(B, C, T, H, W)``, the
            pytorchvideo convention. ``"THWC"`` returns the raw uint8 clip and
            requires ``transform=None``.
        return_meta: Whether to include the passthrough metadata columns in the
            returned dict.
        dataset_name: Name for this dataset. Defaults to the values of the
            ``dataset`` column, joined with ``+``.
        fast: Deprecated and ignored; decoding always seeks. Accepted only so
            that ``omnifall.load_video_dataset(..., fast=...)`` keeps working.

    Raises:
        ValueError: For missing columns or invalid arguments.

    Example::

        train = OmniFallVideoDataset(
            hf_train, sampling="random", seed=0,
            transform=VideoTransform("train"),
        )
        val = OmniFallVideoDataset(hf_val, sampling="uniform",
                                   transform=VideoTransform("val"))
    """

    def __init__(
        self,
        hf_dataset: Any,
        *,
        num_frames: int = 16,
        target_fps: float = 15.0,
        sampling: str = "uniform",
        transform: Callable[..., Any] | None = None,
        seed: int | None = None,
        on_error: ErrorPolicy = "raise",
        max_retries: int = 10,
        output_format: OutputFormat = "TCHW",
        return_meta: bool = True,
        dataset_name: str | None = None,
        fast: bool | None = None,
    ) -> None:
        required = {"video", "label", "start", "end"}
        missing = required - set(hf_dataset.column_names)
        if missing:
            raise ValueError(
                f"HF dataset is missing required columns: {sorted(missing)}. "
                "Did you pass video=True to omnifall.load()?"
            )
        if sampling not in SAMPLING_STRATEGIES:
            raise ValueError(
                f"sampling must be one of {SAMPLING_STRATEGIES}, got {sampling!r}."
            )
        if on_error not in ERROR_POLICIES:
            raise ValueError(
                f"on_error must be one of {ERROR_POLICIES}, got {on_error!r}."
            )
        if output_format not in OUTPUT_FORMATS:
            raise ValueError(
                f"output_format must be one of {OUTPUT_FORMATS}, got {output_format!r}."
            )
        if output_format == "THWC" and transform is not None:
            raise ValueError(
                "output_format='THWC' returns the raw decoded clip and cannot be "
                "combined with a transform. Use 'TCHW' or 'CTHW', or drop the "
                "transform."
            )
        if num_frames < 1:
            raise ValueError(f"num_frames must be >= 1, got {num_frames}.")
        if max_retries < 1:
            raise ValueError(f"max_retries must be >= 1, got {max_retries}.")
        if fast is not None:
            warnings.warn(
                "OmniFallVideoDataset(fast=...) is deprecated and ignored: "
                "decoding always seeks to the segment and falls back to a "
                "sequential scan of the same file only when seeking yields "
                "nothing.",
                DeprecationWarning,
                stacklevel=2,
            )

        self.dataset = hf_dataset
        self.num_frames = num_frames
        self.target_fps = target_fps
        self.sampling = sampling
        self.seed = seed
        self.on_error = on_error
        self.max_retries = max_retries
        self._output_format = output_format
        self.return_meta = return_meta

        if dataset_name is not None:
            self.dataset_name = dataset_name
        elif "dataset" in hf_dataset.column_names:
            self.dataset_name = "+".join(sorted(set(hf_dataset.unique("dataset"))))
        else:
            self.dataset_name = "unknown"

        self._meta_columns = [c for c in hf_dataset.column_names if c != "video"]
        self._warned_ambiguous_layout = False
        self.transform = transform

    @property
    def transform(self) -> Callable | None:
        """The per-clip transform, or ``None`` for raw frames.

        Assigning to this recomputes how the transform is called. Two facts are
        derived from the object and would otherwise go stale: whether it accepts
        an ``rng`` keyword, and which layout it declares via ``output_format``.
        A stale ``rng`` flag is the dangerous one --- the transform would silently
        fall back to global randomness and quietly stop being reproducible, with
        nothing in the output to show for it.
        """
        return self._transform

    @transform.setter
    def transform(self, value: Callable | None) -> None:
        if value is not None and self._output_format == "THWC":
            raise ValueError(
                "output_format='THWC' returns the raw decoded clip and cannot be "
                "combined with a transform. Set output_format='TCHW' or 'CTHW' "
                "first, or leave the transform as None."
            )
        self._transform = value
        self._transform_takes_rng = _accepts_rng(value)
        self._transform_layout: str | None = getattr(value, "output_format", None)

    @property
    def output_format(self) -> str:
        """Layout of ``pixel_values``: ``"TCHW"``, ``"CTHW"`` or ``"THWC"``.

        Validated on assignment. An unchecked attribute here was silently
        accepted and then produced normalized floats where the documented raw
        uint8 clip was expected.
        """
        return self._output_format

    @output_format.setter
    def output_format(self, value: str) -> None:
        if value not in OUTPUT_FORMATS:
            raise ValueError(
                f"output_format must be one of {OUTPUT_FORMATS}, got {value!r}."
            )
        if value == "THWC" and self._transform is not None:
            raise ValueError(
                "output_format='THWC' returns the raw decoded clip and cannot be "
                "combined with a transform. Clear the transform first."
            )
        self._output_format = value

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Decode segment *idx*.

        Args:
            idx: Row index into the underlying HuggingFace dataset.

        Returns:
            A dict with ``pixel_values`` (a tensor, or ``None`` under
            ``on_error="skip"``), ``label``, ``label_str``, and -- when
            ``return_meta`` is set -- the passthrough columns ``path``,
            ``dataset``, ``subject``, ``cam``, ``start`` and ``end``, plus the
            ``fall-da`` compatibility keys ``start_time``, ``end_time``,
            ``segment_duration``, ``video_path`` (the *relative* path, same as
            ``path``) and ``video_file`` (the absolute file on disk).

        Raises:
            VideoUnavailableError: If the row has no video file, regardless of
                ``on_error``.
            VideoDecodeError: Under ``on_error="raise"``, or under
                ``on_error="retry"`` once the retry budget is exhausted.
        """
        requested = idx
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(
                f"Index {requested} is out of range for a dataset with "
                f"{len(self)} segments."
            )

        try:
            return self._load_item(idx)
        except VideoDecodeError as exc:
            if self.on_error == "raise":
                raise
            if self.on_error == "skip":
                logger.warning("Skipping segment %d: %s", idx, exc)
                sample = self._metadata(idx)
                sample["pixel_values"] = None
                sample["error"] = str(exc)
                return sample
            return self._retry(idx, exc)

    def _retry(self, idx: int, first_error: VideoDecodeError) -> dict[str, Any]:
        """Substitute random other indices for a failing one (legacy policy)."""
        rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item()))
        error: Exception = first_error
        for attempt in range(1, self.max_retries + 1):
            substitute = int(rng.integers(0, len(self)))
            if substitute == idx and len(self) > 1:
                substitute = (substitute + 1) % len(self)
            logger.warning(
                "on_error='retry': segment %d failed (%s); substituting segment "
                "%d instead (attempt %d/%d). The batch will NOT contain the "
                "sample that was asked for.",
                idx,
                error,
                substitute,
                attempt,
                self.max_retries,
            )
            try:
                return self._load_item(substitute)
            except (VideoDecodeError, VideoUnavailableError) as exc:
                error = exc
        raise VideoDecodeError(
            str(self.dataset[idx]["video"]),
            float(self.dataset[idx]["start"]),
            float(self.dataset[idx]["end"]),
            f"Failed to load segment {idx} and {self.max_retries} random "
            f"substitutes; last error: {error}",
        )

    def _load_item(self, idx: int) -> dict[str, Any]:
        """Decode and transform one segment without any error handling."""
        row = self.dataset[idx]
        video_path = row["video"]

        if video_path is None:
            raise VideoUnavailableError(
                f"No video file for segment {idx} of dataset "
                f"{row.get('dataset', self.dataset_name)!r} (path={row.get('path')!r}). "
                "Run `omnifall prepare "
                f"{row.get('dataset', '<dataset>')}` to download it, or set "
                "OMNIFALL_ROOT to a directory holding the videos as "
                "{OMNIFALL_ROOT}/{dataset}/video/{path}.mp4"
            )

        rng = self._rng(idx)
        clip = decode_segment(
            video_path,
            start=float(row["start"]),
            end=float(row["end"]),
            num_frames=self.num_frames,
            target_fps=self.target_fps,
            sampling=self.sampling,
            rng=rng,
        )

        sample = self._metadata(idx, row=row)
        sample["pixel_values"] = self._to_output(clip, rng)
        return sample

    def _rng(self, idx: int) -> np.random.Generator:
        """Build the per-item random generator.

        With ``seed`` set the stream depends only on ``(seed, idx)``, so it is
        identical across processes, epochs and worker counts. Without a seed it
        is drawn from torch's global RNG, which ``DataLoader`` seeds per worker
        -- numpy's global state is deliberately not used, because ``DataLoader``
        does *not* reseed it per worker.
        """
        if self.seed is None:
            return np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item()))
        return np.random.default_rng([int(self.seed), int(idx)])

    def _to_output(self, clip: np.ndarray, rng: np.random.Generator) -> torch.Tensor:
        """Apply the transform and put the result into :attr:`output_format`."""
        if self.transform is None:
            tensor = torch.from_numpy(np.ascontiguousarray(clip))
            return _convert_layout(tensor, "THWC", self.output_format)

        if self._transform_takes_rng:
            result = self.transform(clip, rng=rng)
        else:
            result = self.transform(clip)

        if isinstance(result, dict):
            if "pixel_values" not in result:
                raise ValueError(
                    "A transform returning a dict must provide 'pixel_values'; "
                    f"got keys {sorted(result)}."
                )
            tensor = result["pixel_values"]
        else:
            tensor = result

        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                "The transform must return a torch.Tensor (optionally wrapped in "
                f"a dict under 'pixel_values'), got {type(tensor).__name__}."
            )
        if tensor.ndim != 4:
            raise ValueError(
                f"The transform must return a 4D clip tensor, got shape "
                f"{tuple(tensor.shape)}."
            )

        return _convert_layout(tensor, self._source_layout(tensor), self.output_format)

    def _source_layout(self, tensor: torch.Tensor) -> str:
        """Determine the layout the transform produced."""
        if self._transform_layout is not None:
            return str(self._transform_layout)

        inferred = _infer_layout(tuple(tensor.shape))
        if inferred is not None:
            return inferred

        if not self._warned_ambiguous_layout:
            self._warned_ambiguous_layout = True
            warnings.warn(
                f"Cannot tell which layout the transform produced from shape "
                f"{tuple(tensor.shape)}; assuming it is already "
                f"{self.output_format!r}. Give the transform an 'output_format' "
                "attribute to make this explicit.",
                RuntimeWarning,
                stacklevel=3,
            )
        return self.output_format

    def _metadata(self, idx: int, row: dict[str, Any] | None = None) -> dict[str, Any]:
        """Build the non-pixel part of a sample.

        The key set is a superset of what ``fall-da``'s ``OmnifallVideoDataset``
        emits, so that it can be swapped for this class without touching the
        training loop. Note that ``video_path`` is the **relative** dataset path
        (``fall-da`` semantics, identical to ``path``); the absolute file on
        disk is under the separate key ``video_file``.

        Args:
            idx: Row index.
            row: The already-fetched row, if the caller has one.

        Returns:
            The sample dict without ``pixel_values``.

        Raises:
            KeyError: If the row carries a label id outside ``range(16)``.
        """
        if row is None:
            row = self.dataset[idx]

        label = row["label"]
        try:
            label_str = IDX2LABEL[int(label)]
        except KeyError as exc:
            raise KeyError(
                f"Segment {idx} of {self.dataset_name!r} has label id {label}, "
                f"which is not one of the {len(IDX2LABEL)} OmniFall activity "
                "labels. The dataset is inconsistent with omnifall.ACTIVITY_LABELS."
            ) from exc

        sample: dict[str, Any] = {"label": label, "label_str": label_str}
        if not self.return_meta:
            return sample

        for column in self._meta_columns:
            sample.setdefault(column, row[column])
        # fall-da / omnifall 0.1.0 key names, kept so existing code keeps working.
        sample["start_time"] = row["start"]
        sample["end_time"] = row["end"]
        sample["segment_duration"] = float(row["end"]) - float(row["start"])
        sample["video_path"] = row["path"]
        sample["video_file"] = row["video"]
        return sample

    @property
    def targets(self) -> torch.Tensor:
        """All class labels as a tensor, for samplers and class statistics."""
        return torch.tensor(self.dataset["label"], dtype=torch.long)

    def meta(self, idx: int) -> Any:
        """Container metadata of the video file backing segment *idx*.

        Args:
            idx: Row index.

        Returns:
            The :class:`~omnifall._decode.VideoMeta` for the file. Cached per
            file, so calling this for every segment is cheap.

        Raises:
            VideoUnavailableError: If the row has no video file.
        """
        video_path = self.dataset[idx]["video"]
        if video_path is None:
            raise VideoUnavailableError(
                f"No video file for segment {idx} of {self.dataset_name!r}."
            )
        return probe(video_path)

    def __repr__(self) -> str:
        return (
            f"OmniFallVideoDataset(name={self.dataset_name!r}, "
            f"segments={len(self)}, num_frames={self.num_frames}, "
            f"target_fps={self.target_fps}, sampling={self.sampling!r}, "
            f"output_format={self.output_format!r})"
        )


def _accepts_rng(transform: Callable[..., Any] | None) -> bool:
    """Whether *transform* can be passed an ``rng`` keyword argument.

    Getting this wrong is expensive and silent: a ``False`` here means the
    seeded per-item generator never reaches the transform, which then falls back
    to global randomness and quietly stops being reproducible, with nothing in
    the output to show for it.

    The earlier version special-cased plain functions and otherwise inspected
    ``type(x).__call__``, which reported ``False`` for
    :func:`functools.partial`, for bound methods, and for any callable that
    forwards ``**kwargs``. ``inspect.signature`` already follows all three
    correctly, so it is asked directly.

    A ``**kwargs`` forwarder is treated as accepting ``rng``: it will not raise
    on the keyword, and passing a generator that is then ignored is a visible
    no-op, whereas withholding one silently breaks reproducibility. Between two
    imperfect guesses, prefer the one that fails loudly.
    """
    if transform is None:
        return False
    try:
        signature = inspect.signature(transform)
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return False

    for name, parameter in signature.parameters.items():
        if name == "rng" and parameter.kind is not inspect.Parameter.POSITIONAL_ONLY:
            return True
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
    return False


class MultiOmniFallDataset(Dataset):
    """Concatenate several :class:`OmniFallVideoDataset` instances.

    Items carry a ``domain_id`` (the index of the sub-dataset) and a
    ``domain_name``, which is what multi-source training in ``fall-da`` uses.

    Args:
        datasets: The datasets to concatenate, in order. Empty sub-datasets are
            allowed and are simply never indexed.

    Raises:
        ValueError: If *datasets* is empty.

    Example::

        multi = MultiOmniFallDataset([cmdfall_ds, le2i_ds])
        multi[0]["domain_name"]
    """

    def __init__(self, datasets: list[OmniFallVideoDataset]) -> None:
        if not datasets:
            raise ValueError("MultiOmniFallDataset needs at least one dataset.")

        self.datasets = list(datasets)
        self._sizes = [len(d) for d in self.datasets]
        self._cumulative = np.cumsum(self._sizes)

        logger.info(
            "MultiOmniFallDataset: %d datasets, %d total segments",
            len(self.datasets),
            len(self),
        )

    def __len__(self) -> int:
        return int(self._cumulative[-1])

    def _locate(self, idx: int) -> tuple[int, int]:
        """Map a global index to ``(dataset_index, local_index)``.

        ``searchsorted(..., side="right")`` returns the first position whose
        cumulative count strictly exceeds *idx*, which is the sub-dataset that
        owns it. Using ``side="right"`` also skips over empty sub-datasets,
        whose cumulative entries are duplicates of their predecessor's.

        Args:
            idx: Global index; negative indices count from the end.

        Returns:
            The sub-dataset index and the index within that sub-dataset.

        Raises:
            IndexError: If *idx* is out of range.
        """
        total = len(self)
        requested = idx
        if idx < 0:
            idx += total
        if not 0 <= idx < total:
            raise IndexError(
                f"Index {requested} is out of range for MultiOmniFallDataset "
                f"with {total} segments."
            )
        ds_idx = int(np.searchsorted(self._cumulative, idx, side="right"))
        offset = 0 if ds_idx == 0 else int(self._cumulative[ds_idx - 1])
        return ds_idx, idx - offset

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ds_idx, local_idx = self._locate(idx)
        sample = self.datasets[ds_idx][local_idx]
        sample["domain_id"] = ds_idx
        sample["domain_name"] = self.datasets[ds_idx].dataset_name
        return sample

    @property
    def targets(self) -> torch.Tensor:
        """All class labels across all sub-datasets, concatenated in order."""
        return torch.cat([d.targets for d in self.datasets])

    @property
    def domain_ids(self) -> torch.Tensor:
        """The sub-dataset index of every segment, in order."""
        ids: list[int] = []
        for i, dataset in enumerate(self.datasets):
            ids.extend([i] * len(dataset))
        return torch.tensor(ids, dtype=torch.long)

    @property
    def dataset_names(self) -> list[str]:
        """The name of each sub-dataset, in order."""
        return [d.dataset_name for d in self.datasets]

    def get_dataset_statistics(self) -> dict[str, dict[str, Any]]:
        """Per-sub-dataset segment counts and class distributions.

        Returns:
            Mapping from dataset name to ``{"total_segments", "class_distribution"}``.
        """
        stats: dict[str, dict[str, Any]] = {}
        for dataset in self.datasets:
            unique, counts = torch.unique(dataset.targets, return_counts=True)
            stats[dataset.dataset_name] = {
                "total_segments": len(dataset),
                "class_distribution": {int(c): int(n) for c, n in zip(unique, counts)},
            }
        return stats

    def __repr__(self) -> str:
        parts = ", ".join(f"{d.dataset_name}({len(d)})" for d in self.datasets)
        return (
            f"MultiOmniFallDataset({len(self.datasets)} datasets: {parts}, "
            f"total={len(self)})"
        )
