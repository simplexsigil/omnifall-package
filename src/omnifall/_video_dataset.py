"""PyTorch video dataset wrapping HuggingFace Dataset from ``omnifall.load()``.

Provides ``OmniFallVideoDataset`` for single-dataset loading and
``MultiOmniFallDataset`` for combining multiple datasets with proper indexing.

Video decoding uses PyAV (``av`` package).
"""

from __future__ import annotations

import bisect
import logging
import math
import random
from collections.abc import Callable
from typing import Any

import av
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class OmniFallVideoDataset(Dataset):
    """PyTorch dataset that decodes video segments from an OmniFall HF Dataset.

    Each row in the underlying HF Dataset represents one temporal segment
    (with ``path``, ``label``, ``start``, ``end``, and a ``video`` column
    containing the absolute file path). This class handles video decoding
    via PyAV and temporal segment sampling.

    Args:
        hf_dataset: A ``datasets.Dataset`` from ``omnifall.load(config, video=True)``.
            Must contain columns: ``video``, ``label``, ``start``, ``end``.
        target_fps: Target FPS for frame sampling.
        num_frames: Number of frames to extract per segment.
        transform: Optional callable ``(frames: list[np.ndarray]) -> dict``.
            If provided, must return a dict with at least ``"pixel_values"``.
            If None, returns raw frames as ``np.ndarray[T, H, W, C]``.
        max_retries: Max retries on video decode errors before raising.
        fast: Use PTS-based seeking (fast) or sequential decode (slow).
        dataset_name: Name for this dataset (used by ``MultiOmniFallDataset``).
    """

    def __init__(
        self,
        hf_dataset: Any,
        target_fps: float = 15.0,
        num_frames: int = 16,
        transform: Callable | None = None,
        max_retries: int = 10,
        fast: bool = True,
        dataset_name: str | None = None,
    ) -> None:
        required = {"video", "label", "start", "end"}
        missing = required - set(hf_dataset.column_names)
        if missing:
            raise ValueError(
                f"HF dataset is missing required columns: {missing}. "
                "Did you pass video=True to omnifall.load()?"
            )

        self.dataset = hf_dataset
        self.target_fps = target_fps
        self.num_frames = num_frames
        self.transform = transform
        self.max_retries = max_retries
        self._use_fast = fast

        # Derive dataset_name from the 'dataset' column if available
        if dataset_name is not None:
            self.dataset_name = dataset_name
        elif "dataset" in hf_dataset.column_names:
            names = set(hf_dataset.unique("dataset"))
            self.dataset_name = "+".join(sorted(names))
        else:
            self.dataset_name = "unknown"

        # Cache column names for metadata passthrough
        self._meta_columns = [
            c
            for c in hf_dataset.column_names
            if c not in ("video", "label", "start", "end")
        ]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        original_idx = idx
        retries = 0
        while retries < self.max_retries:
            try:
                return self._load_item(idx)
            except Exception as e:
                retries += 1
                if retries >= self.max_retries:
                    logger.error(
                        "Failed to load video after %d retries. "
                        "Original index: %d, last tried index: %d. "
                        "Last error: %s",
                        self.max_retries,
                        original_idx,
                        idx,
                        e,
                    )
                    raise
                idx = random.randint(0, len(self.dataset) - 1)

        raise RuntimeError(
            f"Failed to load a valid video after {self.max_retries} attempts"
        )

    def _load_item(self, idx: int) -> dict[str, Any]:
        row = self.dataset[idx]
        video_path = row["video"]

        if video_path is None:
            raise FileNotFoundError(
                f"Video path is None for index {idx} (path={row.get('path')}). "
                "This segment has no downloadable video."
            )

        frames = self._load_video(video_path, idx)

        if self.transform is not None:
            result = self.transform(frames)
        else:
            result = {"frames": np.stack(frames)}

        result["label"] = row["label"]
        result["start_time"] = row["start"]
        result["end_time"] = row["end"]

        for col in self._meta_columns:
            if col not in result:
                result[col] = row[col]

        return result

    # ------------------------------------------------------------------
    # Temporal sampling
    # ------------------------------------------------------------------

    def _get_segment_offset(self, idx: int, fps: float) -> int:
        """Compute a random start frame within the annotated segment.

        Uses the corrected clip-duration calculation:
        ``clip_duration_sec = (num_frames - 1) / target_fps``.
        """
        row = self.dataset[idx]
        start_frame = int(row["start"] * fps)
        end_frame = int(row["end"] * fps)
        segment_frames = end_frame - start_frame

        clip_duration_sec = (self.num_frames - 1) / self.target_fps
        required_frames = int(clip_duration_sec * fps) + 1

        if segment_frames <= required_frames:
            return start_frame
        else:
            max_offset = segment_frames - required_frames
            return start_frame + random.randint(0, int(max_offset))

    # ------------------------------------------------------------------
    # Video decoding (PyAV)
    # ------------------------------------------------------------------

    def _load_video(
        self, path: str, idx: int
    ) -> list[np.ndarray]:
        if self._use_fast:
            return self._load_video_fast(path, idx)
        return self._load_video_slow(path, idx)

    def _load_video_fast(
        self, path: str, idx: int
    ) -> list[np.ndarray]:
        """PTS-based seeking for efficient frame extraction."""
        try:
            with av.open(path) as container:
                vs = next(s for s in container.streams if s.type == "video")

                rate = vs.average_rate or vs.base_rate
                if not rate or rate.denominator == 0:
                    raise ValueError("Cannot determine FPS")
                fps = float(rate)

                tb = vs.time_base
                if not tb:
                    raise ValueError("Missing time_base")
                tb = float(tb)

                frame_cnt = (
                    None if vs.frames in (0, None) else int(vs.frames)
                )

                begin_frame = (
                    self._get_segment_offset(idx, fps) if frame_cnt else 0
                )

                desired_timestamps = [
                    (begin_frame / fps) + n / self.target_fps
                    for n in range(self.num_frames)
                ]
                desired_pts = [int(ts / tb) for ts in desired_timestamps]

                if desired_pts:
                    try:
                        container.seek(
                            desired_pts[0],
                            any_frame=False,
                            backward=True,
                            stream=vs,
                        )
                    except av.error.FFmpegError:
                        container.seek(0, stream=vs)

                frames: list[np.ndarray] = []
                want_idx = 0
                prev = None
                for f in container.decode(vs):
                    if f.pts is None:
                        continue

                    while (
                        want_idx < len(desired_pts)
                        and f.pts >= desired_pts[want_idx]
                    ):
                        if prev and abs(
                            prev.pts - desired_pts[want_idx]
                        ) < abs(f.pts - desired_pts[want_idx]):
                            frames.append(prev.to_ndarray(format="rgb24"))
                        else:
                            frames.append(f.to_ndarray(format="rgb24"))
                        want_idx += 1

                    if want_idx == len(desired_pts):
                        break
                    prev = f

                if not frames:
                    logger.warning("%s: fallback to slow loader", path)
                    return self._load_video_slow(path, idx)

                # Pad by repeating last frame
                if len(frames) < self.num_frames:
                    last = frames[-1]
                    while len(frames) < self.num_frames:
                        frames.append(last)

                return frames

        except Exception as e:
            logger.error("Error reading video %s: %s", path, e, exc_info=True)
            raise RuntimeError(f"Failed to process video {path}") from e

    def _load_video_slow(
        self, path: str, idx: int
    ) -> list[np.ndarray]:
        """Sequential decode fallback."""
        try:
            with av.open(path) as container:
                vs = next(s for s in container.streams if s.type == "video")

                rate = vs.average_rate
                if rate and rate.denominator != 0:
                    fps = float(rate.numerator / rate.denominator)
                else:
                    raise ValueError(f"Cannot determine FPS for {path}")

                target_interval = max(1, round(fps / self.target_fps))

                frames: list[np.ndarray] = []
                for i, frame in enumerate(container.decode(vs)):
                    if i % target_interval == 0:
                        frames.append(frame.to_ndarray(format="rgb24"))

            if not frames:
                raise ValueError(f"No frames decoded from {path}")

            if len(frames) < self.num_frames:
                last = frames[-1]
                while len(frames) < self.num_frames:
                    frames.append(last)
            else:
                start_index = self._get_segment_offset(idx, fps)
                # Convert absolute frame offset to index in the subsampled list
                start_index_sub = start_index // target_interval
                start_index_sub = min(
                    start_index_sub, max(0, len(frames) - self.num_frames)
                )
                frames = frames[start_index_sub : start_index_sub + self.num_frames]

            return frames

        except Exception as e:
            logger.error("Error reading video %s: %s", path, e)
            raise RuntimeError(f"Failed to process video {path}") from e

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def targets(self) -> torch.Tensor:
        """All class labels as a tensor (for samplers / statistics)."""
        return torch.tensor(self.dataset["label"])

    def __repr__(self) -> str:
        return (
            f"OmniFallVideoDataset(name={self.dataset_name!r}, "
            f"segments={len(self)}, fps={self.target_fps}, "
            f"num_frames={self.num_frames})"
        )


class MultiOmniFallDataset(Dataset):
    """Wrapper for multiple ``OmniFallVideoDataset`` instances.

    Enables training on multiple datasets simultaneously with proper
    cumulative indexing.

    Args:
        datasets: List of ``OmniFallVideoDataset`` instances.
    """

    def __init__(self, datasets: list[OmniFallVideoDataset]) -> None:
        self.datasets = datasets
        self._sizes = [len(d) for d in datasets]
        self._cumulative = np.cumsum(self._sizes)

        total = int(self._cumulative[-1]) if len(self._cumulative) else 0
        logger.info(
            "MultiOmniFallDataset: %d datasets, %d total segments",
            len(datasets),
            total,
        )

    def __len__(self) -> int:
        return int(self._cumulative[-1]) if len(self._cumulative) else 0

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ds_idx = bisect.bisect_right(self._cumulative, idx)
        local_idx = idx if ds_idx == 0 else idx - int(self._cumulative[ds_idx - 1])

        sample = self.datasets[ds_idx][local_idx]
        sample["domain_id"] = ds_idx
        sample["domain_name"] = self.datasets[ds_idx].dataset_name
        return sample

    @property
    def targets(self) -> torch.Tensor:
        """All class labels across all datasets."""
        return torch.cat([d.targets for d in self.datasets])

    @property
    def domain_ids(self) -> torch.Tensor:
        """Domain ID for each segment (dataset index)."""
        ids: list[int] = []
        for i, d in enumerate(self.datasets):
            ids.extend([i] * len(d))
        return torch.tensor(ids)

    @property
    def dataset_names(self) -> list[str]:
        return [d.dataset_name for d in self.datasets]

    def get_dataset_statistics(self) -> dict[str, dict[str, Any]]:
        """Per-dataset segment counts and class distributions."""
        stats: dict[str, dict[str, Any]] = {}
        for d in self.datasets:
            targets = d.targets
            unique, counts = torch.unique(targets, return_counts=True)
            stats[d.dataset_name] = {
                "total_segments": len(d),
                "class_distribution": {
                    int(c): int(n) for c, n in zip(unique, counts)
                },
            }
        return stats

    def __repr__(self) -> str:
        parts = ", ".join(
            f"{d.dataset_name}({len(d)})" for d in self.datasets
        )
        return f"MultiOmniFallDataset({len(self.datasets)} datasets: {parts}, total={len(self)})"
