"""Video segment decoding for OmniFall.

This module is intentionally free of any dataset concerns: it knows how to turn
``(path, start, end)`` into a fixed number of RGB frames and nothing else.
Everything is expressed in **seconds** and presentation timestamps; a frame
count is never required, because several of the OmniFall component datasets ship
containers whose frame count is unreliable or absent.

Requires the optional ``av`` (PyAV) and ``numpy`` dependencies. This module is
imported lazily by :mod:`omnifall`, never at package import time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal, Protocol, Sequence

import av
import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "SAMPLING_STRATEGIES",
    "VideoDecodeError",
    "VideoMeta",
    "decode_segment",
    "probe",
    "segment_timestamps",
]

#: Supported temporal sampling strategies for :func:`decode_segment`.
SAMPLING_STRATEGIES = ("uniform", "center", "random")

Sampling = Literal["uniform", "center", "random"]


class _RNG(Protocol):
    """Minimal random-number interface required by :func:`decode_segment`."""

    def uniform(self, low: float, high: float) -> float:  # pragma: no cover
        ...


class VideoDecodeError(RuntimeError):
    """Raised when a video segment cannot be decoded.

    The message always names the file and the requested segment bounds so that
    a failure in a ``DataLoader`` worker can be traced back to a single row of
    the dataset without re-running anything.

    Attributes:
        path: Path of the video file that failed.
        start: Segment start in seconds.
        end: Segment end in seconds.
    """

    def __init__(
        self,
        path: str,
        start: float,
        end: float,
        message: str,
    ) -> None:
        self.path = path
        self.start = start
        self.end = end
        super().__init__(
            f"{message} (file={path!r}, segment=[{start:.3f}s, {end:.3f}s])"
        )


@dataclass(frozen=True)
class VideoMeta:
    """Container-level metadata for a video file.

    Attributes:
        path: Path the metadata was read from.
        fps: Average frame rate in frames per second.
        duration: Stream duration in seconds, or ``None`` if the container does
            not report one.
        width: Coded frame width in pixels.
        height: Coded frame height in pixels.
        n_frames: Number of frames reported by the container, or ``None`` when
            the container reports no (or a zero) frame count. Never rely on
            this being present.
        start_time: Stream start time in seconds (usually ``0.0``).
    """

    path: str
    fps: float
    duration: float | None
    width: int
    height: int
    n_frames: int | None
    start_time: float


@lru_cache(maxsize=4096)
def probe(path: str) -> VideoMeta:
    """Read container metadata for *path*.

    Results are cached (per process, so per ``DataLoader`` worker), which
    matters because OmniFall stores many temporal segments per video file --
    ``cmdfall`` alone has roughly 30 segments per file.

    Args:
        path: Path to a video file.

    Returns:
        The :class:`VideoMeta` for the file.

    Raises:
        VideoDecodeError: If the file cannot be opened, has no video stream, or
            reports no usable frame rate.
    """
    try:
        with av.open(path) as container:
            stream = _video_stream(container, path)
            fps = _stream_fps(stream, path)
            time_base = _stream_time_base(stream, path)

            if stream.duration is not None:
                duration: float | None = float(stream.duration) * time_base
            elif container.duration is not None:
                duration = float(container.duration) / 1_000_000.0
            else:
                duration = None

            n_frames = None if stream.frames in (0, None) else int(stream.frames)
            start_time = (
                0.0
                if stream.start_time is None
                else float(stream.start_time) * time_base
            )

            return VideoMeta(
                path=path,
                fps=fps,
                duration=duration,
                width=int(stream.codec_context.width),
                height=int(stream.codec_context.height),
                n_frames=n_frames,
                start_time=start_time,
            )
    except VideoDecodeError:
        raise
    except Exception as exc:  # noqa: BLE001 - re-raised with full context
        raise VideoDecodeError(
            path, float("nan"), float("nan"), f"Failed to probe video: {exc!r}"
        ) from exc


def _uniform(rng: _RNG | None, low: float, high: float) -> float:
    """Draw from ``rng``, or from numpy's global RNG when *rng* is ``None``.

    Args:
        rng: Random source, or ``None`` for the global RNG.
        low: Inclusive lower bound.
        high: Exclusive upper bound.

    Returns:
        A sample from ``[low, high)``.

    Raises:
        TypeError: If *rng* is not ``None`` and has no ``uniform`` method.
    """
    if rng is None:
        return float(np.random.uniform(low, high))
    if not hasattr(rng, "uniform"):
        raise TypeError(
            f"rng must have a uniform(low, high) method, got "
            f"{type(rng).__name__}. Pass None to use numpy's global RNG."
        )
    return float(rng.uniform(low, high))


def segment_timestamps(
    start: float,
    end: float,
    num_frames: int,
    target_fps: float,
    sampling: Sampling = "uniform",
    rng: _RNG | None = None,
) -> tuple[list[float], int]:
    """Compute the wanted presentation timestamps for one segment.

    All *requested* timestamps lie inside ``[start, end]`` by construction. The
    frames actually returned can fall marginally outside it: the decoder picks
    the frame nearest each requested timestamp, and for a timestamp close to a
    boundary the nearest frame may sit just beyond it. The excursion is bounded
    by half a frame interval — 20 ms at 25 fps — so it never reaches a
    neighbouring action, but the clip is not strictly confined to the segment.

    Args:
        start: Segment start in seconds.
        end: Segment end in seconds.
        num_frames: Number of frames to sample.
        target_fps: Frame rate of the sampled clip (used by ``"center"`` and
            ``"random"``; ignored by ``"uniform"``).
        sampling: One of :data:`SAMPLING_STRATEGIES`. See
            :func:`decode_segment` for the semantics.
        rng: Random source with a ``uniform(low, high)`` method. Used only by
            ``sampling="random"``. ``None`` means "draw from numpy's global
            RNG", which is well defined but not reproducible; pass a generator
            to get reproducible offsets.

    Returns:
        A tuple ``(timestamps, n_wanted)`` where ``timestamps`` is a sorted list
        of at most *num_frames* timestamps in seconds and ``n_wanted`` is
        *num_frames*. When ``len(timestamps) < n_wanted`` the caller must pad by
        repeating the last decoded frame.

    Raises:
        ValueError: If the arguments are inconsistent (``end < start``,
            ``num_frames < 1``, or an unknown *sampling*).
        TypeError: If *rng* is given but has no ``uniform(low, high)`` method.
    """
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}")
    if target_fps <= 0:
        raise ValueError(f"target_fps must be > 0, got {target_fps}")
    if end < start:
        raise ValueError(
            f"Segment end ({end}) is before segment start ({start}). "
            "Check the 'start'/'end' columns of the dataset row."
        )
    if sampling not in SAMPLING_STRATEGIES:
        raise ValueError(
            f"Unknown sampling={sampling!r}. Expected one of {SAMPLING_STRATEGIES}."
        )

    span = end - start

    if sampling == "uniform":
        if num_frames == 1:
            return [start + span / 2.0], num_frames
        step = span / (num_frames - 1)
        return [start + i * step for i in range(num_frames)], num_frames

    window = (num_frames - 1) / target_fps

    if window <= span:
        if sampling == "center":
            offset = start + (span - window) / 2.0
        else:
            offset = start + float(_uniform(rng, 0.0, span - window))
        return [offset + i / target_fps for i in range(num_frames)], num_frames

    # The segment is shorter than the requested window. Clamp to the segment and
    # let the caller pad by repeating the last frame. This is the only fallback
    # in this module: it is well defined, unavoidable, and never silently swaps
    # in data from outside the annotated segment.
    stamps = [start + i / target_fps for i in range(num_frames)]
    stamps = [t for t in stamps if t <= end] or [start]
    return stamps, num_frames


def decode_segment(
    path: str,
    *,
    start: float,
    end: float,
    num_frames: int = 16,
    target_fps: float = 15.0,
    sampling: Sampling = "uniform",
    rng: _RNG | None = None,
    backend: str = "pyav",
) -> np.ndarray:
    """Decode exactly *num_frames* RGB frames from ``[start, end]`` of *path*.

    Sampling strategies:
        ``"uniform"``
            *num_frames* timestamps spread evenly over the whole segment,
            endpoints included. Deterministic, ignores *target_fps*, and never
            needs padding. This is the right choice for validation and test.
        ``"center"``
            A window of ``(num_frames - 1) / target_fps`` seconds centred in the
            segment, sampled at *target_fps*. Deterministic.
        ``"random"``
            The same window length as ``"center"`` but at a random offset drawn
            from *rng*. This is the choice for training.

    The window length ``(num_frames - 1) / target_fps`` is deliberately the same
    formula ``fall-da``'s ``VideoDataset.get_random_offset`` uses, so numbers
    produced with ``sampling="random"`` stay comparable to previously published
    OmniFall results. What changed is only *how* the offset is applied: in
    seconds against the annotation, rather than through a container frame count.

    If the segment is shorter than the window required by ``"center"`` /
    ``"random"``, the window is clamped to the segment and the clip is padded by
    repeating the last decoded frame. The same padding covers segments that are
    annotated past the end of their file, which **is expected in the published
    OmniFall annotations** -- 209 of 52,618 segments (0.4%) overrun, up to 7.4 s
    in ``edf`` and 3.8 s in ``cmdfall``. Overrun must therefore never be turned
    into a hard error: that would reject real published data. This padding is
    the only fallback in this function.

    Args:
        path: Path to the video file.
        start: Segment start in seconds.
        end: Segment end in seconds.
        num_frames: Number of frames to return.
        target_fps: Sampling rate of the extracted clip in frames per second.
        sampling: One of :data:`SAMPLING_STRATEGIES`.
        rng: Random source with ``uniform(low, high)``, used only by
            ``sampling="random"``. ``None`` draws from numpy's global RNG:
            well defined, but not reproducible. Reproducibility requires
            passing a generator, which ``OmniFallVideoDataset`` does whenever
            its ``seed`` is set.
        backend: Decoding backend. Only ``"pyav"`` is implemented.

    Returns:
        A ``uint8`` array of shape ``(num_frames, H, W, 3)`` in RGB order.

    Raises:
        VideoDecodeError: If the file cannot be opened or no frame at all can be
            decoded from it.
        ValueError: For invalid arguments (see :func:`segment_timestamps`) or an
            unsupported *backend*.
    """
    if backend != "pyav":
        raise ValueError(
            f"backend={backend!r} is not supported. Only 'pyav' is implemented."
        )

    wanted, n_wanted = segment_timestamps(
        start, end, num_frames, target_fps, sampling, rng
    )

    try:
        with av.open(path) as container:
            stream = _video_stream(container, path)
            time_base = _stream_time_base(stream, path)
            start_pts = 0 if stream.start_time is None else int(stream.start_time)
            wanted_pts = [round(t / time_base) + start_pts for t in wanted]

            frames = _decode_at_pts(container, stream, wanted_pts, seek=True)

            if not frames:
                # A genuine, logged retry of the *same* sample: some containers
                # have broken seek indices, so scan from the head instead.
                logger.warning(
                    "Seeking yielded no frames for %s [%.3f, %.3f]; "
                    "retrying the same segment with a sequential scan.",
                    path,
                    start,
                    end,
                )
                container.seek(0, backward=True, any_frame=False, stream=stream)
                frames = _decode_at_pts(container, stream, wanted_pts, seek=False)

            if not frames:
                raise VideoDecodeError(
                    path, start, end, "No frames could be decoded for this segment"
                )
    except VideoDecodeError:
        raise
    except Exception as exc:  # noqa: BLE001 - re-raised with full context
        raise VideoDecodeError(
            path, start, end, f"Failed to decode video segment: {exc!r}"
        ) from exc

    if len(frames) < n_wanted:
        frames.extend([frames[-1]] * (n_wanted - len(frames)))

    clip = np.stack(frames[:n_wanted])
    if clip.dtype != np.uint8:  # pragma: no cover - PyAV always gives uint8
        raise VideoDecodeError(
            path, start, end, f"Expected uint8 frames, got dtype {clip.dtype}"
        )
    return clip


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _video_stream(container: Any, path: str) -> Any:
    """Return the first video stream of *container*."""
    for stream in container.streams:
        if stream.type == "video":
            return stream
    raise VideoDecodeError(
        path, float("nan"), float("nan"), "File contains no video stream"
    )


def _stream_fps(stream: Any, path: str) -> float:
    """Return the frame rate of *stream* in frames per second."""
    for rate in (stream.average_rate, stream.guessed_rate, stream.base_rate):
        if rate is not None and rate.denominator != 0 and float(rate) > 0:
            return float(rate)
    raise VideoDecodeError(
        path,
        float("nan"),
        float("nan"),
        "Cannot determine the frame rate of the video stream",
    )


def _stream_time_base(stream: Any, path: str) -> float:
    """Return the time base of *stream* in seconds per PTS unit."""
    time_base = stream.time_base
    if time_base is None or time_base.denominator == 0 or float(time_base) <= 0:
        raise VideoDecodeError(
            path,
            float("nan"),
            float("nan"),
            "Video stream has no usable time_base",
        )
    return float(time_base)


def _decode_at_pts(
    container: Any,
    stream: Any,
    wanted_pts: Sequence[int],
    *,
    seek: bool,
) -> list[np.ndarray]:
    """Decode the frame nearest to each timestamp in *wanted_pts*.

    Args:
        container: An open PyAV container.
        stream: The video stream to decode.
        wanted_pts: Sorted, non-decreasing presentation timestamps.
        seek: Whether to seek to the first wanted timestamp first. Pass ``False``
            to scan the stream from its current position.

    Returns:
        The decoded frames as ``(H, W, 3)`` uint8 arrays, at most
        ``len(wanted_pts)`` of them; shorter if the stream ended early.
    """
    if seek and wanted_pts:
        try:
            container.seek(
                int(wanted_pts[0]), any_frame=False, backward=True, stream=stream
            )
        except av.error.FFmpegError:
            container.seek(0, any_frame=False, backward=True, stream=stream)

    frames: list[np.ndarray] = []
    want_idx = 0
    prev = None

    for frame in container.decode(stream):
        if frame.pts is None:
            continue

        # Emit every wanted timestamp that this frame has caught up with,
        # choosing whichever of the previous / current frame is nearer.
        while want_idx < len(wanted_pts) and frame.pts >= wanted_pts[want_idx]:
            target = wanted_pts[want_idx]
            if prev is not None and abs(prev.pts - target) < abs(frame.pts - target):
                frames.append(prev.to_ndarray(format="rgb24"))
            else:
                frames.append(frame.to_ndarray(format="rgb24"))
            want_idx += 1

        if want_idx == len(wanted_pts):
            break
        prev = frame

    # The stream ended before all timestamps were reached: use the last frame we
    # actually saw for the remaining ones (the caller pads identically).
    if want_idx < len(wanted_pts) and prev is not None:
        tail = prev.to_ndarray(format="rgb24")
        frames.extend([tail] * (len(wanted_pts) - want_idx))

    return frames
