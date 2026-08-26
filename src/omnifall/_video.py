"""Attach absolute video file paths to a loaded OmniFall table.

The public entry point is :func:`add_video`. It works for every per-segment
config the Hub serves --- present and future --- because it joins the
``dataset`` and ``path`` columns rather than consulting a list of config names.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Mapping

from ._configs import NON_SEGMENT_CONFIGS
from ._resolve import (
    ResolutionReport,
    build_report,
    canonical_dataset,
    resolve_table,
)

__all__ = ["MissingVideosError", "add_video", "resolution_report"]


class MissingVideosError(FileNotFoundError):
    """Raised by :func:`add_video` with ``strict=True`` when videos are absent.

    Subclasses :class:`FileNotFoundError` so that existing ``except
    FileNotFoundError`` handlers keep working. The message is the full
    :meth:`omnifall._resolve.ResolutionReport.summary`.

    Attributes:
        report: The availability report that triggered the error.
    """

    def __init__(self, report: ResolutionReport) -> None:
        super().__init__(report.summary())
        self.report = report


def _reject_non_segment_config(config: str | None) -> None:
    """Raise for configs that are not per-segment tables.

    Args:
        config: Config name the table was loaded with, or ``None``.

    Raises:
        ValueError: If *config* is an annotation-only config.
    """
    if config is not None and config in NON_SEGMENT_CONFIGS:
        raise ValueError(
            f"Config {config!r} holds one row per video rather than per "
            f"annotated segment, so its rows carry no 'start'/'end' and cannot "
            f"drive segment decoding. Load a segment config such as "
            f"'of-syn' to attach videos. "
            f"({', '.join(sorted(NON_SEGMENT_CONFIGS))} do carry resolvable "
            f"'dataset' and 'path' columns, so omitting the config argument "
            f"resolves them anyway --- use omnifall.resolution_report to "
            f"inspect their availability.)"
        )


def _normalised_overrides(
    video_dirs: Mapping[str, str | Path] | None,
) -> dict[str, Path]:
    """Map user-supplied directory overrides onto ``dataset`` column values.

    Args:
        video_dirs: Directories keyed by any spelling accepted by
            :func:`omnifall._resolve.canonical_dataset`.

    Returns:
        Directories keyed by the exact ``dataset`` column value.

    Raises:
        KeyError: If a key is not a component dataset of OmniFall.
        ValueError: If two keys normalise to the same dataset.
    """
    out: dict[str, Path] = {}
    for key, value in (video_dirs or {}).items():
        name = canonical_dataset(key)
        if name in out and Path(value) != out[name]:
            raise ValueError(
                f"video_dirs names dataset {name!r} twice with different "
                f"directories ({out[name]} and {value})."
            )
        out[name] = Path(value)
    return out


def _prepare_missing(
    ds,
    overrides: dict[str, Path],
    *,
    consent: bool,
) -> dict[str, Path]:
    """Obtain the videos of every referenced dataset that has none yet.

    Args:
        ds: The loaded table.
        overrides: Directories supplied by the caller.
        consent: Passed through to the preparation code, where it stands in for
            the interactive license acknowledgement.

    Returns:
        *overrides* extended with the directory each newly prepared dataset was
        written to.

    Raises:
        RuntimeError: If the preparation module is unavailable.
    """
    try:
        from ._prepare import ensure_dataset
    except ImportError as exc:  # pragma: no cover - packaging accident
        raise RuntimeError(
            "download=True needs omnifall._prepare, which could not be "
            "imported. Obtain the videos manually and point OMNIFALL_ROOT at "
            "them instead."
        ) from exc

    report = build_report(ds, overrides=overrides, check=False)
    resolved = dict(overrides)
    for name, availability in report.per_dataset.items():
        if availability.prepared:
            continue
        resolved[name] = Path(
            ensure_dataset(name, consent=consent, video_dir=overrides.get(name))
        )
    return resolved


def resolution_report(
    dataset,
    config: str | None = None,
    *,
    video_dirs: Mapping[str, str | Path] | None = None,
    check: bool = True,
) -> ResolutionReport:
    """Report which of a table's videos are available, without modifying it.

    Args:
        dataset: A :class:`datasets.Dataset` or :class:`datasets.DatasetDict`
            loaded from the OmniFall Hub repository.
        config: Config the table was loaded with. Optional, and used only to
            produce a better error for annotation-only configs.
        video_dirs: Per-dataset video directory overrides, e.g.
            ``{"cmdfall": "/data/cmdfall/video"}``.
        check: Stat each distinct file. With ``False`` only the presence of the
            per-dataset directories is inspected.

    Returns:
        The availability report; :meth:`ResolutionReport.summary` renders it.
    """
    _reject_non_segment_config(config)
    return resolve_table(
        dataset,
        overrides=_normalised_overrides(video_dirs),
        check=check,
    )[1]


def add_video(
    dataset,
    config: str | None = None,
    *,
    video_dirs: Mapping[str, str | Path] | None = None,
    check: bool = True,
    strict: bool = False,
    download: bool = False,
    consent: bool = False,
):
    """Add a ``video`` column of absolute file paths to a loaded table.

    Each row's file is ``{video_dir(dataset)}/{path}{ext}``, where the video
    directory comes from the layers described in :mod:`omnifall._cache`. The
    config name plays no part in resolution, so every per-segment config the
    Hub serves is supported, including ones added after this release.

    Args:
        dataset: A :class:`datasets.Dataset` or :class:`datasets.DatasetDict`
            loaded from the OmniFall Hub repository. An existing ``video``
            column is replaced.
        config: Config the table was loaded with. Optional, and used only to
            produce a better error for annotation-only configs.
        video_dirs: Per-dataset video directory overrides, keyed by any
            spelling accepted by
            :func:`omnifall._resolve.canonical_dataset`.
        check: Stat each distinct file and write ``None`` for rows whose video
            is absent. With ``False`` the column is filled unconditionally,
            which is faster but may point at files that do not exist.
        strict: Raise :class:`MissingVideosError` when anything is missing,
            instead of warning once.
        download: Obtain the videos of unprepared datasets first. Only the
            components OmniFall may redistribute can be obtained this way; the
            rest still have to be requested from their original authors.
        consent: Acknowledge the OOPS license non-interactively. Only
            meaningful together with ``download``.

    Returns:
        The table with an additional ``video`` column of type ``str`` (or
        ``None`` where ``check`` found no file). A
        :class:`datasets.DatasetDict` in, a :class:`datasets.DatasetDict` out.

    Raises:
        ValueError: If *config* is an annotation-only config, or if the table
            lacks the ``dataset``/``path`` columns.
        MissingVideosError: If ``strict`` and any referenced video is absent.
        FileNotFoundError: If an explicitly named video directory or
            ``OMNIFALL_ROOT`` does not exist.

    Warns:
        UserWarning: Once, with the full availability report, when videos are
            missing and ``strict`` is false.
    """
    from datasets import DatasetDict

    _reject_non_segment_config(config)
    overrides = _normalised_overrides(video_dirs)

    if download:
        overrides = _prepare_missing(dataset, overrides, consent=consent)

    columns, report = resolve_table(dataset, overrides=overrides, check=check)

    if not report.complete:
        if strict:
            raise MissingVideosError(report)
        warnings.warn(report.summary(), UserWarning, stacklevel=2)

    if isinstance(dataset, DatasetDict):
        by_split = dict(columns)
        return DatasetDict(
            {
                name: _with_video(split, by_split[name])
                for name, split in dataset.items()
            }
        )
    return _with_video(dataset, columns[0][1])


def _with_video(split, video_column: list[str | None]):
    """Return *split* with *video_column* attached, replacing any old one."""
    if "video" in split.column_names:
        split = split.remove_columns("video")
    return split.add_column("video", video_column)
