"""Turn ``(dataset, path)`` pairs into absolute video file paths.

Every config published by the OmniFall Hub repository carries a ``dataset``
column, and every ``path`` value is relative to that dataset's own video root.
Resolution is therefore a join of two columns against a per-dataset directory
and needs no knowledge of which config the table came from.

Two properties of the label tables shape the implementation:

* Segments vastly outnumber videos --- the ``cs`` config holds 54,150 segments
  over 2,979 distinct video files --- so existence checks are cached per
  distinct file path rather than performed per row.
* A single table may mix up to ten component datasets, each with its own root,
  so rows are grouped by ``dataset`` and each root is resolved once.
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from ._cache import dataset_video_dir_with_layer, dir_holds_videos, video_ext
from ._constants import DATASETS, ENV_ROOT

__all__ = [
    "Availability",
    "ResolutionReport",
    "canonical_dataset",
    "datasets_in",
    "resolve_paths",
    "build_report",
]

#: How many absent files are quoted per dataset in :meth:`ResolutionReport.summary`.
_N_MISSING_EXAMPLES = 3

#: Spellings that appear in config names but not in the ``dataset`` column.
#: Case is handled separately, so only genuinely different strings belong here.
_DATASET_ALIASES: dict[str, str] = {
    "of-itw": "OOPS",
}

#: Lower-cased spelling -> exact ``dataset`` column value.
_CANONICAL: dict[str, str] = {name.lower(): name for name in DATASETS}
_CANONICAL.update(_DATASET_ALIASES)


def canonical_dataset(name: str) -> str:
    """Map any spelling of a component dataset onto its ``dataset`` column value.

    Config names are lower-cased where the ``dataset`` column is not
    (``gmdcsa24`` vs ``GMDCSA24``, ``oops`` vs ``OOPS``), and the in-the-wild
    component is called ``of-itw`` as a config but ``OOPS`` as a dataset.

    Args:
        name: Any spelling, e.g. ``"GMDCSA24"``, ``"gmdcsa24"`` or ``"of-itw"``.

    Returns:
        The exact ``dataset`` column value.

    Raises:
        KeyError: If *name* is not a component dataset of OmniFall.
    """
    key = name.strip().lower()
    try:
        return _CANONICAL[key]
    except KeyError:
        raise KeyError(
            f"{name!r} is not a component dataset of OmniFall. "
            f"Known datasets (as spelled in the 'dataset' column): "
            f"{', '.join(sorted(DATASETS))}."
        ) from None


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Availability:
    """Video availability of a single component dataset.

    Attributes:
        dataset: Exact ``dataset`` column value.
        prepared: Whether the resolved directory holds any videos at all.
        video_dir: The directory ``path`` values were resolved against.
        source_layer: Which layer supplied *video_dir* --- ``"argument"``,
            ``"env:<VAR>"``, ``"OMNIFALL_ROOT"`` or ``"cache"``.
        n_referenced: Rows of the loaded table belonging to this dataset.
        n_missing: Referenced rows whose video file is absent. When existence
            was not checked this is either ``0`` (directory holds videos) or
            *n_referenced* (directory holds none).
        missing_examples: A few absolute paths that were expected but absent.
        suggestion: A nearby directory that *does* hold the referenced videos,
            found by probing an absent example. Set when *video_dir* is one
            level above the videos, which is the usual mistake when pointing
            an override at ``{root}/{dataset}`` instead of
            ``{root}/{dataset}/video``.
    """

    dataset: str
    prepared: bool
    video_dir: Path
    source_layer: str
    n_referenced: int
    n_missing: int
    missing_examples: list[str] = field(default_factory=list)
    suggestion: Path | None = None

    @property
    def complete(self) -> bool:
        """Whether every referenced row of this dataset resolved to a file."""
        return self.n_missing == 0


@dataclass(frozen=True)
class ResolutionReport:
    """Video availability of a whole loaded table.

    Attributes:
        per_dataset: One :class:`Availability` per component dataset present in
            the table, keyed by ``dataset`` column value.
        n_rows: Total rows across all splits.
        n_resolved: Rows whose video file was found (or, when existence was not
            checked, whose dataset directory holds videos).
        checked: Whether individual files were stat-ed.
    """

    per_dataset: dict[str, Availability]
    n_rows: int
    n_resolved: int
    checked: bool = True

    @property
    def complete(self) -> bool:
        """Whether every row of the table resolved to an available video."""
        return self.n_resolved == self.n_rows

    @property
    def missing_datasets(self) -> list[str]:
        """Component datasets with at least one unavailable video, sorted."""
        return sorted(d for d, a in self.per_dataset.items() if not a.complete)

    def summary(self) -> str:
        """Render a human-readable, multi-line availability report.

        Returns:
            A report suitable for a warning message or for printing by the CLI.
        """
        verb = "found" if self.checked else "expected"
        head = (
            f"OmniFall videos: {self.n_resolved}/{self.n_rows} rows {verb} "
            f"({len(self.per_dataset) - len(self.missing_datasets)}/"
            f"{len(self.per_dataset)} component datasets complete)"
        )
        lines = [head]

        width = max((len(d) for d in self.per_dataset), default=0)
        for name in sorted(self.per_dataset):
            av = self.per_dataset[name]
            state = "ok     " if av.complete else "MISSING"
            lines.append(
                f"  {name:<{width}}  {state}  "
                f"{av.n_referenced - av.n_missing}/{av.n_referenced} rows  "
                f"[{av.source_layer}] {av.video_dir}"
            )
            for example in av.missing_examples:
                lines.append(f"  {'':<{width}}    absent: {example}")
            if av.suggestion is not None:
                lines.append(
                    f"  {'':<{width}}    but found under {av.suggestion} --- "
                    f"point the [{av.source_layer}] layer there instead"
                )

        if not self.complete:
            lines.append("")
            lines.append(self._hint())
        return "\n".join(lines)

    def _hint(self) -> str:
        """Return an actionable next step for the datasets that are missing."""
        missing = self.missing_datasets
        obtainable = [d for d in missing if d in DATASETS and DATASETS[d].obtainable]
        manual = [d for d in missing if d not in obtainable]
        parts = [
            f"Point {ENV_ROOT} at a tree laid out as "
            f"{{{ENV_ROOT}}}/{{dataset}}/video/{{path}}.mp4, or set "
            f"OMNIFALL_VIDEO_ROOT__<dataset> per dataset."
        ]
        if obtainable:
            parts.append(
                f"Obtainable without a request to the original authors: "
                f"{', '.join(sorted(obtainable))} --- run "
                f"`omnifall prepare {' '.join(sorted(obtainable))}`."
            )
        if manual:
            homes = ", ".join(
                f"{d} ({DATASETS[d].homepage})" if d in DATASETS else d
                for d in sorted(manual)
            )
            parts.append(f"Must be obtained from the original authors: {homes}.")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Column inspection
# ---------------------------------------------------------------------------


def _splits(ds) -> list[tuple[str, object]]:
    """Return ``(split_name, Dataset)`` pairs for a ``Dataset``/``DatasetDict``.

    Args:
        ds: A :class:`datasets.Dataset` or :class:`datasets.DatasetDict`.

    Returns:
        One pair per split; a bare ``Dataset`` yields a single pair whose name
        is the empty string.

    Raises:
        TypeError: If *ds* is neither of the two supported types.
    """
    from datasets import Dataset, DatasetDict

    if isinstance(ds, DatasetDict):
        return list(ds.items())
    if isinstance(ds, Dataset):
        return [("", ds)]
    raise TypeError(
        f"Expected a datasets.Dataset or datasets.DatasetDict, got "
        f"{type(ds).__name__}."
    )


def _require_columns(split_name: str, table) -> None:
    """Raise unless *table* carries the columns needed for video resolution."""
    missing = [c for c in ("dataset", "path") if c not in table.column_names]
    if missing:
        where = f" of split {split_name!r}" if split_name else ""
        raise ValueError(
            f"Column(s) {', '.join(missing)}{where} are required to resolve "
            f"video files, but the table has {table.column_names}. "
            f"Annotation-only configs (metadata-syn, framewise-syn) are not "
            f"per-segment tables and have no video counterpart."
        )


def _dataset_column(table) -> list[str]:
    """Read the ``dataset`` column once, without materialising Python rows."""
    return table.data.column("dataset").to_pylist()


def datasets_in(ds) -> dict[str, int]:
    """Count rows per component dataset in a loaded table.

    The ``dataset`` column is read once per split straight from the Arrow
    table, which avoids the per-row Python objects that ``ds["dataset"]``
    would build.

    Args:
        ds: A :class:`datasets.Dataset` or :class:`datasets.DatasetDict`.

    Returns:
        Mapping from ``dataset`` column value to row count, aggregated over all
        splits.

    Raises:
        ValueError: If a split has no ``dataset`` column.
        TypeError: If *ds* is not a supported type.
    """
    counts: Counter[str] = Counter()
    for split_name, table in _splits(ds):
        _require_columns(split_name, table)
        counts.update(_dataset_column(table))
    return dict(counts)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def resolve_paths(
    datasets_col: Sequence[str],
    paths_col: Sequence[str],
    *,
    roots: Mapping[str, Path],
    check: bool,
) -> list[str | None]:
    """Join a ``dataset`` and a ``path`` column into absolute video paths.

    Args:
        datasets_col: ``dataset`` column values.
        paths_col: ``path`` column values, without extension and without a
            dataset prefix, e.g. ``"Coffee_room_01/video_1"``.
        roots: Directory each dataset's ``path`` values are relative to. Every
            value occurring in *datasets_col* must be present.
        check: Stat each distinct file and yield ``None`` for absent ones.
            Distinct paths are stat-ed once; segment tables reference the same
            video file many times over.

    Returns:
        One absolute path per row, or ``None`` where *check* is enabled and the
        file is absent.

    Raises:
        ValueError: If the two columns differ in length.
        KeyError: If *roots* lacks an entry for a dataset that occurs in
            *datasets_col*.
    """
    if len(datasets_col) != len(paths_col):
        raise ValueError(
            f"dataset column has {len(datasets_col)} rows but path column has "
            f"{len(paths_col)}."
        )

    prefixes: dict[str, str] = {}
    extensions: dict[str, str] = {}
    for name in set(datasets_col):
        try:
            root = roots[name]
        except KeyError:
            raise KeyError(
                f"No video directory was resolved for dataset {name!r}, which "
                f"occurs in the table. Known datasets: "
                f"{', '.join(sorted(DATASETS))}."
            ) from None
        prefixes[name] = os.path.join(str(root), "")
        extensions[name] = video_ext(name)

    exists: dict[str, bool] = {}
    out: list[str | None] = []
    for name, rel in zip(datasets_col, paths_col):
        full = f"{prefixes[name]}{rel}{extensions[name]}"
        if not check:
            out.append(full)
            continue
        hit = exists.get(full)
        if hit is None:
            hit = os.path.exists(full)
            exists[full] = hit
        out.append(full if hit else None)
    return out


def resolve_roots(
    names: Iterable[str],
    *,
    overrides: Mapping[str, Path] | None = None,
) -> dict[str, tuple[str, Path]]:
    """Resolve the video directory of each named component dataset.

    Args:
        names: ``dataset`` column values.
        overrides: Directories supplied by the caller, keyed by any spelling
            accepted by :func:`canonical_dataset`.

    Returns:
        Mapping from ``dataset`` column value to a ``(layer, directory)`` pair.

    Raises:
        KeyError: If a key of *overrides* is not a component dataset.
        FileNotFoundError: If an override or environment variable names a
            location that does not exist.
    """
    by_dataset = {canonical_dataset(k): Path(v) for k, v in (overrides or {}).items()}
    resolved: dict[str, tuple[str, Path]] = {}
    for name in dict.fromkeys(names):
        # Rejects a 'dataset' value this build does not know about rather than
        # guessing a layout and an extension for it.
        canonical_dataset(name)
        resolved[name] = dataset_video_dir_with_layer(
            name, override=by_dataset.get(name)
        )
    return resolved


#: Directories tried, relative to a failing video directory, when diagnosing a
#: dataset that resolved nothing. Kept tiny and used only for reporting.
_NEARBY_PROBES: tuple[str, ...] = ("video",)


def _probe_nearby(directory: Path, dataset: str, rel: str | None) -> Path | None:
    """Look for a nearby directory that does hold *rel*.

    Pointing an override at ``{root}/{dataset}`` rather than
    ``{root}/{dataset}/video`` is the mistake worth catching, and one ``stat``
    against a known-absent path settles it. This never changes resolution ---
    it only lets the report name the directory that would have worked.

    Args:
        directory: The video directory that produced no hits.
        dataset: Exact ``dataset`` column value.
        rel: A ``path`` value that was not found under *directory*, or ``None``
            when nothing was missing.

    Returns:
        The directory that holds *rel*, or ``None``.
    """
    if rel is None:
        return None
    for probe in _NEARBY_PROBES:
        candidate = directory / probe
        if os.path.exists(candidate / f"{rel}{video_ext(dataset)}"):
            return candidate
    return None


def resolve_table(
    ds,
    *,
    overrides: Mapping[str, Path] | None = None,
    check: bool = True,
) -> tuple[list[tuple[str, list[str | None]]], ResolutionReport]:
    """Resolve every split of a table and report on what was found.

    This is the single place where resolution actually happens;
    :func:`build_report` and :func:`omnifall._video.add_video` are thin
    wrappers that keep or discard the resolved columns.

    Args:
        ds: A :class:`datasets.Dataset` or :class:`datasets.DatasetDict`.
        overrides: Per-dataset video directories supplied by the caller.
        check: Stat each distinct file.

    Returns:
        A pair of ``(split_name, video_column)`` entries in split order, and the
        matching :class:`ResolutionReport`.
    """
    splits = _splits(ds)
    for split_name, table in splits:
        _require_columns(split_name, table)

    referenced: Counter[str] = Counter()
    for _, table in splits:
        referenced.update(_dataset_column(table))
    roots = resolve_roots(referenced, overrides=overrides)

    columns: list[tuple[str, list[str | None]]] = []
    missing: Counter[str] = Counter()
    examples: dict[str, list[str]] = {name: [] for name in referenced}
    first_absent_rel: dict[str, str] = {}

    for split_name, table in splits:
        names = _dataset_column(table)
        paths = table.data.column("path").to_pylist()
        resolved = resolve_paths(
            names,
            paths,
            roots={n: d for n, (_, d) in roots.items()},
            check=check,
        )
        columns.append((split_name, resolved))
        if not check:
            continue
        for name, value, rel in zip(names, resolved, paths):
            if value is not None:
                continue
            missing[name] += 1
            first_absent_rel.setdefault(name, rel)
            bucket = examples[name]
            if len(bucket) < _N_MISSING_EXAMPLES:
                # Segments share video files, so quote distinct files only.
                _, root = roots[name]
                example = str(root / f"{rel}{video_ext(name)}")
                if example not in bucket:
                    bucket.append(example)

    per_dataset: dict[str, Availability] = {}
    for name, n_referenced in referenced.items():
        layer, directory = roots[name]
        if check and missing[name] < n_referenced:
            # At least one file was found, so the directory is populated ---
            # no need to walk it a second time.
            prepared = True
        else:
            prepared = dir_holds_videos(directory, name)
        if check:
            n_missing = missing[name]
        else:
            # Without stat-ing, an empty directory is the only thing that can
            # be stated with certainty --- and it means nothing will load.
            n_missing = 0 if prepared else n_referenced
        per_dataset[name] = Availability(
            dataset=name,
            prepared=prepared,
            video_dir=directory,
            source_layer=layer,
            n_referenced=n_referenced,
            n_missing=n_missing,
            missing_examples=examples[name],
            suggestion=_probe_nearby(directory, name, first_absent_rel.get(name)),
        )

    n_rows = sum(referenced.values())
    n_resolved = n_rows - sum(a.n_missing for a in per_dataset.values())
    report = ResolutionReport(
        per_dataset=per_dataset,
        n_rows=n_rows,
        n_resolved=n_resolved,
        checked=check,
    )
    return columns, report


def build_report(
    ds,
    *,
    overrides: Mapping[str, Path] | None = None,
    check: bool = True,
) -> ResolutionReport:
    """Report which of a table's videos are available, without modifying it.

    Args:
        ds: A :class:`datasets.Dataset` or :class:`datasets.DatasetDict`.
        overrides: Per-dataset video directories supplied by the caller.
        check: Stat each distinct file. With ``False`` only the presence of the
            per-dataset directories is inspected, which is fast but coarse.

    Returns:
        The availability report.
    """
    return resolve_table(ds, overrides=overrides, check=check)[1]
