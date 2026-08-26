"""Obtaining the videos of each OmniFall component.

OmniFall's annotations come from the Hub; its videos mostly do not. This module
is what acts on :data:`omnifall._sources.SOURCES`: it downloads what can be
downloaded, extracts it safely, converts the original directory trees into the
one OmniFall addresses, and --- crucially --- tells the user honestly when it
cannot do any of that.

The layout every component must end up in is::

    {video_dir}/{path}{ext}

where ``video_dir`` comes from :func:`omnifall._cache.dataset_video_dir` and the
``path`` values are exactly the ones in the Hub file ``labels/{dataset}.csv``.
That file is the authority, and :func:`verify` checks a prepared tree against
it. Nothing here guesses.

Design rules
------------
* Fail fast. A missing converter raises; it never silently prepares a partial
  tree and reports success.
* Never buffer a large download in memory, and never leave a half-written file
  where a complete one is expected.
* A source that needs a browser, a form or an e-mail is not "broken" --- it is
  :class:`DatasetNotAvailableError` with instructions.
"""

from __future__ import annotations

import bisect
import csv
import hashlib
import html
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import IO, Callable, Iterable, Iterator, Mapping, Sequence
from urllib.parse import urlencode

from ._cache import (
    dataset_download_dir,
    dataset_video_dir,
    get_cache_dir,
    get_download_dir,
    is_dataset_prepared,
    video_ext,
)
from ._constants import HF_REPO_ID
from ._sources import SOURCES, Source, get_source

__all__ = [
    "DatasetNotAvailableError",
    "ConversionNotImplementedError",
    "VerifyReport",
    "ConvertReport",
    "ensure_dataset",
    "prepare",
    "prepare_all",
    "prepare_oops",
    "status",
    "verify",
    "convert",
    "required_paths",
    "datasets_in_config",
    "download",
    "extract_archive",
    "convert_tree",
    "require_ffmpeg",
    "require_ffprobe",
    "download_locations",
    "up_fall_links",
    "FFMPEG_ENV_VAR",
]

#: Chunk size for streaming downloads and copies.
_CHUNK = 1 << 20

#: How often the progress line is refreshed, in seconds.
_PROGRESS_INTERVAL = 0.5


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DatasetNotAvailableError(RuntimeError):
    """Raised when a component's videos cannot be obtained automatically.

    The message always names both *why* (the source needs a browser, a form or
    an e-mail request) and *where* the files have to end up, so that a user who
    reads only the exception still knows what to do next.

    Attributes:
        dataset: The component that could not be prepared.
        video_dir: The directory the videos must eventually occupy.
        source: The registry entry that explains the manual route.
        download_dir: The download root in force, so that the directory named
            in the message is the one the next command will actually read.
    """

    def __init__(
        self,
        dataset: str,
        video_dir: Path,
        source: Source,
        download_dir: str | Path | None = None,
    ) -> None:
        self.dataset = dataset
        self.video_dir = video_dir
        self.source = source
        instructions = source.instructions.strip() or (
            f"No automated source is recorded for {dataset}. See "
            f"{source.homepage or 'the original authors'}."
        )
        reason = (
            "require an access request to the original authors"
            if source.gated
            else "cannot be downloaded without a browser session"
        )
        # The instructions say to put the files "in the directory named
        # above"; this is what names it. An error that tells the user to do
        # something without saying where is not an instruction.
        where = download_locations(dataset, download_dir).describe()
        super().__init__(
            f"{dataset}: the videos {reason}, so omnifall cannot fetch them "
            f"for you.\n\n{instructions}\n\n"
            f"Put what you download here, then run "
            f"'omnifall prepare {dataset}' again:\n{where}\n\n"
            f"Either way the videos must end up as {video_dir}/{{path}}"
            f"{_ext(dataset)}, for example "
            f"{video_dir}/{_example_path(dataset)}{_ext(dataset)}.\n"
            f"Run 'omnifall verify {dataset}' afterwards to check the result."
        )


class ConversionNotImplementedError(NotImplementedError):
    """Raised when a source can be fetched but not yet reshaped.

    Several components ship their videos in a layout that differs from
    OmniFall's, and the code that bridges the two has not been written for all
    of them yet. This is deliberately a loud, specific failure rather than a
    partial preparation.

    Attributes:
        dataset: The component whose converter is missing.
        detail: What precisely remains to be implemented.
    """

    def __init__(self, dataset: str, detail: str) -> None:
        self.dataset = dataset
        self.detail = detail
        source = SOURCES.get(dataset)
        route = ""
        if source is not None and source.automatable:
            route = (
                f"\nThe download itself IS automated: run "
                f"'omnifall prepare {dataset} --download-only' to fetch and "
                f"unpack the original release, then convert it yourself."
            )
        super().__init__(
            f"{dataset}: converting the original tree into OmniFall's layout "
            f"is not implemented yet.\n\n{detail.strip()}{route}\n"
            f"Once the files are in place, 'omnifall verify {dataset}' will "
            f"confirm the result."
        )


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------


def _ext(dataset: str) -> str:
    """Return *dataset*'s video extension, tolerating unknown names."""
    try:
        return video_ext(dataset)
    except KeyError:
        return ".mp4"


#: One representative ``path`` per component, used only in error messages.
_EXAMPLE_PATHS: dict[str, str] = {
    "caucafall": "adl/HopS1",
    "cmdfall": "colors/S10P12K1",
    "edf": "jianjun/view1/rgb/jianjun_view1_rgb",
    "GMDCSA24": "Subject_1/ADL/01",
    "le2i": "Coffee_room_01/video_1",
    "mcfd": "chute01/cam1",
    "occu": "jiayan/view1/rgb/jiayan_view1_rgb",
    "up_fall": "Subject1/Activity1/Trial1/Subject1Activity1Trial1Camera1",
    "OOPS": "falls/BestFailsofWeek2July2016_FailArmy9",
    "of-syn": "fall/fall_ch_001",
}


def _example_path(dataset: str) -> str:
    """Return a representative ``path`` value for *dataset*."""
    return _EXAMPLE_PATHS.get(dataset, "<path>")


def _human(n: int | None) -> str:
    """Format a byte count for humans, or ``"unknown size"`` for ``None``."""
    if n is None:
        return "-"
    step = 1024.0
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < step or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= step
    return f"{value:.1f} TiB"


class _Progress:
    """A one-line progress indicator on stderr.

    Writes nothing at all when disabled or when stderr is not a terminal, so
    that piped output and log files stay clean.

    Args:
        label: Text shown before the counter.
        total: Expected final count, or ``None`` when it is not known.
        unit: ``"B"`` to format counts as byte sizes, anything else to print
            them as plain integers.
        enabled: Set to ``False`` to silence the indicator entirely.
    """

    def __init__(
        self,
        label: str,
        total: int | None = None,
        *,
        unit: str = "B",
        enabled: bool = True,
    ) -> None:
        self.label = label
        self.total = total
        self.unit = unit
        self.enabled = enabled and sys.stderr.isatty()
        self.count = 0
        self._start = time.monotonic()
        self._last = 0.0

    def advance(self, n: int) -> None:
        """Add *n* to the running count and refresh the line if it is due."""
        self.count += n
        now = time.monotonic()
        if now - self._last >= _PROGRESS_INTERVAL:
            self._last = now
            self._render(now)

    def _render(self, now: float) -> None:
        if not self.enabled:
            return
        done = _human(self.count) if self.unit == "B" else f"{self.count}"
        if self.total:
            of = _human(self.total) if self.unit == "B" else f"{self.total}"
            pct = 100.0 * self.count / self.total
            body = f"{done} / {of} ({pct:5.1f}%)"
        else:
            body = done
        elapsed = max(now - self._start, 1e-6)
        rate = self.count / elapsed
        speed = f"{_human(int(rate))}/s" if self.unit == "B" else f"{rate:.0f}/s"
        sys.stderr.write(f"\r  {self.label}: {body}  {speed}   ")
        sys.stderr.flush()

    def close(self) -> None:
        """Render a final line and move the cursor off it."""
        if not self.enabled:
            return
        self._render(time.monotonic())
        sys.stderr.write("\n")
        sys.stderr.flush()

    def __enter__(self) -> "_Progress":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


#: Name of the directory, inside a component's download directory, that holds
#: the unpacked original release. Reserved: omnifall extracts into it, and a
#: user who has the release unpacked already can put it there instead, which is
#: the whole manual route for cmdfall.
_UNPACKED = "unpacked"


def _work_dir(dataset: str, download_dir: str | Path | None = None) -> Path:
    """Return the directory holding *dataset*'s original release.

    Args:
        dataset: Exact ``dataset`` column value.
        download_dir: An explicit download root, replacing the resolved one.

    Returns:
        ``{download_dir}/{dataset}``. It is not created here.
    """
    return dataset_download_dir(dataset, override=download_dir)


@dataclass(frozen=True)
class DownloadLocations:
    """Where one component's original release is fetched to and looked for.

    This is the answer ``omnifall sources`` prints, and it is the same answer
    the downloader acts on --- which is the point. An instruction that names a
    different place from the one the code reads is an instruction that stops
    being true, so there is exactly one function computing both.

    Attributes:
        dataset: The component.
        directory: ``{download_dir}/{dataset}``, holding the archives.
        unpacked: ``{directory}/unpacked``, holding the unpacked release.
        names: Archive file names expected in :attr:`directory`, where they are
            recorded. Empty where :attr:`pattern` describes them instead.
        pattern: Shape of the archive names, for a source with too many to
            list. ``None`` where :attr:`names` is complete.
        count: How many archives the release is served as.
        accepts_unpacked: Whether :attr:`unpacked` is a route into this
            component. It is for most of them; of-syn's archive already holds
            OmniFall's own layout and unpacks into the video directory, and
            OOPS is streamed member by member out of a 45 GB tarball rather
            than unpacked at all, so for those two only the archive counts.
    """

    dataset: str
    directory: Path
    unpacked: Path
    names: tuple[str, ...]
    pattern: str | None
    count: int
    accepts_unpacked: bool = True

    def paths(self) -> tuple[Path, ...]:
        """Return the full path of every recorded archive name."""
        return tuple(self.directory / name for name in self.names)

    def describe(self) -> str:
        """Return a few lines naming the files and the directory to put them in.

        Returns:
            Text ending without a newline, suitable for both ``omnifall
            sources`` and an error message.
        """
        lines = [f"  download to: {self.directory}"]
        if self.names and len(self.names) <= 3:
            lines.append(f"  file name:   {', '.join(self.names)}")
        elif self.names:
            lines.append(
                f"  file names:  {self.names[0]} ... {self.names[-1]} "
                f"({len(self.names)} files)"
            )
        elif self.pattern:
            lines.append(
                f"  file names:  {self.pattern} ({self.count} files)"
            )
        if self.accepts_unpacked:
            lines.append(f"  or unpack the release into: {self.unpacked}")
        return "\n".join(lines)


def download_locations(
    dataset: str, download_dir: str | Path | None = None
) -> DownloadLocations:
    """Return where *dataset*'s release is downloaded to and looked for.

    Args:
        dataset: Exact ``dataset`` column value.
        download_dir: An explicit download root, replacing the resolved one.

    Returns:
        The locations, whether or not anything is there yet.

    Raises:
        KeyError: If *dataset* is not an OmniFall component.
    """
    source = get_source(dataset)
    directory = _work_dir(dataset, download_dir)
    return DownloadLocations(
        dataset=dataset,
        directory=directory,
        unpacked=directory / _UNPACKED,
        names=source.archive_names(),
        pattern=source.file_pattern,
        count=source.n_archives,
        accepts_unpacked=dataset not in _ARCHIVE_ONLY,
    )


#: Components for which an unpacked directory is not a route in. See
#: :attr:`DownloadLocations.accepts_unpacked`.
_ARCHIVE_ONLY: frozenset[str] = frozenset({"of-syn", "OOPS"})


def local_archive(
    dataset: str, download_dir: str | Path | None = None
) -> Path | None:
    """Return a single-archive component's archive, if it is already on disk.

    Args:
        dataset: Exact ``dataset`` column value.
        download_dir: An explicit download root.

    Returns:
        The archive's path, or ``None`` when the component is served as several
        archives or none is there.
    """
    locations = download_locations(dataset, download_dir)
    if len(locations.names) != 1:
        return None
    candidate = locations.directory / locations.names[0]
    return candidate if candidate.is_file() else None


# ---------------------------------------------------------------------------
# The label files are the authority on what a prepared tree must contain
# ---------------------------------------------------------------------------


def required_paths(dataset: str, *, refresh: bool = False) -> tuple[str, ...]:
    """Return every ``path`` value *dataset* must provide a video for.

    Read from the Hub file ``labels/{dataset}.csv``, which is the authoritative
    list --- not from any count hard-coded in this package.

    Args:
        dataset: Exact ``dataset`` column value, e.g. ``"le2i"``.
        refresh: Bypass the local Hub cache and re-download the label file.

    Returns:
        The unique ``path`` values, sorted, without file extensions.

    Raises:
        KeyError: If *dataset* is not an OmniFall component.
        RuntimeError: If the label file has no ``path`` column.
    """
    get_source(dataset)  # validates the name, with a helpful message
    from huggingface_hub import hf_hub_download

    csv_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=f"labels/{dataset}.csv",
        repo_type="dataset",
        force_download=refresh,
    )
    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "path" not in reader.fieldnames:
            raise RuntimeError(
                f"labels/{dataset}.csv has no 'path' column "
                f"(columns: {reader.fieldnames}). The Hub layout changed; "
                f"upgrade the omnifall package."
            )
        return tuple(sorted({row["path"] for row in reader}))


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VerifyReport:
    """The result of checking a prepared tree against the Hub label file.

    Attributes:
        dataset: The component that was checked.
        video_dir: The directory the check was performed in.
        required: How many distinct ``path`` values the labels reference.
        present: How many of those exist as non-empty files.
        missing: The ``path`` values with no file at all.
        empty: The ``path`` values whose file exists but has zero bytes.
        extra: How many video files in the tree the labels never reference.
            Extras are harmless --- several releases carry more views than
            OmniFall uses --- so they never make a report incomplete.
    """

    dataset: str
    video_dir: Path
    required: int
    present: int
    missing: tuple[str, ...]
    empty: tuple[str, ...]
    extra: int

    @property
    def complete(self) -> bool:
        """Whether every required video is present and non-empty."""
        return not self.missing and not self.empty

    @property
    def n_expected(self) -> int:
        """How many videos the Hub label file requires.

        Read from ``labels/{dataset}.csv``, never from
        ``omnifall._constants.DATASETS[...].n_videos`` --- the label file is the
        authority and the two must not be allowed to drift.
        """
        return self.required

    @property
    def n_present(self) -> int:
        """How many required videos exist as non-empty files."""
        return self.present

    @property
    def n_missing(self) -> int:
        """How many required videos are absent or zero-length.

        Counts the empty files too: a zero-byte video is missing in every sense
        that matters to a decoder.
        """
        return len(self.missing) + len(self.empty)

    def missing_examples(self, limit: int = 10) -> tuple[str, ...]:
        """Return up to *limit* of the ``path`` values that are not usable.

        Args:
            limit: How many to return. Missing files come first, then
                zero-length ones.

        Returns:
            ``path`` values, without file extension.
        """
        return (self.missing + self.empty)[:limit]

    @property
    def fraction(self) -> float:
        """Share of required videos that are present, in ``[0, 1]``."""
        if self.required == 0:
            return 1.0
        return self.present / self.required

    def summary(self) -> str:
        """Return a single-line verdict."""
        verdict = "OK" if self.complete else "INCOMPLETE"
        return (
            f"{self.dataset:<10} {verdict:<11} "
            f"{self.present}/{self.required} present "
            f"({100 * self.fraction:.1f}%)"
        )

    def render(self, *, max_listed: int = 10) -> str:
        """Return a multi-line report naming the first missing paths.

        Args:
            max_listed: How many missing (and empty) paths to name before
                summarising the rest as a count.

        Returns:
            A report suitable for printing to a terminal.
        """
        lines = [self.summary(), f"  directory: {self.video_dir}"]
        ext = _ext(self.dataset)
        if self.extra:
            lines.append(
                f"  {self.extra} extra {ext} file(s) not referenced by the "
                f"labels (harmless)"
            )
        for label, items in (("missing", self.missing), ("empty", self.empty)):
            if not items:
                continue
            lines.append(f"  {len(items)} {label}:")
            for item in items[:max_listed]:
                lines.append(f"    {item}{ext}")
            if len(items) > max_listed:
                lines.append(f"    ... and {len(items) - max_listed} more")
        if not self.complete:
            source = SOURCES.get(self.dataset)
            if source is not None and not source.automatable:
                lines.append(
                    f"  obtain the rest manually: omnifall sources "
                    f"{self.dataset}"
                )
            else:
                lines.append(f"  retry with: omnifall prepare {self.dataset}")
        return "\n".join(lines)


def verify(
    dataset: str,
    *,
    video_dir: str | Path | None = None,
    refresh: bool = False,
) -> VerifyReport:
    """Check a prepared tree against the authoritative Hub label file.

    This is the function that answers "did my manual download actually work?".
    It resolves the dataset's video directory exactly the way video loading
    does, so a report that says ``OK`` means the loader will find every file.

    Args:
        dataset: Exact ``dataset`` column value.
        video_dir: Directory to check, overriding the usual resolution order.
        refresh: Re-download the label file instead of using the Hub cache.

    Returns:
        A :class:`VerifyReport`.

    Raises:
        KeyError: If *dataset* is not an OmniFall component.
    """
    wanted = required_paths(dataset, refresh=refresh)
    directory = dataset_video_dir(dataset, override=video_dir)
    ext = _ext(dataset)

    missing: list[str] = []
    empty: list[str] = []
    present = 0
    for path in wanted:
        candidate = directory / f"{path}{ext}"
        try:
            size = candidate.stat().st_size
        except (OSError, ValueError):
            missing.append(path)
            continue
        if size == 0:
            empty.append(path)
        else:
            present += 1

    extra = 0
    if directory.is_dir():
        wanted_set = set(wanted)
        for found in directory.rglob(f"*{ext}"):
            if not found.is_file():
                continue
            rel = found.relative_to(directory).with_suffix("").as_posix()
            if rel not in wanted_set:
                extra += 1

    return VerifyReport(
        dataset=dataset,
        video_dir=directory,
        required=len(wanted),
        present=present,
        missing=tuple(missing),
        empty=tuple(empty),
        extra=extra,
    )


def datasets_in_config(config: str, *, refresh: bool = False) -> tuple[str, ...]:
    """Return the component datasets a Hub config draws rows from.

    Determined by loading the config and reading its ``dataset`` column, which
    every config carries --- so this keeps working for configs added to the Hub
    after this release, and needs no hard-coded table.

    Args:
        config: A config name, e.g. ``"of-sta-cs"``. Deprecated spellings are
            accepted; the Hub still serves them.
        refresh: Bypass the ``datasets`` HTTP cache.

    Returns:
        The component names present in the config, in registry order.

    Raises:
        ValueError: If the config has no ``dataset`` column.
    """
    from datasets import load_dataset

    from ._resolve import datasets_in

    ds = load_dataset(
        HF_REPO_ID,
        config,
        download_mode="force_redownload" if refresh else None,
    )
    found = set(datasets_in(ds))
    # Return in registry order so output is stable between runs.
    ordered = tuple(name for name in SOURCES if name in found)
    unknown = sorted(found - set(SOURCES))
    if unknown:
        raise ValueError(
            f"config {config!r} references component dataset(s) this build of "
            f"omnifall does not know: {unknown}. Upgrade the omnifall package."
        )
    return ordered


def _spread(items: Sequence[str], count: int) -> tuple[str, ...]:
    """Return *count* of *items*, evenly spaced and including both ends.

    Args:
        items: The sequence to sample.
        count: How many to take. Fewer are returned when *items* is shorter.

    Returns:
        The sample, in the original order.
    """
    if len(items) <= count or count <= 1:
        return tuple(items[:count] if count <= 1 else items)
    step = (len(items) - 1) / (count - 1)
    return tuple(items[round(index * step)] for index in range(count))


def _holds_own_videos(
    dataset: str, video_dir: str | Path | None, *, sample: int = 5
) -> bool:
    """Report whether a directory holds videos *this* component actually needs.

    :func:`omnifall._cache.is_dataset_prepared` asks only whether a directory
    contains a file with the right extension. Every component uses ``.mp4``, so
    under a single ``video_dir`` override that question has the same answer ten
    times over: one prepared component makes all ten look present, gated
    cmdfall included. This asks the question that actually distinguishes them.

    A handful of the component's own ``path`` values is enough --- no other
    component's tree contains them --- and far cheaper than :func:`verify`,
    which is still the right tool for "is it complete".

    Args:
        dataset: Exact ``dataset`` column value.
        video_dir: Directory override, or ``None`` for the usual resolution.
        sample: How many of the required paths to look for.

    Returns:
        ``True`` if every sampled path exists as a non-empty file.

    Raises:
        Exception: Propagated from :func:`required_paths` if the Hub label file
            cannot be read. Only reached for a directory that does hold videos:
            the cheap negative below needs no network.
    """
    if not is_dataset_prepared(dataset, override=video_dir):
        return False
    directory = dataset_video_dir(dataset, override=video_dir)
    ext = _ext(dataset)
    for path in _spread(required_paths(dataset), sample):
        try:
            if (directory / f"{path}{ext}").stat().st_size == 0:
                return False
        except OSError:
            return False
    return True


def status(
    *,
    video_dir: str | Path | None = None,
    strict: bool | None = None,
) -> dict[str, bool]:
    """Report which components have videos on disk.

    This answers "is there something there", not "is it complete" --- use
    :func:`verify` for the latter.

    Args:
        video_dir: Directory override applied to every component.
        strict: Whether to look for videos each component actually needs rather
            than for any video at all. Defaults to ``True`` when *video_dir* is
            given and ``False`` otherwise, because a single override points all
            ten components at one directory and the cheap check cannot tell
            them apart there; without an override each component has its own
            directory and the cheap check is both correct and offline.

    Returns:
        A mapping from component name to whether its video directory holds
        that component's videos, in registry order.

    Raises:
        Exception: Under *strict*, propagated from :func:`required_paths` if
            the Hub label file cannot be read for a directory that does hold
            videos.
    """
    if strict is None:
        strict = video_dir is not None
    if not strict:
        return {
            name: is_dataset_prepared(name, override=video_dir)
            for name in SOURCES
        }
    return {name: _holds_own_videos(name, video_dir) for name in SOURCES}


# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------


def download(
    url: str,
    dest: str | Path,
    *,
    expected_bytes: int | None = None,
    sha256: str | None = None,
    resume: bool = True,
    progress: bool = True,
    label: str | None = None,
    verbose: bool = True,
    expect_magic: bytes | None = None,
    what: str | None = None,
) -> Path:
    """Stream *url* to *dest*, resuming a previous attempt where possible.

    The body is written to a ``.part`` sibling and moved into place only once
    the transfer completes, so an interrupted run never leaves a truncated file
    that later looks finished.

    A file that is already at *dest* is checked, not trusted: whatever
    guarantees a fresh download would have to satisfy, a reused one has to
    satisfy too. Otherwise the second run of a command is weaker than the
    first, which is exactly backwards --- the file has had more time to be
    corrupted, truncated or replaced.

    Args:
        url: The URL to fetch. Redirects are followed, which matters for
            sources that hand out short-lived presigned links.
        dest: Final path for the downloaded file.
        expected_bytes: Size the finished file must have. A mismatch is an
            error, not a warning.
        sha256: Lower-case hex digest the finished file must have, checked on a
            reused file as well as on a fresh one. Only pass a digest that was
            actually measured. When none is passed and *verbose* is set, that
            is announced rather than passed over: without a digest the only
            integrity evidence is the byte count.
        resume: Whether to ask the server to continue a partial ``.part`` file.
            Servers that ignore ``Range`` cause a clean restart from zero.
        progress: Whether to show a progress line on an interactive stderr.
        label: Text for the progress line; defaults to the file name.
        verbose: Whether to note on stderr what was and was not checked.
        expect_magic: Leading bytes the finished file must have, e.g. ``b"PK"``
            for a zip. Several of these sources answer a perfectly ordinary
            HTTP 200 with an HTML page instead of the data --- a bot check, a
            Drive quota notice, an expired link --- and without this the page
            is written out under the archive's name and only fails later, at
            extraction, having thrown the real error away. An HTML content type
            is rejected before the body is read at all.
        what: The thing being fetched, named in errors, e.g.
            ``"mcfd chute01.zip"``. Defaults to the URL.

    Returns:
        The path to the completed file.

    Raises:
        RuntimeError: On an HTTP error, a size mismatch, a digest mismatch or a
            body that is not what *expect_magic* says it must be --- whether
            the file was just downloaded or found already in place.
    """
    import requests

    dest = Path(dest)
    subject = what or url
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")

    if dest.exists():
        size = dest.stat().st_size
        if expected_bytes is not None and size != expected_bytes:
            raise RuntimeError(
                f"{dest} already exists but is {size} bytes, "
                f"expected {expected_bytes}. Delete it and retry."
            )
        if sha256 is not None:
            digest = _sha256(dest, progress=progress)
            if digest != sha256.lower():
                raise RuntimeError(
                    f"{dest} already exists but has sha256 {digest}, expected "
                    f"{sha256.lower()}. The file is corrupt, truncated or not "
                    f"what it claims to be; it was NOT used. Delete it and "
                    f"retry to download it again."
                )
        _check_magic(dest, expect_magic, subject, reused=True)
        _report_integrity(dest, expected_bytes, sha256, verbose, reused=True)
        return dest

    already = part.stat().st_size if (resume and part.exists()) else 0
    headers = {"Range": f"bytes={already}-"} if already else {}

    with requests.get(
        url, stream=True, headers=headers, timeout=(30, 300), allow_redirects=True
    ) as response:
        if response.status_code not in (200, 206):
            raise RuntimeError(
                f"downloading {subject} failed with HTTP "
                f"{response.status_code} ({url}). If the source requires a "
                f"browser session, obtain the file manually and point omnifall "
                f"at it with --archive."
            )
        # A page where an archive was asked for. Reject it now, before a few
        # kilobytes of HTML are written out under the archive's name.
        kind = (response.headers.get("Content-Type") or "").split(";")[0].strip()
        if expect_magic is not None and kind.lower() in _NOT_AN_ARCHIVE:
            raise RuntimeError(_challenge_message(subject, url, kind))
        # A server that ignores our Range header restarts the body at zero;
        # appending would corrupt the file, so start over instead.
        if already and response.status_code == 200:
            already = 0
        mode = "ab" if already else "wb"

        remaining = response.headers.get("Content-Length")
        total = expected_bytes
        if total is None and remaining is not None:
            total = int(remaining) + already

        response.raw.decode_content = True
        with _Progress(
            label or dest.name, total, enabled=progress
        ) as bar, open(part, mode) as handle:
            bar.advance(already)
            for chunk in response.iter_content(chunk_size=_CHUNK):
                if chunk:
                    handle.write(chunk)
                    bar.advance(len(chunk))

    # Before the size check: a challenge page is short, so a size mismatch is
    # the symptom the user would otherwise be shown, and it explains nothing.
    _check_magic(part, expect_magic, subject, reused=False, url=url)

    size = part.stat().st_size
    if expected_bytes is not None and size != expected_bytes:
        raise RuntimeError(
            f"{subject} produced {size} bytes but {expected_bytes} were "
            f"expected ({url}). The partial file is kept at {part}; rerun to "
            f"resume, or delete it to start over."
        )
    if sha256 is not None:
        digest = _sha256(part, progress=progress)
        if digest != sha256.lower():
            part.unlink()
            raise RuntimeError(
                f"{url} has sha256 {digest}, expected {sha256.lower()}. "
                f"The download was discarded."
            )
    part.replace(dest)
    _report_integrity(dest, expected_bytes, sha256, verbose, reused=False)
    return dest


#: Content types that are never an archive. A source answering with one of
#: these where data was expected is serving a bot check, a quota notice or an
#: error page, whatever its status code says.
_NOT_AN_ARCHIVE: frozenset[str] = frozenset(
    {"text/html", "application/xhtml+xml", "text/plain", "application/json"}
)


def _challenge_message(subject: str, url: str, detail: str) -> str:
    """Return the error text for a source that served a page, not the data.

    Args:
        subject: What was being fetched, e.g. ``"mcfd chute01.zip"``.
        url: The URL it was fetched from.
        detail: What came back instead, e.g. a content type.

    Returns:
        The message body.
    """
    dataset = subject.split()[0]
    return (
        f"{subject}: the server returned {detail} where an archive was "
        f"expected ({url}).\n"
        f"That is a web page, not data --- typically a bot check, a rate "
        f"limit, a Google Drive quota notice or an expired link. It has NOT "
        f"been saved as an archive.\n"
        f"Try again later, or download the file in a browser and pass it to "
        f"omnifall directly:\n"
        f"    omnifall prepare {dataset} --archive <file-or-directory>\n"
        f"Run 'omnifall sources {dataset}' for the exact file names and the "
        f"directory to put them in."
    )


def _check_magic(
    path: Path,
    expect_magic: bytes | None,
    subject: str,
    *,
    reused: bool,
    url: str = "",
) -> None:
    """Raise unless *path* begins with *expect_magic*.

    Args:
        path: The downloaded file.
        expect_magic: Required leading bytes, or ``None`` to skip the check.
        subject: What was being fetched, for the message.
        reused: Whether *path* was found in place rather than downloaded. A
            fresh download that failed this is deleted; a file the user put
            there is left alone, because deleting someone else's file to
            punish it for being the wrong file is not this function's call.
        url: Where it came from, for the message.

    Raises:
        RuntimeError: If the leading bytes do not match.
    """
    if expect_magic is None:
        return
    with open(path, "rb") as handle:
        head = handle.read(len(expect_magic))
    if head == expect_magic:
        return
    if not reused:
        path.unlink(missing_ok=True)
    where = f"the file already at {path}" if reused else "the download"
    raise RuntimeError(
        _challenge_message(
            subject, url or str(path), f"a body starting {head!r}"
        )
        + f"\n({where} did not begin with {expect_magic!r}.)"
    )


def _report_integrity(
    dest: Path,
    expected_bytes: int | None,
    sha256: str | None,
    verbose: bool,
    *,
    reused: bool,
) -> None:
    """Say on stderr what was checked about *dest*, and what was not.

    Silence about an unverified multi-gigabyte download reads as a clean bill
    of health, which it is not. None of the entries in
    :data:`omnifall._sources.SOURCES` currently declares a digest --- the
    original sites publish none that could be recorded honestly --- so for most
    components the only evidence is a byte count, and this says so.

    Args:
        dest: The file in question.
        expected_bytes: The declared size, if any.
        sha256: The declared digest, if any.
        verbose: Whether to print at all.
        reused: Whether *dest* was found in place rather than downloaded.
    """
    if not verbose:
        return
    where = "reusing" if reused else "downloaded"
    if sha256 is not None:
        note = "sha256 verified"
    elif expected_bytes is not None:
        note = (
            "no sha256 is published for this source, so only its byte count "
            "was checked"
        )
    else:
        note = (
            "no sha256 and no expected size are known for this source, so its "
            "contents were NOT verified"
        )
    print(f"  {where} {dest.name}: {note}", file=sys.stderr)


def _sha256(path: Path, *, progress: bool = True) -> str:
    """Return the lower-case hex sha256 of *path*, streaming it in chunks."""
    digest = hashlib.sha256()
    total = path.stat().st_size
    with _Progress(f"hashing {path.name}", total, enabled=progress) as bar:
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(_CHUNK)
                if not chunk:
                    break
                digest.update(chunk)
                bar.advance(len(chunk))
    return digest.hexdigest()


def download_hub_file(
    filename: str,
    *,
    repo_id: str = HF_REPO_ID,
    progress: bool = True,
) -> Path:
    """Download one file from the OmniFall Hub repository.

    Args:
        filename: Repository-relative file name.
        repo_id: Repository to read from.
        progress: Whether ``huggingface_hub`` may draw its own progress bar.

    Returns:
        Path to the file inside the Hub cache.
    """
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
        )
    )


# ---------------------------------------------------------------------------
# Archive extraction
# ---------------------------------------------------------------------------


def _reject(name: str, reason: str) -> None:
    """Raise a uniform error for a rejected archive member."""
    raise RuntimeError(
        f"refusing to extract archive member {name!r}: {reason}. "
        f"The archive is malformed or hostile; it was not extracted."
    )


def safe_member_target(dest: Path, name: str) -> Path:
    """Resolve archive member *name* under *dest*, refusing to escape it.

    Rejects absolute paths, Windows drive letters, and any name that walks out
    of *dest* via ``..`` --- the classic tar/zip traversal attacks.

    Args:
        dest: Directory the archive is being extracted into.
        name: Member name as recorded in the archive.

    Returns:
        The absolute path the member may be written to.

    Raises:
        RuntimeError: If the member would land outside *dest*.
    """
    if not name or name in (".", "./"):
        return dest.resolve()
    if name.startswith(("/", "\\")):
        _reject(name, "absolute path")
    if len(name) > 1 and name[1] == ":":
        _reject(name, "drive-qualified path")
    normalised = name.replace("\\", "/")
    if any(part == ".." for part in PurePosixPath(normalised).parts):
        _reject(name, "path traversal via '..'")

    base = dest.resolve()
    target = (base / normalised).resolve()
    if target != base and base not in target.parents:
        _reject(name, f"resolves outside the destination ({target})")
    return target


def _checked_tar_members(
    tar: tarfile.TarFile, dest: Path, wanted: set[str] | None
) -> Iterator[tarfile.TarInfo]:
    """Yield the members of *tar* that are safe, and wanted, to extract."""
    base = dest.resolve()
    for member in tar:
        if wanted is not None and member.name not in wanted:
            continue
        target = safe_member_target(dest, member.name)
        if member.issym() or member.islnk():
            link = member.linkname.replace("\\", "/")
            if link.startswith("/"):
                _reject(member.name, "absolute link target")
            resolved = (target.parent / link).resolve()
            if resolved != base and base not in resolved.parents:
                _reject(member.name, f"link escapes destination ({resolved})")
        elif not (member.isfile() or member.isdir()):
            _reject(member.name, "not a regular file, directory or link")
        yield member


def _extract_tar(
    archive: Path | tarfile.TarFile,
    dest: Path,
    *,
    members: Sequence[str] | None = None,
    progress: bool = True,
    label: str = "extracting",
) -> int:
    """Extract a tar archive into *dest*, validating every member."""
    wanted = set(members) if members is not None else None
    opened = isinstance(archive, tarfile.TarFile)
    tar = archive if opened else tarfile.open(archive, "r:*")
    count = 0
    try:
        with _Progress(label, len(wanted) if wanted else None, unit="n",
                       enabled=progress) as bar:
            for member in _checked_tar_members(tar, dest, wanted):
                # filter="data" is defence in depth on 3.12+; the checks above
                # already cover every case it rejects, and older Pythons rely
                # on them alone.
                if sys.version_info >= (3, 12):
                    tar.extract(member, dest, filter="data")
                else:
                    tar.extract(member, dest)
                if member.isfile():
                    count += 1
                    bar.advance(1)
    finally:
        if not opened:
            tar.close()
    return count


def _extract_zip(
    archive: Path,
    dest: Path,
    *,
    members: Sequence[str] | None = None,
    progress: bool = True,
    label: str = "extracting",
) -> int:
    """Extract a zip archive into *dest*, validating every member."""
    wanted = set(members) if members is not None else None
    count = 0
    with zipfile.ZipFile(archive) as zf:
        entries = [
            info
            for info in zf.infolist()
            if wanted is None or info.filename in wanted
        ]
        for info in entries:
            safe_member_target(dest, info.filename)
        with _Progress(label, len(entries), unit="n", enabled=progress) as bar:
            for info in entries:
                zf.extract(info, dest)
                if not info.is_dir():
                    count += 1
                    bar.advance(1)
    return count


def extract_archive(
    archive: str | Path,
    dest: str | Path,
    *,
    fmt: str | None = None,
    members: Sequence[str] | None = None,
    progress: bool = True,
) -> int:
    """Extract *archive* into *dest*, refusing any member that escapes it.

    Every member name is validated before anything is written: absolute paths,
    drive letters, ``..`` components and links pointing out of *dest* are
    rejected outright, and the archive is not partially extracted past the
    offending member.

    Args:
        archive: Path to a ``.tar``, ``.tar.gz``/``.tgz`` or ``.zip`` file.
        dest: Directory to extract into; created if absent.
        fmt: ``"tar"``, ``"tar.gz"`` or ``"zip"``. Inferred from the file name
            when omitted.
        members: Extract only these member names, in archive order.
        progress: Whether to show a progress line on an interactive stderr.

    Returns:
        The number of regular files written.

    Raises:
        RuntimeError: If a member is unsafe, or the format cannot be inferred.
    """
    archive = Path(archive)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    if fmt is None:
        name = archive.name.lower()
        if name.endswith((".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")):
            fmt = "tar.gz"
        elif name.endswith(".tar"):
            fmt = "tar"
        elif name.endswith(".zip"):
            fmt = "zip"
        else:
            raise RuntimeError(
                f"cannot tell the archive format of {archive} from its name; "
                f"pass fmt='tar', 'tar.gz' or 'zip'."
            )

    label = f"extracting {archive.name}"
    if fmt in ("tar", "tar.gz"):
        return _extract_tar(
            archive, dest, members=members, progress=progress, label=label
        )
    if fmt == "zip":
        return _extract_zip(
            archive, dest, members=members, progress=progress, label=label
        )
    raise RuntimeError(
        f"unsupported archive format {fmt!r}; expected 'tar', 'tar.gz' or 'zip'."
    )


# ---------------------------------------------------------------------------
# ffmpeg
# ---------------------------------------------------------------------------


#: Environment variable naming an ffmpeg installation that is not on ``PATH``.
#: It may hold the path to the ``ffmpeg`` binary itself or the directory that
#: contains it; ``ffprobe`` is looked for beside it either way. ``PATH`` is
#: always searched first, so setting this never shadows a working install.
FFMPEG_ENV_VAR = "OMNIFALL_FFMPEG"


def _find_tool(name: str) -> str | None:
    """Return the path to *name*, searching ``PATH`` then :data:`FFMPEG_ENV_VAR`.

    Args:
        name: ``"ffmpeg"`` or ``"ffprobe"``.

    Returns:
        A usable path, or ``None`` if neither route finds an executable.
    """
    found = shutil.which(name)
    if found is not None:
        return found
    hint = os.environ.get(FFMPEG_ENV_VAR, "").strip()
    if not hint:
        return None
    candidate = Path(hint).expanduser()
    if candidate.is_dir():
        candidate = candidate / name
    elif candidate.name != name:
        candidate = candidate.with_name(name)
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    return None


def _require_tool(name: str) -> str:
    """Return the path to *name*, or raise with installation instructions."""
    found = _find_tool(name)
    if found is None:
        raise RuntimeError(
            f"{name} is required to convert this dataset (its videos are not "
            f"shipped as MP4) but was not found on PATH.\n"
            f"  conda:  conda install -c conda-forge ffmpeg\n"
            f"  debian: sudo apt install ffmpeg\n"
            f"  macos:  brew install ffmpeg\n"
            f"If ffmpeg is installed somewhere that is not on PATH, set "
            f"{FFMPEG_ENV_VAR} to the ffmpeg binary, or to the directory that "
            f"holds ffmpeg and ffprobe."
        )
    return found


def require_ffmpeg() -> str:
    """Return the path to the ``ffmpeg`` binary.

    Returns:
        Path to ``ffmpeg``, taken from ``PATH`` or from
        :data:`FFMPEG_ENV_VAR`.

    Raises:
        RuntimeError: If ``ffmpeg`` cannot be found. Several components ship
            AVI files, raw frame dumps or image sequences and genuinely cannot
            be converted without it.
    """
    return _require_tool("ffmpeg")


def require_ffprobe() -> str:
    """Return the path to the ``ffprobe`` binary.

    Conversion checks every file it produces, so ``ffprobe`` is as much a
    requirement as ``ffmpeg`` itself: without it a wrong frame rate or a
    truncated encode would pass unnoticed.

    Returns:
        Path to ``ffprobe``, taken from ``PATH`` or from
        :data:`FFMPEG_ENV_VAR`.

    Raises:
        RuntimeError: If ``ffprobe`` cannot be found.
    """
    return _require_tool("ffprobe")


@dataclass(frozen=True)
class _Stream:
    """What ``ffprobe`` reports about the first video stream of a file.

    Attributes:
        frames: Frame count, or ``None`` when the container does not record
            one.
        duration: Container duration in seconds, or ``None``.
        width: Frame width in pixels, or ``None``.
        height: Frame height in pixels, or ``None``.
    """

    frames: int | None
    duration: float | None
    width: int | None
    height: int | None


def _probe(path: str | Path) -> _Stream:
    """Return ffprobe's view of the first video stream of *path*.

    Args:
        path: A media file.

    Returns:
        A :class:`_Stream`. Fields ffprobe reports as ``N/A`` are ``None``.

    Raises:
        RuntimeError: If ffprobe cannot open the file at all.
    """
    binary = require_ffprobe()
    completed = subprocess.run(
        [
            binary, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=nb_frames,width,height:format=duration",
            "-of", "default=noprint_wrappers=1", str(path),
        ],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        tail = "\n".join(completed.stderr.strip().splitlines()[-8:])
        raise RuntimeError(
            f"ffprobe could not read {path} (exit {completed.returncode}):"
            f"\n{tail}"
        )
    fields: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        key, _, value = line.partition("=")
        if value and value != "N/A":
            fields[key.strip()] = value.strip()

    def _int(key: str) -> int | None:
        try:
            return int(fields[key])
        except (KeyError, ValueError):
            return None

    def _float(key: str) -> float | None:
        try:
            return float(fields[key])
        except (KeyError, ValueError):
            return None

    return _Stream(
        frames=_int("nb_frames"),
        duration=_float("duration"),
        width=_int("width"),
        height=_int("height"),
    )


@lru_cache(maxsize=None)
def _ffmpeg_has_option(name: str) -> bool:
    """Whether the available ffmpeg advertises the option *name*.

    ffmpeg's command line is not stable across major releases, and one of the
    recipes this module reproduces uses a combination that ffmpeg 7 started
    rejecting. Asking the binary what it supports is more honest than parsing
    its version string.

    Args:
        name: An option name without its leading dash, e.g.
            ``"enc_time_base"``.

    Returns:
        ``True`` if the option appears in ``ffmpeg -h full``.
    """
    completed = subprocess.run(
        [require_ffmpeg(), "-hide_banner", "-h", "full"],
        capture_output=True,
        text=True,
    )
    text = completed.stdout + completed.stderr
    # The listing spells an option either bare or with a stream specifier,
    # e.g. "-enc_time_base[:<stream_spec>] <ratio>".
    return re.search(rf"^\s*-{re.escape(name)}(\[|\s)", text, re.M) is not None


def _packet_times(path: Path) -> list[float]:
    """Return the presentation timestamps of *path*'s video packets, in seconds.

    Reads the container index rather than decoding, so this stays cheap even
    for a long recording.

    Args:
        path: An MP4.

    Returns:
        The timestamps ffprobe reports, in file order.

    Raises:
        RuntimeError: If ffprobe fails.
    """
    completed = subprocess.run(
        [
            require_ffprobe(), "-v", "error", "-select_streams", "v:0",
            "-show_entries", "packet=pts_time", "-of", "csv=p=0", str(path),
        ],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        tail = "\n".join(completed.stderr.strip().splitlines()[-8:])
        raise RuntimeError(
            f"ffprobe could not read the packet timestamps of {path} "
            f"(exit {completed.returncode}):\n{tail}"
        )
    times = []
    for line in completed.stdout.splitlines():
        value = line.strip().rstrip(",")
        if value and value != "N/A":
            times.append(float(value))
    return times


def _ffmpeg_command(args: Sequence[str], *, piped: bool) -> list[str]:
    """Return the full ffmpeg command line for *args*.

    Args:
        args: Arguments after the standard prefix.
        piped: Whether the input arrives on stdin, in which case ``-nostdin``
            is left out --- it would close the very pipe the frames go into.

    Returns:
        The argv list, ready to run and to quote into an error message.
    """
    prefix = ["-loglevel", "error", "-y"] if piped else [
        "-nostdin", "-loglevel", "error", "-y"
    ]
    return [require_ffmpeg(), *prefix, *args]


def _run_ffmpeg(
    args: Sequence[str],
    *,
    what: str,
    feed: Callable[[IO[bytes]], None] | None = None,
) -> None:
    """Run ffmpeg with *args*, raising with the full command and stderr.

    Args:
        args: Arguments after the standard ``-nostdin -loglevel error -y``
            prefix. Input options such as ``-r`` must precede their ``-i``,
            exactly as on a command line.
        what: A phrase completing "ffmpeg failed while ...", used in the error.
        feed: Called with ffmpeg's stdin when the input is piped raw video.
            ffmpeg's stderr is collected in a temporary file rather than a pipe
            so that a large frame stream cannot deadlock against a full stderr
            buffer.

    Raises:
        RuntimeError: If ffmpeg exits non-zero. The message carries the exact
            command and the tail of its stderr.
    """
    cmd = _ffmpeg_command(args, piped=feed is not None)
    if feed is None:
        completed = subprocess.run(cmd, capture_output=True, text=True)
        code, stderr = completed.returncode, completed.stderr
    else:
        with tempfile.TemporaryFile(mode="w+") as log:
            process = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log
            )
            assert process.stdin is not None  # stdin=PIPE guarantees it
            try:
                feed(process.stdin)
            except BrokenPipeError:
                pass  # ffmpeg died early; its stderr below says why
            finally:
                try:
                    process.stdin.close()
                except BrokenPipeError:
                    pass
                code = process.wait()
            log.seek(0)
            stderr = log.read()
    if code != 0:
        tail = "\n".join(stderr.strip().splitlines()[-12:])
        raise RuntimeError(
            f"ffmpeg failed while {what} (exit {code}).\n"
            f"  command: {shlex.join(cmd)}\n"
            f"  stderr:\n{tail}"
        )


def transcode(src: str | Path, dst: str | Path, *, crf: int = 20) -> None:
    """Re-encode a single video file to H.264 MP4.

    Args:
        src: Any container ffmpeg can read, e.g. an AVI.
        dst: Destination ``.mp4`` path; parent directories are created.
        crf: x264 quality factor; lower is better and larger.
    """
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        ["-i", str(src), "-c:v", "libx264", "-crf", str(crf),
         "-pix_fmt", "yuv420p", "-an", str(dst)],
        what=f"transcoding {src}",
    )


def frames_to_mp4(
    pattern: str | Path,
    dst: str | Path,
    *,
    fps: float = 30.0,
    crf: int = 20,
) -> None:
    """Encode a numbered image sequence into an MP4.

    Args:
        pattern: An ffmpeg input pattern such as ``".../frame_%05d.png"``, or a
            glob such as ``".../*.png"`` (which selects ffmpeg's glob demuxer).
        dst: Destination ``.mp4`` path; parent directories are created.
        fps: Frame rate to stamp on the output.
        crf: x264 quality factor.
    """
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    pattern = str(pattern)
    head = ["-framerate", str(fps)]
    if "*" in pattern or "?" in pattern:
        head = ["-pattern_type", "glob", *head]
    _run_ffmpeg(
        [*head, "-i", pattern, "-c:v", "libx264", "-crf", str(crf),
         "-pix_fmt", "yuv420p", str(dst)],
        what=f"encoding frames from {pattern}",
    )


def _encode(
    args: Sequence[str],
    out: Path,
    *,
    what: str,
    expected_frames: int | None = None,
    feed: Callable[[IO[bytes]], None] | None = None,
    check: Callable[[Path], None] | None = None,
) -> None:
    """Encode one video, then check it, then move it into place.

    The MP4 is written to a ``.part`` sibling and renamed only once every check
    has passed, so an interrupted or subtly wrong run never leaves a file that a
    later idempotent pass would mistake for a finished one.

    Args:
        args: Everything ffmpeg needs except the output path, which is
            appended together with an explicit ``-f mp4``.
        out: Final ``.mp4`` path; parent directories are created.
        what: A phrase completing "ffmpeg failed while ...".
        expected_frames: Frame count the result must have. ``None`` skips the
            check, which is right only where no independent count exists.
        feed: Passed to :func:`_run_ffmpeg` for piped raw-video input.
        check: An extra check on the finished file, called with the ``.part``
            path before it is renamed. It must raise to reject the result.

    Raises:
        RuntimeError: If ffmpeg fails, if the result is empty, if its frame
            count differs from *expected_frames*, or if *check* rejects it.
            Every message carries the exact command that was run.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part")
    full = [*args, "-f", "mp4", str(part)]
    shown = shlex.join(_ffmpeg_command(full, piped=feed is not None))
    try:
        _run_ffmpeg(full, what=what, feed=feed)
        if not part.is_file() or part.stat().st_size == 0:
            raise RuntimeError(
                f"ffmpeg reported success but produced no data for {out}.\n"
                f"  command: {shown}"
            )
        if expected_frames is not None:
            found = _probe(part).frames
            if found is None:
                raise RuntimeError(
                    f"ffprobe cannot count the frames of the file just written "
                    f"for {out}, so the conversion cannot be checked.\n"
                    f"  command: {shown}"
                )
            if found != expected_frames:
                raise RuntimeError(
                    f"{out} was encoded with {found} frames but the source has "
                    f"{expected_frames}. The conversion parameters are wrong "
                    f"for this file; refusing to keep it.\n"
                    f"  command: {shown}"
                )
        if check is not None:
            check(part)
        part.replace(out)
    finally:
        if part.exists():
            part.unlink()


# ---------------------------------------------------------------------------
# Reshaping an original tree into OmniFall's
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConvertReport:
    """What :func:`convert_tree` did.

    Attributes:
        dataset: The component that was converted.
        dst: The video directory that was written into.
        written: How many videos were newly created.
        skipped: How many were already present and left alone.
    """

    dataset: str
    dst: Path
    written: int
    skipped: int

    def summary(self) -> str:
        """Return a single-line description."""
        return (
            f"{self.dataset}: wrote {self.written}, kept {self.skipped} "
            f"existing, in {self.dst}"
        )


def convert_tree(
    src: str | Path,
    dst: str | Path,
    mapping: Mapping[str, str],
    *,
    dataset: str = "",
    fps: float = 30.0,
    frame_glob: str = "*.png",
    overwrite: bool = False,
    progress: bool = True,
) -> ConvertReport:
    """Copy or transcode an original tree into OmniFall's ``{path}.mp4`` layout.

    Every source named by *mapping* must exist. Missing sources are collected
    and reported together rather than failing on the first one, so a user who
    unpacked an incomplete download learns the full extent of the problem at
    once.

    Args:
        src: Root of the unpacked original release.
        dst: OmniFall video directory to write into.
        mapping: Maps a path relative to *src* to an OmniFall ``path`` value
            without extension. A source that is a directory is treated as an
            image sequence and encoded with ffmpeg; a source that is already an
            MP4 is copied; anything else is transcoded.
        dataset: Component name, used only for reporting.
        fps: Frame rate used when encoding image sequences.
        frame_glob: Glob matching the frames inside a sequence directory.
        overwrite: Rewrite outputs that already exist.
        progress: Whether to show a progress line on an interactive stderr.

    Returns:
        A :class:`ConvertReport`.

    Raises:
        FileNotFoundError: If any source named by *mapping* is absent.
        RuntimeError: If ffmpeg is needed but unavailable, or a conversion
            fails.
    """
    src = Path(src)
    dst = Path(dst)
    if not src.is_dir():
        raise FileNotFoundError(f"source directory does not exist: {src}")

    absent = [rel for rel in mapping if not (src / rel).exists()]
    if absent:
        shown = "\n".join(f"    {rel}" for rel in sorted(absent)[:10])
        more = f"\n    ... and {len(absent) - 10} more" if len(absent) > 10 else ""
        raise FileNotFoundError(
            f"{len(absent)} of {len(mapping)} expected source files are "
            f"missing under {src}:\n{shown}{more}\n"
            f"The unpacked release is incomplete, or {src} is not its root."
        )

    written = 0
    skipped = 0
    with _Progress(
        f"converting {dataset or src.name}", len(mapping), unit="n",
        enabled=progress,
    ) as bar:
        for rel, target in sorted(mapping.items()):
            source = src / rel
            out = dst / f"{target}.mp4"
            if out.exists() and out.stat().st_size > 0 and not overwrite:
                skipped += 1
                bar.advance(1)
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                frames_to_mp4(source / frame_glob, out, fps=fps)
            elif source.suffix.lower() == ".mp4":
                shutil.copyfile(source, out)
            else:
                transcode(source, out)
            written += 1
            bar.advance(1)

    return ConvertReport(
        dataset=dataset or src.name, dst=dst, written=written, skipped=skipped
    )


def _locate_root(base: Path, *markers: str, depth: int = 3) -> Path:
    """Return the directory under *base* that contains all *markers*.

    Archives are inconsistent about whether they nest everything under a single
    top-level directory, so the conversion functions locate their real root
    rather than assuming one.

    Args:
        base: Directory to search from.
        markers: Names that must all exist directly inside the answer.
        depth: How many levels below *base* to search.

    Returns:
        The matching directory.

    Raises:
        FileNotFoundError: If no directory within *depth* levels qualifies.
    """
    queue: list[tuple[Path, int]] = [(base, 0)]
    while queue:
        current, level = queue.pop(0)
        if all((current / marker).exists() for marker in markers):
            return current
        if level >= depth:
            continue
        try:
            children = sorted(p for p in current.iterdir() if p.is_dir())
        except OSError:
            continue
        queue.extend((child, level + 1) for child in children)
    if _expand_nested_archives(base, markers, depth=depth):
        return _locate_root(base, *markers, depth=depth)

    raise FileNotFoundError(
        f"could not find a directory under {base} containing "
        f"{', '.join(repr(m) for m in markers)} within {depth} levels. "
        f"Is {base} really the unpacked release?"
    )


def _archive_stem(name: str, suffixes: Sequence[str]) -> str | None:
    """Return *name* without its archive suffix, or ``None`` if it has none.

    Longest suffix first, so ``x.tar.gz`` does not come back as ``x.tar``.
    """
    for suffix in sorted(suffixes, key=len, reverse=True):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return None


def _expand_nested_archives(
    base: Path, markers: Sequence[str], *, depth: int = 3
) -> bool:
    """Unpack archives that stand where *markers* expect directories.

    Some releases are an archive of archives. Le2i's ``FallDataset.zip`` holds
    ``Coffee_room_01.zip``, ``Office.zip`` and four more, rather than the scene
    directories themselves. A converter tested against a copy somebody had
    already expanded by hand never meets that layer, so this exists to make the
    downloaded and hand-unpacked routes end in the same tree.

    Only archives whose name matches a marker are touched, and each is expanded
    beside itself. Nothing else in the tree is disturbed.

    Args:
        base: Directory to search from.
        markers: The names :func:`_locate_root` was looking for.
        depth: How many levels below *base* to search.

    Returns:
        Whether anything was expanded, i.e. whether a retry is worthwhile.
    """
    suffixes = (".zip", ".tar", ".tar.gz", ".tgz")
    expanded = False
    queue: list[tuple[Path, int]] = [(base, 0)]
    while queue:
        current, level = queue.pop(0)
        # A marker standing as an archive means this directory *is* the
        # archive-of-archives layer, so every archive beside it belongs to the
        # same release. Expanding only the marker-named ones would leave the
        # other scenes packed and the conversion short of files -- Le2i names
        # just two of its six scenes as markers.
        if any(
            (current / f"{marker}{suffix}").is_file()
            and not (current / marker).is_dir()
            for marker in markers
            for suffix in suffixes
        ):
            for archive in sorted(current.iterdir()):
                if not archive.is_file():
                    continue
                stem = _archive_stem(archive.name, suffixes)
                if stem is None or (current / stem).is_dir():
                    continue
                print(f"  expanding nested archive {archive.name}")
                extract_archive(archive, current, progress=False)
                expanded = True
        if level >= depth:
            continue
        try:
            children = sorted(p for p in current.iterdir() if p.is_dir())
        except OSError:
            continue
        queue.extend((child, level + 1) for child in children)
    return expanded


# ---------------------------------------------------------------------------
# The conversion recipes
#
# Every ffmpeg invocation below is the one the OmniFall authors actually used,
# taken from https://github.com/simplexsigil/omnifall-helper-scripts
# (video_conversion/) and checked against the published videos: for each
# component the codec, profile, frame rate, pixel format and frame count of a
# converted file were read back with ffprobe and matched the recipe. Do not
# "improve" these parameters. The published annotations are timed against the
# files these commands produce.
# ---------------------------------------------------------------------------

#: The x264 analysis options every staged-video component was encoded with.
_X264_OPTS = (
    "me=umh:subme=7:merange=24:psy-rd=1.0:aq-mode=3:aq-strength=0.8:"
    "rc-lookahead=60"
)

#: The encoder settings shared by le2i, caucafall, mcfd, cmdfall and up_fall:
#: a Baseline-profile stream with no B-frames and a single reference frame, so
#: that decoding stays cheap for dataloaders that seek constantly.
_X264_BASELINE: tuple[str, ...] = (
    "-c:v", "libx264",
    "-preset", "veryslow",
    "-tune", "fastdecode",
    "-profile:v", "baseline",
    "-refs", "1",
    "-bf", "0",
    "-x264opts", _X264_OPTS,
)


@dataclass(frozen=True)
class _Job:
    """One source recording and the OmniFall video it has to become.

    Attributes:
        target: The OmniFall ``path`` value, without extension.
        source: The file or directory the frames come from. Its existence is
            checked for every job before any encoding starts.
        out: The ``.mp4`` to write.
        encode: Does the work. Called from a worker thread.
    """

    target: str
    source: Path
    out: Path
    encode: Callable[[], None]


def _default_workers() -> int:
    """Return the default worker count: eight, or fewer on a small machine."""
    return max(1, min(8, os.cpu_count() or 1))


def _run_jobs(
    dataset: str,
    dst: Path,
    jobs: Sequence[_Job],
    *,
    overwrite: bool = False,
    workers: int | None = None,
    progress: bool = True,
) -> ConvertReport:
    """Run *jobs* in a thread pool, stopping at the first failure.

    Every source is checked before anything is encoded, so an incomplete
    download is reported in full rather than one file at a time. Outputs that
    already exist and are non-empty are left alone unless *overwrite* is set,
    which makes a re-run after an interruption cheap.

    Args:
        dataset: Component name, for messages and the report.
        dst: The OmniFall video directory being written into.
        jobs: What to convert.
        overwrite: Re-encode outputs that are already there.
        workers: Size of the pool; :func:`_default_workers` when omitted.
        progress: Whether to show a progress line on an interactive stderr.

    Returns:
        A :class:`ConvertReport`.

    Raises:
        FileNotFoundError: If any source is absent.
        RuntimeError: If any encode fails or produces an unexpected file. The
            first such error propagates and the remaining jobs are cancelled.
    """
    absent = [job for job in jobs if not job.source.exists()]
    if absent:
        shown = "\n".join(f"    {job.source}" for job in absent[:10])
        more = (
            f"\n    ... and {len(absent) - 10} more" if len(absent) > 10 else ""
        )
        raise FileNotFoundError(
            f"{len(absent)} of {len(jobs)} source recordings for {dataset} are "
            f"missing:\n{shown}{more}\n"
            f"The unpacked release is incomplete, or it is not the release "
            f"this converter expects."
        )

    todo = [
        job
        for job in jobs
        if overwrite or not (job.out.is_file() and job.out.stat().st_size > 0)
    ]
    skipped = len(jobs) - len(todo)
    written = 0
    pool_size = _default_workers() if workers is None else max(1, int(workers))

    with _Progress(
        f"converting {dataset}", len(todo), unit="n", enabled=progress
    ) as bar:
        pool = ThreadPoolExecutor(max_workers=pool_size)
        try:
            futures = [pool.submit(job.encode) for job in todo]
            for future in as_completed(futures):
                future.result()  # re-raises with the exact ffmpeg command
                written += 1
                bar.advance(1)
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            pool.shutdown(wait=True)

    return ConvertReport(
        dataset=dataset, dst=dst, written=written, skipped=skipped
    )


def _norm(name: str) -> str:
    """Reduce a file or directory name to a shape-insensitive key.

    Releases spell the same thing several ways --- ``Lecture room`` against
    ``Lecture_room``, ``video (10).avi`` against ``video_10.mp4`` --- so the
    le2i mapping is built on this rather than on any one spelling.

    Args:
        name: A single path component, with or without an extension.

    Returns:
        The name lower-cased, with every run of non-alphanumeric characters
        collapsed to a single underscore.
    """
    return re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_").lower()


def _transcode_job(
    target: str,
    source: Path,
    out: Path,
    *,
    args: Sequence[str],
    what: str,
    input_args: Sequence[str] = (),
) -> _Job:
    """Build a job that re-encodes one existing video file.

    The source's own frame count is read at encode time and the result is
    required to match it, which is what catches a frame-rate mistake: an input
    rate override must change the *duration* of the output and nothing else.

    Args:
        target: The OmniFall ``path`` value.
        source: The video to read.
        out: The ``.mp4`` to write.
        args: Output options, appended after ``-i``.
        what: A phrase completing "ffmpeg failed while ...".
        input_args: Options that must precede ``-i``, such as ``-r 30``.

    Returns:
        A :class:`_Job`.

    Raises:
        RuntimeError: At encode time, if the source's own frame count cannot be
            read. Encoding it unchecked is not an option: the check is the only
            thing standing between a frame-rate slip and a whole component of
            silently misaligned annotations.
    """

    def encode() -> None:
        frames = _probe(source).frames
        if frames is None:
            raise RuntimeError(
                f"ffprobe cannot count the frames of {source}, so the video "
                f"converted from it could not be checked. Refusing to encode "
                f"{target} unverified."
            )
        _encode(
            [*input_args, "-i", str(source), *args],
            out,
            what=what,
            expected_frames=frames,
        )

    return _Job(target=target, source=source, out=out, encode=encode)


# ---------------------------------------------------------------------------
# Per-dataset conversion
# ---------------------------------------------------------------------------


def _convert_gmdcsa24(
    src: Path,
    dst: Path,
    *,
    overwrite: bool = False,
    workers: int | None = None,
    progress: bool = True,
) -> ConvertReport:
    """Convert the GMDCSA24 GitHub checkout into OmniFall's layout.

    The repository stores the 160 MP4s directly, under subject directories
    spelled with a space (``Subject 1``) where OmniFall uses an underscore
    (``Subject_1``). Nothing is re-encoded.

    Args:
        src: Root of the unpacked repository tarball.
        dst: OmniFall video directory for ``GMDCSA24``.
        overwrite: Rewrite outputs that already exist.
        workers: Ignored --- this conversion copies files rather than encoding
            them, so a pool would only add contention on one disk.
        progress: Whether to show a progress line.

    Returns:
        A :class:`ConvertReport`.
    """
    root = _locate_root(src, "Subject 1", "Subject 2")
    mapping = {
        f"{path.replace('Subject_', 'Subject ', 1)}.mp4": path
        for path in required_paths("GMDCSA24")
    }
    return convert_tree(
        root, dst, mapping, dataset="GMDCSA24",
        overwrite=overwrite, progress=progress,
    )


def _convert_of_syn(
    src: Path,
    dst: Path,
    *,
    overwrite: bool = False,
    workers: int | None = None,
    progress: bool = True,
) -> ConvertReport:
    """No-op: the OF-Syn tar already carries OmniFall's exact layout.

    Its members are ``./{label}/{stem}.mp4``, so extraction into the video
    directory is the whole conversion.

    Args:
        src: Ignored.
        dst: The video directory the tar was extracted into.
        overwrite: Ignored; there is nothing to rewrite.
        workers: Ignored; there is nothing to encode.
        progress: Ignored.

    Returns:
        A :class:`ConvertReport` counting the extracted files as skipped.
    """
    existing = sum(1 for _ in dst.rglob("*.mp4")) if dst.is_dir() else 0
    return ConvertReport(dataset="of-syn", dst=dst, written=0, skipped=existing)


def _not_implemented(dataset: str, detail: str) -> Callable[..., ConvertReport]:
    """Return a converter that explains precisely what is still missing."""

    def _raise(
        src: Path,
        dst: Path,
        *,
        overwrite: bool = False,
        workers: int | None = None,
        progress: bool = True,
    ) -> ConvertReport:
        raise ConversionNotImplementedError(dataset, detail)

    return _raise


def _unresolved(
    dataset: str, root: Path, targets: Sequence[str], expected: str
) -> None:
    """Raise because some required recordings could not be located.

    Args:
        dataset: Component name.
        root: The directory that was searched.
        targets: The OmniFall ``path`` values with no source.
        expected: Where each of them was expected, in prose.

    Raises:
        FileNotFoundError: Always.
    """
    shown = "\n".join(f"    {target}" for target in sorted(targets)[:10])
    more = f"\n    ... and {len(targets) - 10} more" if len(targets) > 10 else ""
    raise FileNotFoundError(
        f"{len(targets)} of the recordings {dataset} needs could not be "
        f"located under {root}:\n{shown}{more}\n"
        f"Expected {expected}\n"
        f"The unpacked release is incomplete, or it is not the release this "
        f"converter expects."
    )


def _index_by_stem(
    root: Path, suffix: str, *, depth_key: Callable[[Path], tuple[str, ...]]
) -> dict[tuple[str, ...], Path]:
    """Index every ``*{suffix}`` file under *root* by a normalised key.

    Args:
        root: Directory to walk.
        suffix: File extension including the dot, e.g. ``".avi"``.
        depth_key: Maps a path relative to *root* to the key it is filed under.

    Returns:
        The index.

    Raises:
        RuntimeError: If two different files claim the same key --- which would
            make the mapping ambiguous, and is never something to guess about.
    """
    index: dict[tuple[str, ...], Path] = {}
    for found in sorted(root.rglob(f"*{suffix}")):
        if not found.is_file():
            continue
        key = depth_key(found.relative_to(root))
        previous = index.get(key)
        if previous is not None:
            raise RuntimeError(
                f"two source files map to the same OmniFall name under "
                f"{root}:\n    {previous}\n    {found}\n"
                f"Remove the duplicate copy and retry; guessing which one the "
                f"labels refer to is not acceptable."
            )
        index[key] = found
    return index


def _convert_le2i(
    src: Path,
    dst: Path,
    *,
    overwrite: bool = False,
    workers: int | None = None,
    progress: bool = True,
) -> ConvertReport:
    """Transcode the Le2i release into OmniFall's layout.

    The release groups AVI files by scene, but not consistently: four scenes
    nest their videos in a ``Videos/`` subdirectory and two do not, one scene
    directory is spelled ``Lecture room`` where OmniFall writes
    ``Lecture_room``, and the files themselves are named ``video (10).avi``
    where OmniFall writes ``video_10.mp4``. The mapping is therefore built on
    (scene, stem) keys normalised by :func:`_norm` rather than on any single
    spelling.

    Encoding follows ``video_conversion/le2i.sh``: Baseline profile, 25-frame
    GOP, CRF 20, ``yuv420p``. The sources are 25 fps raw video and carry an
    audio track, which is copied through exactly as the published files do.

    Args:
        src: Root of the unpacked ``FallDataset.zip``.
        dst: OmniFall video directory for ``le2i``.
        overwrite: Re-encode outputs that already exist.
        workers: Size of the conversion pool.
        progress: Whether to show a progress line.

    Returns:
        A :class:`ConvertReport`.
    """
    root = _locate_root(src, "Coffee_room_01", "Office")
    index = _index_by_stem(
        root,
        ".avi",
        depth_key=lambda rel: (_norm(rel.parts[0]), _norm(rel.stem)),
    )

    jobs: list[_Job] = []
    missing: list[str] = []
    for path in required_paths("le2i"):
        scene, _, stem = path.rpartition("/")
        source = index.get((_norm(scene), _norm(stem)))
        if source is None:
            missing.append(path)
            continue
        jobs.append(
            _transcode_job(
                path,
                source,
                dst / f"{path}.mp4",
                args=[
                    "-pix_fmt", "yuv420p", *_X264_BASELINE,
                    "-g", "25", "-crf", "20",
                ],
                what=f"transcoding le2i {path}",
            )
        )
    if missing:
        _unresolved(
            "le2i", root, missing,
            "one AVI per scene named 'video (N).avi' or 'video_N.avi', "
            "directly in the scene directory or in its 'Videos' "
            "subdirectory.",
        )
    return _run_jobs(
        "le2i", dst, jobs,
        overwrite=overwrite, workers=workers, progress=progress,
    )


def _convert_caucafall(
    src: Path,
    dst: Path,
    *,
    overwrite: bool = False,
    workers: int | None = None,
    progress: bool = True,
) -> ConvertReport:
    """Transcode the CAUCAFall release into OmniFall's layout.

    The Mendeley release is organised by subject
    (``CAUCAFall/Subject.7/Fall left/``) and each activity directory holds both
    the PNG frames and an AVI the authors assembled from them. OmniFall uses
    that AVI: it is named exactly like the OmniFall stem
    (``FallLeftS7.avi``), which makes the mapping unambiguous, and it is one
    frame shorter than the PNG sequence --- so re-encoding the PNGs would
    silently shift every annotation in the file by one frame.

    Encoding follows ``video_conversion/caucafall.sh``: Baseline profile,
    20-frame GOP, CRF 24, and no pixel-format override (the sources are already
    ``yuv420p``).

    Args:
        src: Root of the unpacked Mendeley download.
        dst: OmniFall video directory for ``caucafall``.
        overwrite: Re-encode outputs that already exist.
        workers: Size of the conversion pool.
        progress: Whether to show a progress line.

    Returns:
        A :class:`ConvertReport`.
    """
    root = _locate_root(src, "Subject.1", "Subject.10")
    index = _index_by_stem(
        root, ".avi", depth_key=lambda rel: (_norm(rel.stem),)
    )

    jobs: list[_Job] = []
    missing: list[str] = []
    for path in required_paths("caucafall"):
        stem = path.rpartition("/")[2]
        source = index.get((_norm(stem),))
        if source is None:
            missing.append(path)
            continue
        jobs.append(
            _transcode_job(
                path,
                source,
                dst / f"{path}.mp4",
                args=[*_X264_BASELINE, "-g", "20", "-crf", "24"],
                what=f"transcoding caucafall {path}",
            )
        )
    if missing:
        _unresolved(
            "caucafall", root, missing,
            "one AVI per activity directory, named after the activity and "
            "the subject, e.g. Subject.7/Fall left/FallLeftS7.avi.",
        )
    return _run_jobs(
        "caucafall", dst, jobs,
        overwrite=overwrite, workers=workers, progress=progress,
    )


def _convert_mcfd(
    src: Path,
    dst: Path,
    *,
    overwrite: bool = False,
    workers: int | None = None,
    progress: bool = True,
) -> ConvertReport:
    """Transcode the Multiple Cameras Fall Dataset into OmniFall's layout.

    The layout is already OmniFall's --- ``dataset/chute01/cam1.avi`` --- but
    the frame rate is not. The release stores its recordings with a 120 fps
    header even though they were captured at 30 fps, so
    ``video_conversion/mcfd.sh`` passes ``-r 30`` *before* ``-i``. That is an
    input-rate override, not a resample: no frame is added or dropped, the
    output simply lasts four times as long. Moving that flag after ``-i``, or
    leaving it out, produces videos that play four times too fast and misaligns
    every segment boundary in ``labels/mcfd.csv``. The frame-count check on
    each output is what would catch a resample.

    Args:
        src: Root of the unpacked release.
        dst: OmniFall video directory for ``mcfd``.
        overwrite: Re-encode outputs that already exist.
        workers: Size of the conversion pool.
        progress: Whether to show a progress line.

    Returns:
        A :class:`ConvertReport`.
    """
    root = _locate_root(src, "chute01", "chute24")
    jobs = [
        _transcode_job(
            path,
            root / f"{path}.avi",
            dst / f"{path}.mp4",
            args=[
                *_X264_BASELINE,
                "-g", "30", "-keyint_min", "30", "-crf", "24",
            ],
            what=f"transcoding mcfd {path}",
            input_args=["-r", "30"],
        )
        for path in required_paths("mcfd")
    ]
    return _run_jobs(
        "mcfd", dst, jobs,
        overwrite=overwrite, workers=workers, progress=progress,
    )


def _convert_cmdfall(
    src: Path,
    dst: Path,
    *,
    overwrite: bool = False,
    workers: int | None = None,
    progress: bool = True,
) -> ConvertReport:
    """Transcode the CMDFall RGB recordings into OmniFall's layout.

    OmniFall uses only the ``colors`` stream, whose file names already match
    its ``path`` values. The release carries more of them than OmniFall
    references, plus depth, skeleton and per-clip directories; everything not
    named by ``labels/cmdfall.csv`` is ignored.

    Encoding follows ``video_conversion/cmdfall_colors.sh``: Baseline profile,
    20-frame GOP, CRF 20, and no pixel-format override --- the sources are
    full-range ``yuvj420p`` MJPEG and the published videos keep that.

    Args:
        src: Root of the unpacked release, or the directory holding
            ``colors/``.
        dst: OmniFall video directory for ``cmdfall``.
        overwrite: Re-encode outputs that already exist.
        workers: Size of the conversion pool.
        progress: Whether to show a progress line.

    Returns:
        A :class:`ConvertReport`.
    """
    root = _locate_root(src, "colors")
    jobs = [
        _transcode_job(
            path,
            root / f"{path}.avi",
            dst / f"{path}.mp4",
            args=[*_X264_BASELINE, "-g", "20", "-crf", "20"],
            what=f"transcoding cmdfall {path}",
        )
        for path in required_paths("cmdfall")
    ]
    return _run_jobs(
        "cmdfall", dst, jobs,
        overwrite=overwrite, workers=workers, progress=progress,
    )


# ---------------------------------------------------------------------------
# edf and occu: raw .bin frame dumps
# ---------------------------------------------------------------------------


def _bin_frame_shape(path: Path) -> tuple[int, int]:
    """Return the ``(width, height)`` of one EDF/OCCU ``.bin`` frame.

    The format, taken from ``video_conversion/edf_occu/create_rgb_video.py``,
    is a four-``uint16`` header followed by planar 8-bit RGB. The header's two
    size fields are transposed relative to the image that comes out of the
    decode: ``header[3]`` is the width and ``header[2]`` the height. The
    original script calls them ``rows`` and ``cols`` respectively, which is why
    it prints its resolution the wrong way round; the pixels are unaffected.

    Args:
        path: A ``.bin`` frame.

    Returns:
        ``(width, height)`` in pixels.

    Raises:
        RuntimeError: If the file is too short, or its length disagrees with
            its own header.
    """
    import numpy as np

    header = np.fromfile(path, dtype=np.uint16, count=4)
    if header.size < 4:
        raise RuntimeError(f"{path} is too short to be an EDF/OCCU frame.")
    width, height = int(header[3]), int(header[2])
    expected = 4 + (3 * width * height + 1) // 2
    actual = path.stat().st_size // 2
    if width <= 0 or height <= 0 or actual != expected:
        raise RuntimeError(
            f"{path} does not look like an EDF/OCCU RGB frame: its header says "
            f"{width}x{height}, which needs {expected} uint16 words, but the "
            f"file holds {actual}."
        )
    return width, height


def _read_bin_frame(path: Path, width: int, height: int) -> object:
    """Decode one EDF/OCCU ``.bin`` frame into an ``(H, W, 3)`` RGB array.

    Args:
        path: The frame to read.
        width: Width every frame of the sequence must have.
        height: Height every frame of the sequence must have.

    Returns:
        A ``numpy`` array of shape ``(height, width, 3)``, channel order RGB.

    Raises:
        RuntimeError: If the frame's size differs from the sequence's, which
            would make a single video out of two different captures.
    """
    import numpy as np

    data = np.fromfile(path, dtype=np.uint16)
    rows, cols = int(data[3]), int(data[2])
    if (rows, cols) != (width, height):
        raise RuntimeError(
            f"{path} is {rows}x{cols} but the rest of the sequence is "
            f"{width}x{height}. Refusing to encode a sequence whose frame "
            f"size changes."
        )
    return data[4:].view(np.uint8).reshape(3, rows, cols).transpose((2, 1, 0))


def _convert_bin_sequences(
    dataset: str,
    src: Path,
    dst: Path,
    *,
    overwrite: bool = False,
    workers: int | None = None,
    progress: bool = True,
) -> ConvertReport:
    """Encode the RGB ``.bin`` dumps of edf or occu into OmniFall's layout.

    Both releases store one subject per directory, doubled
    (``EDF/jianjun/jianjun/view1/rgb/000001.bin``), and carry a third view plus
    depth streams that OmniFall does not use. Frames are taken in sorted file
    order, which is numeric because the names are zero-padded.

    The frames are piped to ffmpeg as raw RGB rather than written out as PNGs
    first. That is what ``create_rgb_video.py`` does by way of a temporary
    directory, and PNG is lossless, so the encoder sees the same bytes either
    way --- without spending gigabytes of scratch space per sequence. The
    encoder settings are that script's: plain libx264 at CRF 22, 30 fps,
    ``yuv420p``, and deliberately *no* Baseline profile, which is why the
    published edf and occu videos are High profile where every other component
    is Baseline.

    Args:
        dataset: ``"edf"`` or ``"occu"``.
        src: Root of the unpacked release.
        dst: OmniFall video directory for *dataset*.
        overwrite: Re-encode outputs that already exist.
        workers: Size of the conversion pool.
        progress: Whether to show a progress line.

    Returns:
        A :class:`ConvertReport`.
    """
    paths = required_paths(dataset)
    subjects = sorted({path.split("/")[0] for path in paths})
    root = _locate_root(src, *subjects)

    def build(path: str) -> Path:
        """Return the directory holding *path*'s frames, or a missing one."""
        parts = path.split("/")
        subject, view, stream = parts[0], parts[1], parts[2]
        doubled = root / subject / subject / view / stream
        if doubled.is_dir():
            return doubled
        return root / subject / view / stream

    jobs: list[_Job] = []
    for path in paths:
        source = build(path)
        out = dst / f"{path}.mp4"
        jobs.append(
            _Job(
                target=path,
                source=source,
                out=out,
                encode=_bin_encoder(dataset, path, source, out),
            )
        )
    return _run_jobs(
        dataset, dst, jobs,
        overwrite=overwrite, workers=workers, progress=progress,
    )


def _bin_encoder(
    dataset: str, target: str, source: Path, out: Path
) -> Callable[[], None]:
    """Return the callable that encodes one ``.bin`` frame directory."""

    def encode() -> None:
        frames = sorted(source.glob("*.bin"))
        if not frames:
            raise RuntimeError(
                f"{source} holds no .bin frames, so {dataset} {target} cannot "
                f"be encoded."
            )
        width, height = _bin_frame_shape(frames[0])

        def feed(stream: IO[bytes]) -> None:
            for frame in frames:
                stream.write(_read_bin_frame(frame, width, height).tobytes())

        _encode(
            [
                "-f", "rawvideo",
                "-pix_fmt", "rgb24",
                "-video_size", f"{width}x{height}",
                "-framerate", "30",
                "-i", "-",
                "-vcodec", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "22",
            ],
            out,
            what=f"encoding {dataset} {target} from {len(frames)} .bin frames",
            expected_frames=len(frames),
            feed=feed,
        )

    return encode


def _convert_edf(
    src: Path,
    dst: Path,
    *,
    overwrite: bool = False,
    workers: int | None = None,
    progress: bool = True,
) -> ConvertReport:
    """Encode the EDF RGB frame dumps into OmniFall's layout.

    See :func:`_convert_bin_sequences` for the format and the encoder
    settings.

    Args:
        src: Root of the unpacked ``EDF.zip``.
        dst: OmniFall video directory for ``edf``.
        overwrite: Re-encode outputs that already exist.
        workers: Size of the conversion pool.
        progress: Whether to show a progress line.

    Returns:
        A :class:`ConvertReport`.
    """
    return _convert_bin_sequences(
        "edf", src, dst,
        overwrite=overwrite, workers=workers, progress=progress,
    )


def _convert_occu(
    src: Path,
    dst: Path,
    *,
    overwrite: bool = False,
    workers: int | None = None,
    progress: bool = True,
) -> ConvertReport:
    """Encode the OCCU RGB frame dumps into OmniFall's layout.

    Identical in shape to :func:`_convert_edf`; the two releases share a paper,
    a Zenodo record and a file format.

    Args:
        src: Root of the unpacked ``OCCU.zip``.
        dst: OmniFall video directory for ``occu``.
        overwrite: Re-encode outputs that already exist.
        workers: Size of the conversion pool.
        progress: Whether to show a progress line.

    Returns:
        A :class:`ConvertReport`.
    """
    return _convert_bin_sequences(
        "occu", src, dst,
        overwrite=overwrite, workers=workers, progress=progress,
    )


# ---------------------------------------------------------------------------
# up_fall: timestamped PNG frames at a genuinely variable frame rate
# ---------------------------------------------------------------------------

#: The factor ``up_fall.py`` scales concat durations by before ffmpeg divides
#: them back out through ``setpts``. The concat demuxer rounds each duration to
#: the timebase of the image it belongs to, which is 1/25 s; blowing the
#: durations up by a million first makes that rounding negligible, and dividing
#: the timestamps back down afterwards recovers microsecond precision.
_UP_FALL_TIMESCALE = 1_000_000

#: The MP4 timescale UP-Fall's variable-rate videos are stamped with.
_UP_FALL_TRACK_TIMESCALE = 90_000


def _up_fall_rate_args() -> tuple[str, ...]:
    """Return the flags that keep UP-Fall's frame timestamps at full precision.

    ``up_fall.py`` asks for ``-r 1000000``, which on the ffmpeg of its day set
    the encoder's timebase fine enough for the divided-down timestamps to
    survive. ffmpeg 7 rejects an output frame rate alongside a variable-rate
    ``-vsync``, and without it the encoder inherits a 1/25 s timebase and snaps
    every frame onto a 40 ms grid --- which is exactly the drift against
    ``labels/up_fall.csv`` this whole detour exists to avoid.

    Where ffmpeg offers ``-enc_time_base`` (6.0 and later) the timebase is
    therefore set directly instead. That was checked against the published
    videos: every frame lands on the same 1/90000 tick either way. The only
    difference is the container's total duration, which comes out one nominal
    frame period longer because the final sample is given a real duration
    rather than a rounded-to-zero one.

    Returns:
        The flags to place before the encoder options.
    """
    if _ffmpeg_has_option("enc_time_base"):
        return ("-enc_time_base", f"1/{_UP_FALL_TRACK_TIMESCALE}")
    return ("-r", str(_UP_FALL_TIMESCALE))


def _up_fall_time(frame: Path) -> datetime:
    """Return the capture time encoded in an UP-Fall frame's file name.

    Args:
        frame: A file named ``%Y-%m-%dT%H_%M_%S.<microseconds>.png``.

    Returns:
        The timestamp, to microsecond precision.

    Raises:
        RuntimeError: If the name does not carry a timestamp. Falling back to a
            constant frame rate here would misplace every annotation in the
            file, so this is fatal rather than approximated.
    """
    parts = frame.name.rsplit(".", 2)
    try:
        stamp = datetime.strptime(parts[0], "%Y-%m-%dT%H_%M_%S")
        return stamp + timedelta(microseconds=int(parts[1]))
    except (IndexError, ValueError) as error:
        raise RuntimeError(
            f"{frame} is not named like an UP-Fall frame "
            f"(%Y-%m-%dT%H_%M_%S.<microseconds>.png), so its capture time is "
            f"unknown and the video cannot be timed correctly."
        ) from error


def _check_up_fall_timing(
    out: Path, target: str, times: Sequence[datetime]
) -> None:
    """Raise unless every frame of *out* sits where its source timestamp says.

    The frame count alone would not catch a lost timebase: a video with the
    right number of frames on a 40 ms grid looks fine until its annotations are
    read. This compares the encoded presentation timestamps against the capture
    times the file names carry, which is the property the labels depend on.

    Args:
        out: The encoded MP4.
        target: The OmniFall ``path`` value, for the error message.
        times: Capture times of the source frames, in order.

    Raises:
        RuntimeError: If any frame is off by more than one tick of the track
            timescale.
    """
    tolerance = 2.0 / _UP_FALL_TRACK_TIMESCALE
    encoded = _packet_times(out)
    if len(encoded) != len(times):
        raise RuntimeError(
            f"up_fall {target}: the encoded video has {len(encoded)} packets "
            f"but {len(times)} frames went in."
        )
    origin = times[0]
    for index, (stamp, seen) in enumerate(zip(times, encoded)):
        wanted = (stamp - origin).total_seconds()
        if abs(seen - wanted) > tolerance:
            raise RuntimeError(
                f"up_fall {target}: frame {index} was encoded at {seen:.6f}s "
                f"but its file name says {wanted:.6f}s. The frame timing was "
                f"lost, so the segment boundaries in labels/up_fall.csv would "
                f"not line up. Refusing to keep the file."
            )


def _up_fall_encoder(target: str, source: Path, out: Path) -> Callable[[], None]:
    """Return the callable that encodes one UP-Fall frame directory."""

    def encode() -> None:
        frames = sorted(source.glob("*.png"))
        if not frames:
            raise RuntimeError(
                f"{source} holds no PNG frames, so up_fall {target} cannot be "
                f"encoded."
            )
        times = [_up_fall_time(frame) for frame in frames]

        handle = tempfile.NamedTemporaryFile(
            mode="w", suffix=".ffconcat", delete=False
        )
        try:
            handle.write("ffconcat version 1.0\n")
            for index, frame in enumerate(frames):
                handle.write(f"file {shlex.quote(str(frame.resolve()))}\n")
                if index == len(frames) - 1:
                    continue
                seconds = (times[index + 1] - times[index]).total_seconds()
                if seconds <= 0:
                    raise RuntimeError(
                        f"up_fall {target}: {frames[index + 1].name} is not "
                        f"later than {frame.name}, so the frame order and the "
                        f"timestamps disagree. Refusing to guess."
                    )
                handle.write(
                    f"duration {round(seconds * _UP_FALL_TIMESCALE, 8)}\n"
                )
            handle.close()
            timescale = _UP_FALL_TRACK_TIMESCALE
            _encode(
                [
                    "-f", "concat", "-safe", "0", "-i", handle.name,
                    "-vsync", "2",
                    "-copyts",
                    *_up_fall_rate_args(),
                    *_X264_BASELINE,
                    "-g", "18",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    "-video_track_timescale", str(timescale),
                    "-vf",
                    f"settb=1/{timescale},setpts=PTS/{_UP_FALL_TIMESCALE}",
                    "-crf", "24",
                ],
                out,
                what=f"encoding up_fall {target} from {len(frames)} frames",
                expected_frames=len(frames),
                check=lambda written: _check_up_fall_timing(
                    written, target, times
                ),
            )
        finally:
            handle.close()
            os.unlink(handle.name)

    return encode


def _convert_up_fall(
    src: Path,
    dst: Path,
    *,
    overwrite: bool = False,
    workers: int | None = None,
    progress: bool = True,
) -> ConvertReport:
    """Encode the UP-Fall camera frames into OmniFall's layout.

    The camera archives unpack to one directory per recording, at exactly the
    OmniFall ``path`` --- ``Subject1/Activity1/Trial1/
    Subject1Activity1Trial1Camera1/`` --- holding PNG frames whose names are
    their capture timestamps.

    Those timestamps are the point. UP-Fall was captured over a network at a
    rate that wanders either side of 18 fps, so ``video_conversion/up_fall/
    up_fall.py`` builds an ffconcat file with the real inter-frame gaps and
    encodes a genuinely variable-rate video with a 90 kHz timescale. The
    published videos are variable-rate accordingly. Stamping a constant frame
    rate on them instead would drift against ``labels/up_fall.csv``, whose
    boundaries are in seconds, by a growing amount across each recording.

    Args:
        src: Root of the unpacked camera archives.
        dst: OmniFall video directory for ``up_fall``.
        overwrite: Re-encode outputs that already exist.
        workers: Size of the conversion pool.
        progress: Whether to show a progress line.

    Returns:
        A :class:`ConvertReport`.
    """
    paths = required_paths("up_fall")
    subjects = sorted(
        {path.split("/")[0] for path in paths},
        key=lambda name: int(name.removeprefix("Subject")),
    )
    root = _locate_root(src, subjects[0], subjects[-1])
    jobs = [
        _Job(
            target=path,
            source=root / path,
            out=dst / f"{path}.mp4",
            encode=_up_fall_encoder(path, root / path, dst / f"{path}.mp4"),
        )
        for path in paths
    ]
    return _run_jobs(
        "up_fall", dst, jobs,
        overwrite=overwrite, workers=workers, progress=progress,
    )


_convert_oops = _not_implemented(
    "OOPS",
    "OOPS is prepared by streaming, not by converting an unpacked tree. Use "
    "'omnifall prepare OOPS', or omnifall._oops.prepare_oops() with a local "
    "copy of video_and_anns.tar.gz.",
)


#: Maps a component to the function that reshapes its unpacked release.
_CONVERTERS: dict[str, Callable[..., ConvertReport]] = {
    "GMDCSA24": _convert_gmdcsa24,
    "of-syn": _convert_of_syn,
    "OOPS": _convert_oops,
    "le2i": _convert_le2i,
    "edf": _convert_edf,
    "occu": _convert_occu,
    "caucafall": _convert_caucafall,
    "mcfd": _convert_mcfd,
    "up_fall": _convert_up_fall,
    "cmdfall": _convert_cmdfall,
}

#: Components this package can obtain end to end, without the user touching a
#: browser. OOPS is here even though ``_convert_oops`` raises: it is prepared by
#: streaming the archive rather than by reshaping an unpacked tree, so it never
#: goes through a converter at all.
#:
#: Only cmdfall is absent, and not for want of a converter --- it has one. Its
#: videos are released per e-mail request, so there is nothing to automate.
#: ``omnifall sources cmdfall`` names the directory to unpack it into, after
#: which ``omnifall prepare cmdfall`` finishes the job like any other
#: component.
IMPLEMENTED_CONVERSIONS: frozenset[str] = frozenset(
    {
        "GMDCSA24", "of-syn", "OOPS", "le2i", "edf", "occu",
        "caucafall", "mcfd", "up_fall",
    }
)


def convert(
    dataset: str,
    src: str | Path,
    *,
    video_dir: str | Path | None = None,
    overwrite: bool = False,
    workers: int | None = None,
    progress: bool = True,
) -> ConvertReport:
    """Reshape an already-downloaded release into OmniFall's layout.

    This is the entry point for the manual route: obtain the original data by
    whatever means the source demands, unpack it, then point this at it. The
    result is checked against ``labels/{dataset}.csv`` before this returns, so
    a report that comes back is a tree the loader can read end to end.

    Args:
        dataset: Exact ``dataset`` column value.
        src: Root of the unpacked original release.
        video_dir: OmniFall video directory to write into; resolved the usual
            way when omitted.
        overwrite: Rewrite videos that already exist.
        workers: How many videos to encode at once. Defaults to eight, or the
            number of CPUs where that is smaller.
        progress: Whether to show a progress line.

    Returns:
        A :class:`ConvertReport`.

    Raises:
        KeyError: If *dataset* is not an OmniFall component.
        ConversionNotImplementedError: If no converter exists for it yet.
        RuntimeError: If the converted tree is still incomplete afterwards.
    """
    get_source(dataset)
    target = dataset_video_dir(dataset, override=video_dir)
    target.mkdir(parents=True, exist_ok=True)
    report = _CONVERTERS[dataset](
        Path(src), target,
        overwrite=overwrite, workers=workers, progress=progress,
    )
    check = verify(dataset, video_dir=video_dir)
    if not check.complete:
        raise RuntimeError(
            f"{dataset} was converted but the result is incomplete:\n"
            f"{check.render()}"
        )
    return report


# ---------------------------------------------------------------------------
# Per-dataset acquisition
# ---------------------------------------------------------------------------


#: Leading bytes each archive format must have, used to tell a real download
#: from a bot-check or quota page that arrived with an HTTP 200.
_ARCHIVE_MAGIC: dict[str, bytes] = {
    "zip": b"PK\x03\x04",
    "tar.gz": b"\x1f\x8b",
}


def _magic_for(source: Source) -> bytes | None:
    """Return the leading bytes *source*'s archives must start with."""
    if source.archive_format is None:
        return None
    return _ARCHIVE_MAGIC.get(source.archive_format)


def _nonempty(directory: Path) -> bool:
    """Report whether *directory* exists and holds anything at all."""
    return directory.is_dir() and any(directory.iterdir())


def _extract_marker(locations: DownloadLocations, name: str) -> Path:
    """Return the file recording that one archive has been fully extracted.

    Extraction is not atomic, so "the unpacked directory is not empty" cannot
    distinguish a finished extraction from an interrupted one, nor from a
    release the user unpacked there personally. A marker written *after* an
    archive is extracted can.

    Args:
        locations: Where the component's release lives.
        name: The archive's file name.

    Returns:
        The marker path. It may not exist.
    """
    return locations.directory / ".extracted" / f"{name}.done"


def _acquire_archives(
    dataset: str,
    locations: DownloadLocations,
    *,
    progress: bool = True,
) -> list[Path]:
    """Return every archive of *dataset*, downloading the ones not already there.

    A file already in the download directory is used without touching the
    network, whether omnifall put it there or the user did. That is the whole
    of the "bring your own download" path: there is no separate code for it,
    because a separate path is one that stops being tested.

    Args:
        dataset: A component whose archive names are recorded.
        locations: Where its release lives.
        progress: Whether to show a progress line.

    Returns:
        The archives, in :attr:`Source.files` order.

    Raises:
        DatasetNotAvailableError: If an archive is absent and cannot be
            fetched.
        RuntimeError: If a download fails, or answers with a page.
    """
    source = get_source(dataset)
    names = source.archive_names()
    if not names:
        raise RuntimeError(
            f"{dataset} records no archive names, so this routine cannot "
            f"acquire it. This is an omnifall bug."
        )

    present = [n for n in names if (locations.directory / n).is_file()]
    if present and len(present) == len(names):
        print(
            f"  found all {len(names)} archive(s) already in "
            f"{locations.directory}; not downloading",
            file=sys.stderr,
        )
    elif present:
        print(
            f"  found {len(present)} of {len(names)} archive(s) already in "
            f"{locations.directory}; fetching the other "
            f"{len(names) - len(present)}",
            file=sys.stderr,
        )

    magic = _magic_for(source)
    archives: list[Path] = []
    for name in names:
        dest = locations.directory / name
        if not dest.is_file() and not source.automatable:
            raise DatasetNotAvailableError(
                dataset, dataset_video_dir(dataset), source,
                locations.directory.parent,
            )
        url = (
            source.file_url(name)
            if source.url_template is not None
            else source.url
        )
        assert url is not None  # guaranteed by Source.automatable
        download(
            url,
            dest,
            # Only a size actually measured on the archive itself. approx_bytes
            # is a figure to show the user and is not always the same quantity
            # -- GMDCSA24's is the sum of the repository's blobs, which is not
            # what a gzipped tarball of them weighs.
            expected_bytes=source.bytes_of(name),
            # codeload builds its tarball on the fly: no Content-Length, no
            # Range, so a resumed request would append to a fresh body.
            resume=dataset != "GMDCSA24",
            progress=progress,
            label=name,
            expect_magic=magic,
            what=f"{dataset} {name}",
        )
        archives.append(dest)
    return archives


def _extract_all(
    dataset: str,
    archives: Sequence[Path],
    locations: DownloadLocations,
    *,
    progress: bool = True,
) -> Path:
    """Extract *archives* into the component's unpacked directory.

    Args:
        dataset: The component.
        archives: Archives to extract, in order.
        locations: Where its release lives.
        progress: Whether to show a progress line.

    Returns:
        The unpacked directory.
    """
    source = get_source(dataset)
    for archive in archives:
        marker = _extract_marker(locations, archive.name)
        if marker.is_file():
            continue
        extract_archive(
            archive,
            locations.unpacked,
            fmt=source.archive_format,
            progress=progress,
        )
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(f"{archive}\n{archive.stat().st_size}\n")
    return locations.unpacked


def _unpack_given(
    dataset: str, given: Path, locations: DownloadLocations, *, progress: bool
) -> Path:
    """Interpret an ``--archive`` value and return an unpacked release.

    Three things are accepted, and which one was assumed is always printed ---
    guessing quietly between them is how a user ends up debugging a converter
    when the real mistake was one directory up.

    Args:
        dataset: The component.
        given: A single archive file, a directory of archives, or a directory
            holding the already-unpacked release.
        locations: Where the component's release would normally live.
        progress: Whether to show a progress line.

    Returns:
        The directory holding the unpacked release.

    Raises:
        FileNotFoundError: If *given* does not exist.
    """
    source = get_source(dataset)
    given = Path(given).expanduser()
    if not given.exists():
        raise FileNotFoundError(
            f"--archive names {given}, which does not exist. Pass either one "
            f"downloaded archive, a directory holding the archives, or a "
            f"directory holding the already-unpacked release."
        )

    if given.is_file():
        print(f"  using the archive {given}", file=sys.stderr)
        extract_archive(
            given, locations.unpacked, fmt=source.archive_format,
            progress=progress,
        )
        return locations.unpacked

    inside = [given / name for name in source.archive_names()]
    found = [path for path in inside if path.is_file()]
    if found:
        print(
            f"  using {len(found)} archive(s) found in {given}",
            file=sys.stderr,
        )
        for archive in found:
            extract_archive(
                archive, locations.unpacked, fmt=source.archive_format,
                progress=progress,
            )
        return locations.unpacked

    print(f"  using the unpacked release at {given}", file=sys.stderr)
    return given


def _obtain_unpacked(
    dataset: str,
    *,
    download_dir: str | Path | None = None,
    archive: str | Path | None = None,
    progress: bool = True,
) -> Path:
    """Return a directory holding *dataset*'s unpacked original release.

    The single funnel every component except up_fall goes through, whether its
    archives are downloaded now, were downloaded earlier, were placed by hand,
    or are being pointed at with ``--archive``.

    Args:
        dataset: Exact ``dataset`` column value.
        download_dir: An explicit download root.
        archive: A local archive or unpacked release to use instead of
            downloading.
        progress: Whether to show a progress line.

    Returns:
        The directory the converter should read.

    Raises:
        DatasetNotAvailableError: If nothing local is available and the source
            cannot be fetched.
    """
    source = get_source(dataset)
    locations = download_locations(dataset, download_dir)

    if archive is not None:
        return _unpack_given(dataset, Path(archive), locations, progress=progress)

    names = source.archive_names()
    have = [n for n in names if (locations.directory / n).is_file()]

    # Nothing left to extract, but a release is sitting in the reserved
    # directory: either the user put it there, or an earlier run did and the
    # archives have since been deleted to reclaim the space.
    if not have and _nonempty(locations.unpacked):
        print(
            f"  using the release already unpacked at {locations.unpacked}",
            file=sys.stderr,
        )
        return locations.unpacked

    if not names:
        # cmdfall: no archive name can be recorded, because what the authors
        # send is theirs to name. The unpacked directory above is its route.
        raise DatasetNotAvailableError(
            dataset, dataset_video_dir(dataset), source,
            locations.directory.parent,
        )

    archives = _acquire_archives(dataset, locations, progress=progress)
    return _extract_all(dataset, archives, locations, progress=progress)


def _fetch_of_syn(target: Path, *, progress: bool = True) -> None:
    """Download the OF-Syn archive from the Hub and extract it into *target*."""
    source = get_source("of-syn")
    archive = download_hub_file(source.files[0], progress=progress)
    print(f"  archive: {archive}", file=sys.stderr)
    target.mkdir(parents=True, exist_ok=True)
    extract_archive(archive, target, fmt="tar", progress=progress)


def _fetch_and_convert(
    dataset: str,
    target: Path,
    *,
    download_dir: str | Path | None = None,
    archive: str | Path | None = None,
    workers: int | None = None,
    progress: bool = True,
) -> None:
    """Obtain a component's release and convert it into *target*.

    The unpacked original is left in the download directory, because these
    archives are large and re-downloading one to retry a conversion would be
    cruel.

    Args:
        dataset: A component with a converter.
        target: OmniFall video directory to write into.
        download_dir: An explicit download root.
        archive: A local archive or unpacked release to use.
        workers: How many videos to encode at once.
        progress: Whether to show a progress line.
    """
    unpacked = _obtain_unpacked(
        dataset, download_dir=download_dir, archive=archive, progress=progress
    )
    print(f"  converting from {unpacked}", file=sys.stderr)
    _CONVERTERS[dataset](unpacked, target, workers=workers, progress=progress)


# ---------------------------------------------------------------------------
# up_fall: 1,118 Google Drive archives, discovered from the HAR-UP page
# ---------------------------------------------------------------------------

#: An anchor on the HAR-UP page pointing at one camera archive. The page links
#: optical-flow (``Camera1_OF``) and sensor bundles from the same kind of URL,
#: so the anchor *text* is what selects: exactly ``Camera1`` or ``Camera2``.
_UP_FALL_LINK = re.compile(
    r"drive\.google\.com/[^\"']*?[?&]id=([\w-]{20,})[^\"']*?\"[^>]*>"
    r"(Camera[12])</a>"
)

#: The trial each link belongs to. These markers appear in the page text ahead
#: of the anchors they own, which is what makes "nearest preceding" the rule.
_UP_FALL_TRIAL = re.compile(r"Subject(\d+)Activity(\d+)Trial(\d+)")


def up_fall_links(*, progress: bool = True) -> dict[str, str]:
    """Return the Google Drive file id of every UP-Fall camera archive.

    HAR-UP's own downloader authenticates to the Drive API through PyDrive.
    None of that is necessary: the page links each archive directly, and each
    id is served by ``uc?export=download`` without a confirm token. This reads
    the page and builds the map.

    Two things about the page decide the parsing. Its HTML is escaped twice, so
    it is unescaped twice before the anchors are legible. And an anchor does
    not name its own trial --- the trial appears as a
    ``SubjectNActivityMTrialK`` marker earlier in the document --- so each link
    is attributed to the nearest marker preceding it.

    The result is then required to equal ``labels/up_fall.csv`` exactly. A page
    that yields 1,117 links is not a page to download 1,117 archives from; it
    is a page whose shape has changed, and the difference between those two
    readings is a component that verifies as complete while missing a
    recording.

    Args:
        progress: Whether to announce the fetch on stderr.

    Returns:
        A mapping from OmniFall ``path`` value to Drive file id.

    Raises:
        RuntimeError: If the link set does not match the label file exactly, or
            if two paths claim the same id.
    """
    import requests

    source = get_source("up_fall")
    assert source.url is not None  # recorded above
    if progress:
        print(f"  reading camera links from {source.url}", file=sys.stderr)
    response = requests.get(source.url, timeout=(30, 300))
    if response.status_code != 200:
        raise RuntimeError(
            f"the HAR-UP page at {source.url} answered HTTP "
            f"{response.status_code}, so the UP-Fall download links could not "
            f"be read. Try again later, or download the archives yourself; "
            f"'omnifall sources up_fall' says where to put them."
        )

    text = html.unescape(html.unescape(response.text))
    trials = [
        (match.start(), match.group(1), match.group(2), match.group(3))
        for match in _UP_FALL_TRIAL.finditer(text)
    ]
    starts = [trial[0] for trial in trials]

    links: dict[str, str] = {}
    for match in _UP_FALL_LINK.finditer(text):
        index = bisect.bisect_right(starts, match.start()) - 1
        if index < 0:
            continue  # an anchor before any trial marker owns no trial
        _, subject, activity, trial = trials[index]
        camera = match.group(2)
        path = (
            f"Subject{subject}/Activity{activity}/Trial{trial}/"
            f"Subject{subject}Activity{activity}Trial{trial}{camera}"
        )
        previous = links.get(path)
        if previous is not None and previous != match.group(1):
            raise RuntimeError(
                f"the HAR-UP page links two different Drive ids for {path} "
                f"({previous} and {match.group(1)}). Which one the labels "
                f"refer to is not something to guess at."
            )
        links[path] = match.group(1)

    wanted = set(required_paths("up_fall"))
    missing = sorted(wanted - set(links))
    extra = sorted(set(links) - wanted)
    if missing or extra:
        raise RuntimeError(
            f"the HAR-UP page no longer yields the expected camera links: "
            f"{len(links)} found, {len(wanted)} required, {len(missing)} "
            f"missing, {len(extra)} unrecognised.\n"
            f"  missing (first 5): {missing[:5]}\n"
            f"  unrecognised (first 5): {extra[:5]}\n"
            f"The page's layout has changed, so omnifall refuses to download "
            f"whatever subset it can still see --- a partial set that verified "
            f"as complete would be worse than this error. Upgrade the omnifall "
            f"package, or download the archives yourself; 'omnifall sources "
            f"up_fall' says where to put them."
        )
    if progress:
        print(
            f"  {len(links)} camera archives linked, matching all "
            f"{len(wanted)} recordings in labels/up_fall.csv",
            file=sys.stderr,
        )
    return links


#: The form Google Drive shows instead of a large file, and its hidden fields.
_DRIVE_FORM = re.compile(
    r'<form[^>]*\bid="download-form"[^>]*\baction="([^"]+)"', re.I
)
_DRIVE_FIELD = re.compile(
    r'<input\s+type="hidden"\s+name="([^"]+)"\s+value="([^"]*)"', re.I
)


def _drive_direct_url(file_id: str, *, progress: bool = True) -> str:
    """Return a URL that serves one Drive file's bytes rather than a page.

    ``uc?id=...&export=download`` hands back the file directly only while it is
    small. Above roughly 100 MB --- which a good third of the UP-Fall archives
    are --- Drive answers with a "Virus scan warning" page instead, whose form
    carries a per-request ``uuid`` and posts to a different host. Following
    that form is the whole of the extra step; it needs no account, no API key
    and no token of ours.

    A page with no such form is a real refusal (quota exhausted, file
    withdrawn, link expired) and is reported as one rather than retried.

    Args:
        file_id: The Drive file id.
        progress: Unused; kept so callers can pass it uniformly.

    Returns:
        A URL whose body is the archive.

    Raises:
        RuntimeError: On an HTTP error, or on a page that offers no download.
    """
    import requests

    source = get_source("up_fall")
    url = source.file_url(file_id)
    with requests.get(
        url, stream=True, timeout=(30, 300), allow_redirects=True
    ) as response:
        if response.status_code != 200:
            raise RuntimeError(
                f"Google Drive answered HTTP {response.status_code} for file "
                f"id {file_id} ({url})."
            )
        kind = (response.headers.get("Content-Type") or "").split(";")[0]
        if kind.strip().lower() not in _NOT_AN_ARCHIVE:
            return url
        body = response.text

    action = _DRIVE_FORM.search(body)
    if action is None:
        raise RuntimeError(
            _challenge_message(
                f"up_fall (Drive id {file_id})", url, "a page with no download "
                "form on it"
            )
        )
    fields = dict(_DRIVE_FIELD.findall(body))
    if "id" not in fields:
        fields["id"] = file_id
    return f"{html.unescape(action.group(1))}?{urlencode(fields)}"


def _up_fall_zip_name(path: str) -> str:
    """Return the archive file name of one UP-Fall recording.

    Args:
        path: An OmniFall ``path`` value, e.g.
            ``"Subject1/Activity1/Trial1/Subject1Activity1Trial1Camera1"``.

    Returns:
        e.g. ``"Subject1Activity1Trial1Camera1.zip"`` --- the name Google Drive
        itself sends in ``content-disposition``, so a manually downloaded copy
        needs no renaming.
    """
    return f"{path.rpartition('/')[2]}.zip"


def _fetch_up_fall(
    target: Path,
    *,
    download_dir: str | Path | None = None,
    archive: str | Path | None = None,
    workers: int | None = None,
    keep_archives: bool = False,
    progress: bool = True,
) -> None:
    """Download, convert and discard UP-Fall one recording at a time.

    The whole set is of the order of 110 GB of PNG archives that become about
    4.6 hours of video, so fetching everything before converting anything would
    demand scratch space nobody has and would leave an interrupted run with
    nothing usable. Instead each recording is downloaded, encoded, and its
    archive and frames deleted, before the next one starts. A run stopped
    halfway leaves half the component ready and resumes at the recording it did
    not reach.

    Only archives this run downloaded are deleted. One the user put in the
    download directory, or pointed at with ``--archive``, is theirs and is left
    alone.

    Args:
        target: OmniFall video directory for ``up_fall``.
        download_dir: An explicit download root.
        archive: A directory holding the camera archives, or one holding the
            already-unpacked release.
        workers: How many recordings to handle at once. Each occupies about
            100 MB of scratch space while it runs.
        keep_archives: Keep the downloaded archives and extracted frames
            instead of deleting them. Budget the full 110 GB.
        progress: Whether to show a progress line.

    Raises:
        FileNotFoundError: If *archive* does not exist.
        RuntimeError: If an archive is missing and cannot be fetched.
    """
    locations = download_locations("up_fall", download_dir)
    zip_dir = locations.directory
    ours = True

    if archive is not None:
        given = Path(archive).expanduser()
        if not given.exists():
            raise FileNotFoundError(
                f"--archive names {given}, which does not exist."
            )
        if given.is_file():
            raise RuntimeError(
                f"up_fall is served as {locations.count} separate archives, "
                f"so --archive has to name a directory holding them (or the "
                f"already-unpacked release), not the single file {given}."
            )
        if any(given.glob("Subject*Camera*.zip")):
            zip_dir = given
            ours = False
            print(f"  taking camera archives from {given}", file=sys.stderr)
        else:
            print(f"  using the unpacked release at {given}", file=sys.stderr)
            _convert_up_fall(given, target, workers=workers, progress=progress)
            return

    require_ffmpeg()
    source = get_source("up_fall")
    paths = required_paths("up_fall")
    todo = [
        path
        for path in paths
        if not _already_encoded(target / f"{path}.mp4")
    ]
    if not todo:
        print(
            f"  all {len(paths)} recordings are already encoded",
            file=sys.stderr,
        )
        return
    print(
        f"  {len(paths) - len(todo)} of {len(paths)} recordings already "
        f"encoded; {len(todo)} to go",
        file=sys.stderr,
    )

    def have_locally(path: str) -> bool:
        """Report whether this recording needs nothing downloaded."""
        frames = locations.unpacked / path
        if frames.is_dir() and any(frames.glob("*.png")):
            return True
        return (zip_dir / _up_fall_zip_name(path)).is_file()

    # Only read the page when something actually has to be downloaded, so a
    # fully pre-populated download directory needs no network at all.
    links: dict[str, str] = {}
    have = sum(1 for path in todo if have_locally(path))
    if have:
        print(
            f"  {have} of those are already downloaded or unpacked locally",
            file=sys.stderr,
        )
    if have < len(todo):
        if not source.automatable:  # pragma: no cover - recorded as automatable
            raise DatasetNotAvailableError(
                "up_fall", target, source, download_dir
            )
        links = up_fall_links(progress=progress)

    def handle(path: str) -> None:
        """Download, extract, encode and clean up one recording."""
        # The archives hold their PNGs flat at the root, with no directory
        # prefix, so each is extracted into the directory named after the
        # recording it belongs to -- which is what the converter reads, and
        # which is also where a user who unpacked the release themselves would
        # have put them. Frames already there mean nothing has to be fetched.
        frames = locations.unpacked / path
        if frames.is_dir() and any(frames.glob("*.png")):
            _up_fall_encoder(path, frames, target / f"{path}.mp4")()
            return

        zip_path = zip_dir / _up_fall_zip_name(path)
        fetched = False
        if not zip_path.is_file():
            file_id = links.get(path)
            if file_id is None:
                raise RuntimeError(
                    f"up_fall {path}: no archive at {zip_path} and no download "
                    f"link for it. Run 'omnifall sources up_fall' for the file "
                    f"name and the directory to put it in."
                )
            download(
                _drive_direct_url(file_id),
                zip_path,
                progress=False,
                verbose=False,
                expect_magic=_ARCHIVE_MAGIC["zip"],
                what=f"up_fall {zip_path.name}",
            )
            fetched = True

        extract_archive(zip_path, frames, fmt="zip", progress=False)
        try:
            _up_fall_encoder(path, frames, target / f"{path}.mp4")()
        finally:
            if not keep_archives:
                shutil.rmtree(frames, ignore_errors=True)
                # Only what this run downloaded into omnifall's own directory.
                if fetched and ours:
                    zip_path.unlink(missing_ok=True)

    pool_size = _default_workers() if workers is None else max(1, int(workers))
    (target).mkdir(parents=True, exist_ok=True)
    with _Progress(
        "preparing up_fall", len(todo), unit="n", enabled=progress
    ) as bar:
        pool = ThreadPoolExecutor(max_workers=pool_size)
        try:
            futures = {pool.submit(handle, path): path for path in todo}
            for future in as_completed(futures):
                future.result()
                bar.advance(1)
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            pool.shutdown(wait=True)


def _already_encoded(out: Path) -> bool:
    """Report whether *out* is a video an earlier run finished writing."""
    return out.is_file() and out.stat().st_size > 0


def prepare_oops(
    output_dir: str | Path | None = None,
    oops_archive: str | Path | None = None,
    force: bool = False,
    consent: bool = False,
) -> Path:
    """Prepare the OOPS videos used by OF-ItW.

    Kept as a module-level alias so that ``from omnifall import prepare_oops``
    keeps working; the implementation lives in :mod:`omnifall._oops`.

    Args:
        output_dir: Where to place the prepared videos.
        oops_archive: A local copy of ``video_and_anns.tar.gz`` to read instead
            of streaming from the web.
        force: Re-extract videos that already exist.
        consent: Skip the interactive licence prompt.

    Returns:
        The directory holding the prepared videos.
    """
    from ._oops import prepare_oops as _impl

    return _impl(
        output_dir=output_dir,
        oops_archive=oops_archive,
        force=force,
        consent=consent,
    )


# ---------------------------------------------------------------------------
# The public preparation API
# ---------------------------------------------------------------------------


def ensure_dataset(
    dataset: str,
    *,
    consent: bool = False,
    video_dir: Path | None = None,
    workers: int | None = None,
    force: bool = False,
    archive: str | Path | None = None,
    download_dir: str | Path | None = None,
    keep_archives: bool = False,
) -> Path:
    """Make sure *dataset*'s videos are on disk, and return their directory.

    This is the function video loading calls. It returns immediately when the
    videos are already there, and otherwise either fetches them or raises an
    error that says exactly what the user has to do.

    "Already there" is the cheap check from :mod:`omnifall._cache`: a directory
    holding at least one video of the right extension. That is deliberately
    weak, and it is why *force* exists --- a run interrupted halfway through an
    extraction leaves exactly such a directory, and without *force* every later
    call would take the shortcut and report success on a fraction of the data.
    With *force* the shortcut is skipped, the component is prepared again, and
    the result is checked against the Hub label file before this returns.

    Args:
        dataset: Exact ``dataset`` column value.
        consent: Treat licence prompts as accepted. Required for unattended
            use of sources that carry a notice, such as OOPS.
        video_dir: Directory override, resolved by
            :func:`omnifall._cache.dataset_video_dir`.
        workers: How many videos to encode at once where a conversion is
            needed. Defaults to eight, or the number of CPUs where that is
            smaller.
        force: Prepare again even when videos are already present, and fail if
            the result is still incomplete. This repairs a partial tree; it
            does not re-encode videos that are already there, for which
            :func:`convert` takes ``overwrite=True``.
        archive: A local archive, a directory of archives, or an
            already-unpacked release to use instead of downloading. This is the
            escape hatch for every component: whatever the source demands of a
            browser, once the file is on disk this takes it.
        download_dir: An explicit download root, replacing
            ``OMNIFALL_DOWNLOAD_DIR`` and the cache default. Archives are
            fetched into it, and archives already in it are used as they
            stand.
        keep_archives: For up_fall only, whose archives are deleted as they are
            converted: keep them instead.

    Returns:
        The directory *dataset*'s ``path`` values are relative to.

    Raises:
        KeyError: If *dataset* is not an OmniFall component.
        DatasetNotAvailableError: If the videos need a browser, a form or an
            e-mail request and none have been placed locally. The message
            carries the instructions and the exact target directory.
        ConversionNotImplementedError: If the download is automated but the
            reshaping of its tree is not written yet.
    """
    source = get_source(dataset)
    target = dataset_video_dir(dataset, override=video_dir)

    if not force and is_dataset_prepared(dataset, override=video_dir):
        return target

    # A source that cannot be fetched can still be prepared, as long as the
    # user has supplied the data. Only refuse when there is nothing to work
    # from -- which _obtain_unpacked decides, because it is what looks.
    if not source.automatable and archive is None:
        locations = download_locations(dataset, download_dir)
        if not _nonempty(locations.unpacked) and not any(
            path.is_file() for path in locations.paths()
        ):
            raise DatasetNotAvailableError(
                dataset, target, source, download_dir
            )

    if dataset not in IMPLEMENTED_CONVERSIONS and dataset not in _CONVERTERS:
        # Refuse before spending bandwidth: fetching 16 GB only to fail at the
        # conversion step would be worse than saying so up front.
        _CONVERTERS[dataset](Path("."), target)

    if archive is not None:
        scale = f"from {archive}"
    elif source.approx_bytes is None:
        scale = "size unknown"
    else:
        scale = f"{_human(source.approx_bytes)} to download"
    print(f"Preparing {dataset} ({scale}) into {target}", file=sys.stderr)
    # of-syn and OOPS are fetched by routes of their own rather than through
    # _obtain_unpacked, so the download directory is looked at here instead.
    if archive is None and dataset in _ARCHIVE_ONLY:
        archive = local_archive(dataset, download_dir)
        if archive is not None:
            print(
                f"  found {archive} already downloaded; not fetching it again",
                file=sys.stderr,
            )

    if dataset == "of-syn" and archive is None:
        _fetch_of_syn(target, progress=True)
    elif dataset == "of-syn":
        # The tar's members are already "./{label}/{stem}.mp4", so it unpacks
        # straight into the video directory; there is no tree to reshape and no
        # unpacked/ stage to route it through.
        target.mkdir(parents=True, exist_ok=True)
        extract_archive(Path(archive), target, fmt="tar", progress=True)
    elif dataset == "OOPS" and archive is None:
        prepare_oops(output_dir=target, consent=consent)
    elif dataset == "OOPS":
        prepare_oops(output_dir=target, oops_archive=archive, consent=consent)
    elif dataset == "up_fall":
        _fetch_up_fall(
            target,
            download_dir=download_dir,
            archive=archive,
            workers=workers,
            keep_archives=keep_archives,
            progress=True,
        )
    else:
        _fetch_and_convert(
            dataset,
            target,
            download_dir=download_dir,
            archive=archive,
            workers=workers,
            progress=True,
        )

    report = verify(dataset, video_dir=video_dir)
    if not report.complete:
        raise RuntimeError(
            f"{dataset} was prepared but is incomplete:\n{report.render()}"
        )
    return target


def prepare(
    dataset: str,
    *,
    consent: bool = False,
    video_dir: str | Path | None = None,
    force: bool = False,
    download_only: bool = False,
    archive: str | Path | None = None,
    download_dir: str | Path | None = None,
    workers: int | None = None,
    keep_archives: bool = False,
) -> Path:
    """Obtain *dataset*'s videos, with a licence prompt where one is due.

    The user-facing counterpart to :func:`ensure_dataset`: it prints what it is
    about to do and asks for confirmation on sources whose licence requires it,
    unless *consent* is set.

    Args:
        dataset: Exact ``dataset`` column value.
        consent: Skip interactive licence prompts.
        video_dir: Directory override.
        force: Prepare again even if videos are already present, and fail if
            the result is still incomplete. This is the way to repair a tree
            left half-finished by an interrupted run: without it, a directory
            holding a single video counts as prepared and nothing is fetched.
        download_only: Fetch and unpack the original release into the download
            directory without reshaping it.
        archive: A local archive, a directory of archives, or an
            already-unpacked release to use instead of downloading.
        download_dir: An explicit download root, replacing
            ``OMNIFALL_DOWNLOAD_DIR`` and the cache default.
        workers: How many videos to encode at once.
        keep_archives: For up_fall, whose archives are deleted as they are
            converted: keep them instead.

    Returns:
        The dataset's video directory, or --- with *download_only* --- the
        directory the original release was unpacked into.

    Raises:
        DatasetNotAvailableError: If no automated source exists and nothing has
            been supplied locally.
        ConversionNotImplementedError: If the tree cannot be reshaped yet.
    """
    source = get_source(dataset)
    target = dataset_video_dir(dataset, override=video_dir)

    if download_only:
        if dataset == "up_fall":
            raise RuntimeError(
                "up_fall cannot be downloaded without converting: it is "
                f"{download_locations('up_fall', download_dir).count} archives "
                "of PNG frames, about 110 GB, and omnifall keeps the run "
                "within reasonable scratch space precisely by converting and "
                "discarding each one as it goes. Run 'omnifall prepare "
                "up_fall --keep-archives' if you want the archives kept."
            )
        if not source.automatable and archive is None:
            raise DatasetNotAvailableError(
                dataset, target, source, download_dir
            )
        return _obtain_unpacked(
            dataset, download_dir=download_dir, archive=archive
        )

    if not force and is_dataset_prepared(dataset, override=video_dir):
        print(f"{dataset}: already prepared at {target}")
        return target

    return ensure_dataset(
        dataset,
        consent=consent,
        video_dir=video_dir,
        force=force,
        archive=archive,
        download_dir=download_dir,
        workers=workers,
        keep_archives=keep_archives,
    )


def prepare_all(
    datasets: Iterable[str] | None = None,
    *,
    consent: bool = False,
    video_dir: str | Path | None = None,
    force: bool = False,
    download_dir: str | Path | None = None,
    workers: int | None = None,
    keep_archives: bool = False,
) -> dict[str, str]:
    """Prepare several components, carrying on past the ones that cannot be.

    Unlike :func:`prepare`, a component that needs a manual download does not
    abort the run --- the point of this function is to get everything that
    *can* be obtained, then report the rest in one place.

    Args:
        datasets: Components to prepare; all of them when omitted.
        consent: Skip interactive licence prompts.
        video_dir: Directory override.
        force: Prepare again even where videos are already present.
        download_dir: An explicit download root.
        workers: How many videos to encode at once.
        keep_archives: For up_fall: keep its archives instead of deleting them
            as they are converted.

    Returns:
        A mapping from component name to a one-line outcome: ``"ok"``,
        ``"already prepared"``, or the reason it was skipped.
    """
    names = list(datasets) if datasets is not None else list(SOURCES)
    results: dict[str, str] = {}
    for name in names:
        try:
            if not force and is_dataset_prepared(name, override=video_dir):
                results[name] = "already prepared"
                continue
            prepare(
                name,
                consent=consent,
                video_dir=video_dir,
                force=force,
                download_dir=download_dir,
                workers=workers,
                keep_archives=keep_archives,
            )
            results[name] = "ok"
        except DatasetNotAvailableError:
            results[name] = "manual download required"
        except ConversionNotImplementedError:
            results[name] = "conversion not implemented"
        except Exception as error:  # noqa: BLE001 - reported, not swallowed
            results[name] = f"failed: {error.__class__.__name__}: {error}"
    return results
