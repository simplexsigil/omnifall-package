"""Where the video files of a component dataset may live.

Video files are addressed by the pair ``(dataset, path)``. This module answers
only the first half of that: given a ``dataset`` column value, which directory
are its ``path`` values relative to?

Four layers are consulted, most specific first:

1. an explicit directory passed by the caller,
2. ``OMNIFALL_VIDEO_ROOT__<dataset>`` --- a per-dataset environment variable
   using the exact spelling of the ``dataset`` column
   (e.g. ``OMNIFALL_VIDEO_ROOT__GMDCSA24``),
3. ``OMNIFALL_ROOT`` --- a shared root laid out as
   ``{root}/{dataset}/video/{path}.mp4``,
4. the package cache --- ``{cache}/videos/{dataset}/video/{path}.mp4``.

Layers 1 and 2 name a single dataset's video directory directly: the value is
the directory its ``path`` values are relative to, so
``$OMNIFALL_VIDEO_ROOT__le2i/Coffee_room_01/video_1.mp4`` has to be a file. It
is never reinterpreted as a parent of that directory. Because these are
explicit user statements, a value that points nowhere is an error rather than
something to silently skip, and a value that points one level too high is
named as such in the resolution report instead of being silently corrected.

Layers 3 and 4 are general locations that may legitimately hold only a subset
of the ten component datasets, so a dataset missing from ``OMNIFALL_ROOT``
falls through to the cache.
"""

from __future__ import annotations

import os
from pathlib import Path

from ._constants import (
    DATASETS,
    ENV_CACHE,
    ENV_PER_DATASET_PREFIX,
    ENV_ROOT,
    EXPECTED_OOPS_COUNT,
    EXPECTED_SYN_COUNT,
)

__all__ = [
    "get_cache_dir",
    "get_download_dir",
    "dataset_download_dir",
    "ENV_DOWNLOAD_DIR",
    "get_omnifall_root",
    "per_dataset_env_var",
    "dataset_video_dir",
    "dataset_video_dir_candidates",
    "dataset_video_dir_with_layer",
    "dir_holds_videos",
    "is_dataset_prepared",
    "video_ext",
    "get_syn_video_dir",
    "get_oops_video_dir",
    "is_oops_prepared",
    "is_syn_extracted",
]


# ---------------------------------------------------------------------------
# Base locations
# ---------------------------------------------------------------------------


def get_cache_dir() -> Path:
    """Return the package cache directory.

    Honours :data:`omnifall._constants.ENV_CACHE` (``OMNIFALL_CACHE_DIR``) and
    otherwise defaults to ``~/.cache/omnifall``.

    Returns:
        The cache directory. It is not created here.
    """
    env = os.environ.get(ENV_CACHE)
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "omnifall"


#: Names a directory for the original releases, overriding ``{cache}/downloads``.
#:
#: Kept here rather than in :mod:`omnifall._constants` because it names a
#: location, which is what this module is about. It is the one place both
#: halves of the "bring your own download" story meet: everything omnifall
#: fetches is written here, and everything a user fetches by hand is looked for
#: here, under the same names.
ENV_DOWNLOAD_DIR = "OMNIFALL_DOWNLOAD_DIR"


def get_download_dir(override: str | Path | None = None) -> Path:
    """Return the directory original releases are downloaded to and read from.

    This is deliberately one known place rather than a private temporary
    directory. A user who cannot let omnifall reach a source --- because it
    needs a browser, or a login, or because the machine has no network --- can
    put the archive here by hand under the name
    :func:`omnifall._sources.Source.archive_names` gives, and the ordinary
    ``omnifall prepare`` then finds it instead of downloading.

    Args:
        override: An explicit directory, which wins over everything else.

    Returns:
        The download directory. It is not created here.
    """
    if override is not None:
        return Path(override).expanduser()
    env = os.environ.get(ENV_DOWNLOAD_DIR)
    if env:
        return Path(env).expanduser()
    return get_cache_dir() / "downloads"


def dataset_download_dir(
    dataset: str, override: str | Path | None = None
) -> Path:
    """Return the download directory of one component.

    Args:
        dataset: Exact ``dataset`` column value, e.g. ``"mcfd"``.
        override: An explicit download root, replacing the resolved one.

    Returns:
        ``{download_dir}/{dataset}``. It is not created here.
    """
    return get_download_dir(override) / dataset


def get_omnifall_root() -> Path | None:
    """Return the shared video root, or ``None`` when it is not configured.

    Returns:
        The value of ``OMNIFALL_ROOT`` as a path, or ``None``.

    Raises:
        FileNotFoundError: If ``OMNIFALL_ROOT`` is set but is not a directory.
            A misspelled root is a user error worth reporting immediately ---
            silently ignoring it would send every lookup to an empty cache.
    """
    env = os.environ.get(ENV_ROOT)
    if not env:
        return None
    root = Path(env).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(
            f"{ENV_ROOT} is set to {env!r}, which is not an existing directory. "
            f"It must point at a tree laid out as "
            f"{{{ENV_ROOT}}}/{{dataset}}/video/{{path}}.mp4. "
            f"Unset {ENV_ROOT} to fall back to the package cache "
            f"({get_cache_dir()})."
        )
    return root


def per_dataset_env_var(dataset: str) -> str:
    """Return the name of the per-dataset override variable for *dataset*.

    Args:
        dataset: Exact ``dataset`` column value, e.g. ``"GMDCSA24"``.

    Returns:
        e.g. ``"OMNIFALL_VIDEO_ROOT__GMDCSA24"``.
    """
    return f"{ENV_PER_DATASET_PREFIX}{dataset}"


# ---------------------------------------------------------------------------
# Layered resolution
# ---------------------------------------------------------------------------


def _explicit_dir(label: str, value: str | Path, dataset: str, hint: str) -> Path:
    """Validate an explicitly named video directory.

    An explicit location is taken literally: it *is* the directory ``path``
    values are relative to, so ``{value}/{path}{ext}`` has to be a file. It is
    never reinterpreted as a parent of that directory --- a wrong value fails
    loudly here, and :mod:`omnifall._resolve` recognises the common
    "one level too high" mistake from the real path data and names the right
    directory in its report.

    Args:
        label: Layer name used in messages, e.g. ``"argument"``.
        value: The directory the user named.
        dataset: Exact ``dataset`` column value, for the extension and message.
        hint: How to undo the setting, appended to error messages.

    Returns:
        *value* as a path.

    Raises:
        FileNotFoundError: If *value* is not an existing directory.
    """
    base = Path(value).expanduser()
    if not base.is_dir():
        nested = base / "video"
        extra = f" Did you mean {nested}?" if nested.is_dir() else ""
        raise FileNotFoundError(
            f"{label} names {str(value)!r} as the video directory of "
            f"{dataset}, but that directory does not exist. It must be the "
            f"directory {dataset}'s 'path' values are relative to, so that "
            f"{base}/{{path}}{_ext(dataset)} is a file.{extra} {hint}"
        )
    return base


def dataset_video_dir_candidates(
    dataset: str,
    *,
    override: str | Path | None = None,
) -> list[tuple[str, Path]]:
    """Return every directory *dataset*'s ``path`` values could be relative to.

    Args:
        dataset: Exact ``dataset`` column value, e.g. ``"cmdfall"``.
        override: Directory supplied by the caller, taking precedence over all
            environment variables.

    Returns:
        ``(layer, directory)`` pairs, most specific first. The layer name is
        one of ``"argument"``, ``"env:<VAR>"``, ``"OMNIFALL_ROOT"`` or
        ``"cache"``. The last entry is always the cache layer, so the list is
        never empty.

    Raises:
        FileNotFoundError: If *override* or the per-dataset environment
            variable names a directory that does not exist, or if
            ``OMNIFALL_ROOT`` is set but is not a directory.
    """
    candidates: list[tuple[str, Path]] = []

    if override is not None:
        candidates.append(
            (
                "argument",
                _explicit_dir(
                    "argument video_dirs",
                    override,
                    dataset,
                    "Pass a directory that exists, or omit it to fall back to "
                    f"{ENV_ROOT} or the cache.",
                ),
            )
        )

    env_var = per_dataset_env_var(dataset)
    env_value = os.environ.get(env_var)
    if env_value:
        candidates.append(
            (
                f"env:{env_var}",
                _explicit_dir(
                    f"Environment variable {env_var}",
                    env_value,
                    dataset,
                    f"Unset {env_var} to fall back to {ENV_ROOT} or the cache.",
                ),
            )
        )

    root = get_omnifall_root()
    if root is not None:
        candidates.append((ENV_ROOT, root / dataset / "video"))

    candidates.append(("cache", get_cache_dir() / "videos" / dataset / "video"))
    return candidates


def dataset_video_dir(
    dataset: str,
    *,
    override: str | Path | None = None,
) -> Path:
    """Return the directory *dataset*'s ``path`` values are relative to.

    Joining ``f"{path}{video_ext}"`` onto the result yields the video file.

    Args:
        dataset: Exact ``dataset`` column value.
        override: Directory supplied by the caller.

    Returns:
        The first candidate directory that exists. When none exists the cache
        candidate is returned, so callers can report a concrete "not prepared"
        location instead of guessing.

    Raises:
        FileNotFoundError: Propagated from
            :func:`dataset_video_dir_candidates` for explicit locations that do
            not exist.
    """
    return dataset_video_dir_with_layer(dataset, override=override)[1]


def dataset_video_dir_with_layer(
    dataset: str,
    *,
    override: str | Path | None = None,
) -> tuple[str, Path]:
    """Like :func:`dataset_video_dir`, but also return which layer won.

    Args:
        dataset: Exact ``dataset`` column value.
        override: Directory supplied by the caller.

    Returns:
        A ``(layer, directory)`` pair.

    Note:
        An **explicitly named** location -- the ``override`` argument or the
        per-dataset environment variable -- always wins, empty or not. The
        caller said where to look, and quietly resolving somewhere else is
        unsafe: this function also names the destination that preparation writes
        into, so overriding an explicit choice can send writes at a directory
        the caller never mentioned, such as a shared read-only master copy.

        Among the *implicit* locations -- ``OMNIFALL_ROOT`` and the cache -- a
        directory holding at least one video beats one that merely exists. An
        interrupted preparation leaves an empty ``{root}/{dataset}/video``
        behind, and picking it would shadow a fully populated cache: the videos
        are there, the loader reports them missing, and nothing says why.
    """
    candidates = dataset_video_dir_candidates(dataset, override=override)

    explicit = [c for c in candidates if c[0] == "argument" or c[0].startswith("env:")]
    if explicit:
        return explicit[0]

    for layer, directory in candidates:
        if _holds_a_video(directory, dataset):
            return layer, directory
    for layer, directory in candidates:
        if directory.is_dir():
            return layer, directory
    return candidates[-1]


def _holds_a_video(directory: Path, dataset: str) -> bool:
    """Whether *directory* contains at least one file of *dataset*'s type.

    Short-circuits on the first hit rather than walking the tree — cmdfall has
    over a thousand files and this runs on every resolution.
    """
    if not directory.is_dir():
        return False
    suffix = video_ext(dataset)
    for path in directory.rglob(f"*{suffix}"):
        if path.is_file():
            return True
    return False


# ---------------------------------------------------------------------------
# Preparation checks
# ---------------------------------------------------------------------------


def video_ext(dataset: str) -> str:
    """Return the video file extension used by *dataset*.

    Args:
        dataset: Exact ``dataset`` column value.

    Returns:
        The extension, including the leading dot.

    Raises:
        KeyError: If *dataset* is not a component dataset of OmniFall.
    """
    try:
        return DATASETS[dataset].video_ext
    except KeyError:
        raise KeyError(
            f"{dataset!r} is not a component dataset known to this build of "
            f"omnifall. Known datasets: {', '.join(sorted(DATASETS))}. "
            f"If the Hub repository has gained a new component dataset, "
            f"upgrade the omnifall package."
        ) from None


def _ext(dataset: str) -> str:
    """Return *dataset*'s extension, defaulting to ``.mp4`` for error messages.

    Used only where an unknown name must not derail the construction of a
    diagnostic; resolution itself goes through :func:`video_ext`.
    """
    info = DATASETS.get(dataset)
    return info.video_ext if info is not None else ".mp4"


def dir_holds_videos(directory: Path, dataset: str) -> bool:
    """Report whether *directory* contains at least one video of *dataset*.

    The search is recursive but stops at the first hit, so it stays cheap on
    directories holding tens of thousands of files.

    Args:
        directory: Directory to inspect.
        dataset: Exact ``dataset`` column value, used for the extension.

    Returns:
        ``True`` if the directory exists and holds a matching file.
    """
    if not directory.is_dir():
        return False
    return next(directory.rglob(f"*{_ext(dataset)}"), None) is not None


def is_dataset_prepared(
    dataset: str,
    *,
    override: str | Path | None = None,
) -> bool:
    """Report whether *dataset*'s video directory holds any videos at all.

    This is a cheap heuristic, not a completeness check: the search stops at
    the first matching file rather than walking tens of thousands of entries.
    Use the per-row resolution in :mod:`omnifall._resolve` when it matters
    whether a *specific* video is present.

    Args:
        dataset: Exact ``dataset`` column value.
        override: Directory supplied by the caller.

    Returns:
        ``True`` if the resolved directory exists and contains at least one
        file with the dataset's video extension.
    """
    return dir_holds_videos(dataset_video_dir(dataset, override=override), dataset)


def _count_videos(directory: Path, ext: str, *, stop_at: int) -> int:
    """Count video files under *directory*, giving up once *stop_at* is hit."""
    if not directory.is_dir():
        return 0
    count = 0
    for _ in directory.rglob(f"*{ext}"):
        count += 1
        if count >= stop_at:
            break
    return count


# ---------------------------------------------------------------------------
# Backwards-compatible wrappers used by the preparation code
# ---------------------------------------------------------------------------


def get_syn_video_dir(override: str | Path | None = None) -> Path:
    """Return the directory holding the extracted OF-Syn videos."""
    return dataset_video_dir("of-syn", override=override)


def get_oops_video_dir(override: str | Path | None = None) -> Path:
    """Return the directory holding the prepared OF-ItW (OOPS) videos."""
    return dataset_video_dir("OOPS", override=override)


def is_oops_prepared(oops_dir: str | Path | None = None) -> bool:
    """Report whether *all* expected OOPS videos have been prepared.

    Stricter than :func:`is_dataset_prepared`: preparation code uses this to
    decide whether an interrupted extraction has to be resumed, so a partially
    filled directory must count as not prepared.

    Args:
        oops_dir: Directory to inspect. Defaults to the resolved OOPS
            video directory.

    Returns:
        ``True`` if at least :data:`EXPECTED_OOPS_COUNT` ``.mp4`` files are
        present.
    """
    directory = get_oops_video_dir(oops_dir)
    return (
        _count_videos(directory, _ext("OOPS"), stop_at=EXPECTED_OOPS_COUNT)
        >= EXPECTED_OOPS_COUNT
    )


def is_syn_extracted(syn_dir: str | Path | None = None) -> bool:
    """Report whether *all* expected OF-Syn videos have been extracted.

    Args:
        syn_dir: Directory to inspect. Defaults to the resolved OF-Syn
            video directory.

    Returns:
        ``True`` if at least :data:`EXPECTED_SYN_COUNT` ``.mp4`` files are
        present.
    """
    directory = get_syn_video_dir(syn_dir)
    return (
        _count_videos(directory, _ext("of-syn"), stop_at=EXPECTED_SYN_COUNT)
        >= EXPECTED_SYN_COUNT
    )
