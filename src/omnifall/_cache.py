"""Cache directory management for omnifall."""

from __future__ import annotations

import os
from pathlib import Path

from ._constants import EXPECTED_OOPS_COUNT, EXPECTED_SYN_COUNT


def get_omnifall_root() -> Path | None:
    """Return the OMNIFALL_ROOT path if set, else None.

    When OMNIFALL_ROOT is set, all video paths are resolved locally
    as ``{OMNIFALL_ROOT}/{dataset}/video/{path}.mp4`` and no downloads
    are performed.
    """
    env = os.environ.get("OMNIFALL_ROOT")
    if env:
        return Path(env)
    return None


def get_cache_dir() -> Path:
    """Return the omnifall cache directory.

    Respects the OMNIFALL_CACHE_DIR environment variable.
    Defaults to ~/.cache/omnifall.
    """
    env = os.environ.get("OMNIFALL_CACHE_DIR")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "omnifall"


def get_syn_video_dir() -> Path:
    """Return the directory for extracted OF-Syn videos."""
    return get_cache_dir() / "of-syn-videos"


def get_oops_video_dir() -> Path:
    """Return the directory for prepared OOPS videos."""
    return get_cache_dir() / "oops_prepared"


def is_oops_prepared(oops_dir: Path | None = None) -> bool:
    """Check whether OOPS videos have been prepared.

    Verifies that the falls/ subdirectory contains at least EXPECTED_OOPS_COUNT
    .mp4 files.
    """
    if oops_dir is None:
        oops_dir = get_oops_video_dir()
    falls_dir = oops_dir / "falls"
    if not falls_dir.is_dir():
        return False
    mp4_count = sum(1 for f in falls_dir.iterdir() if f.suffix == ".mp4")
    return mp4_count >= EXPECTED_OOPS_COUNT


def is_syn_extracted(syn_dir: Path | None = None) -> bool:
    """Check whether OF-Syn videos have been fully extracted.

    Checks that the directory contains at least EXPECTED_SYN_COUNT .mp4 files.
    """
    if syn_dir is None:
        syn_dir = get_syn_video_dir()
    if not syn_dir.is_dir():
        return False
    mp4_count = sum(1 for _ in syn_dir.rglob("*.mp4"))
    return mp4_count >= EXPECTED_SYN_COUNT
