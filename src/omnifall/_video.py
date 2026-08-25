"""Video download, extraction, and path resolution for OmniFall."""

from __future__ import annotations

import tarfile
from pathlib import Path

from datasets import Dataset, DatasetDict
from huggingface_hub import hf_hub_download

from ._cache import (
    get_omnifall_root,
    get_oops_video_dir,
    get_syn_video_dir,
    is_oops_prepared,
    is_syn_extracted,
)
from ._constants import (
    ALL_VIDEO_CONFIGS,
    DOWNLOADABLE_VIDEO_CONFIGS,
    HF_REPO_ID,
    NO_VIDEO_CONFIGS,
    OOPS_VIDEO_CONFIGS,
    STAGED_ONLY_CONFIGS,
    SYN_VIDEO_ARCHIVE,
    SYN_VIDEO_CONFIGS,
    TO_ALL_STAGED_SYN_TRAIN_CONFIGS,
    TO_ALL_STAGED_TRAIN_CONFIGS,
    TO_ALL_SYN_TRAIN_CONFIGS,
)


def download_and_extract_syn(syn_dir: Path | None = None) -> Path:
    """Download and extract OF-Syn videos from HF Hub.

    Args:
        syn_dir: Directory to extract videos into. Defaults to the cache dir.

    Returns:
        Path to the directory containing extracted video files.
    """
    if syn_dir is None:
        syn_dir = get_syn_video_dir()

    if is_syn_extracted(syn_dir):
        return syn_dir

    print("Downloading OF-Syn video archive from HF Hub (~9.1GB)...")
    archive_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=SYN_VIDEO_ARCHIVE,
        repo_type="dataset",
    )

    print(f"Extracting to {syn_dir}...")
    syn_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r") as tar:
        tar.extractall(path=syn_dir)

    print(f"OF-Syn videos extracted to: {syn_dir}")
    return syn_dir


# ---------------------------------------------------------------------------
# OMNIFALL_ROOT path resolution (local data, no downloads)
# ---------------------------------------------------------------------------


def _resolve_root_video_path(
    path: str, dataset: str, omnifall_root: Path
) -> str:
    """Resolve a video path using OMNIFALL_ROOT.

    Layout: ``{OMNIFALL_ROOT}/{dataset}/video/{path}.mp4``
    """
    video_path = omnifall_root / dataset / "video" / f"{path}.mp4"
    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found at {video_path}. "
            f"OMNIFALL_ROOT is set to '{omnifall_root}'. "
            f"Expected layout: {{OMNIFALL_ROOT}}/{{dataset}}/video/{{path}}.mp4"
        )
    return str(video_path)


def _add_video_column_root(
    ds: Dataset, omnifall_root: Path
) -> Dataset:
    """Add video column using OMNIFALL_ROOT for all rows."""
    paths = ds["path"]
    datasets = ds["dataset"]
    video_paths = [
        _resolve_root_video_path(p, d, omnifall_root)
        for p, d in zip(paths, datasets)
    ]
    return ds.add_column("video", video_paths)


# ---------------------------------------------------------------------------
# Download-based path resolution (cache mode)
# ---------------------------------------------------------------------------


def _resolve_syn_video_path(path: str, syn_dir: Path) -> str:
    """Resolve an OF-Syn video path to an absolute file path.

    The path column is like ``lie_down/lie_down_el_098`` (no extension).
    """
    return str(syn_dir / f"{path}.mp4")


def _resolve_oops_video_path(path: str, oops_dir: Path) -> str:
    """Resolve an OF-ItW video path to an absolute file path.

    The path column is like ``falls/xyz`` (no extension).
    Prepared OOPS videos are stored as ``oops_dir/falls/xyz.mp4``.
    """
    return str(oops_dir / f"{path}.mp4")


def _needs_syn_videos(config: str, split: str | None) -> bool:
    """Check if the given config/split needs OF-Syn videos."""
    if config in SYN_VIDEO_CONFIGS:
        return True
    # to-all: syn+staged_syn train configs have syn in ALL splits
    if config in (TO_ALL_SYN_TRAIN_CONFIGS | TO_ALL_STAGED_SYN_TRAIN_CONFIGS):
        return True  # syn in train/val AND test
    # to-all: staged-only train configs have syn in test only
    if config in TO_ALL_STAGED_TRAIN_CONFIGS:
        if split is None:
            return True  # DatasetDict: test has syn
        return split == "test"
    return False


def _needs_oops_videos(config: str, split: str | None) -> bool:
    """Check if the given config/split needs OOPS videos."""
    if config in OOPS_VIDEO_CONFIGS:
        return True
    # All to-all configs need OOPS for test
    if config in (TO_ALL_SYN_TRAIN_CONFIGS | TO_ALL_STAGED_TRAIN_CONFIGS
                  | TO_ALL_STAGED_SYN_TRAIN_CONFIGS):
        if split is None:
            return True
        return split == "test"
    return False


def _add_video_column_download(
    ds: Dataset,
    config: str,
    split_name: str,
    syn_dir: Path | None,
    oops_dir: Path | None,
) -> Dataset:
    """Add a 'video' column using download-based resolution for a single split.

    Staged dataset rows (not OOPS or of-syn) get None as their video path
    since staged videos are not available for download. Use OMNIFALL_ROOT
    for full video support including staged datasets.
    """
    need_syn = _needs_syn_videos(config, split_name)
    need_oops = _needs_oops_videos(config, split_name)

    if not need_syn and not need_oops:
        return ds

    paths = ds["path"]
    dataset_col = ds["dataset"] if "dataset" in ds.column_names else None

    video_paths = []
    for i, p in enumerate(paths):
        ds_name = dataset_col[i] if dataset_col is not None else None

        if ds_name == "OOPS":
            video_paths.append(_resolve_oops_video_path(p, oops_dir))
        elif ds_name == "of-syn":
            video_paths.append(_resolve_syn_video_path(p, syn_dir))
        elif ds_name is None and need_oops and not need_syn:
            # Pure OOPS config (of-itw) without dataset column
            video_paths.append(_resolve_oops_video_path(p, oops_dir))
        elif ds_name is None and need_syn and not need_oops:
            # Pure syn config without dataset column
            video_paths.append(_resolve_syn_video_path(p, syn_dir))
        else:
            # Staged dataset row in mixed split: not downloadable
            video_paths.append(None)

    return ds.add_column("video", video_paths)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def add_video(
    dataset: Dataset | DatasetDict,
    config: str,
    *,
    syn_video_dir: str | Path | None = None,
    oops_video_dir: str | Path | None = None,
    download_syn: bool = True,
    consent: bool = False,
) -> Dataset | DatasetDict:
    """Add a 'video' column with absolute file paths to a loaded dataset.

    Args:
        dataset: A Dataset or DatasetDict loaded from OmniFall.
        config: The config name used when loading the dataset.
        syn_video_dir: Directory containing extracted OF-Syn videos.
            If None, uses the default cache directory. Ignored with OMNIFALL_ROOT.
        oops_video_dir: Directory containing prepared OOPS videos.
            If None, uses the default cache directory. Ignored with OMNIFALL_ROOT.
        download_syn: If True (default), automatically download and extract
            OF-Syn videos when needed. Set to False to skip. Ignored with OMNIFALL_ROOT.
        consent: If True, skip the interactive OOPS license consent prompt
            when auto-preparing OOPS videos. Ignored with OMNIFALL_ROOT.

    Returns:
        The dataset with an additional 'video' column containing file paths.

    Raises:
        ValueError: If the config has no video source.
        FileNotFoundError: If OMNIFALL_ROOT is set but expected files are missing,
            or if required videos are not available.
    """
    if config in NO_VIDEO_CONFIGS:
        raise ValueError(
            f"Config '{config}' has no associated video files (metadata only). "
            f"Video loading is supported for: {sorted(ALL_VIDEO_CONFIGS)}"
        )

    if config not in ALL_VIDEO_CONFIGS:
        raise ValueError(
            f"Unknown config '{config}' for video loading. "
            f"Supported configs: {sorted(ALL_VIDEO_CONFIGS)}"
        )

    # --- OMNIFALL_ROOT mode: local resolution, no downloads ---
    omnifall_root = get_omnifall_root()
    if omnifall_root is not None:
        if not omnifall_root.is_dir():
            raise FileNotFoundError(
                f"OMNIFALL_ROOT is set to '{omnifall_root}' but directory does not exist."
            )
        if isinstance(dataset, DatasetDict):
            return DatasetDict({
                split_name: _add_video_column_root(ds, omnifall_root)
                for split_name, ds in dataset.items()
            })
        else:
            return _add_video_column_root(dataset, omnifall_root)

    # --- Download mode: syn/OOPS via cache ---
    if config in STAGED_ONLY_CONFIGS:
        raise ValueError(
            f"Config '{config}' contains only staged dataset videos which "
            f"are not available for download. Set the OMNIFALL_ROOT environment "
            f"variable to point to a local copy of the datasets. "
            f"Layout: {{OMNIFALL_ROOT}}/{{dataset}}/video/{{path}}.mp4"
        )

    syn_dir = Path(syn_video_dir) if syn_video_dir else get_syn_video_dir()
    oops_dir = Path(oops_video_dir) if oops_video_dir else get_oops_video_dir()

    # Check if we need syn videos and ensure they're available
    any_split_needs_syn = any(
        _needs_syn_videos(config, s)
        for s in (dataset.keys() if isinstance(dataset, DatasetDict) else [None])
    )
    if any_split_needs_syn:
        if download_syn:
            syn_dir = download_and_extract_syn(syn_dir)
        elif not is_syn_extracted(syn_dir):
            raise FileNotFoundError(
                f"OF-Syn videos not found at {syn_dir}. "
                "Set download_syn=True to auto-download, or download manually."
            )

    # Check if we need OOPS videos and auto-prepare if necessary
    any_split_needs_oops = any(
        _needs_oops_videos(config, s)
        for s in (dataset.keys() if isinstance(dataset, DatasetDict) else [None])
    )
    if any_split_needs_oops and not is_oops_prepared(oops_dir):
        from ._oops import prepare_oops

        print("OOPS videos are required but not yet prepared.")
        oops_dir = prepare_oops(output_dir=oops_dir, consent=consent)

    # Add video column
    if isinstance(dataset, DatasetDict):
        return DatasetDict({
            split_name: _add_video_column_download(
                ds, config, split_name, syn_dir, oops_dir
            )
            for split_name, ds in dataset.items()
        })
    else:
        return _add_video_column_download(
            dataset, config, None, syn_dir, oops_dir
        )
