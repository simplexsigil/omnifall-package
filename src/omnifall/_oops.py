"""OOPS video preparation for OF-ItW.

Streams the OOPS dataset archive and extracts only the 818 videos used in
OF-ItW, renamed to match the OF-ItW path convention.
"""

from __future__ import annotations

import csv
import os
import subprocess
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download

from ._cache import get_oops_video_dir, is_oops_prepared
from ._constants import (
    EXPECTED_OOPS_COUNT,
    HF_REPO_ID,
    OOPS_LICENSE_TEXT,
    OOPS_MAPPING_FILE,
    OOPS_URL,
)


def _load_mapping() -> dict[str, str]:
    """Download and load the OOPS-to-ITW filename mapping from the HF Hub."""
    mapping_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=OOPS_MAPPING_FILE,
        repo_type="dataset",
    )
    mapping = {}
    with open(mapping_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row["oops_path"]] = row["itw_path"]
    return mapping


def _to_stdout_cmd(
    source: str, member: str
) -> tuple[str | list[str], bool]:
    """Build command to extract a single tar member to stdout."""
    if source.startswith("http://") or source.startswith("https://"):
        return f'curl -sL "{source}" | tar -xzf - --to-stdout "{member}"', True
    elif source.endswith(".tar.gz") or source.endswith(".tgz"):
        return ["tar", "-xzf", source, "--to-stdout", member], False
    else:
        return ["tar", "-xf", source, "--to-stdout", member], False


def _extract_videos(
    source: str, mapping: dict[str, str], output_dir: str
) -> int:
    """Stream through the OOPS archive and extract matching videos.

    The archive has a nested structure: the outer tar contains
    oops_dataset/video.tar.gz, which contains the actual video files.
    """
    total = len(mapping)
    print(f"Extracting {total} videos from OOPS archive...")
    if source.startswith("http"):
        print("(Streaming ~45GB from web, no local disk space needed)")
        print("(This may take 30-60 minutes depending on connection speed)")
    else:
        print("(Reading from local archive)")

    os.makedirs(os.path.join(output_dir, "falls"), exist_ok=True)

    found = 0
    remaining = set(mapping.keys())

    cmd, use_shell = _to_stdout_cmd(source, "oops_dataset/video.tar.gz")
    proc = subprocess.Popen(
        cmd, shell=use_shell, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )

    try:
        with tarfile.open(fileobj=proc.stdout, mode="r|gz") as tar:
            for member in tar:
                if not remaining:
                    break
                if member.name in remaining:
                    itw_path = mapping[member.name]
                    out_path = os.path.join(output_dir, itw_path)

                    f = tar.extractfile(member)
                    if f is not None:
                        with open(out_path, "wb") as out_f:
                            while True:
                                chunk = f.read(1024 * 1024)
                                if not chunk:
                                    break
                                out_f.write(chunk)
                        f.close()
                        found += 1
                        remaining.discard(member.name)
                        if found % 50 == 0:
                            print(f"  Extracted {found}/{total} videos...")
    finally:
        proc.stdout.close()
        proc.wait()

    print(f"Extracted {found}/{total} videos.")
    if remaining:
        print(f"WARNING: {len(remaining)} videos not found in archive:")
        for p in sorted(remaining)[:10]:
            print(f"  {p}")
        if len(remaining) > 10:
            print(f"  ... and {len(remaining) - 10} more")
    return found


def prepare_oops(
    output_dir: str | Path | None = None,
    oops_archive: str | Path | None = None,
    force: bool = False,
    consent: bool = False,
) -> Path:
    """Prepare OOPS videos for OF-ItW.

    Downloads and extracts only the 818 videos used in OF-ItW from the OOPS
    dataset archive. The archive is streamed (~45GB) and only the relevant
    videos (~2.6GB) are saved to disk.

    Args:
        output_dir: Directory to place prepared videos. Defaults to
            ~/.cache/omnifall/oops_prepared (or OMNIFALL_CACHE_DIR).
        oops_archive: Path to an already-downloaded OOPS archive
            (video_and_anns.tar.gz). If not provided, streams from the web.
        force: If True, re-extract even if videos already exist.
        consent: If True, skip the interactive license consent prompt.

    Returns:
        Path to the prepared video directory.
    """
    if output_dir is None:
        out = get_oops_video_dir()
    else:
        out = Path(output_dir)

    if not force and is_oops_prepared(out):
        print(f"OOPS videos already prepared at: {out}")
        return out

    mapping = _load_mapping()

    if not consent:
        print(OOPS_LICENSE_TEXT % len(mapping))
        print(f"\nOutput directory: {out}")
        answer = input("\nDo you agree and want to proceed? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            raise RuntimeError("OOPS video preparation cancelled by user.")

    out.mkdir(parents=True, exist_ok=True)

    if oops_archive:
        source = str(Path(oops_archive).resolve())
        if not Path(source).exists():
            raise FileNotFoundError(f"Archive not found: {source}")
    else:
        source = OOPS_URL

    print(f"Source: {source}")
    found = _extract_videos(source, mapping, str(out))

    print()
    print("=" * 60)
    print("Preparation complete!")
    print(f"  Output directory: {out}")
    print(f"  Videos extracted: {found}/{len(mapping)}")

    return out
