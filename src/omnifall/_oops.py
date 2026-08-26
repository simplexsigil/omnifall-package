"""Streaming extraction of the OOPS videos used by OF-ItW.

OF-ItW is a 818-video subset of the OOPS dataset. The OOPS release is a single
45 GB archive, and the subset amounts to roughly 2.6 GB of it, so this module
never stores the archive: it streams the response body through two nested gzip
tar readers and writes out only the members OF-ItW needs, stopping as soon as
the last one has been seen.

The mapping from OOPS file names to OF-ItW ``path`` values comes from the Hub
file ``data_files/oops_video_mapping.csv``.

:mod:`omnifall._prepare` orchestrates; this module knows only about OOPS.
"""

from __future__ import annotations

import csv
import sys
import tarfile
from pathlib import Path
from typing import IO

from ._cache import get_oops_video_dir
from ._constants import (
    HF_REPO_ID,
    OOPS_LICENSE_TEXT,
    OOPS_MAPPING_FILE,
    OOPS_URL,
)

__all__ = [
    "load_mapping",
    "extract_oops",
    "prepare_oops",
]

#: Member of the outer archive that holds the videos.
INNER_ARCHIVE = "oops_dataset/video.tar.gz"

#: Read size for copying members out of the stream.
_CHUNK = 1 << 20


def load_mapping() -> dict[str, str]:
    """Load the OOPS-to-OF-ItW file-name mapping from the Hub.

    The Hub file spells its ``itw_path`` column *with* the ``.mp4`` suffix,
    whereas OmniFall's ``path`` column everywhere else is extension-free. This
    function returns the extension-free form, so that callers can append the
    extension the same way they do for every other component. The suffix is
    required rather than tolerated: a value without it would mean the Hub file
    changed shape, and silently accepting it would put videos at the wrong
    paths.

    Returns:
        A mapping from the member name inside the OOPS video archive to the
        OF-ItW ``path`` value, without extension.

    Raises:
        RuntimeError: If the mapping file lacks the expected columns, or if any
            ``itw_path`` does not end in ``.mp4``.
    """
    from huggingface_hub import hf_hub_download

    mapping_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=OOPS_MAPPING_FILE,
        repo_type="dataset",
    )
    with open(mapping_path, newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if "oops_path" not in fields or "itw_path" not in fields:
            raise RuntimeError(
                f"{OOPS_MAPPING_FILE} has columns {fields}, expected "
                f"'oops_path' and 'itw_path'. The Hub layout changed; upgrade "
                f"the omnifall package."
            )
        rows = [(row["oops_path"], row["itw_path"]) for row in reader]

    odd = [target for _, target in rows if not target.endswith(".mp4")]
    if odd:
        raise RuntimeError(
            f"{OOPS_MAPPING_FILE} has {len(odd)} 'itw_path' value(s) that do "
            f"not end in '.mp4' (e.g. {odd[0]!r}). The Hub layout changed; "
            f"upgrade the omnifall package."
        )
    return {member: target[: -len(".mp4")] for member, target in rows}


def _open_source(source: str) -> tuple[IO[bytes], object | None]:
    """Open *source* as a byte stream of the outer OOPS archive.

    Args:
        source: An ``http(s)://`` URL, or the path of a local
            ``video_and_anns.tar.gz``.

    Returns:
        A ``(stream, closer)`` pair. *closer* is the object that must be closed
        after the stream, or ``None`` when the stream owns itself.

    Raises:
        FileNotFoundError: If a local path does not exist.
        RuntimeError: If an HTTP request fails.
    """
    if source.startswith(("http://", "https://")):
        import requests

        response = requests.get(source, stream=True, timeout=(30, 300))
        if response.status_code != 200:
            raise RuntimeError(
                f"fetching {source} failed with HTTP {response.status_code}. "
                f"Download video_and_anns.tar.gz manually and pass it as "
                f"oops_archive=."
            )
        # decode_content handles a Content-Encoding, which is distinct from the
        # gzip container we are about to read ourselves.
        response.raw.decode_content = True
        return response.raw, response

    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"OOPS archive not found: {path}")
    return open(path, "rb"), None


def _inner_stream(outer: tarfile.TarFile) -> IO[bytes]:
    """Return a byte stream over the nested video archive.

    Args:
        outer: The outer archive, opened in streaming mode.

    Returns:
        A readable stream positioned at the start of :data:`INNER_ARCHIVE`.

    Raises:
        RuntimeError: If the outer archive does not contain the expected member.
    """
    for member in outer:
        if member.name == INNER_ARCHIVE:
            stream = outer.extractfile(member)
            if stream is None:
                raise RuntimeError(
                    f"{INNER_ARCHIVE} is present but not a regular file."
                )
            return stream
    raise RuntimeError(
        f"the OOPS archive does not contain {INNER_ARCHIVE}. Its layout "
        f"changed, or the download is truncated."
    )


def extract_oops(
    source: str,
    mapping: dict[str, str],
    output_dir: Path,
    *,
    resume: bool = True,
) -> int:
    """Stream the OOPS archive and write out the OF-ItW videos.

    Every target's parent directory is created before extraction, so a mapping
    that grows beyond ``falls/`` keeps working. The pass stops as soon as the
    last wanted member has been written, which for a remote source saves
    reading the tail of a 45 GB body.

    Args:
        source: An ``http(s)://`` URL or a local ``video_and_anns.tar.gz``.
        mapping: OOPS member name to OF-ItW ``path``, from :func:`load_mapping`.
        output_dir: OF-ItW video directory to write into.
        resume: Skip targets that already exist as non-empty files. This makes
            an interrupted run cheap to repeat: the stream restarts, but
            nothing is written twice.

    Returns:
        The number of videos present in *output_dir* after the pass, counting
        those a previous run had already written.

    Raises:
        RuntimeError: If the archive layout is not what OF-ItW expects, or if
            some wanted members never appeared in the stream.
    """
    output_dir = Path(output_dir)

    already: set[str] = set()
    if resume:
        for member, target in mapping.items():
            out = output_dir / f"{target}.mp4"
            if out.is_file() and out.stat().st_size > 0:
                already.add(member)

    remaining = set(mapping) - already
    total = len(mapping)
    if not remaining:
        print(f"OOPS: all {total} videos already present.", file=sys.stderr)
        return total

    # Created only once there is something to write, so that pointing this at
    # an already-complete read-only tree stays a pure read.
    for target in mapping.values():
        (output_dir / target).parent.mkdir(parents=True, exist_ok=True)

    if already:
        print(
            f"OOPS: resuming, {len(already)}/{total} already present.",
            file=sys.stderr,
        )
    if source.startswith("http"):
        print(
            "OOPS: streaming ~45 GB from the web; only the ~2.6 GB of wanted "
            "videos is written to disk. This takes 30-60 minutes.",
            file=sys.stderr,
        )

    written = 0
    stream, closer = _open_source(source)
    try:
        with tarfile.open(fileobj=stream, mode="r|gz") as outer:
            inner_stream = _inner_stream(outer)
            with tarfile.open(fileobj=inner_stream, mode="r|gz") as inner:
                for member in inner:
                    if not remaining:
                        break
                    if member.name not in remaining:
                        continue
                    out = output_dir / f"{mapping[member.name]}.mp4"
                    handle = inner.extractfile(member)
                    if handle is None:
                        raise RuntimeError(
                            f"{member.name} is in the archive but is not a "
                            f"regular file."
                        )
                    # Write beside the target and move into place, so an
                    # interruption never leaves a short file that a later
                    # resume would mistake for a finished one.
                    part = out.with_name(out.name + ".part")
                    with handle, open(part, "wb") as sink:
                        while True:
                            chunk = handle.read(_CHUNK)
                            if not chunk:
                                break
                            sink.write(chunk)
                    part.replace(out)
                    remaining.discard(member.name)
                    written += 1
                    if written % 50 == 0:
                        done = written + len(already)
                        print(
                            f"  OOPS: {done}/{total} videos", file=sys.stderr
                        )
    finally:
        stream.close()
        if closer is not None and hasattr(closer, "close"):
            closer.close()  # type: ignore[attr-defined]

    present = written + len(already)
    if remaining:
        shown = "\n".join(f"    {name}" for name in sorted(remaining)[:10])
        more = (
            f"\n    ... and {len(remaining) - 10} more"
            if len(remaining) > 10
            else ""
        )
        raise RuntimeError(
            f"OOPS: {len(remaining)} of {total} wanted videos were not found "
            f"in the archive:\n{shown}{more}\n"
            f"{present} were written. The OOPS release may have changed; "
            f"please report this."
        )

    print(f"OOPS: {present}/{total} videos ready.", file=sys.stderr)
    return present


def prepare_oops(
    output_dir: str | Path | None = None,
    oops_archive: str | Path | None = None,
    force: bool = False,
    consent: bool = False,
) -> Path:
    """Prepare the OOPS videos that OF-ItW references.

    Args:
        output_dir: Directory to place the prepared videos in. Defaults to the
            cache location from :func:`omnifall._cache.get_oops_video_dir`.
        oops_archive: A local copy of ``video_and_anns.tar.gz`` to read instead
            of streaming from the OOPS site.
        force: Re-extract videos that are already present.
        consent: Skip the interactive licence prompt. Set this only when the
            caller has already accepted the OOPS terms.

    Returns:
        The directory holding the prepared videos.

    Raises:
        RuntimeError: If the user declines the licence, or if extraction is
            incomplete.
    """
    out = Path(output_dir) if output_dir is not None else get_oops_video_dir()
    mapping = load_mapping()

    if not consent:
        print(OOPS_LICENSE_TEXT % len(mapping))
        print(f"\nOutput directory: {out}")
        try:
            answer = input("\nDo you agree and want to proceed? [y/N] ")
        except EOFError:
            # No one is there to answer -- a script, a CI job, a notebook
            # kernel. Refusing is right; the licence must be accepted by a
            # person, never by the absence of one. But a bare EOFError says
            # nothing about how to proceed, so name the flag.
            raise RuntimeError(
                "The OOPS licence has to be accepted before its videos can be "
                "downloaded, and there is no interactive terminal to accept it "
                "on. Read the notice above, and if you agree, re-run with "
                "consent=True (or `omnifall prepare OOPS --yes` on the command "
                "line)."
            ) from None
        if answer.strip().lower() not in ("y", "yes"):
            raise RuntimeError("OOPS video preparation cancelled by user.")

    out.mkdir(parents=True, exist_ok=True)

    if oops_archive is not None:
        source = str(Path(oops_archive).resolve())
    else:
        source = OOPS_URL
    print(f"Source: {source}", file=sys.stderr)

    extract_oops(source, mapping, out, resume=not force)
    print(f"OOPS videos ready at: {out}")
    return out
