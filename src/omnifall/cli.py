"""Command-line interface for the omnifall package.

The CLI exists mainly because obtaining OmniFall's videos is a multi-step,
partly manual job: eight of the ten components belong to other authors and live
on their sites. ``omnifall sources`` says where each one comes from,
``omnifall prepare`` fetches the ones that can be fetched, and
``omnifall verify`` checks the result against the label files on the Hub ---
which is the only trustworthy answer to "did my download work?".
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

__all__ = ["main"]


def _get_version() -> str:
    """Return the installed package version, or ``"unknown"``."""
    try:
        from . import __version__

        return __version__
    except ImportError:  # pragma: no cover - only during a broken install
        return "unknown"


def _selected(args: argparse.Namespace) -> list[str]:
    """Return the components named on the command line.

    Args:
        args: A namespace carrying ``datasets`` and, optionally, ``all``.

    Returns:
        The requested component names, validated against the registry.

    Raises:
        SystemExit: If nothing was selected, or an unknown name was given.
    """
    from ._sources import SOURCES

    if getattr(args, "all", False):
        return list(SOURCES)
    names = list(args.datasets)
    if not names:
        raise SystemExit(
            "no dataset given. Name one or more of "
            f"{', '.join(SOURCES)}, or pass --all."
        )
    unknown = [n for n in names if n not in SOURCES]
    if unknown:
        raise SystemExit(
            f"unknown dataset(s): {', '.join(unknown)}. "
            f"Valid names (case-sensitive): {', '.join(SOURCES)}."
        )
    return names


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------


def cmd_info(args: argparse.Namespace) -> int:
    """Print cache locations and which components have videos on disk."""
    from ._cache import (
        dataset_video_dir_with_layer,
        get_cache_dir,
        get_omnifall_root,
    )
    from ._constants import HF_REPO_ID
    from ._prepare import status
    from ._sources import SOURCES

    print(f"omnifall {_get_version()}")
    print(f"Hub repository: {HF_REPO_ID}")
    print()

    root = get_omnifall_root()
    print(f"Cache directory: {get_cache_dir()}")
    if root is None:
        print("OMNIFALL_ROOT:   not set")
    elif root.is_dir():
        print(f"OMNIFALL_ROOT:   {root}")
    else:
        print(f"OMNIFALL_ROOT:   {root}  (WARNING: not a directory)")
    print()

    prepared = status()
    ready = [n for n, ok in prepared.items() if ok]
    print(f"Components with videos on disk: {len(ready)}/{len(SOURCES)}")
    for name in SOURCES:
        layer, directory = dataset_video_dir_with_layer(name)
        mark = "yes" if prepared[name] else "no "
        print(f"  {name:<10} {mark}  [{layer}] {directory}")
    print()
    if len(ready) < len(SOURCES):
        print("Next steps:")
        print("  omnifall status          what each component needs")
        print("  omnifall prepare --all   fetch everything that can be fetched")
        print("  omnifall sources <name>  manual instructions for the rest")
    return 0


# ---------------------------------------------------------------------------
# configs
# ---------------------------------------------------------------------------


def cmd_configs(args: argparse.Namespace) -> int:
    """List the configs the Hub repository currently serves."""
    from ._configs import CONFIG_GROUPS, DEPRECATED_CONFIGS, list_configs

    try:
        names = list_configs(refresh=args.refresh)
    except Exception as error:  # noqa: BLE001 - network failure is the point
        print(
            f"could not list configs from the Hub: {error}", file=sys.stderr
        )
        return 1

    if not args.group:
        for name in names:
            note = ""
            if name in DEPRECATED_CONFIGS:
                note = f"  (deprecated, prefer {DEPRECATED_CONFIGS[name]})"
            print(f"{name}{note}")
        print(f"\n{len(names)} configs")
        return 0

    groups: dict[str, list[str]] = {}
    for name in names:
        if name in DEPRECATED_CONFIGS:
            continue
        label = CONFIG_GROUPS.get(name, "Other")
        groups.setdefault(label, []).append(name)
    for label in sorted(groups):
        print(f"{label}:")
        for name in sorted(groups[label]):
            print(f"  {name}")
        print()
    deprecated = [n for n in names if n in DEPRECATED_CONFIGS]
    if deprecated:
        print(f"Deprecated aliases still served: {len(deprecated)}")
        print("  (see 'omnifall configs' for the full list)")
    return 0


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def cmd_status(args: argparse.Namespace) -> int:
    """Print, per component, whether it is prepared and where it comes from."""
    from ._prepare import IMPLEMENTED_CONVERSIONS, _human, status
    from ._sources import SOURCES

    # With --video-dir every component resolves to the same directory, so
    # "does it hold an .mp4" would answer yes ten times over for one prepared
    # component. status() then looks for each component's own paths instead,
    # which needs the Hub label files.
    try:
        prepared = status(video_dir=args.video_dir)
    except Exception as error:  # noqa: BLE001 - reported, not swallowed
        print(
            f"cannot tell which component the videos in {args.video_dir} "
            f"belong to: reading the Hub label files failed with "
            f"{error.__class__.__name__}: {error}",
            file=sys.stderr,
        )
        return 1
    header = f"{'dataset':<10} {'videos':<9} {'source':<9} {'size':<10} note"
    print(header)
    print("-" * len(header))
    for name, source in SOURCES.items():
        state = "present" if prepared[name] else "-"
        if source.gated:
            note = "access request required"
        elif not source.automatable:
            note = "manual download required"
        elif name not in IMPLEMENTED_CONVERSIONS:
            note = "download automated, conversion pending"
        else:
            note = "fully automated"
        print(
            f"{name:<10} {state:<9} {source.kind:<9} "
            f"{_human(source.approx_bytes):<10} {note}"
        )
    print()
    ready = sum(prepared.values())
    print(f"{ready}/{len(SOURCES)} components have videos on disk.")
    print("Run 'omnifall verify --all' to check completeness against the Hub.")
    return 0


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------


def cmd_prepare(args: argparse.Namespace) -> int:
    """Obtain the videos of the named components."""
    from ._prepare import (
        ConversionNotImplementedError,
        DatasetNotAvailableError,
        prepare,
    )

    names = _selected(args)
    if args.archive is not None and len(names) != 1:
        print(
            f"--archive names one component's data, so exactly one component "
            f"has to be named with it; {len(names)} were selected.",
            file=sys.stderr,
        )
        return 2
    failures = 0
    for name in names:
        # Banner on stderr so it stays interleaved with progress and errors.
        print(f"\n=== {name} ===", file=sys.stderr)
        try:
            where = prepare(
                name,
                consent=args.yes,
                video_dir=args.video_dir,
                force=args.force,
                download_only=args.download_only,
                archive=args.archive,
                download_dir=args.download_dir,
                workers=args.workers,
                keep_archives=args.keep_archives,
            )
            print(f"{name}: ready at {where}")
        except (DatasetNotAvailableError, ConversionNotImplementedError) as e:
            print(str(e), file=sys.stderr)
            failures += 1
        except KeyboardInterrupt:
            print("\ninterrupted; rerun to resume.", file=sys.stderr)
            return 130
        except Exception as error:  # noqa: BLE001 - reported, not swallowed
            print(f"{name}: FAILED: {error}", file=sys.stderr)
            failures += 1

    if failures:
        print(
            f"\n{failures} of {len(names)} component(s) could not be prepared "
            f"automatically. See 'omnifall sources <dataset>'.",
            file=sys.stderr,
        )
    return 1 if failures else 0


def cmd_prepare_oops(args: argparse.Namespace) -> int:
    """Deprecated alias for ``omnifall prepare OOPS``."""
    print(
        "note: 'omnifall prepare-oops' is deprecated; "
        "use 'omnifall prepare OOPS'.",
        file=sys.stderr,
    )
    from ._oops import prepare_oops

    try:
        prepare_oops(
            output_dir=args.output_dir,
            oops_archive=args.oops_archive,
            force=args.force,
            consent=args.yes,
        )
    except KeyboardInterrupt:
        print("\ninterrupted; rerun to resume.", file=sys.stderr)
        return 130
    except Exception as error:  # noqa: BLE001
        print(f"OOPS: FAILED: {error}", file=sys.stderr)
        return 1
    return 0


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


def cmd_verify(args: argparse.Namespace) -> int:
    """Check prepared trees against the label files on the Hub."""
    from ._prepare import verify

    names = _selected(args)
    incomplete = 0
    for name in names:
        try:
            report = verify(
                name, video_dir=args.video_dir, refresh=args.refresh
            )
        except Exception as error:  # noqa: BLE001
            print(f"{name:<10} ERROR       {error}", file=sys.stderr)
            incomplete += 1
            continue
        print(report.render(max_listed=args.max_listed))
        if not report.complete:
            incomplete += 1
    if len(names) > 1:
        print()
        print(f"{len(names) - incomplete}/{len(names)} component(s) complete.")
    return 1 if incomplete else 0


# ---------------------------------------------------------------------------
# convert
# ---------------------------------------------------------------------------


def cmd_convert(args: argparse.Namespace) -> int:
    """Reshape an already-downloaded release into OmniFall's layout."""
    from ._prepare import ConversionNotImplementedError, convert

    try:
        report = convert(
            args.dataset,
            args.src,
            video_dir=args.video_dir,
            overwrite=args.overwrite,
            workers=args.workers,
        )
    except ConversionNotImplementedError as error:
        print(str(error), file=sys.stderr)
        return 1
    except Exception as error:  # noqa: BLE001
        print(f"{args.dataset}: FAILED: {error}", file=sys.stderr)
        return 1
    print(report.summary())
    print(f"Now run: omnifall verify {args.dataset}")
    return 0


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------


def _print_source(
    name: str, verbose: bool, download_dir: Path | None = None
) -> None:
    """Print one registry entry, including where to put a manual download."""
    from ._prepare import IMPLEMENTED_CONVERSIONS, _human, download_locations
    from ._sources import get_source

    source = get_source(name)
    locations = download_locations(name, download_dir)
    print(f"=== {name} ===")
    if source.description:
        print(f"  {source.description}")
    print(f"  kind:      {source.kind}")
    print(f"  url:       {source.url or '(none -- manual download)'}")
    print(f"  size:      {_human(source.approx_bytes)}")
    print(f"  format:    {source.archive_format or '(n/a)'}")
    print(f"  license:   {source.license or '(see homepage)'}")
    print(f"  homepage:  {source.homepage}")
    if source.terms:
        # Where no licence is granted, this quote is the whole of the authors'
        # stated conditions -- almost always "cite us". Print it in full rather
        # than summarising it, and say where it came from.
        print("  the authors' own words, from the page above:")
        for line in textwrap.wrap(source.terms, width=72):
            print(f"    | {line}")
    if source.automatable:
        if name in IMPLEMENTED_CONVERSIONS:
            print(f"  automated: yes -- 'omnifall prepare {name}'")
        else:
            print(
                "  automated: download only -- 'omnifall prepare "
                f"{name} --download-only'; conversion pending"
            )
    else:
        print("  automated: no -- the files have to be obtained by hand")

    # The half that makes "download it yourself" a followable instruction:
    # the exact names, and the exact directory. Printed for every component,
    # automated or not, because an automated source can still be unreachable
    # from where the user happens to be sitting.
    print("  to supply the download yourself:")
    for line in locations.describe().splitlines():
        print(f"  {line}")
    print(f"    then run: omnifall prepare {name}")
    print(f"    or:       omnifall prepare {name} --archive <path>")

    if source.instructions:
        print("  where to get it:")
        for line in source.instructions.strip().splitlines():
            print(f"    {line}")
    if verbose and source.notes:
        print("  notes:")
        for line in source.notes.strip().splitlines():
            print(f"    {line}")
    print()


def cmd_sources(args: argparse.Namespace) -> int:
    """Print where each component's videos come from."""
    from ._sources import (
        ANNOTATION_LICENSE_CONFLICT,
        SOURCES,
        VIDEO_LICENSE_NOTICE,
    )

    names = list(args.datasets) if args.datasets else list(SOURCES)
    unknown = [n for n in names if n not in SOURCES]
    if unknown:
        print(
            f"unknown dataset(s): {', '.join(unknown)}. "
            f"Valid names: {', '.join(SOURCES)}.",
            file=sys.stderr,
        )
        return 1
    for name in names:
        _print_source(
            name,
            verbose=args.verbose or bool(args.datasets),
            download_dir=args.download_dir,
        )
    print(VIDEO_LICENSE_NOTICE)
    print(
        "Entries marked 'see homepage' are ones whose licence this package "
        "could not read\nat the source. Check before redistributing."
    )
    print()
    print(ANNOTATION_LICENSE_CONFLICT)
    return 0


# ---------------------------------------------------------------------------
# cite
# ---------------------------------------------------------------------------


def cmd_cite(args: argparse.Namespace) -> int:
    """Print the BibTeX a user of these components or configs must cite."""
    from ._prepare import datasets_in_config
    from ._sources import OMNIFALL_CITATION, SOURCES, get_source

    names = list(args.names)
    if not names:
        wanted = list(SOURCES)
        source_of = {name: "all components" for name in wanted}
    else:
        wanted = []
        source_of: dict[str, str] = {}
        for name in names:
            if name in SOURCES:
                found = [name]
                origin = "requested"
            else:
                # Not a component, so treat it as a config and ask the Hub
                # which components it actually draws rows from. Every config
                # carries a 'dataset' column, so this needs no lookup table.
                try:
                    found = list(datasets_in_config(name))
                except Exception as error:  # noqa: BLE001
                    print(
                        f"{name!r} is neither a component dataset nor a config "
                        f"that could be loaded: {error}\n"
                        f"Components: {', '.join(SOURCES)}\n"
                        f"Configs:    omnifall configs",
                        file=sys.stderr,
                    )
                    return 1
                origin = f"config {name}"
                print(
                    f"% config {name} draws on: {', '.join(found)}",
                    file=sys.stderr,
                )
            for component in found:
                if component not in wanted:
                    wanted.append(component)
                    source_of[component] = origin

    print("% OmniFall itself -- always cite this.")
    print(OMNIFALL_CITATION)
    print()
    seen: set[str] = set()
    for name in wanted:
        citation = get_source(name).citation
        if not citation or citation == OMNIFALL_CITATION:
            # of-syn is OmniFall's own contribution; it has no separate paper.
            continue
        if citation in seen:
            # edf and occu share a paper.
            continue
        seen.add(citation)
        print(f"% {name}")
        print(citation)
        print()
    print(
        "% Every component keeps its original licence; cite the papers above "
        "in\n% addition to OmniFall. Run 'omnifall sources' for the licences."
    )
    return 0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _add_video_dir(parser: argparse.ArgumentParser) -> None:
    """Add the shared ``--video-dir`` option to *parser*."""
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "directory the dataset's 'path' values are relative to, "
            "overriding OMNIFALL_ROOT and the cache"
        ),
    )


def _add_download_dir(parser: argparse.ArgumentParser) -> None:
    """Add the shared ``--download-dir`` option to *parser*."""
    from ._cache import ENV_DOWNLOAD_DIR

    parser.add_argument(
        "--download-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "where original releases are downloaded to, and where archives "
            f"you downloaded yourself are looked for; overrides "
            f"{ENV_DOWNLOAD_DIR}"
        ),
    )


def _add_workers(parser: argparse.ArgumentParser) -> None:
    """Add the shared ``--workers`` option to *parser*."""
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="N",
        help=(
            "how many videos to encode at once (default: 8, or the number of "
            "CPUs where that is smaller)"
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the ``omnifall`` command."""
    parser = argparse.ArgumentParser(
        prog="omnifall",
        description=(
            "Obtain and check the videos behind the OmniFall dataset. "
            "Annotations come from the HuggingFace Hub; most videos belong to "
            "the original authors and are fetched from their sites."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"omnifall {_get_version()}"
    )
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("info", help="cache locations and preparation state")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("configs", help="list the configs served by the Hub")
    p.add_argument(
        "--group", action="store_true", help="group configs by purpose"
    )
    p.add_argument(
        "--refresh", action="store_true", help="bypass the local Hub cache"
    )
    p.set_defaults(func=cmd_configs)

    p = sub.add_parser(
        "status", help="per-component source, size and preparation state"
    )
    _add_video_dir(p)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("prepare", help="obtain a component's videos")
    p.add_argument("datasets", nargs="*", metavar="DATASET")
    p.add_argument(
        "--all", action="store_true", help="prepare every component"
    )
    p.add_argument(
        "--yes", "-y", action="store_true",
        help="accept licence notices without prompting",
    )
    p.add_argument(
        "--force", action="store_true",
        help="prepare again even if videos are already present",
    )
    p.add_argument(
        "--download-only", action="store_true",
        help=(
            "fetch and unpack the original release into the download "
            "directory without reshaping it into OmniFall's layout"
        ),
    )
    p.add_argument(
        "--archive", type=Path, default=None, metavar="PATH",
        help=(
            "use this already-downloaded archive, directory of archives, or "
            "unpacked release instead of downloading; names one component"
        ),
    )
    p.add_argument(
        "--keep-archives", action="store_true",
        help=(
            "for up_fall: keep each downloaded archive instead of deleting it "
            "once its video is encoded (needs ~110 GB)"
        ),
    )
    _add_download_dir(p)
    _add_workers(p)
    _add_video_dir(p)
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser(
        "verify",
        help="check a prepared tree against the Hub label file",
        description=(
            "Compares the videos on disk against every 'path' value in "
            "labels/<dataset>.csv on the Hub. Exits non-zero if anything is "
            "missing."
        ),
    )
    p.add_argument("datasets", nargs="*", metavar="DATASET")
    p.add_argument("--all", action="store_true", help="verify every component")
    p.add_argument(
        "--refresh", action="store_true",
        help="re-download the label file instead of using the Hub cache",
    )
    p.add_argument(
        "--max-listed", type=int, default=10, metavar="N",
        help="how many missing paths to name before summarising (default: 10)",
    )
    _add_video_dir(p)
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser(
        "convert",
        help="reshape an already-downloaded release into OmniFall's layout",
    )
    p.add_argument("dataset", metavar="DATASET")
    p.add_argument(
        "src", metavar="SRC_DIR", type=Path,
        help="root of the unpacked original release",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="rewrite videos that already exist",
    )
    _add_workers(p)
    _add_video_dir(p)
    p.set_defaults(func=cmd_convert)

    p = sub.add_parser(
        "sources", help="where each component's videos come from"
    )
    p.add_argument("datasets", nargs="*", metavar="DATASET")
    p.add_argument(
        "--verbose", "-v", action="store_true", help="include notes"
    )
    _add_download_dir(p)
    p.set_defaults(func=cmd_sources)

    p = sub.add_parser(
        "cite",
        help="print the BibTeX you must cite",
        description=(
            "Prints the OmniFall entry plus one entry per component you use. "
            "Names may be components (le2i, OOPS, ...) or configs "
            "(of-sta-cs, cs, ...); a config is resolved by loading it and "
            "reading its 'dataset' column, so you get exactly the papers that "
            "config draws on. With no arguments, prints everything."
        ),
    )
    p.add_argument(
        "names", nargs="*", metavar="DATASET_OR_CONFIG",
        help="component datasets and/or Hub config names",
    )
    p.set_defaults(func=cmd_cite)

    p = sub.add_parser(
        "prepare-oops", help="deprecated alias for 'prepare OOPS'"
    )
    p.add_argument("--output-dir", default=None)
    p.add_argument(
        "--oops-archive", default=None,
        help="path to an already-downloaded video_and_anns.tar.gz",
    )
    p.add_argument("--force", action="store_true")
    p.add_argument("--yes", "-y", action="store_true")
    p.set_defaults(func=cmd_prepare_oops)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the ``omnifall`` command.

    Args:
        argv: Arguments to parse; ``sys.argv[1:]`` when omitted.

    Returns:
        A process exit status.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("\ninterrupted.", file=sys.stderr)
        return 130
    except SystemExit as error:
        if isinstance(error.code, str):
            print(error.code, file=sys.stderr)
            return 2
        raise


if __name__ == "__main__":
    sys.exit(main())
