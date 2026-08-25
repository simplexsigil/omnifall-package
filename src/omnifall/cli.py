"""CLI entry points for the omnifall package."""

from __future__ import annotations

import argparse
import sys


def cmd_prepare_oops(args: argparse.Namespace) -> None:
    """Run OOPS video preparation."""
    from ._oops import prepare_oops

    prepare_oops(
        output_dir=args.output_dir,
        oops_archive=args.oops_archive,
        force=args.force,
        consent=args.yes,
    )


def cmd_info(args: argparse.Namespace) -> None:
    """Show cache status and available configs."""
    from ._cache import (
        get_cache_dir,
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
        STAGED_ONLY_CONFIGS,
    )

    cache_dir = get_cache_dir()
    syn_dir = get_syn_video_dir()
    oops_dir = get_oops_video_dir()
    omnifall_root = get_omnifall_root()

    print(f"OmniFall v{_get_version()}")
    print(f"HF Dataset: {HF_REPO_ID}")
    print()

    if omnifall_root is not None:
        print(f"OMNIFALL_ROOT: {omnifall_root}")
        if omnifall_root.is_dir():
            print("  Status: directory exists")
            print(f"  Video-enabled configs: {sorted(ALL_VIDEO_CONFIGS)}")
        else:
            print("  WARNING: directory does not exist!")
    else:
        print("OMNIFALL_ROOT: not set")
        print(f"Cache directory: {cache_dir}")
        print()
        print("Video status:")
        if is_syn_extracted(syn_dir):
            print(f"  OF-Syn videos:  READY ({syn_dir})")
        else:
            print("  OF-Syn videos:  NOT EXTRACTED (auto-downloaded on first use)")
        if is_oops_prepared(oops_dir):
            print(f"  OOPS videos:    READY ({oops_dir})")
        else:
            print("  OOPS videos:    NOT PREPARED (auto-prepared on first use)")
        print()
        print(f"Downloadable video configs: {sorted(DOWNLOADABLE_VIDEO_CONFIGS)}")
        print(f"Staged-only configs (need OMNIFALL_ROOT): {sorted(STAGED_ONLY_CONFIGS)}")


def _get_version() -> str:
    try:
        from . import __version__
        return __version__
    except ImportError:
        return "unknown"


def main(argv: list[str] | None = None) -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="omnifall",
        description="OmniFall dataset companion tool",
    )
    subparsers = parser.add_subparsers(dest="command")

    # prepare-oops
    p_oops = subparsers.add_parser(
        "prepare-oops",
        help="Download and prepare OOPS videos for OF-ItW",
    )
    p_oops.add_argument(
        "--output-dir",
        default=None,
        help="Directory to place prepared videos (default: cache dir)",
    )
    p_oops.add_argument(
        "--oops-archive",
        default=None,
        help="Path to already-downloaded video_and_anns.tar.gz",
    )
    p_oops.add_argument(
        "--force",
        action="store_true",
        help="Re-extract even if videos already exist",
    )
    p_oops.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip interactive license consent prompt",
    )
    p_oops.set_defaults(func=cmd_prepare_oops)

    # info
    p_info = subparsers.add_parser(
        "info",
        help="Show cache status and available configs",
    )
    p_info.set_defaults(func=cmd_info)

    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
