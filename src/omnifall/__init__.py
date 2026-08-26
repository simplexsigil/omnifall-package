"""OmniFall: companion package for the OmniFall fall-detection dataset.

The annotations live on the HuggingFace Hub; the videos of the component
datasets mostly live at their original authors' sites. This package joins the
two, so that::

    import omnifall
    ds = omnifall.load("le2i-cs", video=True)

gives you a ``datasets.Dataset`` whose ``video`` column holds absolute paths to
real files on your disk, and::

    parts = omnifall.load_video_dataset("le2i-cs")

gives you PyTorch datasets that decode the annotated segment of each video on
demand, in the layout HuggingFace ``transformers`` expects.

Set ``OMNIFALL_ROOT`` to a directory laid out as
``{root}/{dataset}/video/{path}.mp4`` if you already have the videos. Otherwise
run ``omnifall status`` to see what can be fetched automatically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._configs import DEPRECATED_CONFIGS, list_configs, preferred_name
from ._constants import (
    ACTIVITY_LABELS,
    DATASETS,
    HF_REPO_ID,
    ITW_DATASETS,
    STAGED_DATASETS,
    SYN_DATASETS,
    DatasetInfo,
)
from ._label_maps import IDX2LABEL, LABEL2IDX
from ._load import load, load_video_dataset

if TYPE_CHECKING:  # pragma: no cover
    from ._collate import collate_fn
    from ._hf import compute_metrics, label_mappings, load_model, trainer_dataset
    from ._prepare import (
        DatasetNotAvailableError,
        ensure_dataset,
        prepare,
        prepare_oops,
        status,
        verify,
    )
    from ._resolve import Availability, ResolutionReport
    from ._transforms import VideoMAETransform, VideoTransform
    from ._video import MissingVideosError, add_video, resolution_report
    from ._video_dataset import (
        MultiOmniFallDataset,
        OmniFallVideoDataset,
        VideoDecodeError,
    )

#: Attribute name -> module it lives in. Resolved on first access so that
#: importing :mod:`omnifall` never pulls in torch, av or transformers.
_LAZY: dict[str, str] = {
    "add_video": "_video",
    "resolution_report": "_video",
    "MissingVideosError": "_video",
    "Availability": "_resolve",
    "ResolutionReport": "_resolve",
    "canonical_dataset": "_resolve",
    "prepare": "_prepare",
    "prepare_oops": "_prepare",
    "ensure_dataset": "_prepare",
    "status": "_prepare",
    "verify": "_prepare",
    "convert": "_prepare",
    "required_paths": "_prepare",
    "VerifyReport": "_prepare",
    "ConvertReport": "_prepare",
    "DatasetNotAvailableError": "_prepare",
    "ConversionNotImplementedError": "_prepare",
    "SOURCES": "_sources",
    "Source": "_sources",
    "OmniFallVideoDataset": "_video_dataset",
    "MultiOmniFallDataset": "_video_dataset",
    "VideoUnavailableError": "_video_dataset",
    "VideoDecodeError": "_decode",
    "decode_segment": "_decode",
    "probe": "_decode",
    "collate_fn": "_collate",
    "VideoTransform": "_transforms",
    "from_model": "_transforms",
    "VideoMAETransform": "_transforms",
    "load_model": "_hf",
    "label_mappings": "_hf",
    "compute_metrics": "_hf",
    "trainer_dataset": "_hf",
}


def __getattr__(name: str):
    """Import optional-dependency attributes on first use."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    import importlib

    try:
        mod = importlib.import_module(f".{module}", __package__)
    except ImportError as exc:
        # Only claim a missing optional dependency when that is actually what
        # went wrong. An ImportError raised from inside our own module is a bug
        # in this package, and dressing it up as "pip install omnifall[video]"
        # would send the user chasing the wrong problem.
        optional = {"torch", "av", "torchvision", "transformers", "numpy"}
        culprit = (exc.name or "").split(".")[0]
        if culprit in optional:
            raise ImportError(
                f"omnifall.{name} needs the optional dependency {culprit!r}. "
                f"Install the extras with: pip install 'omnifall[video]'"
            ) from exc
        raise
    value = getattr(mod, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)


__version__ = "0.2.0"

__all__ = [
    # loading
    "load",
    "load_video_dataset",
    "add_video",
    "resolution_report",
    # configs and datasets
    "list_configs",
    "preferred_name",
    "DEPRECATED_CONFIGS",
    "DATASETS",
    "DatasetInfo",
    "STAGED_DATASETS",
    "ITW_DATASETS",
    "SYN_DATASETS",
    "canonical_dataset",
    # acquisition
    "prepare",
    "prepare_oops",
    "ensure_dataset",
    "convert",
    "status",
    "verify",
    "required_paths",
    "SOURCES",
    "Source",
    "VerifyReport",
    "ConvertReport",
    # torch
    "OmniFallVideoDataset",
    "MultiOmniFallDataset",
    "collate_fn",
    "VideoTransform",
    "VideoMAETransform",
    "from_model",
    "decode_segment",
    "probe",
    # transformers
    "load_model",
    "label_mappings",
    "compute_metrics",
    "trainer_dataset",
    # labels
    "ACTIVITY_LABELS",
    "LABEL2IDX",
    "IDX2LABEL",
    # errors
    #
    # Three distinct failure modes, deliberately distinguishable:
    #   MissingVideosError     -- resolution found absent files (strict=True)
    #   VideoUnavailableError  -- one row has no video file at all
    #   VideoDecodeError       -- the file exists but could not be decoded
    # The first two subclass FileNotFoundError, so a coarse
    # ``except FileNotFoundError`` still catches both.
    "MissingVideosError",
    "VideoUnavailableError",
    "VideoDecodeError",
    "DatasetNotAvailableError",
    "ConversionNotImplementedError",
    # misc
    "Availability",
    "ResolutionReport",
    "HF_REPO_ID",
    "__version__",
]
