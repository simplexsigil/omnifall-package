"""OmniFall: Companion package for the OmniFall fall detection dataset."""

from ._constants import ACTIVITY_LABELS
from ._label_maps import IDX2LABEL, LABEL2IDX
from ._load import load, load_video_dataset

from ._oops import prepare_oops
from ._video import add_video


def __getattr__(name: str):
    # Lazy imports for optional heavy dependencies (torch, av, transformers)
    _lazy = {
        "OmniFallVideoDataset": "_video_dataset",
        "MultiOmniFallDataset": "_video_dataset",
        "collate_fn": "_collate",
        "VideoMAETransform": "_transforms",
    }
    if name in _lazy:
        import importlib

        mod = importlib.import_module(f".{_lazy[name]}", __package__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__version__ = "0.1.0"
__all__ = [
    "load",
    "load_video_dataset",
    "add_video",
    "prepare_oops",
    "ACTIVITY_LABELS",
    "LABEL2IDX",
    "IDX2LABEL",
    "OmniFallVideoDataset",
    "MultiOmniFallDataset",
    "collate_fn",
    "VideoMAETransform",
    "__version__",
]
