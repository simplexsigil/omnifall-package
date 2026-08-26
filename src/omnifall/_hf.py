"""Glue between OmniFall and the HuggingFace training stack.

The PyTorch dataset in :mod:`omnifall._video_dataset` is enough to train with a
plain ``DataLoader``. What this module adds is the small amount of extra
plumbing that ``transformers.Trainer`` expects: a label mapping in the shape
``AutoConfig`` wants, and a way to build a model whose head already matches the
OmniFall label space.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from ._constants import ACTIVITY_LABELS
from ._label_maps import IDX2LABEL, LABEL2IDX

if TYPE_CHECKING:  # pragma: no cover
    from ._video_dataset import OmniFallVideoDataset


def label_mappings() -> tuple[dict[int, str], dict[str, int]]:
    """Return ``(id2label, label2id)`` in the form ``AutoConfig`` expects.

    Example:
        >>> id2label, label2id = omnifall.label_mappings()
        >>> model = AutoModelForVideoClassification.from_pretrained(
        ...     "MCG-NJU/videomae-small-finetuned-kinetics",
        ...     id2label=id2label, label2id=label2id,
        ...     ignore_mismatched_sizes=True,
        ... )
    """
    return dict(IDX2LABEL), dict(LABEL2IDX)


def load_model(
    model_name: str = "MCG-NJU/videomae-small-finetuned-kinetics",
    *,
    labels: list[str] | None = None,
    **kwargs: Any,
):
    """Load a video-classification model with an OmniFall-shaped head.

    Args:
        model_name: Any HuggingFace video-classification checkpoint.
        labels: Override the label space. Defaults to the 16 OmniFall classes.
        **kwargs: Forwarded to ``from_pretrained``.

    Returns:
        A ``transformers`` model whose classifier has ``len(labels)`` outputs.
        The head is randomly initialised whenever it does not match the
        checkpoint --- that is expected and is why
        ``ignore_mismatched_sizes=True`` is set for you.

    Note:
        Feed this model ``pixel_values`` of shape ``(B, T, C, H, W)``, which is
        what :class:`~omnifall.OmniFallVideoDataset` produces by default
        (``output_format="TCHW"``).
    """
    from transformers import AutoModelForVideoClassification

    names = labels if labels is not None else ACTIVITY_LABELS
    id2label = {i: n for i, n in enumerate(names)}
    label2id = {n: i for i, n in enumerate(names)}

    kwargs.setdefault("ignore_mismatched_sizes", True)
    return AutoModelForVideoClassification.from_pretrained(
        model_name,
        num_labels=len(names),
        id2label=id2label,
        label2id=label2id,
        **kwargs,
    )


def compute_metrics(eval_pred: Any) -> dict[str, float]:
    """Accuracy and macro-averaged recall for ``transformers.Trainer``.

    Macro recall (balanced accuracy) is reported alongside plain accuracy
    because OmniFall is heavily imbalanced --- ``other`` and ``lying`` dominate,
    so accuracy alone flatters a model that never predicts ``fall``.

    Args:
        eval_pred: The ``(predictions, label_ids)`` tuple ``Trainer`` passes in.

    Returns:
        ``{"accuracy": ..., "balanced_accuracy": ...}``.
    """
    import numpy as np

    logits, labels = eval_pred[0], eval_pred[1]
    preds = np.asarray(logits).argmax(axis=-1)
    labels = np.asarray(labels)

    accuracy = float((preds == labels).mean()) if labels.size else 0.0

    recalls: list[float] = []
    for cls in np.unique(labels):
        mask = labels == cls
        recalls.append(float((preds[mask] == cls).mean()))
    balanced = float(np.mean(recalls)) if recalls else 0.0

    return {"accuracy": accuracy, "balanced_accuracy": balanced}


def trainer_dataset(
    config: str,
    *,
    model_name: str | None = None,
    num_frames: int = 16,
    target_fps: float = 15.0,
    **kwargs: Any,
) -> dict[str, "OmniFallVideoDataset"]:
    """Build train/validation/test datasets ready for ``transformers.Trainer``.

    Uses ``sampling="random"`` on the train split and ``"uniform"`` elsewhere,
    and derives the spatial transform from *model_name* so the preprocessing
    matches the checkpoint.

    Args:
        config: OmniFall config name.
        model_name: Checkpoint whose image processor supplies image size and
            normalisation. ``None`` uses ImageNet statistics at 224x224.
        num_frames: Frames per clip. Must match the model's ``num_frames``.
        target_fps: Sampling rate used to lay the frames out in time.
        **kwargs: Forwarded to :func:`omnifall.load_video_dataset`.

    Returns:
        A dict keyed by split name.

    Example:
        >>> parts = omnifall.trainer_dataset("le2i-cs")
        >>> trainer = Trainer(
        ...     model=omnifall.load_model(),
        ...     train_dataset=parts["train"],
        ...     eval_dataset=parts["validation"],
        ...     data_collator=omnifall.collate_fn,
        ...     compute_metrics=omnifall.compute_metrics,
        ...     args=TrainingArguments(output_dir="out", remove_unused_columns=False),
        ... )
    """
    from ._load import load_video_dataset
    from ._transforms import VideoTransform

    def _tf(mode: str) -> Any:
        if model_name is not None:
            return VideoTransform.from_model(model_name, mode=mode)
        return VideoTransform(mode=mode)

    transform = {
        "train": _tf("train"),
        "validation": _tf("val"),
        "test": _tf("test"),
    }

    parts = load_video_dataset(
        config,
        num_frames=num_frames,
        target_fps=target_fps,
        sampling="auto",
        transform=transform,
        output_format="TCHW",
        **kwargs,
    )
    if not isinstance(parts, Mapping):  # pragma: no cover - split= was passed
        raise TypeError(
            "trainer_dataset() builds all splits at once; do not pass split=."
        )
    return dict(parts)
