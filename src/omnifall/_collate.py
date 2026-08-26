"""Collate function for OmniFall video dataloaders.

Handles the two things the default ``torch`` collate cannot: metadata columns
that exist in some component datasets but not others (OF-Syn ships demographic
fields the staged datasets do not have), and the ``pixel_values=None`` rows that
``OmniFallVideoDataset(on_error="skip")`` produces.

Requires the optional ``torch`` dependency. Imported lazily by :mod:`omnifall`.
"""

from __future__ import annotations

from typing import Any

import torch

__all__ = ["collate_fn"]


def collate_fn(examples: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate a list of :class:`~omnifall.OmniFallVideoDataset` samples.

    Only keys present in *all* remaining examples end up in the batch, so
    mixing datasets with different metadata columns is safe.

    Samples with ``pixel_values is None`` -- produced by
    ``OmniFallVideoDataset(on_error="skip")`` -- are dropped, which makes the
    batch smaller than ``batch_size``. If every sample in a batch was dropped,
    this raises rather than returning an empty dict, because a training loop
    given ``{}`` fails much later and much less clearly.

    Args:
        examples: Sample dicts from ``OmniFallVideoDataset.__getitem__``.

    Returns:
        A dict with ``pixel_values`` (stacked clip tensors), ``labels`` (a
        ``long`` tensor, renamed from the per-sample ``label`` key because that
        is what HuggingFace models expect), and every other key shared by all
        examples.

    Raises:
        ValueError: If *examples* is empty, or if every example was dropped
            because its video could not be decoded.
    """
    if not examples:
        raise ValueError(
            "collate_fn received an empty list of examples. A DataLoader "
            "should never produce this; check for a sampler that yields no "
            "indices."
        )

    kept = [e for e in examples if e.get("pixel_values") is not None]
    if not kept:
        errors = {e.get("error", "<no error recorded>") for e in examples}
        raise ValueError(
            f"All {len(examples)} samples in this batch failed to decode, so the "
            "batch is empty. Distinct errors: "
            + "; ".join(sorted(errors))
            + ". Use on_error='raise' (the default) to fail at the offending "
            "sample instead."
        )

    shared = [key for key in kept[0] if all(key in e for e in kept)]
    batch: dict[str, Any] = {}

    for key in shared:
        values = [example[key] for example in kept]
        if key == "pixel_values":
            shapes = {tuple(v.shape) for v in values}
            if len(shapes) > 1:
                raise ValueError(
                    "Cannot batch clips of different shapes: "
                    f"{sorted(shapes)}. OmniFall mixes component datasets with "
                    "different video resolutions, so a spatial transform is "
                    "required -- pass transform=omnifall.VideoTransform('val') "
                    "(or 'train') to OmniFallVideoDataset."
                )
            batch["pixel_values"] = torch.stack(values)
        elif key == "label":
            batch["labels"] = torch.tensor(values, dtype=torch.long)
        elif isinstance(values[0], torch.Tensor):
            try:
                batch[key] = torch.stack(values)
            except RuntimeError:
                batch[key] = values
        elif _all_numeric(values):
            batch[key] = torch.tensor(values)
        else:
            batch[key] = values

    return batch


def _all_numeric(values: list[Any]) -> bool:
    """Whether every value is a plain number that ``torch.tensor`` will accept.

    Inspecting only ``values[0]`` made the result depend on batch order: a
    nullable column such as ``subject`` yielded a tensor, a list, or a
    ``RuntimeError`` for the same data depending on where the ``None`` landed
    after shuffling. Booleans are excluded because ``torch.tensor`` turns them
    into a bool tensor, which is not what a metadata column wants.
    """
    return all(
        isinstance(v, (int, float)) and not isinstance(v, bool) for v in values
    )
