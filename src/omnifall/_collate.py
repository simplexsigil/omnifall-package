"""Collate functions for video dataloaders.

Handles batching of video data with mixed metadata formats across datasets
(e.g., OF-Syn demographic metadata not present in staged datasets).
"""

from __future__ import annotations

from typing import Any

import torch


def collate_fn(examples: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate function for video batches.

    Handles mixed datasets where some examples have metadata (OF-Syn
    demographics) and others don't. Only keys present in ALL examples
    are included in the batch.

    Args:
        examples: List of dicts from ``OmniFallVideoDataset.__getitem__``.

    Returns:
        Dictionary with collated tensors and metadata:
            - ``pixel_values``: Batched video tensors ``[B, C, T, H, W]``
            - ``labels``: Batched labels ``[B]``
            - Other keys: metadata if present in ALL examples
    """
    if not examples:
        return {}

    batch: dict[str, Any] = {}

    for key in examples[0]:
        if key == "pixel_values":
            batch["pixel_values"] = torch.stack(
                [example[key] for example in examples]
            )
        elif key == "label":
            batch["labels"] = torch.tensor(
                [example[key] for example in examples], dtype=torch.long
            )
        else:
            if not all(key in example for example in examples):
                continue
            values = [example[key] for example in examples]
            if values and isinstance(values[0], (int, float)):
                batch[key] = torch.tensor(values)
            elif values and isinstance(values[0], torch.Tensor):
                try:
                    batch[key] = torch.stack(values)
                except RuntimeError:
                    batch[key] = values
            else:
                batch[key] = values

    return batch
