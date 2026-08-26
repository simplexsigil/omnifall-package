"""Label index mappings derived from :data:`omnifall.ACTIVITY_LABELS`."""

from __future__ import annotations

from ._constants import ACTIVITY_LABELS, FALL_STATE_GROUPS

LABEL2IDX: dict[str, int] = {name: i for i, name in enumerate(ACTIVITY_LABELS)}
IDX2LABEL: dict[int, str] = {i: name for i, name in enumerate(ACTIVITY_LABELS)}

#: Collapses the 16 classes onto ``fall`` / ``fallen`` / ``other``, the coarse
#: target used by most single-dataset fall-detection benchmarks.
IDX2FALLSTATE: dict[int, str] = {
    LABEL2IDX[name]: group
    for group, names in FALL_STATE_GROUPS.items()
    for name in names
}

FALLSTATE_LABELS: list[str] = list(FALL_STATE_GROUPS)
FALLSTATE2IDX: dict[str, int] = {n: i for i, n in enumerate(FALLSTATE_LABELS)}


def to_fall_state(label: int) -> int:
    """Map a 16-class label id onto the 3-class fall-state id.

    Args:
        label: A class id in ``range(16)``.

    Returns:
        ``0`` for ``fall``, ``1`` for ``fallen``, ``2`` for everything else.

    Raises:
        KeyError: If *label* is not a valid OmniFall class id.
    """
    if label not in IDX2FALLSTATE:
        raise KeyError(
            f"{label!r} is not an OmniFall class id. Valid ids: 0..{len(ACTIVITY_LABELS) - 1}."
        )
    return FALLSTATE2IDX[IDX2FALLSTATE[label]]
