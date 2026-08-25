"""Label index mappings derived from ACTIVITY_LABELS."""

from ._constants import ACTIVITY_LABELS

LABEL2IDX: dict[str, int] = {name: i for i, name in enumerate(ACTIVITY_LABELS)}
IDX2LABEL: dict[int, str] = {i: name for i, name in enumerate(ACTIVITY_LABELS)}
