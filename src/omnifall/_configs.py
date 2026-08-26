"""Catalog of the configs published by the OmniFall Hub repository.

The package deliberately does *not* gate video loading on the config name.
Every config carries a ``dataset`` column and video files are addressed by
``(dataset, path)``, so resolution works uniformly for all of them --- including
configs added to the Hub after this package was released.

What lives here is only what genuinely cannot be derived from a loaded table:
the deprecated-alias map, and the grouping used for human-facing listings.
"""

from __future__ import annotations

from ._constants import HF_REPO_ID

#: Config names the Hub still serves but which have a preferred spelling.
#: Mapping an alias is purely informational --- the alias itself loads fine.
DEPRECATED_CONFIGS: dict[str, str] = {
    "cs-staged": "of-sta-cs",
    "cv-staged": "of-sta-cv",
    "cs-staged-wild": "of-sta-to-all-cs",
    "cv-staged-wild": "of-sta-to-all-cv",
    "of-sta-to-itw-cs": "of-sta-to-all-cs",
    "of-sta-to-itw-cv": "of-sta-to-all-cv",
    "of-sta-itw-cs": "of-sta-to-all-cs",
    "of-sta-itw-cv": "of-sta-to-all-cv",
    "of-syn-to-itw": "of-syn-to-all-cs",
    "of-syn-itw": "of-syn-to-all-cs",
    "of-sta-syn-to-itw-cs": "of-sta-syn-to-all-cs",
    "of-sta-syn-to-itw-cv": "of-sta-syn-to-all-cv",
    "OOPS": "of-itw",
    "caucafall": "caucafall-cs",
    "cmdfall": "cmdfall-cs",
    "edf": "edf-cs",
    "gmdcsa24": "gmdcsa24-cs",
    "le2i": "le2i-cs",
    "mcfd": "mcfd-cs",
    "occu": "occu-cs",
    "up_fall": "up_fall-cs",
}

#: Configs that carry annotations only and have no video counterpart at all.
#: ``metadata-syn`` describes whole videos rather than segments, and
#: ``framewise-syn`` holds per-frame label arrays.
NON_SEGMENT_CONFIGS: frozenset[str] = frozenset({"metadata-syn", "framewise-syn"})

#: The eight staged datasets, in the spelling used by config names.
#: Note that config names lowercase GMDCSA24 while the ``dataset`` column does
#: not; :func:`omnifall._resolve.canonical_dataset` bridges the two.
STAGED_CONFIG_STEMS: tuple[str, ...] = (
    "caucafall",
    "cmdfall",
    "edf",
    "gmdcsa24",
    "le2i",
    "mcfd",
    "occu",
    "up_fall",
)

#: Human-facing grouping used by ``omnifall configs``.
CONFIG_GROUPS: dict[str, str] = {
    "labels": "Annotations only (no splits)",
    "labels-syn": "Annotations only (no splits)",
    "metadata-syn": "Annotations only (no splits)",
    "framewise-syn": "Annotations only (no splits)",
    "of-sta-cs": "Same-domain: staged",
    "of-sta-cv": "Same-domain: staged",
    "of-itw": "Same-domain: in-the-wild",
    "of-syn": "Same-domain: synthetic",
    "of-syn-cross-age": "Same-domain: synthetic",
    "of-syn-cross-ethnicity": "Same-domain: synthetic",
    "of-syn-cross-bmi": "Same-domain: synthetic",
    "cs": "Aggregate (staged + in-the-wild)",
    "cv": "Aggregate (staged + in-the-wild)",
}


def list_configs(*, refresh: bool = False) -> list[str]:
    """Return every config name the Hub repository currently serves.

    Queried live so that configs added after this release are still listed.

    Args:
        refresh: Bypass the ``datasets`` HTTP cache.

    Returns:
        Config names in the order the Hub reports them.
    """
    from datasets import get_dataset_config_names

    return list(
        get_dataset_config_names(
            HF_REPO_ID,
            download_mode="force_redownload" if refresh else None,
        )
    )


def preferred_name(config: str) -> str:
    """Return the non-deprecated spelling of *config*.

    Unknown names are returned unchanged --- the Hub is the authority on which
    configs exist, not this table.
    """
    return DEPRECATED_CONFIGS.get(config, config)
