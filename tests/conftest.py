"""Shared fixtures.

Tests are split into three tiers by what they need:

* no marker      -- pure unit tests, no network, no data.
* ``network``    -- talks to the HuggingFace Hub (annotations only, a few MB).
* ``localdata``  -- needs real video files, i.e. ``OMNIFALL_ROOT`` pointing at a
  prepared copy of the component datasets.

Run the fast tier with ``pytest -m 'not network and not localdata'``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "network: needs the HuggingFace Hub")
    config.addinivalue_line("markers", "localdata: needs real video files")
    config.addinivalue_line("markers", "slow: takes more than a few seconds")


@pytest.fixture(scope="session")
def omnifall_root() -> Path:
    """The prepared video root, or skip.

    Set ``OMNIFALL_ROOT`` to run the ``localdata`` tier.
    """
    env = os.environ.get("OMNIFALL_ROOT")
    if not env:
        pytest.skip("OMNIFALL_ROOT is not set")
    root = Path(env)
    if not root.is_dir():
        pytest.skip(f"OMNIFALL_ROOT={root} is not a directory")
    return root


@pytest.fixture(scope="session")
def small_config() -> str:
    """A config small enough to load repeatedly without being slow.

    ``le2i-cs`` is 967 segments over 192 videos across three splits.
    """
    return "le2i-cs"
