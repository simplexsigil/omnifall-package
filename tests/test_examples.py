"""Executes the example notebooks.

Example code that has drifted from the API is worse than no example, and it
drifts silently — nothing else in the suite imports these. `trainer_dataset`
was already broken by a missing `VideoTransform.from_model` when this was first
run, so these tests earn their keep.
"""

from __future__ import annotations

import pathlib

import pytest

pytest.importorskip("nbformat")
pytest.importorskip("nbclient")

EXAMPLES = pathlib.Path(__file__).parent.parent / "examples"
NOTEBOOKS = sorted(EXAMPLES.glob("*.ipynb"))


def test_examples_directory_is_not_empty() -> None:
    assert NOTEBOOKS, f"no notebooks found in {EXAMPLES}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_valid(path: pathlib.Path) -> None:
    """Structurally valid, and shipped without stale outputs."""
    import nbformat

    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    assert nb.cells, "notebook has no cells"

    executed = [
        i
        for i, c in enumerate(nb.cells)
        if c.cell_type == "code" and (c.get("outputs") or c.get("execution_count"))
    ]
    assert not executed, (
        f"cells {executed} carry outputs; ship notebooks cleared so diffs stay "
        "readable and no local paths leak"
    )


@pytest.mark.network
@pytest.mark.localdata
@pytest.mark.slow
@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_runs(path: pathlib.Path, tmp_path, omnifall_root) -> None:
    """Every cell executes without raising.

    Run against a prepared video root. Cells that would download tens of GB are
    commented out in the notebooks themselves, so this stays offline-ish: it
    touches the Hub for annotations and one small checkpoint, nothing more.
    """
    import shutil

    import nbformat
    from nbclient import NotebookClient

    local = tmp_path / path.name
    shutil.copy(path, local)

    nb = nbformat.read(local, as_version=4)
    client = NotebookClient(
        nb,
        timeout=900,
        kernel_name="python3",
        resources={"metadata": {"path": str(tmp_path)}},
    )
    client.execute()
