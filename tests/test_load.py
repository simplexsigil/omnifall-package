"""Tests for loading annotations from the Hub and attaching video paths.

Marked ``network`` because they talk to the HuggingFace Hub. The annotation
tables are small (the largest config is a few MB), so this tier is fast once
the ``datasets`` cache is warm.
"""

from __future__ import annotations

import pytest
from datasets import Dataset, DatasetDict

import omnifall

pytestmark = pytest.mark.network

#: Columns every per-segment config is required to expose.
SEGMENT_COLUMNS = {"path", "label", "start", "end", "subject", "cam", "dataset"}

#: The exact set of values that may appear in the ``dataset`` column.
COMPONENT_NAMES = set(omnifall.DATASETS)


class TestConfigCatalog:
    def test_hub_serves_configs(self) -> None:
        names = omnifall.list_configs()
        assert len(names) >= 70

    def test_every_deprecated_alias_still_exists_on_the_hub(self) -> None:
        """If the Hub drops an alias, our table should stop advertising it."""
        names = set(omnifall.list_configs())
        stale = sorted(a for a in omnifall.DEPRECATED_CONFIGS if a not in names)
        assert not stale, f"aliases no longer served by the Hub: {stale}"

    def test_every_alias_target_exists_on_the_hub(self) -> None:
        names = set(omnifall.list_configs())
        broken = sorted(
            f"{a} -> {t}"
            for a, t in omnifall.DEPRECATED_CONFIGS.items()
            if t not in names
        )
        assert not broken, f"aliases pointing at non-existent configs: {broken}"


class TestRegistryMatchesTheHub:
    """The static tables must agree with the published annotations.

    Hand-maintained counts drift, and a wrong count in a user-facing report is
    a quiet lie. These tests pin the registry to the label files, which are the
    authority.
    """

    @pytest.mark.parametrize("dataset", sorted(omnifall.DATASETS))
    def test_video_count_matches_label_file(self, dataset: str) -> None:
        import pandas as pd
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            omnifall.HF_REPO_ID,
            f"labels/{dataset}.csv",
            repo_type="dataset",
        )
        expected = pd.read_csv(path)["path"].nunique()
        assert omnifall.DATASETS[dataset].n_videos == expected

    def test_every_component_has_a_label_file(self) -> None:
        from huggingface_hub import HfApi

        files = {
            f.rfilename for f in HfApi().repo_info(omnifall.HF_REPO_ID, repo_type="dataset").siblings
        }
        for name in omnifall.DATASETS:
            assert f"labels/{name}.csv" in files, name


class TestLoad:
    def test_returns_dataset_dict_without_split(self, small_config: str) -> None:
        ds = omnifall.load(small_config)
        assert isinstance(ds, DatasetDict)
        assert set(ds) == {"train", "validation", "test"}

    def test_returns_dataset_with_split(self, small_config: str) -> None:
        ds = omnifall.load(small_config, split="train")
        assert isinstance(ds, Dataset)
        assert len(ds) > 0

    def test_segment_columns(self, small_config: str) -> None:
        ds = omnifall.load(small_config, split="train")
        assert SEGMENT_COLUMNS <= set(ds.column_names)

    def test_label_is_a_classlabel_with_our_names(self, small_config: str) -> None:
        ds = omnifall.load(small_config, split="train")
        assert ds.features["label"].names == omnifall.ACTIVITY_LABELS

    def test_dataset_column_values_are_known(self, small_config: str) -> None:
        ds = omnifall.load(small_config, split="train")
        assert set(ds.unique("dataset")) <= COMPONENT_NAMES

    def test_segments_are_well_formed(self, small_config: str) -> None:
        ds = omnifall.load(small_config, split="train")
        for row in ds.select(range(min(200, len(ds)))):
            assert row["end"] > row["start"] >= 0.0
            assert 0 <= row["label"] < 16
            assert not row["path"].startswith("/")
            assert "." not in row["path"].rsplit("/", 1)[-1]

    def test_video_rejected_for_non_segment_config(self) -> None:
        with pytest.raises(ValueError, match="not a per-segment table"):
            omnifall.load("framewise-syn", video=True)

    @pytest.mark.slow
    def test_all_configs_load_and_expose_dataset_column(self) -> None:
        """The invariant the whole resolution design rests on.

        Video files are addressed by ``(dataset, path)``. If any config were to
        lack a ``dataset`` column, that design would break --- so assert it for
        every config the Hub serves, not just the ones we happen to know about.
        """
        failures: list[str] = []
        for name in omnifall.list_configs():
            try:
                ds = omnifall.load(name)
            except Exception as exc:  # noqa: BLE001 - reporting, not handling
                failures.append(f"{name}: load failed: {type(exc).__name__}: {exc}")
                continue
            for split, part in ds.items():
                if "dataset" not in part.column_names:
                    failures.append(f"{name}[{split}]: no 'dataset' column")
                    continue
                unknown = set(part.unique("dataset")) - COMPONENT_NAMES
                if unknown:
                    failures.append(f"{name}[{split}]: unknown datasets {unknown}")
        assert not failures, "\n".join(failures)


@pytest.mark.localdata
class TestAddVideo:
    def test_video_column_is_absolute_and_exists(
        self, small_config: str, omnifall_root
    ) -> None:
        ds = omnifall.load(small_config, split="train", video=True, strict=True)
        assert "video" in ds.column_names
        from pathlib import Path

        for path in ds["video"][:50]:
            assert path is not None
            assert Path(path).is_absolute()
            assert Path(path).exists()

    def test_strict_resolution_is_complete_for_every_component(
        self, omnifall_root
    ) -> None:
        """Every dataset must resolve from a fully prepared root.

        ``labels`` references all nine non-synthetic components at once, which
        makes it the single best end-to-end check of path resolution.
        """
        report = omnifall.resolution_report(omnifall.load("labels", split="train"))
        assert report.complete, report.summary()

    def test_missing_root_degrades_with_one_warning(
        self, small_config: str, monkeypatch
    ) -> None:
        monkeypatch.delenv("OMNIFALL_ROOT", raising=False)
        monkeypatch.setenv("OMNIFALL_CACHE_DIR", "/nonexistent-omnifall-cache")
        with pytest.warns(UserWarning):
            ds = omnifall.load(small_config, split="train", video=True)
        assert all(v is None for v in ds["video"])

    def test_strict_raises_when_videos_are_absent(
        self, small_config: str, monkeypatch
    ) -> None:
        monkeypatch.delenv("OMNIFALL_ROOT", raising=False)
        monkeypatch.setenv("OMNIFALL_CACHE_DIR", "/nonexistent-omnifall-cache")
        with pytest.raises(omnifall.MissingVideosError):
            omnifall.load(small_config, split="train", video=True, strict=True)
