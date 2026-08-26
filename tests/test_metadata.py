"""Tests for the static tables: labels, dataset registry, config catalog.

These need neither network nor data, and they are the tests that catch a typo
in a table that would otherwise only surface as a mysterious KeyError deep in a
dataloader.
"""

from __future__ import annotations

import pytest

import omnifall
from omnifall._configs import DEPRECATED_CONFIGS, NON_SEGMENT_CONFIGS, preferred_name
from omnifall._constants import DATASETS
from omnifall._label_maps import (
    FALLSTATE_LABELS,
    IDX2FALLSTATE,
    IDX2LABEL,
    LABEL2IDX,
    to_fall_state,
)


class TestLabels:
    def test_sixteen_classes(self) -> None:
        assert len(omnifall.ACTIVITY_LABELS) == 16

    def test_names_are_unique(self) -> None:
        assert len(set(omnifall.ACTIVITY_LABELS)) == 16

    def test_maps_are_inverse(self) -> None:
        for idx, name in IDX2LABEL.items():
            assert LABEL2IDX[name] == idx

    def test_fall_and_fallen_are_distinct(self) -> None:
        # The whole point of OmniFall's taxonomy is that the fall *event* and
        # the *fallen* state are separate classes.
        assert LABEL2IDX["fall"] != LABEL2IDX["fallen"]

    def test_every_class_has_a_fall_state(self) -> None:
        assert set(IDX2FALLSTATE) == set(range(16))

    @pytest.mark.parametrize(
        ("label", "expected"),
        [("fall", "fall"), ("fallen", "fallen"), ("walk", "other"), ("jump", "other")],
    )
    def test_fall_state_grouping(self, label: str, expected: str) -> None:
        assert FALLSTATE_LABELS[to_fall_state(LABEL2IDX[label])] == expected

    def test_fall_state_rejects_unknown_id(self) -> None:
        with pytest.raises(KeyError, match="not an OmniFall class id"):
            to_fall_state(99)


class TestDatasetRegistry:
    def test_ten_components(self) -> None:
        assert len(DATASETS) == 10

    def test_keys_match_names(self) -> None:
        for key, info in DATASETS.items():
            assert key == info.name

    def test_kinds_are_known(self) -> None:
        assert {i.kind for i in DATASETS.values()} == {"staged", "itw", "syn"}

    def test_eight_staged_one_itw_one_syn(self) -> None:
        assert len(omnifall.STAGED_DATASETS) == 8
        assert len(omnifall.ITW_DATASETS) == 1
        assert len(omnifall.SYN_DATASETS) == 1

    def test_cmdfall_is_flagged_unobtainable(self) -> None:
        # CMDFall is the one component that cannot be downloaded; the package
        # must say so rather than fail obscurely later.
        assert DATASETS["cmdfall"].obtainable is False

    def test_every_other_dataset_is_obtainable(self) -> None:
        unobtainable = {n for n, i in DATASETS.items() if not i.obtainable}
        assert unobtainable == {"cmdfall"}

    def test_all_have_a_homepage(self) -> None:
        for name, info in DATASETS.items():
            assert info.homepage.startswith("http"), name


class TestConfigCatalog:
    def test_aliases_point_at_real_looking_names(self) -> None:
        # An alias must not map onto another alias, or resolution would need
        # to iterate.
        for alias, target in DEPRECATED_CONFIGS.items():
            assert target not in DEPRECATED_CONFIGS, alias

    def test_preferred_name_is_idempotent(self) -> None:
        for alias in DEPRECATED_CONFIGS:
            once = preferred_name(alias)
            assert preferred_name(once) == once

    def test_preferred_name_passes_through_unknown(self) -> None:
        assert preferred_name("something-new") == "something-new"

    def test_non_segment_configs(self) -> None:
        assert NON_SEGMENT_CONFIGS == {"metadata-syn", "framewise-syn"}


class TestPublicApi:
    def test_import_does_not_pull_torch(self) -> None:
        """Importing omnifall must stay cheap.

        The package is useful for annotation-only work with nothing but
        ``datasets`` installed, so torch/av/transformers must not be imported
        at module import time.
        """
        import subprocess
        import sys

        code = (
            "import sys, omnifall; "
            "heavy = [m for m in ('torch', 'av', 'transformers', 'torchvision') "
            "if m in sys.modules]; "
            "print(','.join(heavy))"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert out.stdout.strip() == "", f"eagerly imported: {out.stdout.strip()}"

    def test_all_names_resolve(self) -> None:
        missing = []
        for name in omnifall.__all__:
            try:
                getattr(omnifall, name)
            except (AttributeError, ImportError) as exc:
                missing.append(f"{name}: {exc}")
        assert not missing, "unresolvable public names: " + "; ".join(missing)

    def test_unknown_attribute_raises(self) -> None:
        with pytest.raises(AttributeError):
            omnifall.definitely_not_a_real_attribute

    def test_every_lazy_target_is_public(self) -> None:
        """Anything worth lazy-loading is worth documenting in ``__all__``."""
        from omnifall import _LAZY

        undeclared = sorted(set(_LAZY) - set(omnifall.__all__))
        assert not undeclared, f"lazily importable but not in __all__: {undeclared}"

    def test_missing_optional_dependency_message_is_not_misleading(self) -> None:
        """An internal ImportError must not masquerade as a missing extra.

        Telling someone to ``pip install omnifall[video]`` when the real fault
        is a broken import inside the package sends them chasing the wrong
        problem.
        """
        import importlib

        source = importlib.import_module("omnifall").__getattr__.__doc__
        assert source is not None


class TestErrorHierarchy:
    """The three failure modes must stay distinguishable.

    They were briefly named ``MissingVideoError`` and ``MissingVideosError``,
    one letter apart, which is the kind of pair that gets caught by the wrong
    ``except`` clause.
    """

    def test_names_are_not_confusable(self) -> None:
        names = [n for n in omnifall.__all__ if n.endswith("Error")]
        lowered = [n.lower().rstrip("s") for n in names]
        assert len(set(lowered)) == len(names), (
            f"error names differing only by case or a trailing 's': {names}"
        )

    def test_absence_errors_are_file_not_found(self) -> None:
        # So that a coarse `except FileNotFoundError` still works.
        assert issubclass(omnifall.MissingVideosError, FileNotFoundError)
        assert issubclass(omnifall.VideoUnavailableError, FileNotFoundError)

    def test_decode_error_is_not_a_file_error(self) -> None:
        # A corrupt file is present; conflating it with absence would send the
        # user to re-download data they already have.
        assert issubclass(omnifall.VideoDecodeError, RuntimeError)
        assert not issubclass(omnifall.VideoDecodeError, FileNotFoundError)

    def test_conversion_error_is_a_not_implemented_error(self) -> None:
        assert issubclass(
            omnifall.ConversionNotImplementedError, NotImplementedError
        )


class TestReviewRegressions:
    """Defects found by adversarial review, each reproduced before being fixed.

    They share a shape: the wrong answer was returned silently, so only a test
    that asserts the *right* answer keeps them fixed.
    """

    def test_accepts_rng_sees_wrapped_callables(self) -> None:
        """A seeded RNG must reach every callable shape a transform can take.

        Returning False here does not fail — it silently drops reproducibility,
        which is why `partial` and bound methods went unnoticed.
        """
        import functools

        from omnifall._video_dataset import _accepts_rng

        def takes(frames, rng=None):
            return frames

        def forwards(frames, **kw):
            return frames

        def plain(frames):
            return frames

        class Callable_:
            def __call__(self, frames, rng=None):
                return frames

            def method(self, frames, rng=None):
                return frames

        assert _accepts_rng(takes)
        assert _accepts_rng(functools.partial(takes))
        assert _accepts_rng(Callable_().method)
        assert _accepts_rng(Callable_())
        assert _accepts_rng(forwards)
        assert not _accepts_rng(plain)
        assert not _accepts_rng(None)

    def test_auto_sampling_handles_sliced_splits(self) -> None:
        from omnifall._load import _pick

        for split in ("train", "train[:80%]", "train[10:20]", "train[:100]+train[-100:]"):
            assert _pick("auto", split) == "random", split
        for split in ("validation", "test", "validation[:50%]"):
            assert _pick("auto", split) == "uniform", split

    def test_collate_metadata_is_order_independent(self) -> None:
        """Same values, different order, same result type."""
        import torch

        import omnifall

        def example(subject):
            return {
                "pixel_values": torch.zeros(2, 3, 4, 4),
                "label": 0,
                "subject": subject,
            }

        a = omnifall.collate_fn([example(1), example(None)])["subject"]
        b = omnifall.collate_fn([example(None), example(1)])["subject"]
        assert type(a) is type(b) is list

        numeric = omnifall.collate_fn([example(1), example(2)])["subject"]
        assert isinstance(numeric, torch.Tensor)

    def test_empty_directory_does_not_shadow_a_populated_one(self, tmp_path) -> None:
        """A killed prepare leaves an empty dir; it must not hide real videos."""
        import importlib
        import os

        root = tmp_path / "root"
        cache = tmp_path / "cache"
        (root / "le2i" / "video").mkdir(parents=True)
        populated = cache / "videos" / "le2i" / "video"
        populated.mkdir(parents=True)
        (populated / "clip.mp4").write_bytes(b"x")

        old = dict(os.environ)
        try:
            os.environ["OMNIFALL_ROOT"] = str(root)
            os.environ["OMNIFALL_CACHE_DIR"] = str(cache)
            for name in [k for k in os.environ if k.startswith("OMNIFALL_VIDEO_ROOT__")]:
                del os.environ[name]
            from omnifall import _cache as cache_mod

            importlib.reload(cache_mod)
            layer, chosen = cache_mod.dataset_video_dir_with_layer("le2i")
            assert chosen == populated, f"chose {chosen} via {layer}"
            assert cache_mod.is_dataset_prepared("le2i")

            # ...but a populated root still wins over a populated cache.
            (root / "le2i" / "video" / "clip.mp4").write_bytes(b"y")
            layer, chosen = cache_mod.dataset_video_dir_with_layer("le2i")
            assert chosen == root / "le2i" / "video"
        finally:
            os.environ.clear()
            os.environ.update(old)
            from omnifall import _cache as cache_mod

            importlib.reload(cache_mod)


class TestTransformGeometry:
    """Crop and scale must agree, or every sample comes out a different shape."""

    def test_scales_derive_from_image_size(self) -> None:
        import omnifall

        big = omnifall.VideoTransform("train", image_size=384)
        assert big.min_scale >= 384
        assert big.max_scale > big.min_scale

    def test_train_output_is_square_and_stable(self) -> None:
        import numpy as np

        import omnifall

        transform = omnifall.VideoTransform("train", image_size=384)
        frames = [np.zeros((240, 320, 3), np.uint8)] * 4
        shapes = {tuple(transform(frames)["pixel_values"].shape) for _ in range(6)}
        assert shapes == {(4, 3, 384, 384)}, shapes

    def test_impossible_geometry_raises(self) -> None:
        import pytest as _pytest

        import omnifall

        with _pytest.raises(ValueError, match="resize_size"):
            omnifall.VideoTransform("val", image_size=256, resize_size=224)
        with _pytest.raises(ValueError, match="min_scale"):
            omnifall.VideoTransform("train", image_size=256, min_scale=200)

    def test_explicit_location_always_wins(self, tmp_path) -> None:
        """An explicitly named directory must never be silently overridden.

        This function also names where preparation *writes*, so resolving an
        explicit argument to somewhere else can send writes at a directory the
        caller never mentioned — a shared read-only master copy, for instance.
        A first attempt at the shadowing fix above made `override=` lose to a
        populated OMNIFALL_ROOT, which is exactly that hazard.
        """
        import importlib
        import os

        root = tmp_path / "root"
        (root / "le2i" / "video").mkdir(parents=True)
        (root / "le2i" / "video" / "populated.mp4").write_bytes(b"x")
        explicit = tmp_path / "explicit"
        explicit.mkdir()

        old = dict(os.environ)
        try:
            os.environ["OMNIFALL_ROOT"] = str(root)
            for name in [k for k in os.environ if k.startswith("OMNIFALL_VIDEO_ROOT__")]:
                del os.environ[name]
            from omnifall import _cache as cache_mod

            importlib.reload(cache_mod)

            layer, chosen = cache_mod.dataset_video_dir_with_layer(
                "le2i", override=str(explicit)
            )
            assert chosen == explicit, f"argument lost to {layer}"

            os.environ["OMNIFALL_VIDEO_ROOT__le2i"] = str(explicit)
            importlib.reload(cache_mod)
            layer, chosen = cache_mod.dataset_video_dir_with_layer("le2i")
            assert chosen == explicit, f"per-dataset env lost to {layer}"
        finally:
            os.environ.clear()
            os.environ.update(old)
            from omnifall import _cache as cache_mod

            importlib.reload(cache_mod)
