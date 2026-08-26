"""Tests for decoding, sampling and batching.

The invariants that matter here are the ones a silent bug would hide:
evaluation must be reproducible, the tensor layout must be the one the model
expects, and a decode failure must not quietly substitute a different sample.
"""

from __future__ import annotations

import pytest

import omnifall

torch = pytest.importorskip("torch")

pytestmark = [pytest.mark.network, pytest.mark.localdata]


@pytest.fixture(scope="module")
def hf_split(omnifall_root):
    """A small annotated split with resolved video paths."""
    return omnifall.load("le2i-cs", split="validation", video=True, strict=True)


def _dataset(hf_split, **kw):
    from omnifall import OmniFallVideoDataset

    kw.setdefault("num_frames", 8)
    kw.setdefault("target_fps", 15.0)
    return OmniFallVideoDataset(hf_split, **kw)


class TestShape:
    def test_default_layout_is_tchw(self, hf_split) -> None:
        ds = _dataset(hf_split, transform=omnifall.VideoTransform(mode="val"))
        pv = ds[0]["pixel_values"]
        assert pv.shape == (8, 3, 224, 224), pv.shape

    def test_ctwh_layout_available(self, hf_split) -> None:
        ds = _dataset(
            hf_split,
            transform=omnifall.VideoTransform(mode="val"),
            output_format="CTHW",
        )
        assert ds[0]["pixel_values"].shape == (3, 8, 224, 224)

    def test_raw_frames_without_transform(self, hf_split) -> None:
        ds = _dataset(hf_split, output_format="THWC")
        pv = ds[0]["pixel_values"]
        assert pv.shape[0] == 8 and pv.shape[-1] == 3
        assert pv.dtype == torch.uint8

    def test_exactly_num_frames_even_for_short_segments(self, hf_split) -> None:
        ds = _dataset(hf_split, num_frames=16, output_format="THWC")
        # Pick the shortest segments, where padding must kick in.
        durations = [e - s for s, e in zip(hf_split["start"], hf_split["end"])]
        order = sorted(range(len(durations)), key=lambda i: durations[i])
        for idx in order[:10]:
            assert ds[idx]["pixel_values"].shape[0] == 16


class TestDeterminism:
    def test_uniform_sampling_is_reproducible(self, hf_split) -> None:
        a = _dataset(hf_split, sampling="uniform", output_format="THWC")
        b = _dataset(hf_split, sampling="uniform", output_format="THWC")
        for idx in (0, 1, 7):
            assert torch.equal(a[idx]["pixel_values"], b[idx]["pixel_values"])

    def test_center_sampling_is_reproducible(self, hf_split) -> None:
        a = _dataset(hf_split, sampling="center", output_format="THWC")
        b = _dataset(hf_split, sampling="center", output_format="THWC")
        assert torch.equal(a[3]["pixel_values"], b[3]["pixel_values"])

    def test_seeded_random_sampling_is_reproducible(self, hf_split) -> None:
        a = _dataset(hf_split, sampling="random", seed=0, output_format="THWC")
        b = _dataset(hf_split, sampling="random", seed=0, output_format="THWC")
        for idx in (0, 5):
            assert torch.equal(a[idx]["pixel_values"], b[idx]["pixel_values"])

    def test_different_seeds_differ(self, hf_split) -> None:
        a = _dataset(hf_split, sampling="random", seed=0, output_format="THWC")
        b = _dataset(hf_split, sampling="random", seed=1, output_format="THWC")
        # Over several long segments at least one draw must differ.
        differ = any(
            not torch.equal(a[i]["pixel_values"], b[i]["pixel_values"])
            for i in range(min(30, len(a)))
        )
        assert differ


class TestSegmentBounds:
    def test_segment_start_is_inside_the_file(self, hf_split) -> None:
        """A segment must at least begin before the video ends.

        Note the asymmetry: ``end`` is deliberately *not* required to fall
        inside the file. Some published annotations overrun the video by a
        fraction of a second, so the decoder has to clamp rather than fail --- a
        segment whose ``start`` is past the end, by contrast, would be
        unrecoverable and should never occur.
        """
        from omnifall._decode import probe

        for idx in range(min(25, len(hf_split))):
            row = hf_split[idx]
            meta = probe(row["video"])
            assert row["end"] > row["start"]
            assert row["start"] < meta.duration, (
                f"segment starts at {row['start']}s but {row['video']} is only "
                f"{meta.duration}s long"
            )

    def test_decoding_tolerates_annotations_overrunning_the_file(
        self, hf_split
    ) -> None:
        """Clamp, do not crash, when ``end`` is past the last frame.

        Real OmniFall annotations do overrun --- le2i ``Coffee_room_01/video_46``
        is labelled to 11.818s in an 11.68s file. Requesting frames past the end
        must still yield exactly ``num_frames``.
        """
        from omnifall._decode import decode_segment
        from omnifall._decode import probe

        row = hf_split[0]
        meta = probe(row["video"])
        frames = decode_segment(
            row["video"],
            start=max(0.0, meta.duration - 0.3),
            end=meta.duration + 5.0,
            num_frames=8,
            target_fps=15.0,
            sampling="uniform",
        )
        assert frames.shape[0] == 8


@pytest.mark.slow
class TestAnnotationOverrun:
    """Documents a property of the published data, not of this package.

    In the multi-view components, views are annotated once and propagated across
    a per-camera synchronisation offset, so a segment can end past the last frame
    of its video:

        mcfd      85/1352  (6.3%), worst 1.19s   chute05/cam8
        edf       18/508   (3.5%), worst 7.36s   jinhui/view2/rgb/...
        cmdfall   76/42143 (0.2%), worst 3.80s   colors/S22P36K6

    Overall 209 of 52,618 staged and in-the-wild segments (0.4%) overrun. The
    decoder must clamp to the available footage; a strict bounds check would
    reject real data.
    """

    def test_overrun_stays_rare(self, omnifall_root) -> None:
        from omnifall._decode import probe

        ds = omnifall.load("labels", split="train", video=True, strict=True)
        sample = ds.shuffle(seed=0).select(range(2000))

        durations: dict[str, float] = {}
        overrun = 0
        for row in sample:
            path = row["video"]
            if path not in durations:
                durations[path] = probe(path).duration
            if row["end"] - durations[path] > 0.04:
                overrun += 1

        # A sharp rise would mean the annotations or the video files changed.
        assert overrun / len(sample) < 0.03, (
            f"{overrun}/{len(sample)} segments overrun their video; "
            "expected well under 3%"
        )


class TestTransformReassignment:
    """Swapping the transform must not leave derived state stale.

    Two things are derived from the transform object: whether it accepts an
    ``rng`` keyword, and which layout it declares. If the ``rng`` flag went
    stale, a seeded generator would silently stop reaching the transform and
    augmentation would quietly become irreproducible --- with nothing in the
    output to reveal it.
    """

    def test_rng_flag_follows_assignment(self, hf_split) -> None:
        def takes_rng(frames, rng=None):
            return {"pixel_values": torch.zeros(4, 3, 8, 8)}

        ds = _dataset(hf_split, transform=None)
        assert ds._transform_takes_rng is False
        ds.transform = takes_rng
        assert ds._transform_takes_rng is True
        ds.transform = None
        assert ds._transform_takes_rng is False

    def test_declared_layout_follows_assignment(self, hf_split) -> None:
        ds = _dataset(hf_split, transform=None)
        assert ds._transform_layout is None
        ds.transform = omnifall.VideoTransform(mode="val")
        assert ds._transform_layout == "TCHW"

    def test_reassigned_transform_still_gets_the_seeded_rng(self, hf_split) -> None:
        seen: list[object] = []

        def takes_rng(frames, rng=None):
            seen.append(rng)
            return {"pixel_values": torch.zeros(len(frames), 3, 8, 8)}

        ds = _dataset(hf_split, transform=None, sampling="random", seed=0)
        ds.transform = takes_rng
        ds[0]
        assert seen and seen[0] is not None, "seeded rng did not reach the transform"


class TestMetadataPassthrough:
    def test_falllda_compatible_keys(self, hf_split) -> None:
        ds = _dataset(hf_split, output_format="THWC")
        item = ds[0]
        for key in (
            "label",
            "label_str",
            "video_path",
            "start_time",
            "end_time",
            "segment_duration",
            "dataset",
        ):
            assert key in item, key

    def test_relative_and_absolute_paths_are_distinct_keys(self, hf_split) -> None:
        ds = _dataset(hf_split, output_format="THWC")
        item = ds[0]
        assert not item["video_path"].startswith("/")
        assert item["video_file"].startswith("/")
        assert item["video_file"].endswith(item["video_path"] + ".mp4")

    def test_label_str_matches_label(self, hf_split) -> None:
        ds = _dataset(hf_split, output_format="THWC")
        item = ds[0]
        assert item["label_str"] == omnifall.IDX2LABEL[item["label"]]


class TestErrorHandling:
    def test_missing_video_raises_by_default(self, hf_split) -> None:
        broken = hf_split.map(lambda _: {"video": None})
        ds = _dataset(broken, output_format="THWC")
        with pytest.raises(Exception) as exc:
            ds[0]
        assert "prepare" in str(exc.value).lower() or "omnifall_root" in str(
            exc.value
        ).lower()

    def test_decode_error_is_not_silently_substituted(self, hf_split, tmp_path) -> None:
        """``on_error="raise"`` is the default, and it must actually raise.

        The old implementation swapped in a random other sample after a decode
        failure, so a corrupt file quietly turned into duplicated data.
        """
        bad = tmp_path / "not-a-video.mp4"
        bad.write_bytes(b"definitely not a video")
        broken = hf_split.select(range(1)).map(lambda _: {"video": str(bad)})
        ds = _dataset(broken, output_format="THWC")
        with pytest.raises(omnifall.VideoDecodeError):
            ds[0]


class TestCollate:
    def test_batches_are_model_shaped(self, hf_split) -> None:
        from torch.utils.data import DataLoader

        ds = _dataset(hf_split, transform=omnifall.VideoTransform(mode="val"))
        loader = DataLoader(ds, batch_size=4, collate_fn=omnifall.collate_fn)
        batch = next(iter(loader))
        assert batch["pixel_values"].shape == (4, 8, 3, 224, 224)
        assert batch["labels"].shape == (4,)
        assert batch["labels"].dtype == torch.long

    @pytest.mark.slow
    def test_multiworker_loading(self, hf_split) -> None:
        from torch.utils.data import DataLoader

        ds = _dataset(hf_split, transform=omnifall.VideoTransform(mode="val"))
        loader = DataLoader(
            ds, batch_size=8, collate_fn=omnifall.collate_fn, num_workers=4
        )
        seen = 0
        for batch in loader:
            seen += batch["pixel_values"].shape[0]
        assert seen == len(ds)


class TestMultiDataset:
    @pytest.mark.parametrize("sizes", [[2, 3, 1], [1, 1], [5], [3, 1, 4, 1]])
    def test_index_mapping_is_exact(self, sizes: list[int]) -> None:
        """Every global index must map to the right (dataset, local index).

        The boundary indices are what a ``bisect`` off-by-one gets wrong, so
        check all of them rather than a sample.
        """
        from omnifall import MultiOmniFallDataset

        class _Stub:
            def __init__(self, tag: int, n: int) -> None:
                self.tag, self.n = tag, n
                self.dataset_name = f"d{tag}"

            def __len__(self) -> int:
                return self.n

            def __getitem__(self, i: int) -> dict:
                assert 0 <= i < self.n, f"local index {i} out of range for d{self.tag}"
                return {"tag": self.tag, "local": i}

            @property
            def targets(self):
                return torch.zeros(self.n, dtype=torch.long)

        multi = MultiOmniFallDataset([_Stub(t, n) for t, n in enumerate(sizes)])
        assert len(multi) == sum(sizes)

        expected = [(t, i) for t, n in enumerate(sizes) for i in range(n)]
        got = [(multi[g]["tag"], multi[g]["local"]) for g in range(sum(sizes))]
        assert got == expected
