"""Static facts about the OmniFall dataset.

Everything in this module is data, not behaviour. The single most important
design rule of this package lives here:

    Video files are addressed by the pair ``(dataset, path)``, never by the
    config name.

Every config published by the OmniFall Hub repository carries a ``dataset``
column, and every ``path`` value is relative to that dataset's own video root.
Resolution is therefore uniform across all 72 configs and needs no per-config
special casing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

HF_REPO_ID = "simplexsigil2/omnifall"

# ---------------------------------------------------------------------------
# Label space
# ---------------------------------------------------------------------------

#: The 16 activity classes shared by every OmniFall component, in label-id order.
ACTIVITY_LABELS: list[str] = [
    "walk",  # 0
    "fall",  # 1
    "fallen",  # 2
    "sit_down",  # 3
    "sitting",  # 4
    "lie_down",  # 5
    "lying",  # 6
    "stand_up",  # 7
    "standing",  # 8
    "other",  # 9
    "kneel_down",  # 10
    "kneeling",  # 11
    "squat_down",  # 12
    "squatting",  # 13
    "crawl",  # 14
    "jump",  # 15
]

#: Coarse three-way grouping used by several fall-detection benchmarks.
FALL_STATE_GROUPS: dict[str, tuple[str, ...]] = {
    "fall": ("fall",),
    "fallen": ("fallen",),
    "other": tuple(l for l in ACTIVITY_LABELS if l not in ("fall", "fallen")),
}

# ---------------------------------------------------------------------------
# Component datasets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetInfo:
    """Static description of one component dataset of OmniFall.

    Attributes:
        name: The exact string used in the ``dataset`` column of every config,
            and the directory name under ``OMNIFALL_ROOT``.
        kind: ``"staged"``, ``"itw"`` (in-the-wild) or ``"syn"`` (synthetic).
        n_videos: Number of distinct video files referenced by the labels.
        video_ext: File extension of the converted videos under
            ``{root}/{name}/video/``.
        obtainable: Whether the package can obtain the videos without the user
            manually requesting access from the original authors.
        license: License of the original videos.
        citation: BibTeX key / short citation of the originating paper.
        homepage: Landing page of the original dataset.
    """

    name: str
    kind: str
    n_videos: int
    video_ext: str = ".mp4"
    obtainable: bool = True
    license: str = "see homepage"
    citation: str = ""
    homepage: str = ""


#: Every component dataset of OmniFall, keyed by its ``dataset`` column value.
#:
#: ``n_videos`` is the number of distinct ``path`` values in ``labels/{name}.csv``
#: on the Hub, cross-checked against the dataset table in the Hub README. It is
#: used for reporting only, never for control flow --- the label file is always
#: the authority, so :func:`omnifall.verify` counts paths itself rather than
#: trusting this table.
#:
#: Note that ``n_videos`` counts videos OmniFall *annotates*, which can be far
#: fewer than the source dataset ships: cmdfall annotates 384 of its 1,437
#: files.
DATASETS: dict[str, DatasetInfo] = {
    "caucafall": DatasetInfo(
        name="caucafall",
        kind="staged",
        n_videos=100,
        homepage="https://data.mendeley.com/datasets/7w7fccy7ky/4",
    ),
    "cmdfall": DatasetInfo(
        name="cmdfall",
        kind="staged",
        n_videos=384,
        obtainable=False,
        homepage="https://www.mica.edu.vn/perso/Tran-Thi-Thanh-Hai/CMDFALL.html",
    ),
    "edf": DatasetInfo(
        name="edf",
        kind="staged",
        n_videos=10,
        license="CC BY 4.0",
        homepage="https://doi.org/10.5281/zenodo.15494102",
    ),
    "GMDCSA24": DatasetInfo(
        name="GMDCSA24",
        kind="staged",
        n_videos=160,
        homepage="https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos",
    ),
    "le2i": DatasetInfo(
        name="le2i",
        kind="staged",
        n_videos=190,
        homepage="https://search-data.ubfc.fr/imvia/FR-13002091000019-2024-04-09_Fall-Detection-Dataset.html",
    ),
    "mcfd": DatasetInfo(
        name="mcfd",
        kind="staged",
        n_videos=192,
        homepage="https://www.iro.umontreal.ca/~labimage/Dataset/",
    ),
    "occu": DatasetInfo(
        name="occu",
        kind="staged",
        n_videos=10,
        license="CC BY 4.0",
        homepage="https://doi.org/10.5281/zenodo.15494102",
    ),
    "up_fall": DatasetInfo(
        name="up_fall",
        kind="staged",
        n_videos=1118,
        homepage="https://sites.google.com/up.edu.mx/har-up/",
    ),
    "OOPS": DatasetInfo(
        name="OOPS",
        kind="itw",
        n_videos=818,
        license="CC BY-NC-SA 4.0",
        homepage="https://oops.cs.columbia.edu/data/",
    ),
    "of-syn": DatasetInfo(
        name="of-syn",
        kind="syn",
        n_videos=12000,
        homepage=f"https://huggingface.co/datasets/{HF_REPO_ID}",
    ),
}

#: Component datasets grouped by kind.
STAGED_DATASETS: tuple[str, ...] = tuple(
    n for n, i in DATASETS.items() if i.kind == "staged"
)
ITW_DATASETS: tuple[str, ...] = tuple(n for n, i in DATASETS.items() if i.kind == "itw")
SYN_DATASETS: tuple[str, ...] = tuple(n for n, i in DATASETS.items() if i.kind == "syn")

# ---------------------------------------------------------------------------
# Hub file names
# ---------------------------------------------------------------------------

#: OF-Syn video archive on the Hub (~9.7 GB, 12000 AV1-encoded MP4s).
SYN_VIDEO_ARCHIVE = "data_files/omnifall-synthetic_av1.tar"

#: Maps original OOPS file names to OF-ItW ``path`` values.
OOPS_MAPPING_FILE = "data_files/oops_video_mapping.csv"

#: OF-Syn frame-wise HDF5 labels.
SYN_FRAMEWISE_ARCHIVE = "data_files/syn_frame_wise_labels.tar.zst"

OOPS_URL = "https://oops.cs.columbia.edu/data/video_and_anns.tar.gz"

EXPECTED_OOPS_COUNT = 818
EXPECTED_SYN_COUNT = 12000

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------

#: Points at a directory laid out as ``{root}/{dataset}/video/{path}.mp4``.
ENV_ROOT = "OMNIFALL_ROOT"

#: Overrides the default cache location (``~/.cache/omnifall``).
ENV_CACHE = "OMNIFALL_CACHE_DIR"

#: ``OMNIFALL_VIDEO_ROOT__<dataset>`` overrides the root of a single dataset.
ENV_PER_DATASET_PREFIX = "OMNIFALL_VIDEO_ROOT__"

OOPS_LICENSE_TEXT = """\
==========================================================================
OOPS Dataset License Notice
==========================================================================

The OF-ItW component of OmniFall uses videos from the OOPS dataset.
The following notice is from the OOPS dataset website
(https://oops.cs.columbia.edu/data/):

  "By pressing any of the links above, you acknowledge that we do not
   own the copyright to these videos and that they are solely provided
   for non-commercial research and/or educational purposes. This dataset
   is licensed under a Creative Commons Attribution-NonCommercial-
   ShareAlike 4.0 International License."

If you use OF-ItW in your research, please also cite the OOPS paper:

  @inproceedings{epstein2020oops,
    title={Oops! predicting unintentional action in video},
    author={Epstein, Dave and Chen, Boyuan and Vondrick, Carl},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer
               Vision and Pattern Recognition},
    pages={919--929},
    year={2020}
  }

The download will stream ~45GB from the OOPS website and extract
%d videos (~2.6GB disk space).
=========================================================================="""
