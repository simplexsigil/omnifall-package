"""Where the videos of each OmniFall component actually come from.

OmniFall publishes *annotations* on the HuggingFace Hub, but the *videos* of
eight of its ten components belong to the original authors and live on their
sites. This module is the declarative record of that: one :class:`Source` per
component, saying whether the videos can be fetched automatically, from where,
under which licence, and --- when they cannot --- exactly what the user has to
do by hand.

Every entry is followable either way. Nine of the ten can be fetched
unattended; the tenth, cmdfall, cannot and never will be, because access to it
is granted by e-mail. For all ten, :attr:`Source.archive_names` (or
:attr:`Source.file_pattern`, where there are too many to list) names the files
a user who downloads by hand has to produce, and ``omnifall sources`` prints
those names next to the directory to put them in. Dropping them there makes
``omnifall prepare`` succeed without touching the network.

The module contains **data only**. Every piece of behaviour lives in
:mod:`omnifall._prepare`.

Verification policy
-------------------
Every ``url`` in :data:`SOURCES` was probed with a real HTTP request while this
module was written; the observed status code and byte count are recorded in the
comment above the entry. A source whose download could not be reached
programmatically is recorded as ``kind="manual"`` with ``url=None`` and prose
instructions --- never as a guessed URL.

"Could not be reached" means a GET was tried and failed, and the comment says
what the failure was. Two entries here were once marked manual on weaker
evidence than that, and both turned out to be automatable:

* caucafall was probed with HEAD. Its Mendeley download redirects to a
  presigned S3 URL, and a presigned URL is signed for one method --- HEAD
  against it answers 403 however healthy the source is.
* mcfd was probed with a browser User-Agent. The site's Anubis bot check
  challenges browsers specifically, so the browser-shaped request got the
  challenge page and an honest ``python-requests`` one got the zip.

Both are recorded in full above their entries. The lesson they share is worth
stating once: a negative result from a probe that differs from the real
download is not evidence about the download.

Nothing downloadable is left as ``manual`` here. Only cmdfall is, and that is
not a probing question --- access is granted per e-mail request, so there is no
URL to reach.

The :attr:`Source.notes` describing each original layout are not guesses either.
They were read off unpacked copies of the releases themselves, and the
converters in :mod:`omnifall._prepare` were run against those copies and their
output compared frame by frame with the published OmniFall videos. Where a
release turned out to differ from what its own paper says --- MCFD's 120 fps
headers, UP-Fall's PNG rather than JPEG frames --- the note records what is
actually on disk.

Citations
---------
The BibTeX here is copied verbatim from the ``## Citation`` section of the
OmniFall Hub README, i.e. from the OmniFall authors themselves, keys included.
The only edit is that the README ends several entries with ``},`` rather than
``}``, which is not valid BibTeX; those stray commas are stripped. No entry was
reconstructed from memory, and none was reformatted --- pasting this package's
output next to the Hub's own block yields no diff.

Licences
--------
The Hub README assigns a licence to OmniFall's *annotations* only; on the
component videos it says just that they "belong to their respective owners".
A :attr:`Source.license` therefore names a licence only where one was read at
the source itself (a Zenodo record, a Mendeley record, the GitHub repository
metadata, the download page). Everywhere else it says ``"see homepage"``.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._constants import HF_REPO_ID, OOPS_URL, SYN_VIDEO_ARCHIVE

__all__ = [
    "Source",
    "SOURCES",
    "OMNIFALL_CITATION",
    "ANNOTATION_LICENSE",
    "VIDEO_LICENSE_NOTICE",
    "ANNOTATION_LICENSE_CANDIDATES",
    "ANNOTATION_LICENSE_CONFLICT",
    "get_source",
    "automatable_datasets",
    "manual_datasets",
]


#: Licence of everything OmniFall itself publishes: the annotations and the
#: split definitions.
#:
#: The Hub repository currently states this two different ways, and the
#: difference is substantive --- ShareAlike imposes an obligation that plain NC
#: does not:
#:
#: * the machine-readable card metadata says ``cc-by-nc-4.0``. This is what the
#:   dataset page displays, what the search filters use, and what any tool
#:   reading ``HfApi().repo_info(...).cardData`` gets.
#: * the README badge and its "## License" prose both say CC BY-NC-SA 4.0.
#:
#: This package refuses to pick a winner, because getting a licence wrong in a
#: redistribution tool is worse than admitting the ambiguity. Both readings are
#: reported, and :data:`ANNOTATION_LICENSE_CONFLICT` is what the CLI prints.
ANNOTATION_LICENSE = "CC BY-NC-SA 4.0 (README prose) / cc-by-nc-4.0 (card metadata)"

#: The two conflicting statements, for tools that want to inspect them.
ANNOTATION_LICENSE_CANDIDATES: tuple[str, ...] = ("CC BY-NC-SA 4.0", "CC BY-NC 4.0")

ANNOTATION_LICENSE_CONFLICT = (
    "NOTE: the OmniFall Hub repository states its annotation licence "
    "inconsistently. The card metadata says 'cc-by-nc-4.0' while the README "
    "badge and License section say 'CC BY-NC-SA 4.0'. The ShareAlike clause is "
    "the difference. Confirm with the dataset authors before relying on either."
)

#: What the Hub README says about the videos, quoted verbatim. It deliberately
#: does not assign a licence per component, which is why several entries below
#: record ``license="see homepage"``.
VIDEO_LICENSE_NOTICE = (
    "The annotations and split definitions are released under CC BY-NC-SA 4.0. "
    "The original video data belongs to their respective owners and should be "
    "obtained from the original sources."
)


@dataclass(frozen=True)
class Source:
    """Where and how the videos of one component dataset can be obtained.

    Attributes:
        dataset: The component's spelling in the ``dataset`` column of every
            OmniFall config, and its directory name under ``OMNIFALL_ROOT``.
        kind: How the source is reached. ``"http"`` for a plain URL,
            ``"hf"`` for a file in the OmniFall Hub repository, ``"zenodo"``,
            ``"mendeley"``, ``"kaggle"`` and ``"gdrive"`` for those hosts, and
            ``"manual"`` when no automated route exists.
        url: The download URL, or ``None`` for ``kind="manual"``. For
            ``kind="hf"`` this is the repository id, and :attr:`files` names the
            files within it. Where :attr:`url_template` is set this is the index
            or landing page instead, and is not itself downloaded.
        files: Specific archive names, where the source offers several. These
            are also the names a manually-downloaded copy must carry in the
            download directory, which is why they are the server's own
            spellings rather than tidied-up ones.
        url_template: For a source served as many files from one place: the URL
            of a single file, with a ``{file}`` placeholder taking a value from
            :attr:`files`. ``None`` where :attr:`url` is the whole download.
        file_bytes: Sizes of :attr:`files`, in the same order, as reported by
            the server. Empty where they were not measured.
        file_pattern: How the archives are named, for a source with too many
            files to list one by one. Prose, e.g.
            ``"Subject{s}Activity{a}Trial{t}Camera{c}.zip"``.
        file_count: How many archives the source is served as, where
            :attr:`files` is too long to enumerate. ``None`` otherwise, in
            which case ``len(files)`` is the count.
        approx_bytes: Size of the download in bytes, as reported by the server.
            For a multi-file source this is the sum over :attr:`files`.
            ``None`` when the server does not report one.
        archive_format: ``"tar"``, ``"tar.gz"`` or ``"zip"``; ``None`` when the
            source is not a single archive.
        gated: Whether a form, an e-mail request or a login stands between the
            user and the data. Gated sources are never fetched automatically.
        instructions: What the user must do by hand. Shown by
            ``omnifall sources`` and raised inside
            :class:`omnifall._prepare.DatasetNotAvailableError`.
        license: Licence of the original videos, where it could be verified at
            the source. ``"see homepage"`` where it could not --- asserting an
            unverified licence in a tool that helps people redistribute data is
            worse than admitting ignorance.
        terms: The conditions of use, quoted verbatim from the source's own
            page. For several components this is the *only* statement there is:
            no licence is granted, only a request to cite. Paraphrasing that
            would misrepresent what the authors asked for, so it is reproduced
            exactly, with the page it came from.
        citation: BibTeX entry for the originating paper.
        homepage: Landing page of the original dataset.
        notes: Anything a user needs to know that is not an instruction ---
            typically how the original tree differs from the OmniFall tree.
        description: One line about the component, taken from the dataset table
            in the OmniFall Hub README.
    """

    dataset: str
    kind: str
    url: str | None
    files: tuple[str, ...] = ()
    url_template: str | None = None
    file_bytes: tuple[int, ...] = ()
    file_pattern: str | None = None
    file_count: int | None = None
    approx_bytes: int | None = None
    archive_format: str | None = None
    gated: bool = False
    instructions: str = ""
    license: str = ""
    terms: str = ""
    citation: str = ""
    homepage: str = ""
    notes: str = ""
    description: str = ""

    @property
    def automatable(self) -> bool:
        """Whether :mod:`omnifall._prepare` may fetch this source unattended."""
        return self.kind != "manual" and not self.gated and self.url is not None

    @property
    def n_archives(self) -> int:
        """How many separate archives the release is served as."""
        if self.file_count is not None:
            return self.file_count
        return len(self.files) or 1

    def archive_names(self) -> tuple[str, ...]:
        """Return the file names the download directory is expected to hold.

        These are the names :mod:`omnifall._prepare` writes when it downloads,
        and the names it looks for when it does not. A user placing a file by
        hand has to match one of them, so ``omnifall sources`` prints exactly
        this.

        :attr:`files` may carry a path rather than a bare name where the source
        needs one --- of-syn's entry is a Hub repository path --- so the
        directory part is dropped here. What lands in a download directory is a
        file, not a tree.

        Returns:
            The archive names, or an empty tuple where the set is discovered at
            run time rather than recorded here --- in which case
            :attr:`file_pattern` describes their shape instead.
        """
        return tuple(name.rsplit("/", 1)[-1] for name in self.files)

    def file_url(self, name: str) -> str:
        """Return the URL one archive of a multi-file source is served at.

        Where :attr:`files` records the whole set --- mcfd's 24 scenarios ---
        membership is checked, because a name outside it would be a guessed
        URL. Where it is empty the keys are not names but handles discovered at
        run time (up_fall's Google Drive ids, read off the HAR-UP page), and
        there is nothing here to check them against; the discovery step in
        :mod:`omnifall._prepare` is what validates those, by requiring the set
        it finds to equal ``labels/up_fall.csv`` exactly.

        Args:
            name: One of :attr:`files`, or a run-time handle for a source whose
                file set is discovered rather than recorded.

        Returns:
            The URL to download.

        Raises:
            ValueError: If this source is not served as separate files, or
                *name* is outside a recorded :attr:`files`.
        """
        if self.url_template is None:
            raise ValueError(
                f"{self.dataset} is not served as separate files; its whole "
                f"download is {self.url}"
            )
        if self.files and name not in self.files:
            raise ValueError(
                f"{name!r} is not one of the {len(self.files)} archives "
                f"recorded for {self.dataset}"
            )
        return self.url_template.format(file=name)

    def bytes_of(self, name: str) -> int | None:
        """Return the recorded size of one archive, or ``None`` if unmeasured.

        Args:
            name: One of :attr:`files`.

        Returns:
            The byte count the server reported, or ``None``.
        """
        if not self.file_bytes or name not in self.files:
            return None
        return self.file_bytes[self.files.index(name)]


#: The OmniFall paper itself. Every user of any component must cite this.
# ---------------------------------------------------------------------------
# BibTeX, copied verbatim from the "## Citation" section of the OmniFall Hub
# README (hf/README.md). The only edit is that the README ends several entries
# with "}," rather than "}", which is not valid BibTeX; those stray trailing
# commas are stripped. Nothing else -- no author, title, year or key -- is
# altered, so pasting this next to the Hub's own block produces no diff and no
# duplicate-key clash.
#
# There is no separate entry for of-syn: the synthetic component is OmniFall's
# own contribution and is covered by the omnifall entry. edf and occu share a
# single entry, as they share a paper.
# ---------------------------------------------------------------------------

OMNIFALL_CITATION = r"""@misc{omnifall,
      title={OmniFall: From Staged Through Synthetic to Wild, A Unified Multi-Domain Dataset for Robust Fall Detection},
      author={David Schneider and Zdravko Marinov and Rafael Baur and Zeyun Zhong and Rodi Düger and Rainer Stiefelhagen},
      year={2025},
      eprint={2505.19889},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.19889},
}"""

_CITE_CMDFALL = r"""@inproceedings{omnifall_cmdfall,
  title={A multi-modal multi-view dataset for human fall analysis and preliminary investigation on modality},
  author={Tran, Thanh-Hai and Le, Thi-Lan and Pham, Dinh-Tan and Hoang, Van-Nam and Khong, Van-Minh and Tran, Quoc-Toan and Nguyen, Thai-Son and Pham, Cuong},
  booktitle={2018 24th International Conference on Pattern Recognition (ICPR)},
  pages={1947--1952},
  year={2018},
  organization={IEEE}
}"""

_CITE_UP_FALL = r"""@article{omnifall_up-fall,
  title={UP-fall detection dataset: A multimodal approach},
  author={Mart{\'\i}nez-Villase{\~n}or, Lourdes and Ponce, Hiram and Brieva, Jorge and Moya-Albor, Ernesto and N{\'u}{\~n}ez-Mart{\'\i}nez, Jos{\'e} and Pe{\~n}afort-Asturiano, Carlos},
  journal={Sensors},
  volume={19},
  number={9},
  pages={1988},
  year={2019},
  publisher={MDPI}
}"""

_CITE_LE2I = r"""@article{omnifall_le2i,
  title={Optimized spatio-temporal descriptors for real-time fall detection: comparison of support vector machine and Adaboost-based classification},
  author={Charfi, Imen and Miteran, Johel and Dubois, Julien and Atri, Mohamed and Tourki, Rached},
  journal={Journal of Electronic Imaging},
  volume={22},
  number={4},
  pages={041106--041106},
  year={2013},
  publisher={Society of Photo-Optical Instrumentation Engineers}
}"""

_CITE_GMDCSA = r"""@article{omnifall_gmdcsa,
  title={GMDCSA-24: A dataset for human fall detection in videos},
  author={Alam, Ekram and Sufian, Abu and Dutta, Paramartha and Leo, Marco and Hameed, Ibrahim A},
  journal={Data in Brief},
  volume={57},
  pages={110892},
  year={2024},
  publisher={Elsevier}
}"""

_CITE_CAUCA = r"""@article{omnifall_cauca,
  title={Dataset CAUCAFall},
  author={Eraso, Jose Camilo and Mu{\~n}oz, Elena and Mu{\~n}oz, Mariela and Pinto, Jesus},
  journal={Mendeley Data},
  volume={4},
  year={2022}
}"""

_CITE_EDF_OCCU = r"""@inproceedings{omnifall_edf_occu,
  title={Evaluating depth-based computer vision methods for fall detection under occlusions},
  author={Zhang, Zhong and Conly, Christopher and Athitsos, Vassilis},
  booktitle={International symposium on visual computing},
  pages={196--207},
  year={2014},
  organization={Springer}
}"""

_CITE_MCFD = r"""@article{omnifall_mcfd,
  title={Multiple cameras fall dataset},
  author={Auvinet, Edouard and Rougier, Caroline and Meunier, Jean and St-Arnaud, Alain and Rousseau, Jacqueline},
  journal={DIRO-Universit{\'e} de Montr{\'e}al, Tech. Rep},
  volume={1350},
  pages={24},
  year={2010}
}"""

_CITE_OOPS = r"""@inproceedings{omnifall_oops,
  title={Oops! predicting unintentional action in video},
  author={Epstein, Dave and Chen, Boyuan and Vondrick, Carl},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={919--929},
  year={2020}
}"""


#: One entry per component dataset, keyed by the ``dataset`` column value.
SOURCES: dict[str, Source] = {
    # -- of-syn ------------------------------------------------------------
    # VERIFIED: the archive is listed by the Hub API and is 9,716,480,000 B.
    # Members are "./{label}/{stem}.mp4", i.e. the OmniFall tree with a "./"
    # prefix, so extraction alone produces the required layout.
    "of-syn": Source(
        dataset="of-syn",
        kind="hf",
        url=HF_REPO_ID,
        files=(SYN_VIDEO_ARCHIVE,),
        approx_bytes=9_716_480_000,
        archive_format="tar",
        # of-syn is OmniFall's own data, so its licence IS the annotation
        # licence -- which is exactly the one the repository states two ways.
        # Naming a single side here would reintroduce the assertion that
        # ANNOTATION_LICENSE_CONFLICT exists to avoid.
        license=ANNOTATION_LICENSE,
        description=(
            "12,000 diffusion-generated videos with demographic diversity; "
            "single view, 19,228 segments, 16.88h."
        ),
        citation=OMNIFALL_CITATION,
        homepage=f"https://huggingface.co/datasets/{HF_REPO_ID}",
        notes=(
            "The 12,000 synthetic videos are the only videos OmniFall hosts "
            "itself. AV1-encoded; decoding needs a recent libdav1d/PyAV."
        ),
    ),
    # -- OOPS --------------------------------------------------------------
    # VERIFIED: HEAD -> HTTP 200, content-length 47,904,996,151 (44.6 GiB).
    # Only ~2.6 GB of that is needed, so the archive is streamed rather than
    # stored; see omnifall._oops.
    "OOPS": Source(
        dataset="OOPS",
        kind="http",
        url=OOPS_URL,
        files=("video_and_anns.tar.gz",),
        approx_bytes=47_904_996_151,
        archive_format="tar.gz",
        # Verified: this is the notice the OOPS site itself carries, quoted
        # verbatim in _constants.OOPS_LICENSE_TEXT.
        license="CC BY-NC-SA 4.0",
        description=(
            "Genuine accidents from the OOPS dataset; single view, 818 "
            "videos, 4,022 segments, 2.65h."
        ),
        terms=(
            # Quoted from https://oops.cs.columbia.edu/data/ . Note the first
            # clause: the CC grant covers the authors' work, NOT the underlying
            # footage, which is third-party. That is why omnifall streams and
            # extracts OOPS rather than mirroring it.
            "By pressing any of the links above, you acknowledge that we do "
            "not own the copyright to these videos and that they are solely "
            "provided for non-commercial research and/or educational "
            "purposes. This dataset is licensed under a Creative Commons "
            "Attribution-NonCommercial-ShareAlike 4.0 International License."
        ),
        citation=_CITE_OOPS,
        homepage="https://oops.cs.columbia.edu/data/",
        notes=(
            "The 818 OF-ItW videos are a subset of the OOPS release. The "
            "45 GB archive is streamed and only the needed members are "
            "written to disk, so no 45 GB of scratch space is required. "
            "File-name mapping comes from the Hub file "
            "data_files/oops_video_mapping.csv."
        ),
    ),
    # -- GMDCSA24 ----------------------------------------------------------
    # VERIFIED: codeload tarball -> HTTP 200 (default branch is "master", not
    # "main"). The GitHub tree API lists exactly 160 .mp4 blobs, 1,109,791,938 B
    # in total, and their paths map onto the 160 required OmniFall paths
    # bijectively ("Subject N/ADL/01.mp4" -> "Subject_N/ADL/01"). Checked:
    # 160/160 match, no missing, no extras.
    # codeload generates the tarball on the fly and sends no Content-Length and
    # no Range support, so this download cannot be resumed.
    #
    # approx_bytes here is the sum of the 160 blobs, NOT the size of the
    # tarball, and file_bytes is deliberately left empty because of it. The two
    # differ: a measured download came to 1,107,243,445 B against the
    # 1,109,791,938 B of blob content. Nor would pinning that measurement be
    # right -- codeload compresses on demand, so the exact byte count is the
    # server's to vary. This is why _acquire_archives takes expected_bytes from
    # file_bytes alone and never from approx_bytes: one is a measurement of the
    # thing being downloaded, the other is a figure to show the user.
    "GMDCSA24": Source(
        dataset="GMDCSA24",
        kind="http",
        url=(
            "https://codeload.github.com/ekramalam/"
            "GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos/"
            "tar.gz/refs/heads/master"
        ),
        # codeload sends no file name of its own, so this is the name omnifall
        # writes and the name it looks for. A tarball downloaded from GitHub's
        # own "Download ZIP"/tarball link arrives close enough to it that
        # renaming is the only step.
        files=("GMDCSA24-master.tar.gz",),
        approx_bytes=1_109_791_938,
        archive_format="tar.gz",
        # Verified: the GitHub API reports spdx_id "MIT" for the repository
        # that holds the videos.
        license="MIT (repository licence)",
        description=(
            "Single view, 160 videos, 458 segments, 0.36h, 2.80s average "
            "segment."
        ),
        citation=_CITE_GMDCSA,
        homepage=(
            "https://github.com/ekramalam/"
            "GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos"
        ),
        notes=(
            "The videos are committed to the repository itself. Conversion is "
            "a rename only: the source spells subject directories "
            "'Subject 1' with a space, OmniFall spells them 'Subject_1'."
        ),
    ),
    # -- edf ---------------------------------------------------------------
    # VERIFIED: Zenodo record 15494102, "EDF and OCCU Fall Detection Datasets".
    # Range request -> HTTP 206, content-range ".../16030813960". Resumable.
    # This DOI is the one the OmniFall authors link from the Hub README.
    "edf": Source(
        dataset="edf",
        kind="zenodo",
        url="https://zenodo.org/api/records/15494102/files/EDF.zip/content",
        files=("EDF.zip",),
        file_bytes=(16_030_813_960,),
        approx_bytes=16_030_813_960,
        archive_format="zip",
        # Verified: the Zenodo record reports CC BY 4.0.
        license="CC BY 4.0",
        description=(
            "Multi-view (2 views), 10 videos, 254 segments, 0.22h, 3.14s "
            "average segment."
        ),
        citation=_CITE_EDF_OCCU,
        homepage="https://doi.org/10.5281/zenodo.15494102",
        notes=(
            "16 GB because the release carries depth and skeleton data "
            "alongside RGB, and a third camera view. OmniFall needs only the "
            "10 RGB sequences, which unpack to {subject}/{subject}/view{1,2}/"
            "rgb/ as one .bin file per frame: a four-uint16 header followed by "
            "planar 8-bit RGB. omnifall decodes them and encodes 30 fps MP4s "
            "with ffmpeg, which is therefore required."
        ),
    ),
    # -- occu --------------------------------------------------------------
    # VERIFIED: same Zenodo record; Range -> HTTP 206, ".../10831118897".
    "occu": Source(
        dataset="occu",
        kind="zenodo",
        url="https://zenodo.org/api/records/15494102/files/OCCU.zip/content",
        files=("OCCU.zip",),
        file_bytes=(10_831_118_897,),
        approx_bytes=10_831_118_897,
        archive_format="zip",
        # Verified: the same Zenodo record reports CC BY 4.0.
        license="CC BY 4.0",
        description=(
            "Multi-view (2 views), 10 videos, 245 segments, 0.25h, 3.54s "
            "average segment."
        ),
        citation=_CITE_EDF_OCCU,
        homepage="https://doi.org/10.5281/zenodo.15494102",
        notes=(
            "Shares a Zenodo record, a paper and a file format with edf. "
            "OmniFall needs the 10 RGB sequences, which unpack to "
            "{subject}/{subject}/view{1,2}/rgb/ as per-frame .bin dumps."
        ),
    ),
    # -- le2i --------------------------------------------------------------
    # VERIFIED: dl_data.php?file=101 -> HTTP 303 to a presigned S3 URL for
    # FallDataset.zip, which answers a Range request with HTTP 206 and
    # content-range ".../9608701260". Resumable.
    # The presigned URL expires after 300 s, so every request (including every
    # resume) must start from the dl_data.php URL and follow the redirect.
    "le2i": Source(
        dataset="le2i",
        kind="http",
        url="https://search-data.ubfc.fr/imvia/dl_data.php?file=101",
        files=("FallDataset.zip",),
        file_bytes=(9_608_701_260,),
        approx_bytes=9_608_701_260,
        archive_format="zip",
        # Verified: the ubfc landing page links creativecommons.org/
        # licenses/by-nc-sa/3.0/.
        license="CC BY-NC-SA 3.0",
        description=(
            "Single view, 190 videos, 967 segments, 0.79h, 2.95s average "
            "segment."
        ),
        citation=_CITE_LE2I,
        homepage=(
            "https://search-data.ubfc.fr/imvia/"
            "FR-13002091000019-2024-04-09_Fall-Detection-Dataset.html"
        ),
        notes=(
            "DOI 10.25666/DATAUBFC-2024-04-09. The release ships AVI files "
            "grouped by scene (Coffee_room_01, Coffee_room_02, Home_01, "
            "Home_02, Lecture room, Office); OmniFall needs them transcoded "
            "to MP4 as {scene}/video_{n}.mp4, which needs ffmpeg. The naming "
            "does not survive that step untouched: the videos are called "
            "'video (7).avi', four of the six scenes nest them in a 'Videos' "
            "subdirectory and two do not, and the lecture-room directory is "
            "spelled with a space where OmniFall uses an underscore."
        ),
    ),
    # -- caucafall ---------------------------------------------------------
    # VERIFIED: the Mendeley bulk-zip endpoint answers a Range request with
    # HTTP 302 to a presigned S3 URL, which returns HTTP 206, content-type
    # application/zip, content-range ".../8326349365" and a body starting
    # "PK\x03\x04". Resumable. Content-disposition names the file
    # "Dataset CAUCAFall.zip", which is the name used below so that a copy
    # downloaded in a browser drops into the download directory unrenamed.
    #
    # An earlier pass recorded this source as unreachable. That conclusion came
    # from probing with HEAD: the S3 URL is presigned for GET, so HEAD against
    # it answers 403 no matter what. A GET works, and always did. The
    # per-folder listing endpoints really are browser-only, but the bulk zip is
    # the whole release anyway, so nothing is lost.
    #
    # Downloaded in full and checked: 40,209 members, of which exactly 100 are
    # the .avi files the converter needs, at
    # "Dataset CAUCAFall/CAUCAFall/Subject.N/<activity>/<Stem>.avi".
    "caucafall": Source(
        dataset="caucafall",
        kind="mendeley",
        url="https://data.mendeley.com/public-api/zip/7w7fccy7ky/download/4",
        files=("Dataset CAUCAFall.zip",),
        file_bytes=(8_326_349_365,),
        approx_bytes=8_326_349_365,
        archive_format="zip",
        # Verified: the Mendeley API reports data_licence.short_name
        # "CC BY 4.0" for this record.
        license="CC BY 4.0",
        description=(
            "Single view, 100 videos, 258 segments, 0.28h, 3.85s average "
            "segment."
        ),
        citation=_CITE_CAUCA,
        homepage="https://data.mendeley.com/datasets/7w7fccy7ky/4",
        instructions=(
            "'omnifall prepare caucafall' fetches this automatically.\n"
            "To download it yourself instead, open "
            "https://data.mendeley.com/datasets/7w7fccy7ky/4 in a browser and "
            "press 'Download all'. You get 'Dataset CAUCAFall.zip'."
        ),
        notes=(
            "The release is organised by subject (CAUCAFall/Subject.7/Fall "
            "left/) and each activity directory holds both the PNG frames and "
            "an AVI the authors assembled from them, named exactly like the "
            "OmniFall stem (FallLeftS7.avi). OmniFall transcodes that AVI, "
            "which needs ffmpeg. It is deliberately not built from the PNGs: "
            "the authors' AVI is one frame shorter than the frame sequence, "
            "and the published annotations are timed against the AVI. The 100 "
            "OmniFall videos are grouped as adl/ (50), side/ (30), "
            "backwards/ (10) and forward/ (10). 7.8 GiB is downloaded for "
            "0.28h of video because the archive carries the PNG frames and "
            "per-frame annotation text files as well. NOTE: "
            "_constants.DATASETS['caucafall'] currently points at Mendeley id "
            "7dwjhd7kfz; the OmniFall Hub README links 7w7fccy7ky/4, which is "
            "what this entry uses."
        ),
    ),
    # -- mcfd --------------------------------------------------------------
    # VERIFIED: all 24 archives probed with a Range request. Each answered
    # HTTP 206, content-type application/zip, body starting "PK\x03\x04";
    # the sizes below are their content-range totals and add up to
    # 3,782,812,240 B. The index page returns HTTP 200 with the title
    # "Multiple cameras fall dataset" and links exactly these 24 names.
    #
    # The site does sit behind an Anubis proof-of-work check, and an earlier
    # pass recorded this source as unreachable because of it. Anubis
    # challenges browsers, not tools: it keys off a "Mozilla" User-Agent.
    # Probed side by side, a request sent as "Mozilla/5.0" gets HTTP 200 with
    # a text/html challenge body where a zip was asked for, while the same
    # request sent as "curl/8.0.1", "Wget/1.21", "python-requests/2.31" or the
    # honest omnifall agent gets the zip. omnifall therefore does NOT spoof a
    # browser here -- doing so is what breaks it. Never set a Mozilla-like
    # User-Agent on this host.
    #
    # The challenge may still appear (the policy is the site's to change), so
    # _prepare checks the content type and the leading "PK" of every archive
    # and refuses to write a challenge page as a zip.
    "mcfd": Source(
        dataset="mcfd",
        kind="http",
        url="https://www.iro.umontreal.ca/~labimage/Dataset/",
        url_template=(
            "https://www.iro.umontreal.ca/~labimage/Dataset/chute-zip/{file}"
        ),
        files=tuple(f"chute{n:02d}.zip" for n in range(1, 25)),
        file_bytes=(
            179_574_453, 94_202_585, 108_043_138, 116_722_738,
            85_869_909, 153_944_790, 103_814_364, 79_060_375,
            106_705_518, 108_015_268, 90_414_078, 102_982_012,
            135_474_451, 164_473_313, 128_236_000, 143_141_531,
            182_512_862, 109_046_444, 116_006_266, 113_184_380,
            135_469_254, 129_637_064, 647_752_573, 448_528_874,
        ),
        approx_bytes=3_782_812_240,
        archive_format="zip",
        # Not verified: the download page states no licence and the Hub README
        # states none for the original videos.
        license="see homepage",
        description=(
            "Multi-view (8 views), 192 videos, 169 segments, 0.20h, 4.26s "
            "average segment."
        ),
        terms=(
            # Quoted from https://www.iro.umontreal.ca/~labimage/Dataset/ .
            # This is the page's ENTIRE statement of conditions: no licence is
            # granted and redistribution is not addressed, so the videos remain
            # all-rights-reserved to the authors.
            "If you use this dataset and publish your work, be sure to "
            "correctly reference our technical report as \"E. Auvinet, "
            "C. Rougier, J.Meunier, A. St-Arnaud, J. Rousseau, \"Multiple "
            "cameras fall dataset\", Technical report 1350, DIRO - "
            "Universite de Montreal, July 2010.\""
        ),
        citation=_CITE_MCFD,
        homepage="https://www.iro.umontreal.ca/~labimage/Dataset/",
        instructions=(
            "'omnifall prepare mcfd' fetches this automatically.\n"
            "To download it yourself instead, open "
            "https://www.iro.umontreal.ca/~labimage/Dataset/ in a browser and "
            "take all 24 scenario archives, chute01.zip ... chute24.zip."
        ),
        notes=(
            "The original release ships per-camera AVI files at "
            "dataset/chute01/cam1.avi, which is OmniFall's layout already; "
            "conversion needs ffmpeg only to re-encode them. Those AVIs carry "
            "a 120 fps header for footage captured at 30, so the conversion "
            "reinterprets the input rate rather than resampling. Anything "
            "that skips that step plays four times too fast and misaligns "
            "every segment boundary in the labels. The download is served as "
            "24 separate zips, one per scenario, so an interrupted run resumes "
            "at the archive it stopped on rather than at the start."
        ),
    ),
    # -- up_fall -----------------------------------------------------------
    # VERIFIED, and it needs no Google Drive API and no OAuth. HAR-UP's own
    # downloader uses PyDrive, but the page does not need it: every camera
    # archive is linked from https://sites.google.com/up.edu.mx/har-up/ as a
    # plain drive.google.com URL carrying the file id, and each such id is
    # served by https://drive.google.com/uc?id=...&export=download without a
    # confirm token.
    #
    # Probed: the page returns HTTP 200. Its body is HTML-escaped twice, so it
    # has to be unescaped twice before the anchors are readable. Doing that
    # and taking every Camera1/Camera2 anchor, attributed to the nearest
    # PRECEDING SubjectNActivityMTrialK marker, yields exactly 1,118 links --
    # measured against labels/up_fall.csv: 1,118 required, 0 missing, 0 extra,
    # 0 ids claimed by two paths. _prepare asserts that and refuses to
    # download a subset if the page ever changes shape.
    #
    # One archive was fetched end to end as a check:
    # Subject1Activity1Trial1Camera1, id 1x-gpsGcP1jMAvWAZ9oM1O7MsGW8VaCma ->
    # HTTP 200, content-disposition filename="Subject1Activity1Trial1Camera1
    # .zip", 99,660,843 B, body starting "PK\x03\x04", holding 195 timestamped
    # PNGs FLAT at the archive root (no directory prefix), which is why
    # _prepare extracts each archive into its own {path}/ directory.
    #
    # That archive downloads in one request, but not every archive does, and
    # assuming otherwise is a trap this entry exists to disarm. Drive serves a
    # file directly only while it is small; past roughly 100 MB it answers with
    # a "Virus scan warning" page carrying a form. Observed:
    # Subject2Activity10Trial1Camera1, id 19hOc1qujQGLHSUKIo9u1FXdzqL_PkPWf,
    # 232 MB -> HTTP 200, text/html, form posting id/export/confirm/uuid to
    # https://drive.usercontent.google.com/download. Following that form
    # yields the zip. The uuid is issued per request, so the page has to be
    # read each time rather than a URL being cached; see _drive_direct_url.
    # Still no account, API key or OAuth of any kind.
    #
    # approx_bytes is left None on purpose: only a handful of the 1,118
    # archives were measured, and multiplying one of them out would be a
    # guess dressed up as a figure. The notes give the honest order.
    "up_fall": Source(
        dataset="up_fall",
        kind="gdrive",
        url="https://sites.google.com/up.edu.mx/har-up/",
        url_template="https://drive.google.com/uc?id={file}&export=download",
        file_pattern="Subject{s}Activity{a}Trial{t}Camera{c}.zip",
        file_count=1118,
        approx_bytes=None,
        archive_format="zip",
        # Not verified; the Hub README states no licence for the original
        # videos.
        license="see homepage",
        description=(
            "Multi-view (2 views), 1,118 videos, 1,213 segments, 4.59h, "
            "13.63s average segment."
        ),
        terms=(
            # Quoted from https://sites.google.com/up.edu.mx/har-up/ .
            # As with mcfd, this is the whole statement: a citation request and
            # a copyright line, with no licence and no redistribution grant.
            "If you use this data set, please cite as follows: Lourdes "
            "Martinez-Villasenor, Hiram Ponce, Jorge Brieva, Ernesto "
            "Moya-Albor, Jose Nunez-Martinez, Carlos Penafort-Asturiano, "
            "\"UP-Fall Detection Dataset: A Multimodal Approach\", Sensors "
            "19(9), 1988: 2019, doi:10.3390/s19091988. "
            "-- Copyright 2017 - 2019. Universidad Panamericana."
        ),
        citation=_CITE_UP_FALL,
        homepage="https://sites.google.com/up.edu.mx/har-up/",
        instructions=(
            "'omnifall prepare up_fall' fetches this automatically: it reads "
            "the Google Drive link for each of the 1,118 camera archives off "
            "https://sites.google.com/up.edu.mx/har-up/ and downloads, "
            "converts and discards them one trial at a time. No Google "
            "account, API key or OAuth flow is needed. Expect it to take a "
            "long time and roughly 110 GB of transfer.\n"
            "To download them yourself instead, open that page in a browser "
            "and take the Camera1 and Camera2 links only -- NOT Camera1_OF or "
            "Camera2_OF (optical flow), 'Features' or 'DataSet'. Keep each "
            "file's own name, Subject{s}Activity{a}Trial{t}Camera{c}.zip."
        ),
        notes=(
            "The camera ZIPs contain PNG frame sequences, not videos, so "
            "conversion needs ffmpeg. Each frame is named for the moment it "
            "was captured, and those timestamps matter: the cameras recorded "
            "over a network at a rate that wanders either side of 18 fps, so "
            "OmniFall's videos are genuinely variable-rate, built from the "
            "real inter-frame gaps. A constant frame rate would drift against "
            "labels/up_fall.csv, whose boundaries are in seconds. Note that "
            "17 subjects x 11 activities x 3 trials x 2 cameras = 1,122, but "
            "four of those recordings are absent from the OmniFall labels; "
            "1,118 is the correct target.\n"
            "Each archive is around 100 MB, so the whole set is of the order "
            "of 110 GB -- but omnifall never needs that much scratch space at "
            "once. It downloads one trial, encodes its MP4, deletes the PNGs "
            "and the archive, and moves on, so an interrupted run leaves the "
            "trials it finished ready to use and resumes at the next one. "
            "Pass keep_archives=True (CLI: --keep-archives) to hold on to the "
            "downloaded zips, and budget the full 110 GB if you do.\n"
            "The bundle at Google Drive id 1JBGU5W2uq9rl8h7bJNt2lN4SjfZnFxmQ, "
            "which the HAR-UP page also links, is NOT the camera package: it "
            "is CompleteDataSet.csv, 31 MB, 118,748 rows of accelerometer, "
            "gyroscope, EEG and infrared readings. It contains no video and is "
            "of no use to OmniFall. It has been mistaken for the video "
            "download before, which is why it is named here."
        ),
    ),
    # -- cmdfall -----------------------------------------------------------
    # GATED BY DESIGN: access is granted per e-mail request by the MICA
    # institute. There is no URL to verify and there never will be one.
    "cmdfall": Source(
        dataset="cmdfall",
        kind="manual",
        url=None,
        approx_bytes=None,
        archive_format=None,
        gated=True,
        # Not verified: access is granted case by case and the Hub README
        # states no licence for the original videos.
        license="see homepage",
        description=(
            "Multi-view (7 views), 384 videos, 6,026 segments, 7.12h, "
            "4.25s average segment."
        ),
        terms=(
            # Quoted from
            # https://www.mica.edu.vn/perso/Tran-Thi-Thanh-Hai/CMDFALL.html .
            # "free for research purpose" is a permission to use, not to
            # redistribute, and access is by individual request.
            "Our CMDFALL dataset is available and free for research purpose. "
            "If you want to use our dataset, please contact "
            "thanh-hai.tran@mica.edu.vn."
        ),
        citation=_CITE_CMDFALL,
        homepage=(
            "https://www.mica.edu.vn/perso/Tran-Thi-Thanh-Hai/CMDFALL.html"
        ),
        instructions=(
            "CMDFall is not publicly downloadable, so omnifall cannot fetch "
            "it. Everything after that is the same as for any other "
            "component.\n"
            "1. Visit https://www.mica.edu.vn/perso/Tran-Thi-Thanh-Hai/"
            "CMDFALL.html and e-mail the MICA institute to request access, "
            "stating your institution and intended research use.\n"
            "2. When they grant access, download the RGB ('colors') "
            "recordings and unpack them, so that a 'colors' directory "
            "exists.\n"
            "3. Unpack the release into the directory named above, or leave "
            "it where it is and pass 'omnifall prepare cmdfall --archive "
            "<dir>'.\n"
            "OmniFall needs 384 videos named colors/S{s}P{p}K{k}.mp4, where "
            "K is the Kinect index."
        ),
        notes=(
            "OmniFall uses only the RGB stream of 384 of the recordings, even "
            "though the full CMDFall release has far more (7 views, plus "
            "depth, skeleton and per-clip cut-outs). They unpack to "
            "colors/S{s}P{p}K{k}.avi, which is OmniFall's layout already: "
            "20 fps MJPEG, re-encoded with ffmpeg. The full-range pixel "
            "format of the MJPEG sources is deliberately carried over rather "
            "than converted."
        ),
    ),
}


def get_source(dataset: str) -> Source:
    """Return the :class:`Source` for *dataset*.

    Args:
        dataset: A ``dataset`` column value, e.g. ``"le2i"`` or ``"OOPS"``.

    Returns:
        The registry entry for that component.

    Raises:
        KeyError: If *dataset* is not an OmniFall component. The message lists
            the valid spellings, which are case-sensitive.
    """
    try:
        return SOURCES[dataset]
    except KeyError:
        raise KeyError(
            f"unknown OmniFall component {dataset!r}; "
            f"expected one of {sorted(SOURCES)}"
        ) from None


def automatable_datasets() -> tuple[str, ...]:
    """Return the components whose videos this package can fetch unattended."""
    return tuple(n for n, s in SOURCES.items() if s.automatable)


def manual_datasets() -> tuple[str, ...]:
    """Return the components the user has to obtain by hand."""
    return tuple(n for n, s in SOURCES.items() if not s.automatable)
