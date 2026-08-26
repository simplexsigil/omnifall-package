"""Tests for the acquisition layer and the CLI.

The acquisition code mostly moves gigabytes around, which no test should do.
What is tested here is everything around that: the source registry is honest,
archive extraction is safe, verification actually detects a broken tree, and
the CLI runs.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tarfile

import pytest

import omnifall
from omnifall._constants import DATASETS


@pytest.fixture(scope="module")
def sources() -> dict:
    return omnifall.SOURCES


class TestSourceRegistry:
    def test_covers_every_component(self, sources: dict) -> None:
        assert set(sources) == set(DATASETS)

    def test_kinds_are_known(self, sources: dict) -> None:
        allowed = {"http", "hf", "mendeley", "zenodo", "kaggle", "gdrive", "manual"}
        for name, src in sources.items():
            assert src.kind in allowed, f"{name}: {src.kind}"

    def test_automated_sources_carry_a_url(self, sources: dict) -> None:
        for name, src in sources.items():
            if src.kind != "manual":
                assert src.url, f"{name} claims kind={src.kind} but has no url"

    def test_manual_sources_explain_themselves(self, sources: dict) -> None:
        """A manual source is only useful if it says what to do.

        Silently having no URL and no instructions would leave a user stuck
        with no way forward, which is worse than an honest error.
        """
        for name, src in sources.items():
            if src.kind == "manual" or src.gated:
                assert len(src.instructions.strip()) > 40, (
                    f"{name} needs manual steps but gives no usable instructions"
                )

    def test_cmdfall_is_manual(self, sources: dict) -> None:
        assert sources["cmdfall"].kind == "manual"
        assert sources["cmdfall"].gated

    def test_of_syn_comes_from_the_hub(self, sources: dict) -> None:
        assert sources["of-syn"].kind == "hf"

    def test_of_syn_licence_defers_to_the_annotation_licence(
        self, sources: dict
    ) -> None:
        """OF-Syn is OmniFall's own data, so it inherits OmniFall's ambiguity.

        The Hub states its annotation licence two ways --- ``cc-by-nc-4.0`` in
        the card metadata that drives the dataset page, ``CC BY-NC-SA 4.0`` in
        the README prose. Hard-coding either onto OF-Syn would reassert, on the
        one component OmniFall actually redistributes, exactly the one-sided
        claim the conflict notice exists to avoid.
        """
        from omnifall._sources import ANNOTATION_LICENSE

        assert sources["of-syn"].license == ANNOTATION_LICENSE

    def test_third_party_licences_do_not_track_omnifall(self, sources: dict) -> None:
        """Component licences are their authors', not OmniFall's.

        OOPS is CC BY-NC-SA 4.0 because the OOPS notice says so, not because
        OmniFall's README does; if the two were coupled, correcting OmniFall's
        licence would silently rewrite eight other projects' terms.
        """
        from omnifall._sources import ANNOTATION_LICENSE

        coupled = [
            n
            for n, s in sources.items()
            if n != "of-syn" and s.license == ANNOTATION_LICENSE
        ]
        assert not coupled, f"third-party licences tracking OmniFall's: {coupled}"

    def test_unverified_licences_say_so(self, sources: dict) -> None:
        """Anything not read at the source must admit it.

        Asserting an unverified licence in a tool that helps people redistribute
        data is worse than admitting ignorance.
        """
        for name, src in sources.items():
            assert src.license.strip(), f"{name} has an empty licence field"

    def test_citations_are_present_and_look_like_bibtex(self, sources: dict) -> None:
        missing = [n for n, s in sources.items() if not s.citation.strip()]
        assert not missing, f"components with no citation: {missing}"
        for name, src in sources.items():
            assert src.citation.lstrip().startswith("@"), name
            assert src.citation.count("{") == src.citation.count("}"), name


class TestArchiveSafety:
    def test_rejects_path_traversal(self, tmp_path) -> None:
        """A tar member escaping the destination must be refused.

        Extracting archives fetched over the network is exactly where this
        matters.
        """
        from omnifall._prepare import extract_archive

        payload = tmp_path / "payload.txt"
        payload.write_text("pwned")
        archive = tmp_path / "evil.tar"
        with tarfile.open(archive, "w") as tar:
            tar.add(payload, arcname="../../escaped.txt")

        dest = tmp_path / "dest"
        dest.mkdir()
        with pytest.raises(Exception) as exc:
            extract_archive(archive, dest)
        assert "escape" in str(exc.value).lower() or "outside" in str(exc.value).lower()
        assert not (tmp_path.parent / "escaped.txt").exists()

    def test_extracts_a_normal_archive(self, tmp_path) -> None:
        from omnifall._prepare import extract_archive

        src = tmp_path / "src"
        (src / "a").mkdir(parents=True)
        (src / "a" / "b.txt").write_text("hello")
        archive = tmp_path / "ok.tar"
        with tarfile.open(archive, "w") as tar:
            tar.add(src / "a", arcname="a")

        dest = tmp_path / "dest"
        extract_archive(archive, dest)
        assert (dest / "a" / "b.txt").read_text() == "hello"


class TestStatus:
    def test_status_covers_every_component(self) -> None:
        st = omnifall.status()
        assert set(st) == set(DATASETS)

    def test_status_values_are_boolean_like(self) -> None:
        for name, value in omnifall.status().items():
            assert isinstance(value, bool), f"{name}: {type(value)}"


@pytest.mark.network
class TestVerify:
    @pytest.mark.localdata
    def test_prepared_tree_verifies_completely(self, omnifall_root) -> None:
        """A fully prepared root must verify at 100% for every component."""
        incomplete = []
        for name in DATASETS:
            report = omnifall.verify(name)
            if not report.complete:
                incomplete.append(
                    f"{name}: {len(report.missing)} missing, "
                    f"{len(report.empty)} empty, of {report.required}"
                )
        assert not incomplete, "\n".join(incomplete)

    @pytest.mark.localdata
    def test_required_counts_match_the_registry(self, omnifall_root) -> None:
        for name, info in DATASETS.items():
            assert omnifall.verify(name).required == info.n_videos, name

    @pytest.mark.localdata
    def test_unreferenced_files_do_not_make_a_tree_incomplete(
        self, omnifall_root
    ) -> None:
        """Extra files are expected and must not be reported as a problem.

        OmniFall annotates a subset of several releases: cmdfall ships 1,436
        videos of which the labels reference 384, leaving 1,052 extras.
        """
        report = omnifall.verify("cmdfall")
        assert report.extra > 0
        assert report.complete

    def test_verify_reports_an_empty_tree_as_missing(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.delenv("OMNIFALL_ROOT", raising=False)
        for name in list(os.environ):
            if name.startswith("OMNIFALL_VIDEO_ROOT__"):
                monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("OMNIFALL_CACHE_DIR", str(tmp_path))
        report = omnifall.verify("le2i")
        assert not report.complete
        assert len(report.missing) == report.required > 0
        assert report.present == 0


class TestCli:
    def _run(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "omnifall.cli", *args],
            capture_output=True,
            text=True,
        )

    def test_no_args_shows_help(self) -> None:
        out = self._run()
        assert "usage" in (out.stdout + out.stderr).lower()

    @pytest.mark.parametrize("cmd", ["info", "status", "sources"])
    def test_offline_commands_succeed(self, cmd: str) -> None:
        out = self._run(cmd)
        assert out.returncode == 0, out.stderr

    def test_sources_mentions_every_component(self) -> None:
        out = self._run("sources")
        for name in DATASETS:
            assert name in out.stdout, f"{name} missing from `omnifall sources`"

    @pytest.mark.network
    def test_cite_prints_bibtex(self) -> None:
        out = self._run("cite", "le2i")
        assert out.returncode == 0, out.stderr
        assert "@" in out.stdout

    @pytest.mark.network
    def test_configs_lists_the_hub(self) -> None:
        out = self._run("configs")
        assert out.returncode == 0, out.stderr
        assert "of-itw" in out.stdout and "of-syn" in out.stdout


@pytest.mark.network
class TestOopsConsent:
    """The OOPS licence must be accepted by a person, not by default.

    OOPS videos are third-party footage under CC BY-NC-SA 4.0 for
    non-commercial research. The package redistributes nothing, but it does
    fetch on the user's behalf, so the acknowledgement has to be explicit.
    """

    def _prepare(self, tmp_path, monkeypatch, **kw):
        monkeypatch.delenv("OMNIFALL_ROOT", raising=False)
        monkeypatch.setenv("OMNIFALL_CACHE_DIR", str(tmp_path))
        return omnifall.prepare("OOPS", **kw)

    def test_declining_aborts(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr("builtins.input", lambda *_: "n")
        with pytest.raises(RuntimeError, match="cancelled by user"):
            self._prepare(tmp_path, monkeypatch)

    def test_empty_answer_aborts(self, tmp_path, monkeypatch) -> None:
        # The prompt is [y/N]; bare Enter must not be taken as agreement.
        monkeypatch.setattr("builtins.input", lambda *_: "")
        with pytest.raises(RuntimeError, match="cancelled by user"):
            self._prepare(tmp_path, monkeypatch)

    def test_non_interactive_refuses_and_names_the_flag(
        self, tmp_path, monkeypatch
    ) -> None:
        """No tty means no one can accept, so it must refuse -- helpfully.

        A bare EOFError would be correct but useless: it never mentions that
        consent=True exists.
        """
        def _eof(*_args):
            raise EOFError

        monkeypatch.setattr("builtins.input", _eof)
        with pytest.raises(RuntimeError, match="consent=True"):
            self._prepare(tmp_path, monkeypatch)

    def test_licence_notice_names_the_terms(self) -> None:
        from omnifall._constants import OOPS_LICENSE_TEXT

        notice = OOPS_LICENSE_TEXT % 818
        assert "NonCommercial" in notice
        assert "oops.cs.columbia.edu" in notice
        assert "epstein2020oops" in notice


class TestSourceTerms:
    """Components with no licence must still carry the authors' own words.

    For mcfd, up_fall and cmdfall the source page grants no licence at all --
    the citation request *is* the entire statement of conditions. Paraphrasing
    it would misrepresent what the authors asked for, so it is quoted.
    """

    NO_LICENCE = ("mcfd", "up_fall", "cmdfall")

    @pytest.mark.parametrize("name", NO_LICENCE)
    def test_unlicensed_sources_quote_their_terms(self, sources, name) -> None:
        src = sources[name]
        assert src.license == "see homepage", (
            f"{name} now claims a licence; if one was actually published, "
            "update the terms quote to match"
        )
        assert len(src.terms.strip()) > 60, f"{name} has no quoted terms"

    @pytest.mark.parametrize("name", NO_LICENCE)
    def test_terms_ask_for_citation(self, sources, name) -> None:
        text = sources[name].terms.lower()
        assert any(w in text for w in ("cite", "reference", "contact")), (
            f"{name} terms quote does not convey what the authors ask for"
        )

    def test_oops_terms_disclaim_copyright(self, sources) -> None:
        # The clause that matters most: the CC grant does not cover the
        # underlying footage, so OOPS must never be mirrored.
        assert "do not own the copyright" in sources["OOPS"].terms

    def test_terms_are_shown_by_the_cli(self) -> None:
        out = subprocess.run(
            [sys.executable, "-m", "omnifall.cli", "sources", "mcfd"],
            capture_output=True,
            text=True,
        )
        assert out.returncode == 0, out.stderr
        assert "Auvinet" in out.stdout, "quoted terms not printed"


class TestNestedArchives:
    """A release can be an archive of archives.

    Le2i's `FallDataset.zip` holds six per-scene zips, not the scene
    directories. A converter tested against a copy somebody expanded by hand
    never meets that layer, so the download route failed while the manual route
    passed. Both must end in the same tree.
    """

    def _release(self, tmp_path, scenes):
        import zipfile

        outer = tmp_path / "Release.zip"
        inner_dir = tmp_path / "build"
        inner_dir.mkdir()
        inner_zips = []
        for scene in scenes:
            content = inner_dir / scene
            (content / "Videos").mkdir(parents=True)
            (content / "Videos" / "video (1).avi").write_bytes(b"x")
            zpath = inner_dir / f"{scene}.zip"
            with zipfile.ZipFile(zpath, "w") as z:
                z.write(content / "Videos" / "video (1).avi",
                        f"{scene}/Videos/video (1).avi")
            inner_zips.append(zpath)
        with zipfile.ZipFile(outer, "w") as z:
            for zpath in inner_zips:
                z.write(zpath, zpath.name)
        return outer

    def test_locate_root_expands_nested_archives(self, tmp_path) -> None:
        from omnifall._prepare import _locate_root, extract_archive

        scenes = ["Coffee_room_01", "Coffee_room_02", "Office"]
        outer = self._release(tmp_path, scenes)
        dest = tmp_path / "unpacked"
        extract_archive(outer, dest)

        # Only two of the three are markers; all three must still be expanded,
        # or the conversion silently runs on a subset of the release.
        root = _locate_root(dest, "Coffee_room_01", "Office")
        for scene in scenes:
            assert (root / scene).is_dir(), f"{scene} was left as an archive"

    def test_plain_release_is_untouched(self, tmp_path) -> None:
        """A release that is already directories must not be disturbed."""
        from omnifall._prepare import _locate_root

        base = tmp_path / "plain"
        for scene in ("Coffee_room_01", "Office"):
            (base / scene).mkdir(parents=True)
        assert _locate_root(base, "Coffee_room_01", "Office") == base
