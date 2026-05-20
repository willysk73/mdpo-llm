"""Tests for the read-only ``mdpo-llm lint`` scanner (T-19)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mdpo_llm.cli_lint import (
    DanglingFinding,
    LINT_EXTENSIONS,
    LintReport,
    ResidueFinding,
    cmd_lint,
    format_human_report,
    lint_directory,
    main,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Residue detection — one case per supported source script.
# ---------------------------------------------------------------------------


class TestResidueDetection:
    def test_hangul_in_english_target_is_residue(self, tmp_path: Path) -> None:
        _write(
            tmp_path / "doc.md",
            "Introduction\n\nThis paragraph has a Korean term 회원 in it.\n",
        )
        report = lint_directory(tmp_path, target_lang="en")
        assert len(report.residue) == 1
        finding = report.residue[0]
        assert finding.line == 3
        assert "ko" in finding.languages

    def test_cjk_in_english_target_is_residue(self, tmp_path: Path) -> None:
        _write(tmp_path / "doc.md", "Pure Chinese 中文 leaked through.\n")
        report = lint_directory(tmp_path, target_lang="en")
        assert len(report.residue) == 1
        assert "zh" in report.residue[0].languages

    def test_japanese_kana_in_english_target_is_residue(self, tmp_path: Path) -> None:
        _write(tmp_path / "doc.md", "Hiragana あいう and katakana カタカナ.\n")
        report = lint_directory(tmp_path, target_lang="en")
        assert len(report.residue) == 1
        assert "ja" in report.residue[0].languages

    def test_target_language_chars_are_not_residue(self, tmp_path: Path) -> None:
        _write(tmp_path / "doc.md", "순수한 한국어 문장입니다.\n")
        report = lint_directory(tmp_path, target_lang="ko")
        assert report.residue == ()

    def test_latin_target_does_not_flag_latin_prose(self, tmp_path: Path) -> None:
        # ``--target fr`` (or any other Latin-script target) MUST NOT
        # flag ordinary French / German / Spanish prose as residue:
        # the language module's coarse ``en`` pattern ([A-Za-z])
        # matches every Latin letter, so a raw detect_languages-only
        # check would produce universal false positives here.
        _write(
            tmp_path / "doc.md",
            "Bonjour le monde, ceci est une phrase française.\n"
            "Hola mundo, esta es una frase en español.\n"
            "Hallo Welt, dies ist ein deutscher Satz.\n",
        )
        report = lint_directory(tmp_path, target_lang="fr")
        assert report.residue == ()

    def test_english_in_korean_target_is_not_flagged(self, tmp_path: Path) -> None:
        # English / Latin-script leakage in a Korean target is NOT
        # surfaced by the lint scanner — it focuses on the non-Latin
        # scripts (ko / ja / zh) where leakage is visually
        # unambiguous. Disambiguating English-the-language from
        # Latin-script identifiers / brand names is the structural
        # validator's job.
        _write(tmp_path / "doc.md", "Mostly English with a Korean 단어.\n")
        report = lint_directory(tmp_path, target_lang="ko")
        assert report.residue == ()

    def test_residue_truncates_long_lines(self, tmp_path: Path) -> None:
        long_line = "한국어 " + "x" * 200 + "\n"
        _write(tmp_path / "doc.md", long_line)
        report = lint_directory(tmp_path, target_lang="en")
        assert len(report.residue) == 1
        assert len(report.residue[0].text) <= 80

    def test_japanese_kanji_under_japanese_target_is_not_residue(
        self, tmp_path: Path
    ) -> None:
        # Japanese kanji is in CJK Unified Ideographs, which is the
        # ``zh`` pattern's range. Under ``--target ja`` every
        # kanji-bearing Japanese line would otherwise be flagged as
        # ``zh`` residue — this test pins the target-aware
        # suppression that mirrors residue_pass._resolve_residue_pattern.
        _write(
            tmp_path / "doc.md",
            "純粋な日本語の文章。\n"  # kana + kanji mix
            "漢字だけの行。\n"  # kana + kanji mix
            "完全漢字漢字漢字漢字\n",  # kanji-only line
        )
        report = lint_directory(tmp_path, target_lang="ja")
        assert report.residue == ()

    def test_kana_under_chinese_target_is_residue(self, tmp_path: Path) -> None:
        # The other direction MUST still flag: kana never appears in
        # natural Chinese, so a kana-bearing line under ``--target zh``
        # is real residue.
        _write(tmp_path / "doc.md", "中文文本 含 ひらがな\n")
        report = lint_directory(tmp_path, target_lang="zh")
        assert len(report.residue) == 1
        assert "ja" in report.residue[0].languages

    def test_korean_under_japanese_target_is_residue(
        self, tmp_path: Path
    ) -> None:
        # The CJK-overlap suppression must NOT silence non-CJK residue.
        _write(tmp_path / "doc.md", "日本語と 한국어 が混在。\n")
        report = lint_directory(tmp_path, target_lang="ja")
        assert len(report.residue) == 1
        assert "ko" in report.residue[0].languages

    def test_locale_subtag_normalised(self, tmp_path: Path) -> None:
        # ``zh-CN`` and ``zh-TW`` share the ``zh`` primary subtag, so a
        # pure-Chinese line under target=zh-CN must NOT be flagged.
        _write(tmp_path / "doc.md", "中文文本测试。\n")
        report = lint_directory(tmp_path, target_lang="zh-CN")
        assert report.residue == ()


# ---------------------------------------------------------------------------
# Dangling-reference detection: present in source / present in target / missing.
# ---------------------------------------------------------------------------


class TestDanglingReferences:
    def _setup_tree(self, tmp_path: Path) -> tuple[Path, Path]:
        target = tmp_path / "target"
        source = tmp_path / "source"
        target.mkdir()
        source.mkdir()
        # File present in TARGET tree only.
        (target / "guide.pdf").write_bytes(b"%PDF-1.4\n")
        # File present in SOURCE tree only.
        (source / "spec.pdf").write_bytes(b"%PDF-1.4\n")
        return target, source

    def test_reference_present_in_target_is_resolved(self, tmp_path: Path) -> None:
        target, source = self._setup_tree(tmp_path)
        _write(target / "doc.md", "See `guide.pdf` for details.\n")
        report = lint_directory(target, target_lang="en", source_root=source)
        assert report.dangling == ()

    def test_reference_present_in_source_is_resolved(self, tmp_path: Path) -> None:
        target, source = self._setup_tree(tmp_path)
        _write(target / "doc.md", "Original spec lives at `spec.pdf`.\n")
        report = lint_directory(target, target_lang="en", source_root=source)
        assert report.dangling == ()

    def test_missing_reference_is_reported(self, tmp_path: Path) -> None:
        target, source = self._setup_tree(tmp_path)
        _write(target / "doc.md", "Broken link to `missing.pdf` here.\n")
        report = lint_directory(target, target_lang="en", source_root=source)
        assert len(report.dangling) == 1
        assert report.dangling[0].reference == "missing.pdf"
        assert report.dangling[0].line == 1

    def test_missing_reference_without_source_root(self, tmp_path: Path) -> None:
        target = tmp_path / "target"
        target.mkdir()
        _write(target / "doc.md", "Reference to `nowhere.pdf`.\n")
        report = lint_directory(target, target_lang="en")
        assert len(report.dangling) == 1
        assert report.dangling[0].reference == "nowhere.pdf"

    def test_extensions_covered(self, tmp_path: Path) -> None:
        target = tmp_path / "target"
        target.mkdir()
        for ext in LINT_EXTENSIONS:
            (target / f"present.{ext}").write_bytes(b"")
        body_lines = [f"See `present.{ext}` and `missing.{ext}`." for ext in LINT_EXTENSIONS]
        _write(target / "doc.md", "\n".join(body_lines) + "\n")
        report = lint_directory(target, target_lang="en")
        missing_refs = {d.reference for d in report.dangling}
        for ext in LINT_EXTENSIONS:
            assert f"missing.{ext}" in missing_refs
            assert f"present.{ext}" not in missing_refs

    def test_angle_bracket_reference(self, tmp_path: Path) -> None:
        target = tmp_path / "target"
        target.mkdir()
        (target / "found.png").write_bytes(b"")
        _write(target / "doc.md", "Inline <found.png> and <gone.png>.\n")
        report = lint_directory(target, target_lang="en")
        refs = {d.reference for d in report.dangling}
        assert refs == {"gone.png"}

    def test_url_refs_are_skipped(self, tmp_path: Path) -> None:
        target = tmp_path / "target"
        target.mkdir()
        _write(
            target / "doc.md",
            "Remote: `https://example.com/file.pdf` and "
            "<http://example.com/x.png>.\n",
        )
        report = lint_directory(target, target_lang="en")
        assert report.dangling == ()

    def test_case_insensitive_matching(self, tmp_path: Path) -> None:
        target = tmp_path / "target"
        target.mkdir()
        (target / "Image.PNG").write_bytes(b"")
        _write(target / "doc.md", "Inline `image.png` reference.\n")
        report = lint_directory(target, target_lang="en")
        assert report.dangling == ()

    def test_basename_match_ignores_path_prefix(self, tmp_path: Path) -> None:
        target = tmp_path / "target"
        target.mkdir()
        (target / "assets").mkdir()
        (target / "assets" / "logo.svg").write_bytes(b"")
        _write(target / "doc.md", "Refer to `docs/old/logo.svg` here.\n")
        report = lint_directory(target, target_lang="en")
        assert report.dangling == ()


# ---------------------------------------------------------------------------
# Empty directory + edge cases.
# ---------------------------------------------------------------------------


class TestEmptyAndEdge:
    def test_empty_directory_is_clean(self, tmp_path: Path) -> None:
        report = lint_directory(tmp_path, target_lang="en")
        assert report.files_scanned == 0
        assert report.residue == ()
        assert report.dangling == ()
        assert not report.has_findings()

    def test_unreadable_file_is_tolerated(self, tmp_path: Path) -> None:
        # Binary content (invalid UTF-8) MUST NOT abort the walk.
        (tmp_path / "a.md").write_bytes(b"\xff\xfe not utf-8")
        _write(tmp_path / "b.md", "Plain ASCII.\n")
        report = lint_directory(tmp_path, target_lang="en")
        assert report.files_scanned == 2
        # Only b.md contributes to residue scanning; a.md is silently skipped.
        assert report.residue == ()

    def test_files_scanned_counts_md_only(self, tmp_path: Path) -> None:
        _write(tmp_path / "a.md", "ok\n")
        _write(tmp_path / "b.md", "ok\n")
        (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")
        report = lint_directory(tmp_path, target_lang="en")
        assert report.files_scanned == 2


# ---------------------------------------------------------------------------
# Exit-code semantics via the CLI entry point.
# ---------------------------------------------------------------------------


class TestExitCodes:
    def test_clean_tree_exits_zero(self, tmp_path: Path, capsys) -> None:
        _write(tmp_path / "doc.md", "Plain English.\n")
        rc = main([
            "lint",
            str(tmp_path),
            "--target",
            "en",
        ])
        assert rc == 0

    def test_findings_default_exit_zero(self, tmp_path: Path, capsys) -> None:
        # Without --exit-non-zero-on-findings, findings are reported but
        # the command still exits 0 — the brief's default.
        _write(tmp_path / "doc.md", "Korean 한국어 in English target.\n")
        rc = main([
            "lint",
            str(tmp_path),
            "--target",
            "en",
        ])
        assert rc == 0

    def test_findings_exit_non_zero_when_flag_set(
        self, tmp_path: Path, capsys
    ) -> None:
        _write(tmp_path / "doc.md", "Korean 한국어 in English target.\n")
        rc = main([
            "lint",
            str(tmp_path),
            "--target",
            "en",
            "--exit-non-zero-on-findings",
        ])
        assert rc == 1

    def test_missing_directory_exits_two(self, tmp_path: Path, capsys) -> None:
        rc = main([
            "lint",
            str(tmp_path / "does-not-exist"),
            "--target",
            "en",
        ])
        assert rc == 2
        err = capsys.readouterr().err
        assert "does not exist" in err

    def test_file_passed_as_directory_exits_two(
        self, tmp_path: Path, capsys
    ) -> None:
        bad = tmp_path / "not-a-dir.md"
        bad.write_text("x", encoding="utf-8")
        rc = main([
            "lint",
            str(bad),
            "--target",
            "en",
        ])
        assert rc == 2
        assert "not a directory" in capsys.readouterr().err

    def test_invalid_source_root_exits_two(self, tmp_path: Path, capsys) -> None:
        _write(tmp_path / "doc.md", "ok\n")
        bad = tmp_path / "no-source"
        rc = main([
            "lint",
            str(tmp_path),
            "--target",
            "en",
            "--source-root",
            str(bad),
        ])
        assert rc == 2


# ---------------------------------------------------------------------------
# JSON output schema is stable.
# ---------------------------------------------------------------------------


class TestJsonOutput:
    def test_json_output_schema(self, tmp_path: Path, capsys) -> None:
        target = tmp_path / "target"
        target.mkdir()
        (target / "found.pdf").write_bytes(b"")
        _write(
            target / "doc.md",
            "Korean 한국어 residue.\nMissing `gone.pdf` reference.\n",
        )
        rc = main([
            "lint",
            str(target),
            "--target",
            "en",
            "--json",
        ])
        assert rc == 0
        out = capsys.readouterr().out
        payload = json.loads(out)
        # Top-level keys MUST be exactly the documented set.
        assert set(payload.keys()) == {"files_scanned", "residue", "dangling"}
        assert payload["files_scanned"] == 1

        assert isinstance(payload["residue"], list)
        assert len(payload["residue"]) == 1
        r = payload["residue"][0]
        assert set(r.keys()) == {"file", "line", "text", "languages"}
        assert r["line"] == 1
        assert r["languages"] == ["ko"]
        assert isinstance(r["file"], str)

        assert isinstance(payload["dangling"], list)
        assert len(payload["dangling"]) == 1
        d = payload["dangling"][0]
        assert set(d.keys()) == {"file", "line", "reference"}
        assert d["line"] == 2
        assert d["reference"] == "gone.pdf"

    def test_empty_tree_json_is_well_formed(self, tmp_path: Path, capsys) -> None:
        rc = main([
            "lint",
            str(tmp_path),
            "--target",
            "en",
            "--json",
        ])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload == {
            "files_scanned": 0,
            "residue": [],
            "dangling": [],
        }


# ---------------------------------------------------------------------------
# Human-readable report shape (smoke-level).
# ---------------------------------------------------------------------------


class TestHumanReport:
    def test_passed_marker_on_clean_tree(self) -> None:
        report = LintReport(files_scanned=3, residue=(), dangling=())
        text = format_human_report(report, target_lang="en")
        assert "Scanned 3 markdown file(s)." in text
        assert "PASSED" in text

    def test_issues_marker_when_findings_present(self) -> None:
        report = LintReport(
            files_scanned=1,
            residue=(
                ResidueFinding(
                    file="doc.md", line=2, text="some text", languages=("ko",)
                ),
            ),
            dangling=(
                DanglingFinding(file="doc.md", line=3, reference="x.pdf"),
            ),
        )
        text = format_human_report(report, target_lang="en")
        assert "FOUND ISSUES" in text
        assert "doc.md:2:" in text
        assert "doc.md:3:" in text
        assert "x.pdf" in text


# ---------------------------------------------------------------------------
# Top-level CLI integration through the main argparse tree.
# ---------------------------------------------------------------------------


class TestTopLevelCLI:
    def test_lint_subcommand_registered(self, tmp_path: Path, capsys) -> None:
        from mdpo_llm.__main__ import main as main_cli

        _write(tmp_path / "doc.md", "Plain English.\n")
        rc = main_cli([
            "lint",
            str(tmp_path),
            "--target",
            "en",
            "--json",
        ])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["files_scanned"] == 1
        assert payload["residue"] == []
        assert payload["dangling"] == []
