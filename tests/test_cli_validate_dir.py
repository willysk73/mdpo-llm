"""Tests for the whole-tree validation report ``mdpo-llm validate-dir`` (T-21)."""

from __future__ import annotations

import json
from pathlib import Path

import polib
import pytest

from mdpo_llm.cli_validate_dir import (
    CrossReferenceIssue,
    FileSummary,
    LLMValidatorFinding,
    StructuralValidatorFinding,
    ValidateDirReport,
    add_validate_dir_subparser,
    cmd_validate_dir,
    format_human_report,
    main,
    validate_directory,
)


def _write(path: Path, content: str = "x\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _po(
    path: Path,
    entries: list[dict],
) -> Path:
    """Write a PO with the supplied entries.

    Each ``entries`` row is ``{msgctxt, msgid, msgstr, tcomment?, flags?}``.
    ``flags`` defaults to ``[]``; ``tcomment`` defaults to ``""``.
    """
    po = polib.POFile()
    po.metadata = {"Content-Type": "text/plain; charset=UTF-8"}
    for row in entries:
        po.append(
            polib.POEntry(
                msgctxt=row.get("msgctxt"),
                msgid=row["msgid"],
                msgstr=row.get("msgstr", ""),
                tcomment=row.get("tcomment", ""),
                flags=row.get("flags", []),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    po.save(str(path))
    return path


# ---------------------------------------------------------------------------
# Per-file summary basics: fuzzy count + structural validator findings.
# ---------------------------------------------------------------------------


class TestPerFileSummary:
    def test_clean_tree_has_no_findings(self, tmp_path: Path) -> None:
        source = tmp_path / "src"
        target = tmp_path / "tgt"
        _write(source / "doc.md", "Source.\n")
        _write(target / "doc.md", "Target.\n")
        _po(
            target / "doc.po",
            [{"msgctxt": "para::0", "msgid": "Source.", "msgstr": "Target."}],
        )
        report = validate_directory(target, source)
        assert report.files == (
            FileSummary(
                target_file="doc.md",
                po_file="doc.po",
                source_present=True,
                fuzzy_count=0,
                structural_count=0,
            ),
        )
        assert report.cross_reference == ()
        assert not report.has_findings()

    def test_fuzzy_entries_are_counted(self, tmp_path: Path) -> None:
        source = tmp_path / "src"
        target = tmp_path / "tgt"
        _write(source / "doc.md")
        _write(target / "doc.md")
        _po(
            target / "doc.po",
            [
                {
                    "msgctxt": "p0",
                    "msgid": "Hello",
                    "msgstr": "Bonjour",
                    "flags": ["fuzzy"],
                },
                {
                    "msgctxt": "p1",
                    "msgid": "World",
                    "msgstr": "Monde",
                },
            ],
        )
        report = validate_directory(target, source)
        assert len(report.files) == 1
        assert report.files[0].fuzzy_count == 1
        assert report.files[0].structural_count == 0

    def test_structural_findings_are_parsed_from_tcomment(
        self, tmp_path: Path
    ) -> None:
        source = tmp_path / "src"
        target = tmp_path / "tgt"
        _write(source / "doc.md")
        _write(target / "doc.md")
        _po(
            target / "doc.po",
            [
                {
                    "msgctxt": "p0",
                    "msgid": "Hello",
                    "msgstr": "Bonjour",
                    "tcomment": "validator: heading_level: source has h2 but target has h3",
                    "flags": ["fuzzy"],
                },
            ],
        )
        report = validate_directory(target, source)
        assert report.files[0].structural_count == 1
        finding = report.files[0].structural_findings[0]
        assert finding.msgctxt == "p0"
        assert "heading_level" in finding.reasons

    def test_llm_lines_are_counted_always_materialised_on_flag(
        self, tmp_path: Path
    ) -> None:
        source = tmp_path / "src"
        target = tmp_path / "tgt"
        _write(source / "doc.md")
        _write(target / "doc.md")
        _po(
            target / "doc.po",
            [
                {
                    "msgctxt": "p0",
                    "msgid": "Hello",
                    "msgstr": "Bonjour",
                    "tcomment": "validator: llm: tone drifted too formal",
                    "flags": ["fuzzy"],
                },
            ],
        )
        # Default (no flag) → LLM lines hidden BUT counted (aggregate
        # must not under-report; see T-21 review cycle 2 P3).
        report = validate_directory(target, source)
        assert report.files[0].llm_findings == ()
        assert report.files[0].llm_count == 1
        assert report.aggregate()["total_llm_validator_findings"] == 1
        assert report.files[0].structural_count == 0
        # With --include-llm-validator → text surfaced too.
        report = validate_directory(
            target, source, include_llm_validator=True
        )
        assert len(report.files[0].llm_findings) == 1
        assert report.files[0].llm_count == 1
        assert report.files[0].llm_findings[0].reason == "tone drifted too formal"

    def test_llm_prefix_is_not_double_counted_as_structural(
        self, tmp_path: Path
    ) -> None:
        # The LLM prefix ``validator: llm: …`` is a strict subprefix of
        # ``validator: …`` — pin that the parser does NOT classify a
        # single tcomment line under both buckets.
        source = tmp_path / "src"
        target = tmp_path / "tgt"
        _write(source / "doc.md")
        _write(target / "doc.md")
        _po(
            target / "doc.po",
            [
                {
                    "msgctxt": "p0",
                    "msgid": "Hello",
                    "msgstr": "Bonjour",
                    "tcomment": "validator: llm: tone drifted too formal",
                    "flags": ["fuzzy"],
                },
            ],
        )
        report = validate_directory(
            target, source, include_llm_validator=True
        )
        assert report.files[0].structural_count == 0
        assert len(report.files[0].llm_findings) == 1

    def test_mixed_tcomment_lines_split_across_buckets(
        self, tmp_path: Path
    ) -> None:
        # processor.py appends multiple validator lines with ``\n``
        # separators; ensure both kinds land in the right bucket.
        source = tmp_path / "src"
        target = tmp_path / "tgt"
        _write(source / "doc.md")
        _write(target / "doc.md")
        _po(
            target / "doc.po",
            [
                {
                    "msgctxt": "p0",
                    "msgid": "Hello",
                    "msgstr": "Bonjour",
                    "tcomment": (
                        "validator: placeholder_roundtrip: token "
                        "⟦P:1⟧ missing from output\n"
                        "validator: llm: tone drifted too formal"
                    ),
                    "flags": ["fuzzy"],
                },
            ],
        )
        report = validate_directory(
            target, source, include_llm_validator=True
        )
        assert report.files[0].structural_count == 1
        assert len(report.files[0].llm_findings) == 1

    def test_missing_po_yields_zero_counts(self, tmp_path: Path) -> None:
        source = tmp_path / "src"
        target = tmp_path / "tgt"
        _write(source / "doc.md")
        _write(target / "doc.md")
        report = validate_directory(target, source)
        assert report.files[0].po_file is None
        assert report.files[0].fuzzy_count == 0
        assert report.files[0].structural_count == 0

    def test_corrupt_po_does_not_abort_walk(self, tmp_path: Path) -> None:
        source = tmp_path / "src"
        target = tmp_path / "tgt"
        _write(source / "doc.md")
        _write(target / "doc.md")
        (target / "doc.po").write_text(
            "this is not a valid PO file at all\n", encoding="utf-8"
        )
        # Should still complete and report zero counts (best-effort).
        report = validate_directory(target, source)
        assert report.files[0].fuzzy_count == 0
        assert report.files[0].structural_count == 0


# ---------------------------------------------------------------------------
# Cross-reference section.
# ---------------------------------------------------------------------------


class TestCrossReference:
    def test_target_without_source_is_flagged(self, tmp_path: Path) -> None:
        source = tmp_path / "src"
        target = tmp_path / "tgt"
        _write(source / "live.md")
        _write(target / "live.md")
        _write(target / "orphan.md")
        report = validate_directory(target, source)
        assert any(
            x.kind == "target-without-source" and x.path == "orphan.md"
            for x in report.cross_reference
        )

    def test_source_without_target_is_flagged(self, tmp_path: Path) -> None:
        source = tmp_path / "src"
        target = tmp_path / "tgt"
        _write(source / "pending.md")
        # No target written.
        target.mkdir()
        report = validate_directory(target, source)
        assert any(
            x.kind == "source-without-target" and x.path == "pending.md"
            for x in report.cross_reference
        )

    def test_cross_reference_is_sorted_for_byte_stability(
        self, tmp_path: Path
    ) -> None:
        source = tmp_path / "src"
        target = tmp_path / "tgt"
        _write(source / "a.md")
        _write(source / "b.md")
        _write(target / "z.md")
        report = validate_directory(target, source)
        kinds_paths = [(x.kind, x.path) for x in report.cross_reference]
        assert kinds_paths == sorted(kinds_paths)

    def test_validate_dir_only_flags_does_not_act(self, tmp_path: Path) -> None:
        # T-21 brief: "this report only flags, T-20 acts". Pin that the
        # orphan target file is still on disk after validate-dir runs.
        source = tmp_path / "src"
        target = tmp_path / "tgt"
        _write(source / "live.md")
        _write(target / "live.md")
        orphan = _write(target / "orphan.md")
        validate_directory(target, source)
        assert orphan.exists()


# ---------------------------------------------------------------------------
# Aggregate counters.
# ---------------------------------------------------------------------------


def test_aggregate_counts_match_per_file(tmp_path: Path) -> None:
    source = tmp_path / "src"
    target = tmp_path / "tgt"
    _write(source / "a.md")
    _write(source / "b.md")
    _write(target / "a.md")
    _write(target / "b.md")
    _po(
        target / "a.po",
        [
            {
                "msgctxt": "p0",
                "msgid": "Hello",
                "msgstr": "Bonjour",
                "tcomment": "validator: heading_level: drift",
                "flags": ["fuzzy"],
            },
        ],
    )
    _po(
        target / "b.po",
        [
            {
                "msgctxt": "p0",
                "msgid": "Hi",
                "msgstr": "Salut",
                "tcomment": "validator: llm: tone off",
                "flags": ["fuzzy"],
            },
        ],
    )
    report = validate_directory(target, source, include_llm_validator=True)
    agg = report.aggregate()
    assert agg["files_scanned"] == 2
    assert agg["po_files_scanned"] == 2
    assert agg["total_fuzzy"] == 2
    assert agg["total_structural_findings"] == 1
    assert agg["total_llm_validator_findings"] == 1
    assert agg["total_cross_reference_issues"] == 0


# ---------------------------------------------------------------------------
# Lint folding (--include-lint).
# ---------------------------------------------------------------------------


class TestLintFolding:
    def test_lint_findings_attach_to_matching_file(self, tmp_path: Path) -> None:
        source = tmp_path / "src"
        target = tmp_path / "tgt"
        _write(source / "doc.md", "Intro.\n")
        # Target has residue (Korean chars) under --target en.
        _write(
            target / "doc.md",
            "Translated intro with a Korean term 회원.\n",
        )
        report = validate_directory(
            target, source, target_lang="en", include_lint=True
        )
        assert report.lint_ran is True
        assert report.files[0].residue
        assert any("ko" in r.languages for r in report.files[0].residue)

    def test_dangling_reference_attaches_to_file(self, tmp_path: Path) -> None:
        source = tmp_path / "src"
        target = tmp_path / "tgt"
        _write(source / "doc.md")
        _write(
            target / "doc.md",
            "See `missing-attachment.pdf` for details.\n",
        )
        report = validate_directory(
            target, source, target_lang="en", include_lint=True
        )
        assert report.files[0].dangling
        assert report.files[0].dangling[0].reference == "missing-attachment.pdf"

    def test_dangling_resolves_via_source_root(self, tmp_path: Path) -> None:
        # The brief specifies source_dir doubles as --source-root for
        # the lint folding so attachments present only in source still
        # resolve.
        source = tmp_path / "src"
        target = tmp_path / "tgt"
        _write(source / "doc.md")
        _write(source / "spec.pdf", "")
        _write(target / "doc.md", "See `spec.pdf` for details.\n")
        report = validate_directory(
            target, source, target_lang="en", include_lint=True
        )
        assert report.files[0].dangling == ()

    def test_include_lint_requires_target(self, tmp_path: Path) -> None:
        source = tmp_path / "src"
        target = tmp_path / "tgt"
        _write(source / "doc.md")
        _write(target / "doc.md")
        with pytest.raises(ValueError, match="requires --target"):
            validate_directory(target, source, include_lint=True)

    def test_uppercase_md_extension_is_flagged_as_lint_skipped(
        self, tmp_path: Path
    ) -> None:
        # cli_lint.lint_directory uses a case-sensitive ``*.md`` glob,
        # while validate-dir's walk is case-insensitive (matching the
        # rest of the pipeline). A ``README.MD`` sibling would
        # otherwise appear in the report with zero residue/dangling
        # findings even when it contains lint issues — that is the
        # T-21 review-cycle-1 P2 from Codex. Pin that the per-file
        # ``lint_scanned`` flag and the aggregate ``lint_coverage_gap``
        # surface the discrepancy.
        source = tmp_path / "src"
        target = tmp_path / "tgt"
        _write(source / "doc.md")
        _write(target / "doc.md", "Clean.\n")
        _write(
            target / "README.MD",
            "Translated intro with a Korean term 회원.\n",
        )
        report = validate_directory(
            target, source, target_lang="en", include_lint=True
        )
        by_name = {f.target_file: f for f in report.files}
        assert by_name["doc.md"].lint_scanned is True
        # The uppercase-suffix sibling MUST be flagged as not scanned
        # even though it appears in the walk.
        assert by_name["README.MD"].lint_scanned is False
        # Aggregate counter reflects the gap so CI can react.
        assert report.aggregate()["lint_coverage_gap"] == 1

    def test_lint_findings_for_nested_files_use_posix_keys(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # T-21 review cycle 2 P2: on Windows, cli_lint.lint_directory
        # stores nested rel-paths with backslashes (because it builds
        # them via ``str(path.relative_to(...))``), while validate-dir's
        # target_rels are POSIX. A naive bucketing would drop every
        # nested-directory finding silently. Simulate the Windows
        # shape by monkey-patching the lint_directory helper to return
        # a backslash-keyed report and pin that we still attach the
        # findings to the right per-file row.
        from mdpo_llm import cli_validate_dir as mod
        from mdpo_llm.cli_lint import LintReport, ResidueFinding

        source = tmp_path / "src"
        target = tmp_path / "tgt"
        _write(source / "guide/intro.md")
        _write(
            target / "guide/intro.md",
            "Some text with Korean 회원 leaked through.\n",
        )

        def _fake(target_dir, *, target_lang, source_root=None):
            return LintReport(
                files_scanned=1,
                residue=(
                    ResidueFinding(
                        # Windows-style backslash path; the helper
                        # MUST normalise it before bucketing.
                        file="guide\\intro.md",
                        line=1,
                        text="Some text with Korean 회원",
                        languages=("ko",),
                    ),
                ),
                dangling=(),
            )

        monkeypatch.setattr(mod, "lint_directory", _fake)
        report = validate_directory(
            target, source, target_lang="en", include_lint=True
        )
        by_name = {f.target_file: f for f in report.files}
        assert by_name["guide/intro.md"].residue, (
            "nested lint finding was bucketed under a backslash key "
            "and silently dropped — _build_lint_index must normalise "
            "lint file fields to POSIX before lookup"
        )

    def test_lint_scanned_is_false_when_lint_disabled(
        self, tmp_path: Path
    ) -> None:
        # Without --include-lint, every file's lint_scanned must be
        # False (and lint_coverage_gap must be 0 — the gap is only
        # meaningful when lint actually ran).
        source = tmp_path / "src"
        target = tmp_path / "tgt"
        _write(source / "doc.md")
        _write(target / "doc.md")
        report = validate_directory(target, source)
        assert all(not f.lint_scanned for f in report.files)
        assert report.aggregate()["lint_coverage_gap"] == 0


# ---------------------------------------------------------------------------
# PO directory override.
# ---------------------------------------------------------------------------


def test_po_dir_override_locates_pos_outside_target(tmp_path: Path) -> None:
    source = tmp_path / "src"
    target = tmp_path / "tgt"
    po_root = tmp_path / "pos"
    _write(source / "doc.md")
    _write(target / "doc.md")
    _po(
        po_root / "doc.po",
        [
            {
                "msgctxt": "p0",
                "msgid": "Hello",
                "msgstr": "Bonjour",
                "flags": ["fuzzy"],
            },
        ],
    )
    report = validate_directory(target, source, po_dir=po_root)
    assert report.files[0].po_file == "doc.po"
    assert report.files[0].fuzzy_count == 1


# ---------------------------------------------------------------------------
# Usage errors and CLI exit codes.
# ---------------------------------------------------------------------------


class TestCliBoundary:
    def test_missing_target_dir_is_usage_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        source = tmp_path / "src"
        _write(source / "doc.md")
        rc = main(
            [
                "validate-dir",
                str(tmp_path / "does-not-exist"),
                "--source",
                str(source),
            ]
        )
        assert rc == 2
        captured = capsys.readouterr()
        assert "does not exist" in captured.err

    def test_missing_source_dir_is_usage_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        target = tmp_path / "tgt"
        target.mkdir()
        rc = main(
            [
                "validate-dir",
                str(target),
                "--source",
                str(tmp_path / "missing-source"),
            ]
        )
        assert rc == 2
        assert "does not exist" in capsys.readouterr().err

    def test_include_lint_without_target_is_usage_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        source = tmp_path / "src"
        target = tmp_path / "tgt"
        _write(source / "doc.md")
        _write(target / "doc.md")
        rc = main(
            [
                "validate-dir",
                str(target),
                "--source",
                str(source),
                "--include-lint",
            ]
        )
        assert rc == 2
        assert "--include-lint requires --target" in capsys.readouterr().err

    def test_findings_without_flag_still_exits_zero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        source = tmp_path / "src"
        target = tmp_path / "tgt"
        _write(source / "live.md")
        _write(target / "live.md")
        _write(target / "orphan.md")  # produces a cross-reference issue
        rc = main(
            [
                "validate-dir",
                str(target),
                "--source",
                str(source),
            ]
        )
        assert rc == 0  # findings exist but no --exit-non-zero-on-findings

    def test_findings_with_flag_exits_one(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        source = tmp_path / "src"
        target = tmp_path / "tgt"
        _write(source / "live.md")
        _write(target / "live.md")
        _write(target / "orphan.md")
        rc = main(
            [
                "validate-dir",
                str(target),
                "--source",
                str(source),
                "--exit-non-zero-on-findings",
            ]
        )
        assert rc == 1

    def test_json_output_is_valid_json(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        source = tmp_path / "src"
        target = tmp_path / "tgt"
        _write(source / "doc.md")
        _write(target / "doc.md")
        _po(
            target / "doc.po",
            [
                {
                    "msgctxt": "p0",
                    "msgid": "Hi",
                    "msgstr": "Salut",
                    "tcomment": "validator: heading_level: drift",
                    "flags": ["fuzzy"],
                },
            ],
        )
        rc = main(
            [
                "validate-dir",
                str(target),
                "--source",
                str(source),
                "--json",
            ]
        )
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["aggregate"]["total_structural_findings"] == 1
        assert payload["aggregate"]["total_fuzzy"] == 1
        assert payload["llm_validator_ran"] is False

    def test_subparser_attaches_to_main_module(self, tmp_path: Path) -> None:
        # Smoke-test that the verb is reachable via the package's
        # ``python -m mdpo_llm`` entry point (T-21 must wire the
        # subparser into __main__.py).
        from mdpo_llm.__main__ import build_parser

        parser = build_parser()
        source = tmp_path / "src"
        target = tmp_path / "tgt"
        _write(source / "doc.md")
        _write(target / "doc.md")
        ns = parser.parse_args(
            [
                "validate-dir",
                str(target),
                "--source",
                str(source),
            ]
        )
        assert ns.command == "validate-dir"
        assert callable(ns.func)


# ---------------------------------------------------------------------------
# Human-readable rendering smoke test.
# ---------------------------------------------------------------------------


def test_human_report_renders_aggregate_block(tmp_path: Path) -> None:
    source = tmp_path / "src"
    target = tmp_path / "tgt"
    _write(source / "doc.md")
    _write(target / "doc.md")
    _po(
        target / "doc.po",
        [
            {
                "msgctxt": "p0",
                "msgid": "Hi",
                "msgstr": "Salut",
                "tcomment": "validator: heading_level: drift",
                "flags": ["fuzzy"],
            },
        ],
    )
    report = validate_directory(target, source)
    rendered = format_human_report(report)
    assert "Aggregate:" in rendered
    assert "total_fuzzy: 1" in rendered
    assert "FOUND ISSUES" in rendered


def test_human_report_pass_when_clean(tmp_path: Path) -> None:
    source = tmp_path / "src"
    target = tmp_path / "tgt"
    _write(source / "doc.md")
    _write(target / "doc.md")
    report = validate_directory(target, source)
    rendered = format_human_report(report)
    assert "PASSED" in rendered
