"""Whole-tree validation report for translated Markdown directories (T-21).

``mdpo-llm validate-dir <target_dir> --source <source_dir>`` walks an
already-translated directory and aggregates three signal sources into a
single per-file roll-up so reviewers do not have to grep per-file PO
trees by hand:

1. **Per-document PO files** — fuzzy entry count, plus the
   ``tcomment`` lines the post-translation validators stamp onto
   flagged entries. Two prefixes are recognised, both written by
   :mod:`mdpo_llm.processor`:

   * ``validator: <check>: <detail>; …`` — the structural validator
     (placeholder round-trip, fence count, heading drift, language
     stability, …). Surfaced as a count by default.
   * ``validator: llm: <reason>`` — the T-16 LLM grader's final
     rejection reason. Surfaced as text under
     ``--include-llm-validator``.

2. **Cross-reference issues** — source files with no target and
   target files with no source. Mirror layout only (target relative
   path == source relative path). This overlaps with the T-20
   cleanup verb deliberately: ``validate-dir`` only flags, T-20
   acts.

3. **T-19 lint findings** — opt-in via ``--include-lint``. Reuses
   :func:`mdpo_llm.cli_lint.lint_directory` as a library helper so
   residue and dangling-reference semantics stay consistent with the
   ``mdpo-llm lint`` CLI rather than re-implementing the scan here.

The verb is read-only: no PO writes, no LLM calls, no filesystem
mutation. Intended as a CI gate (via ``--exit-non-zero-on-findings``)
and as a one-shot reviewer summary.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import polib

from .cli_lint import (
    DanglingFinding,
    LINT_EXTENSIONS,
    LintReport,
    ResidueFinding,
    lint_directory,
)


# Extension match for the directory walk. Lower-cased before lookup so
# ``README.MD`` on a case-sensitive filesystem still participates in
# both the target walk and the mirror cross-reference — same rule
# cleanup_ops uses, kept consistent here.
TARGET_EXTENSIONS: frozenset[str] = frozenset({".md"})


# Validator tcomment prefixes. Mirrors the strings
# :mod:`mdpo_llm.processor` writes when flagging entries fuzzy so the
# parser stays in sync with the producer; see processor.py:6131-6134
# (structural) and processor.py:6273 (LLM). The LLM line is a strict
# subprefix of the structural line, so order matters — match LLM
# first.
_LLM_PREFIX = "validator: llm: "
_STRUCTURAL_PREFIX = "validator: "


@dataclass(frozen=True)
class LLMValidatorFinding:
    """One ``validator: llm: <reason>`` tcomment line lifted out of a PO entry.

    ``msgctxt`` is the entry's PO context key — usually a parser
    block identifier. The reviewer uses it to jump back to the
    offending block in the source / target Markdown.
    """

    file: str
    msgctxt: str
    reason: str


@dataclass(frozen=True)
class StructuralValidatorFinding:
    """One ``validator: <reasons>`` tcomment line (non-LLM).

    ``reasons`` is the full semicolon-joined list of ``check: detail``
    pairs the structural validator emitted; we surface it verbatim
    rather than re-parse so future validator checks introduce no
    schema churn in this scanner.
    """

    file: str
    msgctxt: str
    reasons: str


@dataclass(frozen=True)
class CrossReferenceIssue:
    """One mirror-layout cross-reference mismatch.

    ``kind`` is ``"source-without-target"`` (source file has no
    corresponding target on disk) or ``"target-without-source"``
    (target file is an orphan — its source has been deleted or moved).
    Both are POSIX relative paths against their respective roots.
    """

    kind: str
    path: str


@dataclass(frozen=True)
class FileSummary:
    """Per-file roll-up of every signal collected for one target Markdown.

    ``llm_count`` is the number of ``validator: llm: <reason>``
    tcomment lines observed on this file's PO and is ALWAYS
    populated when a PO was parsed, even when
    ``--include-llm-validator`` was off (the flag only controls
    whether the full per-line text in ``llm_findings`` is
    materialised). Without this split, aggregate consumers would
    under-report LLM validator failures unless they also opted into
    the verbose per-line output.

    ``lint_scanned`` reports whether the T-19 lint pass actually
    covered this file. Set to ``True`` only when ``--include-lint``
    was active AND the file's path matched ``cli_lint.lint_directory``'s
    case-sensitive ``*.md`` glob (always ``False`` when
    ``--include-lint`` was off). Uppercase-suffix siblings
    (``README.MD``) appear in the walk via the case-insensitive
    ``suffix.lower()`` rule the rest of the pipeline uses, but T-19's
    scanner skips them — surfacing ``lint_scanned=False`` keeps a
    clean residue/dangling row from giving false reassurance on
    those files.
    """

    target_file: str
    po_file: Optional[str]
    source_present: bool
    fuzzy_count: int
    structural_count: int
    llm_count: int = 0
    structural_findings: Tuple[StructuralValidatorFinding, ...] = ()
    llm_findings: Tuple[LLMValidatorFinding, ...] = ()
    residue: Tuple[ResidueFinding, ...] = ()
    dangling: Tuple[DanglingFinding, ...] = ()
    lint_scanned: bool = False

    def has_findings(self) -> bool:
        return bool(
            self.fuzzy_count
            or self.structural_count
            or self.llm_count
            or self.residue
            or self.dangling
        )


@dataclass(frozen=True)
class ValidateDirReport:
    """Aggregate report returned by :func:`validate_directory`.

    ``files`` is sorted by ``target_file`` for byte-stable output;
    ``cross_reference`` is sorted by ``(kind, path)``. ``lint_ran``
    records whether ``--include-lint`` was active so the human
    renderer can mark a zero-finding lint section as "scanned and
    clean" vs "not scanned".
    """

    target_dir: str
    source_dir: str
    files: Tuple[FileSummary, ...]
    cross_reference: Tuple[CrossReferenceIssue, ...]
    llm_validator_ran: bool
    lint_ran: bool

    def aggregate(self) -> dict:
        return {
            "files_scanned": len(self.files),
            "po_files_scanned": sum(1 for f in self.files if f.po_file),
            "total_fuzzy": sum(f.fuzzy_count for f in self.files),
            "total_structural_findings": sum(f.structural_count for f in self.files),
            "total_llm_validator_findings": sum(f.llm_count for f in self.files),
            "total_residue": sum(len(f.residue) for f in self.files),
            "total_dangling": sum(len(f.dangling) for f in self.files),
            "total_cross_reference_issues": len(self.cross_reference),
            # Coverage signal, not a finding: counts files that ``--include-lint``
            # could not scan because of T-19's case-sensitive ``*.md`` glob
            # (uppercase ``.MD`` siblings, mostly). Always ``0`` when
            # ``--include-lint`` was off. Surfaced in ``aggregate`` so CI
            # consumers can detect the gap without having to inspect
            # per-file rows.
            "lint_coverage_gap": (
                sum(1 for f in self.files if not f.lint_scanned)
                if self.lint_ran
                else 0
            ),
        }

    def has_findings(self) -> bool:
        agg = self.aggregate()
        return any(
            agg[k]
            for k in (
                "total_fuzzy",
                "total_structural_findings",
                "total_llm_validator_findings",
                "total_residue",
                "total_dangling",
                "total_cross_reference_issues",
            )
        )

    def to_dict(self) -> dict:
        return {
            "target_dir": self.target_dir,
            "source_dir": self.source_dir,
            "llm_validator_ran": self.llm_validator_ran,
            "lint_ran": self.lint_ran,
            "files": [
                {
                    "target_file": f.target_file,
                    "po_file": f.po_file,
                    "source_present": f.source_present,
                    "lint_scanned": f.lint_scanned,
                    "fuzzy_count": f.fuzzy_count,
                    "structural_count": f.structural_count,
                    "llm_count": f.llm_count,
                    "structural_findings": [
                        {
                            "file": s.file,
                            "msgctxt": s.msgctxt,
                            "reasons": s.reasons,
                        }
                        for s in f.structural_findings
                    ],
                    "llm_findings": [
                        {
                            "file": l.file,
                            "msgctxt": l.msgctxt,
                            "reason": l.reason,
                        }
                        for l in f.llm_findings
                    ],
                    "residue": [
                        {
                            "file": r.file,
                            "line": r.line,
                            "text": r.text,
                            "languages": list(r.languages),
                        }
                        for r in f.residue
                    ],
                    "dangling": [
                        {
                            "file": d.file,
                            "line": d.line,
                            "reference": d.reference,
                        }
                        for d in f.dangling
                    ],
                }
                for f in self.files
            ],
            "cross_reference": [
                {"kind": x.kind, "path": x.path} for x in self.cross_reference
            ],
            "aggregate": self.aggregate(),
        }


def _scan_markdown_rels(root: Path) -> set[str]:
    """Return POSIX-form relative paths of every Markdown file under ``root``.

    POSIX form so the mirror cross-reference comparison stays portable
    across Windows checkouts. Extension match is case-insensitive to
    mirror the ``suffix.lower()`` rule the rest of the pipeline
    (``cleanup_ops``, ``processor.process_directory``) already uses;
    otherwise a live ``README.MD`` source on a case-sensitive FS would
    look missing while the target walker still considers it.
    """
    result: set[str] = set()
    for path in root.rglob("*"):
        try:
            if not path.is_file():
                continue
        except OSError:
            continue
        if path.suffix.lower() not in TARGET_EXTENSIONS:
            continue
        try:
            result.add(path.relative_to(root).as_posix())
        except ValueError:
            continue
    return result


def _parse_po_findings(
    po_path: Path,
    *,
    display_file: str,
    include_llm: bool,
) -> Tuple[
    int,
    List[StructuralValidatorFinding],
    int,
    List[LLMValidatorFinding],
]:
    """Return ``(fuzzy_count, structural_findings, llm_count, llm_findings)``.

    Unreadable / unparseable PO files yield ``(0, [], 0, [])`` — the
    scanner is best-effort and a single corrupt PO MUST NOT abort
    the whole validate-dir walk. Obsolete entries are ignored because
    they are not part of the active translation surface.

    ``llm_count`` is the total of ``validator: llm: …`` tcomment
    lines observed and is ALWAYS reported, regardless of
    ``include_llm``. Only the per-line text materialisation
    (``llm_findings``) is gated by the flag — otherwise the
    aggregate counters would under-report LLM validator failures
    unless the operator also opted into the verbose per-line output.
    """
    try:
        po_file = polib.pofile(str(po_path), encoding="utf-8")
    except Exception:
        return 0, [], 0, []

    fuzzy_count = 0
    structural: List[StructuralValidatorFinding] = []
    llm: List[LLMValidatorFinding] = []
    llm_count = 0
    for entry in po_file:
        if entry.obsolete:
            continue
        if "fuzzy" in entry.flags:
            fuzzy_count += 1
        tcomment = entry.tcomment or ""
        if not tcomment:
            continue
        msgctxt = entry.msgctxt or ""
        for raw_line in tcomment.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            # LLM prefix is checked first because it is a strict
            # subprefix of the structural prefix — matching structural
            # first would steal every llm line.
            if line.startswith(_LLM_PREFIX):
                llm_count += 1
                if include_llm:
                    llm.append(
                        LLMValidatorFinding(
                            file=display_file,
                            msgctxt=msgctxt,
                            reason=line[len(_LLM_PREFIX) :],
                        )
                    )
                continue
            if line.startswith(_STRUCTURAL_PREFIX):
                structural.append(
                    StructuralValidatorFinding(
                        file=display_file,
                        msgctxt=msgctxt,
                        reasons=line[len(_STRUCTURAL_PREFIX) :],
                    )
                )
    return fuzzy_count, structural, llm_count, llm


def _resolve_po_path(
    target_rel: str, *, target_dir: Path, po_dir: Optional[Path]
) -> Path:
    """Return the per-document PO path for a target Markdown rel-path.

    Mirrors :meth:`MarkdownProcessor.process_directory`'s default
    layout: PO is at ``po_root / target_rel.with_suffix(".po")``.
    ``po_root`` is the ``--po-dir`` override when supplied, else
    ``target_dir``. Mirror layout only — translate-paths layouts whose
    PO is keyed off ``src_rel`` instead of ``tgt_rel`` are out of
    scope here (see module docstring).
    """
    po_root = po_dir if po_dir is not None else target_dir
    return po_root / Path(target_rel).with_suffix(".po")


def _posix_key(raw: str) -> str:
    """Normalise a ``file`` field from a lint finding to a POSIX rel-path key.

    ``cli_lint.lint_directory`` stores relative paths via
    ``str(path.relative_to(...))``, which on Windows yields backslash-
    separated strings like ``guide\\doc.md``. validate-dir's target
    rel-paths come from ``Path.as_posix()`` (forward slashes), so a
    naive equality lookup would silently drop every nested-directory
    finding on Windows. Replacing the OS separator before bucketing
    keeps the lookup correct on every platform.
    """
    return raw.replace("\\", "/")


def _build_lint_index(
    report: LintReport,
) -> Tuple[dict[str, List[ResidueFinding]], dict[str, List[DanglingFinding]]]:
    """Bucket a :class:`LintReport` by its ``file`` field.

    The ``ResidueFinding`` / ``DanglingFinding`` ``file`` strings are
    normalised to POSIX via :func:`_posix_key` so the bucket keys
    match the rel-paths produced by :func:`_scan_markdown_rels` on
    every platform — see the helper's docstring for the Windows
    motivation.
    """
    residue_by_file: dict[str, List[ResidueFinding]] = {}
    dangling_by_file: dict[str, List[DanglingFinding]] = {}
    for r in report.residue:
        residue_by_file.setdefault(_posix_key(r.file), []).append(r)
    for d in report.dangling:
        dangling_by_file.setdefault(_posix_key(d.file), []).append(d)
    return residue_by_file, dangling_by_file


def validate_directory(
    target_dir: Path | str,
    source_dir: Path | str,
    *,
    po_dir: Optional[Path | str] = None,
    target_lang: Optional[str] = None,
    include_llm_validator: bool = False,
    include_lint: bool = False,
) -> ValidateDirReport:
    """Build a whole-tree validation report.

    Args:
        target_dir: Directory of translated Markdown files. Must
            exist and be a directory.
        source_dir: Source directory the translation ran against.
            Used for the mirror-layout cross-reference between
            source rel-paths and target rel-paths.
        po_dir: Optional override for the per-document PO root.
            Defaults to ``target_dir`` (matching the layout
            ``process_directory`` writes when ``--po-dir`` was not
            used).
        target_lang: BCP 47 locale of the translated tree. Required
            when ``include_lint`` is true (the T-19 residue scan
            needs it to score script families); ignored otherwise.
        include_llm_validator: When true, materialise the
            ``validator: llm: <reason>`` tcomment lines as
            :class:`LLMValidatorFinding` entries on the per-file
            summary. Structural findings are always counted; their
            full text is also surfaced for parity in JSON output.
        include_lint: When true, run the T-19 lint scan over the
            target tree (with ``source_dir`` as the optional
            ``--source-root`` for dangling-reference resolution) and
            fold residue / dangling findings onto the matching
            per-file summary.

    Returns:
        :class:`ValidateDirReport` with sorted per-file summaries,
        cross-reference issues, and aggregate counters.

    Raises:
        FileNotFoundError: ``target_dir`` or ``source_dir`` does
            not exist.
        NotADirectoryError: any of the supplied roots exists but is
            not a directory.
        ValueError: ``include_lint=True`` without ``target_lang``.
    """
    target_path = Path(target_dir)
    source_path = Path(source_dir)
    po_root: Optional[Path] = Path(po_dir) if po_dir is not None else None

    if not target_path.exists():
        raise FileNotFoundError(
            f"target directory does not exist: {target_path}"
        )
    if not target_path.is_dir():
        raise NotADirectoryError(
            f"target path is not a directory: {target_path}"
        )
    if not source_path.exists():
        raise FileNotFoundError(
            f"source directory does not exist: {source_path}"
        )
    if not source_path.is_dir():
        raise NotADirectoryError(
            f"source path is not a directory: {source_path}"
        )
    if po_root is not None and not po_root.is_dir():
        raise NotADirectoryError(
            f"po-dir path is not a directory: {po_root}"
        )
    if include_lint and not target_lang:
        raise ValueError(
            "--include-lint requires --target <lang>: the T-19 residue "
            "scan needs a target locale to score detected script "
            "families."
        )

    target_rels = _scan_markdown_rels(target_path)
    source_rels = _scan_markdown_rels(source_path)

    residue_by_file: dict[str, List[ResidueFinding]] = {}
    dangling_by_file: dict[str, List[DanglingFinding]] = {}
    lint_scanned_rels: set[str] = set()
    if include_lint:
        lint_report = lint_directory(
            target_path,
            target_lang=target_lang or "",
            source_root=source_path,
        )
        residue_by_file, dangling_by_file = _build_lint_index(lint_report)
        # Mirror cli_lint.lint_directory's discovery rule so the
        # per-file ``lint_scanned`` flag is honest about coverage.
        # T-19 uses ``rglob("*.md")``, which is CASE-SENSITIVE on
        # POSIX filesystems; uppercase ``.MD`` siblings appear in
        # ``target_rels`` (because ``_scan_markdown_rels`` matches
        # ``suffix.lower()`` like the rest of the pipeline) but the
        # lint pass never reads them. Recording the lint-covered set
        # lets the per-file row distinguish "scanned and clean" from
        # "not scanned" so a clean report does not give false
        # reassurance on those files.
        for md in target_path.rglob("*.md"):
            try:
                if not md.is_file():
                    continue
            except OSError:
                continue
            try:
                lint_scanned_rels.add(md.relative_to(target_path).as_posix())
            except ValueError:
                continue

    files: List[FileSummary] = []
    for target_rel in sorted(target_rels):
        po_path = _resolve_po_path(
            target_rel, target_dir=target_path, po_dir=po_root
        )
        po_present = po_path.is_file()
        po_rel: Optional[str] = None
        if po_present:
            try:
                po_rel = po_path.relative_to(
                    po_root if po_root is not None else target_path
                ).as_posix()
            except ValueError:
                po_rel = str(po_path)
        fuzzy = 0
        structural_findings: List[StructuralValidatorFinding] = []
        llm_findings: List[LLMValidatorFinding] = []
        llm_count = 0
        if po_present:
            (
                fuzzy,
                structural_findings,
                llm_count,
                llm_findings,
            ) = _parse_po_findings(
                po_path,
                display_file=target_rel,
                include_llm=include_llm_validator,
            )

        files.append(
            FileSummary(
                target_file=target_rel,
                po_file=po_rel,
                source_present=target_rel in source_rels,
                fuzzy_count=fuzzy,
                structural_count=len(structural_findings),
                llm_count=llm_count,
                structural_findings=tuple(structural_findings),
                llm_findings=tuple(llm_findings),
                residue=tuple(residue_by_file.get(target_rel, ())),
                dangling=tuple(dangling_by_file.get(target_rel, ())),
                lint_scanned=include_lint and target_rel in lint_scanned_rels,
            )
        )

    cross: List[CrossReferenceIssue] = []
    for rel in sorted(source_rels - target_rels):
        cross.append(
            CrossReferenceIssue(kind="source-without-target", path=rel)
        )
    for rel in sorted(target_rels - source_rels):
        cross.append(
            CrossReferenceIssue(kind="target-without-source", path=rel)
        )

    return ValidateDirReport(
        target_dir=str(target_path),
        source_dir=str(source_path),
        files=tuple(files),
        cross_reference=tuple(cross),
        llm_validator_ran=include_llm_validator,
        lint_ran=include_lint,
    )


def format_human_report(report: ValidateDirReport) -> str:
    """Render ``report`` as the default human-readable summary.

    Layout: a per-file table (one bullet line per file with the
    headline counters, followed by indented sub-bullets when any
    sub-finding category is non-empty), then the cross-reference
    section, then the aggregate counters. The layout intentionally
    mirrors :func:`mdpo_llm.cli_lint.format_human_report` so an
    operator who already reads ``mdpo-llm lint`` output recognises
    the shape immediately.
    """
    lines: List[str] = []
    agg = report.aggregate()
    lines.append(f"Scanned {agg['files_scanned']} markdown file(s).")
    lines.append(f"PO files found: {agg['po_files_scanned']}.")

    # Surface files whose lint pass was skipped due to the case-sensitive
    # ``*.md`` glob (uppercase ``.MD`` siblings) so the report does not
    # silently imply they were checked. Reported once at the top of the
    # findings section regardless of whether the file itself has any
    # other finding, otherwise a clean tcomment / fuzzy row could hide
    # the coverage gap entirely.
    if report.lint_ran:
        lint_skipped = [f for f in report.files if not f.lint_scanned]
        if lint_skipped:
            lines.append("")
            lines.append(
                f"Lint coverage gap ({len(lint_skipped)} file(s) not "
                "scanned — case-sensitive '*.md' only):"
            )
            for f in lint_skipped:
                lines.append(f"  {f.target_file}")

    files_with_findings = [f for f in report.files if f.has_findings()]
    if files_with_findings:
        lines.append("")
        lines.append(f"Per-file findings ({len(files_with_findings)}):")
        for f in files_with_findings:
            header_bits = [
                f"fuzzy={f.fuzzy_count}",
                f"structural={f.structural_count}",
                f"llm={f.llm_count}",
            ]
            if report.lint_ran:
                header_bits.append(f"residue={len(f.residue)}")
                header_bits.append(f"dangling={len(f.dangling)}")
                if not f.lint_scanned:
                    header_bits.append("lint=skipped")
            lines.append(f"  {f.target_file}: " + " ".join(header_bits))
            for s in f.structural_findings:
                ctx = f"@{s.msgctxt}" if s.msgctxt else ""
                lines.append(f"    structural{ctx}: {s.reasons}")
            for l in f.llm_findings:
                ctx = f"@{l.msgctxt}" if l.msgctxt else ""
                lines.append(f"    llm{ctx}: {l.reason}")
            for r in f.residue:
                langs = ",".join(r.languages)
                lines.append(
                    f"    residue:{r.line}: [{langs}] {r.text}"
                )
            for d in f.dangling:
                lines.append(
                    f"    dangling:{d.line}: missing: {d.reference}"
                )

    if report.cross_reference:
        lines.append("")
        lines.append(
            f"Cross-reference issues ({len(report.cross_reference)}):"
        )
        for x in report.cross_reference:
            lines.append(f"  [{x.kind}] {x.path}")

    lines.append("")
    lines.append("Aggregate:")
    for key in (
        "files_scanned",
        "po_files_scanned",
        "total_fuzzy",
        "total_structural_findings",
        "total_llm_validator_findings",
        "total_residue",
        "total_dangling",
        "total_cross_reference_issues",
        "lint_coverage_gap",
    ):
        lines.append(f"  {key}: {agg[key]}")

    lines.append("")
    if report.has_findings():
        lines.append("validate-dir FOUND ISSUES.")
    else:
        lines.append(
            "validate-dir PASSED — no fuzzy entries, validator findings, "
            "lint hits, or cross-reference mismatches."
        )
    return "\n".join(lines)


def add_validate_dir_subparser(
    sub: "argparse._SubParsersAction[argparse.ArgumentParser]",
) -> argparse.ArgumentParser:
    """Register the ``validate-dir`` subcommand on the top-level argparse.

    Kept as a public helper so :mod:`mdpo_llm.__main__` can attach the
    subparser without importing the implementation symbols
    individually, matching :func:`add_lint_subparser` and
    :func:`add_cleanup_subparser`.
    """
    p = sub.add_parser(
        "validate-dir",
        help=(
            "Whole-tree validation report: per-file fuzzy / validator "
            "counts + cross-reference + optional T-19 lint folding."
        ),
        description=(
            "Walk a directory of translated markdown files and aggregate "
            "fuzzy counts, structural / LLM validator tcomment lines, "
            "mirror-layout cross-reference issues, and (with "
            "--include-lint) the T-19 residue / dangling-reference scan. "
            "Read-only: no PO writes, no LLM calls."
        ),
    )
    p.add_argument(
        "target_dir",
        help="Translated markdown directory to validate (scanned recursively).",
    )
    p.add_argument(
        "--source",
        dest="source_dir",
        required=True,
        help=(
            "Source directory the translation ran against. Required for "
            "the cross-reference section; also used as the lint "
            "scanner's --source-root when --include-lint is set."
        ),
    )
    p.add_argument(
        "--po-dir",
        dest="po_dir",
        type=Path,
        default=None,
        help=(
            "Optional override for the per-document PO directory. "
            "Defaults to TARGET_DIR (matching the default layout "
            "translate-dir writes when --po-dir was not used)."
        ),
    )
    p.add_argument(
        "--target",
        default=None,
        help=(
            "BCP 47 locale of the translated tree (e.g. 'en', 'ko'). "
            "Required only with --include-lint; the residue scan needs "
            "a target locale to score detected script families."
        ),
    )
    p.add_argument(
        "--include-llm-validator",
        dest="include_llm_validator",
        action="store_true",
        help=(
            "Materialise 'validator: llm: <reason>' tcomment lines on "
            "the per-file summary. Structural validator findings are "
            "always counted; this flag adds the T-16 LLM grader's "
            "rejection reasons by file."
        ),
    )
    p.add_argument(
        "--include-lint",
        dest="include_lint",
        action="store_true",
        help=(
            "Fold T-19 lint findings (source-language residue + "
            "dangling doc references; extensions: "
            + " ".join("." + e for e in LINT_EXTENSIONS)
            + ") onto the matching per-file summary. Requires --target."
        ),
    )
    p.add_argument(
        "--json",
        dest="json_out",
        action="store_true",
        help=(
            "Emit the report as JSON to stdout instead of the "
            "human-readable summary. Schema: "
            "{target_dir, source_dir, llm_validator_ran, lint_ran, "
            "files: [{target_file, po_file, source_present, "
            "fuzzy_count, structural_count, structural_findings, "
            "llm_findings, residue, dangling}], "
            "cross_reference: [{kind, path}], aggregate: {...}}."
        ),
    )
    p.add_argument(
        "--exit-non-zero-on-findings",
        dest="exit_non_zero_on_findings",
        action="store_true",
        help=(
            "Exit with code 1 when any finding is reported (default: "
            "always exit 0 unless a usage error occurs). Use this in CI "
            "to fail the build on a non-clean tree."
        ),
    )
    p.set_defaults(func=cmd_validate_dir)
    return p


def cmd_validate_dir(args: argparse.Namespace) -> int:
    """``mdpo-llm validate-dir`` entry point.

    Exit code contract:
      * ``2`` — usage error (missing / non-directory ``target_dir``,
        ``--source``, ``--po-dir``, or ``--include-lint`` without
        ``--target``). Surfaced before any filesystem walk.
      * ``1`` — findings reported AND ``--exit-non-zero-on-findings``
        was passed. The scanner itself succeeded; this is a
        configurable CI-failure signal, not a runtime error.
      * ``0`` — otherwise (no findings, or findings without the
        opt-in flag).
    """
    try:
        report = validate_directory(
            target_dir=args.target_dir,
            source_dir=args.source_dir,
            po_dir=getattr(args, "po_dir", None),
            target_lang=getattr(args, "target", None),
            include_llm_validator=getattr(args, "include_llm_validator", False),
            include_lint=getattr(args, "include_lint", False),
        )
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if getattr(args, "json_out", False):
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(format_human_report(report))

    if (
        getattr(args, "exit_non_zero_on_findings", False)
        and report.has_findings()
    ):
        return 1
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Standalone entry point: ``python -m mdpo_llm.cli_validate_dir …``.

    Mirrors the ``mdpo-llm validate-dir`` subcommand surface so the
    module can be driven directly without going through the top-level
    parser — convenient for ad-hoc runs and for tests that exercise
    the CLI surface in isolation.
    """
    parser = argparse.ArgumentParser(prog="mdpo-llm-validate-dir")
    sub = parser.add_subparsers(dest="command", required=True)
    add_validate_dir_subparser(sub)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
