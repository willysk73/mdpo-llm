"""Read-only lint scanner for translated markdown trees (T-19).

``mdpo-llm lint <directory> --target <lang>`` walks a directory of
already-translated markdown files and reports two classes of issue
without ever issuing an LLM call or touching a PO file:

1. **Source-language residue** — lines whose text still carries
   characters from a language other than ``--target``. The detection
   reuses :func:`mdpo_llm.language.detect_languages` so the script
   ranges stay consistent with the rest of the pipeline (residue
   post-pass, validator). A line is residue-bearing when its detected
   primary-subtag set, with the target's primary subtag removed, is
   non-empty.

2. **Dangling doc references** — backticked or angle-bracketed
   filenames whose basename is not present in either the scanned
   target tree or, optionally, the ``--source-root`` tree.
    only covered PDF refs; this scanner generalises to
   the common artefact extensions (`pdf png jpg jpeg gif svg md csv
   json xlsx docx`). URLs (anything containing ``://``) are skipped
   because their existence cannot be checked on disk.

The scanner is pure observability: no LLM calls, no PO writes, no
mutation of the scanned tree. It is meant to surface follow-up work
for human reviewers and as a CI gate via ``--exit-non-zero-on-findings``.

The  reference implementation drove its
known-filename set from ``filename_map.json``; per BOARD direction
mdpo-llm replaces that artefact with ``_paths.po`` / ``path_map.json``,
so this scanner derives the known set straight from the filesystem
instead of reintroducing a parallel mapping file.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

from .language import _resolve_primary, detect_languages
from .residue_pass import SOURCE_LANG_PATTERNS


# Primary BCP 47 subtags eligible to be flagged as residue. Reuses the
# residue-pass source-language set ({"ko", "ja", "zh"}) so the lint
# semantics stay consistent with the rest of the pipeline AND we avoid
# the language module's coarse ``en`` pattern (``[A-Za-z]``), which
# would otherwise classify ordinary French / German / Spanish prose as
# English residue and yield universal false positives under
# ``--target fr`` / etc. The trade-off is intentional: residue
# detection focuses on the non-Latin scripts where leakage is
# visually unambiguous; English / other Latin-script leakage in a
# Korean tree is left to the existing structural validator (T-16
# language_stability) where script-vs-language disambiguation is
# tractable.
RESIDUE_LANG_SUBTAGS: frozenset[str] = frozenset(SOURCE_LANG_PATTERNS.keys())


# Per-target suppression of script subtags whose Unicode range overlaps
# the target's natural script. ``zh``'s pattern is just the CJK
# Unified Ideographs block, which Japanese kanji uses natively — so
# under ``--target ja`` every kanji-bearing Japanese line would
# otherwise be reported as ``zh`` residue. The residue post-pass
# handles the equivalent edge case in
# :func:`mdpo_llm.residue_pass._resolve_residue_pattern`; we mirror
# that behaviour here so the lint stays target-aware instead of
# producing systematic false positives on CJK-overlap pairs.
#
# Pairs (read as "target → suppressed residue subtags"):
#   * ``ja`` → ``{"zh"}``: kanji is normal Japanese.
#
# Two related pairs need no entry here:
#   * ``zh → zh``: the leading ``detected.discard(target_primary)``
#     already removes ``zh`` for a Chinese target.
#   * ``ja → ja`` for kana: ``detected.discard("ja")`` removes it.
#     A pure-kana ``ja`` line under ``--target zh`` IS legitimate
#     residue (kana never appears in Chinese), so we deliberately
#     leave that direction unsuppressed.
_TARGET_SUPPRESSED_RESIDUE: dict[str, frozenset[str]] = {
    "ja": frozenset({"zh"}),
}


# Doc artefact extensions tracked for dangling-reference detection
# (lowercased, no leading dot). Generalised from 's
# PDF-only scan to cover the common documentation attachments
# mdpo-llm operators ship alongside markdown trees.
LINT_EXTENSIONS: Tuple[str, ...] = (
    "pdf",
    "png",
    "jpg",
    "jpeg",
    "gif",
    "svg",
    "md",
    "csv",
    "json",
    "xlsx",
    "docx",
)


_EXT_ALTERNATION = "|".join(LINT_EXTENSIONS)

# Inline backtick reference: a single-backtick run on one line whose
# trimmed body ends with one of the tracked extensions. ``[^`\n]+?``
# refuses backticks inside the body so multi-backtick fences cannot
# overlap, and the lazy quantifier keeps the match tight against the
# closing backtick. Case-insensitive so ``FILE.PDF`` shaped refs are
# still caught.
_REF_BACKTICK_RE = re.compile(
    rf"`([^`\n]+?\.(?:{_EXT_ALTERNATION}))`",
    re.IGNORECASE,
)

# Autolink-style angle-bracket reference: ``<file.ext>``. ``[^<>\s]``
# rejects whitespace and nested brackets so a stray ``<`` in prose
# does not consume across a line; URLs (``http://…``) match too but
# are filtered downstream via :func:`_is_url`.
_REF_ANGLE_RE = re.compile(
    rf"<([^<>\s]+?\.(?:{_EXT_ALTERNATION}))>",
    re.IGNORECASE,
)

# Truncation width applied to the human-readable preview of a
# residue-bearing line. Matches the  reference so
# tooling parsing the human report sees the same cap regardless of
# which implementation produced it.
_LINE_PREVIEW_CHARS = 80


@dataclass(frozen=True)
class ResidueFinding:
    """One markdown line carrying non-target-language characters."""

    file: str
    line: int
    text: str
    languages: Tuple[str, ...]


@dataclass(frozen=True)
class DanglingFinding:
    """One doc-artefact reference whose basename is missing from every scanned tree."""

    file: str
    line: int
    reference: str


@dataclass(frozen=True)
class LintReport:
    """Aggregate result of one :func:`lint_directory` call."""

    files_scanned: int
    residue: Tuple[ResidueFinding, ...]
    dangling: Tuple[DanglingFinding, ...]

    def has_findings(self) -> bool:
        return bool(self.residue or self.dangling)

    def to_dict(self) -> dict:
        return {
            "files_scanned": self.files_scanned,
            "residue": [
                {
                    "file": f.file,
                    "line": f.line,
                    "text": f.text,
                    "languages": list(f.languages),
                }
                for f in self.residue
            ],
            "dangling": [
                {
                    "file": d.file,
                    "line": d.line,
                    "reference": d.reference,
                }
                for d in self.dangling
            ],
        }


def _collect_known_filenames(roots: Iterable[Path]) -> set[str]:
    """Return the lowercased basenames of every tracked-extension file under each root.

    Roots that do not exist or are not directories are skipped silently
    — the caller may pass an optional ``--source-root`` that the user
    omitted, and a missing optional root should not be a fatal error
    here. Symlink loops are avoided by ``rglob``'s default behaviour
    (the stdlib refuses to descend through cycles).
    """
    known: set[str] = set()
    for root in roots:
        if root is None:
            continue
        if not root.exists() or not root.is_dir():
            continue
        for path in root.rglob("*"):
            try:
                if not path.is_file():
                    continue
            except OSError:
                # Broken symlinks raise on ``is_file()``; skip rather
                # than abort the whole scan for one unreachable entry.
                continue
            ext = path.suffix.lstrip(".").lower()
            if ext in LINT_EXTENSIONS:
                known.add(path.name.lower())
    return known


def _iter_references(line: str) -> Iterator[str]:
    """Yield every backticked or angle-bracketed doc-artefact reference on ``line``.

    Order is backticked first, then angle-bracketed — stable so that
    JSON output is reproducible regardless of which pattern matched
    first lexically. Duplicates within the same line are NOT
    de-duplicated here so the caller still records each location.
    """
    for m in _REF_BACKTICK_RE.finditer(line):
        yield m.group(1)
    for m in _REF_ANGLE_RE.finditer(line):
        yield m.group(1)


def _is_url(token: str) -> bool:
    """Return True when ``token`` looks like a URL we cannot existence-check.

    The check is intentionally loose: ``://`` is the only universally
    reliable signal for "this is a remote resource, not a local
    filename". Scheme-relative refs (``//example.com/x.pdf``) are
    rare in markdown trees and would be a false positive on a literal
    relative path that happens to start with ``//`` — accept the
    edge case rather than over-engineer it.
    """
    return "://" in token


def _detect_residue_languages(line: str, target_primary: str) -> Tuple[str, ...]:
    """Return non-target residue subtags detected in ``line``, sorted for stable output.

    Intersected with :data:`RESIDUE_LANG_SUBTAGS` so only the
    non-Latin scripts the rest of the pipeline treats as source
    languages can be flagged — preventing the false positives that a
    raw ``detect_languages`` call would produce under any
    Latin-script target (``en``, ``fr``, ``de``, ``es``, ...) where
    the language module's ``en`` pattern (``[A-Za-z]``) matches
    ordinary target-language prose.
    """
    detected = detect_languages(line) & RESIDUE_LANG_SUBTAGS
    detected.discard(target_primary)
    detected -= _TARGET_SUPPRESSED_RESIDUE.get(target_primary, frozenset())
    return tuple(sorted(detected))


def lint_file(
    path: Path,
    *,
    target_lang: str,
    known_filenames: frozenset[str],
    relative_to: Optional[Path] = None,
) -> Tuple[List[ResidueFinding], List[DanglingFinding]]:
    """Lint a single markdown file.

    Returns ``(residue_findings, dangling_findings)``. Unreadable
    files (binary content, permission denied, transient I/O errors)
    yield two empty lists — the scan is best-effort and a single
    unreadable file MUST NOT abort the directory walk.
    """
    target_primary = _resolve_primary(target_lang)
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return [], []

    display = (
        str(path.relative_to(relative_to)) if relative_to is not None else str(path)
    )
    residue: List[ResidueFinding] = []
    dangling: List[DanglingFinding] = []
    for line_num, line in enumerate(content.splitlines(), start=1):
        langs = _detect_residue_languages(line, target_primary)
        if langs:
            residue.append(
                ResidueFinding(
                    file=display,
                    line=line_num,
                    text=line.strip()[:_LINE_PREVIEW_CHARS],
                    languages=langs,
                )
            )
        for ref in _iter_references(line):
            if _is_url(ref):
                continue
            basename = Path(ref).name.lower()
            if basename and basename not in known_filenames:
                dangling.append(
                    DanglingFinding(
                        file=display,
                        line=line_num,
                        reference=ref,
                    )
                )
    return residue, dangling


def lint_directory(
    target_dir: Path,
    *,
    target_lang: str,
    source_root: Optional[Path] = None,
) -> LintReport:
    """Lint every ``*.md`` file under ``target_dir``.

    The known-filename set is computed once from ``target_dir`` and
    the optional ``source_root`` together — a reference is considered
    resolved when its basename exists in either tree, matching the
    brief's "file present in source OR file present in target" rule.

    Markdown files are processed in sorted path order so JSON output
    is reproducible across runs and filesystems with different walk
    ordering.
    """
    roots: List[Path] = [target_dir]
    if source_root is not None:
        roots.append(source_root)
    known_filenames = frozenset(_collect_known_filenames(roots))

    md_files = sorted(target_dir.rglob("*.md"))
    all_residue: List[ResidueFinding] = []
    all_dangling: List[DanglingFinding] = []
    scanned = 0
    for md in md_files:
        try:
            if not md.is_file():
                continue
        except OSError:
            continue
        scanned += 1
        residue, dangling = lint_file(
            md,
            target_lang=target_lang,
            known_filenames=known_filenames,
            relative_to=target_dir,
        )
        all_residue.extend(residue)
        all_dangling.extend(dangling)

    return LintReport(
        files_scanned=scanned,
        residue=tuple(all_residue),
        dangling=tuple(all_dangling),
    )


def format_human_report(report: LintReport, target_lang: str) -> str:
    """Render ``report`` as the default human-readable report.

    Stable layout: one header line with the scan total, then a
    residue block (when non-empty) followed by a dangling block
    (when non-empty), then a closing PASSED / ISSUES line. Each
    finding line is shaped ``  <file>:<line>: <detail>`` so editor
    "jump to line" parsers can navigate the report directly.
    """
    lines: List[str] = []
    lines.append(f"Scanned {report.files_scanned} markdown file(s).")

    if report.residue:
        lines.append("")
        lines.append(
            f"Found {len(report.residue)} line(s) with non-target language "
            f"residue (target={target_lang}):"
        )
        for f in report.residue:
            langs = ",".join(f.languages)
            lines.append(f"  {f.file}:{f.line}: [{langs}] {f.text}")

    if report.dangling:
        lines.append("")
        lines.append(f"Found {len(report.dangling)} dangling reference(s):")
        for d in report.dangling:
            lines.append(f"  {d.file}:{d.line}: missing: {d.reference}")

    lines.append("")
    if report.has_findings():
        lines.append("Lint check FOUND ISSUES.")
    else:
        lines.append("Lint check PASSED — no residue or dangling references.")
    return "\n".join(lines)


def add_lint_subparser(sub: "argparse._SubParsersAction[argparse.ArgumentParser]") -> argparse.ArgumentParser:
    """Register the ``lint`` subcommand on the top-level argparse.

    Kept as a public helper so :mod:`mdpo_llm.__main__` can attach the
    subparser without importing the implementation symbols
    individually, and so library callers building their own CLI driver
    can reuse the same definition.
    """
    p = sub.add_parser(
        "lint",
        help=(
            "Read-only scanner: source-language residue + dangling doc "
            "references in a translated markdown tree."
        ),
        description=(
            "Walk a directory of translated markdown files and report "
            "two classes of issue: (1) lines containing characters from "
            "a language other than --target ('source-language residue') "
            "and (2) backticked or angle-bracketed doc artefact "
            "references whose basename does not exist in either the "
            "scanned tree or the optional --source-root. Read-only: no "
            "LLM calls, no PO writes."
        ),
    )
    p.add_argument(
        "directory",
        help="Directory of translated markdown files to lint (scanned recursively).",
    )
    p.add_argument(
        "--target",
        required=True,
        help=(
            "BCP 47 locale of the translated tree (e.g. 'en', 'ko'). "
            "Lines whose detected script set contains any subtag other "
            "than this locale's primary subtag are reported as "
            "source-language residue."
        ),
    )
    p.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help=(
            "Optional path to the SOURCE tree. Doc artefact references "
            "(extensions: " + " ".join("." + e for e in LINT_EXTENSIONS) + ") "
            "are considered resolved when their basename exists in "
            "either the target tree OR this source tree; anything not "
            "in either is flagged as a dangling reference. URLs "
            "(anything containing '://') are skipped."
        ),
    )
    p.add_argument(
        "--json",
        dest="json_out",
        action="store_true",
        help=(
            "Emit structured findings as JSON to stdout instead of the "
            "human-readable report. Schema: "
            "{files_scanned: int, residue: [{file, line, text, languages: [str]}], "
            "dangling: [{file, line, reference}]}."
        ),
    )
    p.add_argument(
        "--exit-non-zero-on-findings",
        dest="exit_non_zero_on_findings",
        action="store_true",
        help=(
            "Exit with code 1 when any finding is reported (default: "
            "always exit 0 unless a usage error occurs). Use this in CI "
            "to fail the build on residue or dangling references."
        ),
    )
    p.set_defaults(func=cmd_lint)
    return p


def cmd_lint(args: argparse.Namespace) -> int:
    """``mdpo-llm lint`` entry point (dispatched from :mod:`mdpo_llm.__main__`).

    Exit code contract:
      * ``2`` — usage error (missing / non-directory ``directory`` or
        ``--source-root``). Surfaced before any filesystem walk.
      * ``1`` — findings reported AND ``--exit-non-zero-on-findings``
        was passed. The scanner itself succeeded; this is a
        configurable CI-failure signal, not a runtime error.
      * ``0`` — otherwise (no findings, or findings without the
        opt-in flag).
    """
    directory = Path(args.directory)
    if not directory.exists():
        print(f"error: directory does not exist: {directory}", file=sys.stderr)
        return 2
    if not directory.is_dir():
        print(f"error: not a directory: {directory}", file=sys.stderr)
        return 2

    source_root: Optional[Path] = args.source_root
    if source_root is not None and not source_root.is_dir():
        print(
            f"error: --source-root is not a directory: {source_root}",
            file=sys.stderr,
        )
        return 2

    report = lint_directory(
        directory,
        target_lang=args.target,
        source_root=source_root,
    )

    if args.json_out:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(format_human_report(report, args.target))

    if args.exit_non_zero_on_findings and report.has_findings():
        return 1
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Standalone entry point: ``python -m mdpo_llm.cli_lint …``.

    Mirrors the contract of the ``mdpo-llm lint`` subcommand so the
    module can be invoked directly without going through the top-level
    parser — convenient for ad-hoc runs and for tests that want to
    drive the CLI surface without registering the full subcommand
    tree.
    """
    parser = argparse.ArgumentParser(prog="mdpo-llm-lint")
    sub = parser.add_subparsers(dest="command", required=True)
    add_lint_subparser(sub)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
