"""Command-line entry point for mdpo-llm.

Usage:
    python -m mdpo_llm translate SOURCE TARGET [options]
    python -m mdpo_llm translate-dir SOURCE_DIR TARGET_DIR [options]
    python -m mdpo_llm estimate SOURCE [options]
    python -m mdpo_llm report SOURCE PO [options]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from . import __version__
from .processor import MarkdownProcessor


def _add_shared_flags(parser: argparse.ArgumentParser) -> None:
    """Flags used by every subcommand that constructs a MarkdownProcessor."""
    parser.add_argument(
        "--model",
        required=True,
        help="LiteLLM model string (e.g. gpt-4o, anthropic/claude-sonnet-4-5-20250929).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=40,
        help="Max entries per batched call (0 disables batching).",
    )
    parser.add_argument(
        "--batch-max-chars",
        type=int,
        default=8000,
        help="Soft cap on source characters per batch.",
    )
    parser.add_argument(
        "--max-reference-pairs",
        type=int,
        default=5,
        help="Max similar reference pairs per entry/batch.",
    )


def _add_translate_flags(parser: argparse.ArgumentParser) -> None:
    """Flags that only make sense when actually issuing translations."""
    _add_shared_flags(parser)
    parser.add_argument(
        "--target",
        required=True,
        help="BCP 47 target locale (e.g. ko, ja, zh-CN).",
    )
    parser.add_argument(
        "--validation",
        choices=["off", "conservative", "strict"],
        default="off",
        help="Post-translation structural validation.",
    )
    parser.add_argument(
        "--glossary",
        type=Path,
        default=None,
        help="Path to a JSON glossary file.",
    )
    parser.add_argument(
        "--extra-instructions",
        type=str,
        default=None,
        help="Additional prompt instructions (tone, domain, audience).",
    )
    parser.add_argument(
        "--prompt-cache",
        action="store_true",
        help="Mark the stable system prefix as cacheable.",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="After translating, copy msgstr back to msgid.",
    )
    parser.add_argument(
        "--json-receipt",
        type=Path,
        default=None,
        metavar="PATH",
        help="Also write the run receipt (tokens, cost, duration) as JSON to PATH.",
    )


def _add_estimate_flags(parser: argparse.ArgumentParser) -> None:
    """Estimate never issues API calls, so locale/validation flags are irrelevant.

    ``--target`` is accepted but optional (the processor constructor still
    requires a locale and we default to ``en`` when callers omit it).
    """
    _add_shared_flags(parser)
    parser.add_argument(
        "--target",
        default="en",
        help="BCP 47 locale (optional — unused by the estimator, but the "
        "processor constructor requires one).",
    )


def _build_processor(args: argparse.Namespace) -> MarkdownProcessor:
    return MarkdownProcessor(
        model=args.model,
        target_lang=args.target,
        batch_size=args.batch_size,
        batch_max_chars=args.batch_max_chars,
        validation=getattr(args, "validation", "off"),
        max_reference_pairs=args.max_reference_pairs,
        extra_instructions=getattr(args, "extra_instructions", None),
        glossary_path=getattr(args, "glossary", None),
        enable_prompt_cache=getattr(args, "prompt_cache", False),
    )


def _print_result(obj: Any) -> None:
    if hasattr(obj, "to_dict"):
        obj = obj.to_dict()
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def _write_receipt_json(receipt: Any, path: Path) -> None:
    """Dump a receipt dataclass / dict to ``path`` as JSON."""
    payload = receipt.to_dict() if hasattr(receipt, "to_dict") else dict(receipt)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _emit_receipt(result: Any, json_receipt_path: Optional[Path]) -> None:
    """Print the human-readable receipt block and optionally dump JSON.

    No-op when ``result`` has no ``receipt`` attribute / the receipt is
    ``None`` (e.g. legacy callers that bypass the standard processor flow).
    """
    receipt = getattr(result, "receipt", None) if not isinstance(result, dict) else result.get("receipt")
    if receipt is None:
        return
    # ``receipt`` is a ``Receipt`` instance, but hasattr("render") keeps this
    # resilient to pure-dict round-trips (e.g. after ``json.loads``).
    if hasattr(receipt, "render"):
        print(receipt.render(), file=sys.stderr)
    if json_receipt_path is not None:
        _write_receipt_json(receipt, json_receipt_path)


def cmd_translate(args: argparse.Namespace) -> int:
    processor = _build_processor(args)
    json_receipt = getattr(args, "json_receipt", None)
    try:
        result = processor.process_document(
            Path(args.source),
            Path(args.target_file),
            Path(args.po) if args.po else None,
            inplace=args.inplace,
        )
    except BaseException as exc:
        # Even a mid-run failure may have billed tokens; surface what we
        # have so the operator / CI sees the real cost of the failed run.
        partial = getattr(exc, "partial_receipt", None)
        if partial is not None:
            if hasattr(partial, "render"):
                print(partial.render(), file=sys.stderr)
            if json_receipt is not None:
                _write_receipt_json(partial, json_receipt)
        raise
    _print_result(result)
    _emit_receipt(result, json_receipt)
    stats = result.translation_stats
    return 1 if (stats.failed or stats.validation_failed) else 0


def cmd_translate_dir(args: argparse.Namespace) -> int:
    processor = _build_processor(args)
    json_receipt = getattr(args, "json_receipt", None)
    try:
        result = processor.process_directory(
            Path(args.source_dir),
            Path(args.target_dir),
            Path(args.po_dir) if args.po_dir else None,
            glob=args.glob,
            inplace=args.inplace,
            max_workers=args.max_workers,
        )
    except BaseException as exc:
        partial = getattr(exc, "partial_receipt", None)
        if partial is not None:
            if hasattr(partial, "render"):
                print(partial.render(), file=sys.stderr)
            if json_receipt is not None:
                _write_receipt_json(partial, json_receipt)
        raise
    summary: Dict[str, Any] = {
        "source_dir": result.source_dir,
        "target_dir": result.target_dir,
        "po_dir": result.po_dir,
        "files_processed": result.files_processed,
        "files_failed": result.files_failed,
        "files_skipped": result.files_skipped,
    }
    _print_result(summary)
    _emit_receipt(result, json_receipt)
    return 1 if result.files_failed else 0


def cmd_estimate(args: argparse.Namespace) -> int:
    processor = _build_processor(args)
    estimate = processor.estimate(
        Path(args.source), Path(args.po) if args.po else None
    )
    _print_result(estimate)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    processor = MarkdownProcessor(model="unused", target_lang=args.target or "en")
    report = processor.export_report(Path(args.source), Path(args.po))
    sys.stdout.write(report)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mdpo-llm",
        description="Incremental Markdown translation with LLMs and PO files.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable INFO-level logging."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_tr = sub.add_parser("translate", help="Translate a single markdown file.")
    _add_translate_flags(p_tr)
    p_tr.add_argument("source", help="Source markdown file.")
    p_tr.add_argument("target_file", help="Target markdown file.")
    p_tr.add_argument("--po", default=None, help="PO file path (default: target with .po).")
    p_tr.set_defaults(func=cmd_translate)

    p_dir = sub.add_parser("translate-dir", help="Translate a directory tree.")
    _add_translate_flags(p_dir)
    p_dir.add_argument("source_dir", help="Source directory.")
    p_dir.add_argument("target_dir", help="Target directory.")
    p_dir.add_argument("--po-dir", default=None, help="PO output directory.")
    p_dir.add_argument("--glob", default="**/*.md", help="Glob pattern for markdown files.")
    p_dir.add_argument("--max-workers", type=int, default=4, help="Concurrent file workers.")
    p_dir.set_defaults(func=cmd_translate_dir)

    p_est = sub.add_parser("estimate", help="Estimate pending blocks and tokens (no API calls).")
    _add_estimate_flags(p_est)
    p_est.add_argument("source", help="Source markdown file.")
    p_est.add_argument("--po", default=None, help="PO file path.")
    p_est.set_defaults(func=cmd_estimate)

    p_rep = sub.add_parser("report", help="Print a translation coverage report for an existing PO.")
    p_rep.add_argument("source", help="Source markdown file.")
    p_rep.add_argument("po", help="PO file path.")
    p_rep.add_argument("--target", default=None, help="BCP 47 locale (unused for reporting).")
    p_rep.set_defaults(func=cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
