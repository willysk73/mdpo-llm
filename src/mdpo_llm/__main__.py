"""Command-line entry point for mdpo-llm.

Usage:
    python -m mdpo_llm translate SOURCE TARGET [options]
    python -m mdpo_llm translate-dir SOURCE_DIR TARGET_DIR [options]
    python -m mdpo_llm refine SOURCE REFINED [options]
    python -m mdpo_llm refine-dir SOURCE_DIR REFINED_DIR [options]
    python -m mdpo_llm estimate SOURCE [options]
    python -m mdpo_llm report SOURCE PO [options]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from . import __version__
from .processor import MarkdownProcessor, ProgressEvent


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
        "--batch-concurrency",
        type=int,
        default=1,
        help=(
            "EXPERIMENTAL. Max batches from the same file to send in "
            "parallel after the first batch has seeded the reference "
            "pool. Off by default (1 = deterministic sequential "
            "batching). Ignored when --batch-size=0 (sequential path)."
        ),
    )
    parser.add_argument(
        "--max-reference-pairs",
        type=int,
        default=5,
        help="Max similar reference pairs per entry/batch.",
    )


def _add_translate_flags(
    parser: argparse.ArgumentParser,
    *,
    target_help: str = "BCP 47 target locale (e.g. ko, ja, zh-CN).",
) -> None:
    """Flags that only make sense when actually issuing translations.

    ``target_help`` lets the ``refine`` subcommands override the
    ``--target`` blurb so users don't read it as "translate to this
    locale" when the value actually names the source/preserved language.
    """
    _add_shared_flags(parser)
    parser.add_argument(
        "--target",
        required=True,
        help=target_help,
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
        "--glossary-mode",
        choices=["instruction", "placeholder"],
        default="instruction",
        help=(
            "How to feed glossary terms to the LLM. 'instruction' "
            "(default) appends a glossary block to the system prompt. "
            "'placeholder' substitutes each term with an opaque "
            "\u27e6P:N\u27e7 token pre-call and restores the "
            "target-language form (or the original term for "
            "do-not-translate entries) post-call."
        ),
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
        help=(
            "[DEPRECATED — removed in v0.5] After translating, copy msgstr "
            "back to msgid. Use the `refine` subcommand (or `translate "
            "--refine-first`) instead to polish the source while keeping "
            "the original msgid intact."
        ),
    )
    parser.add_argument(
        "--json-receipt",
        type=Path,
        default=None,
        metavar="PATH",
        help="Also write the run receipt (tokens, cost, duration) as JSON to PATH.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help=(
            "Disable the progress bar even on a TTY. Progress is also "
            "auto-suppressed when stderr is not a TTY, under -v, or when "
            "MDPO_NO_PROGRESS is set."
        ),
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


def _build_processor(
    args: argparse.Namespace,
    progress_callback: Optional[Callable[[ProgressEvent], None]] = None,
    *,
    mode: str = "translate",
) -> MarkdownProcessor:
    return MarkdownProcessor(
        model=args.model,
        target_lang=args.target,
        batch_size=args.batch_size,
        batch_max_chars=args.batch_max_chars,
        batch_concurrency=getattr(args, "batch_concurrency", 1),
        validation=getattr(args, "validation", "off"),
        max_reference_pairs=args.max_reference_pairs,
        extra_instructions=getattr(args, "extra_instructions", None),
        glossary_path=getattr(args, "glossary", None),
        glossary_mode=getattr(args, "glossary_mode", "instruction"),
        enable_prompt_cache=getattr(args, "prompt_cache", False),
        progress_callback=progress_callback,
        mode=mode,
    )


def _progress_enabled(args: argparse.Namespace) -> bool:
    """Return True when a progress bar should render.

    Suppressed on non-TTY (CI, redirected output), when ``-v`` turns on
    structured logging (logs would interleave and scramble the bar),
    when ``NO_COLOR`` / ``MDPO_NO_PROGRESS`` are set (standard opt-outs),
    and when the user passed ``--no-progress``.
    """
    if getattr(args, "verbose", False):
        return False
    if getattr(args, "no_progress", False):
        return False
    if os.environ.get("MDPO_NO_PROGRESS"):
        return False
    try:
        return sys.stderr.isatty()
    except Exception:
        return False


class _RichFileProgress:
    """Progress hook for single-document ``translate``.

    One rich ``Progress`` with a single task tracking the document's
    batches / entries. Opens on ``document_start`` and closes on
    ``document_end`` (also on uncaught exceptions via the outer CLI
    ``finally`` — see :func:`_with_progress`).

    Rendered on stderr so it cannot corrupt the JSON result on stdout
    when callers redirect / pipe the command output.
    """

    def __init__(self) -> None:
        from rich.console import Console
        from rich.progress import (
            BarColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
        )

        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            transient=False,
            console=Console(stderr=True),
        )
        self._task_id: Optional[int] = None
        self._started = False

    def __call__(self, event: ProgressEvent) -> None:
        if event.kind == "document_start":
            if not self._started:
                self._progress.start()
                self._started = True
            desc = Path(event.path).name if event.path else "translate"
            self._task_id = self._progress.add_task(
                desc, total=max(event.total or 0, 1)
            )
            # Zero-work documents still render a filled bar so the user
            # sees "nothing to do" rather than a stuck empty bar.
            if not event.total:
                self._progress.update(self._task_id, completed=1)
        elif event.kind == "document_progress" and self._task_id is not None:
            self._progress.update(
                self._task_id, completed=event.index, total=event.total
            )
        elif event.kind == "document_end":
            self.close()

    def close(self) -> None:
        if self._started:
            self._progress.stop()
            self._started = False


class _RichDirectoryProgress:
    """Progress hook for ``translate-dir``.

    One bar for overall file completion; per-file document/batch events
    are ignored (too noisy across concurrent workers). Rendered on
    stderr so it cannot corrupt the JSON result on stdout when callers
    redirect / pipe the command output.
    """

    def __init__(self) -> None:
        from rich.console import Console
        from rich.progress import (
            BarColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
        )

        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]files"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TextColumn("{task.fields[current]}"),
            transient=False,
            console=Console(stderr=True),
        )
        self._task_id: Optional[int] = None
        self._started = False

    def __call__(self, event: ProgressEvent) -> None:
        if event.kind == "directory_start":
            if not self._started:
                self._progress.start()
                self._started = True
            self._task_id = self._progress.add_task(
                "files", total=max(event.total or 0, 1), current=""
            )
            if not event.total:
                self._progress.update(self._task_id, completed=1)
        elif event.kind == "file_start" and self._task_id is not None:
            name = Path(event.path).name if event.path else ""
            self._progress.update(self._task_id, current=name)
        elif event.kind == "file_end" and self._task_id is not None:
            # rich.progress is thread-safe: ``advance`` takes an internal
            # lock, so concurrent worker callbacks are safe to ``advance``
            # the shared task.
            self._progress.advance(self._task_id, 1)
        elif event.kind == "directory_end":
            self.close()

    def close(self) -> None:
        if self._started:
            self._progress.stop()
            self._started = False


def _make_progress_hook(
    args: argparse.Namespace, kind: str
) -> tuple[Optional[Callable[[ProgressEvent], None]], Optional[Any]]:
    """Return ``(callback, closer)`` for the active CLI command.

    Returns ``(None, None)`` when progress is disabled or ``rich`` isn't
    installed. ``closer`` is an object with ``.close()`` — the CLI calls
    it in a ``finally`` so an interrupted run still leaves a clean
    terminal (rich leaves cursor state dirty otherwise).
    """
    if not _progress_enabled(args):
        return None, None
    try:
        if kind == "directory":
            hook = _RichDirectoryProgress()
        else:
            hook = _RichFileProgress()
    except ImportError:
        # rich is an optional dep — progress silently disabled.
        return None, None
    return hook, hook


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


def _handle_usage_error(exc: ValueError) -> int:
    """Convert a processor-layer usage ``ValueError`` into a CLI usage
    error: stderr message + exit code 2.

    The refine / refine-first preconditions in ``MarkdownProcessor``
    surface as ``ValueError`` (aliased paths, inplace+refine,
    refine_first without refine_lang, colliding PO paths, etc.).
    Those are deterministic user-input errors, not runtime failures,
    so the CLI reports them exactly like the pre-call argparse-style
    usage messages rather than letting the traceback surface.
    """
    print(f"error: {exc}", file=sys.stderr)
    return 2


def cmd_translate(args: argparse.Namespace) -> int:
    hook, closer = _make_progress_hook(args, kind="file")
    processor = _build_processor(args, progress_callback=hook)
    json_receipt = getattr(args, "json_receipt", None)
    refine_first = getattr(args, "refine_first", False)
    refined_path = getattr(args, "refined_path", None)
    refine_lang = getattr(args, "refine_lang", None)
    refined_po_path = getattr(args, "refined_po_path", None)
    if refine_first and refined_path is None:
        print(
            "error: --refine-first requires --refined-path PATH",
            file=sys.stderr,
        )
        return 2
    if refine_first and not refine_lang:
        print(
            "error: --refine-first requires --refine-lang LOCALE "
            "(BCP 47 locale of the source document).",
            file=sys.stderr,
        )
        return 2
    try:
        try:
            result = processor.process_document(
                Path(args.source),
                Path(args.target_file),
                Path(args.po) if args.po else None,
                inplace=args.inplace,
                refined_path=Path(refined_path) if refined_path else None,
                refine_first=refine_first,
                refine_lang=refine_lang,
                refined_po_path=(
                    Path(refined_po_path) if refined_po_path else None
                ),
            )
        except ValueError as exc:
            if closer is not None:
                closer.close()
            return _handle_usage_error(exc)
        except BaseException as exc:
            # Stop the live progress renderer BEFORE emitting the
            # partial-receipt block. While rich is active it still owns
            # the cursor and would overwrite / interleave the receipt on
            # a TTY — exactly in the failure case it is meant to help
            # with. Closing twice is a no-op.
            if closer is not None:
                closer.close()
            # Even a mid-run failure may have billed tokens; surface what we
            # have so the operator / CI sees the real cost of the failed run.
            partial = getattr(exc, "partial_receipt", None)
            if partial is not None:
                if hasattr(partial, "render"):
                    print(partial.render(), file=sys.stderr)
                if json_receipt is not None:
                    _write_receipt_json(partial, json_receipt)
            raise
    finally:
        if closer is not None:
            closer.close()
    _print_result(result)
    _emit_receipt(result, json_receipt)
    stats = result.translation_stats
    return 1 if (stats.failed or stats.validation_failed) else 0


def cmd_translate_dir(args: argparse.Namespace) -> int:
    hook, closer = _make_progress_hook(args, kind="directory")
    processor = _build_processor(args, progress_callback=hook)
    json_receipt = getattr(args, "json_receipt", None)
    refine_first = getattr(args, "refine_first", False)
    refined_dir = getattr(args, "refined_dir", None)
    refine_lang = getattr(args, "refine_lang", None)
    refined_po_dir = getattr(args, "refined_po_dir", None)
    if refine_first and refined_dir is None:
        print(
            "error: --refine-first requires --refined-dir PATH",
            file=sys.stderr,
        )
        return 2
    if refine_first and not refine_lang:
        print(
            "error: --refine-first requires --refine-lang LOCALE "
            "(BCP 47 locale of the source document).",
            file=sys.stderr,
        )
        return 2
    try:
        try:
            result = processor.process_directory(
                Path(args.source_dir),
                Path(args.target_dir),
                Path(args.po_dir) if args.po_dir else None,
                glob=args.glob,
                inplace=args.inplace,
                max_workers=args.max_workers,
                refined_dir=Path(refined_dir) if refined_dir else None,
                refine_first=refine_first,
                refine_lang=refine_lang,
                refined_po_dir=(
                    Path(refined_po_dir) if refined_po_dir else None
                ),
            )
        except ValueError as exc:
            if closer is not None:
                closer.close()
            return _handle_usage_error(exc)
        except BaseException as exc:
            # Stop the bar before the partial-receipt print; see
            # ``cmd_translate`` for the rationale.
            if closer is not None:
                closer.close()
            partial = getattr(exc, "partial_receipt", None)
            if partial is not None:
                if hasattr(partial, "render"):
                    print(partial.render(), file=sys.stderr)
                if json_receipt is not None:
                    _write_receipt_json(partial, json_receipt)
            raise
    finally:
        if closer is not None:
            closer.close()
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


def cmd_refine(args: argparse.Namespace) -> int:
    if args.inplace:
        # Refine mode forbids source/msgid overwrite by contract.  The
        # processor raises ValueError on this combination; catch it at
        # the CLI boundary so users get a clean usage error instead of
        # a traceback.
        print(
            "error: --inplace is incompatible with `refine`. Refine "
            "never overwrites the source document — the refined output "
            "goes to REFINED_FILE. Drop --inplace.",
            file=sys.stderr,
        )
        return 2
    hook, closer = _make_progress_hook(args, kind="file")
    processor = _build_processor(args, progress_callback=hook, mode="refine")
    json_receipt = getattr(args, "json_receipt", None)
    try:
        try:
            result = processor.process_document(
                Path(args.source),
                Path(args.refined_file),
                Path(args.po) if args.po else None,
                inplace=False,
            )
        except ValueError as exc:
            if closer is not None:
                closer.close()
            return _handle_usage_error(exc)
        except BaseException as exc:
            if closer is not None:
                closer.close()
            partial = getattr(exc, "partial_receipt", None)
            if partial is not None:
                if hasattr(partial, "render"):
                    print(partial.render(), file=sys.stderr)
                if json_receipt is not None:
                    _write_receipt_json(partial, json_receipt)
            raise
    finally:
        if closer is not None:
            closer.close()
    _print_result(result)
    _emit_receipt(result, json_receipt)
    stats = result.translation_stats
    return 1 if (stats.failed or stats.validation_failed) else 0


def cmd_refine_dir(args: argparse.Namespace) -> int:
    if args.inplace:
        print(
            "error: --inplace is incompatible with `refine-dir`. Refine "
            "never overwrites the source — refined output goes under "
            "REFINED_DIR. Drop --inplace.",
            file=sys.stderr,
        )
        return 2
    hook, closer = _make_progress_hook(args, kind="directory")
    processor = _build_processor(args, progress_callback=hook, mode="refine")
    json_receipt = getattr(args, "json_receipt", None)
    try:
        try:
            result = processor.process_directory(
                Path(args.source_dir),
                Path(args.refined_dir),
                Path(args.po_dir) if args.po_dir else None,
                glob=args.glob,
                inplace=False,
                max_workers=args.max_workers,
            )
        except ValueError as exc:
            if closer is not None:
                closer.close()
            return _handle_usage_error(exc)
        except BaseException as exc:
            if closer is not None:
                closer.close()
            partial = getattr(exc, "partial_receipt", None)
            if partial is not None:
                if hasattr(partial, "render"):
                    print(partial.render(), file=sys.stderr)
                if json_receipt is not None:
                    _write_receipt_json(partial, json_receipt)
            raise
    finally:
        if closer is not None:
            closer.close()
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


def _parse_langs_csv(value: str) -> list[str]:
    """Split a comma-separated language list; drop blanks and duplicates
    while preserving caller order.

    Used by ``translate-multi`` so operators can write
    ``--langs ko,ja,zh-CN`` without caring about spaces around commas.
    """
    out: list[str] = []
    seen: set = set()
    for raw in value.split(","):
        lang = raw.strip()
        if not lang:
            continue
        if lang in seen:
            continue
        seen.add(lang)
        out.append(lang)
    return out


def cmd_translate_multi(args: argparse.Namespace) -> int:
    langs = _parse_langs_csv(args.langs)
    if not langs:
        print(
            "error: --langs must contain at least one locale.",
            file=sys.stderr,
        )
        return 2
    target_template: str = args.target_template
    if "{lang}" not in target_template:
        print(
            "error: --target-template must contain the literal substring "
            "'{lang}' — it is expanded per-locale (e.g. out/{lang}/source.md).",
            file=sys.stderr,
        )
        return 2
    po_template: Optional[str] = getattr(args, "po_template", None)
    if po_template is not None and "{lang}" not in po_template:
        print(
            "error: --po-template must contain the literal substring "
            "'{lang}' when supplied.",
            file=sys.stderr,
        )
        return 2

    # Path templates are user input — an unmatched brace or an extra
    # placeholder like ``{version}`` would otherwise bubble up from
    # str.format as a raw KeyError / ValueError traceback instead of a
    # clean usage error.  Only ``{lang}`` is supported and the CLI
    # already validated the literal substring above; anything else is
    # rejected here with a specific error message.
    try:
        target_paths: Dict[str, Path] = {
            lang: Path(target_template.format(lang=lang)) for lang in langs
        }
    except (KeyError, ValueError, IndexError) as exc:
        print(
            "error: --target-template contains unsupported format "
            f"expressions ({exc}). Only the literal placeholder "
            "'{lang}' is supported; any other brace expression must be "
            "removed or escaped.",
            file=sys.stderr,
        )
        return 2
    po_paths: Optional[Dict[str, Path]] = None
    if po_template:
        try:
            po_paths = {
                lang: Path(po_template.format(lang=lang)) for lang in langs
            }
        except (KeyError, ValueError, IndexError) as exc:
            print(
                "error: --po-template contains unsupported format "
                f"expressions ({exc}). Only the literal placeholder "
                "'{lang}' is supported; any other brace expression must be "
                "removed or escaped.",
                file=sys.stderr,
            )
            return 2

    # target_lang is ignored by process_document_multi but the
    # processor constructor still requires a locale; use the first lang
    # as a harmless placeholder so the validator helpers still resolve.
    setattr(args, "target", getattr(args, "target", None) or langs[0])

    hook, closer = _make_progress_hook(args, kind="file")
    processor = _build_processor(args, progress_callback=hook)
    json_receipt = getattr(args, "json_receipt", None)
    try:
        try:
            result = processor.process_document_multi(
                Path(args.source),
                target_langs=langs,
                target_paths=target_paths,
                po_paths=po_paths,
            )
        except ValueError as exc:
            if closer is not None:
                closer.close()
            return _handle_usage_error(exc)
        except BaseException as exc:
            if closer is not None:
                closer.close()
            partial = getattr(exc, "partial_receipt", None)
            if partial is not None:
                if hasattr(partial, "render"):
                    print(partial.render(), file=sys.stderr)
                if json_receipt is not None:
                    _write_receipt_json(partial, json_receipt)
            raise
    finally:
        if closer is not None:
            closer.close()

    summary: Dict[str, Any] = {
        "source_path": result["source_path"],
        "target_langs": result["target_langs"],
        "by_lang": {
            lang: {
                "target_path": pr["target_path"],
                "po_path": pr["po_path"],
                "blocks_count": pr["blocks_count"],
                "translation_stats": pr["translation_stats"].to_dict()
                if hasattr(pr["translation_stats"], "to_dict")
                else pr["translation_stats"],
                "coverage": pr["coverage"].to_dict()
                if hasattr(pr["coverage"], "to_dict")
                else pr["coverage"],
            }
            for lang, pr in result["by_lang"].items()
        },
    }
    _print_result(summary)
    receipt = result["receipt"]
    if hasattr(receipt, "render"):
        print(receipt.render(), file=sys.stderr)
    if json_receipt is not None:
        _write_receipt_json(receipt, json_receipt)

    any_failed = any(
        pr["translation_stats"]["failed"]
        or pr["translation_stats"]["validation_failed"]
        for pr in result["by_lang"].values()
    )
    return 1 if any_failed else 0


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
    p_tr.add_argument(
        "--refine-first",
        action="store_true",
        help=(
            "Run a refine pass on the source before translating. Requires "
            "--refined-path for the intermediate output."
        ),
    )
    p_tr.add_argument(
        "--refined-path",
        default=None,
        help="Intermediate refined markdown path used by --refine-first.",
    )
    p_tr.add_argument(
        "--refined-po-path",
        default=None,
        help=(
            "Refine-pass PO file path. Defaults to REFINED_PATH with a "
            ".po suffix; override when the default PO would collide with "
            "the translate-pass PO (e.g. TARGET and REFINED_PATH share a "
            "stem but differ by extension like out.md / out.markdown)."
        ),
    )
    p_tr.add_argument(
        "--refine-lang",
        default=None,
        help=(
            "BCP 47 locale the refine pass should preserve "
            "(the SOURCE language of the document, e.g. 'en'). "
            "REQUIRED with --refine-first; no default — using --target "
            "would pin refine to the translate target and corrupt the run."
        ),
    )
    p_tr.set_defaults(func=cmd_translate)

    p_dir = sub.add_parser("translate-dir", help="Translate a directory tree.")
    _add_translate_flags(p_dir)
    p_dir.add_argument("source_dir", help="Source directory.")
    p_dir.add_argument("target_dir", help="Target directory.")
    p_dir.add_argument("--po-dir", default=None, help="PO output directory.")
    p_dir.add_argument("--glob", default="**/*.md", help="Glob pattern for markdown files.")
    p_dir.add_argument("--max-workers", type=int, default=4, help="Concurrent file workers.")
    p_dir.add_argument(
        "--refine-first",
        action="store_true",
        help=(
            "Run a refine pass on each source file before translating. "
            "Requires --refined-dir."
        ),
    )
    p_dir.add_argument(
        "--refined-dir",
        default=None,
        help="Intermediate refined directory used by --refine-first.",
    )
    p_dir.add_argument(
        "--refined-po-dir",
        default=None,
        help=(
            "Refine-pass PO output directory. Defaults to refined-dir "
            "alongside each file's .po; override when the default POs "
            "would collide with the translate-pass POs (same stem, "
            "different extensions)."
        ),
    )
    p_dir.add_argument(
        "--refine-lang",
        default=None,
        help=(
            "BCP 47 locale the refine pass should preserve "
            "(the SOURCE language, e.g. 'en'). REQUIRED with "
            "--refine-first; no default."
        ),
    )
    p_dir.set_defaults(func=cmd_translate_dir)

    p_rf = sub.add_parser(
        "refine",
        help="Polish a markdown file in its source language (no translation).",
    )
    _add_translate_flags(
        p_rf,
        target_help=(
            "BCP 47 locale of the SOURCE document — the language the "
            "refine pass MUST preserve (e.g. 'en', 'ja'). refine never "
            "translates; any non-source locale here will produce "
            "wrong-language output and trip language_stability."
        ),
    )
    p_rf.add_argument("source", help="Source markdown file.")
    p_rf.add_argument(
        "refined_file", help="Refined markdown output (never overwrites source)."
    )
    p_rf.add_argument("--po", default=None, help="PO file path (default: refined with .po).")
    p_rf.set_defaults(func=cmd_refine)

    p_rfd = sub.add_parser(
        "refine-dir",
        help="Polish a directory tree in its source language (no translation).",
    )
    _add_translate_flags(
        p_rfd,
        target_help=(
            "BCP 47 locale of the SOURCE documents — the language the "
            "refine pass MUST preserve (e.g. 'en', 'ja'). refine never "
            "translates; any non-source locale here will produce "
            "wrong-language output and trip language_stability."
        ),
    )
    p_rfd.add_argument("source_dir", help="Source directory.")
    p_rfd.add_argument("refined_dir", help="Refined output directory.")
    p_rfd.add_argument("--po-dir", default=None, help="PO output directory.")
    p_rfd.add_argument("--glob", default="**/*.md", help="Glob pattern for markdown files.")
    p_rfd.add_argument(
        "--max-workers", type=int, default=4, help="Concurrent file workers."
    )
    p_rfd.set_defaults(func=cmd_refine_dir)

    p_multi = sub.add_parser(
        "translate-multi",
        help=(
            "Translate a single markdown file into MULTIPLE target "
            "languages in one batched call."
        ),
    )
    _add_shared_flags(p_multi)
    p_multi.add_argument(
        "--langs",
        required=True,
        help=(
            "Comma-separated BCP 47 locales (e.g. 'ko,ja,zh-CN'). "
            "Duplicates dropped, order preserved."
        ),
    )
    p_multi.add_argument(
        "--target-template",
        required=True,
        help=(
            "Per-locale output markdown path template containing the "
            "literal substring '{lang}', e.g. 'out/{lang}/README.md'."
        ),
    )
    p_multi.add_argument(
        "--po-template",
        default=None,
        help=(
            "Optional per-locale PO path template containing '{lang}'. "
            "Defaults to the target template with a '.po' suffix."
        ),
    )
    p_multi.add_argument(
        "--validation",
        choices=["off", "conservative", "strict"],
        default="off",
        help="Post-translation structural validation.",
    )
    p_multi.add_argument(
        "--glossary",
        type=Path,
        default=None,
        help=(
            "Path to a JSON glossary file. Per-locale entries in the "
            "file are resolved for the FIRST lang in --langs only; "
            "callers with stronger per-locale needs should run "
            "single-target 'translate' per lang instead."
        ),
    )
    p_multi.add_argument(
        "--glossary-mode",
        choices=["instruction", "placeholder"],
        default="instruction",
        help="How to feed glossary terms to the LLM (see 'translate').",
    )
    p_multi.add_argument(
        "--extra-instructions",
        type=str,
        default=None,
        help="Additional prompt instructions (tone, domain, audience).",
    )
    p_multi.add_argument(
        "--prompt-cache",
        action="store_true",
        help="Mark the stable system prefix as cacheable.",
    )
    p_multi.add_argument(
        "--json-receipt",
        type=Path,
        default=None,
        metavar="PATH",
        help="Also write the run receipt (tokens, cost, duration) as JSON to PATH.",
    )
    p_multi.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the progress bar even on a TTY.",
    )
    p_multi.add_argument("source", help="Source markdown file.")
    p_multi.set_defaults(func=cmd_translate_multi)

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
