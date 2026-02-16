"""
Main Markdown Translator orchestrator class.
"""

import inspect
from pathlib import Path
from typing import Any, Dict, List
import logging

logger = logging.getLogger(__name__)

import polib

from .language import LanguageCode, contains_language

from .llm_interface import LLMInterface
from .manager import POManager
from .parser import BlockParser
from .reconstructor import DocumentReconstructor
from .reference_pool import ReferencePool


class MarkdownProcessor:
    """Main orchestrator for markdown process workflow."""

    SKIP_TYPES = ["hr"]  # Block types to skip processing

    def __init__(self, llm_interface: LLMInterface, max_reference_pairs: int = 5):
        """
        Initialize the processor.

        Args:
            llm_interface: Custom LLM interface, if provided
            max_reference_pairs: Maximum number of similar reference pairs
                to pass as context to the LLM per entry.
        """
        self.parser = BlockParser()
        self.po_manager = POManager(skip_types=self.SKIP_TYPES)
        self.reconstructor = DocumentReconstructor(skip_types=self.SKIP_TYPES)
        self.llm = llm_interface
        self.max_reference_pairs = max_reference_pairs

        # Detect whether the LLM accepts reference_pairs
        self._llm_accepts_reference_pairs = self._check_llm_accepts_reference_pairs()

    def _check_llm_accepts_reference_pairs(self) -> bool:
        """Check if the LLM's process() method accepts a reference_pairs parameter."""
        try:
            sig = inspect.signature(self.llm.process)
            return "reference_pairs" in sig.parameters
        except (ValueError, TypeError):
            return False

    def process_document(
        self, source_path: Path, target_path: Path, po_path: Path, inplace: bool = False
    ) -> Dict[str, Any]:
        """
        Process a markdown document.

        Uses local instances of parser, po_manager, and reconstructor for
        thread safety (so process_directory can call this concurrently).

        Args:
            source_path: Path to source markdown file
            target_path: Path for processed markdown file
            po_path: Path for PO file

        Returns:
            Dictionary with translation results and statistics
        """
        # Local instances for thread safety
        parser = BlockParser()
        po_manager = POManager(skip_types=self.SKIP_TYPES)
        reconstructor = DocumentReconstructor(skip_types=self.SKIP_TYPES)

        po_file = None
        try:
            # Step 1: Read and parse source document
            source = source_path.read_text(encoding="utf-8")
            source_lines = source.splitlines(keepends=True)
            blocks = parser.segment_markdown(
                [line.rstrip("\n") for line in source_lines]
            )

            # Step 2: Sync with PO file
            po_file = po_manager.load_or_create_po(po_path)
            po_manager.sync_po(po_file, blocks, parser.context_id)

            # Step 3: process sequentially with reference context
            translation_stats = self._process_entries_sequential(
                po_file, po_manager, inplace=inplace
            )

            # Step 4: Get coverage stats
            coverage = reconstructor.get_process_coverage(
                blocks, po_file, parser.context_id
            )

            # Step 5: Rebuild processed document
            processed_content = reconstructor.rebuild_markdown(
                source_lines, blocks, po_file, parser.context_id
            )

            if inplace:
                self._match_ctxt(
                    processed_content=processed_content,
                    parser=parser,
                    po_manager=po_manager,
                )

            # Step 6: Save processed document
            self._save_processed_document(processed_content, target_path)

            return {
                "source_path": str(source_path),
                "target_path": str(target_path),
                "po_path": str(po_path),
                "blocks_count": len(blocks),
                "coverage": coverage,
                "translation_stats": translation_stats,
            }
        finally:
            if po_file is not None:
                po_manager.save_po(po_file, po_path)

    def process_directory(
        self,
        source_dir: Path,
        target_dir: Path,
        po_dir: Path,
        glob: str = "**/*.md",
        inplace: bool = False,
        max_workers: int = 4,
    ) -> Dict[str, Any]:
        """
        Process all markdown files in a directory tree.

        Files are processed concurrently (each file has its own sequential
        entry processing with its own reference pool).

        Args:
            source_dir: Root directory containing source markdown files
            target_dir: Root directory for processed output (mirrors source structure)
            po_dir: Root directory for PO files (mirrors source structure)
            glob: Glob pattern to match markdown files
            inplace: Whether to use inplace mode for processing
            max_workers: Maximum number of files to process concurrently

        Returns:
            Dictionary with aggregate results and per-file details
        """
        import concurrent.futures

        source_dir = Path(source_dir)
        target_dir = Path(target_dir)
        po_dir = Path(po_dir)

        matched_files = sorted(source_dir.glob(glob))

        results: List[Dict[str, Any]] = []
        files_processed = 0
        files_failed = 0
        files_skipped = 0

        def _process_one(source_file: Path):
            relative_path = source_file.relative_to(source_dir)
            target_path = target_dir / relative_path
            po_path_file = po_dir / relative_path.with_suffix(".po")
            return self.process_document(
                source_file, target_path, po_path_file, inplace=inplace
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(_process_one, sf): sf for sf in matched_files
            }
            for future in concurrent.futures.as_completed(future_to_file):
                source_file = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)

                    newly_processed = result.get("translation_stats", {}).get("processed", 0)
                    if newly_processed == 0:
                        files_skipped += 1
                    else:
                        files_processed += 1
                except Exception:
                    logger.exception("Failed to process %s", source_file)
                    files_failed += 1
                    results.append({"source_path": str(source_file), "error": True})

        return {
            "source_dir": str(source_dir),
            "target_dir": str(target_dir),
            "po_dir": str(po_dir),
            "files_processed": files_processed,
            "files_failed": files_failed,
            "files_skipped": files_skipped,
            "results": results,
        }

    def _match_ctxt(self, processed_content: str, parser=None, po_manager=None):
        p = parser or self.parser
        pm = po_manager or self.po_manager
        processed_lines = processed_content.splitlines(keepends=True)
        blocks = p.segment_markdown(
            [line.rstrip("\n") for line in processed_lines]
        )
        pm.redraw_context(blocks, p.context_id)

    def _process_entry(self, entry, reference_pairs=None):
        try:
            if reference_pairs is not None and self._llm_accepts_reference_pairs:
                processed = self.llm.process(entry.msgid, reference_pairs=reference_pairs)
            else:
                processed = self.llm.process(entry.msgid)
            return (entry, processed, None)
        except Exception as e:
            return (entry, None, e)

    def _process_entries_sequential(
        self, po_file: polib.POFile, po_manager: POManager, inplace: bool = False
    ) -> Dict[str, int]:
        """Process entries sequentially with growing reference context.

        Builds a ReferencePool seeded from existing translations, then
        iterates entries in PO order (= document order after sync_po).
        Each successfully translated entry is added to the pool so that
        subsequent entries benefit from the context.

        Args:
            po_file: The PO file with synced entries.
            po_manager: The POManager instance to use.
            inplace: Whether to update msgid to match msgstr.

        Returns:
            Dictionary with processed/failed/skipped counts.
        """
        pool = ReferencePool(max_results=self.max_reference_pairs)
        pool.seed_from_po(po_file)

        stats = {"processed": 0, "failed": 0, "skipped": 0}

        for entry in po_file:
            if entry.obsolete:
                continue

            block_type = self._extract_block_type_from_msgctxt(entry.msgctxt)
            if block_type in self.SKIP_TYPES:
                stats["skipped"] += 1
                continue

            if block_type == "code":
                if not contains_language(entry.msgid, [LanguageCode.KO]):
                    entry.msgstr = entry.msgid
                    po_manager.mark_entry_processed(entry)
                    stats["skipped"] += 1
                    continue

            needs_translation = (not entry.msgstr) or ("fuzzy" in entry.flags)
            if not needs_translation:
                continue

            similar = pool.find_similar(entry.msgid)

            try:
                entry_obj, processed, error = self._process_entry(
                    entry, reference_pairs=similar or None
                )
                if error is None:
                    entry_obj.msgstr = processed
                    if inplace:
                        entry_obj.msgid = processed
                    po_manager.mark_entry_processed(entry_obj)
                    stats["processed"] += 1
                    pool.add(entry.msgid, processed)
                else:
                    print(f"Failed to process entry {entry_obj.msgctxt}: {str(error)}")
                    stats["failed"] += 1
            except Exception as exc:
                print(
                    f"Unexpected error for entry {getattr(entry, 'msgctxt', None)}: {exc}"
                )
                stats["failed"] += 1

        return stats

    def _extract_block_type_from_msgctxt(self, msgctxt: str) -> str:
        """Extract block type from msgctxt, between '::' and ':'."""
        if msgctxt:
            start = msgctxt.find("::")
            if start != -1:
                start += 2
                end = msgctxt.find(":", start)
                if end != -1:
                    return msgctxt[start:end]
        return ""

    def _process_entries(
        self, po_file: polib.POFile, inplace: bool = False
    ) -> Dict[str, int]:
        """Process unprocessed entries in PO file."""
        stats = {"processed": 0, "failed": 0, "skipped": 0}

        for entry in po_file:
            if entry.obsolete:
                continue

            block_type = self._extract_block_type_from_msgctxt(entry.msgctxt)
            if block_type in self.SKIP_TYPES:
                stats["skipped"] += 1
                continue

            needs_translation = (not entry.msgstr) or ("fuzzy" in entry.flags)
            if not needs_translation:
                continue

            try:
                entry_obj, processed, error = self._process_entry(entry)
                if error is None:
                    entry_obj.msgstr = processed
                    if inplace:
                        entry_obj.msgid = processed
                    self.po_manager.mark_entry_processed(entry_obj)
                    stats["processed"] += 1
                else:
                    print(f"Failed to process entry {entry_obj.msgctxt}: {str(error)}")
                    stats["failed"] += 1
            except Exception as exc:
                print(
                    f"Unexpected error for entry {getattr(entry, 'msgctxt', None)}: {exc}"
                )
                stats["failed"] += 1

        return stats

    def _process_entries_concurrent(
        self, po_file: "polib.POFile", inplace: bool = False, max_workers: int = 10
    ) -> Dict[str, int]:
        """Process unprocessed entries in PO file concurrently."""
        import concurrent.futures
        import threading

        stats = {"processed": 0, "failed": 0, "skipped": 0}
        entries_to_process = []

        for entry in po_file:
            if entry.obsolete:
                continue

            block_type = self._extract_block_type_from_msgctxt(entry.msgctxt)
            if block_type in self.SKIP_TYPES:
                stats["skipped"] += 1
                continue

            if block_type == "code":
                if not contains_language(entry.msgid, [LanguageCode.KO]):
                    entry.msgstr = entry.msgid
                    self.po_manager.mark_entry_processed(entry)
                    stats["skipped"] += 1
                    continue

            needs_translation = (not entry.msgstr) or ("fuzzy" in entry.flags)
            if not needs_translation:
                continue

            entries_to_process.append(entry)

        stop_event = threading.Event()
        future_to_entry = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            try:
                future_to_entry = {
                    executor.submit(self._process_entry, entry): entry
                    for entry in entries_to_process
                }
                for future in concurrent.futures.as_completed(future_to_entry):
                    entry = future_to_entry[future]
                    try:
                        entry_obj, processed, error = future.result()
                        if error is None:
                            entry_obj.msgstr = processed
                            if inplace:
                                entry_obj.msgid = processed
                            self.po_manager.mark_entry_processed(entry_obj)
                            stats["processed"] += 1
                        else:
                            print(
                                f"Failed to process entry {entry_obj.msgctxt}: {str(error)}"
                            )
                            stats["failed"] += 1
                    except concurrent.futures.CancelledError:
                        stats["failed"] += 1
                    except Exception as exc:
                        print(
                            f"Unexpected error for entry {getattr(entry, 'msgctxt', None)}: {exc}"
                        )
                        stats["failed"] += 1

            except KeyboardInterrupt:
                stop_event.set()

                for f in list(future_to_entry.keys()):
                    f.cancel()

                executor.shutdown(cancel_futures=True)
                logging.info(
                    "Ctrl+C pressed — cancelling pending tasks and shutting down…"
                )

        return stats

    def _extract_block_type(self, context_id: str) -> str:
        """Extract block type from context ID."""
        if "::" in context_id:
            parts = context_id.split("::")
            if len(parts) >= 2:
                return parts[1]
        return "unknown"

    def _save_processed_document(self, processed_content: str, target_path: Path):
        """Save the processed document to the target path."""
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(processed_content, encoding="utf-8", newline="\n")

    def get_translation_stats(self, source_path: Path, po_path: Path) -> Dict[str, Any]:
        """Get detailed translation statistics."""
        source_lines = source_path.read_text(encoding="utf-8").splitlines(keepends=True)
        blocks = self.parser.segment_markdown(
            [line.rstrip("\n") for line in source_lines]
        )
        po_file = self.po_manager.load_or_create_po(po_path)

        coverage = self.reconstructor.get_process_coverage(
            blocks, po_file, self.parser.context_id
        )
        po_stats = self.po_manager.get_processing_stats(po_file)

        return {
            "file_stats": {
                "source_path": str(source_path),
                "po_path": str(po_path),
                "total_lines": len(source_lines),
                "total_blocks": len(blocks),
            },
            "coverage": coverage,
            "po_stats": po_stats,
        }

    def export_report(self, source_path: Path, po_path: Path) -> str:
        """Export detailed translation report."""
        source_lines = source_path.read_text(encoding="utf-8").splitlines(keepends=True)
        blocks = self.parser.segment_markdown(
            [line.rstrip("\n") for line in source_lines]
        )
        po_file = self.po_manager.load_or_create_po(po_path)

        return self.reconstructor.export_translation_report(
            str(source_path), blocks, po_file, self.parser.context_id
        )
