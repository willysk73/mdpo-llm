"""
Main Markdown Translator orchestrator class.
"""

from pathlib import Path
from typing import Any, Dict
import logging

import polib

from .language import LanguageCode, contains_language

from .llm_interface import LLMInterface
from .manager import POManager
from .parser import BlockParser
from .reconstructor import DocumentReconstructor


class MarkdownProcessor:
    """Main orchestrator for markdown process workflow."""

    SKIP_TYPES = ["hr"]  # Block types to skip processing

    def __init__(self, llm_interface: LLMInterface):
        """
        Initialize the processor.

        Args:
            llm_interface: Custom LLM interface, if provided
            llm_type: Type of LLM to use if no interface provided
            **llm_kwargs: Additional arguments for LLM initialization
        """
        self.parser = BlockParser()
        self.po_manager = POManager(skip_types=self.SKIP_TYPES)
        self.reconstructor = DocumentReconstructor(skip_types=self.SKIP_TYPES)
        self.llm = llm_interface

    def process_document(
        self, source_path: Path, target_path: Path, po_path: Path, inplace: bool = False
    ) -> Dict[str, Any]:
        """
        Process a markdown document.

        Args:
            source_path: Path to source markdown file
            target_path: Path for  markdown file
            po_path: Path for PO file

        Returns:
            Dictionary with translation results and statistics
        """
        po_file = None
        try:
            # Step 1: Read and parse source document
            source = source_path.read_text(encoding="utf-8")
            source_lines = source.splitlines(keepends=True)
            blocks = self.parser.segment_markdown(
                [line.rstrip("\n") for line in source_lines]
            )

            # Step 2: Sync with PO file
            po_file = self.po_manager.load_or_create_po(po_path)
            self.po_manager.sync_po(po_file, blocks, self.parser.context_id)

            # Step 3: process
            translation_stats = self._process_entries_concurrent(
                po_file, inplace=inplace, max_workers=10
            )

            # Step 4: Get coverage stats
            coverage = self.reconstructor.get_process_coverage(
                blocks, po_file, self.parser.context_id
            )

            # Step 5: Rebuild processed document
            processed_content = self.reconstructor.rebuild_markdown(
                source_lines, blocks, po_file, self.parser.context_id
            )

            if inplace:
                self._match_ctxt(processed_content=processed_content)

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
                self.po_manager.save_po(po_file, po_path)

    def _match_ctxt(self, processed_content: str):
        processed_lines = processed_content.splitlines(keepends=True)
        blocks = self.parser.segment_markdown(
            [line.rstrip("\n") for line in processed_lines]
        )
        self.po_manager.redraw_context(blocks, self.parser.context_id)

    def _process_entry(self, entry):
        try:
            processed = self.llm.process(entry.msgid)
            return (entry, processed, None)
        except Exception as e:
            return (entry, None, e)

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
