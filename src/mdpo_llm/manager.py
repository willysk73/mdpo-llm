"""
PO file management for processing workflow.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import polib


class POManager:
    """Manages GNU gettext PO files for process tracking."""

    def __init__(self, po_path: Optional[Path] = None, skip_types: List[str] = []):
        """Initialize PO manager with optional path."""
        self.po_path = po_path
        self.po_file = None
        self.skip_types = skip_types

    def _remove_obsolete_entries(self, po_file: Optional[polib.POFile] = None) -> None:
        """Remove obsolete entries from PO file."""
        po = po_file or self.po_file
        if not po:
            raise ValueError("No PO file loaded")

        for entry in po.obsolete_entries():
            po.remove(entry)

    def load_or_create_po(
        self, po_path: Optional[Path] = None, target_lang: Optional[str] = None
    ) -> polib.POFile:
        """Load existing PO file or create new one.

        Args:
            po_path: Path to the PO file.
            target_lang: BCP 47 locale string.  Sets the ``Language``
                metadata header on *new* files; existing files are not
                overwritten.
        """
        path = po_path or self.po_path
        if not path:
            raise ValueError("No PO file path provided")

        self.po_path = path

        if path.exists():
            self.po_file = polib.pofile(path.read_text(encoding="utf-8"))
        else:
            self.po_file = polib.POFile()
            metadata = {"Content-Type": "text/plain; charset=UTF-8"}
            if target_lang:
                metadata["Language"] = target_lang
            self.po_file.metadata = metadata

        return self.po_file

    def sync_po(
        self, po_file: polib.POFile, blocks: List[Dict[str, Any]], context_id_func
    ) -> None:
        """Sync parsed blocks with PO file entries."""
        seen = set()

        for block in blocks:
            ctx = context_id_func(block)
            seen.add(ctx)
            text = block["text"]

            if block["type"] in self.skip_types:
                # Still track in PO for change detection, but don't process
                entry = next((e for e in po_file if e.msgctxt == ctx), None)
                if entry is None:
                    # Create entry but mark as non-processable (empty msgstr, no fuzzy flag)
                    po_file.append(polib.POEntry(msgctxt=ctx, msgid=text, msgstr=""))
                else:
                    # Update source text but don't mark as fuzzy (no process needed)
                    entry.msgid = text
                continue

            # Process processable blocks (headings, paragraphs, lists, quotes, tables)
            entry = next((e for e in po_file if e.msgctxt == ctx), None)
            if entry is None:
                po_file.append(polib.POEntry(msgctxt=ctx, msgid=text, msgstr=""))
            else:
                if entry.msgid != text:
                    entry.msgid = text
                    if "fuzzy" not in entry.flags:
                        entry.flags.append("fuzzy")

        # Mark missing entries as obsolete
        for entry in po_file:
            if entry.msgctxt not in seen:
                entry.obsolete = True

        self._remove_obsolete_entries(po_file)

    def redraw_context(self, blocks: List[Dict[str, Any]], context_id_func) -> None:
        old_metadata = getattr(self.po_file, "metadata", {}) if self.po_file is not None else {}
        self.po_file = polib.POFile()
        metadata = {"Content-Type": "text/plain; charset=UTF-8"}
        if old_metadata.get("Language"):
            metadata["Language"] = old_metadata["Language"]
        self.po_file.metadata = metadata
        po_file = self.po_file

        for block in blocks:
            ctx = context_id_func(block)
            text = block["text"]

            if block["type"] in self.skip_types:
                # Still track in PO for change detection, but don't process
                entry = next((e for e in po_file if e.msgctxt == ctx), None)
                if entry is None:
                    # Create entry but mark as non-processable (empty msgstr, no fuzzy flag)
                    po_file.append(polib.POEntry(msgctxt=ctx, msgid=text, msgstr=""))
                else:
                    # Update source text but don't mark as fuzzy (no process needed)
                    entry.msgid = text
                continue

            # Process processable blocks (headings, paragraphs, lists, quotes, tables)
            entry = next((e for e in po_file if e.msgctxt == ctx), None)
            if entry is None:
                po_file.append(polib.POEntry(msgctxt=ctx, msgid=text, msgstr=text))

    def save_po(
        self, po_file: Optional[polib.POFile] = None, path: Optional[Path] = None
    ) -> None:
        """Save PO file to disk."""
        po = po_file or self.po_file
        save_path = path or self.po_path

        if not po:
            raise ValueError("No PO file to save")
        if not save_path:
            raise ValueError("No path provided for saving PO file")

        save_path.parent.mkdir(parents=True, exist_ok=True)
        po.save(save_path.as_posix(), newline="\n")

    def get_unprocessed_entries(
        self, po_file: Optional[polib.POFile] = None
    ) -> List[polib.POEntry]:
        """Get entries that need processing."""
        po = po_file or self.po_file
        if not po:
            raise ValueError("No PO file loaded")

        return [
            e for e in po if not e.obsolete and (not e.msgstr or "fuzzy" in e.flags)
        ]

    def get_fuzzy_entries(
        self, po_file: Optional[polib.POFile] = None
    ) -> List[polib.POEntry]:
        """Get entries marked as fuzzy (need reprocessing)."""
        po = po_file or self.po_file
        if not po:
            raise ValueError("No PO file loaded")

        return [e for e in po if not e.obsolete and "fuzzy" in e.flags]

    def mark_entry_processed(self, entry: polib.POEntry) -> None:
        """Mark entry as processed by removing fuzzy flag."""
        if "fuzzy" in entry.flags:
            entry.flags.remove("fuzzy")

    def get_processing_stats(
        self, po_file: Optional[polib.POFile] = None
    ) -> Dict[str, int]:
        """Get process statistics."""
        po = po_file or self.po_file
        if not po:
            raise ValueError("No PO file loaded")

        stats = {
            "total": 0,
            "processed": 0,
            "fuzzy": 0,
            "unprocessed": 0,
            "obsolete": 0,
        }

        for entry in po:
            if entry.obsolete:
                stats["obsolete"] += 1
                continue

            stats["total"] += 1

            if "fuzzy" in entry.flags:
                stats["fuzzy"] += 1
            elif entry.msgstr:
                stats["processed"] += 1
            else:
                stats["unprocessed"] += 1

        return stats
