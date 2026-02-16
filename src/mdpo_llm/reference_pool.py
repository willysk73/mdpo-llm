"""
Reference pool for translation context.

Maintains a growing pool of (source, translation) pairs and finds
similar entries via difflib.SequenceMatcher to provide few-shot
context to the LLM.
"""

from difflib import SequenceMatcher
from typing import List, Tuple

import polib


class ReferencePool:
    """Pool of (source, translation) pairs used as few-shot context."""

    def __init__(self, max_results: int = 5):
        """
        Initialize the reference pool.

        Args:
            max_results: Maximum number of similar pairs to return from find_similar.
        """
        self.max_results = max_results
        self._pairs: List[Tuple[str, str]] = []

    def seed_from_po(self, po_file: polib.POFile) -> None:
        """Populate pool from already-translated PO entries.

        Picks up non-obsolete, non-fuzzy entries with non-empty msgstr.

        Args:
            po_file: A loaded polib.POFile instance.
        """
        for entry in po_file:
            if entry.obsolete:
                continue
            if "fuzzy" in entry.flags:
                continue
            if not entry.msgstr:
                continue
            self._pairs.append((entry.msgid, entry.msgstr))

    def add(self, source: str, translation: str) -> None:
        """Append a new (source, translation) pair to the pool.

        Args:
            source: Original source text.
            translation: Translated text.
        """
        self._pairs.append((source, translation))

    def find_similar(self, source_text: str) -> List[Tuple[str, str]]:
        """Return the top-K most similar (source, translation) pairs.

        Uses difflib.SequenceMatcher ratio for similarity scoring.
        Excludes exact self-matches (where source == source_text).
        Results are sorted by similarity, most similar first.

        Args:
            source_text: The source text to find similar pairs for.

        Returns:
            List of (source, translation) tuples, most similar first.
        """
        if not self._pairs:
            return []

        scored = []
        for src, tgt in self._pairs:
            if src == source_text:
                continue
            ratio = SequenceMatcher(None, source_text, src).ratio()
            scored.append((ratio, src, tgt))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [(src, tgt) for _, src, tgt in scored[: self.max_results]]

    def __len__(self) -> int:
        return len(self._pairs)
