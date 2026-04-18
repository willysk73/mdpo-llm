"""
JSON-batch translator.

Issues one LLM call for many blocks at once using a JSON wire format.
Partitions by entry count and total characters, validates all keys round-trip,
and recursively bisects on any failure. A single-entry batch that still fails
returns an empty dict so the caller can fall back to the per-entry path.
"""

import json
import logging
from typing import Any, Callable, Dict, Iterator, Optional

logger = logging.getLogger(__name__)


class BatchTranslator:
    """Translate multiple blocks in a single LLM call via JSON wire format."""

    def __init__(
        self,
        call_llm: Callable[[Dict[str, str]], str],
        max_entries: int = 40,
        max_chars: int = 8000,
    ):
        """
        Args:
            call_llm: Callable that takes the batch's ``{key: source}`` dict
                and returns the raw LLM response string (expected to contain
                a JSON object).  The caller owns message construction
                (system prompt, references, glossary, JSON response mode).
            max_entries: Max items per JSON request.
            max_chars: Soft cap on total source characters per request.
        """
        self._call_llm = call_llm
        self.max_entries = max_entries
        self.max_chars = max_chars

    def translate(self, items: Dict[str, str]) -> Dict[str, str]:
        """Translate ``items`` by batching into LLM calls.

        Returns a ``{key: translation}`` mapping.  Keys absent from the return
        signal the caller should fall back to per-entry translation.
        """
        if not items:
            return {}

        result: Dict[str, str] = {}
        for chunk in self._partition(items):
            result.update(self._translate_chunk(chunk))
        return result

    def _partition(self, items: Dict[str, str]) -> Iterator[Dict[str, str]]:
        """Yield sub-dicts each within ``max_entries`` and ``max_chars``."""
        current: Dict[str, str] = {}
        current_chars = 0
        for k, v in items.items():
            src_len = len(v)
            if current and (
                len(current) >= self.max_entries
                or current_chars + src_len > self.max_chars
            ):
                yield current
                current = {}
                current_chars = 0
            current[k] = v
            current_chars += src_len
        if current:
            yield current

    def _translate_chunk(self, chunk: Dict[str, str]) -> Dict[str, str]:
        """Translate one chunk; bisect recursively on any failure."""
        try:
            raw = self._call_llm(chunk)
        except Exception as exc:
            logger.warning(
                "Batch call raised (%d items): %s; bisecting", len(chunk), exc
            )
            return self._bisect(chunk)

        parsed = self._parse_response(raw)
        if parsed is None:
            logger.warning(
                "Batch response unparseable (%d items); bisecting", len(chunk)
            )
            return self._bisect(chunk)

        good: Dict[str, str] = {}
        for k in chunk:
            v = parsed.get(k)
            if isinstance(v, str):
                good[k] = v

        missing = {k: chunk[k] for k in chunk if k not in good}
        if not missing:
            return good

        logger.info(
            "Batch missing %d/%d keys; retrying subset", len(missing), len(chunk)
        )
        good.update(self._bisect(missing))
        return good

    def _bisect(self, chunk: Dict[str, str]) -> Dict[str, str]:
        """Halve chunk and translate each half.  Single-entry failure → {}."""
        if len(chunk) <= 1:
            return {}
        keys = list(chunk.keys())
        mid = len(keys) // 2
        left = {k: chunk[k] for k in keys[:mid]}
        right = {k: chunk[k] for k in keys[mid:]}
        out: Dict[str, str] = {}
        out.update(self._translate_chunk(left))
        out.update(self._translate_chunk(right))
        return out

    @staticmethod
    def _parse_response(raw: Any) -> Optional[Dict[str, Any]]:
        """Extract a JSON object from raw model output.

        Handles plain JSON, code-fenced (```json …```), and prose-wrapped
        responses where a JSON object is embedded.
        """
        if not isinstance(raw, str):
            return None
        s = raw.strip()
        if not s:
            return None

        # Direct parse.
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

        # Strip code fence (```json … ``` or ``` … ```).
        if s.startswith("```"):
            lines = s.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            inner = "\n".join(lines).strip()
            try:
                obj = json.loads(inner)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass

        # Last resort: slice the outermost {...} substring.
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end > start:
            try:
                obj = json.loads(s[start : end + 1])
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass

        return None
