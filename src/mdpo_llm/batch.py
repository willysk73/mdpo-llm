"""
JSON-batch translator.

Issues one LLM call for many blocks at once using a JSON wire format.
Partitions by entry count and total characters, validates all keys round-trip,
and recursively bisects on any failure. A single-entry batch that still fails
returns an empty dict so the caller can fall back to the per-entry path.
"""

import json
import logging
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional

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


class MultiTargetBatchTranslator(BatchTranslator):
    """Translate multiple blocks into N target languages in one LLM call.

    The wire format is a JSON object where each key is an opaque block id
    and each value is another JSON object mapping BCP 47 locale → translated
    Markdown.  Source-side decomposition (placeholders, glossary hits,
    reference lookup) runs once per block regardless of how many languages
    fan out, so re-using this class across languages amortises a single
    input-token bill across N outputs.

    Partial per-lang coverage for a given block IS kept: each block
    returns whichever languages came back with a well-typed string
    value, and the caller backfills any missing langs with single-target
    per-entry calls (see :meth:`MarkdownProcessor._translate_group_multi`).
    A block whose value is missing entirely — not a dict at all, or a
    dict with no valid per-lang strings — triggers the same bisection
    strategy as the single-target :class:`BatchTranslator`.  A
    single-entry chunk that still fails returns an empty dict so the
    caller can fall back to N independent single-target calls.
    """

    def __init__(
        self,
        call_llm: Callable[[Dict[str, str]], str],
        target_langs: Iterable[str],
        max_entries: int = 40,
        max_chars: int = 8000,
    ):
        """
        Args:
            call_llm: Callable that takes the batch's ``{key: source}``
                dict and returns the raw LLM response string (a JSON
                object whose values are per-language dicts).  The caller
                owns message construction.
            target_langs: Non-empty iterable of BCP 47 locale strings.
                Duplicates are ignored but order is preserved — callers
                that care about stable ordering (e.g. for snapshot tests)
                get deterministic behaviour.
            max_entries: Max source blocks per JSON request.
            max_chars: Soft cap on total SOURCE characters per request —
                the cap is not scaled by ``len(target_langs)`` because
                output-token growth from fan-out is bounded by the model
                (not by the request-side character budget), and inflating
                the cap would just push input-token costs higher without
                a matching safety benefit.
        """
        super().__init__(call_llm, max_entries=max_entries, max_chars=max_chars)
        seen: set = set()
        ordered: List[str] = []
        for lang in target_langs:
            if not isinstance(lang, str) or not lang:
                raise ValueError(
                    "target_langs entries must be non-empty strings"
                )
            if lang in seen:
                continue
            seen.add(lang)
            ordered.append(lang)
        if not ordered:
            raise ValueError("target_langs must not be empty")
        self.target_langs: tuple = tuple(ordered)

    def translate(self, items: Dict[str, str]) -> Dict[str, Dict[str, str]]:
        """Translate ``items`` into every configured target language.

        Returns a ``{key: {lang: translation}}`` mapping.  A key absent
        from the return signals the caller should fall back to per-entry
        translation for that block across all languages.  A key present
        with PARTIAL language coverage (some langs delivered, others
        missing) is emitted as-is: the per-lang keys that came back are
        preserved, and the caller is responsible for falling back on
        the missing langs.  This preserves whatever useful output the
        model actually produced rather than throwing away partial work.
        """
        if not items:
            return {}

        result: Dict[str, Dict[str, str]] = {}
        for chunk in self._partition(items):
            result.update(self._translate_chunk_multi(chunk))
        return result

    def _translate_chunk_multi(
        self, chunk: Dict[str, str]
    ) -> Dict[str, Dict[str, str]]:
        """Translate one chunk; bisect recursively on any failure."""
        try:
            raw = self._call_llm(chunk)
        except Exception as exc:
            logger.warning(
                "Multi-target batch call raised (%d items, %d langs): %s; bisecting",
                len(chunk),
                len(self.target_langs),
                exc,
            )
            return self._bisect_multi(chunk)

        parsed = self._parse_response(raw)
        if parsed is None:
            logger.warning(
                "Multi-target batch response unparseable (%d items); bisecting",
                len(chunk),
            )
            return self._bisect_multi(chunk)

        good: Dict[str, Dict[str, str]] = {}
        for k in chunk:
            v = parsed.get(k)
            if not isinstance(v, dict):
                continue
            langs_ok: Dict[str, str] = {}
            for lang in self.target_langs:
                val = v.get(lang)
                if isinstance(val, str):
                    langs_ok[lang] = val
            # Partial coverage IS returned: any lang that came back with
            # a well-typed string is worth keeping so the caller can
            # commit it without re-billing those tokens.  Missing langs
            # per block are the caller's responsibility to backfill
            # (typically via a single-target per-entry fallback — see
            # ``MarkdownProcessor._translate_group_multi``).  Only keys
            # with NO valid lang at all trigger bisection.
            if langs_ok:
                good[k] = langs_ok

        missing = {k: chunk[k] for k in chunk if k not in good}
        if not missing:
            return good

        logger.info(
            "Multi-target batch missing %d/%d keys; retrying subset",
            len(missing),
            len(chunk),
        )
        good.update(self._bisect_multi(missing))
        return good

    def _bisect_multi(
        self, chunk: Dict[str, str]
    ) -> Dict[str, Dict[str, str]]:
        """Halve chunk and translate each half.  Single-entry failure → {}."""
        if len(chunk) <= 1:
            return {}
        keys = list(chunk.keys())
        mid = len(keys) // 2
        left = {k: chunk[k] for k in keys[:mid]}
        right = {k: chunk[k] for k in keys[mid:]}
        out: Dict[str, Dict[str, str]] = {}
        out.update(self._translate_chunk_multi(left))
        out.update(self._translate_chunk_multi(right))
        return out
