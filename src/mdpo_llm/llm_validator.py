"""
LLM-based translation grader.

Issues a single batched JSON call that reads ``{key: {source, output}}``
pairs and returns ``{key: {binary_score, reason}}`` grades.  Mirrors
:class:`mdpo_llm.batch.BatchTranslator`'s partition + bisection
scaffolding so a malformed validator response on a large batch
degrades into single-entry calls instead of failing the whole batch.

The retry orchestration that consumes the grades — accumulating reject
reasons across attempts, switching to a fallback model halfway through
the budget, marking residual failures fuzzy — lives in
:class:`mdpo_llm.processor.MarkdownProcessor`; this module only owns
the validator wire format.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, Literal, Optional

from .batch import BatchTranslator

logger = logging.getLogger(__name__)

GradeScore = Literal["yes", "no"]


@dataclass(frozen=True)
class BinaryGrade:
    """One validator verdict.

    ``binary_score`` is the literal pass/fail flag the prompt asks for;
    ``reason`` is a short human-readable explanation (always populated
    on the wire, even on PASS — the prompt requires it because passing
    sibling keys feed intra-batch context to retries on failed keys).
    Implemented as a plain frozen dataclass rather than a Pydantic
    model so the package keeps its current "no extra runtime deps"
    posture; ``BatchTranslator._parse_response`` already reuses
    ``json.loads`` for the same wire format, and
    :meth:`from_payload` does the literal-score validation that a
    Pydantic ``Literal`` type would have done.
    """

    binary_score: GradeScore
    reason: str

    @classmethod
    def from_payload(cls, payload: Any) -> Optional["BinaryGrade"]:
        """Coerce a parsed JSON value into a :class:`BinaryGrade`.

        Returns ``None`` when the payload is missing required fields,
        has a non-literal score, or is not a JSON object — so the
        caller can route the key into the bisection retry path the
        same way :class:`BatchTranslator` does on a missing key.
        """
        if not isinstance(payload, dict):
            return None
        score = payload.get("binary_score")
        if score not in ("yes", "no"):
            return None
        reason_raw = payload.get("reason")
        # ``None`` and missing both round-trip to an empty string so a
        # passing grade with no reason field still yields a valid
        # object.  Non-string types are coerced via ``str()`` so an
        # over-eager model that returns a number doesn't poison the
        # downstream prompt assembly.
        if reason_raw is None:
            reason = ""
        elif isinstance(reason_raw, str):
            reason = reason_raw.strip()
        else:
            reason = str(reason_raw).strip()
        return cls(binary_score=score, reason=reason)


GradePayload = Dict[str, str]
"""Wire-level value shape: ``{"source": str, "output": str}``."""


class LLMValidator:
    """Grade many ``(source, output)`` pairs in a single LLM call.

    The constructor takes a ``call_llm`` closure that owns message
    construction (system prompt, target-lang context, JSON response
    mode) — this class only marshals the wire object, partitions the
    batch, and bisects on parse failure.  Validator-level retries with
    accumulated rejection reasons and the fallback-model switch live
    in the caller (see ``MarkdownProcessor._llm_validate_and_retry``).
    """

    def __init__(
        self,
        call_llm: Callable[[Dict[str, GradePayload]], str],
        max_entries: int = 40,
        max_chars: int = 8000,
    ):
        """
        Args:
            call_llm: Callable that takes the validator batch's
                ``{key: {source, output}}`` dict and returns the raw
                LLM response (a JSON object whose keys mirror the
                input and whose values are
                ``{binary_score, reason}``).
            max_entries: Max items per validator request.
            max_chars: Soft cap on total ``len(source) + len(output)``
                characters per request.  Validator output is much
                smaller than translation output, so this can be looser
                than the translator's cap; the brief calls for the
                same default for accounting simplicity.
        """
        self._call_llm = call_llm
        self.max_entries = max_entries
        self.max_chars = max_chars

    def grade(
        self, items: Dict[str, GradePayload]
    ) -> Dict[str, BinaryGrade]:
        """Grade every entry in ``items``.

        Returns a ``{key: BinaryGrade}`` mapping.  Keys absent from
        the return are the validator equivalent of "could not get a
        verdict" — the caller treats them as failed and lets the
        retry budget handle them, the same way an unparseable
        translation falls into single-entry bisection in
        :class:`BatchTranslator`.
        """
        if not items:
            return {}
        out: Dict[str, BinaryGrade] = {}
        for chunk in self._partition(items):
            out.update(self._grade_chunk(chunk))
        return out

    def _partition(
        self, items: Dict[str, GradePayload]
    ) -> Iterator[Dict[str, GradePayload]]:
        current: Dict[str, GradePayload] = {}
        current_chars = 0
        for k, payload in items.items():
            chars = len(payload.get("source", "")) + len(
                payload.get("output", "")
            )
            if current and (
                len(current) >= self.max_entries
                or current_chars + chars > self.max_chars
            ):
                yield current
                current = {}
                current_chars = 0
            current[k] = payload
            current_chars += chars
        if current:
            yield current

    def _grade_chunk(
        self, chunk: Dict[str, GradePayload]
    ) -> Dict[str, BinaryGrade]:
        try:
            raw = self._call_llm(chunk)
        except Exception as exc:
            logger.warning(
                "Validator call raised (%d items): %s; bisecting",
                len(chunk),
                exc,
            )
            return self._bisect(chunk)

        parsed = BatchTranslator._parse_response(raw)
        if parsed is None:
            logger.warning(
                "Validator response unparseable (%d items); bisecting",
                len(chunk),
            )
            return self._bisect(chunk)

        good: Dict[str, BinaryGrade] = {}
        for k in chunk:
            grade = BinaryGrade.from_payload(parsed.get(k))
            if grade is not None:
                good[k] = grade

        missing = {k: chunk[k] for k in chunk if k not in good}
        if not missing:
            return good

        logger.info(
            "Validator missing %d/%d keys; retrying subset",
            len(missing),
            len(chunk),
        )
        good.update(self._bisect(missing))
        return good

    def _bisect(
        self, chunk: Dict[str, GradePayload]
    ) -> Dict[str, BinaryGrade]:
        """Halve chunk and grade each half. Single-entry failure → ``{}``.

        Mirrors :meth:`BatchTranslator._bisect` so a single key whose
        grading consistently fails (model OOMs on the pair, JSON
        contract repeatedly broken) doesn't block the rest of the
        batch — that key drops out of the result, the caller treats
        it as ungraded, and the surrounding retry budget decides what
        to do with it.
        """
        if len(chunk) <= 1:
            return {}
        keys = list(chunk.keys())
        mid = len(keys) // 2
        left = {k: chunk[k] for k in keys[:mid]}
        right = {k: chunk[k] for k in keys[mid:]}
        out: Dict[str, BinaryGrade] = {}
        out.update(self._grade_chunk(left))
        out.update(self._grade_chunk(right))
        return out
