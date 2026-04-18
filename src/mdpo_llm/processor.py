"""
Main Markdown Translator orchestrator class.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
import warnings
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

logger = logging.getLogger(__name__)

import litellm
import polib

from .batch import BatchTranslator, MultiTargetBatchTranslator
from .manager import POManager
from .parser import BlockParser, slugify_path_segment
from .placeholder import (
    ANCHOR_PATTERN,
    BUILTIN_PATTERNS,
    HTML_ATTR_PATTERN,
    Placeholder,
    PlaceholderMap,
    PlaceholderRegistry,
    check_round_trip,
    check_structural_position,
)
from .prompts import Prompts
from .reconstructor import DocumentReconstructor
from .reference_pool import ReferencePool
from .results import BatchStats, Coverage, DirectoryResult, ProcessResult, Receipt
from .validator import (
    ValidationIssue,
    check_language_stability,
    validate as validate_translation,
)


ValidationMode = Literal["off", "conservative", "strict"]
GlossaryMode = Literal["instruction", "placeholder"]
Mode = Literal["translate", "refine"]


_INPLACE_DEPRECATION_MESSAGE = (
    "`inplace=True` is deprecated and will be removed in v0.5. "
    "Use `mode='refine'` with a separate `refined_path` to produce a "
    "polished version of the source while keeping the original `msgid` "
    "intact. See README 'Refine mode' for migration details."
)


@dataclass(frozen=True)
class ProgressEvent:
    """Lightweight event emitted by :class:`MarkdownProcessor` progress hook.

    ``kind`` is one of ``"directory_start"``, ``"file_start"``,
    ``"file_end"``, ``"directory_end"``, ``"document_start"``,
    ``"document_progress"``, ``"document_end"``. Other fields are optional
    and depend on ``kind``. The library is UI-agnostic — callers decide
    how to render.
    """

    kind: str
    path: Optional[str] = None
    index: Optional[int] = None
    total: Optional[int] = None
    status: Optional[str] = None


ProgressCallback = Callable[[ProgressEvent], None]


class _UsageAccumulator:
    """Sum token usage and API-call counts across a single run.

    Tolerant of missing / non-numeric ``usage`` fields: providers that don't
    report usage (or mock responses in tests) still increment ``api_calls``
    without corrupting token totals.
    """

    __slots__ = ("input_tokens", "output_tokens", "api_calls")

    def __init__(self) -> None:
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.api_calls: int = 0

    def record(self, response: Any) -> None:
        self.api_calls += 1
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        self.input_tokens += _coerce_int(getattr(usage, "prompt_tokens", 0))
        self.output_tokens += _coerce_int(getattr(usage, "completion_tokens", 0))


def _coerce_int(value: Any) -> int:
    """Return ``int(value)`` when it is a real number, else ``0``.

    Guards against ``MagicMock`` responses (which would otherwise return a
    truthy object) and providers that stuff ``None`` into usage fields.
    """
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return 0
    return 0


def _resolve_pricing(model: str) -> Tuple[Optional[float], Optional[float]]:
    """Return ``(input_per_1m_usd, output_per_1m_usd)`` from ``litellm.model_cost``.

    Returns ``(None, None)`` when the model isn't listed or the per-token
    rates are missing — CLI renderers surface this as ``"—"`` so unpriced
    models don't silently report ``$0.00``.
    """
    try:
        cost_table = getattr(litellm, "model_cost", {}) or {}
    except Exception:
        return None, None

    candidates = [model]
    if "/" in model:
        candidates.append(model.split("/", 1)[1])

    for key in candidates:
        entry = cost_table.get(key)
        if not isinstance(entry, dict):
            continue
        ipt = entry.get("input_cost_per_token")
        opt = entry.get("output_cost_per_token")
        if isinstance(ipt, (int, float)) and isinstance(opt, (int, float)):
            return float(ipt) * 1_000_000.0, float(opt) * 1_000_000.0
    return None, None


def _build_receipt(
    *,
    model: str,
    target_lang: str,
    source_path: Optional[str],
    target_path: Optional[str],
    po_path: Optional[str],
    usage: _UsageAccumulator,
    duration_seconds: float,
) -> Receipt:
    in_per_1m, out_per_1m = _resolve_pricing(model)
    if in_per_1m is not None and out_per_1m is not None:
        input_cost = usage.input_tokens * in_per_1m / 1_000_000.0
        output_cost = usage.output_tokens * out_per_1m / 1_000_000.0
        total_cost: Optional[float] = input_cost + output_cost
    else:
        input_cost = None
        output_cost = None
        total_cost = None

    return Receipt(
        model=model,
        target_lang=target_lang,
        source_path=source_path,
        target_path=target_path,
        po_path=po_path,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.input_tokens + usage.output_tokens,
        api_calls=usage.api_calls,
        duration_seconds=duration_seconds,
        input_cost_per_1m_usd=in_per_1m,
        output_cost_per_1m_usd=out_per_1m,
        input_cost_usd=input_cost,
        output_cost_usd=output_cost,
        total_cost_usd=total_cost,
    )


class MarkdownProcessor:
    """Main orchestrator for markdown process workflow."""

    SKIP_TYPES = ["hr"]  # Block types to skip processing

    def __init__(
        self,
        model: str,
        target_lang: str,
        max_reference_pairs: int = 5,
        extra_instructions: str | None = None,
        post_process: Callable[[str], str] | None = None,
        glossary: Dict[str, str | None] | None = None,
        glossary_path: Path | str | None = None,
        batch_size: int = 40,
        batch_max_chars: int = 8000,
        batch_concurrency: int = 1,
        validation: ValidationMode = "off",
        enable_prompt_cache: bool = False,
        progress_callback: Optional[ProgressCallback] = None,
        placeholders: Optional[PlaceholderRegistry] = None,
        glossary_mode: GlossaryMode = "instruction",
        mode: Mode = "translate",
        **litellm_kwargs,
    ):
        """
        Initialize the processor.

        Args:
            model: Any LiteLLM model string (e.g. ``"gpt-4"``,
                ``"anthropic/claude-sonnet-4-5-20250929"``, ``"gemini/gemini-pro"``).
            target_lang: BCP 47 locale string (e.g. ``"ko"``).  Baked into
                the system prompt sent to the LLM.
            max_reference_pairs: Maximum number of similar reference pairs
                to pass as context to the LLM per entry (or per batch).
            extra_instructions: Additional instructions appended to the
                default translation prompt.
            post_process: Optional callable applied to every LLM response
                before it is stored.
            glossary: Inline glossary mapping terms to translations.
                ``None`` values mean "do not translate".
            glossary_path: Path to a JSON glossary file.
            batch_size: Max entries per batched LLM call.  ``0`` disables
                batching entirely and restores the v0.2 per-entry path.
            batch_max_chars: Soft cap on total source characters per batch.
            batch_concurrency: EXPERIMENTAL. When ``> 1``, batches within a
                single document may fly in parallel after the first batch
                has seeded the reference pool.  The first section-aware
                group is always processed sequentially so subsequent
                groups inherit at least one fresh few-shot pair for
                terminology / tone; the remaining groups are submitted to
                a thread pool of size ``batch_concurrency``.  Defaults to
                ``1`` (no intra-file concurrency), which preserves v0.4
                determinism.  Ignored on the sequential path
                (``batch_size == 0``) and when a document partitions into
                a single group (nothing to parallelise after seeding).
            validation: Post-translation validation mode.  ``"off"`` (default),
                ``"conservative"`` (structural checks), or ``"strict"`` (adds
                inline-code count check).  Failing entries are marked fuzzy.
            enable_prompt_cache: Pass ``cache_control`` hints on the stable
                system prefix so providers that support prompt caching
                (Anthropic native, OpenAI automatic) can reuse tokens across
                batches and re-runs.
            progress_callback: Optional one-argument callable invoked with a
                :class:`ProgressEvent` at key points during
                ``process_document`` / ``process_directory``. Emitted from
                worker threads in the directory path, so callers that touch
                shared UI state must handle thread-safety themselves. A
                callback that raises is logged and suppressed so progress
                rendering never breaks the actual work.
            placeholders: Optional :class:`PlaceholderRegistry` whose patterns
                are registered IN ADDITION to the always-on T-6 built-ins
                (``{#anchor-id}`` Kramdown attributes and raw-HTML attribute
                pairs like ``class="bare"``).  Every match is substituted
                with an opaque ``\u27e6P:N\u27e7`` token before the source
                block is sent to the LLM and restored afterwards; the
                round-trip check flags any mapped token that is missing,
                duplicated, or unexpected in the output as a structural
                fail — independent of the ``validation`` mode.
                ``None`` (default) runs with only the T-6 built-ins;
                passing an extra registry layers additional patterns on top.
            mode: ``"translate"`` (default) issues cross-language translations
                using :attr:`Prompts.TRANSLATE_SYSTEM_TEMPLATE` and runs the
                translate-purpose validator (target-language-presence check).
                ``"refine"`` issues same-language clarity/grammar polish using
                :attr:`Prompts.REFINE_SYSTEM_TEMPLATE`, selects the batched
                refine prompt, and runs the refine-purpose validator
                (language-stability check replaces target-language-presence).
                In refine mode ``target_lang`` names the source/output
                language (refine never switches languages), ``inplace`` must
                be ``False``, and the PO ``msgid`` is NEVER overwritten — a
                processed entry stores the refined text in ``msgstr`` and
                the original source stays authoritative.
            glossary_mode: How the configured glossary is fed to the LLM.
                ``"instruction"`` (default, v0.4 back-compat) appends a
                glossary block to the system prompt.  ``"placeholder"``
                substitutes each matching term with an opaque
                ``\u27e6P:N\u27e7`` token before the call and restores the
                target-language form (or the original term for
                do-not-translate entries) afterwards.  Matching uses
                case-sensitive word-boundary regex (``\\bterm\\b``);
                trailing morphology is NOT matched ("APIs" does not hit
                "API") — a false-negative is preferred over a mid-word
                false-positive.  Terms whose first or last character isn't
                a word character (e.g. ``.NET``, ``C++``) are silently
                skipped because the ``\\b`` anchors would reject them.
                Ignored when no glossary is configured.
            **litellm_kwargs: Extra keyword arguments forwarded to
                ``litellm.completion()``.
        """
        self.parser = BlockParser()
        self.po_manager = POManager(skip_types=self.SKIP_TYPES)
        self.reconstructor = DocumentReconstructor(skip_types=self.SKIP_TYPES)
        self.model = model
        self.target_lang = target_lang
        self.max_reference_pairs = max_reference_pairs
        self._extra_instructions = extra_instructions
        self._post_process = post_process
        # Keep the raw inputs so :meth:`process_document_multi` can
        # re-resolve locale-specific glossary entries for each target
        # language — the baked ``self._glossary`` below is pinned to
        # ``target_lang`` and would silently feed wrong-locale
        # replacements into other langs otherwise.
        self._glossary_inline = (
            dict(glossary) if glossary else None
        )
        self._glossary_file: Optional[Dict[str, Any]] = None
        if glossary_path:
            self._glossary_file = json.loads(
                Path(glossary_path).read_text(encoding="utf-8")
            )
        self._glossary = self._resolve_glossary(glossary, glossary_path)
        self.batch_size = batch_size
        self.batch_max_chars = batch_max_chars
        try:
            concurrency_value = int(batch_concurrency)
        except (TypeError, ValueError):
            concurrency_value = 1
        # Clamp non-positive values to ``1`` so a 0/negative kwarg
        # degrades to the safe sequential path instead of raising deep
        # inside the thread pool.
        self.batch_concurrency = max(1, concurrency_value)
        self.validation: ValidationMode = validation
        self.enable_prompt_cache = enable_prompt_cache
        self._progress_callback = progress_callback
        self._placeholders = placeholders
        self.glossary_mode: GlossaryMode = glossary_mode
        if mode not in ("translate", "refine"):
            raise ValueError(
                f"mode must be 'translate' or 'refine', got {mode!r}"
            )
        self.mode: Mode = mode
        if mode == "refine" and self._glossary:
            # Translation glossaries typically map terms to a
            # target-language form (e.g. ``"pull request" → "풀 리퀘스트"``).
            # Applying those mappings during a same-language refine
            # would deterministically inject target-language strings
            # into the refined output — either via the instruction-mode
            # prompt block or via placeholder-mode decode — which
            # violates the refine contract.  Disable glossary handling
            # in refine mode entirely; callers who need refine-specific
            # vocabulary enforcement should swap the glossary when they
            # switch modes or run refine as a standalone pass.
            self._glossary = None
        # Effective registry merges user-supplied patterns with glossary
        # patterns when glossary_mode=="placeholder"; ``_encode_source``
        # calls into this (not ``_placeholders`` directly) so every
        # callsite picks up glossary substitutions consistently.
        self._effective_registry = self._build_effective_registry(placeholders)
        self._litellm_kwargs = litellm_kwargs
        # Thread-local hand-off slot so ``process_directory`` can expose each
        # worker's accumulator to ``process_document`` without changing that
        # method's public signature (which existing tests monkey-patch).
        self._tls = threading.local()

    # ----- progress hook -----

    def _emit_progress(self, **fields: Any) -> None:
        """Invoke ``progress_callback`` if set; swallow exceptions.

        A broken UI callback must never abort translation — if rendering
        fails the actual API calls and PO writes should still complete.
        """
        cb = self._progress_callback
        if cb is None:
            return
        try:
            cb(ProgressEvent(**fields))
        except Exception:
            logger.exception("progress_callback raised; continuing")

    def _collect_pending_entries(
        self, po_file: polib.POFile
    ) -> Tuple[List[polib.POEntry], int]:
        """Return ``(pending, skipped_count)`` for a PO file.

        Centralised so ``process_document`` can compute the progress
        ``total`` once — and emit ``document_start`` / ``document_end``
        around the whole per-document pipeline (rebuild + save included)
        rather than around just the translation loop.
        """
        pending: List[polib.POEntry] = []
        skipped = 0
        for entry in po_file:
            if entry.obsolete:
                continue
            block_type = self._extract_block_type_from_msgctxt(entry.msgctxt)
            if block_type in self.SKIP_TYPES:
                skipped += 1
                continue
            needs_translation = (not entry.msgstr) or ("fuzzy" in entry.flags)
            if not needs_translation:
                continue
            pending.append(entry)
        return pending, skipped

    # ----- glossary -----

    def _resolve_glossary(self, glossary, glossary_path):
        resolved: Dict[str, str | None] = {}

        if glossary_path:
            raw = json.loads(Path(glossary_path).read_text(encoding="utf-8"))
            for term, value in raw.items():
                if value is None:
                    resolved[term] = None
                elif isinstance(value, str):
                    resolved[term] = value
                elif isinstance(value, dict):
                    resolved[term] = value.get(self.target_lang)

        if glossary:
            resolved.update(glossary)

        return resolved or None

    def _resolve_glossary_for_lang(
        self, target_lang: str
    ) -> Optional[Dict[str, Optional[str]]]:
        """Re-resolve the configured glossary for an arbitrary locale.

        Used by :meth:`process_document_multi` so each target language
        gets its own locale-specific glossary resolution rather than
        inheriting ``self.target_lang``'s pinning.  Falls back to the
        baked ``self._glossary`` when the processor was built without
        either a ``glossary_path`` or an inline ``glossary``, so callers
        who never configured one pay no overhead.
        """
        if not self._glossary_inline and self._glossary_file is None:
            return self._glossary
        resolved: Dict[str, Optional[str]] = {}
        if self._glossary_file is not None:
            for term, value in self._glossary_file.items():
                if value is None:
                    resolved[term] = None
                elif isinstance(value, str):
                    resolved[term] = value
                elif isinstance(value, dict):
                    resolved[term] = value.get(target_lang)
        if self._glossary_inline:
            resolved.update(self._glossary_inline)
        return resolved or None

    def _format_glossary_for_lang(
        self, source_text: str, target_lang: str
    ) -> Optional[str]:
        """Per-lang counterpart of :meth:`_format_glossary`."""
        glossary = self._resolve_glossary_for_lang(target_lang)
        if not glossary or self.glossary_mode != "instruction":
            return None
        relevant = {k: v for k, v in glossary.items() if k in source_text}
        if not relevant:
            return None
        lines = []
        for term, translation in relevant.items():
            if translation is None:
                lines.append(f'- "{term}" \u2192 do not translate')
            else:
                lines.append(f'- "{term}" \u2192 "{translation}"')
        return "Glossary (use these exact forms, do not alter):\n" + "\n".join(lines)

    def _format_glossary(self, source_text: str) -> str | None:
        # In placeholder mode the glossary is fed to the LLM as opaque
        # tokens in the user message, so the instruction-mode block is
        # redundant and would leak target-language terms that the model
        # might echo outside the protected spans.
        if not self._glossary or self.glossary_mode != "instruction":
            return None

        relevant = {k: v for k, v in self._glossary.items() if k in source_text}
        if not relevant:
            return None

        lines = []
        for term, translation in relevant.items():
            if translation is None:
                lines.append(f'- "{term}" \u2192 do not translate')
            else:
                lines.append(f'- "{term}" \u2192 "{translation}"')
        return "Glossary (use these exact forms, do not alter):\n" + "\n".join(lines)

    def _sibling_refine_processor(self, target_lang: str) -> "MarkdownProcessor":
        """Build a refine-mode clone sharing this processor's config.

        Used by :meth:`process_document` to run the first pass of a
        ``refine_first=True`` translate composition without mutating
        ``self.mode`` (which would be thread-unsafe and leak state on
        failure).  Shared knobs — batching, placeholders, extra
        instructions, LiteLLM kwargs — are copied through so the refine
        pass uses the same provider and the same guardrails as the
        translate pass.

        The translation glossary is deliberately **not** forwarded: its
        target-language mappings are defined for the translate pass and
        would leak target-language terms into the source-language refine
        output (deterministically under
        ``glossary_mode="placeholder"``, where decode rewrites tokens to
        the glossary replacement).  Callers who want refine-specific
        glossary behaviour can run the refine pass standalone with its
        own configured glossary.

        The progress callback is also withheld: forwarding it would make
        one public ``process_document`` call emit TWO independent
        ``document_start``/``document_end`` lifecycles (one per pass),
        which consumers (including the CLI's rich bar) correlate 1:1
        with public calls.  The translate-pass callback alone drives the
        outer progress; refine-pass progress would double-count or
        overwrite the UI state.
        """
        return MarkdownProcessor(
            model=self.model,
            target_lang=target_lang,
            max_reference_pairs=self.max_reference_pairs,
            extra_instructions=self._extra_instructions,
            post_process=self._post_process,
            glossary=None,
            batch_size=self.batch_size,
            batch_max_chars=self.batch_max_chars,
            batch_concurrency=self.batch_concurrency,
            validation=self.validation,
            enable_prompt_cache=self.enable_prompt_cache,
            progress_callback=None,
            placeholders=self._placeholders,
            glossary_mode=self.glossary_mode,
            mode="refine",
            **self._litellm_kwargs,
        )

    # ----- per-entry messaging -----

    def _build_messages(
        self,
        source_text: str,
        reference_pairs=None,
        *,
        glossary_source: Optional[str] = None,
    ):
        if self.mode == "refine":
            instruction = Prompts.REFINE_INSTRUCTION
            system_template = Prompts.REFINE_SYSTEM_TEMPLATE
        else:
            instruction = Prompts.TRANSLATE_INSTRUCTION
            system_template = Prompts.TRANSLATE_SYSTEM_TEMPLATE
        if self._extra_instructions:
            instruction += "\n" + self._extra_instructions
        system_content = system_template.format(
            lang=self.target_lang,
            instruction=instruction,
        )

        # Glossary detection runs on the raw, unencoded source so a term
        # sitting inside a protected span (e.g. a URL or anchor that a
        # pattern turned into ``\u27e6P:N\u27e6``) still triggers the
        # glossary block.  ``glossary_source`` defaults to ``source_text``
        # so callers that don't encode stay identical to the pre-T-4 path,
        # which keeps sequential and batched processing consistent for
        # the same input.
        glossary_block = self._format_glossary(
            glossary_source if glossary_source is not None else source_text
        )
        if glossary_block:
            system_content += "\n\n" + glossary_block

        messages: List[Dict[str, Any]] = [self._system_message(system_content)]

        if reference_pairs:
            for ref_src, ref_tgt in reference_pairs:
                messages.append({"role": "user", "content": ref_src})
                messages.append({"role": "assistant", "content": ref_tgt})

        messages.append({"role": "user", "content": source_text})
        return messages

    def _system_message(self, text: str) -> Dict[str, Any]:
        """Build a system message, applying prompt-cache markers when enabled.

        ``cache_control`` in content-part form is an Anthropic-specific
        schema.  Emitting it unconditionally breaks providers that reject
        non-string ``content`` and is redundant for OpenAI (which caches
        eligible prefixes automatically) and for Gemini (which uses its own
        ``cachedContent`` mechanism).  So we only rewrite the message shape
        for Anthropic models; other providers get a plain string and benefit
        from their native caching where applicable.
        """
        if self.enable_prompt_cache and self._is_anthropic_model():
            return {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": text,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        return {"role": "system", "content": text}

    def _is_anthropic_model(self) -> bool:
        m = (self.model or "").lower()
        return (
            m.startswith("anthropic/")
            or m.startswith("claude")
            or "/claude" in m  # e.g. "bedrock/anthropic.claude-…"
            or "anthropic." in m
        )

    def _supports_json_mode(self) -> bool:
        """Return True when the current model advertises ``response_format`` support.

        Falls back to ``False`` when LiteLLM cannot give a concrete answer
        (older versions, custom adapters, probe exceptions).  Forcing JSON
        mode against a provider that rejects the flag drives BatchTranslator
        into a full bisection tree of failing API calls before the per-entry
        fallback fires — the exact pathology this gate exists to avoid.
        """
        try:
            params = litellm.get_supported_openai_params(model=self.model)
        except Exception:
            return False
        if isinstance(params, (list, tuple, set)):
            return "response_format" in params
        return False

    # ----- placeholder pipeline -----

    def _encode_source(self, text: str) -> Tuple[str, PlaceholderMap]:
        """Run the effective placeholder registry over ``text``.

        The effective registry always starts with the T-6 built-ins
        (``anchor`` + ``html_attr``) and layers user-supplied patterns
        and glossary placeholder patterns on top (see
        :meth:`_build_effective_registry`).  Returns ``(encoded, mapping)``;
        the mapping may be empty when no pattern matched, so callers can
        rely on ``bool(mapping)`` to tell an active encoding from a no-op.
        """
        encoded, mapping = self._effective_registry.encode(text)
        mapping = self._apply_glossary_replacements(mapping)
        return encoded, mapping

    def _build_effective_registry(
        self, user_registry: Optional[PlaceholderRegistry]
    ) -> PlaceholderRegistry:
        """Compose the registry the processor actually uses on every call.

        Registration order determines the tie-break: encode sorts
        candidates by ``(earliest start, longest match)`` and Python's
        stable sort keeps insertion order on remaining ties, so
        **earlier-registered wins equal-start/equal-length overlaps**.
        Patterns are therefore registered in descending priority:

            1. User-supplied ``placeholders`` — the caller's explicit
               intent.  Registered first so a user pattern wins on
               exact-match ties.  Additionally, any user pattern whose
               ``name`` matches a built-in (``anchor`` / ``html_attr``)
               SUPPRESSES that built-in entirely: the caller's regex
               and predicate replace the defaults end-to-end, which
               gives "stricter predicate" overrides a path that
               actually works.  No-opt-out applies to the *behaviour*
               (something named ``anchor`` / ``html_attr`` is still
               always on), not to the specific regex.
            2. T-6 built-ins (``anchor``, ``html_attr``) — no opt-out
               flag except the by-name override above.  These protect
               Markdown anchor IDs and raw HTML attribute pairs that
               have been observed to mangle in real-world runs.
            3. Glossary patterns when ``glossary_mode == "placeholder"``
               and a glossary is configured.  Each term becomes a
               case-sensitive word-boundary pattern (``\\bterm\\b``);
               :meth:`_apply_glossary_replacements` stamps each entry
               with the target-language form so decode emits the
               translation rather than the original source span.

        **Morphology rule** — trailing morphology is NOT matched.
        ``"APIs"`` does not hit a glossary term ``"API"`` because the
        trailing ``'s'`` is a word character and breaks the trailing
        ``\\b``.  The choice is deliberate: a false-negative (no match)
        falls through to the LLM's normal translation path, while a
        false-positive would corrupt neighbouring text mid-word — the
        more expensive failure mode.

        **Non-word-boundary terms** (``.NET``, ``C++``) are skipped by
        :meth:`_compile_glossary_pattern` because ``\\b`` anchors would
        reject the entire term; they fall through to the LLM, consistent
        with the same prefer-false-negative rule.
        """
        registry = PlaceholderRegistry()
        # Tracks same-name user patterns whose ``replace_builtin=True``
        # flag explicitly suppresses the default built-in.  Without
        # that opt-in a user pattern named ``anchor`` / ``html_attr``
        # LAYERS on top of the default so every ``{#...}`` anchor /
        # HTML attribute is still tokenized even when the user's
        # regex or predicate would reject it — the T-6 "always-on /
        # no opt-out" contract.
        suppress_default: Dict[
            str,
            List[
                Tuple["re.Pattern[str]", Optional[Callable[[str, int, int], bool]]]
            ],
        ] = {}
        if user_registry is not None:
            for p in user_registry.patterns:
                registry.register(
                    p.name,
                    p.regex,
                    predicate=p.predicate,
                    replace_builtin=p.replace_builtin,
                )
                if p.replace_builtin:
                    suppress_default.setdefault(p.name, []).append(
                        (p.regex, p.predicate)
                    )
        for name, pattern, predicate in BUILTIN_PATTERNS:
            if name in suppress_default:
                continue
            registry.register(name, pattern, predicate=predicate)
        # Stash the override (regex, predicate) pairs per name so
        # ``_apply_validation`` runs ``check_structural_position`` with
        # the SAME matching behaviour the user configured for encoding
        # — a stricter predicate narrows the check's match set too,
        # instead of falling back to the default and flagging spans
        # the override never tokenized.
        self._builtin_overrides = suppress_default
        need_glossary = bool(self._glossary) and self.glossary_mode == "placeholder"
        if need_glossary:
            # Sort longest-first so "pull request" gets registered before
            # "pull"; the encode overlap resolver also prefers the longer
            # match on tie, but keeping insertion order sorted is defensive.
            terms = sorted(self._glossary.keys(), key=len, reverse=True)
            for term in terms:
                pattern = self._compile_glossary_pattern(term)
                if pattern is None:
                    continue
                registry.register(f"glossary:{term}", pattern)
        return registry

    @staticmethod
    def _compile_glossary_pattern(term: str) -> Optional["re.Pattern[str]"]:
        """Case-sensitive word-boundary regex for a glossary term.

        Returns ``None`` when ``term`` starts or ends with a non-word
        character because ``\\b`` anchors reject such a pattern at both
        ends (``\\b.NET\\b`` never matches ``.NET``).  Skipped terms
        silently fall through to the LLM's normal translation path —
        the same prefer-false-negative principle that guides the
        morphology rule.
        """
        if not term:
            return None
        if not (term[0].isalnum() or term[0] == "_"):
            return None
        if not (term[-1].isalnum() or term[-1] == "_"):
            return None
        return re.compile(rf"\b{re.escape(term)}\b")

    def _apply_glossary_replacements(
        self, mapping: PlaceholderMap
    ) -> PlaceholderMap:
        """Stamp glossary entries with their target-language replacement.

        Pattern names of the form ``glossary:<term>`` are rewritten so
        :meth:`PlaceholderRegistry.decode` restores the token to the
        target-language form (``self._glossary[term]``) rather than the
        original source span.  A ``None`` glossary value means "do not
        translate" — the replacement falls back to the matched source
        text so decode emits the term verbatim.

        Non-glossary entries (user-registered patterns, pre-existing
        literal tokens) are returned untouched so their identity-decode
        behaviour is preserved.
        """
        if (
            not mapping.items
            or not self._glossary
            or self.glossary_mode != "placeholder"
        ):
            return mapping
        new_items: List[Placeholder] = []
        changed = False
        for p in mapping.items:
            if not p.pattern_name.startswith("glossary:"):
                new_items.append(p)
                continue
            target = self._glossary.get(p.original)
            replacement = target if target is not None else p.original
            new_items.append(
                Placeholder(
                    token=p.token,
                    original=p.original,
                    pattern_name=p.pattern_name,
                    replacement=replacement,
                )
            )
            changed = True
        if not changed:
            return mapping
        return PlaceholderMap(items=new_items)

    def _decode_translation(
        self, text: str, mapping: Optional[PlaceholderMap]
    ) -> str:
        """Restore tokens in ``text`` using ``mapping``.

        ``None`` or an empty mapping is a no-op, so callers can stay
        placeholder-oblivious when no registry is configured.
        """
        if not mapping:
            return text
        return PlaceholderRegistry.decode(text, mapping)

    def _call_llm(
        self,
        source_text: str,
        reference_pairs=None,
        usage: Optional[_UsageAccumulator] = None,
    ):
        encoded_source, mapping = self._encode_source(source_text)
        messages = self._build_messages(
            encoded_source,
            reference_pairs,
            glossary_source=source_text,
        )
        response = litellm.completion(
            model=self.model, messages=messages, **self._litellm_kwargs
        )
        if usage is not None:
            usage.record(response)
        raw = response.choices[0].message.content
        # Run ``post_process`` BEFORE decoding so the round-trip check,
        # which reads ``last_encoded_response`` below, sees the final
        # text that will be committed to the PO entry.  If a caller's
        # hook edits a protected span (normalises URLs, strips
        # punctuation, etc.) after decode, the change would otherwise
        # slip past validation and corrupt the preserved content.  When
        # placeholders are active the hook sees ``\u27e6P:N\u27e7``
        # tokens — callers who want the decoded form should apply their
        # transform outside the pipeline.
        if self._post_process:
            raw = self._post_process(raw)
        # Stash the post-processed encoded response + mapping so
        # ``_apply_validation`` (same thread, same pipeline step) runs
        # the round-trip check against the exact text that will be
        # decoded and stored.  Cleared inside ``_apply_validation`` so a
        # later per-entry call on the same thread never inherits stale
        # context.
        self._tls.last_encoded_response = raw
        self._tls.last_placeholder_map = mapping
        return self._decode_translation(raw, mapping)

    # ----- batch messaging -----

    def _build_batch_messages(
        self,
        items: Dict[str, str],
        reference_pairs: Optional[List[tuple]] = None,
        glossary_block: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if self.mode == "refine":
            instruction = Prompts.BATCH_REFINE_INSTRUCTION
            system_template = Prompts.BATCH_REFINE_SYSTEM_TEMPLATE
        else:
            instruction = Prompts.BATCH_TRANSLATE_INSTRUCTION
            system_template = Prompts.BATCH_TRANSLATE_SYSTEM_TEMPLATE
        if self._extra_instructions:
            instruction += "\n" + self._extra_instructions
        system_content = system_template.format(
            lang=self.target_lang,
            instruction=instruction,
        )
        if glossary_block:
            system_content += "\n\n" + glossary_block
        if reference_pairs:
            ref_lines = [
                "Reference translations (maintain this tone and terminology):",
            ]
            for src, tgt in reference_pairs:
                ref_lines.append(f"- SRC: {src}\n  TGT: {tgt}")
            system_content += "\n\n" + "\n".join(ref_lines)

        user_payload = json.dumps(items, ensure_ascii=False)
        return [
            self._system_message(system_content),
            {"role": "user", "content": user_payload},
        ]

    def _make_batch_caller(
        self,
        reference_pairs: Optional[List[tuple]],
        glossary_block: Optional[str],
        usage: Optional[_UsageAccumulator] = None,
    ) -> Callable[[Dict[str, str]], str]:
        """Return a ``call_llm`` closure suitable for ``BatchTranslator``."""

        def _call(items: Dict[str, str]) -> str:
            messages = self._build_batch_messages(
                items, reference_pairs=reference_pairs, glossary_block=glossary_block
            )
            call_kwargs = dict(self._litellm_kwargs)
            # Request JSON object responses only when the configured model
            # advertises ``response_format`` support.  Providers that reject
            # the flag would otherwise fail the whole batch on every call
            # and force BatchTranslator into a costly bisection loop.
            if self._supports_json_mode():
                call_kwargs.setdefault("response_format", {"type": "json_object"})
            response = litellm.completion(
                model=self.model, messages=messages, **call_kwargs
            )
            if usage is not None:
                usage.record(response)
            return response.choices[0].message.content

        return _call

    # ----- main orchestration -----

    def process_document(
        self,
        source_path: Path,
        target_path: Path,
        po_path: Path | None = None,
        inplace: bool = False,
        *,
        refined_path: Path | None = None,
        refine_first: bool = False,
        refine_lang: Optional[str] = None,
        refined_po_path: Path | None = None,
    ) -> ProcessResult:
        """
        Process a markdown document.

        In the default translate mode (``self.mode == "translate"``) the
        LLM translates ``source_path`` into ``target_lang`` and writes the
        result to ``target_path``.  In refine mode (``self.mode == "refine"``)
        the LLM polishes ``source_path`` in its original language and writes
        the refined result to ``refined_path`` (or ``target_path`` when
        ``refined_path`` is omitted).  Refine never overwrites the PO
        ``msgid`` and is incompatible with ``inplace=True``.

        Setting ``refine_first=True`` in translate mode runs a refine pass
        before the translate pass: a sibling processor refines the source
        into ``refined_path`` (required), then the translate pass reads
        from ``refined_path`` and writes to ``target_path``.  Both passes
        bill into the returned :class:`Receipt`.  ``refine_lang`` picks the
        refine pass's language; defaults to ``self.target_lang``'s value
        only when the caller's flow is symmetric — otherwise pass it
        explicitly.
        """
        if inplace:
            warnings.warn(
                _INPLACE_DEPRECATION_MESSAGE,
                DeprecationWarning,
                stacklevel=2,
            )
        if self.mode == "refine" and inplace:
            raise ValueError(
                "inplace=True is incompatible with mode='refine'. "
                "Refine never overwrites the source or its PO msgid; "
                "use refined_path instead."
            )
        if refine_first and self.mode != "translate":
            raise ValueError(
                "refine_first=True only applies in mode='translate'."
            )
        tls_usage_installed_here = False
        if self.mode == "refine" and refined_path is not None:
            # In refine mode, honour refined_path as the output when the
            # caller supplies it; otherwise fall through to target_path so
            # the signature stays compatible with translate callers.
            target_path = Path(refined_path)
        if self.mode == "refine":
            # Refine contract: the original source document is NEVER
            # overwritten.  Compare resolved absolute paths so relative
            # vs absolute spellings (``./a.md`` vs ``a.md``) still trip
            # the guard before ``_save_processed_document`` runs.
            # ``Path.resolve(strict=False)`` is tolerant of missing
            # output paths (refined_path may not exist yet on first run).
            try:
                src_resolved = Path(source_path).resolve(strict=False)
                out_resolved = Path(target_path).resolve(strict=False)
            except OSError:
                src_resolved = Path(source_path).absolute()
                out_resolved = Path(target_path).absolute()
            if src_resolved == out_resolved:
                raise ValueError(
                    "Refine mode forbids writing the refined output back "
                    "on top of the source document. "
                    f"source_path and refined/target path both resolve to "
                    f"{src_resolved}."
                )
        if refine_first:
            if refined_path is None:
                raise ValueError(
                    "refine_first=True requires refined_path "
                    "(the intermediate refined Markdown location)."
                )
            if not refine_lang:
                # Falling back to ``self.target_lang`` is actively
                # harmful for typical en→ko / en→ja runs: it would pin
                # the refine pass to the translation TARGET language, so
                # the refine step ends up translating the source and the
                # downstream translate step becomes a no-op in the wrong
                # language.  Force the caller to name the refine
                # language explicitly — there is no safe default.
                raise ValueError(
                    "refine_first=True requires refine_lang "
                    "(the BCP 47 locale of the SOURCE document). "
                    "Do not rely on target_lang — refine is same-language "
                    "and target_lang names the translate pass's target."
                )
            refined_path = Path(refined_path)
            refine_proc = self._sibling_refine_processor(
                target_lang=refine_lang
            )
            refine_po = (
                Path(refined_po_path)
                if refined_po_path is not None
                else refined_path.with_suffix(".po")
            )
            # Distinctness guard: the refine and translate passes must
            # not share output artefacts.  If ``refined_path`` aliases
            # ``target_path`` (or their default POs collide), the
            # translate pass reopens the PO that the refine pass just
            # marked processed, so every unchanged block looks already
            # translated and the final document silently stays in the
            # source language.  Force the caller to pick distinct
            # paths / POs before we burn tokens on a no-op run.  The
            # default PO derivation (``target.with_suffix(".po")`` /
            # ``refined_path.with_suffix(".po")``) is folded into the
            # same check so callers who only pass the ``.md`` paths are
            # still protected.
            final_po_candidate = (
                Path(po_path) if po_path is not None
                else Path(target_path).with_suffix(".po")
            )
            try:
                refined_resolved = refined_path.resolve(strict=False)
                target_resolved = Path(target_path).resolve(strict=False)
                refine_po_resolved = refine_po.resolve(strict=False)
                final_po_resolved = final_po_candidate.resolve(strict=False)
            except OSError:
                refined_resolved = refined_path.absolute()
                target_resolved = Path(target_path).absolute()
                refine_po_resolved = refine_po.absolute()
                final_po_resolved = final_po_candidate.absolute()
            if refined_resolved == target_resolved:
                raise ValueError(
                    "refine_first=True requires distinct refined_path "
                    "and target_path; sharing a path would let the "
                    "translate pass reopen the refine-pass output."
                )
            if refine_po_resolved == final_po_resolved:
                raise ValueError(
                    "refine_first=True requires distinct PO files for "
                    "the refine and translate passes. Sharing a PO "
                    "means translate would see the refine entries as "
                    "already processed and skip translation entirely. "
                    "Pass an explicit ``po_path`` and/or "
                    "``refined_po_path`` that resolve to different "
                    "files."
                )
            # ``sync_po`` on the translate pass runs against blocks
            # parsed from the REFINED intermediate, so any pre-existing
            # translate PO (keyed on the original source msgids) gets
            # its entries detached on first refine-first run and
            # ``_remove_obsolete_entries`` wipes them outright — the
            # new translate PO starts empty of prior translation work.
            # Stash the old msgid→msgstr pairs on thread-local so the
            # translate pass's reference pool can seed from them as
            # few-shot context: they don't match the refined msgids
            # exactly, but the LLM still benefits from prior
            # terminology / tone when translating the refined version.
            carryover: List[Tuple[str, str]] = []
            if final_po_candidate.exists():
                try:
                    prior_po = polib.pofile(str(final_po_candidate))
                    for entry in prior_po:
                        if entry.obsolete or not entry.msgstr:
                            continue
                        if "fuzzy" in entry.flags:
                            continue
                        carryover.append((entry.msgid, entry.msgstr))
                except (OSError, UnicodeDecodeError, IOError):
                    # A malformed / unreadable prior PO shouldn't abort
                    # the refine-first flow; translate pass will proceed
                    # without carryover context.
                    pass
            if carryover:
                self._tls.refine_first_carryover = carryover
            # Install (or reuse) the thread-local usage slot BEFORE
            # invoking the refine pass so a mid-flight failure still
            # has somewhere to deposit billed tokens — otherwise a
            # crash between the refine call and the post-call merge
            # below drops the refine pass's cost on the floor.
            existing = getattr(self._tls, "usage", None)
            if existing is None:
                existing = _UsageAccumulator()
                self._tls.usage = existing
                # Track that the refine-first branch planted the
                # accumulator so the ``finally`` block can clear it on
                # exit.  Without that cleanup a subsequent
                # ``process_document`` call on the same thread would
                # pick up the stale usage and inflate its receipt with
                # this run's tokens.
                tls_usage_installed_here = True

            def _merge_refine_usage(receipt_like: Any) -> None:
                if receipt_like is None:
                    return
                existing.input_tokens += getattr(receipt_like, "input_tokens", 0)
                existing.output_tokens += getattr(receipt_like, "output_tokens", 0)
                existing.api_calls += getattr(receipt_like, "api_calls", 0)

            try:
                refine_result = refine_proc.process_document(
                    source_path=source_path,
                    target_path=refined_path,
                    po_path=refine_po,
                )
            except BaseException as refine_exc:
                # A refine-pass crash that happened AFTER LLM calls is
                # annotated with ``partial_receipt`` (see the main
                # process_document error handler).  Fold that usage in
                # before propagating so the outer ``partial_receipt``
                # the caller sees includes every billed token across
                # both passes — the whole point of the receipt.
                _merge_refine_usage(getattr(refine_exc, "partial_receipt", None))
                raise
            # Hand the combined usage forward so the translate pass's
            # receipt reflects tokens from BOTH passes; without this the
            # caller's cost view would omit the refine step.  The
            # sibling's billed usage is already materialised in its
            # ``receipt`` so we don't have to scrape a stale thread-local.
            _merge_refine_usage(refine_result.receipt)
            # The translate pass works against the refined intermediate
            # end-to-end: its PO is keyed on refined msgids, so every
            # piece of returned metadata (ProcessResult.source_path,
            # Receipt.source_path, progress-event path) must also name
            # the refined file.  Otherwise downstream PO helpers
            # (``get_translation_stats``, ``export_report``,
            # ``estimate``) re-read the caller's ORIGINAL source,
            # resync the PO against its unrefined blocks, and mark
            # every refined-msgid entry as obsolete.  The caller
            # already knows their own source path from the call args;
            # emitting the refined path back makes the PO/source
            # pairing unambiguous for automation.
            read_path: Path = refined_path
            source_path = refined_path
        else:
            read_path = Path(source_path)
        if po_path is None:
            po_path = Path(target_path).with_suffix(".po")
        parser = BlockParser()
        po_manager = POManager(skip_types=self.SKIP_TYPES)
        reconstructor = DocumentReconstructor(skip_types=self.SKIP_TYPES)
        # ``process_directory`` hands its per-worker accumulator in via this
        # thread-local slot so tokens already billed before an exception
        # still reach the directory-level receipt.  Single-file calls find
        # no slot and allocate their own.
        usage = getattr(self._tls, "usage", None) or _UsageAccumulator()
        start = time.monotonic()

        po_file = None
        po_saved = False
        source_path_str = str(source_path)
        document_started = False
        try:
            source = read_path.read_text(encoding="utf-8")
            source_lines = source.splitlines(keepends=True)
            blocks = parser.segment_markdown(
                [line.rstrip("\n") for line in source_lines]
            )

            po_file = po_manager.load_or_create_po(
                po_path, target_lang=self.target_lang
            )
            po_manager.sync_po(po_file, blocks, parser.context_id)

            # Compute progress total up-front so ``document_start`` /
            # ``document_end`` bracket the ENTIRE per-document pipeline
            # (translation + reconstruction + inplace + save), not just
            # the translation loop.  Emitting ``document_end`` from
            # inside the inner helpers would mark failed runs as
            # completed if rebuild / save / inplace raised after
            # translation finished.
            pending, initial_skipped = self._collect_pending_entries(po_file)
            if self.batch_size and self.batch_size > 0:
                progress_total = len(self._section_aware_groups(pending))
            else:
                progress_total = len(pending)
            self._emit_progress(
                kind="document_start",
                path=source_path_str,
                total=progress_total,
            )
            document_started = True

            if self.batch_size and self.batch_size > 0:
                translation_stats = self._process_entries_batched(
                    po_file,
                    po_manager,
                    inplace=inplace,
                    usage=usage,
                    source_path=source_path_str,
                    pending=pending,
                    initial_skipped=initial_skipped,
                )
            else:
                translation_stats = self._process_entries_sequential(
                    po_file,
                    po_manager,
                    inplace=inplace,
                    usage=usage,
                    source_path=source_path_str,
                    pending=pending,
                    initial_skipped=initial_skipped,
                )

            coverage_dict = reconstructor.get_process_coverage(
                blocks, po_file, parser.context_id
            )

            processed_content = reconstructor.rebuild_markdown(
                source_lines, blocks, po_file, parser.context_id
            )

            if inplace:
                # Capture validator/fuzzy state BEFORE the redraw so the
                # replacement PO can inherit fuzzy flags and tcomments.
                # Without this, a validator failure followed by
                # ``inplace`` writes a fresh entry with no fuzzy flag
                # and the next run treats the bad translation as fully
                # processed.
                #
                # Keying by msgid alone would collide on repeated labels
                # ("Overview", "OK", …).  Instead we track each msgid's
                # Nth occurrence in document order and match the Nth old
                # entry to the Nth new entry with the same msgid.
                preserved: Dict[tuple, Dict[str, Any]] = {}
                seen_old: Dict[str, int] = {}
                for entry in po_file:
                    if entry.obsolete or not entry.msgid:
                        continue
                    n = seen_old.get(entry.msgid, 0)
                    seen_old[entry.msgid] = n + 1
                    preserved[(entry.msgid, n)] = {
                        "flags": list(entry.flags),
                        "tcomment": entry.tcomment,
                    }

                self._match_ctxt(
                    processed_content=processed_content,
                    parser=parser,
                    po_manager=po_manager,
                )
                po_file = po_manager.po_file

                seen_new: Dict[str, int] = {}
                for entry in po_file:
                    if entry.obsolete or not entry.msgid:
                        continue
                    n = seen_new.get(entry.msgid, 0)
                    seen_new[entry.msgid] = n + 1
                    carried = preserved.get((entry.msgid, n))
                    if not carried:
                        continue
                    for flag in carried["flags"]:
                        if flag not in entry.flags:
                            entry.flags.append(flag)
                    if carried["tcomment"]:
                        entry.tcomment = carried["tcomment"]

            self._save_processed_document(processed_content, target_path)

            receipt = _build_receipt(
                model=self.model,
                target_lang=self.target_lang,
                source_path=str(source_path),
                target_path=str(target_path),
                po_path=str(po_path),
                usage=usage,
                duration_seconds=time.monotonic() - start,
            )

            result = ProcessResult(
                source_path=str(source_path),
                target_path=str(target_path),
                po_path=str(po_path),
                blocks_count=len(blocks),
                coverage=Coverage(**coverage_dict),
                translation_stats=BatchStats(**translation_stats),
                receipt=receipt,
            )

            po_manager.save_po(po_file, po_path)
            po_saved = True
            return result
        except BaseException as exc:
            # Only annotate when LLM calls were actually billed — a
            # zero-usage receipt on a pre-API failure (missing source,
            # bad PO) would mislead operators into thinking a call was
            # made.  This keeps the CLI fallback path meaningful.
            if usage.api_calls > 0:
                partial = _build_receipt(
                    model=self.model,
                    target_lang=self.target_lang,
                    source_path=str(source_path),
                    target_path=str(target_path),
                    po_path=str(po_path),
                    usage=usage,
                    duration_seconds=time.monotonic() - start,
                )
                try:
                    exc.partial_receipt = partial  # type: ignore[attr-defined]
                except (AttributeError, TypeError):
                    # Immutable / slot-based exceptions can't carry the
                    # attribute; propagate without it rather than masking
                    # the original failure with a secondary error.
                    pass
            # Preserve translations already written into the PO on
            # failure — unless save_po itself was the failure, in which
            # case retrying would likely raise the same error.
            if po_file is not None and not po_saved:
                try:
                    po_manager.save_po(po_file, po_path)
                except Exception:
                    logger.exception(
                        "PO save failed during error handling for %s", po_path
                    )
            raise
        finally:
            # Match every ``document_start`` with a ``document_end``
            # AFTER rebuild / inplace / save have all either committed
            # or raised.  UIs can then close their bars without
            # mis-marking failed runs as completed.
            if document_started:
                self._emit_progress(
                    kind="document_end", path=source_path_str
                )
            # ``refine_first=True`` on a single-call path plants a
            # usage accumulator on ``self._tls`` so the translate-pass
            # receipt picks up the refine-pass tokens.  Clear it now
            # — otherwise a subsequent ``process_document`` on the
            # same thread would start from this run's counts and
            # inflate every later receipt.  The directory path
            # owns its own accumulator via ``_process_one`` and is
            # unaffected because it never enters this branch.
            if tls_usage_installed_here:
                self._tls.usage = None
            # Same hygiene for the refine-first carryover slot: if
            # ``_process_entries_*`` never ran (e.g. parse failed before
            # reference-pool seeding), the stashed carryover would leak
            # into the next call on this thread.
            if getattr(self._tls, "refine_first_carryover", None):
                self._tls.refine_first_carryover = None

    def _translate_path_segments(
        self,
        *,
        source_dir: Path,
        matched_files: List[Path],
        paths_po_path: Path,
        usage: _UsageAccumulator,
        previous_map: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """Translate the distinct path segments across ``matched_files``
        and return a source → target relative-path mapping.

        Scope:

        - Each segment (directory name or markdown stem) is catalogued in
          ``_paths.po`` as a single entry keyed by ``msgctxt="path::segment::<raw>"``.
          File extensions are preserved verbatim and never translated.
        - Translations flow through :meth:`_call_llm` (same prompt pipeline
          as content blocks), so caching, glossary, validation, and
          ``usage`` accounting all behave the way the rest of the directory
          run does.
        - Outputs are piped through :func:`slugify_path_segment` to
          normalise whitespace / strip filesystem-unsafe characters, and a
          per-directory uniqueness pass appends ``-2``, ``-3``, … when two
          source segments collapse to the same slug inside the same
          directory.  Sibling directories are handled independently —
          ``a/x`` and ``b/x`` never contend for the same slot.
        - Missing / empty / failed translations fall back to the sanitized
          source segment so every source path still maps to SOMETHING.

        Returns a dict keyed by the source file's POSIX relative path
        (``"guide/intro.md"``) with the translated relative path as value.
        The caller composes the on-disk target path by joining this value
        under ``target_dir``.
        """
        unique_segments: List[str] = []
        seen_segments: set = set()

        # Collect segments in deterministic first-appearance order so the
        # resulting _paths.po diff is stable across runs with unchanged
        # input.  Dotfile-only segments (``""``, ``"."``, ``".."``) are
        # preserved verbatim and never translated — they're filesystem
        # navigation tokens, not content.
        def _is_translatable(seg: str) -> bool:
            return bool(seg) and seg not in {".", ".."} and not seg.startswith(".")

        for source_file in matched_files:
            rel = source_file.relative_to(source_dir)
            parent_parts = list(rel.parts[:-1])
            segments_in_file = parent_parts + [rel.stem]
            for seg in segments_in_file:
                if not _is_translatable(seg):
                    continue
                if seg in seen_segments:
                    continue
                seen_segments.add(seg)
                unique_segments.append(seg)

        po_manager = POManager(skip_types=[])
        po_file = po_manager.load_or_create_po(
            paths_po_path, target_lang=self.target_lang
        )

        def _ctx_for(seg: str) -> str:
            return f"path::segment::{seg}"

        # Add / update entries for segments seen in THIS run.  Entries
        # whose segments aren't matched by the current glob are left
        # untouched — a narrower-glob retry must not flush cached
        # translations (or operator edits) for out-of-scope paths.
        # That mirrors the partial-rerun behaviour already guaranteed
        # for ``path_map.json``: stale PO rows accumulate but never get
        # silently deleted.  Operators who genuinely want to prune can
        # edit ``_paths.po`` by hand, same as the per-document PO flow.
        entry_by_ctx = {e.msgctxt: e for e in po_file if e.msgctxt}
        for seg in unique_segments:
            ctx = _ctx_for(seg)
            entry = entry_by_ctx.get(ctx)
            if entry is None:
                po_file.append(polib.POEntry(msgctxt=ctx, msgid=seg, msgstr=""))
                continue
            if entry.msgid != seg:
                entry.msgid = seg
                if "fuzzy" not in entry.flags:
                    entry.flags.append("fuzzy")

        # Scope translation to segments actually used by THIS run so a
        # narrower-glob retry never spends tokens on (or silently
        # re-translates) an out-of-scope fuzzy entry that an earlier
        # full run left in ``_paths.po``.  Operators who want to
        # re-translate preserved entries can either widen the glob or
        # edit ``_paths.po`` and clear ``msgstr`` themselves.
        current_segment_set = set(unique_segments)
        pending = [
            e
            for e in po_file
            if not e.obsolete
            and e.msgid in current_segment_set
            and ((not e.msgstr) or "fuzzy" in e.flags)
        ]

        for entry in pending:
            try:
                translated = self._call_llm(entry.msgid, usage=usage)
            except Exception as exc:
                logger.warning(
                    "Path segment translation failed for %r: %s", entry.msgid, exc
                )
                continue
            if translated is None:
                continue
            entry.msgstr = translated.strip()
            po_manager.mark_entry_processed(entry)

        # Save eagerly so a downstream crash during file-level translation
        # doesn't discard the segment translations we already billed for.
        po_manager.save_po(po_file, paths_po_path)

        # Raw source segment → sanitized translated slug lookup.  Empty /
        # missing translations fall back to a sanitized source segment so
        # every path still resolves; filesystem-unsafe characters in the
        # raw source are also stripped so the fallback is always writable.
        def _safe_slug_for(raw: str) -> str:
            """Return a guaranteed-filesystem-safe slug for ``raw``.

            Falls back to a deterministic ``segment-<sha1>`` form when
            :func:`slugify_path_segment` strips every character — a
            source segment legally composed of characters that are
            reserved on Windows (``?``, ``*``, ``<``, ``>`` …) would
            otherwise leak straight through as an unwritable path on
            that platform.  The hash is stable across runs so the
            published mapping and on-disk filename stay reproducible.
            """
            slug = slugify_path_segment(raw)
            if slug:
                return slug
            digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
            return f"segment-{digest}"

        segment_slug: Dict[str, str] = {}
        for entry in po_file:
            if entry.obsolete or not entry.msgctxt:
                continue
            raw_msgstr = entry.msgstr or ""
            slug = slugify_path_segment(raw_msgstr)
            if not slug:
                slug = _safe_slug_for(entry.msgid)
            segment_slug[entry.msgid] = slug

        # Per-directory uniqueness: two segments whose TRANSLATED parent
        # is the same (i.e. they land in the same on-disk directory) and
        # whose slugified translations collide get ``-N`` disambiguators.
        # Keying by the TRANSLATED parent — not the raw source parent —
        # matches filesystem semantics: two distinct source dirs that
        # happen to translate to the same slug already get disambiguated
        # at the outer level, so their children share one namespace.
        # Raw-keyed buckets would treat them as separate namespaces even
        # after the outer-level disambig converges the paths, letting
        # descendants shadow each other on disk.
        assigned: Dict[Tuple[Tuple[str, ...], str], str] = {}
        used_at_level: Dict[Tuple[str, ...], set] = {}

        # Pre-seed ``assigned`` AND ``used_at_level`` for out-of-scope
        # preserved entries whose translation is UNCHANGED from the
        # current PO state.  The cache hit makes in-scope resolutions
        # for the same raw segment return the preserved slug (so
        # directories get shared instead of spuriously disambiguated
        # with ``-2``).  For entries whose translation has CHANGED
        # between runs (operator edited ``_paths.po``; LLM output
        # differs on a retranslate), do NOT pre-seed — letting the
        # fresh resolution compute from ``segment_slug`` is what makes
        # rename propagate to in-scope files on partial reruns.
        # Out-of-scope preserved files stay at their old on-disk path;
        # the operator needs a full-tree rerun to migrate them too.
        current_source_set = {
            sf.relative_to(source_dir).as_posix() for sf in matched_files
        }
        for src_rel, old_tgt_rel in (previous_map or {}).items():
            if src_rel in current_source_set:
                continue
            try:
                src_parts = Path(src_rel).parts
                tgt_parts = Path(old_tgt_rel).parts
            except (TypeError, ValueError):
                continue
            if not src_parts or not tgt_parts:
                continue
            if len(src_parts) != len(tgt_parts):
                # Length mismatch (e.g. a prior run with a different
                # flag set) — the segment-to-segment alignment would
                # be ambiguous, so skip this entry rather than risk
                # seeding an incorrect mapping.
                continue
            src_last_stem = Path(src_parts[-1]).stem
            src_last_suffix = Path(src_parts[-1]).suffix
            tgt_last_stem = Path(tgt_parts[-1]).stem
            tgt_last_suffix = Path(tgt_parts[-1]).suffix
            if src_last_suffix != tgt_last_suffix:
                continue
            src_walk = list(src_parts[:-1]) + [src_last_stem]
            tgt_walk = list(tgt_parts[:-1]) + [tgt_last_stem]
            translated_acc: Tuple[str, ...] = ()
            for raw, translated in zip(src_walk, tgt_walk):
                if _is_translatable(raw):
                    current_slug = segment_slug.get(raw)
                    if current_slug is None:
                        current_slug = _safe_slug_for(raw)
                    if current_slug == translated:
                        # Unchanged: cache the mapping and reserve the
                        # slot so in-scope lookups share the directory.
                        key = (translated_acc, raw)
                        assigned.setdefault(key, translated)
                        used_at_level.setdefault(translated_acc, set()).add(
                            translated.casefold()
                        )
                    # Renamed: skip.  In-scope resolution uses the new
                    # slug freely; the preserved file at the OLD slug
                    # lives on a disjoint path and cannot collide.
                translated_acc = translated_acc + (translated,)

        def _resolve(translated_parent: Tuple[str, ...], raw_seg: str) -> str:
            if not _is_translatable(raw_seg):
                return raw_seg
            key = (translated_parent, raw_seg)
            cached = assigned.get(key)
            if cached is not None:
                return cached
            base = segment_slug.get(raw_seg) or _safe_slug_for(raw_seg)
            # Collision detection uses ``str.casefold`` so sibling
            # translations like ``Guide`` and ``guide`` — distinct under
            # Linux ext4 but ALIASES on default Windows NTFS and
            # macOS APFS (case-insensitive by default) — still receive
            # ``-2`` disambiguation.  Without casefolding those siblings
            # clobber each other on the majority of operator platforms
            # even though the in-memory string set accepts both.
            used = used_at_level.setdefault(translated_parent, set())
            candidate = base
            n = 2
            while candidate.casefold() in used:
                candidate = f"{base}-{n}"
                n += 1
            used.add(candidate.casefold())
            assigned[key] = candidate
            return candidate

        path_map: Dict[str, str] = {}
        # Deterministic assignment order: sort by source posix relpath so
        # disambiguation collisions always resolve the same way across
        # runs.  Without this, two bare-``--translate-paths`` invocations
        # could disagree on which colliding sibling gets the base slug.
        for source_file in sorted(
            matched_files, key=lambda p: p.relative_to(source_dir).as_posix()
        ):
            rel = source_file.relative_to(source_dir)
            parent_parts = tuple(rel.parts[:-1])
            stem = rel.stem
            suffix = rel.suffix

            translated_parents: List[str] = []
            translated_accumulated: Tuple[str, ...] = ()
            for raw in parent_parts:
                translated = _resolve(translated_accumulated, raw)
                translated_parents.append(translated)
                translated_accumulated = translated_accumulated + (translated,)
            translated_stem = _resolve(translated_accumulated, stem)
            translated_rel = Path(*translated_parents, translated_stem + suffix)
            path_map[rel.as_posix()] = translated_rel.as_posix()

        return path_map

    def process_directory(
        self,
        source_dir: Path,
        target_dir: Path,
        po_dir: Path | None = None,
        glob: str = "**/*.md",
        inplace: bool = False,
        max_workers: int = 4,
        *,
        refined_dir: Path | None = None,
        refine_first: bool = False,
        refine_lang: Optional[str] = None,
        refined_po_dir: Path | None = None,
        translate_paths: bool = False,
    ) -> DirectoryResult:
        """
        Process all markdown files in a directory tree.

        Refine mode / ``refine_first`` lift the single-file semantics: the
        per-file refined output lands under ``refined_dir`` preserving the
        relative path of each source file.  When ``refine_first=True``,
        ``refined_dir`` is required so each file's refined intermediate
        has a deterministic location.

        ``translate_paths`` (opt-in, default ``False``) treats filesystem
        path segments (directory names and markdown file stems) as their
        own "block type".  A dedicated ``_paths.po`` catalog is created
        beside the effective PO directory (or under ``target_dir`` when
        ``po_dir`` is omitted) and drives the per-segment translations; a
        ``path_map.json`` is emitted alongside the translated tree so
        downstream link-rewriters / sitemaps / CI can resolve the
        source → target path pairing without re-running the translator.
        Link text and URLs inside translated Markdown are deliberately
        NOT rewritten — that's a separate cross-reference problem, and
        scoping it here would invalidate every translated document's
        internal anchors.
        """
        import concurrent.futures

        # The DeprecationWarning for ``inplace`` is raised by
        # ``process_document`` per file so the deduplication rule applied
        # by the ``warnings`` module is driven by a single call-site and
        # emissions are consistent whether the caller used the single-file
        # or directory API.
        #
        # Mirror the refine preconditions that ``process_document``
        # enforces.  Without these, an invalid invocation (e.g.
        # ``mode="refine"`` with ``inplace=True``) would only surface
        # inside each worker, ``_process_one`` would catch the
        # ``ValueError`` into the ``files_failed`` tally, and the caller
        # would get a partial DirectoryResult instead of a clean usage
        # error.  Fail fast so the contract is consistent whether the
        # caller uses the single-file or directory API.
        if self.mode == "refine" and inplace:
            raise ValueError(
                "inplace=True is incompatible with mode='refine'. "
                "Refine never overwrites the source or its PO msgid; "
                "use refined_dir instead."
            )
        if refine_first and self.mode != "translate":
            raise ValueError(
                "refine_first=True only applies in mode='translate'."
            )
        if refine_first and refined_dir is None:
            raise ValueError(
                "refine_first=True requires refined_dir for process_directory."
            )
        if refine_first and not refine_lang:
            raise ValueError(
                "refine_first=True requires refine_lang "
                "(the BCP 47 locale of the SOURCE documents)."
            )
        source_dir = Path(source_dir)
        target_dir = Path(target_dir)
        refined_dir_path = Path(refined_dir) if refined_dir is not None else None
        refined_po_dir_path = (
            Path(refined_po_dir) if refined_po_dir is not None else None
        )

        # Path-collision guards mirror the per-file checks that
        # ``process_document`` performs so deterministic bad inputs
        # surface as a single ``ValueError`` up front — rather than
        # every worker raising inside ``_process_one`` and degrading
        # the run to a partial ``DirectoryResult`` with files_failed
        # entries that hide the real configuration bug.
        def _resolve_dir(p: Path) -> Path:
            try:
                return p.resolve(strict=False)
            except OSError:
                return p.absolute()

        src_res = _resolve_dir(source_dir)
        tgt_res = _resolve_dir(target_dir)
        po_res = _resolve_dir(Path(po_dir)) if po_dir is not None else None
        refined_res = (
            _resolve_dir(refined_dir_path)
            if refined_dir_path is not None
            else None
        )
        refined_po_res = (
            _resolve_dir(refined_po_dir_path)
            if refined_po_dir_path is not None
            else None
        )

        if self.mode == "refine":
            effective_out = refined_res if refined_res is not None else tgt_res
            if src_res == effective_out:
                raise ValueError(
                    "Refine directory run forbids writing refined output "
                    f"on top of the source tree ({src_res})."
                )

        if refine_first:
            # All four artefact directories (source, refined, target,
            # refine PO, translate PO) must be distinct — otherwise a
            # single run can silently clobber or reopen its own prior
            # outputs.  ``refined_res is not None`` is guaranteed by
            # the earlier ``refine_first requires refined_dir`` guard.
            if refined_res == tgt_res:
                raise ValueError(
                    "refine_first=True requires distinct refined_dir "
                    "and target_dir; sharing a directory would let the "
                    "translate pass reopen the refine-pass output."
                )
            if src_res == refined_res:
                raise ValueError(
                    "refine_first=True requires refined_dir to differ "
                    "from source_dir; refine would overwrite the source."
                )
            if src_res == tgt_res:
                raise ValueError(
                    "refine_first=True requires target_dir to differ "
                    "from source_dir; translate would overwrite the source."
                )
            if (
                po_res is not None
                and refined_po_res is not None
                and po_res == refined_po_res
            ):
                raise ValueError(
                    "refine_first=True requires distinct PO directories "
                    "for the refine and translate passes. A shared PO "
                    "dir means translate would see every refine entry "
                    "as already processed and skip translation."
                )

        matched_files = sorted(source_dir.glob(glob))
        start = time.monotonic()

        # Directory where path-translation catalog files (_paths.po,
        # path_map.json) live.  Refine mode with ``refined_dir`` routes
        # the processed tree to ``refined_dir``, so anchor the path-map
        # there too; everything else lands under ``target_dir``.
        path_artefacts_dir = (
            refined_dir_path
            if self.mode == "refine" and refined_dir_path is not None
            else target_dir
        )

        # Opt-in path translation: build a segment catalog in ``_paths.po``
        # and compose a source → target relative-path map BEFORE the thread
        # pool starts, so each worker's ``_process_one`` knows the
        # translated target for its file.  Tokens billed by the per-segment
        # LLM calls are recorded into ``path_usage`` and merged into the
        # directory-level receipt alongside per-worker totals.
        path_usage = _UsageAccumulator()
        path_map_relative: Dict[str, str] = {}
        paths_po_path: Optional[Path] = None
        path_map_json_path: Optional[Path] = None
        previous_map: Dict[str, str] = {}
        if translate_paths:
            paths_po_dir = po_dir if po_dir is not None else path_artefacts_dir
            paths_po_path = Path(paths_po_dir) / "_paths.po"
            path_map_json_path = Path(path_artefacts_dir) / "path_map.json"
            # Load the previous run's map NOW (before the segment
            # catalog runs) so ``_translate_path_segments`` can pre-seed
            # its disambig namespace with preserved out-of-scope
            # entries.  A narrower-glob retry otherwise reshuffles
            # prior ``-N`` suffixes and overwrites outputs for sources
            # it didn't touch.
            if path_map_json_path.exists():
                try:
                    raw_prev = json.loads(
                        path_map_json_path.read_text(encoding="utf-8")
                    )
                    if isinstance(raw_prev, dict):
                        previous_map = {
                            k: v
                            for k, v in raw_prev.items()
                            if isinstance(k, str) and isinstance(v, str)
                        }
                except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                    # A corrupt prior map shouldn't abort the run; we
                    # simply skip preserved-slug seeding and cleanup.
                    previous_map = {}
            # Synthesize implicit mirror-path entries for source files
            # whose source-relative target still sits in the EFFECTIVE
            # output tree (``refined_dir`` under refine mode with a
            # separate refined_dir, else ``target_dir``) without a
            # matching ``path_map.json`` row.  That happens on the
            # FIRST ``--translate-paths`` run against a tree that was
            # previously populated by the default source-mirror path
            # (or by an earlier run without the flag): the new
            # localized outputs would otherwise land beside the legacy
            # mirror files and static-site deploys would publish both
            # copies.  Probing the wrong tree (``target_dir`` in refine
            # mode with a separate refined_dir) would also silently
            # miss the legacy files.  Treating the mirror as
            # ``src_rel -> src_rel`` in ``previous_map`` routes it
            # through the same cleanup pass that handles any other
            # translation change, so the legacy file is removed exactly
            # when the new translated target differs from the
            # source-relative path.
            try:
                artefacts_dir_abs = Path(path_artefacts_dir).resolve(
                    strict=False
                )
            except OSError:
                artefacts_dir_abs = Path(path_artefacts_dir).absolute()
            for source_file in matched_files:
                src_rel = source_file.relative_to(source_dir).as_posix()
                if src_rel in previous_map:
                    continue
                try:
                    mirror_abs = (Path(path_artefacts_dir) / src_rel).resolve(
                        strict=False
                    )
                except OSError:
                    continue
                try:
                    mirror_abs.relative_to(artefacts_dir_abs)
                except ValueError:
                    continue
                if mirror_abs.is_file():
                    previous_map[src_rel] = src_rel
            # Filename collision guard: ``_paths.po`` is fixed by the
            # feature contract, so a source file whose per-document PO
            # would resolve to the same path corrupts both the segment
            # catalog AND the document PO on every run (each overwrites
            # the other's state).  Detect up front and fail with a
            # specific usage error — a deep inconsistency would be far
            # harder to diagnose from symptoms alone.
            if matched_files:
                try:
                    paths_po_abs = paths_po_path.resolve(strict=False)
                except OSError:
                    paths_po_abs = paths_po_path.absolute()
                # Match the per-document PO routing in ``_process_one``:
                # when ``po_dir`` is omitted AND ``translate_paths`` is
                # on, the default PO lives under the EFFECTIVE output
                # tree (``path_artefacts_dir``), not ``target_dir``.
                # Checking the wrong root here would let a source named
                # ``_paths.md`` slip past the guard in refine-mode runs
                # and silently corrupt both the segment catalog and the
                # per-file PO on every invocation.
                per_file_po_root = (
                    Path(po_dir) if po_dir is not None else Path(path_artefacts_dir)
                )
                for source_file in matched_files:
                    rel = source_file.relative_to(source_dir)
                    candidate = per_file_po_root / rel.with_suffix(".po")
                    try:
                        cand_abs = candidate.resolve(strict=False)
                    except OSError:
                        cand_abs = candidate.absolute()
                    if cand_abs == paths_po_abs:
                        raise ValueError(
                            "translate_paths=True reserves the filename "
                            "'_paths.po' for the segment catalog, which "
                            f"collides with the per-document PO for "
                            f"{source_file!s}. Rename the source file (e.g. "
                            "'paths.md' or 'docs-paths.md') or pass an "
                            "explicit --po-dir that does not contain a "
                            "source document mapped to '_paths.po'."
                        )
                path_map_relative = self._translate_path_segments(
                    source_dir=source_dir,
                    matched_files=matched_files,
                    paths_po_path=paths_po_path,
                    usage=path_usage,
                    previous_map=previous_map,
                )

        results: List[Any] = []
        files_processed = 0
        files_failed = 0
        files_skipped = 0
        # Hold each worker's accumulator so a mid-run exception still
        # contributes its billed tokens to the directory-level receipt.
        worker_usages: List[_UsageAccumulator] = []
        # Track which source files actually produced on-disk output in
        # this run.  ``--translate-paths`` uses this to (a) keep
        # path_map.json honest — advertising a target path is only safe
        # when the worker actually wrote that file — and (b) gate the
        # stale-file cleanup so a transient per-file failure doesn't
        # delete the prior good output when the replacement was never
        # produced.
        successful_source_posix: set = set()

        self._emit_progress(
            kind="directory_start",
            path=str(source_dir),
            total=len(matched_files),
        )

        def _process_one(source_file: Path):
            relative_path = source_file.relative_to(source_dir)
            # When ``translate_paths`` is active, the worker writes the
            # translated markdown to the TRANSLATED target path so
            # filesystem layout matches the localized filenames.  PO
            # files remain keyed on the SOURCE relative path so
            # incremental re-runs still find their previously-processed
            # entries; renaming the output file doesn't invalidate the
            # translation memory.
            translated_relative = Path(
                path_map_relative.get(relative_path.as_posix(), relative_path.as_posix())
            )
            target_path = target_dir / translated_relative
            if po_dir is not None:
                po_path_file: Optional[Path] = (
                    po_dir / relative_path.with_suffix(".po")
                )
            elif translate_paths:
                # When ``po_dir`` is omitted, ``process_document`` would
                # default the per-file PO to ``target_path.with_suffix(".po")``
                # — but with ``translate_paths`` active ``target_path`` has
                # already been localized, so the PO moves whenever a path
                # translation changes (e.g. the operator edits ``_paths.po``).
                # That breaks the feature's advertised source-relative PO
                # stability for the default layout.  Pin the PO to the
                # source-relative location under the EFFECTIVE output
                # tree (``refined_dir`` in refine mode, else
                # ``target_dir``) so incremental re-runs keep hitting
                # the same cache even when localized filenames drift —
                # and so refine runs don't write .po files into the
                # source tree when the caller has pointed ``target_dir``
                # at it (refine mode permits that alias because refine
                # output goes to ``refined_dir``, not ``target_dir``).
                po_path_file = (
                    path_artefacts_dir / relative_path.with_suffix(".po")
                )
            else:
                po_path_file = None
            # ``process_document`` in refine mode REWRITES its
            # ``target_path`` kwarg with ``refined_path`` before writing
            # the output, so the refined Markdown actually lands at
            # ``refined_path_file``.  Applying the translated relative
            # path here (not the raw one) keeps the refined tree layout
            # consistent with ``path_map.json`` — otherwise the map
            # advertises localized paths that the refine-mode writer
            # never creates and every downstream consumer of the map
            # breaks under ``mode='refine'`` with ``refined_dir``.
            refined_path_file = (
                refined_dir_path / translated_relative
                if refined_dir_path is not None
                else None
            )
            refined_po_path_file = (
                refined_po_dir_path / relative_path.with_suffix(".po")
                if refined_po_dir_path is not None
                else None
            )
            u = _UsageAccumulator()
            # Plant ``u`` on the processor's thread-local so
            # ``process_document`` records into it — even when a subclass /
            # test monkey-patches the method signature.  Clearing the slot
            # afterwards prevents one worker's accumulator leaking into the
            # next task that lands on this thread.
            self._tls.usage = u
            self._emit_progress(kind="file_start", path=str(source_file))
            try:
                # Only forward refine-specific kwargs when a refine pathway
                # is actually active.  Test suites that monkey-patch
                # ``process_document`` with the pre-T-7 signature
                # (``source, target, po, inplace``) keep working when
                # callers don't opt into refine mode.
                extra: Dict[str, Any] = {}
                if refined_path_file is not None:
                    extra["refined_path"] = refined_path_file
                if refine_first:
                    extra["refine_first"] = True
                if refine_lang is not None:
                    extra["refine_lang"] = refine_lang
                if refined_po_path_file is not None:
                    extra["refined_po_path"] = refined_po_path_file
                result = self.process_document(
                    source_file,
                    target_path,
                    po_path_file,
                    inplace=inplace,
                    **extra,
                )
                return result, u, None
            except Exception as exc:
                return None, u, exc
            finally:
                self._tls.usage = None

        try:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                future_to_file = {
                    executor.submit(_process_one, sf): sf for sf in matched_files
                }
                for future in concurrent.futures.as_completed(future_to_file):
                    source_file = future_to_file[future]
                    file_status = "failed"
                    try:
                        result, worker_usage, worker_error = future.result()
                    except Exception:
                        # _process_one swallows task exceptions into the tuple;
                        # reaching here means the executor itself blew up.
                        logger.exception("Worker crashed for %s", source_file)
                        files_failed += 1
                        results.append({"source_path": str(source_file), "error": True})
                        self._emit_progress(
                            kind="file_end",
                            path=str(source_file),
                            status=file_status,
                        )
                        continue

                    worker_usages.append(worker_usage)

                    if worker_error is not None:
                        logger.exception(
                            "Failed to process %s", source_file, exc_info=worker_error
                        )
                        files_failed += 1
                        results.append({"source_path": str(source_file), "error": True})
                        self._emit_progress(
                            kind="file_end",
                            path=str(source_file),
                            status=file_status,
                        )
                        continue

                    results.append(result)
                    # Record that this source file produced on-disk
                    # output — even when entry-level failures flip
                    # ``files_failed``, the translated markdown was
                    # written by ``_save_processed_document`` before
                    # return.  Only hard worker errors (caught above)
                    # leave the target missing.
                    try:
                        successful_source_posix.add(
                            source_file.relative_to(source_dir).as_posix()
                        )
                    except ValueError:
                        pass

                    stats = result.translation_stats
                    if stats.failed > 0 or stats.validation_failed > 0:
                        # Entry-level failures still count as a file-level
                        # failure so automation can detect partial errors
                        # via ``files_failed`` / the CLI exit code.
                        files_failed += 1
                        file_status = "failed"
                    elif stats.processed == 0:
                        files_skipped += 1
                        file_status = "skipped"
                    else:
                        files_processed += 1
                        file_status = "processed"
                    self._emit_progress(
                        kind="file_end",
                        path=str(source_file),
                        status=file_status,
                    )
        finally:
            self._emit_progress(
                kind="directory_end", path=str(source_dir)
            )

        duration = time.monotonic() - start
        dir_usage = _UsageAccumulator()
        for u in worker_usages:
            dir_usage.input_tokens += u.input_tokens
            dir_usage.output_tokens += u.output_tokens
            dir_usage.api_calls += u.api_calls
        # Fold the path-translation billing into the directory receipt so
        # operators see the full cost of the run — skipping it would
        # under-report whenever ``--translate-paths`` issued API calls.
        dir_usage.input_tokens += path_usage.input_tokens
        dir_usage.output_tokens += path_usage.output_tokens
        dir_usage.api_calls += path_usage.api_calls

        # Emit path_map.json after workers finish so the file is only
        # written on directory completion, alongside the (possibly partial)
        # translated tree.  Writing it before the workers would leave stale
        # mappings on disk if a worker failure aborts the run.
        if translate_paths and path_map_json_path is not None:
            # ``previous_map`` was loaded BEFORE path-segment translation
            # so the disambig pass could pre-seed out-of-scope
            # assignments.  Reuse the same snapshot here for cleanup /
            # merge — re-reading the file would spuriously diff against
            # the empty map when the segment helper overwrote it, and
            # deleting the helper's already-loaded snapshot risks
            # drift between pre- and post-worker views.
            # Restrict the public mapping to sources whose worker
            # actually wrote a translated file.  Hard worker failures
            # leave the target missing on disk, and advertising a
            # non-existent path in path_map.json would break every
            # downstream consumer (link rewriters, sitemap jobs) that
            # trusts the file as ground truth.  Preserve prior map
            # entries for sources that didn't run this time (narrower
            # glob, retry-specific invocation) — but only when the
            # source still exists on disk, so mappings for genuinely
            # deleted files don't linger forever — so repeated partial
            # runs converge on a complete map rather than shrinking.
            try:
                source_dir_resolved = Path(source_dir).resolve(strict=False)
            except OSError:
                source_dir_resolved = Path(source_dir).absolute()

            def _source_still_on_disk(src_rel: str) -> bool:
                # Defend against a corrupt / hand-edited prior map that
                # contains ``..`` or absolute paths: confine the file
                # existence probe to the current ``source_dir`` tree.
                if not src_rel:
                    return False
                try:
                    candidate = (Path(source_dir) / src_rel).resolve(
                        strict=False
                    )
                except OSError:
                    return False
                try:
                    candidate.relative_to(source_dir_resolved)
                except ValueError:
                    return False
                return candidate.is_file()

            published_map: Dict[str, str] = {}
            for src_rel, old_tgt in previous_map.items():
                # Start from the prior map so a narrower-glob rerun or
                # targeted retry does not shrink the published map.
                # Drop entries whose source file has been removed from
                # the tree — those are genuinely stale.
                if _source_still_on_disk(src_rel):
                    published_map[src_rel] = old_tgt
            for src_rel, tgt_rel in path_map_relative.items():
                if src_rel in successful_source_posix:
                    # Override with this run's successful update.
                    published_map[src_rel] = tgt_rel
            try:
                path_map_json_path.parent.mkdir(parents=True, exist_ok=True)
                path_map_json_path.write_text(
                    json.dumps(
                        published_map,
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            except OSError:
                logger.exception(
                    "Failed to write path_map.json at %s", path_map_json_path
                )
            # Delete stale translated outputs.  ``old_target`` only comes
            # off disk when the REPLACEMENT was actually written this
            # run (or the source file is gone from the tree entirely) —
            # otherwise a transient per-file failure after an operator
            # edits ``_paths.po`` would delete the last good translation
            # without producing a new one, which is permanent data loss
            # from a recoverable error.
            try:
                artefacts_resolved = Path(path_artefacts_dir).resolve(
                    strict=False
                )
            except OSError:
                artefacts_resolved = Path(path_artefacts_dir).absolute()
            current_targets_resolved: set = set()
            for src_rel_cur, new_rel in path_map_relative.items():
                if src_rel_cur not in successful_source_posix:
                    # The planned target was never written; keep it
                    # out of the "current" set so the cleanup below
                    # doesn't treat an unrelated-but-equal prior target
                    # as already claimed by this run.
                    continue
                try:
                    current_targets_resolved.add(
                        (Path(path_artefacts_dir) / new_rel).resolve(
                            strict=False
                        )
                    )
                except OSError:
                    continue
            for src_rel, old_tgt_rel in previous_map.items():
                if path_map_relative.get(src_rel) == old_tgt_rel:
                    continue
                planned_new = path_map_relative.get(src_rel)
                # Partial-rerun safety: a narrower glob or targeted
                # retry leaves sources outside scope missing from
                # ``path_map_relative``, but those files still exist
                # on disk and must NOT be deleted.  Only treat a
                # source as "removed" when the file is genuinely gone
                # from the source tree.
                source_removed = (
                    planned_new is None
                    and not _source_still_on_disk(src_rel)
                )
                replacement_ready = (
                    planned_new is not None
                    and src_rel in successful_source_posix
                )
                if not (source_removed or replacement_ready):
                    # Replacement run failed OR source outside this
                    # run's glob; keep the old file so deploys still
                    # have SOMETHING for this source.
                    continue
                try:
                    old_target_abs = (
                        Path(path_artefacts_dir) / old_tgt_rel
                    ).resolve(strict=False)
                except OSError:
                    continue
                try:
                    old_target_abs.relative_to(artefacts_resolved)
                except ValueError:
                    continue
                if old_target_abs in current_targets_resolved:
                    continue
                if not old_target_abs.is_file():
                    continue
                try:
                    old_target_abs.unlink()
                except OSError:
                    logger.exception(
                        "Failed to remove stale translated output %s",
                        old_target_abs,
                    )

        # In refine mode with an explicit ``refined_dir``, per-file output
        # is routed to ``refined_dir`` — not ``target_dir`` — so report
        # that as the effective output path on the aggregate result and
        # receipt.  Downstream tooling keys off these fields to locate
        # outputs and would otherwise point at the unused target_dir.
        effective_output_dir = (
            refined_dir_path
            if self.mode == "refine" and refined_dir_path is not None
            else target_dir
        )

        dir_receipt = _build_receipt(
            model=self.model,
            target_lang=self.target_lang,
            source_path=str(source_dir),
            target_path=str(effective_output_dir),
            po_path=str(po_dir) if po_dir is not None else None,
            usage=dir_usage,
            duration_seconds=duration,
        )

        return DirectoryResult(
            source_dir=str(source_dir),
            target_dir=str(effective_output_dir),
            po_dir=str(po_dir) if po_dir is not None else None,
            files_processed=files_processed,
            files_failed=files_failed,
            files_skipped=files_skipped,
            results=results,
            receipt=dir_receipt,
        )

    # ----- sequential path (v0.2 fallback when batch_size=0) -----

    def _process_entry(self, entry, reference_pairs=None, usage=None):
        try:
            processed = self._call_llm(
                entry.msgid, reference_pairs=reference_pairs, usage=usage
            )
            return (entry, processed, None)
        except Exception as e:
            return (entry, None, e)

    def _process_entries_sequential(
        self,
        po_file: polib.POFile,
        po_manager: POManager,
        inplace: bool = False,
        usage: Optional[_UsageAccumulator] = None,
        source_path: Optional[str] = None,
        pending: Optional[List[polib.POEntry]] = None,
        initial_skipped: int = 0,
    ) -> Dict[str, int]:
        pool = ReferencePool(max_results=self.max_reference_pairs)
        pool.seed_from_po(po_file)
        # Consume any refine-first carryover parked on TLS — these
        # are (source_msgid, translation) pairs from a pre-existing
        # translate PO whose entries ``sync_po`` just invalidated
        # because the refined intermediate changed the msgids.  They
        # still serve as useful few-shot references for tone and
        # terminology even though the keys don't align.  Clear the
        # slot so a later call on the same thread doesn't inherit
        # stale context.
        carryover = getattr(self._tls, "refine_first_carryover", None)
        if carryover:
            for src, tgt in carryover:
                pool.add(src, tgt)
            self._tls.refine_first_carryover = None

        stats: Dict[str, int] = {
            "processed": 0,
            "failed": 0,
            "skipped": initial_skipped,
        }

        if pending is None:
            pending, _skipped = self._collect_pending_entries(po_file)
            stats["skipped"] = _skipped

        total = len(pending)
        for i, entry in enumerate(pending, start=1):
            similar = pool.find_similar(entry.msgid)

            try:
                entry_obj, processed, error = self._process_entry(
                    entry, reference_pairs=similar or None, usage=usage
                )
                if error is not None:
                    logger.warning(
                        "Failed to translate entry %s: %s",
                        entry_obj.msgctxt, error,
                    )
                    stats["failed"] += 1
                    continue

                if processed is not None and processed.strip() == entry.msgid.strip():
                    # Code blocks legitimately round-trip unchanged — rule 3
                    # of the translation instruction tells the LLM to keep
                    # code as-is and only translate comments / user-facing
                    # strings — so output==source is the expected outcome,
                    # not a failure signal. Suppress the warning for the
                    # ``code`` block type only; real regressions in prose
                    # still surface.  Refine mode is same-language: a
                    # well-written paragraph may come back verbatim, so
                    # suppress the entire warning in that mode.
                    block_type = self._extract_block_type_from_msgctxt(entry.msgctxt)
                    if block_type != "code" and self.mode != "refine":
                        logger.warning(
                            "LLM returned untranslated output for entry %s",
                            entry.msgctxt,
                        )

                if not self._apply_validation(entry_obj, processed, inplace, pool):
                    stats["failed"] += 1
                    continue

                # Preserve the original source msgid for the reference pool;
                # the inplace assignment below would otherwise rewrite it.
                source_msgid = entry_obj.msgid
                entry_obj.msgstr = processed
                if inplace:
                    entry_obj.msgid = processed
                po_manager.mark_entry_processed(entry_obj)
                stats["processed"] += 1
                pool.add(source_msgid, processed)
            except Exception as exc:
                logger.warning(
                    "Unexpected error for entry %s: %s",
                    getattr(entry, "msgctxt", None), exc,
                )
                stats["failed"] += 1
            finally:
                self._emit_progress(
                    kind="document_progress",
                    path=source_path,
                    index=i,
                    total=total,
                )

        return stats

    # ----- batched path -----

    def _process_entries_batched(
        self,
        po_file: polib.POFile,
        po_manager: POManager,
        inplace: bool = False,
        usage: Optional[_UsageAccumulator] = None,
        source_path: Optional[str] = None,
        pending: Optional[List[polib.POEntry]] = None,
        initial_skipped: int = 0,
    ) -> Dict[str, int]:
        pool = ReferencePool(max_results=self.max_reference_pairs)
        pool.seed_from_po(po_file)
        # Consume any refine-first carryover parked on TLS — these
        # are (source_msgid, translation) pairs from a pre-existing
        # translate PO whose entries ``sync_po`` just invalidated
        # because the refined intermediate changed the msgids.  They
        # still serve as useful few-shot references for tone and
        # terminology even though the keys don't align.  Clear the
        # slot so a later call on the same thread doesn't inherit
        # stale context.
        carryover = getattr(self._tls, "refine_first_carryover", None)
        if carryover:
            for src, tgt in carryover:
                pool.add(src, tgt)
            self._tls.refine_first_carryover = None

        stats: Dict[str, int] = {
            "processed": 0,
            "failed": 0,
            "skipped": initial_skipped,
            "batched_calls": 0,
            "per_entry_calls": 0,
            "validated": 0,
            "validation_failed": 0,
        }

        if pending is None:
            pending, _skipped = self._collect_pending_entries(po_file)
            stats["skipped"] = _skipped

        if not pending:
            return stats

        groups = self._section_aware_groups(pending)
        total = len(groups)

        concurrency = max(1, int(self.batch_concurrency or 1))
        # Intra-file concurrency is experimental and only kicks in when it
        # can actually amortise: ``concurrency == 1`` is the deterministic
        # v0.4 path, and a single group has nothing to parallelise after
        # the seed batch. Fall through to the plain sequential loop for
        # both cases so that opt-in callers don't pay ThreadPoolExecutor
        # setup cost on documents that don't benefit.
        if concurrency <= 1 or total <= 1:
            for i, group in enumerate(groups, start=1):
                self._translate_group(
                    group, po_manager, pool, stats, inplace=inplace, usage=usage
                )
                self._emit_progress(
                    kind="document_progress",
                    path=source_path,
                    index=i,
                    total=total,
                )
            return stats

        self._run_groups_concurrent(
            groups,
            po_manager=po_manager,
            pool=pool,
            stats=stats,
            inplace=inplace,
            usage=usage,
            source_path=source_path,
            total=total,
            concurrency=concurrency,
        )
        return stats

    def _run_groups_concurrent(
        self,
        groups: List[List[polib.POEntry]],
        *,
        po_manager: POManager,
        pool: ReferencePool,
        stats: Dict[str, int],
        inplace: bool,
        usage: Optional[_UsageAccumulator],
        source_path: Optional[str],
        total: int,
        concurrency: int,
    ) -> None:
        """Run batches in parallel after one seed batch populates the pool.

        The first group is processed synchronously so its translations
        land in ``pool`` before any worker thread calls ``find_similar``.
        Without that seed step, a cold-start document with no prior PO
        would hand every worker an empty pool and defeat the whole point
        of intra-file batching (consistent terminology via few-shot).
        The remaining groups are submitted to a ``ThreadPoolExecutor`` of
        width ``concurrency``. Each worker accumulates into a LOCAL stats
        dict and usage accumulator; shared state (``stats``, ``usage``,
        and the reference ``pool``) is guarded by a single lock so the
        hot path — the LLM call inside ``BatchTranslator`` — stays
        lock-free. Progress events are emitted as futures complete; the
        ``index`` field therefore reflects completion order rather than
        source order, matching the behaviour already documented for
        ``process_directory`` (events emitted from worker threads).
        """
        import concurrent.futures

        # Seed: first group runs on the calling thread so its
        # translations populate ``pool`` before any worker thread calls
        # ``pool.find_similar``. Without this the first N workers would
        # race on an empty pool and lose the few-shot benefit the batch
        # path exists to provide.
        self._translate_group(
            groups[0], po_manager, pool, stats, inplace=inplace, usage=usage
        )
        self._emit_progress(
            kind="document_progress",
            path=source_path,
            index=1,
            total=total,
        )

        remaining = groups[1:]
        if not remaining:
            return

        shared_lock = threading.Lock()
        # Signal used by workers to bail out fast once a peer fails.
        # ``future.cancel()`` alone is racy: a thread may pick up the
        # next queued future in the tiny window between one worker's
        # exception and the main thread's ``as_completed`` response,
        # so the drained future still fires its LLM call and over-
        # bills the user.  Checking the event at worker entry turns
        # that window into a cheap no-op return.
        abort_event = threading.Event()

        def _merge_usage(local_usage: _UsageAccumulator) -> None:
            """Fold a worker's billed tokens into the shared accumulator.

            Called from a ``finally`` block inside ``_run_one`` so that
            an exception raised AFTER LLM calls were billed — for
            example from a user-supplied ``post_process`` hook or a
            validator helper that crashes mid-group — still
            contributes those tokens to ``partial_receipt``.  Merging
            only on the success path would silently drop billed usage
            exactly in the scenario the partial-receipt contract is
            meant to surface.
            """
            if usage is None:
                return
            with shared_lock:
                usage.input_tokens += local_usage.input_tokens
                usage.output_tokens += local_usage.output_tokens
                usage.api_calls += local_usage.api_calls

        def _run_one(group: List[polib.POEntry]) -> Dict[str, int]:
            # Local stats / usage so the worker can mutate them without a
            # lock; ``_translate_group`` only needs the shared lock when
            # it talks to the pool.  Seed the dict with every key the
            # shared tally tracks so the merge loop doesn't accidentally
            # invent keys that ``BatchStats(**stats)`` can't accept.
            local_stats: Dict[str, int] = {k: 0 for k in stats}
            local_usage = _UsageAccumulator()
            # A peer worker has already failed — skip the LLM call
            # entirely rather than burn more tokens on work that the
            # caller will discard.  Returning the zero-filled
            # ``local_stats`` keeps the merge loop in the main thread
            # valid if it somehow still consumes this future (it
            # won't: the main thread re-raises the first failure
            # before iterating the rest).
            if abort_event.is_set():
                return local_stats
            try:
                self._translate_group(
                    group,
                    po_manager,
                    pool,
                    local_stats,
                    inplace=inplace,
                    usage=local_usage,
                    pool_lock=shared_lock,
                )
            except BaseException:
                # Set the abort flag BEFORE the ``finally`` clause
                # propagates the exception so that any peer worker
                # which is currently in its own ``_run_one`` entry
                # check sees the signal; without that the peer could
                # still issue one more LLM call in the narrow window
                # between the failure and the main thread's
                # ``as_completed`` loop noticing it.
                abort_event.set()
                raise
            finally:
                _merge_usage(local_usage)
            return local_stats

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=concurrency
        ) as executor:
            futures = [executor.submit(_run_one, group) for group in remaining]
            # Emit ``document_progress`` with a monotonically-increasing
            # index so progress bars observe "N of total completed"
            # rather than the worker's source-order group ID.  Using
            # the source index here would bounce around (e.g. 1, 4, 2,
            # 3) as ``as_completed`` releases futures in completion
            # order — UIs that interpret ``index / total`` as a
            # fraction would flash forward and then regress, which is
            # exactly what the progress contract forbids.  ``total``
            # stays the group count so the final progress event still
            # lands at ``index == total``.
            completed_index = 1  # seed batch already counted
            try:
                for future in concurrent.futures.as_completed(futures):
                    # Let worker exceptions propagate: the caller
                    # already wraps the batched path in the same
                    # error annotation logic that covers the
                    # sequential loop, so a mid-run crash still
                    # attaches ``partial_receipt`` and saves whatever
                    # the PO managed to record before the failure.
                    # ``_run_one`` has already merged billed usage in
                    # its ``finally`` block so partial receipts stay
                    # correct.
                    local_stats = future.result()
                    with shared_lock:
                        for k, v in local_stats.items():
                            stats[k] = stats.get(k, 0) + v
                    completed_index += 1
                    self._emit_progress(
                        kind="document_progress",
                        path=source_path,
                        index=completed_index,
                        total=total,
                    )
            except BaseException:
                # First worker failure aborts the document.  Cancel
                # every queued future so the pool doesn't start new
                # batches after the caller has already committed to
                # failing — without this the ``with`` block's
                # ``shutdown(wait=True)`` would drain the queue, each
                # drained batch would issue its own LLM call, and
                # ``partial_receipt`` would over-bill users on
                # exactly the concurrent partial-failure runs the
                # receipt contract is meant to surface honestly.
                # ``cancel()`` is a no-op for futures already running
                # — those can't be interrupted mid-call, but their
                # billed tokens still merge via the worker's
                # ``finally`` block so the receipt stays accurate.
                for f in futures:
                    f.cancel()
                raise

    def _section_aware_groups(
        self, entries: List[polib.POEntry]
    ) -> List[List[polib.POEntry]]:
        """Partition ``entries`` into section-aligned groups.

        Respects ``batch_size`` / ``batch_max_chars`` as hard limits.  Crosses
        a top-level section boundary only when the current group already
        covers at least half the batch budget, so coherent sections stay
        together where practical.
        """
        if not entries:
            return []

        size_cap = max(1, self.batch_size)
        char_cap = max(1, self.batch_max_chars)
        soft_size = max(1, size_cap // 2)

        groups: List[List[polib.POEntry]] = []
        current: List[polib.POEntry] = []
        current_chars = 0
        current_section: Optional[str] = None

        for e in entries:
            section = self._top_level_section(e.msgctxt)
            size_would_exceed = len(current) + 1 > size_cap
            chars_would_exceed = current_chars + len(e.msgid) > char_cap
            section_changed = (
                current_section is not None
                and section != current_section
                and len(current) >= soft_size
            )

            if current and (size_would_exceed or chars_would_exceed or section_changed):
                groups.append(current)
                current = []
                current_chars = 0

            current.append(e)
            current_chars += len(e.msgid)
            current_section = section

        if current:
            groups.append(current)
        return groups

    @staticmethod
    def _top_level_section(msgctxt: Optional[str]) -> Optional[str]:
        """Return the first path segment from ``msgctxt`` (``a/b::para:0`` → ``a``)."""
        if not msgctxt:
            return None
        sep = msgctxt.find("::")
        prefix = msgctxt[:sep] if sep != -1 else msgctxt
        slash = prefix.find("/")
        return prefix[:slash] if slash != -1 else prefix

    def _translate_group(
        self,
        group: List[polib.POEntry],
        po_manager: POManager,
        pool: ReferencePool,
        stats: Dict[str, int],
        inplace: bool,
        usage: Optional[_UsageAccumulator] = None,
        *,
        pool_lock: Optional[threading.Lock] = None,
    ) -> None:
        entry_by_ctx: Dict[str, polib.POEntry] = {e.msgctxt: e for e in group}

        # Pool access (reads via ``find_similar`` inside
        # ``_collect_references`` and the per-entry fallback, writes via
        # ``pool.add`` below) is serialised with ``pool_lock`` when the
        # caller opted into intra-file concurrency.  The lock is narrow
        # on purpose: the expensive LLM call happens OUTSIDE it, so
        # parallel workers overlap API latency while the cheap
        # SequenceMatcher scan / list append stay ordered.  When
        # ``pool_lock`` is ``None`` (default / sequential path) the
        # pool is not shared and no synchronisation is needed.
        pool_cm: AbstractContextManager[Any] = (
            pool_lock if pool_lock is not None else nullcontext()
        )

        # Reference lookup and glossary matching operate on human-readable
        # source text — encoding would obscure the substrings they compare
        # against — so collect them from ``entry.msgid`` before encoding
        # swaps pattern matches for opaque tokens.
        raw_sources = [e.msgid for e in group]
        with pool_cm:
            references = self._collect_references(raw_sources, pool)
        glossary_block = self._collect_glossary_block(raw_sources)

        # Encode each item once; stash per-ctx maps so we can decode the
        # batched response and run the round-trip check per entry below.
        items: Dict[str, str] = {}
        mappings: Dict[str, Optional[PlaceholderMap]] = {}
        for e in group:
            encoded, mapping = self._encode_source(e.msgid)
            items[e.msgctxt] = encoded
            mappings[e.msgctxt] = mapping

        base_caller = self._make_batch_caller(references, glossary_block, usage=usage)

        # Count every real LLM call made from inside BatchTranslator —
        # including any internal partitioning and recursive bisection —
        # so ``ProcessResult.translation_stats.batched_calls`` reflects
        # true API usage, not just the number of section groups.
        call_count = 0

        def counting_caller(chunk: Dict[str, str]) -> str:
            nonlocal call_count
            call_count += 1
            return base_caller(chunk)

        translator = BatchTranslator(
            counting_caller,
            max_entries=self.batch_size,
            max_chars=self.batch_max_chars,
        )

        try:
            translated = translator.translate(items)
        except Exception as exc:
            logger.warning("Batch translator crashed: %s; falling back", exc)
            translated = {}
        finally:
            stats["batched_calls"] = stats.get("batched_calls", 0) + call_count

        for ctx, entry in entry_by_ctx.items():
            processed: Optional[str] = None
            mapping = mappings.get(ctx)
            if ctx in translated:
                raw = translated[ctx]
                # Run ``post_process`` BEFORE decoding so a hook cannot
                # silently rewrite a restored protected span — the stash
                # below must contain the exact text that will be stored.
                # Matches the ordering used in ``_call_llm``.
                if self._post_process:
                    raw = self._post_process(raw)
                self._tls.last_encoded_response = raw
                self._tls.last_placeholder_map = mapping
                processed = self._decode_translation(raw, mapping)
            else:
                stats["per_entry_calls"] += 1
                with pool_cm:
                    similar = pool.find_similar(entry.msgid)
                try:
                    processed = self._call_llm(
                        entry.msgid, reference_pairs=similar or None, usage=usage
                    )
                except Exception as exc:
                    logger.warning(
                        "Per-entry fallback failed for %s: %s", ctx, exc
                    )
                    stats["failed"] += 1
                    continue

            if processed is None:
                stats["failed"] += 1
                continue

            if processed.strip() == entry.msgid.strip():
                # See the sequential path for rationale: code blocks are
                # instructed to pass through unchanged so output==source
                # is not a failure; only warn for non-code block types so
                # prose regressions stay visible.  Refine mode is
                # same-language and a well-written paragraph can round-trip
                # verbatim, so the whole warning is suppressed there.
                block_type = self._extract_block_type_from_msgctxt(entry.msgctxt)
                if block_type != "code" and self.mode != "refine":
                    logger.warning(
                        "LLM returned untranslated output for entry %s", entry.msgctxt
                    )

            if not self._apply_validation(entry, processed, inplace, pool, stats):
                stats["failed"] += 1
                continue

            # Capture the pre-inplace msgid so the reference pool is keyed on
            # the original source text rather than the translation-against-
            # translation pair that ``inplace=True`` would otherwise produce.
            source_msgid = entry.msgid
            entry.msgstr = processed
            if inplace:
                entry.msgid = processed
            po_manager.mark_entry_processed(entry)
            stats["processed"] += 1
            with pool_cm:
                pool.add(source_msgid, processed)

    # ----- multi-target translation -----

    def process_document_multi(
        self,
        source_path: Path,
        target_langs: List[str],
        target_paths: Dict[str, Path],
        po_paths: Optional[Dict[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Translate one Markdown document into several target languages in
        a single batched LLM call per source group.

        Source-side decomposition (placeholder substitution, reference
        lookup, glossary matching) runs ONCE per block regardless of the
        number of target languages, so the input-token bill is amortised
        across every language while only the output tokens grow with
        ``len(target_langs)``.  The batched wire format returns
        ``{block_id: {lang: translation}}``; any missing per-lang entry
        forces the affected blocks into bisection / per-language
        single-target fallback, preserving the deterministic behaviour
        of the single-target batched path.

        The processor's own ``target_lang`` / ``mode`` are ignored here —
        multi-target is translate-only and the caller names every
        per-call locale explicitly.  ``inplace`` is NOT supported (the
        contract is one source -> many targets; overwriting the single
        source msgid with N different translations is undefined).  The
        refine-first composition is out of scope for this method.

        Args:
            source_path: Markdown source file.
            target_langs: BCP 47 locale list; order is preserved, and
                duplicate entries are removed with a warning.  Must be
                non-empty.
            target_paths: Per-language output markdown path.  Every lang
                in ``target_langs`` MUST have an entry.
            po_paths: Per-language PO path.  When omitted (or a lang is
                missing), defaults to ``target_paths[lang]`` with a
                ``.po`` suffix.  Callers who want a per-lang PO
                directory layout should pre-compute paths and pass them
                explicitly.

        Returns:
            A dict with:
                - ``by_lang``: ``Dict[str, ProcessResult]`` — one entry
                  per target language (PO path, target markdown path,
                  coverage, per-lang stats; ``receipt=None`` because
                  tokens are billed ONCE across all languages).
                - ``receipt``: a single :class:`Receipt` summarising the
                  multi-target run.  ``target_lang`` is a
                  comma-separated list so operators still see which
                  languages were in scope.
                - ``source_path``: str.
                - ``target_langs``: list of locales actually processed
                  (duplicates removed).

        The human-runnable "canonical-seeded" comparison (translate to
        one anchor lang, then reuse its output as few-shot context for
        other langs) is already expressible via the existing
        single-target :meth:`process_document` repeated per lang.  This
        method ships alongside for direct A/B comparison without any
        additional machinery — nothing in the call graph depends on
        live API access.
        """
        if self.mode != "translate":
            raise ValueError(
                "process_document_multi requires mode='translate'. "
                "Multi-target refine is not supported — refine is "
                "same-language by contract."
            )
        if not target_langs:
            raise ValueError("target_langs must not be empty.")
        # Placeholder-mode glossary bakes the target-language replacement
        # into the PlaceholderMap at encode time — the same decoded token
        # would then be restored to a single locale across every
        # per-lang output, silently injecting one language's glossary
        # forms into the others.  The multi-target flow's "shared
        # source-side decomposition" contract is incompatible with that
        # per-lang decoding requirement, so refuse up front and point
        # callers at the safe alternatives.  Guard against ANY
        # configured glossary source (inline dict OR glossary_file),
        # not just ``self._glossary`` — a locale-keyed file whose
        # entries happen not to resolve for the constructor-time
        # ``target_lang`` would silently leave ``self._glossary``
        # empty, so the narrower check would let the multi-target run
        # proceed with the glossary silently dropped instead of
        # failing fast.
        has_glossary_source = bool(self._glossary_inline) or (
            self._glossary_file is not None
        )
        if has_glossary_source and self.glossary_mode == "placeholder":
            raise ValueError(
                "process_document_multi does not support "
                "glossary_mode='placeholder'. The placeholder path bakes "
                "target-language replacements into the PlaceholderMap at "
                "encode time, which cannot fan out to multiple target "
                "languages without corrupting at least one. Either "
                "switch to glossary_mode='instruction' for the "
                "multi-target call, or run the single-target "
                "process_document once per language (the "
                "canonical-seeded baseline)."
            )

        if self.batch_concurrency and self.batch_concurrency > 1:
            # Multi-target does not yet honour ``batch_concurrency``:
            # the shared per-lang pools / stats counters would need the
            # same lock-protected merge logic as
            # :meth:`_run_groups_concurrent`, which is not yet wired
            # through ``_translate_group_multi``.  Log a loud warning
            # so callers who opted in for performance know the flag is
            # being ignored on this call, then fall through to the
            # sequential loop below.  A silent ignore would otherwise
            # mask throughput regressions relative to single-target runs.
            logger.warning(
                "batch_concurrency=%d is not yet honoured by "
                "process_document_multi; running groups sequentially. "
                "Run single-target process_document per lang to exploit "
                "intra-file concurrency.",
                self.batch_concurrency,
            )

        langs: List[str] = []
        seen_langs: set = set()
        for lang in target_langs:
            if not isinstance(lang, str) or not lang:
                raise ValueError(
                    "target_langs entries must be non-empty strings."
                )
            if lang in seen_langs:
                logger.warning(
                    "target_langs contains duplicate %r; ignoring.", lang
                )
                continue
            seen_langs.add(lang)
            langs.append(lang)

        missing_targets = [l for l in langs if l not in target_paths]
        if missing_targets:
            raise ValueError(
                "target_paths missing entries for: " + ", ".join(missing_targets)
            )

        resolved_target_paths: Dict[str, Path] = {
            lang: Path(target_paths[lang]) for lang in langs
        }
        resolved_po_paths: Dict[str, Path] = {}
        for lang in langs:
            p = (po_paths or {}).get(lang) if po_paths else None
            if p is None:
                p = resolved_target_paths[lang].with_suffix(".po")
            resolved_po_paths[lang] = Path(p)

        # Distinctness guard: two languages writing to the same target
        # or PO path would silently clobber each other on save.  Compare
        # resolved absolute paths so ``./a.md`` vs ``a.md`` spellings
        # trip the guard the same way the refine-first checks do.
        def _resolve(p: Path) -> Path:
            try:
                return p.resolve(strict=False)
            except OSError:
                return p.absolute()

        target_resolved: Dict[str, Path] = {
            lang: _resolve(p) for lang, p in resolved_target_paths.items()
        }
        po_resolved: Dict[str, Path] = {
            lang: _resolve(p) for lang, p in resolved_po_paths.items()
        }
        if len({str(p) for p in target_resolved.values()}) != len(langs):
            raise ValueError(
                "target_paths must resolve to distinct paths per language."
            )
        if len({str(p) for p in po_resolved.values()}) != len(langs):
            raise ValueError(
                "po_paths must resolve to distinct paths per language."
            )
        # Also forbid writing a target on top of the source and the
        # source-path aliasing any PO path, mirroring the single-target
        # safeguards.
        source_resolved = _resolve(Path(source_path))
        for lang, p in target_resolved.items():
            if p == source_resolved:
                raise ValueError(
                    f"target_paths[{lang!r}] cannot be the source path "
                    f"({source_resolved})."
                )
        for lang, p in po_resolved.items():
            if p == source_resolved:
                raise ValueError(
                    f"po_paths[{lang!r}] cannot be the source path "
                    f"({source_resolved})."
                )
        # Cross-set collision: a target path and a PO path that resolve
        # to the same file (same lang or different lang) would let the
        # two save loops clobber each other — the PO save rewrites a
        # markdown file, or vice versa — deterministically corrupting
        # output.  Forbid any overlap between the two sets.
        for tgt_lang, tgt_p in target_resolved.items():
            for po_lang, po_p in po_resolved.items():
                if tgt_p == po_p:
                    raise ValueError(
                        f"target_paths[{tgt_lang!r}] and "
                        f"po_paths[{po_lang!r}] resolve to the same "
                        f"file ({tgt_p}); target markdown and PO files "
                        f"must not alias each other across languages."
                    )

        parser = BlockParser()
        usage = getattr(self._tls, "usage", None) or _UsageAccumulator()
        start = time.monotonic()

        po_managers: Dict[str, POManager] = {
            lang: POManager(skip_types=self.SKIP_TYPES) for lang in langs
        }
        po_files: Dict[str, polib.POFile] = {}
        po_saved: Dict[str, bool] = {lang: False for lang in langs}
        source_path_str = str(source_path)
        document_started = False

        try:
            source_text = Path(source_path).read_text(encoding="utf-8")
            source_lines = source_text.splitlines(keepends=True)
            blocks = parser.segment_markdown(
                [line.rstrip("\n") for line in source_lines]
            )

            for lang in langs:
                po_files[lang] = po_managers[lang].load_or_create_po(
                    resolved_po_paths[lang], target_lang=lang
                )
                po_managers[lang].sync_po(
                    po_files[lang], blocks, parser.context_id
                )

            # Per-lang reference pools seeded from the respective PO.
            pools: Dict[str, ReferencePool] = {}
            for lang in langs:
                pool = ReferencePool(max_results=self.max_reference_pairs)
                pool.seed_from_po(po_files[lang])
                pools[lang] = pool

            # Union pending entries across langs — a block is "in play"
            # if any language still needs it.  Entries are matched by
            # msgctxt so the processor can commit per-lang writes
            # independently even when langs are at different coverage.
            pending_by_lang: Dict[str, List[polib.POEntry]] = {}
            initial_skipped_by_lang: Dict[str, int] = {}
            for lang in langs:
                pending_lang, skipped = self._collect_pending_entries(
                    po_files[lang]
                )
                pending_by_lang[lang] = pending_lang
                initial_skipped_by_lang[lang] = skipped

            pending_ctx_any: set = set()
            for pl in pending_by_lang.values():
                pending_ctx_any.update(e.msgctxt for e in pl)

            # Canonical per-lang entry lookup so commit logic below can
            # address the exact POEntry per (lang, ctx) pair rather than
            # scanning the whole PO file each time.
            entry_by_ctx_by_lang: Dict[str, Dict[str, polib.POEntry]] = {}
            for lang in langs:
                entry_by_ctx_by_lang[lang] = {
                    e.msgctxt: e
                    for e in po_files[lang]
                    if not e.obsolete and e.msgctxt
                }

            # Build the ordered "any-lang pending" list in SOURCE
            # document order.  Walk ``blocks`` (the parser's
            # source-order view) so a later-Korean-only entry can't
            # jump ahead of an earlier-Japanese-only entry on
            # incremental runs.  Sorting by per-lang pending subsets
            # would break section-aware grouping and reference-pool
            # seeding whenever the langs' coverage differ.
            ordered_ctx_seen: set = set()
            any_pending_ordered: List[polib.POEntry] = []
            for block in blocks:
                ctx = parser.context_id(block)
                if ctx in ordered_ctx_seen:
                    continue
                if ctx not in pending_ctx_any:
                    continue
                # Pull the canonical POEntry from whichever lang's PO
                # has it — sync_po typically adds every block to every
                # lang's PO, but an externally-edited PO could leave a
                # gap.  Any lang's entry is usable for source-side
                # decomposition (msgid / msgctxt are shared across
                # langs).
                entry: Optional[polib.POEntry] = None
                for lang in langs:
                    candidate = entry_by_ctx_by_lang[lang].get(ctx)
                    if candidate is not None:
                        entry = candidate
                        break
                if entry is None:
                    continue
                ordered_ctx_seen.add(ctx)
                any_pending_ordered.append(entry)

            stats: Dict[str, int] = {
                "processed": 0,
                "failed": 0,
                "skipped": 0,
                "batched_calls": 0,
                "per_entry_calls": 0,
                "validated": 0,
                "validation_failed": 0,
            }

            progress_total: int
            if self.batch_size and self.batch_size > 0:
                groups = self._section_aware_groups(any_pending_ordered)
                progress_total = max(1, len(groups))
            else:
                groups = [[e] for e in any_pending_ordered]
                progress_total = max(1, len(groups))

            self._emit_progress(
                kind="document_start",
                path=source_path_str,
                total=progress_total,
            )
            document_started = True

            if any_pending_ordered:
                for i, group in enumerate(groups, start=1):
                    self._translate_group_multi(
                        group=group,
                        langs=langs,
                        entry_by_ctx_by_lang=entry_by_ctx_by_lang,
                        pending_by_lang=pending_by_lang,
                        po_managers=po_managers,
                        pools=pools,
                        stats=stats,
                        usage=usage,
                    )
                    self._emit_progress(
                        kind="document_progress",
                        path=source_path_str,
                        index=i,
                        total=progress_total,
                    )

            # Per-lang post-processing: rebuild markdown, save PO +
            # target markdown.  Coverage and per-lang translation_stats
            # fall out of the per-lang PO so consumers can see exactly
            # how many blocks each language ended up with.
            by_lang: Dict[str, ProcessResult] = {}
            for lang in langs:
                po_file = po_files[lang]
                reconstructor = DocumentReconstructor(
                    skip_types=self.SKIP_TYPES
                )
                coverage_dict = reconstructor.get_process_coverage(
                    blocks, po_file, parser.context_id
                )
                processed_content = reconstructor.rebuild_markdown(
                    source_lines, blocks, po_file, parser.context_id
                )
                self._save_processed_document(
                    processed_content, resolved_target_paths[lang]
                )
                po_managers[lang].save_po(po_file, resolved_po_paths[lang])
                po_saved[lang] = True

                # Per-language slice of the shared stats counters.  The
                # "per lang" view is important for operators: the shared
                # stats dict tracks bisection / batched-call counters at
                # the call level, but processed / failed / validation
                # and per-entry fallback totals are attributed per lang.
                # ``batched_calls`` remains the shared total because
                # each call served every language — splitting it
                # would misrepresent the billed API-call count.
                lang_stats = {
                    "processed": stats.get(f"_lang_processed_{lang}", 0),
                    "failed": stats.get(f"_lang_failed_{lang}", 0),
                    "skipped": initial_skipped_by_lang[lang],
                    "batched_calls": stats.get("batched_calls", 0),
                    "per_entry_calls": stats.get(
                        f"_lang_per_entry_calls_{lang}", 0
                    ),
                    "validated": stats.get(f"_lang_validated_{lang}", 0),
                    "validation_failed": stats.get(
                        f"_lang_validation_failed_{lang}", 0
                    ),
                }

                by_lang[lang] = ProcessResult(
                    source_path=str(source_path),
                    target_path=str(resolved_target_paths[lang]),
                    po_path=str(resolved_po_paths[lang]),
                    blocks_count=len(blocks),
                    coverage=Coverage(**coverage_dict),
                    translation_stats=BatchStats(**lang_stats),
                    receipt=None,
                )

            receipt = _build_receipt(
                model=self.model,
                target_lang=",".join(langs),
                source_path=str(source_path),
                target_path=",".join(
                    str(resolved_target_paths[lang]) for lang in langs
                ),
                po_path=",".join(
                    str(resolved_po_paths[lang]) for lang in langs
                ),
                usage=usage,
                duration_seconds=time.monotonic() - start,
            )

            return {
                "source_path": str(source_path),
                "target_langs": list(langs),
                "by_lang": by_lang,
                "receipt": receipt,
            }
        except BaseException as exc:
            if usage.api_calls > 0:
                partial = _build_receipt(
                    model=self.model,
                    target_lang=",".join(langs) if langs else "",
                    source_path=str(source_path),
                    target_path=",".join(
                        str(resolved_target_paths[lang]) for lang in langs
                    ) if langs else None,
                    po_path=",".join(
                        str(resolved_po_paths[lang]) for lang in langs
                    ) if langs else None,
                    usage=usage,
                    duration_seconds=time.monotonic() - start,
                )
                try:
                    exc.partial_receipt = partial  # type: ignore[attr-defined]
                except (AttributeError, TypeError):
                    pass
            # Preserve per-lang translations already written to their PO
            # entries on failure — unless a save_po itself raised, which
            # would likely raise again.
            for lang in langs:
                pf = po_files.get(lang)
                if pf is None or po_saved.get(lang):
                    continue
                try:
                    po_managers[lang].save_po(pf, resolved_po_paths[lang])
                except Exception:
                    logger.exception(
                        "PO save failed during multi-target error handling "
                        "for lang=%s path=%s",
                        lang,
                        resolved_po_paths[lang],
                    )
            raise
        finally:
            if document_started:
                self._emit_progress(
                    kind="document_end", path=source_path_str
                )

    def _translate_group_multi(
        self,
        *,
        group: List[polib.POEntry],
        langs: List[str],
        entry_by_ctx_by_lang: Dict[str, Dict[str, polib.POEntry]],
        pending_by_lang: Dict[str, List[polib.POEntry]],
        po_managers: Dict[str, POManager],
        pools: Dict[str, ReferencePool],
        stats: Dict[str, int],
        usage: Optional[_UsageAccumulator] = None,
    ) -> None:
        """Multi-target counterpart of :meth:`_translate_group`.

        Drives a single LLM call that returns ``{ctx: {lang: translation}}``
        for every block in ``group``, then fans the returned translations
        out to each language's PO file.  Blocks missing in the response
        (or returning incomplete per-lang coverage) fall back to
        independent single-target :meth:`_call_llm` calls per lang so
        the caller never ends up with a partial commit.
        """
        if not group:
            return

        pending_ctx_by_lang: Dict[str, set] = {
            lang: {e.msgctxt for e in pending_by_lang[lang]}
            for lang in langs
        }

        # Shared source-side decomposition: encode once per block, same
        # placeholder mapping for every lang.  Skipped when a ctx is
        # fully done across all langs (no pending lang needs it).
        items: Dict[str, str] = {}
        mappings: Dict[str, Optional[PlaceholderMap]] = {}
        ctx_entries: Dict[str, polib.POEntry] = {}
        for e in group:
            if not any(e.msgctxt in pending_ctx_by_lang[l] for l in langs):
                continue
            encoded, mapping = self._encode_source(e.msgid)
            items[e.msgctxt] = encoded
            mappings[e.msgctxt] = mapping
            ctx_entries[e.msgctxt] = e

        if not items:
            return

        raw_sources = [ctx_entries[c].msgid for c in items]

        # Per-lang reference block composed from each lang's pool.
        # Merge into a single per-lang section list so the model sees
        # terminology hints for every target at once without mixing
        # translations between languages.
        per_lang_references: Dict[str, Optional[List[tuple]]] = {}
        for lang in langs:
            per_lang_references[lang] = self._collect_references(
                raw_sources, pools[lang]
            )

        per_lang_glossary_blocks = self._collect_multi_glossary_blocks(
            raw_sources, langs
        )

        # ``batch_size=0`` is documented as the per-entry opt-out — a
        # caller on a provider that misbehaves with JSON-mode batch
        # calls should be able to get N independent single-target
        # calls per block without silently going through the batched
        # wire.  Skip the MultiTargetBatchTranslator entirely in that
        # case; every entry then drops into the per-lang fallback
        # below, which mirrors what ``_process_entries_sequential``
        # does for single-target ``batch_size=0``.
        if self.batch_size == 0:
            translated: Dict[str, Dict[str, str]] = {}
        else:
            base_caller = self._make_multi_batch_caller(
                langs=langs,
                per_lang_references=per_lang_references,
                per_lang_glossary_blocks=per_lang_glossary_blocks,
                usage=usage,
            )

            call_count = 0

            def counting_caller(chunk: Dict[str, str]) -> str:
                nonlocal call_count
                call_count += 1
                return base_caller(chunk)

            translator = MultiTargetBatchTranslator(
                counting_caller,
                target_langs=langs,
                max_entries=self.batch_size,
                max_chars=self.batch_max_chars,
            )

            try:
                translated = translator.translate(items)
            except Exception as exc:
                logger.warning(
                    "Multi-target batch translator crashed: %s; falling back", exc
                )
                translated = {}
            finally:
                stats["batched_calls"] = stats.get("batched_calls", 0) + call_count

        # Fan-out commit: per-lang PO writes.  A block that came back
        # with full coverage commits to every pending lang; otherwise
        # each lang falls back to the single-target per-entry call.
        for ctx, encoded in items.items():
            entry_any = ctx_entries[ctx]
            mapping = mappings.get(ctx)
            returned = translated.get(ctx)
            for lang in langs:
                if ctx not in pending_ctx_by_lang[lang]:
                    continue
                entry = entry_by_ctx_by_lang[lang].get(ctx)
                if entry is None:
                    # The block is not present in this lang's PO —
                    # nothing to commit.  This can happen when
                    # ``sync_po`` drops entries under a rare race, but
                    # is effectively a no-op here.
                    continue

                raw: Optional[str] = None
                if isinstance(returned, dict):
                    lang_val = returned.get(lang)
                    if isinstance(lang_val, str):
                        raw = lang_val

                if raw is None:
                    # Per-lang fallback: run the single-target per-entry
                    # call against the same pool the multi-target path
                    # uses, so reference continuity is preserved.
                    stats["per_entry_calls"] = stats.get("per_entry_calls", 0) + 1
                    stats[f"_lang_per_entry_calls_{lang}"] = (
                        stats.get(f"_lang_per_entry_calls_{lang}", 0) + 1
                    )
                    similar = pools[lang].find_similar(entry.msgid)
                    processed = self._call_lang_single(
                        entry.msgid,
                        target_lang=lang,
                        reference_pairs=similar or None,
                        usage=usage,
                    )
                    if processed is None:
                        stats["failed"] = stats.get("failed", 0) + 1
                        stats[f"_lang_failed_{lang}"] = (
                            stats.get(f"_lang_failed_{lang}", 0) + 1
                        )
                        continue
                    self._commit_multi_entry(
                        entry=entry,
                        processed=processed,
                        lang=lang,
                        po_manager=po_managers[lang],
                        pool=pools[lang],
                        stats=stats,
                    )
                    continue

                # Normal path: post_process, round-trip stash for
                # validation, decode, commit.
                if self._post_process:
                    raw = self._post_process(raw)
                self._tls.last_encoded_response = raw
                self._tls.last_placeholder_map = mapping
                processed = self._decode_translation(raw, mapping)
                self._commit_multi_entry(
                    entry=entry,
                    processed=processed,
                    lang=lang,
                    po_manager=po_managers[lang],
                    pool=pools[lang],
                    stats=stats,
                )

    def _call_lang_single(
        self,
        source_text: str,
        *,
        target_lang: str,
        reference_pairs: Optional[List[tuple]] = None,
        usage: Optional[_UsageAccumulator] = None,
    ) -> Optional[str]:
        """Single-target LLM call used by the multi-target per-lang fallback.

        Mirrors :meth:`_call_llm` but lets the caller override
        ``target_lang`` without mutating ``self.target_lang`` (which is
        ignored under :meth:`process_document_multi`).  Returns ``None``
        on call failure so the caller can count the entry as failed
        rather than aborting the whole group.
        """
        try:
            encoded_source, mapping = self._encode_source(source_text)
            if self.mode == "refine":
                instruction = Prompts.REFINE_INSTRUCTION
                system_template = Prompts.REFINE_SYSTEM_TEMPLATE
            else:
                instruction = Prompts.TRANSLATE_INSTRUCTION
                system_template = Prompts.TRANSLATE_SYSTEM_TEMPLATE
            if self._extra_instructions:
                instruction += "\n" + self._extra_instructions
            system_content = system_template.format(
                lang=target_lang,
                instruction=instruction,
            )
            glossary_block = self._format_glossary_for_lang(
                source_text, target_lang
            )
            if glossary_block:
                system_content += "\n\n" + glossary_block
            messages: List[Dict[str, Any]] = [
                self._system_message(system_content)
            ]
            if reference_pairs:
                for ref_src, ref_tgt in reference_pairs:
                    messages.append({"role": "user", "content": ref_src})
                    messages.append({"role": "assistant", "content": ref_tgt})
            messages.append({"role": "user", "content": encoded_source})
            response = litellm.completion(
                model=self.model, messages=messages, **self._litellm_kwargs
            )
            if usage is not None:
                usage.record(response)
            raw = response.choices[0].message.content
            if self._post_process:
                raw = self._post_process(raw)
            self._tls.last_encoded_response = raw
            self._tls.last_placeholder_map = mapping
            return self._decode_translation(raw, mapping)
        except Exception as exc:
            logger.warning(
                "Multi-target per-lang fallback failed (lang=%s): %s",
                target_lang,
                exc,
            )
            return None

    def _commit_multi_entry(
        self,
        *,
        entry: polib.POEntry,
        processed: str,
        lang: str,
        po_manager: POManager,
        pool: ReferencePool,
        stats: Dict[str, int],
    ) -> None:
        """Validate + commit one (entry, lang) pair for the multi-target path."""
        if processed.strip() == entry.msgid.strip():
            block_type = self._extract_block_type_from_msgctxt(entry.msgctxt)
            if block_type != "code":
                logger.warning(
                    "LLM returned untranslated output for entry %s (lang=%s)",
                    entry.msgctxt,
                    lang,
                )

        # Temporarily override target_lang AND the locale-resolved
        # glossary for validation so the target-language-presence check
        # and glossary-preservation check both report against this
        # call's target locale — not the processor constructor's
        # ``target_lang``.  Without the glossary swap a ja/zh commit is
        # measured against ko's glossary mapping, so a correct ja
        # translation can be flagged fuzzy (missing glossary term) or a
        # forbidden-term leak can slip through unnoticed.
        original_target = self.target_lang
        original_glossary = self._glossary
        self.target_lang = lang
        self._glossary = self._resolve_glossary_for_lang(lang)
        try:
            ok = self._apply_validation(entry, processed, False, pool, stats)
        finally:
            self.target_lang = original_target
            self._glossary = original_glossary

        if not ok:
            stats["failed"] = stats.get("failed", 0) + 1
            stats[f"_lang_failed_{lang}"] = (
                stats.get(f"_lang_failed_{lang}", 0) + 1
            )
            if self.validation != "off":
                stats[f"_lang_validation_failed_{lang}"] = (
                    stats.get(f"_lang_validation_failed_{lang}", 0) + 1
                )
            return

        source_msgid = entry.msgid
        entry.msgstr = processed
        po_manager.mark_entry_processed(entry)
        stats["processed"] = stats.get("processed", 0) + 1
        stats[f"_lang_processed_{lang}"] = (
            stats.get(f"_lang_processed_{lang}", 0) + 1
        )
        if self.validation != "off":
            stats[f"_lang_validated_{lang}"] = (
                stats.get(f"_lang_validated_{lang}", 0) + 1
            )
        pool.add(source_msgid, processed)

    def _collect_multi_glossary_blocks(
        self, sources: List[str], langs: List[str]
    ) -> Dict[str, Optional[str]]:
        """Union glossary terms matched anywhere in the batch, resolved per lang.

        Each returned block is already locale-specific: entries from a
        ``glossary_path`` with per-locale dicts get the right
        target-language form for ``lang`` instead of reusing
        ``self.target_lang``'s resolution.  Returns ``{lang: block_or_None}``.
        """
        if not self._glossary_inline and self._glossary_file is None:
            return {lang: None for lang in langs}
        joined = "\n".join(sources)
        return {
            lang: self._format_glossary_for_lang(joined, lang)
            for lang in langs
        }

    def _build_multi_batch_messages(
        self,
        items: Dict[str, str],
        *,
        langs: List[str],
        per_lang_references: Dict[str, Optional[List[tuple]]],
        per_lang_glossary_blocks: Optional[Dict[str, Optional[str]]] = None,
    ) -> List[Dict[str, Any]]:
        langs_str = ", ".join(langs)
        instruction = Prompts.BATCH_MULTI_TRANSLATE_INSTRUCTION.format(
            langs=langs_str
        )
        if self._extra_instructions:
            instruction += "\n" + self._extra_instructions
        system_content = Prompts.BATCH_MULTI_TRANSLATE_SYSTEM_TEMPLATE.format(
            langs=langs_str,
            instruction=instruction,
        )
        if per_lang_glossary_blocks:
            for lang in langs:
                block = per_lang_glossary_blocks.get(lang)
                if not block:
                    continue
                # Label each glossary block with its locale so the LLM
                # can apply the right mapping when producing that
                # language's value — a single pooled block would let a
                # Korean term's replacement leak into the Japanese
                # output, which is exactly the wrong-terminology path
                # this split was introduced to prevent.
                system_content += f"\n\nGlossary ({lang}):\n" + block

        for lang in langs:
            refs = per_lang_references.get(lang)
            if not refs:
                continue
            ref_lines = [
                f"Reference translations for {lang} (maintain this tone and terminology):",
            ]
            for src, tgt in refs:
                ref_lines.append(f"- SRC: {src}\n  TGT: {tgt}")
            system_content += "\n\n" + "\n".join(ref_lines)

        user_payload = json.dumps(items, ensure_ascii=False)
        return [
            self._system_message(system_content),
            {"role": "user", "content": user_payload},
        ]

    def _make_multi_batch_caller(
        self,
        *,
        langs: List[str],
        per_lang_references: Dict[str, Optional[List[tuple]]],
        per_lang_glossary_blocks: Optional[Dict[str, Optional[str]]] = None,
        usage: Optional[_UsageAccumulator] = None,
    ) -> Callable[[Dict[str, str]], str]:
        def _call(items: Dict[str, str]) -> str:
            messages = self._build_multi_batch_messages(
                items,
                langs=langs,
                per_lang_references=per_lang_references,
                per_lang_glossary_blocks=per_lang_glossary_blocks,
            )
            call_kwargs = dict(self._litellm_kwargs)
            if self._supports_json_mode():
                call_kwargs.setdefault(
                    "response_format", {"type": "json_object"}
                )
            response = litellm.completion(
                model=self.model, messages=messages, **call_kwargs
            )
            if usage is not None:
                usage.record(response)
            return response.choices[0].message.content

        return _call

    def _collect_references(
        self, sources, pool: ReferencePool
    ) -> Optional[List[tuple]]:
        """Deduplicate top-K similar pairs across all sources in the batch."""
        if len(pool) == 0:
            return None
        seen = set()
        merged: List[tuple] = []
        for src in sources:
            for ref_src, ref_tgt in pool.find_similar(src):
                key = (ref_src, ref_tgt)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(key)
                if len(merged) >= self.max_reference_pairs:
                    return merged
        return merged or None

    def _collect_glossary_block(self, sources) -> Optional[str]:
        """Union glossary terms matched anywhere in the batch."""
        if not self._glossary:
            return None
        joined = "\n".join(sources)
        return self._format_glossary(joined)

    def _apply_validation(
        self,
        entry: polib.POEntry,
        processed: str,
        inplace: bool,
        pool: ReferencePool,
        stats: Optional[Dict[str, int]] = None,
    ) -> bool:
        """Run the validator (if enabled) and mark failures fuzzy.

        When the placeholder registry is active, the round-trip check fires
        regardless of ``self.validation`` — a missing/duplicated/unexpected
        token is always a structural fail.  Other structural checks still
        respect the ``validation`` mode.

        Returns ``True`` when the caller should treat the entry as processed,
        ``False`` when validation failed hard enough that the entry stays in
        a failed state (for batched callers that want to count it).
        """
        # Consume the stash set by ``_call_llm`` / ``_translate_group``.
        # Clearing unconditionally stops a later per-entry call on the same
        # thread from inheriting the previous entry's mapping.
        encoded_response = getattr(self._tls, "last_encoded_response", None)
        mapping = getattr(self._tls, "last_placeholder_map", None)
        self._tls.last_encoded_response = None
        self._tls.last_placeholder_map = None

        issues: List[ValidationIssue] = []

        # Round-trip runs on the pre-decode response so it can see tokens.
        # Structural checks below run on ``processed`` (the decoded,
        # post-processed translation) because pattern authors are allowed
        # to protect spans that include Markdown syntax — fenced blocks,
        # inline code, headings — and the encoded text has those glyphs
        # swapped for tokens, which would defeat fence / heading / inline
        # code counts.
        if mapping and encoded_response is not None:
            reason = check_round_trip(encoded_response, mapping)
            if reason:
                issues.append(
                    ValidationIssue("placeholder_roundtrip", reason)
                )
            # Positional guard for the T-6 built-ins: a count-only check
            # lets a model that moves ``\u27e6P:N\u27e7`` elsewhere in the
            # same block pass silently — an anchor slid off its heading
            # into the next paragraph, or an attribute jumped to a
            # different tag, then decodes to a structurally broken
            # document.  Compare the structural context (heading line
            # for anchors, in-tag for HTML attributes) between source
            # and decoded translation and fail on any drift.
            # Scan with the user's override (regex + predicate) when
            # they replaced a built-in, so the structural check still
            # guards the exact spans they tokenized; otherwise fall
            # back to the default regex and let ``_anchor_positions``
            # / ``_attr_tag_signatures`` apply their built-in filters.
            # Multiple same-name overrides are supported: the check
            # runs once per ``(regex, predicate)`` pair so every
            # override keeps its positional guarantee.
            anchor_overrides = self._builtin_overrides.get(
                "anchor", [(ANCHOR_PATTERN, None)]
            )
            html_attr_overrides = self._builtin_overrides.get(
                "html_attr", [(HTML_ATTR_PATTERN, None)]
            )
            position_reasons: List[str] = []
            for a_regex, a_pred in anchor_overrides:
                r = check_structural_position(
                    entry.msgid,
                    processed,
                    anchor_pattern=a_regex,
                    anchor_predicate=a_pred,
                    check_html_attr=False,
                )
                if r:
                    position_reasons.append(r)
            for h_regex, h_pred in html_attr_overrides:
                r = check_structural_position(
                    entry.msgid,
                    processed,
                    html_attr_pattern=h_regex,
                    html_attr_predicate=h_pred,
                    check_anchor=False,
                )
                if r:
                    position_reasons.append(r)
            if position_reasons:
                issues.append(
                    ValidationIssue(
                        "placeholder_position",
                        "; ".join(position_reasons),
                    )
                )

        # Refine mode's same-language contract is a core feature
        # guarantee — it must fire regardless of ``self.validation``.
        # If we gated it on ``validation != "off"``, the default config
        # (``mode="refine"``, ``validation="off"``) would accept a
        # silently-translated response and write it to disk/PO without
        # flagging drift.  Keep this independent of the ``validation``
        # setting, alongside ``placeholder_roundtrip`` above.
        if self.mode == "refine":
            stability = check_language_stability(entry.msgid, processed)
            if stability is not None:
                issues.append(stability)

        if self.validation != "off":
            result = validate_translation(
                entry.msgid,
                processed,
                target_lang=self.target_lang,
                glossary=self._glossary,
                mode=self.validation,
                purpose=self.mode,
            )
            # Deduplicate: when ``validation != "off"`` the refine-purpose
            # branch inside ``validate()`` also runs ``language_stability``
            # and would double-flag the same issue.
            seen_checks = {i.check for i in issues}
            for issue in result.issues:
                if issue.check in seen_checks:
                    continue
                issues.append(issue)

        if not issues:
            if self.validation != "off" and stats is not None:
                stats["validated"] = stats.get("validated", 0) + 1
            return True

        reasons_joined = "; ".join(f"{i.check}: {i.detail}" for i in issues)
        reason = f"validator: {reasons_joined}"
        existing = entry.tcomment or ""
        entry.tcomment = f"{existing}\n{reason}".strip() if existing else reason
        if "fuzzy" not in entry.flags:
            entry.flags.append("fuzzy")
        # Clear any prior translation: re-translating an edited block that
        # then fails validation must not leave the stale msgstr in place,
        # because ``rebuild_markdown`` treats any non-empty msgstr as
        # authoritative and would ship the outdated content.  Blanking it
        # forces the reconstructor to fall back to the source until a human
        # resolves the fuzzy entry.
        entry.msgstr = ""
        # Also skip: writing the rejected output to msgstr (ditto above),
        # propagating it to msgid under inplace (that would turn the bad
        # translation into the next run's source), and seeding the
        # reference pool (suspect translations must not become few-shot
        # context for later blocks).
        if stats is not None:
            stats["validation_failed"] = stats.get("validation_failed", 0) + 1
        return False

    # ----- helpers -----

    def _match_ctxt(self, processed_content: str, parser=None, po_manager=None):
        p = parser or self.parser
        pm = po_manager or self.po_manager
        processed_lines = processed_content.splitlines(keepends=True)
        blocks = p.segment_markdown(
            [line.rstrip("\n") for line in processed_lines]
        )
        pm.redraw_context(blocks, p.context_id)

    def _extract_block_type_from_msgctxt(self, msgctxt: str) -> str:
        if msgctxt:
            start = msgctxt.find("::")
            if start != -1:
                start += 2
                end = msgctxt.find(":", start)
                if end != -1:
                    return msgctxt[start:end]
        return ""

    def _extract_block_type(self, context_id: str) -> str:
        if "::" in context_id:
            parts = context_id.split("::")
            if len(parts) >= 2:
                return parts[1]
        return "unknown"

    def _save_processed_document(self, processed_content: str, target_path: Path):
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(processed_content, encoding="utf-8", newline="\n")

    def get_translation_stats(self, source_path: Path, po_path: Path) -> Dict[str, Any]:
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
        source_lines = source_path.read_text(encoding="utf-8").splitlines(keepends=True)
        blocks = self.parser.segment_markdown(
            [line.rstrip("\n") for line in source_lines]
        )
        po_file = self.po_manager.load_or_create_po(po_path)

        return self.reconstructor.export_translation_report(
            str(source_path), blocks, po_file, self.parser.context_id
        )

    def estimate(
        self, source_path: Path, po_path: Path | None = None
    ) -> Dict[str, Any]:
        """Dry-run cost estimator.

        Counts pending blocks without making API calls and estimates input /
        output tokens via LiteLLM's token counter.  Output is approximated as
        ``1.2×`` the source-token count, which holds for most target languages.

        Args:
            source_path: Path to source markdown.
            po_path: Path to PO file; defaults to ``source_path`` with a
                ``.po`` suffix.  Missing PO ⇒ every translatable block
                is pending.

        Returns:
            Dict with ``pending_blocks``, ``input_tokens``, ``output_tokens``,
            and ``batches`` (expected number of LLM calls under current
            ``batch_size`` / ``batch_max_chars``).
        """
        if po_path is None:
            po_path = Path(source_path).with_suffix(".po")

        parser = BlockParser()
        po_manager = POManager(skip_types=self.SKIP_TYPES)

        source = Path(source_path).read_text(encoding="utf-8")
        source_lines = source.splitlines(keepends=True)
        blocks = parser.segment_markdown(
            [line.rstrip("\n") for line in source_lines]
        )

        if Path(po_path).exists():
            po_file = po_manager.load_or_create_po(Path(po_path))
            po_manager.sync_po(po_file, blocks, parser.context_id)
            pending = [
                e
                for e in po_file
                if not e.obsolete
                and self._extract_block_type_from_msgctxt(e.msgctxt) not in self.SKIP_TYPES
                and ((not e.msgstr) or ("fuzzy" in e.flags))
            ]
        else:
            # No PO yet — synthesise POEntry stubs so partitioning mirrors
            # what the real batched path would do (section-aware splits plus
            # entry/char caps), rather than relying on a global aggregate
            # that undercounts when individual blocks are close to
            # ``batch_max_chars``.
            pending = [
                polib.POEntry(
                    msgctxt=parser.context_id(b), msgid=b["text"], msgstr=""
                )
                for b in blocks
                if b["type"] not in self.SKIP_TYPES
            ]

        sources = [e.msgid for e in pending]
        total_chars = sum(len(s) for s in sources)

        try:
            raw = litellm.token_counter(
                model=self.model, text="\n".join(sources)
            )
            input_tokens = int(raw)
        except Exception:
            input_tokens = max(1, total_chars // 4)  # rough fallback
        output_tokens = int(input_tokens * 1.2)

        if self.batch_size and self.batch_size > 0 and sources:
            # Always drive the estimate through the real section-aware
            # partitioner so the reported batch count matches the actual
            # LLM-call count, regardless of whether a PO exists.
            groups = self._section_aware_groups(pending)
            batches = max(1, len(groups))
        else:
            batches = len(sources)

        return {
            "pending_blocks": len(sources),
            "total_chars": total_chars,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "batches": batches,
        }
