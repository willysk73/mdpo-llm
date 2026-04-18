"""
Main Markdown Translator orchestrator class.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

logger = logging.getLogger(__name__)

import litellm
import polib

from .batch import BatchTranslator
from .manager import POManager
from .parser import BlockParser
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
from .validator import ValidationIssue, validate as validate_translation


ValidationMode = Literal["off", "conservative", "strict"]
GlossaryMode = Literal["instruction", "placeholder"]


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
        validation: ValidationMode = "off",
        enable_prompt_cache: bool = False,
        progress_callback: Optional[ProgressCallback] = None,
        placeholders: Optional[PlaceholderRegistry] = None,
        glossary_mode: GlossaryMode = "instruction",
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
        self._glossary = self._resolve_glossary(glossary, glossary_path)
        self.batch_size = batch_size
        self.batch_max_chars = batch_max_chars
        self.validation: ValidationMode = validation
        self.enable_prompt_cache = enable_prompt_cache
        self._progress_callback = progress_callback
        self._placeholders = placeholders
        self.glossary_mode: GlossaryMode = glossary_mode
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

    # ----- per-entry messaging -----

    def _build_messages(
        self,
        source_text: str,
        reference_pairs=None,
        *,
        glossary_source: Optional[str] = None,
    ):
        instruction = Prompts.TRANSLATE_INSTRUCTION
        if self._extra_instructions:
            instruction += "\n" + self._extra_instructions
        system_content = Prompts.TRANSLATE_SYSTEM_TEMPLATE.format(
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
        instruction = Prompts.BATCH_TRANSLATE_INSTRUCTION
        if self._extra_instructions:
            instruction += "\n" + self._extra_instructions
        system_content = Prompts.BATCH_TRANSLATE_SYSTEM_TEMPLATE.format(
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
    ) -> ProcessResult:
        """
        Process a markdown document.
        """
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
            source = source_path.read_text(encoding="utf-8")
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

    def process_directory(
        self,
        source_dir: Path,
        target_dir: Path,
        po_dir: Path | None = None,
        glob: str = "**/*.md",
        inplace: bool = False,
        max_workers: int = 4,
    ) -> DirectoryResult:
        """
        Process all markdown files in a directory tree.
        """
        import concurrent.futures

        source_dir = Path(source_dir)
        target_dir = Path(target_dir)

        matched_files = sorted(source_dir.glob(glob))
        start = time.monotonic()

        results: List[Any] = []
        files_processed = 0
        files_failed = 0
        files_skipped = 0
        # Hold each worker's accumulator so a mid-run exception still
        # contributes its billed tokens to the directory-level receipt.
        worker_usages: List[_UsageAccumulator] = []

        self._emit_progress(
            kind="directory_start",
            path=str(source_dir),
            total=len(matched_files),
        )

        def _process_one(source_file: Path):
            relative_path = source_file.relative_to(source_dir)
            target_path = target_dir / relative_path
            po_path_file = (
                po_dir / relative_path.with_suffix(".po")
                if po_dir is not None
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
                result = self.process_document(
                    source_file, target_path, po_path_file, inplace=inplace
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

        dir_receipt = _build_receipt(
            model=self.model,
            target_lang=self.target_lang,
            source_path=str(source_dir),
            target_path=str(target_dir),
            po_path=str(po_dir) if po_dir is not None else None,
            usage=dir_usage,
            duration_seconds=duration,
        )

        return DirectoryResult(
            source_dir=str(source_dir),
            target_dir=str(target_dir),
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
                    # still surface.
                    block_type = self._extract_block_type_from_msgctxt(entry.msgctxt)
                    if block_type != "code":
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
    ) -> None:
        entry_by_ctx: Dict[str, polib.POEntry] = {e.msgctxt: e for e in group}

        # Reference lookup and glossary matching operate on human-readable
        # source text — encoding would obscure the substrings they compare
        # against — so collect them from ``entry.msgid`` before encoding
        # swaps pattern matches for opaque tokens.
        raw_sources = [e.msgid for e in group]
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
                # prose regressions stay visible.
                block_type = self._extract_block_type_from_msgctxt(entry.msgctxt)
                if block_type != "code":
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
            pool.add(source_msgid, processed)

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

        if self.validation != "off":
            result = validate_translation(
                entry.msgid,
                processed,
                target_lang=self.target_lang,
                glossary=self._glossary,
                mode=self.validation,
            )
            issues.extend(result.issues)

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
