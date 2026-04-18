"""
Main Markdown Translator orchestrator class.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

logger = logging.getLogger(__name__)

import litellm
import polib

from .batch import BatchTranslator
from .manager import POManager
from .parser import BlockParser
from .prompts import Prompts
from .reconstructor import DocumentReconstructor
from .reference_pool import ReferencePool
from .results import BatchStats, Coverage, DirectoryResult, ProcessResult
from .validator import validate as validate_translation


ValidationMode = Literal["off", "conservative", "strict"]


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
        self._litellm_kwargs = litellm_kwargs

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
        if not self._glossary:
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

    def _build_messages(self, source_text: str, reference_pairs=None):
        instruction = Prompts.TRANSLATE_INSTRUCTION
        if self._extra_instructions:
            instruction += "\n" + self._extra_instructions
        system_content = Prompts.TRANSLATE_SYSTEM_TEMPLATE.format(
            lang=self.target_lang,
            instruction=instruction,
        )

        glossary_block = self._format_glossary(source_text)
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

    def _call_llm(self, source_text: str, reference_pairs=None):
        messages = self._build_messages(source_text, reference_pairs)
        response = litellm.completion(
            model=self.model, messages=messages, **self._litellm_kwargs
        )
        result = response.choices[0].message.content
        if self._post_process:
            result = self._post_process(result)
        return result

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

        po_file = None
        try:
            source = source_path.read_text(encoding="utf-8")
            source_lines = source.splitlines(keepends=True)
            blocks = parser.segment_markdown(
                [line.rstrip("\n") for line in source_lines]
            )

            po_file = po_manager.load_or_create_po(po_path, target_lang=self.target_lang)
            po_manager.sync_po(po_file, blocks, parser.context_id)

            if self.batch_size and self.batch_size > 0:
                translation_stats = self._process_entries_batched(
                    po_file, po_manager, inplace=inplace
                )
            else:
                translation_stats = self._process_entries_sequential(
                    po_file, po_manager, inplace=inplace
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
                # Without this, a validator failure followed by ``inplace``
                # writes a fresh entry with no fuzzy flag and the next run
                # treats the bad translation as fully processed.
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

            return ProcessResult(
                source_path=str(source_path),
                target_path=str(target_path),
                po_path=str(po_path),
                blocks_count=len(blocks),
                coverage=Coverage(**coverage_dict),
                translation_stats=BatchStats(**translation_stats),
            )
        finally:
            if po_file is not None:
                po_manager.save_po(po_file, po_path)

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

        results: List[Any] = []
        files_processed = 0
        files_failed = 0
        files_skipped = 0

        def _process_one(source_file: Path):
            relative_path = source_file.relative_to(source_dir)
            target_path = target_dir / relative_path
            po_path_file = (
                po_dir / relative_path.with_suffix(".po")
                if po_dir is not None
                else None
            )
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

                    stats = result.translation_stats
                    if stats.failed > 0 or stats.validation_failed > 0:
                        # Entry-level failures still count as a file-level
                        # failure so automation can detect partial errors
                        # via ``files_failed`` / the CLI exit code.
                        files_failed += 1
                    elif stats.processed == 0:
                        files_skipped += 1
                    else:
                        files_processed += 1
                except Exception:
                    logger.exception("Failed to process %s", source_file)
                    files_failed += 1
                    results.append({"source_path": str(source_file), "error": True})

        return DirectoryResult(
            source_dir=str(source_dir),
            target_dir=str(target_dir),
            po_dir=str(po_dir) if po_dir is not None else None,
            files_processed=files_processed,
            files_failed=files_failed,
            files_skipped=files_skipped,
            results=results,
        )

    # ----- sequential path (v0.2 fallback when batch_size=0) -----

    def _process_entry(self, entry, reference_pairs=None):
        try:
            processed = self._call_llm(entry.msgid, reference_pairs=reference_pairs)
            return (entry, processed, None)
        except Exception as e:
            return (entry, None, e)

    def _process_entries_sequential(
        self, po_file: polib.POFile, po_manager: POManager, inplace: bool = False
    ) -> Dict[str, int]:
        pool = ReferencePool(max_results=self.max_reference_pairs)
        pool.seed_from_po(po_file)

        stats: Dict[str, int] = {"processed": 0, "failed": 0, "skipped": 0}

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

            similar = pool.find_similar(entry.msgid)

            try:
                entry_obj, processed, error = self._process_entry(
                    entry, reference_pairs=similar or None
                )
                if error is not None:
                    logger.warning(
                        "Failed to translate entry %s: %s",
                        entry_obj.msgctxt, error,
                    )
                    stats["failed"] += 1
                    continue

                if processed is not None and processed.strip() == entry.msgid.strip():
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

        return stats

    # ----- batched path -----

    def _process_entries_batched(
        self, po_file: polib.POFile, po_manager: POManager, inplace: bool = False
    ) -> Dict[str, int]:
        pool = ReferencePool(max_results=self.max_reference_pairs)
        pool.seed_from_po(po_file)

        stats: Dict[str, int] = {
            "processed": 0,
            "failed": 0,
            "skipped": 0,
            "batched_calls": 0,
            "per_entry_calls": 0,
            "validated": 0,
            "validation_failed": 0,
        }

        pending: List[polib.POEntry] = []
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
            pending.append(entry)

        if not pending:
            return stats

        for group in self._section_aware_groups(pending):
            self._translate_group(
                group, po_manager, pool, stats, inplace=inplace
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
    ) -> None:
        items: Dict[str, str] = {e.msgctxt: e.msgid for e in group}
        entry_by_ctx: Dict[str, polib.POEntry] = {e.msgctxt: e for e in group}

        references = self._collect_references(items.values(), pool)
        glossary_block = self._collect_glossary_block(items.values())

        base_caller = self._make_batch_caller(references, glossary_block)

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
            if ctx in translated:
                raw = translated[ctx]
                processed = self._post_process(raw) if self._post_process else raw
            else:
                stats["per_entry_calls"] += 1
                similar = pool.find_similar(entry.msgid)
                try:
                    processed = self._call_llm(entry.msgid, reference_pairs=similar or None)
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

        Returns ``True`` when the caller should treat the entry as processed,
        ``False`` when validation failed hard enough that the entry stays in
        a failed state (for batched callers that want to count it).
        """
        if self.validation == "off":
            return True

        result = validate_translation(
            entry.msgid,
            processed,
            target_lang=self.target_lang,
            glossary=self._glossary,
            mode=self.validation,
        )
        if result.ok:
            if stats is not None:
                stats["validated"] = stats.get("validated", 0) + 1
            return True

        reason = f"validator: {result.reasons()}"
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
