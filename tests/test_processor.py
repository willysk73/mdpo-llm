"""Tests for MarkdownProcessor end-to-end and error handling."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mdpo_llm.placeholder import PlaceholderRegistry
from mdpo_llm.processor import MarkdownProcessor


SIMPLE_MD = "# Hello\n\nWorld paragraph.\n\n---\n\nEnd.\n"


@pytest.fixture
def processor(mock_completion):
    return MarkdownProcessor(model="test-model", target_lang="ko")


@pytest.fixture
def source_file(tmp_path):
    p = tmp_path / "source.md"
    p.write_text(SIMPLE_MD, encoding="utf-8")
    return p


@pytest.fixture
def target_file(tmp_path):
    return tmp_path / "target.md"


@pytest.fixture
def po_file(tmp_path):
    return tmp_path / "messages.po"


class TestFullWorkflow:
    def test_process_creates_target(self, processor, source_file, target_file, po_file):
        result = processor.process_document(source_file, target_file, po_file)
        assert target_file.exists()
        content = target_file.read_text(encoding="utf-8")
        assert "[TRANSLATED]" in content

    def test_process_creates_po(self, processor, source_file, target_file, po_file):
        processor.process_document(source_file, target_file, po_file)
        assert po_file.exists()

    def test_result_keys(self, processor, source_file, target_file, po_file):
        result = processor.process_document(source_file, target_file, po_file)
        assert "source_path" in result
        assert "target_path" in result
        assert "po_path" in result
        assert "blocks_count" in result
        assert "coverage" in result
        assert "translation_stats" in result

    def test_stats_keys(self, processor, source_file, target_file, po_file):
        result = processor.process_document(source_file, target_file, po_file)
        stats = result["translation_stats"]
        assert "processed" in stats
        assert "failed" in stats
        assert "skipped" in stats

    def test_hr_skipped(self, processor, source_file, target_file, po_file):
        result = processor.process_document(source_file, target_file, po_file)
        assert result["translation_stats"]["skipped"] >= 1

    def test_po_path_defaults_to_target_with_po_ext(self, processor, source_file, target_file):
        """When po_path is omitted, PO file is created next to target with .po extension."""
        result = processor.process_document(source_file, target_file)
        expected_po = target_file.with_suffix(".po")
        assert expected_po.exists()
        assert result["po_path"] == str(expected_po)


class TestIncrementalProcessing:
    def test_no_reprocessing_unchanged(
        self, processor, source_file, target_file, po_file
    ):
        # First pass
        processor.process_document(source_file, target_file, po_file)

        # Second pass — nothing changed so 0 newly processed
        result = processor.process_document(source_file, target_file, po_file)
        assert result["translation_stats"]["processed"] == 0

    def test_reprocess_changed_block(
        self, processor, source_file, target_file, po_file
    ):
        processor.process_document(source_file, target_file, po_file)

        # Change source file
        source_file.write_text(
            "# Hello\n\nChanged paragraph.\n\n---\n\nEnd.\n", encoding="utf-8"
        )
        result = processor.process_document(source_file, target_file, po_file)
        # The changed paragraph should be reprocessed
        assert result["translation_stats"]["processed"] >= 1


class TestInplaceMode:
    def test_inplace_updates_msgid(self, processor, source_file, target_file, po_file):
        result = processor.process_document(
            source_file, target_file, po_file, inplace=True
        )
        # In inplace mode msgid should be updated to processed text
        po = processor.po_manager.load_or_create_po(po_file)
        for entry in po:
            if entry.msgstr:
                # After inplace, msgid == msgstr
                assert entry.msgid == entry.msgstr


class TestTargetLangFlow:
    def test_target_lang_in_system_message(self, tmp_path, mock_completion):
        """target_lang should appear in the system message sent to the LLM."""
        md = "# Title\n\nParagraph.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")

        processor = MarkdownProcessor(
            model="test-model", target_lang="ko", batch_size=0
        )
        result = processor.process_document(
            source, tmp_path / "target.md", tmp_path / "m.po"
        )
        assert result["translation_stats"]["processed"] >= 1

        # Verify system message contains target lang
        call_args = mock_completion.completion.call_args_list[0]
        messages = call_args.kwargs["messages"]
        system_msg = messages[0]["content"]
        assert "ko" in system_msg


class TestLLMFailureHandling:
    def test_partial_failure_doesnt_crash(self, tmp_path, mock_completion):
        """If LLM fails on one entry, others should still be processed."""
        call_count = 0

        def _failing_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("LLM API error")
            messages = kwargs.get("messages", [])
            source_text = ""
            for msg in reversed(messages):
                if msg["role"] == "user":
                    source_text = msg["content"]
                    break
            mock_response = MagicMock()
            mock_response.choices[0].message.content = f"[TRANSLATED] {source_text}"
            return mock_response

        mock_completion.completion.side_effect = _failing_side_effect

        md = "# Title\n\nPara one.\n\nPara two.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")

        processor = MarkdownProcessor(
            model="test-model", target_lang="ko", batch_size=0
        )
        result = processor.process_document(
            source, tmp_path / "target.md", tmp_path / "m.po"
        )
        stats = result["translation_stats"]
        # At least one failure and one success
        assert stats["failed"] >= 1
        assert stats["processed"] >= 1

    def test_po_saved_on_error(self, tmp_path, mock_completion):
        """PO file should be saved even when processing has errors."""
        mock_completion.completion.side_effect = RuntimeError("fail")

        md = "# Title\n\nParagraph.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")
        po_path = tmp_path / "messages.po"

        processor = MarkdownProcessor(
            model="test-model", target_lang="ko", batch_size=0
        )
        processor.process_document(source, tmp_path / "target.md", po_path)
        # PO should still be saved (finally block)
        assert po_path.exists()


class TestResiduePassIntegration:
    """T-17: opt-in source-language residue post-processing pass.

    Covers the minimal integration touch point called for in the brief:
    the ``residue_pass`` constructor flag actually invokes (or skips)
    the new pass on entries with residue in code spans.
    """

    KOREAN_RESIDUE_MD = "# 한국어 제목\n\n`한국어 함수` 설명.\n"

    def _residue_aware_side_effect(self, on_residue_call):
        """Return a litellm side-effect that distinguishes residue calls.

        The residue pass uses a single-message system + user payload
        (no batched JSON), distinct from both the batched translate
        path (JSON object as user content) and the sequential
        translate path (plain source text).  We tag every call by the
        system prompt's first marker so the test can assert which
        pipeline emitted it.
        """

        def _side_effect(*args, **kwargs):
            messages = kwargs.get("messages", args[0] if args else [])
            user_content = ""
            system_content = ""
            for msg in messages:
                if msg["role"] == "system":
                    system_content = msg["content"]
                elif msg["role"] == "user":
                    user_content = msg["content"]
            mock_response = MagicMock()
            if "repairing a translated" in system_content:
                # Residue pass call — return ASCII-only repair so the
                # placeholder-token count check passes.
                on_residue_call(system_content, user_content)
                if "fenced code block" in system_content:
                    mock_response.choices[0].message.content = (
                        user_content
                        # Replace any Korean characters with ASCII so
                        # the residue detector accepts the result on
                        # any subsequent scan and won't loop.
                    )
                else:
                    mock_response.choices[0].message.content = "REPAIRED"
            else:
                # Initial translate path; for the per-entry source we
                # echo a Korean-bearing translated form so the residue
                # pass has something to detect.  Source content varies
                # between heading / paragraph / code-bearing entries —
                # `[TRANSLATED] {source}` keeps the Korean in place.
                mock_response.choices[0].message.content = (
                    f"[TRANSLATED] {user_content}"
                )
            return mock_response

        return _side_effect

    def test_residue_pass_off_by_default(self, tmp_path):
        source = tmp_path / "source.md"
        source.write_text(self.KOREAN_RESIDUE_MD, encoding="utf-8")
        residue_calls = []
        with patch("mdpo_llm.processor.litellm") as mock_litellm:
            mock_litellm.completion.side_effect = (
                self._residue_aware_side_effect(
                    lambda sysmsg, usr: residue_calls.append((sysmsg, usr))
                )
            )
            mock_litellm.get_supported_openai_params.return_value = []
            processor = MarkdownProcessor(
                model="test-model", target_lang="en", batch_size=0
            )
            processor.process_document(
                source, tmp_path / "target.md", tmp_path / "messages.po"
            )
        assert residue_calls == []

    def test_residue_pass_runs_on_code_only_entries(self, tmp_path):
        """Codex cycle-5 P1: stripping code spans before
        ``detect_languages`` makes a code-only entry look
        language-less.  But code-only entries are exactly what T-17
        is meant to repair — verify the residue pass still triggers
        via the strip-fallback path.
        """
        # Single-paragraph document that's nothing but an inline
        # code span containing source-language characters.  Stripping
        # code spans from the msgid leaves an empty string; the
        # fallback path scans the whole msgid and picks up Korean.
        md = "`한국어`\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")

        residue_calls = []

        def _side_effect(*args, **kwargs):
            messages = kwargs.get("messages", args[0] if args else [])
            user_content = ""
            system_content = ""
            for msg in messages:
                if msg["role"] == "system":
                    system_content = msg["content"]
                elif msg["role"] == "user":
                    user_content = msg["content"]
            mock_response = MagicMock()
            if "repairing a translated" in system_content:
                residue_calls.append(user_content)
                mock_response.choices[0].message.content = "Korean"
            else:
                mock_response.choices[0].message.content = (
                    f"[TRANSLATED] {user_content}"
                )
            return mock_response

        with patch("mdpo_llm.processor.litellm") as mock_litellm:
            mock_litellm.completion.side_effect = _side_effect
            mock_litellm.get_supported_openai_params.return_value = []
            processor = MarkdownProcessor(
                model="test-model",
                target_lang="en",
                batch_size=0,
                residue_pass=True,
            )
            processor.process_document(
                source, tmp_path / "target.md", tmp_path / "messages.po"
            )
        assert len(residue_calls) == 1

    def test_residue_pass_runs_on_kanji_only_japanese_source(self, tmp_path):
        """Codex cycle-5 P2: ``detect_languages`` recognises Japanese
        only via kana, so a kanji-only Japanese source (e.g. ``名前``)
        gets classified as ``zh``.  T-17 still needs to scan for ja
        residue in that case — the processor expands {zh} → {ja, zh}
        so kanji-only Japanese entries with kana / kanji residue in
        their translated code spans get repaired.
        """
        # Paragraph whose natural text is kanji-only (so
        # detect_languages → {"zh"} alone) but whose code span
        # carries leftover katakana the residue pass should pick up
        # only if the {zh}→{ja, zh} expansion actually fires.
        md = "名前 `データ` end.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")

        residue_calls = []

        def _side_effect(*args, **kwargs):
            messages = kwargs.get("messages", args[0] if args else [])
            user_content = ""
            system_content = ""
            for msg in messages:
                if msg["role"] == "system":
                    system_content = msg["content"]
                elif msg["role"] == "user":
                    user_content = msg["content"]
            mock_response = MagicMock()
            if "repairing a translated" in system_content:
                residue_calls.append((system_content, user_content))
                mock_response.choices[0].message.content = "REPAIRED"
            else:
                mock_response.choices[0].message.content = (
                    f"[TRANSLATED] {user_content}"
                )
            return mock_response

        with patch("mdpo_llm.processor.litellm") as mock_litellm:
            mock_litellm.completion.side_effect = _side_effect
            mock_litellm.get_supported_openai_params.return_value = []
            processor = MarkdownProcessor(
                model="test-model",
                target_lang="en",
                batch_size=0,
                residue_pass=True,
            )
            processor.process_document(
                source, tmp_path / "target.md", tmp_path / "messages.po"
            )
        # Residue pass MUST have run on the paragraph entry's
        # `あいう` inline span.  Without the {zh}→{ja, zh} expansion
        # this assertion would fail because detect_languages on
        # ``名前`` returns {"zh"} and ja's hiragana pattern would
        # never be applied.
        assert len(residue_calls) >= 1

    def test_residue_pass_skips_intentional_non_source_cjk_in_code(
        self, tmp_path
    ):
        """Codex cycle-4 P1: an English-source document with an
        intentional CJK code token (e.g. `` `用户.md` ``) must NOT be
        sent through the residue prompts — the source had no CJK
        residue to repair, the token is meaningful content.  The
        msgid-stripped source-lang detection guarantees this:
        ``_strip_code_spans`` removes the code span before language
        detection, so the entry's natural text is identified as
        English-only and ``apply_residue_pass()`` never runs.

        Mixed entries (prose + a CJK code token) are
        intentionally false-negatived rather than false-positived
        because the downstream pass cannot distinguish a legitimate
        identifier from a translation leak.  Glossary placeholder
        mode and ``--placeholder-rules`` are the deterministic
        paths when mixed-entry handling matters.
        """
        md = "# Heading\n\nSee `用户.md` for details.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")

        residue_calls = []

        def _side_effect(*args, **kwargs):
            messages = kwargs.get("messages", args[0] if args else [])
            user_content = ""
            system_content = ""
            for msg in messages:
                if msg["role"] == "system":
                    system_content = msg["content"]
                elif msg["role"] == "user":
                    user_content = msg["content"]
            mock_response = MagicMock()
            if "repairing a translated" in system_content:
                residue_calls.append(user_content)
                mock_response.choices[0].message.content = "USER.md"
            else:
                mock_response.choices[0].message.content = (
                    f"[TRANSLATED] {user_content}"
                )
            return mock_response

        with patch("mdpo_llm.processor.litellm") as mock_litellm:
            mock_litellm.completion.side_effect = _side_effect
            mock_litellm.get_supported_openai_params.return_value = []
            processor = MarkdownProcessor(
                model="test-model",
                target_lang="ko",
                batch_size=0,
                residue_pass=True,
            )
            processor.process_document(
                source, tmp_path / "target.md", tmp_path / "messages.po"
            )
        # The English source had no CJK residue to repair (the CJK
        # token in the code span is intentional content, not a
        # translation leftover).  Zero LLM repair calls.
        assert residue_calls == []

    def test_residue_pass_uses_strict_validator_when_processor_is_strict(
        self, tmp_path
    ):
        """Codex cycle-2 P2: hardcoded ``conservative`` would let a
        residue repair drop / add an inline-code span under
        ``validation="strict"`` — only the strict validator's
        ``inline_code_count`` check would catch that.  Verify the
        residue pass uses the processor's actual validation mode.
        """
        # A paragraph entry with two inline code spans (one Korean).
        # The residue repair will collapse the Korean span into prose
        # text (``Korean fn``), reducing the inline_code count by 1.
        # Strict validation must reject; conservative would accept.
        md = "# Heading\n\nUse `var_a` and `한국어` together.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")

        def _side_effect(*args, **kwargs):
            messages = kwargs.get("messages", args[0] if args else [])
            user_content = ""
            system_content = ""
            for msg in messages:
                if msg["role"] == "system":
                    system_content = msg["content"]
                elif msg["role"] == "user":
                    user_content = msg["content"]
            mock_response = MagicMock()
            if "repairing a translated" in system_content:
                # Drop the backticks in the response so the inline
                # span disappears from the final msgstr.  (We strip
                # the surrounding backticks before returning so the
                # reassembly produces ``Korean fn`` as plain text.)
                mock_response.choices[0].message.content = "Korean fn"
            else:
                mock_response.choices[0].message.content = (
                    f"[TRANSLATED] {user_content}"
                )
            return mock_response

        with patch("mdpo_llm.processor.litellm") as mock_litellm:
            mock_litellm.completion.side_effect = _side_effect
            mock_litellm.get_supported_openai_params.return_value = []
            processor = MarkdownProcessor(
                model="test-model",
                target_lang="en",
                batch_size=0,
                validation="strict",
                residue_pass=True,
            )
            processor.process_document(
                source, tmp_path / "target.md", tmp_path / "messages.po"
            )
            po = processor.po_manager.load_or_create_po(tmp_path / "messages.po")
            para_entry = next(
                e for e in po if "한국어" in e.msgid
            )
            # Strict validator caught the inline-code count mismatch
            # → residue repair was reverted, original msgstr stays
            # (still has both backtick spans, including the residue).
            assert para_entry.msgstr.count("`") == 4

    def test_residue_pass_reverts_repair_that_breaks_fence_count(
        self, tmp_path
    ):
        """A residue repair that strips a fenced delimiter must NOT reach disk.

        Codex cycle-1 P2: ``apply_residue_pass`` only enforces the
        placeholder-token round-trip on the rewritten span. The
        surrounding entry's structural invariants (fence count etc.)
        were validated against the pass-1 ``msgstr`` and aren't
        re-checked anywhere later in the pipeline — so a malformed
        residue repair would otherwise reach the rebuilt markdown.
        ``_run_residue_pass`` runs the conservative validator after
        each repair and reverts on any failure.
        """
        md = "# 한국어 제목\n\n```python\nname = '한국어'\n```\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")

        def _side_effect(*args, **kwargs):
            messages = kwargs.get("messages", args[0] if args else [])
            user_content = ""
            system_content = ""
            for msg in messages:
                if msg["role"] == "system":
                    system_content = msg["content"]
                elif msg["role"] == "user":
                    user_content = msg["content"]
            mock_response = MagicMock()
            if "repairing a translated" in system_content:
                # Sabotage the fenced-block repair: drop the closing
                # fence so the repaired msgstr has fence_count=1
                # instead of the original 2.  The placeholder-token
                # round-trip check would accept this (no tokens
                # involved), so the structural validator is the only
                # safety net.
                mock_response.choices[0].message.content = (
                    "```python\nname = 'name'\n"
                )
            else:
                mock_response.choices[0].message.content = (
                    f"[TRANSLATED] {user_content}"
                )
            return mock_response

        with patch("mdpo_llm.processor.litellm") as mock_litellm:
            mock_litellm.completion.side_effect = _side_effect
            mock_litellm.get_supported_openai_params.return_value = []
            processor = MarkdownProcessor(
                model="test-model",
                target_lang="en",
                batch_size=0,
                residue_pass=True,
            )
            processor.process_document(
                source, tmp_path / "target.md", tmp_path / "messages.po"
            )
            po = processor.po_manager.load_or_create_po(tmp_path / "messages.po")
            # Find the code-block entry; its msgstr must STILL contain
            # both fences because the residue repair was reverted.
            code_entry = next(
                e for e in po if e.msgid.startswith("```python")
            )
            assert code_entry.msgstr.count("```") == 2

    def test_residue_pass_on_invokes_pass(self, tmp_path):
        source = tmp_path / "source.md"
        source.write_text(self.KOREAN_RESIDUE_MD, encoding="utf-8")
        residue_calls = []
        with patch("mdpo_llm.processor.litellm") as mock_litellm:
            mock_litellm.completion.side_effect = (
                self._residue_aware_side_effect(
                    lambda sysmsg, usr: residue_calls.append((sysmsg, usr))
                )
            )
            mock_litellm.get_supported_openai_params.return_value = []
            processor = MarkdownProcessor(
                model="test-model",
                target_lang="en",
                batch_size=0,
                residue_pass=True,
            )
            processor.process_document(
                source, tmp_path / "target.md", tmp_path / "messages.po"
            )
        # The Korean inline code span in the paragraph entry triggers
        # exactly one repair call (heading entry has no code span,
        # so it isn't a residue-pass target).
        assert len(residue_calls) == 1
        # Inline-other prompt was used (not the filename prompt) —
        # the body has whitespace so the filename heuristic rejects it.
        sysmsg, _ = residue_calls[0]
        assert "translate the source-language text" in sysmsg.lower()


class TestExtractBlockType:
    def test_standard_msgctxt(self, mock_completion):
        processor = MarkdownProcessor(model="test-model", target_lang="ko")
        result = processor._extract_block_type_from_msgctxt("intro/setup::para:0")
        assert result == "para"

    def test_heading_type(self, mock_completion):
        processor = MarkdownProcessor(model="test-model", target_lang="ko")
        result = processor._extract_block_type_from_msgctxt("title::heading:0")
        assert result == "heading"

    def test_empty_msgctxt(self, mock_completion):
        processor = MarkdownProcessor(model="test-model", target_lang="ko")
        assert processor._extract_block_type_from_msgctxt("") == ""

    def test_none_msgctxt(self, mock_completion):
        processor = MarkdownProcessor(model="test-model", target_lang="ko")
        assert processor._extract_block_type_from_msgctxt(None) == ""

    def test_no_double_colon(self, mock_completion):
        processor = MarkdownProcessor(model="test-model", target_lang="ko")
        assert processor._extract_block_type_from_msgctxt("nocolon") == ""


class TestGetTranslationStats:
    def test_stats_structure(self, processor, source_file, po_file):
        # Must process first so PO exists
        processor.process_document(
            source_file, source_file.parent / "target.md", po_file
        )
        stats = processor.get_translation_stats(source_file, po_file)
        assert "file_stats" in stats
        assert "coverage" in stats
        assert "po_stats" in stats
        assert "total_lines" in stats["file_stats"]
        assert "total_blocks" in stats["file_stats"]


class TestExportReport:
    def test_report_content(self, processor, source_file, po_file):
        processor.process_document(
            source_file, source_file.parent / "target.md", po_file
        )
        report = processor.export_report(source_file, po_file)
        assert "Translation Report" in report


class TestProcessDirectory:
    """Tests for MarkdownProcessor.process_directory()."""

    def _make_md_file(self, path: Path, content: str = "# Hello\n\nWorld.\n"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def test_po_dir_defaults_to_target_dir(self, tmp_path, mock_completion):
        """When po_dir is omitted, PO files are placed next to target files."""
        src = tmp_path / "src"
        tgt = tmp_path / "tgt"

        self._make_md_file(src / "doc.md", "# Hello\n\nWorld.\n")
        self._make_md_file(src / "sub" / "page.md", "# Sub\n\nNested.\n")

        processor = MarkdownProcessor(model="test-model", target_lang="ko")
        result = processor.process_directory(src, tgt)

        assert result["po_dir"] is None
        # PO files should be next to target files
        assert (tgt / "doc.po").exists()
        assert (tgt / "sub" / "page.po").exists()

    def test_processes_all_md_files(self, tmp_path, mock_completion):
        """Flat directory with multiple .md files — all should be processed."""
        src = tmp_path / "src"
        tgt = tmp_path / "tgt"
        po = tmp_path / "po"

        self._make_md_file(src / "a.md", "# A\n\nAlpha.\n")
        self._make_md_file(src / "b.md", "# B\n\nBravo.\n")
        self._make_md_file(src / "c.md", "# C\n\nCharlie.\n")

        processor = MarkdownProcessor(model="test-model", target_lang="ko")
        result = processor.process_directory(src, tgt, po)

        assert result["files_processed"] + result["files_skipped"] == 3
        assert result["files_failed"] == 0
        assert len(result["results"]) == 3
        # All target files exist
        assert (tgt / "a.md").exists()
        assert (tgt / "b.md").exists()
        assert (tgt / "c.md").exists()

    def test_mirrors_subdirectory_structure(self, tmp_path, mock_completion):
        """Nested directories should be mirrored in target and PO dirs."""
        src = tmp_path / "src"
        tgt = tmp_path / "tgt"
        po = tmp_path / "po"

        self._make_md_file(src / "top.md")
        self._make_md_file(src / "sub" / "nested.md")

        processor = MarkdownProcessor(model="test-model", target_lang="ko")
        processor.process_directory(src, tgt, po)

        assert (tgt / "top.md").exists()
        assert (tgt / "sub" / "nested.md").exists()

    def test_po_uses_po_extension(self, tmp_path, mock_completion):
        """PO files should use .po extension, not .md."""
        src = tmp_path / "src"
        tgt = tmp_path / "tgt"
        po = tmp_path / "po"

        self._make_md_file(src / "doc.md")
        self._make_md_file(src / "sub" / "page.md")

        processor = MarkdownProcessor(model="test-model", target_lang="ko")
        processor.process_directory(src, tgt, po)

        assert (po / "doc.po").exists()
        assert (po / "sub" / "page.po").exists()
        # No .md files in PO dir
        assert not list(po.glob("**/*.md"))

    def test_custom_glob_pattern(self, tmp_path, mock_completion):
        """Non-recursive glob should only match top-level files."""
        src = tmp_path / "src"
        tgt = tmp_path / "tgt"
        po = tmp_path / "po"

        self._make_md_file(src / "top.md")
        self._make_md_file(src / "sub" / "nested.md")

        processor = MarkdownProcessor(model="test-model", target_lang="ko")
        result = processor.process_directory(src, tgt, po, glob="*.md")

        # Only top-level file matched
        total = result["files_processed"] + result["files_skipped"]
        assert total == 1
        assert (tgt / "top.md").exists()
        assert not (tgt / "sub" / "nested.md").exists()

    def test_empty_directory(self, tmp_path, mock_completion):
        """Empty directory returns zero counts."""
        src = tmp_path / "src"
        src.mkdir()
        tgt = tmp_path / "tgt"
        po = tmp_path / "po"

        processor = MarkdownProcessor(model="test-model", target_lang="ko")
        result = processor.process_directory(src, tgt, po)

        assert result["files_processed"] == 0
        assert result["files_failed"] == 0
        assert result["files_skipped"] == 0
        assert result["results"] == []

    def test_single_file_failure_continues(self, tmp_path, mock_completion):
        """One file failing should not stop processing of remaining files."""
        src = tmp_path / "src"
        tgt = tmp_path / "tgt"
        po = tmp_path / "po"

        self._make_md_file(src / "good.md", "# Good\n\nContent.\n")
        # Create a file that will cause process_document to fail
        bad = src / "bad.md"
        bad.parent.mkdir(parents=True, exist_ok=True)
        bad.write_text("# Bad\n\nBad content.\n", encoding="utf-8")

        processor = MarkdownProcessor(model="test-model", target_lang="ko")

        original_process = processor.process_document

        def patched_process(source_path, target_path, po_path, inplace=False):
            if source_path.name == "bad.md":
                raise RuntimeError("Simulated failure")
            return original_process(source_path, target_path, po_path, inplace=inplace)

        with patch.object(processor, "process_document", side_effect=patched_process):
            result = processor.process_directory(src, tgt, po)

        assert result["files_failed"] == 1
        assert result["files_processed"] + result["files_skipped"] >= 1
        assert len(result["results"]) == 2

    def test_inplace_mode_forwarded(self, tmp_path, mock_completion):
        """The inplace flag should be forwarded to process_document."""
        src = tmp_path / "src"
        tgt = tmp_path / "tgt"
        po = tmp_path / "po"

        self._make_md_file(src / "doc.md", "# Title\n\nSome text.\n")

        processor = MarkdownProcessor(model="test-model", target_lang="ko")

        with patch.object(
            processor, "process_document", wraps=processor.process_document
        ) as mock_pd:
            processor.process_directory(src, tgt, po, inplace=True)
            mock_pd.assert_called_once()
            _, kwargs = mock_pd.call_args
            assert kwargs.get("inplace") is True or mock_pd.call_args[0][-1] is True

    def test_return_value_structure(self, tmp_path, mock_completion):
        """Return dict should contain all expected keys."""
        src = tmp_path / "src"
        tgt = tmp_path / "tgt"
        po = tmp_path / "po"
        src.mkdir()

        processor = MarkdownProcessor(model="test-model", target_lang="ko")
        result = processor.process_directory(src, tgt, po)

        assert result["source_dir"] == str(src)
        assert result["target_dir"] == str(tgt)
        assert result["po_dir"] == str(po)
        assert "files_processed" in result
        assert "files_failed" in result
        assert "files_skipped" in result
        assert "results" in result

    def test_max_workers_parameter(self, tmp_path, mock_completion):
        """max_workers parameter is accepted and produces correct results."""
        src = tmp_path / "src"
        tgt = tmp_path / "tgt"
        po = tmp_path / "po"

        self._make_md_file(src / "a.md", "# A\n\nAlpha.\n")
        self._make_md_file(src / "b.md", "# B\n\nBravo.\n")

        processor = MarkdownProcessor(model="test-model", target_lang="ko")
        result = processor.process_directory(src, tgt, po, max_workers=2)

        assert result["files_processed"] + result["files_skipped"] == 2
        assert result["files_failed"] == 0


class TestSequentialProcessing:
    """Tests for sequential entry processing with reference context."""

    def test_entries_processed_in_document_order(self, tmp_path, mock_completion):
        """LLM calls should happen in document order."""
        md = "# First\n\nSecond paragraph.\n\nThird paragraph.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")

        processor = MarkdownProcessor(
            model="test-model", target_lang="ko", batch_size=0
        )
        processor.process_document(source, tmp_path / "target.md", tmp_path / "m.po")

        # Extract user messages from call args (last user message in each call)
        call_order = []
        for call in mock_completion.completion.call_args_list:
            messages = call.kwargs["messages"]
            # Last message is always the source text
            call_order.append(messages[-1]["content"])

        assert len(call_order) >= 3
        assert call_order[0] == "# First"
        assert call_order[1] == "Second paragraph."
        assert call_order[2] == "Third paragraph."

    def test_reference_pairs_grow_over_run(self, tmp_path, mock_completion):
        """First entry should get no reference pairs; later entries should get some."""
        md = "# Title\n\nParagraph one.\n\nParagraph two.\n\nParagraph three.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")

        processor = MarkdownProcessor(
            model="test-model", target_lang="ko", batch_size=0
        )
        processor.process_document(source, tmp_path / "target.md", tmp_path / "m.po")

        # First call: system + user (2 messages)
        # Later calls should have more messages due to few-shot pairs
        call_counts = []
        for call in mock_completion.completion.call_args_list:
            messages = call.kwargs["messages"]
            call_counts.append(len(messages))

        # First call has fewest messages (no reference pairs)
        assert call_counts[0] == 2
        # Later calls should have more (reference pairs add messages)
        assert any(c > 2 for c in call_counts[1:])

    def test_existing_po_seeds_pool(self, tmp_path, mock_completion):
        """Second run after source edit should seed pool from existing translations."""
        md = "# Title\n\nOriginal paragraph.\n\nAnother paragraph.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")
        po_path = tmp_path / "m.po"

        processor = MarkdownProcessor(
            model="test-model", target_lang="ko", batch_size=0
        )
        # First run — translates everything
        processor.process_document(source, tmp_path / "target.md", po_path)

        mock_completion.completion.reset_mock()

        # Edit source — change one paragraph
        source.write_text(
            "# Title\n\nChanged paragraph.\n\nAnother paragraph.\n", encoding="utf-8"
        )
        # Second run — only the changed paragraph needs translation,
        # but the pool is seeded from existing PO translations
        processor.process_document(source, tmp_path / "target.md", po_path)

        # The changed paragraph should have been processed
        assert mock_completion.completion.call_count >= 1

    def test_max_reference_pairs_constructor_arg(self, tmp_path, mock_completion):
        """max_reference_pairs should limit the number of pairs passed."""
        # Many paragraphs to ensure pool grows
        lines = ["# Title\n"]
        for i in range(10):
            lines.append(f"\nParagraph {i}.\n")
        md = "".join(lines)

        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")

        processor = MarkdownProcessor(
            model="test-model",
            target_lang="ko",
            max_reference_pairs=2,
            batch_size=0,
        )
        processor.process_document(source, tmp_path / "target.md", tmp_path / "m.po")

        for call in mock_completion.completion.call_args_list:
            messages = call.kwargs["messages"]
            # Count user/assistant pairs (excluding system and final user)
            pair_messages = messages[1:-1]  # exclude system and final user
            # Each pair = 2 messages, so at most 4 pair messages for max_reference_pairs=2
            assert len(pair_messages) <= 4


class TestRefineGlossaryFilter:
    """T-13: refine mode keeps glossary entries that can only preserve.

    Mapped entries (target-language injections) are dropped as before;
    null-entries and identity mappings are kept so the placeholder
    registry can tokenize them and protect against whitespace reflow
    during refine.
    """

    def test_null_entry_kept_in_refine_mode(self, mock_completion):
        processor = MarkdownProcessor(
            model="test-model",
            target_lang="ko",
            mode="refine",
            glossary={"게임코드": None},
            glossary_mode="placeholder",
        )
        assert processor._glossary == {"게임코드": None}

    def test_mapped_entry_dropped_in_refine_mode(self, mock_completion):
        processor = MarkdownProcessor(
            model="test-model",
            target_lang="ko",
            mode="refine",
            glossary={"pull request": "풀 리퀘스트"},
            glossary_mode="placeholder",
        )
        assert processor._glossary is None

    def test_identity_mapping_kept_in_refine_mode(self, mock_completion):
        processor = MarkdownProcessor(
            model="test-model",
            target_lang="ko",
            mode="refine",
            glossary={"API": "API"},
            glossary_mode="placeholder",
        )
        assert processor._glossary == {"API": "API"}

    def test_mixed_glossary_filters_to_preserving_entries(self, mock_completion):
        processor = MarkdownProcessor(
            model="test-model",
            target_lang="ko",
            mode="refine",
            glossary={
                "게임코드": None,
                "pull request": "풀 리퀘스트",
                "API": "API",
            },
            glossary_mode="placeholder",
        )
        assert processor._glossary == {"게임코드": None, "API": "API"}

    def test_per_locale_resolves_to_none_kept(self, tmp_path, mock_completion):
        # Per-locale dicts live in the JSON file path — the inline
        # ``glossary=`` kwarg is typed as ``{term: str | None}`` so
        # per-locale values go through ``glossary_path`` instead.
        gpath = tmp_path / "g.json"
        gpath.write_text(
            json.dumps({"API": {"ko": None, "ja": "API"}}),
            encoding="utf-8",
        )
        processor = MarkdownProcessor(
            model="test-model",
            target_lang="ko",
            mode="refine",
            glossary_path=gpath,
            glossary_mode="placeholder",
        )
        assert processor._glossary == {"API": None}

    def test_per_locale_resolves_to_identity_kept(self, tmp_path, mock_completion):
        gpath = tmp_path / "g.json"
        gpath.write_text(
            json.dumps({"API": {"ko": None, "ja": "API"}}),
            encoding="utf-8",
        )
        processor = MarkdownProcessor(
            model="test-model",
            target_lang="ja",
            mode="refine",
            glossary_path=gpath,
            glossary_mode="placeholder",
        )
        assert processor._glossary == {"API": "API"}

    def test_per_locale_resolves_to_other_language_dropped(
        self, tmp_path, mock_completion
    ):
        gpath = tmp_path / "g.json"
        gpath.write_text(
            json.dumps({"API": {"ko": "에이피아이", "ja": "API"}}),
            encoding="utf-8",
        )
        processor = MarkdownProcessor(
            model="test-model",
            target_lang="ko",
            mode="refine",
            glossary_path=gpath,
            glossary_mode="placeholder",
        )
        assert processor._glossary is None

    def test_translate_mode_unaffected(self, mock_completion):
        """Filter is refine-only; translate mode must keep mapped entries."""
        processor = MarkdownProcessor(
            model="test-model",
            target_lang="ko",
            mode="translate",
            glossary={"pull request": "풀 리퀘스트", "게임코드": None},
            glossary_mode="placeholder",
        )
        assert processor._glossary == {
            "pull request": "풀 리퀘스트",
            "게임코드": None,
        }

    def test_registry_registers_preserved_entry_in_placeholder_mode(
        self, mock_completion
    ):
        """Kept null-entries surface on the placeholder registry so refine
        tokenizes the span and decode emits it verbatim."""
        processor = MarkdownProcessor(
            model="test-model",
            target_lang="ko",
            mode="refine",
            glossary={"게임코드": None},
            glossary_mode="placeholder",
        )
        names = [p.name for p in processor._effective_registry.patterns]
        assert "glossary:게임코드" in names

    def test_refine_preserves_null_entry_verbatim_end_to_end(
        self, tmp_path, mock_completion
    ):
        """Refine pass tokenizes the null-entry term and decodes it back
        verbatim — the LLM never sees the literal `게임코드`, so it cannot
        reflow the whitespace inside it."""
        md = "# Title\n\nDescribe 게임코드 behaviour.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")

        processor = MarkdownProcessor(
            model="test-model",
            target_lang="ko",
            mode="refine",
            glossary={"게임코드": None},
            glossary_mode="placeholder",
            batch_size=0,
        )
        refined = tmp_path / "refined.md"
        processor.process_document(
            source,
            tmp_path / "target.md",
            tmp_path / "m.po",
            refined_path=refined,
        )
        assert "게임코드" in refined.read_text(encoding="utf-8")
        # Confirm the LLM received the tokenized form, not the raw term —
        # this is the mechanism by which whitespace inside the identifier
        # is protected from refine-time reflow.
        user_contents = [
            msg["content"]
            for call in mock_completion.completion.call_args_list
            for msg in call.kwargs["messages"]
            if msg["role"] == "user"
        ]
        assert any("게임코드" not in c and "⟦P:" in c for c in user_contents)

    def test_refine_drops_mapped_entry_no_injection(
        self, tmp_path, mock_completion
    ):
        """Mapped entries are dropped — the LLM sees the raw source and
        no target-language injection can happen via placeholder decode."""
        md = "# Title\n\nOpen a pull request to merge.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")

        processor = MarkdownProcessor(
            model="test-model",
            target_lang="ko",
            mode="refine",
            glossary={"pull request": "풀 리퀘스트"},
            glossary_mode="placeholder",
            batch_size=0,
        )
        refined = tmp_path / "refined.md"
        processor.process_document(
            source,
            tmp_path / "target.md",
            tmp_path / "m.po",
            refined_path=refined,
        )
        output = refined.read_text(encoding="utf-8")
        assert "풀 리퀘스트" not in output

    def test_sibling_refine_processor_forwards_preserve_only_entries(
        self, mock_completion
    ):
        """``refine_first=True`` builds its refine pass via
        ``_sibling_refine_processor``.  The sibling must inherit
        null-entries and identity mappings from the translate-pass
        glossary (so refine can tokenize and protect them), but mapped
        entries that would inject target-language text must be dropped.
        """
        parent = MarkdownProcessor(
            model="test-model",
            target_lang="ko",
            mode="translate",
            glossary={
                "게임코드": None,
                "pull request": "풀 리퀘스트",
                "API": "API",
            },
            glossary_mode="placeholder",
        )
        sibling = parent._sibling_refine_processor(target_lang="en")
        assert sibling.mode == "refine"
        assert sibling._glossary == {"게임코드": None, "API": "API"}

    def test_sibling_refine_processor_uses_tls_raw_chain(
        self, tmp_path, mock_completion
    ):
        """When ``process_directory(refine_first=True)`` installs the
        per-file raw chain on ``self._tls.refine_sibling_raw``, the
        refine sibling must resolve it for ``refine_lang`` and apply the
        refine filter — NOT fall back to constructor-level state.
        Without this the sibling would miss every per-directory
        ``glossary.json`` cascade entry that the translate pass already
        honours, leaving identifiers exposed to whitespace reflow during
        refine.
        """
        parent = MarkdownProcessor(
            model="test-model",
            target_lang="en",
            mode="translate",
            glossary={"only-constructor": None},
            glossary_mode="placeholder",
        )
        # Simulate the raw merged chain a cascade walk would hand to
        # the worker: per-locale dict for "API" plus a cascade-only
        # null-entry for "게임코드".  The constructor-level term must
        # NOT appear in the sibling glossary — the TLS raw chain wins
        # so the sibling sees exactly what the cascade has merged for
        # this file.
        parent._tls.refine_sibling_raw = {
            "API": {"ko": None, "en": "API"},
            "게임코드": None,
        }
        try:
            sibling = parent._sibling_refine_processor(target_lang="ko")
        finally:
            del parent._tls.refine_sibling_raw
        # Under refine_lang=ko: API resolves to None (kept),
        # 게임코드 is None (kept).  The constructor's "only-constructor"
        # is deliberately absent because the TLS raw chain takes
        # precedence — the cascade already folded constructor overrides
        # into the chain upstream (in ``_merged_raw_chain_for_file``).
        assert sibling._glossary == {"API": None, "게임코드": None}

    def test_sibling_refine_processor_empty_tls_chain_overrides_constructor(
        self, mock_completion
    ):
        """Edge case: a per-directory ``__remove__`` can collapse the
        per-file cascade to an empty map.  When the caller installs an
        explicitly empty ``_tls.refine_sibling_raw``, the sibling MUST
        treat that as "this file has no glossary" — NOT fall back to
        constructor-level state.  Otherwise refine would still protect
        a term the translate pass correctly ignores, breaking
        per-directory opt-outs for `refine_first=True` runs.
        """
        parent = MarkdownProcessor(
            model="test-model",
            target_lang="en",
            mode="translate",
            glossary={"constructor-term": None, "API": "API"},
            glossary_mode="placeholder",
        )
        parent._tls.refine_sibling_raw = {}
        try:
            sibling = parent._sibling_refine_processor(target_lang="ko")
        finally:
            del parent._tls.refine_sibling_raw
        assert sibling._glossary is None

    def test_sibling_refine_processor_filters_mapped_entries_from_tls(
        self, tmp_path, mock_completion
    ):
        """TLS raw chain path must still drop mapped entries in the
        sibling — the filter is applied after per-locale collapse.
        """
        parent = MarkdownProcessor(
            model="test-model",
            target_lang="en",
            mode="translate",
            glossary_mode="placeholder",
        )
        parent._tls.refine_sibling_raw = {
            "API": {"ko": None, "en": "API"},
            "pull request": "풀 리퀘스트",
        }
        try:
            sibling = parent._sibling_refine_processor(target_lang="ko")
        finally:
            del parent._tls.refine_sibling_raw
        assert sibling._glossary == {"API": None}

    def test_sibling_refine_processor_collapses_per_locale_file_glossary(
        self, tmp_path, mock_completion
    ):
        """Per-locale dicts from ``glossary_path`` must be collapsed to
        the refine ``target_lang`` BEFORE the refine filter runs — the
        sibling needs one scalar per term to test against the identity
        rule.
        """
        gpath = tmp_path / "g.json"
        gpath.write_text(
            json.dumps(
                {
                    "API": {"ko": None, "ja": "API", "en": "API"},
                    "checkout": {"ko": "체크아웃", "en": None},
                }
            ),
            encoding="utf-8",
        )
        parent = MarkdownProcessor(
            model="test-model",
            target_lang="ko",
            mode="translate",
            glossary_path=gpath,
            glossary_mode="placeholder",
        )
        sibling_en = parent._sibling_refine_processor(target_lang="en")
        # API(en)=="API" identity kept; checkout(en)=None kept.
        assert sibling_en._glossary == {"API": "API", "checkout": None}

        sibling_ja = parent._sibling_refine_processor(target_lang="ja")
        # API(ja)=="API" identity kept; checkout(ja) falls through to
        # None via ``dict.get``, so it's kept as do-not-translate.
        assert sibling_ja._glossary == {"API": "API", "checkout": None}

    def test_refine_then_translate_chain_preserves_null_entry(
        self, tmp_path, mock_completion
    ):
        """End-to-end: a refine pass feeding a translate pass, both
        configured with the same glossary, keeps a null-entry identifier
        intact across both stages.  This is the core regression T-13
        fixes — before the filter, refine would drop the glossary and
        risk reflowing `게임코드` into `게임 코드`, breaking the
        translate pass's glossary match."""
        glossary = {"게임코드": None}
        md = "# Title\n\nThe 게임코드 appears here.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")

        refine_proc = MarkdownProcessor(
            model="test-model",
            target_lang="ko",
            mode="refine",
            glossary=glossary,
            glossary_mode="placeholder",
            batch_size=0,
        )
        refined = tmp_path / "refined.md"
        refine_proc.process_document(
            source,
            tmp_path / "target-noop.md",
            tmp_path / "refine.po",
            refined_path=refined,
        )
        assert "게임코드" in refined.read_text(encoding="utf-8")

        translate_proc = MarkdownProcessor(
            model="test-model",
            target_lang="ko",
            mode="translate",
            glossary=glossary,
            glossary_mode="placeholder",
            batch_size=0,
        )
        translated = tmp_path / "translated.md"
        translate_proc.process_document(
            refined, translated, tmp_path / "translate.po"
        )
        assert "게임코드" in translated.read_text(encoding="utf-8")


class TestAutoBracketPlaceholders:
    """T-14: auto-register source-language bracket placeholders."""

    def test_auto_register_defaults_on_and_tokenizes_non_ascii_bracket(
        self, mock_completion
    ):
        proc = MarkdownProcessor(model="test-model", target_lang="en")
        assert proc.auto_bracket_placeholders is True
        source = "UI label <\uc804\uc1a1> triggers submit."
        encoded, mapping = proc._encode_source(source)
        assert "<\uc804\uc1a1>" not in encoded
        assert any(
            p.pattern_name == "auto_bracket_angle" and p.original == "<\uc804\uc1a1>"
            for p in mapping
        )

    def test_auto_register_tokenizes_brace_url_parameter(self, mock_completion):
        proc = MarkdownProcessor(model="test-model", target_lang="en")
        source = "path /users/{\uac8c\uc784\ucf54\ub4dc}/profile"
        encoded, mapping = proc._encode_source(source)
        assert "{\uac8c\uc784\ucf54\ub4dc}" not in encoded
        assert any(
            p.pattern_name == "auto_bracket_brace"
            and p.original == "{\uac8c\uc784\ucf54\ub4dc}"
            for p in mapping
        )

    def test_auto_register_leaves_ascii_brackets_alone_for_latin_target(
        self, mock_completion
    ):
        # For a Latin/ASCII target (``en``) the detection rule is
        # "content has a word char NOT in the Latin/ASCII range", so
        # pure-ASCII brackets are target-script and flow through the
        # translate prompt.  Under a CJK target the same brackets
        # would count as non-target-script identifiers and DO match
        # — that case is covered by
        # :meth:`test_auto_register_respects_target_script_for_cjk_target`.
        proc = MarkdownProcessor(model="test-model", target_lang="en")
        source = "path /{page_id}/edit uses {template}."
        encoded, mapping = proc._encode_source(source)
        auto_matches = [
            p for p in mapping if p.pattern_name.startswith("auto_bracket_")
        ]
        assert auto_matches == []
        assert encoded == source

    def test_auto_register_leaves_mustache_templates_alone(self, mock_completion):
        # ``{{한글}}`` would match a naive inner ``{한글}`` pattern; the
        # lookbehind / lookahead guard keeps the runtime template intact
        # so the template engine still sees its own delimiters.
        proc = MarkdownProcessor(model="test-model", target_lang="en")
        source = "template {{\ud55c\uae00}} substitution"
        encoded, mapping = proc._encode_source(source)
        auto_matches = [
            p for p in mapping if p.pattern_name.startswith("auto_bracket_")
        ]
        assert auto_matches == []
        assert encoded == source

    def test_auto_register_opt_out_via_kwarg(self, mock_completion):
        proc = MarkdownProcessor(
            model="test-model",
            target_lang="en",
            auto_bracket_placeholders=False,
        )
        assert proc.auto_bracket_placeholders is False
        source = "Do not tokenize <\uc804\uc1a1> here."
        encoded, mapping = proc._encode_source(source)
        auto_matches = [
            p for p in mapping if p.pattern_name.startswith("auto_bracket_")
        ]
        assert auto_matches == []
        assert encoded == source

    def test_env_var_overrides_kwarg_to_disable(self, monkeypatch, mock_completion):
        monkeypatch.setenv("MDPO_AUTO_BRACKET_PLACEHOLDERS", "0")
        proc = MarkdownProcessor(
            model="test-model",
            target_lang="en",
            auto_bracket_placeholders=True,
        )
        assert proc.auto_bracket_placeholders is False

    def test_env_var_overrides_kwarg_to_enable(self, monkeypatch, mock_completion):
        monkeypatch.setenv("MDPO_AUTO_BRACKET_PLACEHOLDERS", "yes")
        proc = MarkdownProcessor(
            model="test-model",
            target_lang="en",
            auto_bracket_placeholders=False,
        )
        assert proc.auto_bracket_placeholders is True

    def test_env_var_unrecognised_value_falls_back_to_kwarg(
        self, monkeypatch, mock_completion
    ):
        # Typos in the env var must not silently flip the setting — the
        # constructor kwarg stays authoritative unless the env value is
        # a recognised on/off keyword.
        monkeypatch.setenv("MDPO_AUTO_BRACKET_PLACEHOLDERS", "maybe")
        proc_on = MarkdownProcessor(
            model="test-model",
            target_lang="en",
            auto_bracket_placeholders=True,
        )
        assert proc_on.auto_bracket_placeholders is True
        proc_off = MarkdownProcessor(
            model="test-model",
            target_lang="en",
            auto_bracket_placeholders=False,
        )
        assert proc_off.auto_bracket_placeholders is False

    def test_glossary_entry_wins_over_auto_register(self, mock_completion):
        # When a caller-supplied glossary term covers the inner span of a
        # bracket, glossary wins: auto-register defers so the glossary
        # pattern gets to tokenize (and, for mapped entries, decode to
        # the target form).  Without this the outer auto-bracket span
        # would swallow the inner glossary match via overlap resolution.
        proc = MarkdownProcessor(
            model="test-model",
            target_lang="en",
            glossary={"\uac8c\uc784\ucf54\ub4dc": None},
            glossary_mode="placeholder",
        )
        source = "use {\uac8c\uc784\ucf54\ub4dc} to identify the entry"
        encoded, mapping = proc._encode_source(source)
        names = [p.pattern_name for p in mapping]
        # Glossary tokenized the bare term; auto-bracket stayed out.
        assert any(n == "glossary:\uac8c\uc784\ucf54\ub4dc" for n in names)
        assert not any(n.startswith("auto_bracket_") for n in names)
        assert "{\uac8c\uc784\ucf54\ub4dc}" not in encoded
        # Brackets survive in the encoded text; only the inner term is a token.
        assert "{" in encoded and "}" in encoded

    def test_unrelated_glossary_entries_do_not_block_auto_register(
        self, mock_completion
    ):
        # A glossary term that does NOT appear inside the bracket must
        # not prevent auto-register from firing — glossary-defers is
        # scoped to the specific bracket span.
        proc = MarkdownProcessor(
            model="test-model",
            target_lang="en",
            glossary={"unrelated": None},
            glossary_mode="placeholder",
        )
        source = "prefix <\uc804\uc1a1> suffix"
        encoded, mapping = proc._encode_source(source)
        assert "<\uc804\uc1a1>" not in encoded
        assert any(p.pattern_name == "auto_bracket_angle" for p in mapping)

    def test_round_trip_after_llm_preserves_token(self, mock_completion):
        # End-to-end with the mock LLM: the mocked completion prefixes
        # ``[TRANSLATED] `` onto whatever the user content was — since
        # the content is already the encoded source with a placeholder
        # token, the token survives the round trip and decode restores
        # the original bracket.
        proc = MarkdownProcessor(model="test-model", target_lang="en", batch_size=0)
        source = "See <\uc804\uc1a1> label."
        result = proc._call_llm(source)
        assert "<\uc804\uc1a1>" in result
        assert result.startswith("[TRANSLATED] ")

    def test_user_pattern_wins_tie_with_auto_register(self, mock_completion):
        # A user-supplied pattern whose match exactly coincides with an
        # auto-bracket match gets priority via registration order (user
        # patterns first, auto-register last), so the ``pattern_name``
        # on the resulting placeholder comes from the user.
        user_reg = PlaceholderRegistry()
        user_reg.register("explicit_angle", r"<[^<>]+>")
        proc = MarkdownProcessor(
            model="test-model",
            target_lang="en",
            placeholders=user_reg,
        )
        source = "See <\uc804\uc1a1> label."
        _, mapping = proc._encode_source(source)
        assert len(mapping) == 1
        assert mapping.items[0].pattern_name == "explicit_angle"

    def test_instruction_glossary_null_entry_is_protected_by_auto_register(
        self, mock_completion
    ):
        # Regression guard for a cycle-1 P2 finding: in instruction
        # mode ``_build_effective_registry`` does NOT register
        # ``glossary:<term>`` patterns (only placeholder mode does).
        # For null-entries the caller's intent is "preserve verbatim",
        # which is exactly what auto-bracket provides — so it must
        # fire instead of deferring to nothing.
        proc = MarkdownProcessor(
            model="test-model",
            target_lang="en",
            glossary={"\uac8c\uc784\ucf54\ub4dc": None},
            glossary_mode="instruction",
        )
        source = "use {\uac8c\uc784\ucf54\ub4dc} to identify the entry"
        encoded, mapping = proc._encode_source(source)
        names = [p.pattern_name for p in mapping]
        assert any(n == "auto_bracket_brace" for n in names)
        assert "{\uac8c\uc784\ucf54\ub4dc}" not in encoded

    def test_instruction_glossary_mapped_entry_defers_auto_register(
        self, mock_completion
    ):
        # Regression guard for a cycle-7 P1 finding: an explicit
        # ``"\uac8c\uc784\ucf54\ub4dc" -> "GameCode"`` mapping in
        # instruction mode means the LLM should SEE the source term
        # and render the mapping.  Auto-bracketing the whole
        # ``{\uac8c\uc784\ucf54\ub4dc}`` span would freeze the source
        # and silently prevent the caller's translation instruction
        # from applying.  The predicate must defer to mapped-entry
        # terms so the glossary prompt block can drive the output.
        proc = MarkdownProcessor(
            model="test-model",
            target_lang="en",
            glossary={"\uac8c\uc784\ucf54\ub4dc": "GameCode"},
            glossary_mode="instruction",
        )
        source = "use {\uac8c\uc784\ucf54\ub4dc} to identify the entry"
        encoded, mapping = proc._encode_source(source)
        names = [p.pattern_name for p in mapping]
        assert not any(n.startswith("auto_bracket_") for n in names)
        assert encoded == source

    def test_instruction_glossary_identity_mapping_is_protected(
        self, mock_completion
    ):
        # Identity mappings (``"API" -> "API"``) also have a
        # "preserve verbatim" effect in instruction mode, so they
        # stay in the auto-bracket protection path instead of
        # deferring.  A bracketed identity-mapped term with an
        # otherwise non-target-script context still tokenizes.
        proc = MarkdownProcessor(
            model="test-model",
            target_lang="en",
            glossary={"\uac8c\uc784\ucf54\ub4dc": "\uac8c\uc784\ucf54\ub4dc"},
            glossary_mode="instruction",
        )
        source = "use {\uac8c\uc784\ucf54\ub4dc} here"
        _, mapping = proc._encode_source(source)
        assert any(
            p.pattern_name == "auto_bracket_brace" for p in mapping
        )

    def test_registry_cache_key_includes_instruction_mode_mapped_terms(
        self, mock_completion
    ):
        # Regression guard for a cycle-9 P2 finding: the auto-bracket
        # predicate now depends on the mapped-term subset of the
        # active glossary even in instruction mode (cycle-7 fix).
        # ``_registry_for_glossary`` used to cache every instruction-
        # mode registry under ``None`` so two per-file glossaries
        # that differ ONLY in their mappings would share the wrong
        # cached registry during a ``process_directory`` run — a
        # mapped term from one file would suppress auto-bracket in
        # another file, or vice versa.
        proc = MarkdownProcessor(
            model="test-model",
            target_lang="en",
            glossary_mode="instruction",
        )
        reg_mapped = proc._registry_for_glossary(
            {"\uac8c\uc784\ucf54\ub4dc": "GameCode"}
        )
        reg_null = proc._registry_for_glossary(
            {"\uac8c\uc784\ucf54\ub4dc": None}
        )
        # Different glossaries → distinct cached registries (by
        # identity, because the cache key set differs).  If the fix
        # were missing, both keys would collapse to ``None`` and the
        # second call would return the same registry as the first.
        assert reg_mapped is not reg_null

    def test_multi_target_fallback_resolves_glossary_for_fallback_lang(
        self, mock_completion
    ):
        # Regression guard for a cycle-19 P1 finding: when the
        # multi-target fallback rebuilds its placeholder registry
        # it must resolve the glossary for the FALLBACK language,
        # not the processor's constructor locale.  Locale-specific
        # mappings (``{"게임코드": {"en": null, "ja": "GameCode"}}``)
        # otherwise silently drop out of the fallback's defer set
        # and the ja pass never sees the raw term needed to apply
        # the mapping.
        proc = MarkdownProcessor(
            model="test-model",
            target_lang="en",
            glossary={"\uac8c\uc784\ucf54\ub4dc": {"en": None, "ja": "GameCode"}},
            glossary_mode="instruction",
        )
        result = proc._call_lang_single(
            "use {\uac8c\uc784\ucf54\ub4dc} here", target_lang="ja"
        )
        # The ja fallback's predicate defers for the mapped term,
        # so auto-bracket does NOT fire.  The mock returns the
        # bracketed source unchanged (prefixed with ``[TRANSLATED] ``)
        # because the encoded text equals the raw source.
        assert result is not None
        assert "{\uac8c\uc784\ucf54\ub4dc}" in result
        # And the en fallback's null-entry keeps auto-bracket
        # protection, so the bracketed term is tokenized before the
        # mock sees it and then decoded back — end-to-end
        # preservation.
        result_en = proc._call_lang_single(
            "use {\uac8c\uc784\ucf54\ub4dc} here", target_lang="en"
        )
        assert result_en is not None
        assert "{\uac8c\uc784\ucf54\ub4dc}" in result_en

    def test_multi_target_per_lang_fallback_uses_per_lang_registry(
        self, mock_completion
    ):
        # Regression guard for a cycle-18 P1 finding: the union
        # registry installed for the shared-encoding path leaks
        # into the per-lang fallback (``_call_lang_single``) via
        # TLS.  That would cause an English fallback on an
        # ``[en, ko]`` run to still union ``ko``'s Hangul into the
        # target-script, so ``{\uc804\uc1a1}`` would reach the
        # model unprotected even though a direct ``target_lang="en"``
        # call would tokenize it.  The fallback must swap TLS to a
        # per-lang registry for the duration of the call.
        proc = MarkdownProcessor(model="test-model", target_lang="en")
        # Install the multi-target union registry to simulate what
        # process_document_multi does, then call the fallback.
        proc._tls.per_file_registry = proc._build_effective_registry(
            proc._placeholders,
            glossary=None,
            update_builtin_overrides=False,
            target_langs_override=["en", "ko"],
        )
        try:
            result = proc._call_lang_single(
                "use {\uc804\uc1a1} here", target_lang="en"
            )
        finally:
            del proc._tls.per_file_registry
        # Fallback ran with the per-lang English registry, where
        # Hangul IS non-target-script and ``{\uc804\uc1a1}`` gets
        # tokenized before the mock ``litellm.completion`` sees it.
        # The mock returns ``[TRANSLATED] `` + the encoded text,
        # which should then decode back to include the original
        # ``{\uc804\uc1a1}`` verbatim.
        assert result is not None
        assert "{\uc804\uc1a1}" in result

    def test_multi_target_unions_mapped_glossary_across_langs(
        self, mock_completion
    ):
        # Regression guard for a cycle-10 P1 finding: the shared
        # predicate in ``process_document_multi`` must defer for a
        # term that is MAPPED in ANY requested lang, not just the
        # constructor locale.  A per-locale glossary like
        # ``{"\uac8c\uc784\ucf54\ub4dc": {"en": None, "ja": "GameCode"}}``
        # says "preserve for en, map for ja"; auto-bracket must
        # defer so the ja pass sees the raw term and can apply the
        # mapping, even though the constructor was built for ``en``.
        proc = MarkdownProcessor(
            model="test-model",
            target_lang="en",
            glossary={"\uac8c\uc784\ucf54\ub4dc": {"en": None, "ja": "GameCode"}},
            glossary_mode="instruction",
        )
        # Replicate the multi-target install: collect mapped terms
        # across langs and hand them to ``_build_effective_registry``
        # the same way ``process_document_multi`` does.
        raw_chain = dict(proc._glossary_inline or {})
        defer: list = []
        for lang in ("en", "ja"):
            resolved = proc._resolve_raw_for_lang(raw_chain, lang)
            for t, v in resolved.items():
                if v is not None and v != t:
                    defer.append(t)
        proc._tls.per_file_registry = proc._build_effective_registry(
            proc._placeholders,
            glossary=None,
            update_builtin_overrides=False,
            target_langs_override=["en", "ja"],
            auto_bracket_defer_terms=defer,
        )
        try:
            # The mapped term appears only in ja; the shared encoding
            # must defer auto-bracket so the ja pass can render the
            # GameCode mapping.
            source = "use {\uac8c\uc784\ucf54\ub4dc} here"
            encoded, mapping = proc._encode_source(source)
            assert not any(
                p.pattern_name.startswith("auto_bracket_") for p in mapping
            )
            assert encoded == source
        finally:
            del proc._tls.per_file_registry

    def test_multi_target_encoding_unions_script_ranges(
        self, mock_completion, tmp_path
    ):
        # Regression guard for a cycle-9 P1 finding: shared encoding
        # must protect neither source-Hangul nor source-Latin so
        # each per-lang pass (en, ko) can translate / refine its
        # target-script content.  Verified by installing the multi-
        # target registry on the TLS slot directly, then encoding.
        proc = MarkdownProcessor(
            model="test-model",
            target_lang="en",
            glossary_mode="instruction",
        )
        # ``_build_effective_registry`` with the multi-target
        # override is what ``process_document_multi`` installs on
        # TLS; use the same hook to verify end-to-end encoding
        # behaviour without running the full save loop.
        proc._tls.per_file_registry = proc._build_effective_registry(
            proc._placeholders,
            glossary=None,
            update_builtin_overrides=False,
            target_langs_override=["en", "ko"],
        )
        try:
            # Bare-Latin / bare-Hangul: both target-script for one of
            # the langs, so neither auto-brackets.
            for source in (
                "see {game_id} here",
                "see {\uc804\uc1a1} here",
            ):
                _, mapping = proc._encode_source(source)
                assert not any(
                    p.pattern_name.startswith("auto_bracket_") for p in mapping
                ), source
            # Cross-script content (Japanese Hiragana, Cyrillic) is
            # non-target-script for both ko and en → still tokenized.
            _, mapping = proc._encode_source(
                "see {\u3053\u3093\u306b\u3061\u306f} and {\u0441\u043b\u043e\u0432\u043e}"
            )
            auto = [
                p.original for p in mapping
                if p.pattern_name.startswith("auto_bracket_")
            ]
            assert "{\u3053\u3093\u306b\u3061\u306f}" in auto
            assert "{\u0441\u043b\u043e\u0432\u043e}" in auto
        finally:
            del proc._tls.per_file_registry

    def test_auto_register_respects_target_script_for_cjk_target(
        self, mock_completion
    ):
        # Regression guard for a cycle-6 P1 finding: a CJK target
        # must not freeze CJK-only bracket spans — otherwise the
        # refine pass never polishes them and a ko→ja translate
        # pass freezes target-script brackets that should re-render.
        # Mixed-script or ASCII-only bracket content still tokenizes
        # because those are legitimate non-target-script identifiers.
        proc = MarkdownProcessor(model="test-model", target_lang="ko")
        source_cjk_only = "prefix {\uc804\uc1a1} suffix"
        _, mapping = proc._encode_source(source_cjk_only)
        assert not any(
            p.pattern_name.startswith("auto_bracket_") for p in mapping
        )

        source_mixed = "prefix {id_\uac8c\uc784\ucf54\ub4dc} suffix"
        _, mapping = proc._encode_source(source_mixed)
        assert any(
            p.pattern_name == "auto_bracket_brace"
            and p.original == "{id_\uac8c\uc784\ucf54\ub4dc}"
            for p in mapping
        )

    def test_auto_register_refine_mode_on_korean_leaves_cjk_brackets(
        self, mock_completion
    ):
        # End-to-end: refine mode with Korean source/target must
        # leave ``{전송}`` / ``<다음>`` for the refine prompt.
        # Before target-script gating was introduced these spans
        # were frozen and the refiner never saw them — that broke
        # the refine contract for any non-ASCII source language.
        proc = MarkdownProcessor(
            model="test-model", target_lang="ko", mode="refine"
        )
        source = "Label <\ub2e4\uc74c> button and {\uc804\uc1a1} action."
        _, mapping = proc._encode_source(source)
        assert not any(
            p.pattern_name.startswith("auto_bracket_") for p in mapping
        )

    def test_refine_first_sibling_inherits_auto_bracket_setting(
        self, mock_completion
    ):
        # ``refine_first=True`` builds a refine-mode sibling processor;
        # forgetting to propagate ``auto_bracket_placeholders`` would
        # leave refine unprotected while translate was protected, so
        # the two passes would disagree about which spans survive
        # verbatim — exactly the divergence T-13 was built to prevent.
        parent = MarkdownProcessor(
            model="test-model",
            target_lang="en",
            auto_bracket_placeholders=False,
        )
        sibling = parent._sibling_refine_processor(target_lang="en")
        assert sibling.auto_bracket_placeholders is False


def _system_messages_from_calls(mock_completion):
    """Return the system-message string from every captured LLM call."""
    out = []
    for call in mock_completion.completion.call_args_list:
        messages = call.kwargs.get("messages") or (
            call.args[0] if call.args else []
        )
        for msg in messages:
            if msg.get("role") == "system":
                content = msg["content"]
                if isinstance(content, list):
                    # Anthropic prompt-cache shape: [{"type": "text",
                    # "text": "...", "cache_control": ...}]
                    parts = [p.get("text", "") for p in content]
                    content = "".join(parts)
                out.append(content)
                break
    return out


class TestContextInjection:
    """T-18: free-text domain context injection via --context (cascade).

    Covers the end-to-end wiring on top of the unit tests in
    :mod:`tests.test_context_loader`: the constructor reads
    ``context_path``, every translate / refine / validate / multi-target
    prompt assembler picks up the resolved block via
    :meth:`MarkdownProcessor._current_context_text`, and the per-file
    cascade in :meth:`process_directory` walks parent → child.
    """

    HEADER = (
        "**ADDITIONAL CONTEXT (use for proper nouns, terminology, "
        "tone, audience):**"
    )

    def test_constructor_context_path_appears_in_system_prompt(
        self, tmp_path, mock_completion
    ):
        ctx = tmp_path / "brief.md"
        ctx.write_text(
            "Domain: game-security SDK. Audience: senior backend engineers.",
            encoding="utf-8",
        )
        source = tmp_path / "source.md"
        source.write_text("# Title\n\nParagraph.\n", encoding="utf-8")

        processor = MarkdownProcessor(
            model="test-model",
            target_lang="ko",
            batch_size=0,
            context_path=ctx,
        )
        processor.process_document(
            source, tmp_path / "target.md", tmp_path / "m.po"
        )

        systems = _system_messages_from_calls(mock_completion)
        assert systems, "expected at least one LLM call"
        for sys in systems:
            assert self.HEADER in sys
            assert "game-security SDK" in sys

    def test_no_context_means_no_header(self, tmp_path, mock_completion):
        """Without ``--context``, the header MUST NOT appear — otherwise
        a no-config run would pay tokens for an empty block."""
        source = tmp_path / "source.md"
        source.write_text("# Title\n\nParagraph.\n", encoding="utf-8")

        processor = MarkdownProcessor(
            model="test-model", target_lang="ko", batch_size=0
        )
        processor.process_document(
            source, tmp_path / "target.md", tmp_path / "m.po"
        )

        systems = _system_messages_from_calls(mock_completion)
        assert systems
        for sys in systems:
            assert self.HEADER not in sys

    def test_missing_context_path_silent_skip(self, tmp_path, mock_completion):
        """Brief: missing files are silently skipped at every level —
        a typo in ``--context`` should not abort the run, just fall back
        to the empty-context path."""
        source = tmp_path / "source.md"
        source.write_text("# Title\n\nParagraph.\n", encoding="utf-8")

        processor = MarkdownProcessor(
            model="test-model",
            target_lang="ko",
            batch_size=0,
            context_path=tmp_path / "does-not-exist.md",
        )
        # Run does not raise.
        processor.process_document(
            source, tmp_path / "target.md", tmp_path / "m.po"
        )
        systems = _system_messages_from_calls(mock_completion)
        for sys in systems:
            assert self.HEADER not in sys

    def test_empty_context_file_silent_skip(self, tmp_path, mock_completion):
        ctx = tmp_path / "empty.md"
        ctx.write_text("", encoding="utf-8")
        source = tmp_path / "source.md"
        source.write_text("# Title\n\nParagraph.\n", encoding="utf-8")

        processor = MarkdownProcessor(
            model="test-model",
            target_lang="ko",
            batch_size=0,
            context_path=ctx,
        )
        processor.process_document(
            source, tmp_path / "target.md", tmp_path / "m.po"
        )
        systems = _system_messages_from_calls(mock_completion)
        for sys in systems:
            assert self.HEADER not in sys

    def test_directory_cascade_parent_then_child(
        self, tmp_path, mock_completion
    ):
        source_dir = tmp_path / "docs"
        api_dir = source_dir / "api"
        api_dir.mkdir(parents=True, exist_ok=True)
        (source_dir / "context.md").write_text(
            "ROOT-LEVEL-DOMAIN", encoding="utf-8"
        )
        (api_dir / "context.md").write_text(
            "API-SECTION-CONTEXT", encoding="utf-8"
        )
        leaf_md = api_dir / "auth.md"
        leaf_md.write_text("# Auth\n\nSign in.\n", encoding="utf-8")

        target_dir = tmp_path / "out"
        po_dir = tmp_path / "po"

        processor = MarkdownProcessor(
            model="test-model", target_lang="ko", batch_size=0
        )
        processor.process_directory(source_dir, target_dir, po_dir)

        systems = _system_messages_from_calls(mock_completion)
        assert systems
        # Both blocks appear, parent before child.
        sys = systems[0]
        assert "ROOT-LEVEL-DOMAIN" in sys
        assert "API-SECTION-CONTEXT" in sys
        assert sys.index("ROOT-LEVEL-DOMAIN") < sys.index(
            "API-SECTION-CONTEXT"
        )

    def test_cli_override_appended_after_cascade(
        self, tmp_path, mock_completion
    ):
        source_dir = tmp_path / "docs"
        source_dir.mkdir(parents=True, exist_ok=True)
        (source_dir / "context.md").write_text(
            "TREE-CONTEXT", encoding="utf-8"
        )
        leaf_md = source_dir / "page.md"
        leaf_md.write_text("# Page\n\nBody.\n", encoding="utf-8")

        cli_ctx = tmp_path / "cli.md"
        cli_ctx.write_text("OVERRIDE-CONTEXT", encoding="utf-8")

        processor = MarkdownProcessor(
            model="test-model",
            target_lang="ko",
            batch_size=0,
            context_path=cli_ctx,
        )
        processor.process_directory(
            source_dir, tmp_path / "out", tmp_path / "po"
        )

        sys = _system_messages_from_calls(mock_completion)[0]
        assert "TREE-CONTEXT" in sys
        assert "OVERRIDE-CONTEXT" in sys
        # Override is the closest layer — appended LAST in the
        # accumulated context block.
        assert sys.rindex("OVERRIDE-CONTEXT") > sys.rindex(
            "TREE-CONTEXT"
        )

    def test_refine_mode_includes_context(
        self, tmp_path, mock_completion
    ):
        ctx = tmp_path / "brief.md"
        ctx.write_text(
            "Style: tutorial voice; keep code tokens verbatim.",
            encoding="utf-8",
        )
        source = tmp_path / "source.md"
        source.write_text("# Heading\n\nBody text.\n", encoding="utf-8")

        processor = MarkdownProcessor(
            model="test-model",
            target_lang="en",
            batch_size=0,
            mode="refine",
            context_path=ctx,
        )
        processor.process_document(
            source,
            tmp_path / "refined.md",
            tmp_path / "m.po",
            refined_path=tmp_path / "refined.md",
        )

        systems = _system_messages_from_calls(mock_completion)
        assert systems
        for sys in systems:
            assert self.HEADER in sys
            assert "tutorial voice" in sys

    def test_validator_prompt_includes_context(
        self, tmp_path, mock_completion
    ):
        """When ``validation='llm'`` the validator MUST grade against the
        same domain framing the translator saw, so the context block
        appears in the validator system prompt too."""
        ctx = tmp_path / "brief.md"
        ctx.write_text(
            "Domain: payment infrastructure. Tone: precise.",
            encoding="utf-8",
        )

        # Override mock_completion to return both translations and
        # validator grades. We detect validator calls by their system
        # prompt's "validator" wording.
        def _side_effect(*args, **kwargs):
            messages = kwargs.get("messages", args[0] if args else [])
            system_text = ""
            user_text = ""
            for msg in messages:
                if msg.get("role") == "system":
                    sc = msg["content"]
                    if isinstance(sc, list):
                        sc = "".join(p.get("text", "") for p in sc)
                    system_text = sc
                if msg.get("role") == "user":
                    user_text = msg["content"]

            mock_response = MagicMock()
            if "validator" in system_text.lower():
                # Validator: pass everything.
                try:
                    payload = json.loads(user_text)
                except json.JSONDecodeError:
                    payload = {}
                grades = {
                    k: {"binary_score": "yes", "reason": "ok"}
                    for k in payload
                }
                mock_response.choices[0].message.content = json.dumps(grades)
                return mock_response
            # Translator: prefix the source with a Korean glyph so the
            # target-language structural check passes and the LLM
            # validator actually fires (a Latin-only response would be
            # rejected by the conservative pre-gate before grading).
            try:
                parsed = json.loads(user_text)
                if isinstance(parsed, dict):
                    out = {k: f"번역: {v}" for k, v in parsed.items()}
                    mock_response.choices[0].message.content = json.dumps(
                        out, ensure_ascii=False
                    )
                    return mock_response
            except json.JSONDecodeError:
                pass
            mock_response.choices[0].message.content = (
                f"번역: {user_text}"
            )
            return mock_response

        mock_completion.completion.side_effect = _side_effect

        source = tmp_path / "source.md"
        source.write_text("# Title\n\nParagraph.\n", encoding="utf-8")

        processor = MarkdownProcessor(
            model="test-model",
            target_lang="ko",
            batch_size=10,
            validation="llm",
            max_retries=0,
            context_path=ctx,
        )
        processor.process_document(
            source, tmp_path / "target.md", tmp_path / "m.po"
        )

        # Find the validator system message specifically and verify it
        # also carries the context block.
        validator_systems = [
            sys
            for sys in _system_messages_from_calls(mock_completion)
            if "validator" in sys.lower()
        ]
        assert validator_systems, (
            "expected at least one validator call under validation='llm'"
        )
        for sys in validator_systems:
            assert self.HEADER in sys
            assert "payment infrastructure" in sys

    def test_multi_target_shares_context_across_langs(
        self, tmp_path, mock_completion
    ):
        ctx = tmp_path / "brief.md"
        ctx.write_text(
            "Domain: developer documentation; keep API names verbatim.",
            encoding="utf-8",
        )
        source = tmp_path / "source.md"
        source.write_text("# Title\n\nBody.\n", encoding="utf-8")

        processor = MarkdownProcessor(
            model="test-model",
            target_lang="ko",
            batch_size=10,
            glossary_mode="instruction",
            context_path=ctx,
        )

        # The mock_completion fixture round-trips JSON; for multi-
        # target the wire format is {block_id: {lang: text}}, so wrap
        # the translation map per lang.
        def _multi_side_effect(*args, **kwargs):
            messages = kwargs.get("messages", args[0] if args else [])
            user_text = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    user_text = msg["content"]
                    break
            mock_response = MagicMock()
            try:
                payload = json.loads(user_text)
            except json.JSONDecodeError:
                payload = {}
            if isinstance(payload, dict):
                out = {
                    k: {"ko": f"[ko] {v}", "ja": f"[ja] {v}"}
                    for k, v in payload.items()
                }
                mock_response.choices[0].message.content = json.dumps(
                    out, ensure_ascii=False
                )
            else:
                mock_response.choices[0].message.content = "{}"
            return mock_response

        mock_completion.completion.side_effect = _multi_side_effect

        processor.process_document_multi(
            source,
            target_langs=["ko", "ja"],
            target_paths={
                "ko": tmp_path / "ko.md",
                "ja": tmp_path / "ja.md",
            },
        )

        # The single multi-target system prompt carries the shared
        # context block — same brief flows to every lang's output bucket.
        systems = _system_messages_from_calls(mock_completion)
        assert systems
        assert any(self.HEADER in sys for sys in systems)
        assert any("developer documentation" in sys for sys in systems)

    def test_missing_per_directory_context_does_not_break_run(
        self, tmp_path, mock_completion
    ):
        """A directory tree without any ``context.md`` files runs
        identically to the no-context path — the cascade walk should
        not read every parent dir into a "(empty)" placeholder."""
        source_dir = tmp_path / "docs"
        sub = source_dir / "api"
        sub.mkdir(parents=True, exist_ok=True)
        (sub / "page.md").write_text("# Page\n\nBody.\n", encoding="utf-8")

        processor = MarkdownProcessor(
            model="test-model", target_lang="ko", batch_size=0
        )
        processor.process_directory(
            source_dir, tmp_path / "out", tmp_path / "po"
        )
        systems = _system_messages_from_calls(mock_completion)
        assert systems
        for sys in systems:
            assert self.HEADER not in sys

    def test_context_md_excluded_from_directory_glob(
        self, tmp_path, mock_completion
    ):
        """``context.md`` is cascade configuration — process_directory
        MUST NOT pick it up via the default ``**/*.md`` glob and try
        to translate the brief itself."""
        source_dir = tmp_path / "docs"
        sub = source_dir / "api"
        sub.mkdir(parents=True, exist_ok=True)
        (source_dir / "context.md").write_text("ROOT-CTX", encoding="utf-8")
        (sub / "context.md").write_text("API-CTX", encoding="utf-8")
        (sub / "auth.md").write_text(
            "# Auth\n\nSign in.\n", encoding="utf-8"
        )

        processor = MarkdownProcessor(
            model="test-model", target_lang="ko", batch_size=0
        )
        result = processor.process_directory(
            source_dir, tmp_path / "out", tmp_path / "po"
        )

        # Only auth.md is translatable; the two context.md files are
        # excluded so files_processed reflects exactly the one
        # document the user wanted translated.
        assert (tmp_path / "out" / "api" / "auth.md").exists()
        assert not (tmp_path / "out" / "context.md").exists()
        assert not (tmp_path / "out" / "api" / "context.md").exists()
        # Defensive: total files seen by workers excludes config too.
        assert (
            result.files_processed
            + result.files_failed
            + result.files_skipped
        ) == 1

    def test_single_file_does_not_walk_directory_cascade(
        self, tmp_path, mock_completion
    ):
        """Single-file ``process_document`` MUST NOT walk a per-directory
        cascade — that's a ``process_directory`` feature (parity with
        glossary). A ``context.md`` next to the source file should be
        ignored unless the caller explicitly passed ``--context PATH``."""
        source_dir = tmp_path / "docs"
        source_dir.mkdir(parents=True, exist_ok=True)
        (source_dir / "context.md").write_text(
            "FILE-DIR-CONTEXT", encoding="utf-8"
        )
        source = source_dir / "page.md"
        source.write_text("# Title\n\nBody.\n", encoding="utf-8")

        processor = MarkdownProcessor(
            model="test-model", target_lang="ko", batch_size=0
        )
        processor.process_document(
            source, tmp_path / "target.md", tmp_path / "m.po"
        )

        systems = _system_messages_from_calls(mock_completion)
        assert systems
        for sys in systems:
            assert "FILE-DIR-CONTEXT" not in sys
            assert self.HEADER not in sys

    def test_default_glob_skips_context_md_even_when_alone_in_tree(
        self, tmp_path, mock_completion
    ):
        """The auto-skip MUST be driven by the caller's glob pattern,
        not by which files happen to be in the tree right now: a small
        directory whose only ``.md`` files are ``context.md`` configs
        and where the caller used the default broad ``**/*.md`` MUST
        leave them alone — the user never opted into translating
        configuration."""
        source_dir = tmp_path / "docs"
        sub = source_dir / "api"
        sub.mkdir(parents=True, exist_ok=True)
        (source_dir / "context.md").write_text("ROOT-CTX", encoding="utf-8")
        (sub / "context.md").write_text("API-CTX", encoding="utf-8")
        # No other .md files in the tree at all.

        processor = MarkdownProcessor(
            model="test-model", target_lang="ko", batch_size=0
        )
        result = processor.process_directory(
            source_dir, tmp_path / "out", tmp_path / "po"
        )

        # Default glob is ``**/*.md`` — broad — so context.md files
        # are filtered out even though they are the only matches.
        assert not (tmp_path / "out" / "context.md").exists()
        assert not (tmp_path / "out" / "api" / "context.md").exists()
        assert (
            result.files_processed
            + result.files_failed
            + result.files_skipped
        ) == 0

    def test_explicit_context_glob_translates_context_md(
        self, tmp_path, mock_completion
    ):
        """When the caller's glob targets ``context.md`` exclusively,
        the directory loop respects intent and translates the files —
        the auto-skip only applies when the glob ALSO matches non-
        context content (i.e. the configs are incidental)."""
        source_dir = tmp_path / "docs"
        api_dir = source_dir / "api"
        api_dir.mkdir(parents=True, exist_ok=True)
        (source_dir / "context.md").write_text(
            "# Root context heading\n", encoding="utf-8"
        )
        (api_dir / "context.md").write_text(
            "# API context heading\n", encoding="utf-8"
        )
        # No other .md files in the tree — the explicit glob below
        # only matches the two context files.

        processor = MarkdownProcessor(
            model="test-model", target_lang="ko", batch_size=0
        )
        processor.process_directory(
            source_dir,
            tmp_path / "out",
            tmp_path / "po",
            glob="**/context.md",
        )

        # Both context files were translated and written under the
        # output tree — the user's explicit glob is honoured.
        assert (tmp_path / "out" / "context.md").exists()
        assert (tmp_path / "out" / "api" / "context.md").exists()

    def test_cache_key_isolates_distinct_runs(
        self, tmp_path, mock_completion
    ):
        """Reusing one processor across two ``process_directory`` calls
        with DIFFERENT trees must serve each tree's own cascade — the
        per-file context cache key MUST include source_root so the
        second run does not echo the first's brief."""
        # Tree A
        tree_a = tmp_path / "a"
        (tree_a / "sub").mkdir(parents=True, exist_ok=True)
        (tree_a / "context.md").write_text("BRIEF-A", encoding="utf-8")
        (tree_a / "sub" / "page.md").write_text(
            "# Title\n\nBody.\n", encoding="utf-8"
        )
        # Tree B
        tree_b = tmp_path / "b"
        (tree_b / "sub").mkdir(parents=True, exist_ok=True)
        (tree_b / "context.md").write_text("BRIEF-B", encoding="utf-8")
        (tree_b / "sub" / "page.md").write_text(
            "# Title\n\nBody.\n", encoding="utf-8"
        )

        processor = MarkdownProcessor(
            model="test-model", target_lang="ko", batch_size=0
        )
        processor.process_directory(
            tree_a, tmp_path / "out_a", tmp_path / "po_a"
        )
        a_systems = _system_messages_from_calls(mock_completion)
        assert any("BRIEF-A" in s for s in a_systems)

        # Reset captured calls between runs.
        mock_completion.completion.call_args_list.clear()
        processor.process_directory(
            tree_b, tmp_path / "out_b", tmp_path / "po_b"
        )
        b_systems = _system_messages_from_calls(mock_completion)
        # The second tree's calls carry the second tree's brief —
        # the cache must NOT serve BRIEF-A here just because the
        # source filename collides with the first run.
        assert any("BRIEF-B" in s for s in b_systems)
        for s in b_systems:
            assert "BRIEF-A" not in s

    def test_explicit_context_glob_basename_normalises_separators(self):
        """The basename-detection step that decides whether a glob
        explicitly targets ``context.md`` MUST normalise both
        forward- and back-slash separators so Windows-shaped patterns
        (``docs\\\\context.md``) are recognised the same way
        forward-slash forms (``docs/context.md``) are. The full glob
        expansion can't be exercised cross-platform from one test
        environment (pathlib rejects ``**\\\\…`` on POSIX), so we
        verify the detection step in isolation."""
        # Mirror the literal expression used in
        # ``process_directory`` so a refactor of either side trips
        # this test.
        for raw in ("**/context.md", "docs/context.md", "context.md"):
            assert raw.replace("\\", "/").rsplit("/", 1)[-1] == "context.md"
        for raw in (
            "**\\context.md",
            "docs\\context.md",
            "docs\\sub\\context.md",
        ):
            assert raw.replace("\\", "/").rsplit("/", 1)[-1] == "context.md"
        # Broad globs MUST stay broad regardless of separator style.
        for raw in (
            "**/*.md",
            "**\\*.md",
            "*.md",
            "docs/*.md",
            "docs\\*.md",
        ):
            assert raw.replace("\\", "/").rsplit("/", 1)[-1] != "context.md"

    def test_translate_paths_sees_tree_level_context(
        self, tmp_path, mock_completion
    ):
        """``translate_paths=True`` runs filename-segment translation
        BEFORE per-file workers fan out. The tree-level cascade
        (source root's ``context.md`` plus cwd + override) must be
        installed for that phase too — otherwise filename translations
        drift from the content's terminology guidance."""
        source_dir = tmp_path / "docs"
        sub = source_dir / "api"
        sub.mkdir(parents=True, exist_ok=True)
        (source_dir / "context.md").write_text(
            "TREE-LEVEL-BRIEF-FOR-PATHS", encoding="utf-8"
        )
        (sub / "page.md").write_text(
            "# Title\n\nBody.\n", encoding="utf-8"
        )

        processor = MarkdownProcessor(
            model="test-model", target_lang="ko", batch_size=0
        )
        processor.process_directory(
            source_dir,
            tmp_path / "out",
            tmp_path / "po",
            translate_paths=True,
        )

        systems = _system_messages_from_calls(mock_completion)
        # Both the path-segment LLM call(s) AND the per-file body
        # call carry the tree-level brief. Without the segment-phase
        # TLS install, only the body call would have it.
        assert systems
        with_brief = [
            s for s in systems if "TREE-LEVEL-BRIEF-FOR-PATHS" in s
        ]
        # At minimum: one segment call + one body call = >= 2 prompts
        # carrying the brief. Allow more if the segment translator
        # bisects internally.
        assert len(with_brief) >= 2, (
            f"expected the brief in both segment and body prompts; "
            f"got {len(with_brief)} of {len(systems)}"
        )

    def test_refine_first_sibling_sees_directory_cascade(
        self, tmp_path, mock_completion
    ):
        """``refine_first`` builds a refine sibling whose own TLS does
        not inherit the parent's per-file cascade. Without the
        sibling-context propagation in :meth:`_sibling_refine_processor`,
        the refine pass would translate against a context-blind prompt
        while the translate pass (running on the parent) sees the full
        cascaded brief — silent divergence between stages."""
        source_dir = tmp_path / "docs"
        source_dir.mkdir(parents=True, exist_ok=True)
        (source_dir / "context.md").write_text(
            "REFINE-FIRST-DIRECTORY-BRIEF", encoding="utf-8"
        )
        (source_dir / "page.md").write_text(
            "# Title\n\nBody.\n", encoding="utf-8"
        )

        refined_dir = tmp_path / "refined"
        target_dir = tmp_path / "out"

        processor = MarkdownProcessor(
            model="test-model",
            target_lang="ko",
            batch_size=0,
            mode="translate",
        )
        processor.process_directory(
            source_dir,
            target_dir,
            tmp_path / "po",
            refined_dir=refined_dir,
            refine_first=True,
            refine_lang="en",
            refined_po_dir=tmp_path / "po-refine",
        )

        systems = _system_messages_from_calls(mock_completion)
        # Both refine and translate stages issue calls; both should
        # carry the same brief. Find at least two system messages
        # that contain the brief — separate stages.
        with_brief = [
            s for s in systems if "REFINE-FIRST-DIRECTORY-BRIEF" in s
        ]
        assert len(with_brief) >= 2, (
            f"expected refine and translate stages to share the cascaded "
            f"brief; got {len(with_brief)} of {len(systems)} calls "
            f"carrying it"
        )

    def test_extra_instructions_does_not_collide_with_context(
        self, tmp_path, mock_completion
    ):
        """``--extra-instructions`` (existing flag) and ``--context``
        (T-18) are independent — both should appear, neither should
        overwrite the other."""
        ctx = tmp_path / "brief.md"
        ctx.write_text("CONTEXT-MARKER", encoding="utf-8")
        source = tmp_path / "source.md"
        source.write_text("# Title\n\nBody.\n", encoding="utf-8")

        processor = MarkdownProcessor(
            model="test-model",
            target_lang="ko",
            batch_size=0,
            extra_instructions="EXTRA-INSTRUCTION-MARKER",
            context_path=ctx,
        )
        processor.process_document(
            source, tmp_path / "target.md", tmp_path / "m.po"
        )

        sys = _system_messages_from_calls(mock_completion)[0]
        assert "EXTRA-INSTRUCTION-MARKER" in sys
        assert "CONTEXT-MARKER" in sys
        assert self.HEADER in sys


class TestNoTranslateBlocks:
    """Pipeline-level checks for the `no_translate` HTML-comment fence.

    ``SKIP_TYPES`` membership means the translate, refine, residue, and
    validator stages never see these blocks. The LLM mock would otherwise
    prefix every msgstr with ``[TRANSLATED] ``; absence of that prefix
    inside the no_translate region proves the short-circuit is wired up.
    """

    NO_TRANSLATE_INCLUDED = "no_translate" in MarkdownProcessor.SKIP_TYPES

    def test_skip_types_constant(self):
        assert self.NO_TRANSLATE_INCLUDED, (
            "MarkdownProcessor.SKIP_TYPES must include 'no_translate' so the "
            "translate / refine / residue / validate stages skip it."
        )

    def test_form_a_block_round_trips_verbatim(self, tmp_path, mock_completion):
        """End-to-end translate: a `<!-- mdpo:no-translate -->` range is
        copied unchanged into the target file, and its PO entry has empty
        ``msgstr`` (matching the hr precedent in manager.sync_po)."""
        md = (
            "# Title\n"
            "\n"
            "Intro paragraph.\n"
            "\n"
            "<!-- mdpo:no-translate -->\n"
            "Verbatim body.\n"
            "Stays in source language.\n"
            "<!-- /mdpo:no-translate -->\n"
            "\n"
            "Trailing paragraph.\n"
        )
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")
        target = tmp_path / "target.md"
        po_path = tmp_path / "m.po"

        processor = MarkdownProcessor(
            model="test-model", target_lang="ko", batch_size=0
        )
        processor.process_document(source, target, po_path)

        produced = target.read_text(encoding="utf-8")
        # The no_translate range survives verbatim, markers and all.
        assert "<!-- mdpo:no-translate -->\nVerbatim body." in produced
        assert "Stays in source language." in produced
        assert "<!-- /mdpo:no-translate -->" in produced
        # The LLM mock prefixes every translated string with [TRANSLATED];
        # that prefix must NOT appear inside the protected body.
        assert "[TRANSLATED] Verbatim body" not in produced
        assert "[TRANSLATED] Stays in source" not in produced
        # Surrounding paragraphs DID go through the translator.
        assert "[TRANSLATED] Intro paragraph." in produced
        assert "[TRANSLATED] Trailing paragraph." in produced

        # PO entry for the no_translate block exists with empty msgstr.
        po = processor.po_manager.load_or_create_po(po_path)
        no_translate_entries = [e for e in po if "no_translate" in (e.msgctxt or "")]
        assert no_translate_entries, "Expected a PO entry for the no_translate block."
        assert all(e.msgstr == "" for e in no_translate_entries)

    def test_form_b_skip_next_round_trips_verbatim(self, tmp_path, mock_completion):
        md = (
            "<!-- mdpo:skip-next -->\n"
            "Skipped paragraph.\n"
            "\n"
            "Translated paragraph.\n"
        )
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")
        target = tmp_path / "target.md"
        po_path = tmp_path / "m.po"

        processor = MarkdownProcessor(
            model="test-model", target_lang="ko", batch_size=0
        )
        processor.process_document(source, target, po_path)

        produced = target.read_text(encoding="utf-8")
        assert "<!-- mdpo:skip-next -->\nSkipped paragraph." in produced
        assert "[TRANSLATED] Skipped paragraph" not in produced
        assert "[TRANSLATED] Translated paragraph." in produced

    def test_refine_mode_leaves_no_translate_untouched(self, tmp_path, mock_completion):
        """Refine mode also short-circuits on SKIP_TYPES — the protected
        body must never reach the LLM."""
        md = (
            "Intro.\n"
            "\n"
            "<!-- mdpo:no-translate -->\n"
            "PROTECTED-MARKER body.\n"
            "<!-- /mdpo:no-translate -->\n"
            "\n"
            "Outro.\n"
        )
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")
        refined = tmp_path / "refined.md"

        processor = MarkdownProcessor(
            model="test-model",
            target_lang="ko",
            mode="refine",
            batch_size=0,
        )
        processor.process_document(
            source,
            tmp_path / "target.md",
            tmp_path / "m.po",
            refined_path=refined,
        )

        produced = refined.read_text(encoding="utf-8")
        assert "PROTECTED-MARKER body." in produced
        # No translator prefix on the protected body — refine never saw it.
        assert "[TRANSLATED] PROTECTED-MARKER" not in produced
        # And the LLM never received the protected content as input.
        user_contents = [
            msg["content"]
            for call in mock_completion.completion.call_args_list
            for msg in call.kwargs["messages"]
            if msg["role"] == "user"
        ]
        assert not any("PROTECTED-MARKER" in c for c in user_contents)
