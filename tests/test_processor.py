"""Tests for MarkdownProcessor end-to-end and error handling."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

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
