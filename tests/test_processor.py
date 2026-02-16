"""Tests for MarkdownProcessor end-to-end and error handling."""

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

        processor = MarkdownProcessor(model="test-model", target_lang="ko")
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

        processor = MarkdownProcessor(model="test-model", target_lang="ko")
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

        processor = MarkdownProcessor(model="test-model", target_lang="ko")
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

        processor = MarkdownProcessor(model="test-model", target_lang="ko")
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

        processor = MarkdownProcessor(model="test-model", target_lang="ko")
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

        processor = MarkdownProcessor(model="test-model", target_lang="ko")
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
            model="test-model", target_lang="ko", max_reference_pairs=2
        )
        processor.process_document(source, tmp_path / "target.md", tmp_path / "m.po")

        for call in mock_completion.completion.call_args_list:
            messages = call.kwargs["messages"]
            # Count user/assistant pairs (excluding system and final user)
            pair_messages = messages[1:-1]  # exclude system and final user
            # Each pair = 2 messages, so at most 4 pair messages for max_reference_pairs=2
            assert len(pair_messages) <= 4
