"""Tests for MarkdownProcessor end-to-end and error handling."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mdpo_llm.llm_interface import LLMInterface, MockLLMInterface
from mdpo_llm.processor import MarkdownProcessor


SIMPLE_MD = "# Hello\n\nWorld paragraph.\n\n---\n\nEnd.\n"


@pytest.fixture
def processor():
    return MarkdownProcessor(MockLLMInterface())


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
        assert "[MOCK PROCESSING]" in content

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


class TestCodeBlockSkipping:
    def test_code_without_korean_skipped(self, tmp_path):
        """Code blocks without Korean content are skipped (msgstr=msgid)."""
        md = "# Title\n\n```python\nprint('hello')\n```\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")
        target = tmp_path / "target.md"
        po_path = tmp_path / "messages.po"

        processor = MarkdownProcessor(MockLLMInterface())
        result = processor.process_document(source, target, po_path)
        # Code block should be skipped (no Korean)
        assert result["translation_stats"]["skipped"] >= 1

    def test_code_with_korean_processed(self, tmp_path):
        """Code blocks with Korean content should be processed."""
        md = "# Title\n\n```python\n# 한국어 주석\nprint('hello')\n```\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")
        target = tmp_path / "target.md"
        po_path = tmp_path / "messages.po"

        processor = MarkdownProcessor(MockLLMInterface())
        result = processor.process_document(source, target, po_path)
        assert result["translation_stats"]["processed"] >= 1


class TestLLMFailureHandling:
    def test_partial_failure_doesnt_crash(self, tmp_path):
        """If LLM fails on one entry, others should still be processed."""

        class FailingLLM(LLMInterface):
            def __init__(self):
                self.call_count = 0

            def process(self, source_text: str) -> str:
                self.call_count += 1
                if self.call_count == 1:
                    raise RuntimeError("LLM API error")
                return f"[OK] {source_text}"

        md = "# Title\n\nPara one.\n\nPara two.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")

        processor = MarkdownProcessor(FailingLLM())
        result = processor.process_document(
            source, tmp_path / "target.md", tmp_path / "m.po"
        )
        stats = result["translation_stats"]
        # At least one failure and one success
        assert stats["failed"] >= 1
        assert stats["processed"] >= 1

    def test_po_saved_on_error(self, tmp_path):
        """PO file should be saved even when processing has errors."""

        class AlwaysFailLLM(LLMInterface):
            def process(self, source_text: str) -> str:
                raise RuntimeError("fail")

        md = "# Title\n\nParagraph.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")
        po_path = tmp_path / "messages.po"

        processor = MarkdownProcessor(AlwaysFailLLM())
        processor.process_document(source, tmp_path / "target.md", po_path)
        # PO should still be saved (finally block)
        assert po_path.exists()


class TestExtractBlockType:
    def test_standard_msgctxt(self):
        processor = MarkdownProcessor(MockLLMInterface())
        result = processor._extract_block_type_from_msgctxt("intro/setup::para:0")
        assert result == "para"

    def test_heading_type(self):
        processor = MarkdownProcessor(MockLLMInterface())
        result = processor._extract_block_type_from_msgctxt("title::heading:0")
        assert result == "heading"

    def test_empty_msgctxt(self):
        processor = MarkdownProcessor(MockLLMInterface())
        assert processor._extract_block_type_from_msgctxt("") == ""

    def test_none_msgctxt(self):
        processor = MarkdownProcessor(MockLLMInterface())
        assert processor._extract_block_type_from_msgctxt(None) == ""

    def test_no_double_colon(self):
        processor = MarkdownProcessor(MockLLMInterface())
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
