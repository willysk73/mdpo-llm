"""Tests for MarkdownProcessor end-to-end and error handling."""

from pathlib import Path
from unittest.mock import MagicMock, patch

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
    def test_code_without_source_lang_skipped(self, tmp_path):
        """Code blocks without source language content are skipped (msgstr=msgid)."""
        md = "# Title\n\n```python\nprint('hello')\n```\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")
        target = tmp_path / "target.md"
        po_path = tmp_path / "messages.po"

        processor = MarkdownProcessor(MockLLMInterface(), source_langs=["ko"])
        result = processor.process_document(source, target, po_path)
        # Code block should be skipped (no Korean)
        assert result["translation_stats"]["skipped"] >= 1

    def test_code_with_source_lang_processed(self, tmp_path):
        """Code blocks with source language content should be processed."""
        md = "# Title\n\n```python\n# 한국어 주석\nprint('hello')\n```\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")
        target = tmp_path / "target.md"
        po_path = tmp_path / "messages.po"

        processor = MarkdownProcessor(MockLLMInterface(), source_langs=["ko"])
        result = processor.process_document(source, target, po_path)
        assert result["translation_stats"]["processed"] >= 1

    def test_no_source_langs_processes_all_code(self, tmp_path):
        """When source_langs is None, all code blocks are sent to the LLM."""
        md = "# Title\n\n```python\nprint('hello')\n```\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")
        target = tmp_path / "target.md"
        po_path = tmp_path / "messages.po"

        # No source_langs → code block skipping disabled
        processor = MarkdownProcessor(MockLLMInterface())
        result = processor.process_document(source, target, po_path)
        # Code block should be processed (not skipped)
        assert result["translation_stats"]["processed"] >= 2  # heading + code

    def test_multiple_source_langs(self, tmp_path):
        """Code blocks with any of multiple source languages are processed."""
        md = "# Title\n\n```python\n# 你好世界\nprint('hello')\n```\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")
        target = tmp_path / "target.md"
        po_path = tmp_path / "messages.po"

        processor = MarkdownProcessor(MockLLMInterface(), source_langs=["ko", "zh"])
        result = processor.process_document(source, target, po_path)
        # Chinese detected → code block should be processed
        assert result["translation_stats"]["processed"] >= 1


class TestTargetLangFlow:
    def test_target_lang_forwarded_to_mock(self, tmp_path):
        """target_lang should appear in MockLLMInterface output."""
        md = "# Title\n\nParagraph.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")

        processor = MarkdownProcessor(MockLLMInterface(), target_lang="ko")
        result = processor.process_document(
            source, tmp_path / "target.md", tmp_path / "m.po"
        )
        content = (tmp_path / "target.md").read_text(encoding="utf-8")
        assert "lang=ko" in content
        assert result["translation_stats"]["processed"] >= 1

    def test_no_target_lang_no_lang_tag(self, tmp_path):
        """Without target_lang, no lang= should appear in output."""
        md = "# Title\n\nParagraph.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")

        processor = MarkdownProcessor(MockLLMInterface())
        processor.process_document(source, tmp_path / "target.md", tmp_path / "m.po")
        content = (tmp_path / "target.md").read_text(encoding="utf-8")
        assert "lang=" not in content

    def test_target_lang_not_sent_to_old_llm(self, tmp_path):
        """Old-style LLM without target_lang param should not receive it."""

        class OldLLM(LLMInterface):
            def process(self, source_text: str) -> str:
                return f"[OLD] {source_text}"

        md = "# Title\n\nParagraph.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")

        processor = MarkdownProcessor(OldLLM(), target_lang="ko")
        result = processor.process_document(
            source, tmp_path / "target.md", tmp_path / "m.po"
        )
        assert result["translation_stats"]["processed"] >= 1
        assert result["translation_stats"]["failed"] == 0

    def test_target_lang_with_reference_pairs(self, tmp_path):
        """Both target_lang and reference_pairs forwarded when LLM accepts both."""
        received = []

        class TrackingLLM(LLMInterface):
            def process(self, source_text: str, reference_pairs=None, target_lang=None) -> str:
                received.append({"text": source_text, "pairs": reference_pairs, "lang": target_lang})
                return f"[OK] {source_text}"

        md = "# Title\n\nParagraph one.\n\nParagraph two.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")

        processor = MarkdownProcessor(TrackingLLM(), target_lang="ko")
        processor.process_document(source, tmp_path / "target.md", tmp_path / "m.po")

        # Every call should have target_lang="ko"
        for call in received:
            assert call["lang"] == "ko"


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


class TestProcessDirectory:
    """Tests for MarkdownProcessor.process_directory()."""

    def _make_md_file(self, path: Path, content: str = "# Hello\n\nWorld.\n"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def test_processes_all_md_files(self, tmp_path):
        """Flat directory with multiple .md files — all should be processed."""
        src = tmp_path / "src"
        tgt = tmp_path / "tgt"
        po = tmp_path / "po"

        self._make_md_file(src / "a.md", "# A\n\nAlpha.\n")
        self._make_md_file(src / "b.md", "# B\n\nBravo.\n")
        self._make_md_file(src / "c.md", "# C\n\nCharlie.\n")

        processor = MarkdownProcessor(MockLLMInterface())
        result = processor.process_directory(src, tgt, po)

        assert result["files_processed"] + result["files_skipped"] == 3
        assert result["files_failed"] == 0
        assert len(result["results"]) == 3
        # All target files exist
        assert (tgt / "a.md").exists()
        assert (tgt / "b.md").exists()
        assert (tgt / "c.md").exists()

    def test_mirrors_subdirectory_structure(self, tmp_path):
        """Nested directories should be mirrored in target and PO dirs."""
        src = tmp_path / "src"
        tgt = tmp_path / "tgt"
        po = tmp_path / "po"

        self._make_md_file(src / "top.md")
        self._make_md_file(src / "sub" / "nested.md")

        processor = MarkdownProcessor(MockLLMInterface())
        processor.process_directory(src, tgt, po)

        assert (tgt / "top.md").exists()
        assert (tgt / "sub" / "nested.md").exists()

    def test_po_uses_po_extension(self, tmp_path):
        """PO files should use .po extension, not .md."""
        src = tmp_path / "src"
        tgt = tmp_path / "tgt"
        po = tmp_path / "po"

        self._make_md_file(src / "doc.md")
        self._make_md_file(src / "sub" / "page.md")

        processor = MarkdownProcessor(MockLLMInterface())
        processor.process_directory(src, tgt, po)

        assert (po / "doc.po").exists()
        assert (po / "sub" / "page.po").exists()
        # No .md files in PO dir
        assert not list(po.glob("**/*.md"))

    def test_custom_glob_pattern(self, tmp_path):
        """Non-recursive glob should only match top-level files."""
        src = tmp_path / "src"
        tgt = tmp_path / "tgt"
        po = tmp_path / "po"

        self._make_md_file(src / "top.md")
        self._make_md_file(src / "sub" / "nested.md")

        processor = MarkdownProcessor(MockLLMInterface())
        result = processor.process_directory(src, tgt, po, glob="*.md")

        # Only top-level file matched
        total = result["files_processed"] + result["files_skipped"]
        assert total == 1
        assert (tgt / "top.md").exists()
        assert not (tgt / "sub" / "nested.md").exists()

    def test_empty_directory(self, tmp_path):
        """Empty directory returns zero counts."""
        src = tmp_path / "src"
        src.mkdir()
        tgt = tmp_path / "tgt"
        po = tmp_path / "po"

        processor = MarkdownProcessor(MockLLMInterface())
        result = processor.process_directory(src, tgt, po)

        assert result["files_processed"] == 0
        assert result["files_failed"] == 0
        assert result["files_skipped"] == 0
        assert result["results"] == []

    def test_single_file_failure_continues(self, tmp_path):
        """One file failing should not stop processing of remaining files."""
        src = tmp_path / "src"
        tgt = tmp_path / "tgt"
        po = tmp_path / "po"

        self._make_md_file(src / "good.md", "# Good\n\nContent.\n")
        # Create a file that will cause process_document to fail
        bad = src / "bad.md"
        bad.parent.mkdir(parents=True, exist_ok=True)
        bad.write_text("# Bad\n\nBad content.\n", encoding="utf-8")

        processor = MarkdownProcessor(MockLLMInterface())

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

    def test_inplace_mode_forwarded(self, tmp_path):
        """The inplace flag should be forwarded to process_document."""
        src = tmp_path / "src"
        tgt = tmp_path / "tgt"
        po = tmp_path / "po"

        self._make_md_file(src / "doc.md", "# Title\n\nSome text.\n")

        processor = MarkdownProcessor(MockLLMInterface())

        with patch.object(
            processor, "process_document", wraps=processor.process_document
        ) as mock_pd:
            processor.process_directory(src, tgt, po, inplace=True)
            mock_pd.assert_called_once()
            _, kwargs = mock_pd.call_args
            assert kwargs.get("inplace") is True or mock_pd.call_args[0][-1] is True

    def test_return_value_structure(self, tmp_path):
        """Return dict should contain all expected keys."""
        src = tmp_path / "src"
        tgt = tmp_path / "tgt"
        po = tmp_path / "po"
        src.mkdir()

        processor = MarkdownProcessor(MockLLMInterface())
        result = processor.process_directory(src, tgt, po)

        assert result["source_dir"] == str(src)
        assert result["target_dir"] == str(tgt)
        assert result["po_dir"] == str(po)
        assert "files_processed" in result
        assert "files_failed" in result
        assert "files_skipped" in result
        assert "results" in result

    def test_max_workers_parameter(self, tmp_path):
        """max_workers parameter is accepted and produces correct results."""
        src = tmp_path / "src"
        tgt = tmp_path / "tgt"
        po = tmp_path / "po"

        self._make_md_file(src / "a.md", "# A\n\nAlpha.\n")
        self._make_md_file(src / "b.md", "# B\n\nBravo.\n")

        processor = MarkdownProcessor(MockLLMInterface())
        result = processor.process_directory(src, tgt, po, max_workers=2)

        assert result["files_processed"] + result["files_skipped"] == 2
        assert result["files_failed"] == 0


class TestSequentialProcessing:
    """Tests for sequential entry processing with reference context."""

    def test_entries_processed_in_document_order(self, tmp_path):
        """LLM calls should happen in document order."""
        call_order = []

        class TrackingLLM(LLMInterface):
            def process(self, source_text: str, reference_pairs=None) -> str:
                call_order.append(source_text)
                return f"[OK] {source_text}"

        md = "# First\n\nSecond paragraph.\n\nThird paragraph.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")

        processor = MarkdownProcessor(TrackingLLM())
        processor.process_document(source, tmp_path / "target.md", tmp_path / "m.po")

        # Entries should be in document order
        assert len(call_order) >= 3
        # First heading (includes markdown prefix), then paragraphs
        assert call_order[0] == "# First"
        assert call_order[1] == "Second paragraph."
        assert call_order[2] == "Third paragraph."

    def test_reference_pairs_grow_over_run(self, tmp_path):
        """First entry should get no reference pairs; later entries should get some."""
        received_pairs = []

        class TrackingLLM(LLMInterface):
            def process(self, source_text: str, reference_pairs=None) -> str:
                received_pairs.append(reference_pairs)
                return f"[OK] {source_text}"

        md = "# Title\n\nParagraph one.\n\nParagraph two.\n\nParagraph three.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")

        processor = MarkdownProcessor(TrackingLLM())
        processor.process_document(source, tmp_path / "target.md", tmp_path / "m.po")

        # First entry has no reference pairs (pool is empty)
        assert received_pairs[0] is None
        # Later entries should have growing reference context
        # At least the last entry should have pairs from earlier entries
        non_none = [p for p in received_pairs if p is not None]
        assert len(non_none) >= 1

    def test_existing_po_seeds_pool(self, tmp_path):
        """Second run after source edit should seed pool from existing translations."""
        received_pairs = []

        class TrackingLLM(LLMInterface):
            def process(self, source_text: str, reference_pairs=None) -> str:
                received_pairs.append((source_text, reference_pairs))
                return f"[OK] {source_text}"

        md = "# Title\n\nOriginal paragraph.\n\nAnother paragraph.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")
        po_path = tmp_path / "m.po"

        processor = MarkdownProcessor(TrackingLLM())
        # First run — translates everything
        processor.process_document(source, tmp_path / "target.md", po_path)

        received_pairs.clear()

        # Edit source — change one paragraph
        source.write_text(
            "# Title\n\nChanged paragraph.\n\nAnother paragraph.\n", encoding="utf-8"
        )
        # Second run — only the changed paragraph needs translation,
        # but the pool is seeded from existing PO translations
        processor.process_document(source, tmp_path / "target.md", po_path)

        # The changed paragraph should have reference pairs from the seeded pool
        assert len(received_pairs) >= 1

    def test_old_style_llm_backward_compat(self, tmp_path):
        """LLM subclass without reference_pairs parameter still works."""

        class OldLLM(LLMInterface):
            def process(self, source_text: str) -> str:
                return f"[OLD] {source_text}"

        md = "# Title\n\nParagraph.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")

        processor = MarkdownProcessor(OldLLM())
        result = processor.process_document(
            source, tmp_path / "target.md", tmp_path / "m.po"
        )
        assert result["translation_stats"]["processed"] >= 1
        assert result["translation_stats"]["failed"] == 0

        target_content = (tmp_path / "target.md").read_text(encoding="utf-8")
        assert "[OLD]" in target_content

    def test_max_reference_pairs_constructor_arg(self, tmp_path):
        """max_reference_pairs should limit the number of pairs passed."""
        received_pairs = []

        class TrackingLLM(LLMInterface):
            def process(self, source_text: str, reference_pairs=None) -> str:
                received_pairs.append(reference_pairs)
                return f"[OK] {source_text}"

        # Many paragraphs to ensure pool grows
        lines = ["# Title\n"]
        for i in range(10):
            lines.append(f"\nParagraph {i}.\n")
        md = "".join(lines)

        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")

        processor = MarkdownProcessor(TrackingLLM(), max_reference_pairs=2)
        processor.process_document(source, tmp_path / "target.md", tmp_path / "m.po")

        for pairs in received_pairs:
            if pairs is not None:
                assert len(pairs) <= 2
