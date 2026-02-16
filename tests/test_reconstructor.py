"""Tests for DocumentReconstructor rebuild and coverage stats."""

import polib
import pytest

from mdpo_llm.parser import BlockParser
from mdpo_llm.reconstructor import DocumentReconstructor


@pytest.fixture
def reconstructor():
    return DocumentReconstructor(skip_types=["hr"])


@pytest.fixture
def simple_doc():
    """Simple document with heading, para, and hr."""
    text = "# Title\n\nHello world\n\n---\n\nGoodbye\n"
    lines = text.splitlines(keepends=True)
    parser = BlockParser()
    blocks = parser.segment_markdown([l.rstrip("\n") for l in lines])
    return lines, blocks, parser.context_id


@pytest.fixture
def po_with_translations(simple_doc):
    """PO file with all translatable blocks translated."""
    lines, blocks, ctx_func = simple_doc
    po = polib.POFile()
    po.metadata = {"Content-Type": "text/plain; charset=UTF-8"}
    for block in blocks:
        ctx = ctx_func(block)
        if block["type"] == "hr":
            po.append(polib.POEntry(msgctxt=ctx, msgid=block["text"], msgstr=""))
        else:
            po.append(
                polib.POEntry(
                    msgctxt=ctx,
                    msgid=block["text"],
                    msgstr=f"[T] {block['text']}",
                )
            )
    return po


class TestRebuildMarkdown:
    def test_all_translated(self, reconstructor, simple_doc, po_with_translations):
        lines, blocks, ctx_func = simple_doc
        result = reconstructor.rebuild_markdown(
            lines, blocks, po_with_translations, ctx_func
        )
        assert "[T] # Title" in result
        assert "[T] Hello world" in result
        assert "[T] Goodbye" in result

    def test_no_translations(self, reconstructor, simple_doc):
        lines, blocks, ctx_func = simple_doc
        po = polib.POFile()
        po.metadata = {"Content-Type": "text/plain; charset=UTF-8"}
        for block in blocks:
            po.append(
                polib.POEntry(
                    msgctxt=ctx_func(block), msgid=block["text"], msgstr=""
                )
            )
        result = reconstructor.rebuild_markdown(lines, blocks, po, ctx_func)
        # Falls back to original
        assert "# Title" in result
        assert "Hello world" in result

    def test_partial_translations(self, reconstructor, simple_doc):
        lines, blocks, ctx_func = simple_doc
        po = polib.POFile()
        po.metadata = {"Content-Type": "text/plain; charset=UTF-8"}
        for block in blocks:
            ctx = ctx_func(block)
            if block["type"] == "heading":
                po.append(
                    polib.POEntry(msgctxt=ctx, msgid=block["text"], msgstr="[T] Title")
                )
            else:
                po.append(
                    polib.POEntry(msgctxt=ctx, msgid=block["text"], msgstr="")
                )
        result = reconstructor.rebuild_markdown(lines, blocks, po, ctx_func)
        assert "[T] Title" in result
        assert "Hello world" in result  # fallback

    def test_skip_types_copied_as_is(self, reconstructor, simple_doc, po_with_translations):
        lines, blocks, ctx_func = simple_doc
        result = reconstructor.rebuild_markdown(
            lines, blocks, po_with_translations, ctx_func
        )
        assert "---" in result

    def test_gaps_preserved(self, reconstructor):
        """Blank lines between blocks should be preserved."""
        text = "# Title\n\n\nParagraph\n"
        lines = text.splitlines(keepends=True)
        parser = BlockParser()
        blocks = parser.segment_markdown([l.rstrip("\n") for l in lines])
        po = polib.POFile()
        for block in blocks:
            ctx = parser.context_id(block)
            po.append(
                polib.POEntry(msgctxt=ctx, msgid=block["text"], msgstr=block["text"])
            )
        result = reconstructor.rebuild_markdown(lines, blocks, po, parser.context_id)
        # The blank lines between title and paragraph should be present
        assert "\n\n" in result

    def test_trailing_content(self, reconstructor):
        """Content after last block should be preserved."""
        text = "# Title\n\nsome trailing\n"
        lines = text.splitlines(keepends=True)
        parser = BlockParser()
        blocks = parser.segment_markdown([l.rstrip("\n") for l in lines])
        po = polib.POFile()
        for block in blocks:
            ctx = parser.context_id(block)
            po.append(
                polib.POEntry(
                    msgctxt=ctx, msgid=block["text"], msgstr=f"[T] {block['text']}"
                )
            )
        result = reconstructor.rebuild_markdown(lines, blocks, po, parser.context_id)
        # All content should be accounted for
        assert "[T]" in result


class TestGetProcessCoverage:
    def test_full_coverage(self, reconstructor, simple_doc, po_with_translations):
        _, blocks, ctx_func = simple_doc
        stats = reconstructor.get_process_coverage(
            blocks, po_with_translations, ctx_func
        )
        assert stats["total_blocks"] == len(blocks)
        assert stats["translated_blocks"] > 0
        assert stats["untranslated_blocks"] == 0
        assert stats["coverage_percentage"] == 100.0

    def test_zero_coverage(self, reconstructor, simple_doc):
        _, blocks, ctx_func = simple_doc
        po = polib.POFile()
        for block in blocks:
            po.append(
                polib.POEntry(
                    msgctxt=ctx_func(block), msgid=block["text"], msgstr=""
                )
            )
        stats = reconstructor.get_process_coverage(blocks, po, ctx_func)
        assert stats["coverage_percentage"] == 0.0
        assert stats["untranslated_blocks"] == stats["translatable_blocks"]

    def test_skip_types_excluded_from_translatable(
        self, reconstructor, simple_doc, po_with_translations
    ):
        _, blocks, ctx_func = simple_doc
        stats = reconstructor.get_process_coverage(
            blocks, po_with_translations, ctx_func
        )
        hr_count = sum(1 for b in blocks if b["type"] == "hr")
        assert stats["translatable_blocks"] == stats["total_blocks"] - hr_count

    def test_by_type_breakdown(self, reconstructor, simple_doc, po_with_translations):
        _, blocks, ctx_func = simple_doc
        stats = reconstructor.get_process_coverage(
            blocks, po_with_translations, ctx_func
        )
        assert "by_type" in stats
        assert isinstance(stats["by_type"], dict)
        for block_type, type_info in stats["by_type"].items():
            assert "total" in type_info
            assert "translatable" in type_info
            assert "translated" in type_info
            assert "fuzzy" in type_info

    def test_fuzzy_blocks_counted(self, reconstructor, simple_doc):
        _, blocks, ctx_func = simple_doc
        po = polib.POFile()
        for block in blocks:
            entry = polib.POEntry(
                msgctxt=ctx_func(block), msgid=block["text"], msgstr="translated"
            )
            if block["type"] == "para":
                entry.flags.append("fuzzy")
            po.append(entry)
        stats = reconstructor.get_process_coverage(blocks, po, ctx_func)
        assert stats["fuzzy_blocks"] > 0


class TestExportTranslationReport:
    def test_report_format(self, reconstructor, simple_doc, po_with_translations):
        _, blocks, ctx_func = simple_doc
        report = reconstructor.export_translation_report(
            "test.md", blocks, po_with_translations, ctx_func
        )
        assert "# Translation Report" in report
        assert "test.md" in report
        assert "Summary" in report
        assert "Total Blocks" in report
        assert "Coverage" in report
        assert "By Block Type" in report
