"""Tests for BlockParser — the most critical module."""

import pytest

from mdpo_llm.parser import BlockParser


class TestSlugify:
    def test_simple_text(self, parser):
        assert parser.slugify("Hello World") == "hello-world"

    def test_special_characters(self, parser):
        assert parser.slugify("Hello! World?") == "hello-world"

    def test_empty_string(self, parser):
        assert parser.slugify("") == "section"

    def test_only_special_chars(self, parser):
        assert parser.slugify("!@#$%") == "section"

    def test_leading_trailing_dashes(self, parser):
        assert parser.slugify("  hello  ") == "hello"

    def test_multiple_spaces(self, parser):
        assert parser.slugify("hello   world") == "hello-world"


class TestContextId:
    def test_basic_format(self, parser):
        block = {"path": ["intro"], "type": "para", "idx_in_section": 0}
        assert parser.context_id(block) == "intro::para:0"

    def test_nested_path(self, parser):
        block = {"path": ["intro", "setup"], "type": "heading", "idx_in_section": 0}
        assert parser.context_id(block) == "intro/setup::heading:0"

    def test_empty_path(self, parser):
        block = {"path": [], "type": "para", "idx_in_section": 0}
        assert parser.context_id(block) == "::para:0"

    def test_index_increments(self, parser):
        block = {"path": ["intro"], "type": "para", "idx_in_section": 3}
        assert parser.context_id(block) == "intro::para:3"


class TestParseHeading:
    def test_h1(self, parser):
        blocks = parser.segment_markdown(["# Title"])
        assert len(blocks) == 1
        assert blocks[0]["type"] == "heading"
        assert blocks[0]["text"] == "# Title"
        assert blocks[0]["path"] == ["title"]

    def test_h2(self, parser):
        blocks = parser.segment_markdown(["# Top", "## Sub"])
        assert blocks[1]["type"] == "heading"
        assert blocks[1]["path"] == ["top", "sub"]

    def test_h3_nested(self, parser):
        blocks = parser.segment_markdown(["# A", "## B", "### C"])
        assert blocks[2]["path"] == ["a", "b", "c"]

    def test_heading_hierarchy_reset(self, parser):
        lines = ["# First", "## Sub1", "# Second", "## Sub2"]
        blocks = parser.segment_markdown(lines)
        assert blocks[2]["path"] == ["second"]
        assert blocks[3]["path"] == ["second", "sub2"]

    def test_duplicate_heading_slugs(self, parser):
        lines = ["# Setup", "## Setup"]
        blocks = parser.segment_markdown(lines)
        assert blocks[0]["path"] == ["setup"]
        # h2 slug "setup" at level 2 is unique at that level
        assert blocks[1]["path"] == ["setup", "setup"]

    def test_same_level_duplicate_slugs(self, parser):
        lines = ["# Setup", "# Setup"]
        blocks = parser.segment_markdown(lines)
        assert blocks[0]["path"] == ["setup"]
        assert blocks[1]["path"] == ["setup-1"]


class TestParseParagraph:
    def test_single_paragraph(self, parser):
        blocks = parser.segment_markdown(["Hello world"])
        assert len(blocks) == 1
        assert blocks[0]["type"] == "para"
        assert blocks[0]["text"] == "Hello world"

    def test_multiline_paragraph(self, parser):
        blocks = parser.segment_markdown(["Line one", "Line two"])
        assert len(blocks) == 1
        assert blocks[0]["type"] == "para"
        assert "Line one\nLine two" == blocks[0]["text"]

    def test_paragraphs_separated_by_blank(self, parser):
        blocks = parser.segment_markdown(["Para one", "", "Para two"])
        assert len(blocks) == 2
        assert blocks[0]["type"] == "para"
        assert blocks[1]["type"] == "para"


class TestParseCodeBlock:
    def test_fenced_code(self, parser):
        lines = ["```python", "print('hi')", "```"]
        blocks = parser.segment_markdown(lines)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "code"
        assert "print('hi')" in blocks[0]["text"]

    def test_tilde_fence(self, parser):
        lines = ["~~~", "code here", "~~~"]
        blocks = parser.segment_markdown(lines)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "code"

    def test_unclosed_fence(self, parser):
        lines = ["```python", "print('hi')", "more code"]
        blocks = parser.segment_markdown(lines)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "code"
        # unclosed fence consumes to end
        assert blocks[0]["end"] == 3

    def test_code_block_preserves_content(self, parser):
        lines = ["```", "  indented", "    more", "```"]
        blocks = parser.segment_markdown(lines)
        assert "  indented" in blocks[0]["text"]
        assert "    more" in blocks[0]["text"]


class TestParseList:
    def test_unordered_list(self, parser):
        lines = ["- Item one", "- Item two", "- Item three"]
        blocks = parser.segment_markdown(lines)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "ulist"

    def test_ordered_list(self, parser):
        lines = ["1. First", "2. Second", "3. Third"]
        blocks = parser.segment_markdown(lines)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "olist"

    def test_nested_list(self, parser):
        lines = ["- Top", "  - Nested", "- Another"]
        blocks = parser.segment_markdown(lines)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "ulist"
        assert "Nested" in blocks[0]["text"]

    def test_different_bullets(self, parser):
        lines = ["* Star item", "* Star two"]
        blocks = parser.segment_markdown(lines)
        assert blocks[0]["type"] == "ulist"

    def test_plus_bullets(self, parser):
        lines = ["+ Plus item", "+ Plus two"]
        blocks = parser.segment_markdown(lines)
        assert blocks[0]["type"] == "ulist"


class TestParseBlockquote:
    def test_single_blockquote(self, parser):
        lines = ["> Quote line"]
        blocks = parser.segment_markdown(lines)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "quote"

    def test_multiline_blockquote(self, parser):
        lines = ["> Line one", "> Line two"]
        blocks = parser.segment_markdown(lines)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "quote"
        assert "Line one" in blocks[0]["text"]
        assert "Line two" in blocks[0]["text"]


class TestParseTable:
    def test_simple_table(self, parser):
        lines = [
            "| A | B |",
            "|---|---|",
            "| 1 | 2 |",
        ]
        blocks = parser.segment_markdown(lines)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "table"

    def test_table_content(self, parser):
        lines = [
            "| Header 1 | Header 2 |",
            "|----------|----------|",
            "| Cell 1   | Cell 2   |",
        ]
        blocks = parser.segment_markdown(lines)
        assert "Header 1" in blocks[0]["text"]
        assert "Cell 1" in blocks[0]["text"]


class TestParseHR:
    def test_dashes(self, parser):
        blocks = parser.segment_markdown(["---"])
        assert len(blocks) == 1
        assert blocks[0]["type"] == "hr"

    def test_asterisks(self, parser):
        blocks = parser.segment_markdown(["***"])
        assert len(blocks) == 1
        assert blocks[0]["type"] == "hr"

    def test_underscores(self, parser):
        blocks = parser.segment_markdown(["___"])
        assert len(blocks) == 1
        assert blocks[0]["type"] == "hr"

    def test_spaced_hr(self, parser):
        blocks = parser.segment_markdown(["- - -"])
        assert len(blocks) == 1
        assert blocks[0]["type"] == "hr"


class TestComplexDocument:
    def test_mixed_blocks(self, parser, sample_lines):
        blocks = parser.segment_markdown(sample_lines)
        types = [b["type"] for b in blocks]
        assert "heading" in types
        assert "para" in types
        assert "ulist" in types
        assert "code" in types
        assert "table" in types
        assert "quote" in types
        assert "hr" in types
        assert "olist" in types

    def test_block_start_end_coverage(self, parser, sample_lines):
        """Blocks should not overlap."""
        blocks = parser.segment_markdown(sample_lines)
        for i in range(1, len(blocks)):
            assert blocks[i]["start"] >= blocks[i - 1]["end"]

    def test_section_indices_unique_per_type(self, parser, sample_lines):
        """idx_in_section should be unique per (path, type)."""
        blocks = parser.segment_markdown(sample_lines)
        seen = {}
        for b in blocks:
            key = (tuple(b["path"]), b["type"])
            idx = b["idx_in_section"]
            if key not in seen:
                seen[key] = set()
            assert idx not in seen[key], f"Duplicate idx {idx} for {key}"
            seen[key].add(idx)


class TestNoTranslateFormA:
    """Form A — `<!-- mdpo:no-translate -->` / `<!-- /mdpo:no-translate -->` pair."""

    def test_pair_around_paragraph(self, parser):
        lines = [
            "<!-- mdpo:no-translate -->",
            "Verbatim paragraph.",
            "<!-- /mdpo:no-translate -->",
        ]
        blocks = parser.segment_markdown(lines)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "no_translate"
        # Block text includes both markers and the body — reconstructor
        # writes from source_lines, but text is what reaches the PO entry.
        assert blocks[0]["text"] == "\n".join(lines)
        assert blocks[0]["start"] == 0
        assert blocks[0]["end"] == 3

    def test_pair_around_mixed_blocks(self, parser):
        lines = [
            "<!-- mdpo:no-translate -->",
            "Paragraph that stays.",
            "",
            "- list item 1",
            "- list item 2",
            "",
            "```python",
            "print('hi')",
            "```",
            "<!-- /mdpo:no-translate -->",
        ]
        blocks = parser.segment_markdown(lines)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "no_translate"
        assert blocks[0]["end"] == len(lines)
        # Inner content reaches the PO entry verbatim — so nested code
        # fence and list characters live inside text.
        assert "```python" in blocks[0]["text"]
        assert "- list item 1" in blocks[0]["text"]

    def test_pair_unclosed_consumes_to_eof_and_warns(self, parser, caplog):
        lines = [
            "<!-- mdpo:no-translate -->",
            "Body without closer.",
            "Still inside.",
        ]
        with caplog.at_level("WARNING", logger="mdpo_llm.parser"):
            blocks = parser.segment_markdown(lines)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "no_translate"
        assert blocks[0]["end"] == len(lines)
        assert any(
            "Unclosed" in record.message and "mdpo:no-translate" in record.message
            for record in caplog.records
        )

    def test_nested_opener_is_literal(self, parser):
        """A second opener inside an open range is treated as content."""
        lines = [
            "<!-- mdpo:no-translate -->",
            "outer body",
            "<!-- mdpo:no-translate -->",
            "still inside",
            "<!-- /mdpo:no-translate -->",
            "after",
        ]
        blocks = parser.segment_markdown(lines)
        # First close marker (line 5) ends the range; "after" is its own block.
        assert blocks[0]["type"] == "no_translate"
        assert blocks[0]["end"] == 5
        assert "<!-- mdpo:no-translate -->" in blocks[0]["text"].splitlines()[2]
        assert blocks[1]["type"] == "para"
        assert blocks[1]["text"] == "after"

    def test_marker_leading_trailing_whitespace_tolerated(self, parser):
        lines = [
            "   <!--   mdpo:no-translate   -->   ",
            "body",
            "  <!-- /mdpo:no-translate -->",
        ]
        blocks = parser.segment_markdown(lines)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "no_translate"


class TestNoTranslateFormB:
    """Form B — `<!-- mdpo:skip-next -->` marks just the next block."""

    def test_skip_next_with_paragraph(self, parser):
        lines = [
            "<!-- mdpo:skip-next -->",
            "This paragraph is skipped.",
            "",
            "This paragraph is translated.",
        ]
        blocks = parser.segment_markdown(lines)
        assert len(blocks) == 2
        assert blocks[0]["type"] == "no_translate"
        assert blocks[0]["start"] == 0
        assert blocks[0]["end"] == 2
        assert blocks[0]["text"] == "\n".join(lines[0:2])
        assert blocks[1]["type"] == "para"
        assert blocks[1]["text"] == "This paragraph is translated."

    def test_skip_next_with_code_fence(self, parser):
        lines = [
            "<!-- mdpo:skip-next -->",
            "```python",
            "print('hi')",
            "```",
        ]
        blocks = parser.segment_markdown(lines)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "no_translate"
        assert blocks[0]["end"] == len(lines)
        # Code fence content is inside the no_translate block, not a code block.
        assert all(b["type"] != "code" for b in blocks)

    def test_skip_next_at_eof_marker_only(self, parser):
        """Marker as last non-blank line → fail-soft marker-only block."""
        lines = ["<!-- mdpo:skip-next -->"]
        blocks = parser.segment_markdown(lines)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "no_translate"
        assert blocks[0]["text"] == "<!-- mdpo:skip-next -->"

    def test_skip_next_at_eof_with_trailing_blank_lines(self, parser):
        lines = ["<!-- mdpo:skip-next -->", "", ""]
        blocks = parser.segment_markdown(lines)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "no_translate"

    def test_skip_next_blank_line_between_marker_and_block(self, parser):
        lines = [
            "<!-- mdpo:skip-next -->",
            "",
            "Skipped paragraph.",
        ]
        blocks = parser.segment_markdown(lines)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "no_translate"
        assert blocks[0]["end"] == 3

    def test_skip_next_only_affects_next_block(self, parser):
        lines = [
            "<!-- mdpo:skip-next -->",
            "Skipped.",
            "",
            "Translated.",
            "",
            "Also translated.",
        ]
        blocks = parser.segment_markdown(lines)
        types = [b["type"] for b in blocks]
        assert types == ["no_translate", "para", "para"]


class TestNoTranslateAdjacency:
    """Markers placed immediately after prose / lists / headings without
    an intervening blank line must still be recognised. Regression for
    the original cycle-1 finding where paragraph / list continuation
    silently swallowed the marker and sent the protected content to
    the LLM."""

    def test_marker_directly_after_paragraph(self, parser):
        lines = [
            "Some prose without a trailing blank.",
            "<!-- mdpo:no-translate -->",
            "Verbatim body.",
            "<!-- /mdpo:no-translate -->",
        ]
        blocks = parser.segment_markdown(lines)
        types = [b["type"] for b in blocks]
        assert types == ["para", "no_translate"]
        assert blocks[0]["text"] == "Some prose without a trailing blank."
        assert blocks[1]["end"] == len(lines)

    def test_skip_next_directly_after_paragraph(self, parser):
        lines = [
            "Prose.",
            "<!-- mdpo:skip-next -->",
            "Skipped.",
        ]
        blocks = parser.segment_markdown(lines)
        types = [b["type"] for b in blocks]
        assert types == ["para", "no_translate"]
        assert blocks[0]["text"] == "Prose."

    def test_marker_directly_after_list(self, parser):
        lines = [
            "- item one",
            "- item two",
            "<!-- mdpo:no-translate -->",
            "Verbatim.",
            "<!-- /mdpo:no-translate -->",
        ]
        blocks = parser.segment_markdown(lines)
        types = [b["type"] for b in blocks]
        assert types == ["ulist", "no_translate"]
        assert "Verbatim." not in blocks[0]["text"]


class TestNoTranslateHeadingPath:
    """A heading inside a Form A range still updates the parser's slug
    stack so blocks after the range key under the correct section."""

    def test_heading_inside_range_updates_path(self, parser):
        lines = [
            "# Intro",
            "intro text",
            "",
            "<!-- mdpo:no-translate -->",
            "## Protected Section",
            "verbatim body",
            "<!-- /mdpo:no-translate -->",
            "",
            "Following paragraph.",
        ]
        blocks = parser.segment_markdown(lines)
        # Last block is the trailing paragraph.
        trailing = blocks[-1]
        assert trailing["type"] == "para"
        assert trailing["path"] == ["intro", "protected-section"]

    def test_heading_inside_nested_code_fence_does_not_update_path(self, parser):
        """A heading-shaped line inside a nested code fence inside the
        protected range must NOT pollute the slug stack."""
        lines = [
            "# Intro",
            "",
            "<!-- mdpo:no-translate -->",
            "```python",
            "# this is a comment, not a heading",
            "```",
            "<!-- /mdpo:no-translate -->",
            "",
            "Trailing.",
        ]
        blocks = parser.segment_markdown(lines)
        trailing = blocks[-1]
        assert trailing["type"] == "para"
        # Path stays under the real heading, no spurious slug added.
        assert trailing["path"] == ["intro"]

    def test_multiple_headings_inside_range(self, parser):
        lines = [
            "# Top",
            "",
            "<!-- mdpo:no-translate -->",
            "## A",
            "body a",
            "### Deep",
            "body deep",
            "## B",
            "body b",
            "<!-- /mdpo:no-translate -->",
            "",
            "After.",
        ]
        blocks = parser.segment_markdown(lines)
        trailing = blocks[-1]
        assert trailing["type"] == "para"
        # Final heading inside the range is h2 "B" — h3 counter cleared.
        assert trailing["path"] == ["top", "b"]


class TestNoTranslateNonMatching:
    """HTML comments outside the `mdpo:` namespace flow through unchanged."""

    def test_generic_html_comment_is_paragraph(self, parser):
        lines = ["<!-- no-translate -->", "body"]
        blocks = parser.segment_markdown(lines)
        # Generic comment is NOT recognised; both lines fold into one paragraph.
        assert len(blocks) == 1
        assert blocks[0]["type"] == "para"

    def test_case_sensitive_marker(self, parser):
        """`MDPO:no-translate` (uppercase) is NOT recognised."""
        lines = [
            "<!-- MDPO:no-translate -->",
            "body",
            "<!-- /MDPO:no-translate -->",
        ]
        blocks = parser.segment_markdown(lines)
        # Uppercase prefix falls through to paragraph parser.
        assert all(b["type"] != "no_translate" for b in blocks)

    def test_close_marker_alone_is_paragraph(self, parser):
        """A stray closer with no opener is treated as a regular paragraph."""
        lines = ["<!-- /mdpo:no-translate -->"]
        blocks = parser.segment_markdown(lines)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "para"


class TestEdgeCases:
    def test_empty_document(self, parser):
        blocks = parser.segment_markdown([])
        assert blocks == []

    def test_blank_lines_only(self, parser):
        blocks = parser.segment_markdown(["", "", ""])
        assert blocks == []

    def test_single_blank_line(self, parser):
        blocks = parser.segment_markdown([""])
        assert blocks == []

    def test_heading_with_path_tracking(self, parser):
        lines = ["# A", "text", "## B", "more text"]
        blocks = parser.segment_markdown(lines)
        # paragraph under A should have path ["a"]
        assert blocks[1]["path"] == ["a"]
        # paragraph under B should have path ["a", "b"]
        assert blocks[3]["path"] == ["a", "b"]
