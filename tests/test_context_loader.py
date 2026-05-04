"""Free-text domain context cascade tests (T-18).

Covers the pure functions in :mod:`mdpo_llm.context_loader`:

    * cascade walk under ``source_root`` (parent → child concatenation)
    * cwd ``context.md`` appended unless already in the tree walk
    * ``cli_override`` appended LAST (topmost layer)
    * empty / missing files silently skipped at every level
    * single-file callers with ``source_root=None`` collapse to file_dir
    * ``inject_context`` header text + trailing-newline preservation
    * ``read_context_file`` size cap warning is informational, not a fail

These exercise the loader directly so the test stays fast and
deterministic — no ``litellm`` mocking required.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from mdpo_llm.context_loader import (
    ADDITIONAL_CONTEXT_HEADER,
    CONTEXT_FILENAME,
    MAX_CONTEXT_BYTES,
    inject_context,
    read_context_file,
    resolve_context_chain,
)


def _write_context(directory: Path, body: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / CONTEXT_FILENAME
    path.write_text(body, encoding="utf-8")
    return path


class TestCascadeOrder:
    def test_parent_before_child_in_concatenation(self, tmp_path):
        source_root = tmp_path / "docs"
        child = source_root / "api"
        _write_context(source_root, "Root domain: game-security SDK.")
        _write_context(child, "API section: prefer formal voice.")
        file_path = child / "auth.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("body", encoding="utf-8")

        merged = resolve_context_chain(file_path, source_root, cwd=None)

        # Parent block precedes child block, separated by a blank line.
        assert merged == (
            "Root domain: game-security SDK.\n"
            "\n"
            "API section: prefer formal voice."
        )

    def test_deep_tree_walks_every_level(self, tmp_path):
        source_root = tmp_path / "docs"
        mid = source_root / "section"
        leaf = mid / "api"
        _write_context(source_root, "ROOT")
        _write_context(mid, "MID")
        _write_context(leaf, "LEAF")
        file_path = leaf / "page.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("x", encoding="utf-8")

        merged = resolve_context_chain(file_path, source_root, cwd=None)

        assert merged == "ROOT\n\nMID\n\nLEAF"

    def test_intermediate_dir_without_context_inherits_outer(self, tmp_path):
        source_root = tmp_path / "docs"
        leaf = source_root / "section" / "api"
        leaf.mkdir(parents=True, exist_ok=True)
        _write_context(source_root, "ROOT")
        _write_context(leaf, "LEAF")
        file_path = leaf / "page.md"
        file_path.write_text("x", encoding="utf-8")

        merged = resolve_context_chain(file_path, source_root, cwd=None)

        # Mid layer absent → only ROOT + LEAF survive (no empty separator).
        assert merged == "ROOT\n\nLEAF"


class TestEmptyAndMissing:
    def test_missing_file_is_silent_skip(self, tmp_path, caplog):
        source_root = tmp_path / "docs"
        source_root.mkdir(parents=True, exist_ok=True)
        file_path = source_root / "page.md"
        file_path.write_text("x", encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            merged = resolve_context_chain(file_path, source_root, cwd=None)

        assert merged == ""
        # Brief: no 'missing' warning — the cascade walks every dir
        # and absence is the common case.
        assert not any("context" in r.message.lower() for r in caplog.records)

    def test_empty_file_is_silent_skip(self, tmp_path, caplog):
        source_root = tmp_path / "docs"
        _write_context(source_root, "")
        file_path = source_root / "page.md"
        file_path.write_text("x", encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            merged = resolve_context_chain(file_path, source_root, cwd=None)

        assert merged == ""
        # Brief: no 'empty' warning either. Operators only hear about
        # genuinely surprising things (size cap, decode error).
        assert not any("empty" in r.message.lower() for r in caplog.records)

    def test_whitespace_only_file_treated_as_empty(self, tmp_path):
        source_root = tmp_path / "docs"
        _write_context(source_root, "   \n   \n")
        file_path = source_root / "page.md"
        file_path.write_text("x", encoding="utf-8")

        merged = resolve_context_chain(file_path, source_root, cwd=None)

        assert merged == ""


class TestCliOverride:
    def test_override_appended_last(self, tmp_path):
        source_root = tmp_path / "docs"
        leaf = source_root / "api"
        _write_context(source_root, "Root.")
        _write_context(leaf, "Leaf.")
        file_path = leaf / "page.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("x", encoding="utf-8")

        merged = resolve_context_chain(
            file_path, source_root, cwd=None,
            cli_override="Override (CLI).",
        )

        # Override is the LAST block — closest to the user message.
        assert merged.endswith("Override (CLI).")
        assert merged == "Root.\n\nLeaf.\n\nOverride (CLI)."

    def test_override_alone_works_without_cascade(self, tmp_path):
        source_root = tmp_path / "docs"
        source_root.mkdir(parents=True, exist_ok=True)
        file_path = source_root / "page.md"
        file_path.write_text("x", encoding="utf-8")

        merged = resolve_context_chain(
            file_path, source_root, cwd=None, cli_override="Just CLI."
        )

        assert merged == "Just CLI."

    def test_override_empty_string_treated_as_no_override(self, tmp_path):
        source_root = tmp_path / "docs"
        _write_context(source_root, "Cascade.")
        file_path = source_root / "page.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("x", encoding="utf-8")

        merged = resolve_context_chain(
            file_path, source_root, cwd=None, cli_override=""
        )

        assert merged == "Cascade."


class TestCwdLayer:
    def test_cwd_context_appended_when_outside_tree(self, tmp_path):
        source_root = tmp_path / "tree"
        cwd_dir = tmp_path / "cwd"
        _write_context(source_root, "TREE")
        _write_context(cwd_dir, "CWD")
        file_path = source_root / "page.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("x", encoding="utf-8")

        merged = resolve_context_chain(
            file_path, source_root, cwd=cwd_dir
        )

        # Tree cascade first, cwd block after, override (none) skipped.
        assert merged == "TREE\n\nCWD"

    def test_cwd_inside_tree_not_double_applied(self, tmp_path):
        source_root = tmp_path / "docs"
        _write_context(source_root, "ROOT")
        file_path = source_root / "page.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("x", encoding="utf-8")

        # cwd == source_root: ROOT/context.md is already in the tree
        # walk, so the cwd layer must NOT re-add it.
        merged = resolve_context_chain(
            file_path, source_root, cwd=source_root
        )

        assert merged == "ROOT"


class TestSingleFileFallback:
    def test_source_root_none_uses_file_dir_only(self, tmp_path):
        leaf = tmp_path / "leaf"
        _write_context(leaf, "Leaf-only.")
        file_path = leaf / "page.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("x", encoding="utf-8")

        merged = resolve_context_chain(file_path, None, cwd=None)

        assert merged == "Leaf-only."

    def test_file_outside_source_root_falls_back_to_file_dir(self, tmp_path):
        source_root = tmp_path / "docs"
        outside = tmp_path / "elsewhere"
        _write_context(source_root, "ROOT (must not appear)")
        _write_context(outside, "OUTSIDE")
        file_path = outside / "page.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("x", encoding="utf-8")

        merged = resolve_context_chain(file_path, source_root, cwd=None)

        # source_root cascade does NOT leak into a sibling tree;
        # falls back to the file's own directory.
        assert merged == "OUTSIDE"


class TestReadContextFile:
    def test_large_file_warns_but_passes_through(self, tmp_path, caplog):
        big_path = tmp_path / CONTEXT_FILENAME
        body = "x" * (MAX_CONTEXT_BYTES + 1024)
        big_path.write_text(body, encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            text = read_context_file(big_path)

        # Brief: do NOT hard-fail; pass full content through and warn.
        assert text == body
        assert any("soft cap" in r.message for r in caplog.records)

    def test_file_at_or_under_cap_does_not_warn(self, tmp_path, caplog):
        path = tmp_path / CONTEXT_FILENAME
        path.write_text("a" * 1024, encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            read_context_file(path)

        assert not any("soft cap" in r.message for r in caplog.records)

    def test_invalid_utf8_is_warned_and_skipped(self, tmp_path, caplog):
        path = tmp_path / CONTEXT_FILENAME
        path.write_bytes(b"\xff\xfe\x00\x01invalid")

        with caplog.at_level(logging.WARNING):
            text = read_context_file(path)

        assert text == ""
        assert any("UTF-8" in r.message for r in caplog.records)

    def test_missing_file_returns_empty(self, tmp_path):
        text = read_context_file(tmp_path / "nope.md")
        assert text == ""


class TestInjectContext:
    def test_none_context_returns_prompt_unchanged(self):
        prompt = "Translate this.\n"
        assert inject_context(prompt, None) == prompt

    def test_empty_context_returns_prompt_unchanged(self):
        prompt = "Translate this.\n"
        assert inject_context(prompt, "") == prompt

    def test_block_appears_after_existing_content_under_header(self):
        prompt = "Translate this."
        out = inject_context(prompt, "Domain: game-security SDK.")
        assert ADDITIONAL_CONTEXT_HEADER in out
        assert out.endswith(
            f"{ADDITIONAL_CONTEXT_HEADER}\nDomain: game-security SDK."
        )
        # Existing prompt content survives unmodified.
        assert out.startswith("Translate this.")

    def test_trailing_newline_preserved(self):
        prompt = "Translate this.\n"
        out = inject_context(prompt, "Brief.")
        # Trailing newline survives — important so downstream callers
        # that concatenate further sections find the same final-shape.
        assert out.endswith("\n")
        assert ADDITIONAL_CONTEXT_HEADER in out
        # The block sits BEFORE the trailing newline of the input.
        assert out == (
            "Translate this.\n\n"
            f"{ADDITIONAL_CONTEXT_HEADER}\nBrief.\n"
        )

    def test_no_trailing_newline_preserved(self):
        prompt = "Translate this."
        out = inject_context(prompt, "Brief.")
        assert not out.endswith("\n")
        assert out == (
            "Translate this.\n\n"
            f"{ADDITIONAL_CONTEXT_HEADER}\nBrief."
        )

    def test_multiline_context_kept_verbatim(self):
        body = "Line one.\n\nLine two with **bold**.\n- bullet"
        out = inject_context("prompt", body)
        # Newlines, blank lines, markdown markers all preserved
        # because the context is opaque text (brief: "any format").
        assert body in out


class TestSelfReferenceGuard:
    def test_translating_context_md_does_not_inject_itself(self, tmp_path):
        """When the source file IS ``context.md`` (e.g. caller passed
        ``glob='**/context.md'`` to translate them as content), the
        cascade MUST skip its own directory's ``context.md`` so the
        file's text is not appended to its own system prompt as
        ADDITIONAL CONTEXT and then sent again as the user payload."""
        source_root = tmp_path / "docs"
        leaf = source_root / "api"
        _write_context(source_root, "ROOT-BRIEF")
        _write_context(leaf, "LEAF-BRIEF (this is the source file body)")
        # The "source file" IS the leaf's context.md — caller wants to
        # translate it as a document.
        source_path = leaf / CONTEXT_FILENAME

        merged = resolve_context_chain(source_path, source_root, cwd=None)

        # Parent ROOT-BRIEF is in scope (context for the translation),
        # but the file's own body is NOT — otherwise it would be
        # duplicated into both the system prompt and user payload.
        assert "ROOT-BRIEF" in merged
        assert "LEAF-BRIEF" not in merged

    def test_self_reference_skipped_at_tree_root_too(self, tmp_path):
        source_root = tmp_path / "docs"
        _write_context(source_root, "ROOT-BRIEF (this is the source file)")
        # Source file IS the tree root's context.md.
        source_path = source_root / CONTEXT_FILENAME

        merged = resolve_context_chain(source_path, source_root, cwd=None)

        assert merged == ""

    def test_self_reference_via_cwd_is_also_skipped(self, tmp_path):
        """The cwd layer should also be skipped when it would point
        back at the source file (e.g. caller invoked from the same
        directory as the context.md they are translating)."""
        cwd_dir = tmp_path / "cwd"
        _write_context(cwd_dir, "CWD-BRIEF (this is the source)")
        source_path = cwd_dir / CONTEXT_FILENAME

        merged = resolve_context_chain(
            source_path, source_root=None, cwd=cwd_dir
        )

        # cwd would have read the same file; self-reference guard
        # leaves the merged context empty.
        assert merged == ""


class TestHeaderText:
    def test_header_constant_matches_brief(self):
        # Brief specifies the exact header text — keep it stable so
        # downstream prompt-cache hits don't churn on header drift.
        assert ADDITIONAL_CONTEXT_HEADER == (
            "**ADDITIONAL CONTEXT (use for proper nouns, terminology, "
            "tone, audience):**"
        )

    def test_filename_constant_matches_brief(self):
        # Brief: "context.md exactly. No fallbacks (context.txt,
        # CONTEXT.md, etc.) — keep discovery deterministic."
        assert CONTEXT_FILENAME == "context.md"
