"""Tests for prompt template placeholder validation."""

from mdpo_llm.prompts import Prompts


class TestTranslateTemplates:
    def test_system_template_has_placeholders(self):
        tmpl = Prompts.TRANSLATE_SYSTEM_TEMPLATE
        assert "{lang}" in tmpl
        assert "{instruction}" in tmpl

    def test_system_template_formats(self):
        result = Prompts.TRANSLATE_SYSTEM_TEMPLATE.format(
            lang="Korean", instruction="Be concise"
        )
        assert "Korean" in result
        assert "Be concise" in result

    def test_instruction_is_static_string(self):
        assert isinstance(Prompts.TRANSLATE_INSTRUCTION, str)
        assert "Markdown" in Prompts.TRANSLATE_INSTRUCTION


class TestPromptCleanup:
    """v0.4 replaced the old sweeping "keep all code as-is" clauses with
    a split that preserves identifiers/paths/URLs while explicitly
    allowing comment/string translation and source-language-label
    translation inside inline code.

    These assertions lock the shape so a future edit can't silently
    revert to the earlier behaviour.
    """

    TRANSLATING_INSTRUCTIONS = [
        "TRANSLATE_INSTRUCTION",
        "BATCH_TRANSLATE_INSTRUCTION",
        "BATCH_MULTI_TRANSLATE_INSTRUCTION",
    ]

    REFINING_INSTRUCTIONS = [
        "REFINE_INSTRUCTION",
        "BATCH_REFINE_INSTRUCTION",
    ]

    ALL_INSTRUCTIONS = TRANSLATING_INSTRUCTIONS + REFINING_INSTRUCTIONS

    FORBIDDEN_SUBSTRINGS = [
        "keep all code as-is",
        "keep code as-is",
        "Keep inline code",
        "Keep URLs, file paths, identifiers",
        "Keep URLs, file paths, and variable/function names",
        "In code blocks:",
        "In code blocks,",
    ]

    def test_forbidden_pre_v04_rules_gone(self):
        for name in self.ALL_INSTRUCTIONS:
            text = getattr(Prompts, name)
            for bad in self.FORBIDDEN_SUBSTRINGS:
                assert bad not in text, (
                    f"{name} still contains removed pre-v0.4 substring: {bad!r}"
                )

    def test_code_block_rule_split_into_preserve_plus_translate(self):
        # Every instruction now preserves code syntax but calls out
        # translatable natural-language content inside code blocks —
        # the two halves of the split.
        for name in self.ALL_INSTRUCTIONS:
            text = getattr(Prompts, name)
            assert "Inside fenced code blocks" in text, (
                f"{name} missing the new 'Inside fenced code blocks' rule"
            )
            assert "preserve the code itself verbatim" in text, (
                f"{name} missing the identifier preservation clause"
            )
            assert "comments and user-facing string literals" in text, (
                f"{name} missing the explicit comment/string translation "
                f"permission"
            )

    def test_translating_instructions_mandate_comment_translation(self):
        # "MUST still translate" is the load-bearing strengthening that
        # addresses LLMs previously being over-conservative about
        # translating comments/strings inside code blocks.
        for name in self.TRANSLATING_INSTRUCTIONS:
            text = getattr(Prompts, name)
            assert "MUST still translate" in text, (
                f"{name} should tell the model it MUST translate code "
                f"comments / strings, not just MAY"
            )

    def test_refining_instructions_permit_polishing_code_strings(self):
        # Refine doesn't translate so it says MAY, not MUST.
        for name in self.REFINING_INSTRUCTIONS:
            text = getattr(Prompts, name)
            assert "MAY polish" in text, (
                f"{name} should permit polishing code comments / strings"
            )

    def test_inline_code_prose_vs_identifier_split(self):
        # The inline-code rule in v0.4 distinguishes code-literal
        # content (preserve) from human labels (translate) — NOT by
        # source vs target script, which would regress same-script
        # pairs like English → French. The rule must work for
        # `Save` → `Enregistrer` and `게임코드` → `GameCode` alike.
        for name in self.TRANSLATING_INSTRUCTIONS:
            text = getattr(Prompts, name)
            assert "Inside inline code spans" in text, (
                f"{name} missing the inline code span rule"
            )
            assert "code-literal" in text and "human labels" in text, (
                f"{name} must use the code-literal vs human-labels "
                f"distinction, not a script-based gate"
            )
            # Guard against regression to the script-based gate.
            assert "SOURCE language" not in text, (
                f"{name} still uses the script-based gate; the rule "
                f"must be prose-vs-identifier to cover same-script pairs"
            )

    def test_translating_instructions_mention_glossary(self):
        # For critical mappings the prompt directs callers to the
        # glossary. Refine instructions don't translate, so they
        # don't need to point at the glossary.
        for name in self.TRANSLATING_INSTRUCTIONS:
            text = getattr(Prompts, name)
            assert "glossary" in text, (
                f"{name} should reference the glossary for critical "
                f"mappings"
            )

    def test_refine_keeps_inline_code_verbatim(self):
        # Refine is same-language, so there is no "translate to target"
        # justification for touching backticked fragments; those
        # fragments may name exact UI labels or config values the
        # docs must keep matching. v0.4 cycle 4 removed the
        # "prose fragments may be polished" permission after Codex
        # flagged it; lock that out.
        for name in self.REFINING_INSTRUCTIONS:
            text = getattr(Prompts, name)
            assert "preserve the content verbatim" in text, (
                f"{name} must keep inline-code content verbatim "
                f"during refine"
            )
            # Guard against the regressed permission.
            assert "may be polished" not in text, (
                f"{name} must not permit polishing backticked content"
            )

    def test_bare_url_preservation_still_called_out(self):
        # Codex flagged that URLs in prose get corrupted without this;
        # the rule survives the cleanup for that reason.
        for name in self.ALL_INSTRUCTIONS:
            text = getattr(Prompts, name)
            assert "bare URLs and file paths" in text, (
                f"{name} dropped the bare-URL / file-path preservation "
                f"rule — reintroduced in v0.4 because nothing downstream "
                f"restores them"
            )

    def test_interpolation_tokens_still_preserved(self):
        # `{{name}}`, `%s`, `${var}` are format syntax, not identifiers,
        # so they keep their own preservation rule.
        for name in self.ALL_INSTRUCTIONS:
            text = getattr(Prompts, name)
            assert "interpolation tokens" in text, (
                f"{name} must still preserve format-string interpolation "
                f"tokens"
            )
            assert "%s" in text

    def test_placeholder_token_rule_intact(self):
        # U+27E6 / U+27E7 placeholder preservation is the load-bearing
        # hard-protection path; it must survive.
        for name in self.ALL_INSTRUCTIONS:
            text = getattr(Prompts, name)
            assert "\u27e6P:N\u27e7" in text, (
                f"{name} lost the ⟦P:N⟧ placeholder preservation rule"
            )

    def test_batch_multi_instruction_formats_with_langs(self):
        # This instruction is the only one that itself goes through
        # .format(langs=...). Check brace escaping survived so the
        # downstream .format() does not raise.
        formatted = Prompts.BATCH_MULTI_TRANSLATE_INSTRUCTION.format(
            langs="en, ja, zh-CN"
        )
        assert "en, ja, zh-CN" in formatted
        assert "`{{name}}`" in formatted
        assert "`${{var}}`" in formatted
