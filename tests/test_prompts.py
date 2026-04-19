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


class TestPromptShape:
    """v0.4.1 dropped three rule families from the translate prompts:

    - The inline-code code-literal-vs-human-label distinction — LLMs
      over-preserved identifier-shaped source-language tokens like
      `{년}`, leaving target docs littered with source-language text.
    - The format-string interpolation-token preservation rule
      (`{{name}}` / `%s` / `${var}`) — intended for Python / Handlebars
      runtime templates but generalized by the LLM to every `{word}`
      including URL path parameters, which regressed the case above.
    - The bare-URL / file-path preservation rule — prevents legitimate
      translation of illustrative example paths like
      `설치/경로/config.json`.

    Refine prompts are untouched (same-language polish, different
    trade-offs).
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

    FORBIDDEN_SUBSTRINGS_ALL = [
        "keep all code as-is",
        "keep code as-is",
        "Keep inline code",
        "Keep URLs, file paths, identifiers",
        "Keep URLs, file paths, and variable/function names",
        "In code blocks:",
        "In code blocks,",
    ]

    # v0.4.1 dropped these from the translate side only. Refine still
    # keeps analogous rules because same-language polish benefits
    # from backtick verbatim.
    FORBIDDEN_SUBSTRINGS_TRANSLATE_ONLY = [
        "code-literal",
        "human labels",
        "SOURCE language",
        "interpolation tokens",
        "bare URLs and file paths",
    ]

    def test_pre_v04_rules_gone_everywhere(self):
        for name in self.ALL_INSTRUCTIONS:
            text = getattr(Prompts, name)
            for bad in self.FORBIDDEN_SUBSTRINGS_ALL:
                assert bad not in text, (
                    f"{name} still contains removed pre-v0.4 substring: {bad!r}"
                )

    def test_v041_removed_rules_gone_from_translate(self):
        for name in self.TRANSLATING_INSTRUCTIONS:
            text = getattr(Prompts, name)
            for bad in self.FORBIDDEN_SUBSTRINGS_TRANSLATE_ONLY:
                assert bad not in text, (
                    f"{name} should not contain {bad!r} — dropped in v0.4.1"
                )

    def test_code_block_rule_kept(self):
        # Translating + refining instructions all still carry the
        # "preserve code, translate comments/strings" rule for fenced
        # code blocks. This is the main safety net that survived the
        # simplification.
        for name in self.ALL_INSTRUCTIONS:
            text = getattr(Prompts, name)
            assert "Inside fenced code blocks" in text, (
                f"{name} missing the 'Inside fenced code blocks' rule"
            )
            assert "preserve the code itself verbatim" in text, (
                f"{name} missing the identifier preservation clause"
            )
            assert "comments and user-facing string literals" in text, (
                f"{name} missing the explicit comment/string "
                f"translation permission"
            )

    def test_translating_instructions_mandate_comment_translation(self):
        for name in self.TRANSLATING_INSTRUCTIONS:
            text = getattr(Prompts, name)
            assert "MUST still translate" in text, (
                f"{name} should tell the model it MUST translate code "
                f"comments / strings, not just MAY"
            )

    def test_refining_instructions_permit_polishing_code_strings(self):
        for name in self.REFINING_INSTRUCTIONS:
            text = getattr(Prompts, name)
            assert "MAY polish" in text, (
                f"{name} should permit polishing code comments / strings"
            )

    def test_refine_still_keeps_inline_code_verbatim(self):
        # Refine prompts were explicitly left untouched in v0.4.1
        # because same-language polish on backticked content can
        # silently break exact product strings. The cycle-4 guard
        # still applies here.
        for name in self.REFINING_INSTRUCTIONS:
            text = getattr(Prompts, name)
            assert "preserve the content verbatim" in text, (
                f"{name} must keep inline-code content verbatim "
                f"during refine"
            )

    def test_placeholder_token_rule_intact(self):
        for name in self.ALL_INSTRUCTIONS:
            text = getattr(Prompts, name)
            assert "\u27e6P:N\u27e7" in text, (
                f"{name} lost the ⟦P:N⟧ placeholder preservation rule"
            )

    def test_runtime_template_syntax_preserved_narrowly(self):
        # v0.4.1 narrowed the old broad "interpolation tokens" rule
        # to cover ONLY:
        #   - printf-style specifiers (%s, %d, etc.)
        #   - double-brace templates ({{name}} — Handlebars/Jinja)
        #   - dollar-brace templates (${var} — shell)
        # Single-brace tokens ({year}, {년}) are explicitly excluded
        # so URL path parameters can still be translated. This is
        # the distinction that matters — the LLM was previously
        # conflating {{name}} with {년}.
        for name in self.TRANSLATING_INSTRUCTIONS:
            text = getattr(Prompts, name)
            assert "printf-style specifiers" in text, (
                f"{name} should preserve printf-style specifiers"
            )
            assert "%s" in text and "%d" in text
            # BATCH_MULTI goes through .format(), so the literal
            # string here contains {{{{name}}}} which collapses to
            # {{name}} after one format. For the non-batch-multi
            # instructions the literal is already {{name}}. Either
            # way, the canonical rendered form should show {{name}}.
            # Render to what the LLM actually sees:
            if name == "BATCH_MULTI_TRANSLATE_INSTRUCTION":
                rendered = text.format(langs="en")
            else:
                rendered = text
            assert "`{{name}}`" in rendered, (
                f"{name} should preserve Handlebars-style "
                f"double-brace templates"
            )
            assert "`${var}`" in rendered, (
                f"{name} should preserve shell-style dollar-brace "
                f"templates"
            )
            # The critical exclusion: single-brace tokens must NOT
            # be covered by this rule so `{년}` etc. translate.
            assert "single-brace" in text, (
                f"{name} must explicitly exclude single-brace tokens "
                f"from the preservation rule"
            )

    def test_glossary_still_referenced_in_translate(self):
        # For critical mappings (product names, API parameter names)
        # the prompt directs callers to the glossary — this is the
        # only deterministic path left for cross-block identifier
        # stability after the v0.4.1 simplification.
        for name in self.TRANSLATING_INSTRUCTIONS:
            text = getattr(Prompts, name)
            assert "glossary" in text, (
                f"{name} should still mention the glossary for "
                f"critical mappings"
            )

    def test_batch_multi_instruction_formats_with_langs(self):
        formatted = Prompts.BATCH_MULTI_TRANSLATE_INSTRUCTION.format(
            langs="en, ja, zh-CN"
        )
        assert "en, ja, zh-CN" in formatted
