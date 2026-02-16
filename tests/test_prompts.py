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
        # No placeholders — instruction is a plain string now
        assert isinstance(Prompts.TRANSLATE_INSTRUCTION, str)
        assert "Markdown" in Prompts.TRANSLATE_INSTRUCTION
