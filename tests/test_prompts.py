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

    def test_instruction_has_reason(self):
        assert "{reason}" in Prompts.TRANSLATE_INSTRUCTION


class TestValidateTemplates:
    def test_human_template_has_placeholders(self):
        tmpl = Prompts.VALIDATE_HUMAN_TEMPLATE
        assert "{lang}" in tmpl
        assert "{criteria}" in tmpl
        assert "{source}" in tmpl
        assert "{processed}" in tmpl

    def test_human_template_formats(self):
        result = Prompts.VALIDATE_HUMAN_TEMPLATE.format(
            lang="Korean",
            criteria="check quality",
            source="original",
            processed="translated",
        )
        assert "Korean" in result

    def test_criteria_has_lang_placeholder(self):
        assert "{lang}" in Prompts.VALIDATE_CRITERIA


class TestRefineTemplates:
    def test_refine_system_has_placeholders(self):
        tmpl = Prompts.REFINE_SYSTEM_TEMPLATE
        assert "{lang}" in tmpl
        assert "{instruction}" in tmpl

    def test_refine_instruction_has_placeholders(self):
        tmpl = Prompts.REFINE_INSTRUCTION
        assert "{reason}" in tmpl
        assert "{previous_chunk}" in tmpl

    def test_refinement_validate_criteria_has_placeholder(self):
        assert "{previous_chunk}" in Prompts.REFINEMENT_VALIDATE_CRITERIA
