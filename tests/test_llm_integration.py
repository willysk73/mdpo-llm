"""Tests for LiteLLM integration — _build_messages and _call_llm."""

import json
from unittest.mock import MagicMock, patch

import pytest

from mdpo_llm.processor import MarkdownProcessor
from mdpo_llm.prompts import Prompts


@pytest.fixture
def processor():
    """MarkdownProcessor with a test model, without patching litellm."""
    return MarkdownProcessor(model="test-model", target_lang="ko")


class TestBuildMessages:
    def test_system_message_contains_target_lang(self, processor):
        messages = processor._build_messages("Hello")
        system = messages[0]
        assert system["role"] == "system"
        assert "ko" in system["content"]

    def test_user_message_is_source_text(self, processor):
        messages = processor._build_messages("Hello world")
        assert messages[-1] == {"role": "user", "content": "Hello world"}

    def test_no_reference_pairs(self, processor):
        messages = processor._build_messages("Hello")
        # System + user only
        assert len(messages) == 2

    def test_reference_pairs_become_few_shot(self, processor):
        pairs = [("src1", "tgt1"), ("src2", "tgt2")]
        messages = processor._build_messages("Hello", reference_pairs=pairs)
        # System + 2 pairs (4 messages) + user = 6
        assert len(messages) == 6
        assert messages[1] == {"role": "user", "content": "src1"}
        assert messages[2] == {"role": "assistant", "content": "tgt1"}
        assert messages[3] == {"role": "user", "content": "src2"}
        assert messages[4] == {"role": "assistant", "content": "tgt2"}

    def test_custom_system_prompt_overrides_default(self):
        proc = MarkdownProcessor(
            model="test-model", target_lang="ko", system_prompt="Custom instruction"
        )
        messages = proc._build_messages("Hello")
        system_content = messages[0]["content"]
        assert "Custom instruction" in system_content
        # Should NOT contain the default instruction text
        assert "Fenced code blocks" not in system_content

    def test_default_instruction_used_when_no_custom(self, processor):
        messages = processor._build_messages("Hello")
        system_content = messages[0]["content"]
        # Default instruction mentions fenced code blocks
        assert "Fenced code blocks" in system_content


class TestCallLLM:
    def test_litellm_completion_called_with_correct_args(self, mock_completion):
        proc = MarkdownProcessor(model="gpt-4", target_lang="ko")
        proc._call_llm("Hello")

        mock_completion.completion.assert_called_once()
        call_kwargs = mock_completion.completion.call_args
        assert call_kwargs.kwargs["model"] == "gpt-4"
        assert len(call_kwargs.kwargs["messages"]) >= 2

    def test_litellm_kwargs_forwarded(self, mock_completion):
        proc = MarkdownProcessor(
            model="gpt-4", target_lang="ko", temperature=0.5, api_key="test-key"
        )
        proc._call_llm("Hello")

        call_kwargs = mock_completion.completion.call_args.kwargs
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["api_key"] == "test-key"

    def test_post_process_applied(self, mock_completion):
        proc = MarkdownProcessor(
            model="gpt-4",
            target_lang="ko",
            post_process=lambda s: s.upper(),
        )
        result = proc._call_llm("Hello")
        assert result == "[TRANSLATED] Hello".upper()

    def test_no_post_process_returns_raw(self, mock_completion):
        proc = MarkdownProcessor(model="gpt-4", target_lang="ko")
        result = proc._call_llm("Hello")
        assert result == "[TRANSLATED] Hello"

    def test_exception_propagates(self, mock_completion):
        mock_completion.completion.side_effect = RuntimeError("API error")
        proc = MarkdownProcessor(model="gpt-4", target_lang="ko")
        with pytest.raises(RuntimeError, match="API error"):
            proc._call_llm("Hello")


class TestGlossary:
    def test_glossary_terms_in_system_message(self):
        proc = MarkdownProcessor(
            model="test-model", target_lang="ko",
            glossary={"GitHub": None, "API": "API"},
        )
        messages = proc._build_messages("Check the GitHub API docs.")
        system = messages[0]["content"]
        assert "GitHub" in system
        assert "API" in system
        assert "Glossary" in system

    def test_glossary_irrelevant_terms_excluded(self):
        proc = MarkdownProcessor(
            model="test-model", target_lang="ko",
            glossary={"GitHub": None, "Kubernetes": None},
        )
        messages = proc._build_messages("Hello world")
        system = messages[0]["content"]
        assert "Glossary" not in system
        assert "GitHub" not in system or "GitHub" in Prompts.TRANSLATE_INSTRUCTION

    def test_glossary_none_means_do_not_translate(self):
        proc = MarkdownProcessor(
            model="test-model", target_lang="ko",
            glossary={"GitHub": None},
        )
        messages = proc._build_messages("Visit GitHub today.")
        system = messages[0]["content"]
        assert '"GitHub" \u2192 do not translate' in system

    def test_glossary_with_translation(self):
        proc = MarkdownProcessor(
            model="test-model", target_lang="ko",
            glossary={"pull request": "\ud480 \ub9ac\ud018\uc2a4\ud2b8"},
        )
        messages = proc._build_messages("Open a pull request.")
        system = messages[0]["content"]
        assert '"\ud480 \ub9ac\ud018\uc2a4\ud2b8"' in system

    def test_glossary_path_json_loading(self, tmp_path):
        glossary_file = tmp_path / "glossary.json"
        glossary_file.write_text(
            json.dumps({"GitHub": None, "API": "API"}),
            encoding="utf-8",
        )
        proc = MarkdownProcessor(
            model="test-model", target_lang="ko",
            glossary_path=glossary_file,
        )
        messages = proc._build_messages("GitHub API")
        system = messages[0]["content"]
        assert "GitHub" in system
        assert "API" in system

    def test_glossary_path_per_locale_resolution(self, tmp_path):
        glossary_file = tmp_path / "glossary.json"
        glossary_file.write_text(
            json.dumps({
                "pull request": {"ko": "\ud480 \ub9ac\ud018\uc2a4\ud2b8", "ja": "\u30d7\u30eb\u30ea\u30af\u30a8\u30b9\u30c8"},
            }),
            encoding="utf-8",
        )
        proc_ko = MarkdownProcessor(
            model="test-model", target_lang="ko", glossary_path=glossary_file,
        )
        messages = proc_ko._build_messages("Open a pull request.")
        assert '"\ud480 \ub9ac\ud018\uc2a4\ud2b8"' in messages[0]["content"]

        proc_ja = MarkdownProcessor(
            model="test-model", target_lang="ja", glossary_path=glossary_file,
        )
        messages = proc_ja._build_messages("Open a pull request.")
        assert '"\u30d7\u30eb\u30ea\u30af\u30a8\u30b9\u30c8"' in messages[0]["content"]

    def test_glossary_path_locale_not_found_keeps_original(self, tmp_path):
        glossary_file = tmp_path / "glossary.json"
        glossary_file.write_text(
            json.dumps({"pull request": {"ja": "\u30d7\u30eb\u30ea\u30af\u30a8\u30b9\u30c8"}}),
            encoding="utf-8",
        )
        proc = MarkdownProcessor(
            model="test-model", target_lang="ko", glossary_path=glossary_file,
        )
        # "ko" not in the dict → resolved to None (keep original)
        messages = proc._build_messages("Open a pull request.")
        assert "do not translate" in messages[0]["content"]

    def test_glossary_inline_overrides_file(self, tmp_path):
        glossary_file = tmp_path / "glossary.json"
        glossary_file.write_text(
            json.dumps({"GitHub": "file-value"}),
            encoding="utf-8",
        )
        proc = MarkdownProcessor(
            model="test-model", target_lang="ko",
            glossary_path=glossary_file,
            glossary={"GitHub": "inline-value"},
        )
        messages = proc._build_messages("Visit GitHub.")
        system = messages[0]["content"]
        assert '"inline-value"' in system
        assert '"file-value"' not in system

    def test_no_glossary_no_change(self, processor):
        messages = processor._build_messages("Hello world")
        system = messages[0]["content"]
        assert "Glossary" not in system
