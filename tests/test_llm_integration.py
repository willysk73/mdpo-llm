"""Tests for LiteLLM integration — _build_messages and _call_llm."""

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
