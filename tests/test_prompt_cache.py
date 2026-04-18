"""Tests for the prompt-cache integration."""

from pathlib import Path

from mdpo_llm.processor import MarkdownProcessor


def test_anthropic_model_gets_cache_control(tmp_path, mock_completion):
    md = "# Title\n\nParagraph.\n"
    source = tmp_path / "src.md"
    source.write_text(md, encoding="utf-8")
    p = MarkdownProcessor(
        model="anthropic/claude-sonnet-4-5-20250929",
        target_lang="ko",
        batch_size=0,
        enable_prompt_cache=True,
    )
    p.process_document(source, tmp_path / "tgt.md", tmp_path / "m.po")
    call = mock_completion.completion.call_args_list[0]
    system_msg = call.kwargs["messages"][0]
    assert isinstance(system_msg["content"], list)
    part = system_msg["content"][0]
    assert part["type"] == "text"
    assert part["cache_control"] == {"type": "ephemeral"}


def test_non_anthropic_model_gets_plain_content(tmp_path, mock_completion):
    md = "# Title\n\nParagraph.\n"
    source = tmp_path / "src.md"
    source.write_text(md, encoding="utf-8")
    p = MarkdownProcessor(
        model="gpt-4o",
        target_lang="ko",
        batch_size=0,
        enable_prompt_cache=True,
    )
    p.process_document(source, tmp_path / "tgt.md", tmp_path / "m.po")
    call = mock_completion.completion.call_args_list[0]
    system_msg = call.kwargs["messages"][0]
    # String content, not the Anthropic content-parts schema.
    assert isinstance(system_msg["content"], str)


def test_prompt_cache_disabled_by_default(tmp_path, mock_completion):
    md = "# Title\n\nParagraph.\n"
    source = tmp_path / "src.md"
    source.write_text(md, encoding="utf-8")
    p = MarkdownProcessor(
        model="anthropic/claude-sonnet-4-5-20250929",
        target_lang="ko",
        batch_size=0,
    )
    p.process_document(source, tmp_path / "tgt.md", tmp_path / "m.po")
    call = mock_completion.completion.call_args_list[0]
    system_msg = call.kwargs["messages"][0]
    assert isinstance(system_msg["content"], str)
