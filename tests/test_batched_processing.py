"""Integration tests for the batched processing path in MarkdownProcessor."""

import json
from unittest.mock import MagicMock

import pytest

from mdpo_llm.processor import MarkdownProcessor


@pytest.fixture
def batch_processor(mock_completion):
    return MarkdownProcessor(model="test-model", target_lang="ko", batch_size=40)


class TestBatchedProcessing:
    def test_single_llm_call_covers_many_blocks(
        self, tmp_path, mock_completion, batch_processor
    ):
        md = "# T\n\nP1.\n\nP2.\n\nP3.\n\nP4.\n\nP5.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")

        batch_processor.process_document(
            source, tmp_path / "target.md", tmp_path / "m.po"
        )
        # All blocks translated in one batch → one completion call.
        assert mock_completion.completion.call_count == 1

    def test_json_user_payload(self, tmp_path, mock_completion, batch_processor):
        md = "# Title\n\nPara A.\n\nPara B.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")
        batch_processor.process_document(
            source, tmp_path / "target.md", tmp_path / "m.po"
        )
        call = mock_completion.completion.call_args_list[0]
        messages = call.kwargs["messages"]
        user_msg = messages[-1]["content"]
        parsed = json.loads(user_msg)
        assert isinstance(parsed, dict)
        # Three processable blocks.
        assert len(parsed) == 3

    def test_po_populated_after_batch(
        self, tmp_path, mock_completion, batch_processor
    ):
        md = "# Title\n\nParagraph one.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")

        result = batch_processor.process_document(
            source, tmp_path / "target.md", tmp_path / "m.po"
        )
        assert result["translation_stats"]["processed"] >= 2
        content = (tmp_path / "target.md").read_text(encoding="utf-8")
        assert "[TRANSLATED]" in content

    def test_rerun_produces_batch_of_one(
        self, tmp_path, mock_completion, batch_processor
    ):
        md = "# Title\n\nPara A.\n\nPara B.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")
        po_path = tmp_path / "m.po"

        batch_processor.process_document(source, tmp_path / "target.md", po_path)
        first_calls = mock_completion.completion.call_count

        # Change only one paragraph; rerun.
        source.write_text(
            "# Title\n\nPara A.\n\nChanged para.\n", encoding="utf-8"
        )
        batch_processor.process_document(source, tmp_path / "target.md", po_path)
        second_calls = mock_completion.completion.call_count - first_calls
        assert second_calls == 1  # One batch-of-one call.

    def test_bisection_on_bad_json(self, tmp_path, mock_completion):
        """Malformed response on the batch → bisect down to single entries."""
        call_count = {"n": 0}

        def _side_effect(*args, **kwargs):
            call_count["n"] += 1
            messages = kwargs["messages"]
            user_content = messages[-1]["content"]
            try:
                items = json.loads(user_content)
            except json.JSONDecodeError:
                items = None

            mock_response = MagicMock()
            if isinstance(items, dict) and len(items) > 1:
                # Break the first call by returning non-JSON.
                mock_response.choices[0].message.content = "totally bogus"
            elif isinstance(items, dict):
                # Single-entry → succeed.
                translated = {k: f"[TRANSLATED] {v}" for k, v in items.items()}
                mock_response.choices[0].message.content = json.dumps(translated)
            else:
                mock_response.choices[0].message.content = (
                    f"[TRANSLATED] {user_content}"
                )
            return mock_response

        mock_completion.completion.side_effect = _side_effect

        md = "# T\n\nA.\n\nB.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")

        p = MarkdownProcessor(model="test-model", target_lang="ko", batch_size=40)
        result = p.process_document(
            source, tmp_path / "target.md", tmp_path / "m.po"
        )
        # After bisection every block should eventually translate.
        assert result["translation_stats"]["processed"] >= 3

    def test_batch_size_cap_splits_calls(self, tmp_path, mock_completion):
        """batch_size limits items per call."""
        md = "# Title\n" + "".join([f"\nPara {i}.\n" for i in range(10)])
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")

        # batch_size=3 → at least ceil((1 heading + 10 paras) / 3) = 4 calls.
        p = MarkdownProcessor(model="test-model", target_lang="ko", batch_size=3)
        result = p.process_document(
            source, tmp_path / "target.md", tmp_path / "m.po"
        )
        assert mock_completion.completion.call_count >= 4
        # Stats must count every actual LLM call, not just section groups.
        assert (
            result.translation_stats.batched_calls
            == mock_completion.completion.call_count
        )

    def test_json_response_format_kwarg(
        self, tmp_path, mock_completion, batch_processor
    ):
        """When LiteLLM advertises response_format support, it is forwarded.

        The mock returns a non-list shape from ``get_supported_openai_params``
        which falls back to "assume supported", so the kwarg is set.
        """
        md = "# T\n\nPara.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")
        batch_processor.process_document(
            source, tmp_path / "target.md", tmp_path / "m.po"
        )
        call = mock_completion.completion.call_args_list[0]
        assert call.kwargs.get("response_format") == {"type": "json_object"}

    def test_json_mode_skipped_when_provider_lacks_support(
        self, tmp_path, mock_completion
    ):
        """When LiteLLM lists supported params without response_format, it is dropped."""
        mock_completion.get_supported_openai_params.return_value = [
            "temperature",
            "max_tokens",
        ]
        md = "# T\n\nPara.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")
        p = MarkdownProcessor(model="test-model", target_lang="ko", batch_size=40)
        p.process_document(source, tmp_path / "target.md", tmp_path / "m.po")
        call = mock_completion.completion.call_args_list[0]
        assert "response_format" not in call.kwargs


class TestInplaceModeBatched:
    def test_inplace_with_batching(self, tmp_path, mock_completion):
        md = "# Title\n\nSome text here.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")
        po_path = tmp_path / "m.po"

        p = MarkdownProcessor(model="test-model", target_lang="ko", batch_size=40)
        p.process_document(source, tmp_path / "target.md", po_path, inplace=True)

        po = p.po_manager.load_or_create_po(po_path)
        for entry in po:
            if entry.msgstr:
                assert entry.msgid == entry.msgstr


class TestSectionAwareChunking:
    def test_cross_section_boundary_triggers_split(self, tmp_path, mock_completion):
        """Two top-level sections with batch_size past the soft threshold should split."""
        md = (
            "# Section A\n\n"
            "Para A1.\n\n"
            "Para A2.\n\n"
            "# Section B\n\n"
            "Para B1.\n\n"
            "Para B2.\n"
        )
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")
        # batch_size=4 so soft=2; after 2 items into A we're willing to cut on section change.
        p = MarkdownProcessor(model="test-model", target_lang="ko", batch_size=4)
        p.process_document(source, tmp_path / "target.md", tmp_path / "m.po")
        # Should produce at least 2 calls (one per section).
        assert mock_completion.completion.call_count >= 2


class TestValidationGate:
    def test_validation_marks_fuzzy_on_failure(self, tmp_path, mock_completion):
        """When translation fails validator (e.g. lost heading level), mark fuzzy."""

        def _side_effect(*args, **kwargs):
            messages = kwargs.get("messages", [])
            user_content = messages[-1]["content"]
            try:
                items = json.loads(user_content)
            except json.JSONDecodeError:
                items = None

            resp = MagicMock()
            if isinstance(items, dict):
                # Strip heading markers → heading-level check fails.
                out = {}
                for k, v in items.items():
                    stripped = v.lstrip("# ").strip() or "요약"
                    out[k] = stripped
                resp.choices[0].message.content = json.dumps(out)
            else:
                resp.choices[0].message.content = "번역"
            return resp

        mock_completion.completion.side_effect = _side_effect

        md = "# Title\n\n텍스트.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")
        po_path = tmp_path / "m.po"

        p = MarkdownProcessor(
            model="test-model",
            target_lang="ko",
            batch_size=40,
            validation="conservative",
        )
        p.process_document(source, tmp_path / "target.md", po_path)

        po = p.po_manager.load_or_create_po(po_path)
        # At least one entry should be flagged fuzzy by the validator.
        assert any("fuzzy" in e.flags for e in po)


class TestEstimate:
    def test_estimate_returns_expected_keys(self, tmp_path, mock_completion):
        md = "# T\n\nPara one.\n\nPara two.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")
        p = MarkdownProcessor(model="test-model", target_lang="ko", batch_size=40)
        est = p.estimate(source)
        assert est["pending_blocks"] >= 2
        assert est["input_tokens"] > 0
        assert est["output_tokens"] > 0
        assert est["batches"] >= 1
        # No API call should have been made.
        assert mock_completion.completion.call_count == 0
