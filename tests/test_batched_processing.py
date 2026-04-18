"""Integration tests for the batched processing path in MarkdownProcessor."""

import json
from types import SimpleNamespace
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


class TestReceipt:
    """Receipt dataclass is populated on ProcessResult / DirectoryResult."""

    def test_receipt_attached_with_model_and_target(
        self, tmp_path, mock_completion, batch_processor
    ):
        md = "# T\n\nPara.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")
        result = batch_processor.process_document(
            source, tmp_path / "target.md", tmp_path / "m.po"
        )
        receipt = result.receipt
        assert receipt is not None
        assert receipt.model == "test-model"
        assert receipt.target_lang == "ko"
        assert receipt.source_path == str(source)
        assert receipt.target_path == str(tmp_path / "target.md")
        assert receipt.po_path == str(tmp_path / "m.po")
        # At least one LLM call for the batch.
        assert receipt.api_calls >= 1
        # Wall clock is monotonic and non-negative.
        assert receipt.duration_seconds >= 0.0

    def test_receipt_tokens_summed_from_usage(self, tmp_path, mock_completion):
        """Token counts from ``response.usage`` accumulate across calls."""

        def _with_usage(*args, **kwargs):
            messages = kwargs.get("messages", [])
            user_content = messages[-1]["content"]
            try:
                items = json.loads(user_content)
            except json.JSONDecodeError:
                items = None
            mock_response = MagicMock()
            if isinstance(items, dict):
                translated = {k: f"[TRANSLATED] {v}" for k, v in items.items()}
                mock_response.choices[0].message.content = json.dumps(translated)
            else:
                mock_response.choices[0].message.content = f"[TRANSLATED] {user_content}"
            mock_response.usage = SimpleNamespace(
                prompt_tokens=100, completion_tokens=50, total_tokens=150
            )
            return mock_response

        mock_completion.completion.side_effect = _with_usage

        md = "# T\n\nA.\n\nB.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")
        p = MarkdownProcessor(model="test-model", target_lang="ko", batch_size=40)
        result = p.process_document(source, tmp_path / "target.md", tmp_path / "m.po")
        receipt = result.receipt
        # Each call contributes 100/50 → totals are a non-zero multiple.
        assert receipt.input_tokens == 100 * receipt.api_calls
        assert receipt.output_tokens == 50 * receipt.api_calls
        assert receipt.total_tokens == receipt.input_tokens + receipt.output_tokens

    def test_receipt_unpriced_model_reports_none_costs(
        self, tmp_path, mock_completion, batch_processor
    ):
        """Unknown models fall through to ``None`` costs rather than ``$0``."""
        md = "# T\n\nPara.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")
        result = batch_processor.process_document(
            source, tmp_path / "target.md", tmp_path / "m.po"
        )
        receipt = result.receipt
        # ``test-model`` isn't in ``litellm.model_cost``.
        assert receipt.input_cost_per_1m_usd is None
        assert receipt.output_cost_per_1m_usd is None
        assert receipt.total_cost_usd is None
        # Rendering still works and uses the em-dash fallback.
        rendered = receipt.render()
        assert "—" in rendered

    def test_receipt_render_contains_paths_and_tokens(
        self, tmp_path, mock_completion, batch_processor
    ):
        md = "# T\n\nPara.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")
        result = batch_processor.process_document(
            source, tmp_path / "target.md", tmp_path / "m.po"
        )
        rendered = result.receipt.render()
        assert "Translation receipt" in rendered
        assert "test-model" in rendered
        assert "ko" in rendered
        assert str(source) in rendered

    def test_save_po_failure_still_carries_partial_receipt(
        self, tmp_path, mock_completion
    ):
        """A save_po failure happens after LLM calls; the exception must
        still carry a partial Receipt so the billed cost is visible."""

        def _with_usage(*args, **kwargs):
            messages = kwargs.get("messages", [])
            user_content = messages[-1]["content"]
            try:
                items = json.loads(user_content)
            except json.JSONDecodeError:
                items = None
            resp = MagicMock()
            if isinstance(items, dict):
                translated = {k: f"[TRANSLATED] {v}" for k, v in items.items()}
                resp.choices[0].message.content = json.dumps(translated)
            else:
                resp.choices[0].message.content = f"[TRANSLATED] {user_content}"
            resp.usage = SimpleNamespace(
                prompt_tokens=8, completion_tokens=4, total_tokens=12
            )
            return resp

        mock_completion.completion.side_effect = _with_usage

        md = "# T\n\nPara.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")

        p = MarkdownProcessor(model="test-model", target_lang="ko", batch_size=40)

        original_save = p.po_manager.save_po

        def _boom_save(po_file, po_path):
            raise OSError("simulated disk full")

        # Patch the class method via a side-step: the processor creates a
        # fresh POManager inside process_document, so patch the class.
        from mdpo_llm.manager import POManager as PM

        original_pm_save = PM.save_po
        PM.save_po = lambda self, po_file, po_path: _boom_save(po_file, po_path)  # type: ignore[assignment]
        try:
            with pytest.raises(OSError) as excinfo:
                p.process_document(
                    source, tmp_path / "target.md", tmp_path / "m.po"
                )
        finally:
            PM.save_po = original_pm_save  # type: ignore[assignment]

        partial = getattr(excinfo.value, "partial_receipt", None)
        assert partial is not None, (
            "save_po failure after billed LLM calls must surface partial_receipt"
        )
        assert partial.api_calls >= 1
        assert partial.input_tokens >= 8
        assert partial.output_tokens >= 4

    def test_pre_api_failure_omits_partial_receipt(
        self, tmp_path, mock_completion, batch_processor
    ):
        """A failure before any LLM call must NOT attach a zero-usage
        partial_receipt — that would mislead operators into believing
        tokens were billed."""
        missing = tmp_path / "does-not-exist.md"
        with pytest.raises(FileNotFoundError) as excinfo:
            batch_processor.process_document(
                missing, tmp_path / "target.md", tmp_path / "m.po"
            )
        # No receipt attribute — zero-usage receipts are NOT emitted on
        # pre-API failures.
        assert getattr(excinfo.value, "partial_receipt", None) is None

    def test_single_file_exception_carries_partial_receipt(
        self, tmp_path, mock_completion
    ):
        """A post-API failure attaches a partial Receipt to the exception
        so single-file CLI callers can still surface billed tokens."""

        def _with_usage(*args, **kwargs):
            messages = kwargs.get("messages", [])
            user_content = messages[-1]["content"]
            try:
                items = json.loads(user_content)
            except json.JSONDecodeError:
                items = None
            mock_response = MagicMock()
            if isinstance(items, dict):
                translated = {k: f"[TRANSLATED] {v}" for k, v in items.items()}
                mock_response.choices[0].message.content = json.dumps(translated)
            else:
                mock_response.choices[0].message.content = f"[TRANSLATED] {user_content}"
            mock_response.usage = SimpleNamespace(
                prompt_tokens=12, completion_tokens=7, total_tokens=19
            )
            return mock_response

        mock_completion.completion.side_effect = _with_usage

        md = "# T\n\nPara.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")

        p = MarkdownProcessor(model="test-model", target_lang="ko", batch_size=40)

        def _boom(content, target):
            raise RuntimeError("simulated post-API failure")

        p._save_processed_document = _boom  # type: ignore[method-assign]

        with pytest.raises(RuntimeError) as excinfo:
            p.process_document(source, tmp_path / "target.md", tmp_path / "m.po")

        partial = getattr(excinfo.value, "partial_receipt", None)
        assert partial is not None, "exception should carry partial_receipt"
        assert partial.api_calls >= 1
        assert partial.input_tokens >= 12
        assert partial.output_tokens >= 7
        assert partial.model == "test-model"
        assert partial.source_path == str(source)

    def test_directory_receipt_preserves_usage_on_worker_exception(
        self, tmp_path, mock_completion
    ):
        """When a worker raises after issuing LLM calls, its tokens still
        flow into the directory-level receipt.  Dropping them would make
        ``translate-dir`` silently under-report billed usage whenever
        reconstruction / save fails."""

        def _with_usage(*args, **kwargs):
            messages = kwargs.get("messages", [])
            user_content = messages[-1]["content"]
            try:
                items = json.loads(user_content)
            except json.JSONDecodeError:
                items = None
            mock_response = MagicMock()
            if isinstance(items, dict):
                translated = {k: f"[TRANSLATED] {v}" for k, v in items.items()}
                mock_response.choices[0].message.content = json.dumps(translated)
            else:
                mock_response.choices[0].message.content = f"[TRANSLATED] {user_content}"
            mock_response.usage = SimpleNamespace(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            )
            return mock_response

        mock_completion.completion.side_effect = _with_usage

        src = tmp_path / "src"
        src.mkdir()
        (src / "ok.md").write_text("# Title\n\nPara.\n", encoding="utf-8")
        (src / "boom.md").write_text("# Title\n\nPara.\n", encoding="utf-8")

        p = MarkdownProcessor(model="test-model", target_lang="ko", batch_size=40)

        # Force the save step to fail ONLY for boom.md, after LLM calls have
        # already been made and billed.  This mirrors the real-world case
        # Codex flagged: post-API failure dropping its receipt on the floor.
        original_save = p._save_processed_document

        def _failing_save(content, target):
            if target.name == "boom.md":
                raise RuntimeError("simulated save failure")
            return original_save(content, target)

        p._save_processed_document = _failing_save  # type: ignore[method-assign]

        result = p.process_directory(
            src, tmp_path / "out", tmp_path / "po", max_workers=1
        )
        # One file failed, one succeeded.
        assert result.files_failed == 1
        # Aggregate receipt includes BOTH workers' tokens, not just the
        # successful one's.  Each batch call books 10/5 tokens and both
        # files issue at least one call before the failure point.
        assert result.receipt.api_calls >= 2
        assert result.receipt.input_tokens >= 20
        assert result.receipt.output_tokens >= 10

    def test_directory_result_aggregates_tokens(self, tmp_path, mock_completion):
        """DirectoryResult.receipt sums usage across files."""

        def _with_usage(*args, **kwargs):
            messages = kwargs.get("messages", [])
            user_content = messages[-1]["content"]
            try:
                items = json.loads(user_content)
            except json.JSONDecodeError:
                items = None
            mock_response = MagicMock()
            if isinstance(items, dict):
                translated = {k: f"[TRANSLATED] {v}" for k, v in items.items()}
                mock_response.choices[0].message.content = json.dumps(translated)
            else:
                mock_response.choices[0].message.content = f"[TRANSLATED] {user_content}"
            mock_response.usage = SimpleNamespace(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            )
            return mock_response

        mock_completion.completion.side_effect = _with_usage

        src = tmp_path / "src"
        src.mkdir()
        for name in ("a.md", "b.md"):
            (src / name).write_text("# H\n\nPara.\n", encoding="utf-8")

        p = MarkdownProcessor(model="test-model", target_lang="ko", batch_size=40)
        result = p.process_directory(
            src, tmp_path / "out", tmp_path / "po", max_workers=1
        )
        receipt = result.receipt
        assert receipt is not None
        # Two files, each at least one call → aggregate has both.
        per_file_totals = [r.receipt.input_tokens for r in result.results if hasattr(r, "receipt")]
        assert sum(per_file_totals) == receipt.input_tokens
        assert receipt.api_calls == sum(r.receipt.api_calls for r in result.results)


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
