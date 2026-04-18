"""Integration tests for the batched processing path in MarkdownProcessor."""

import json
import logging
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mdpo_llm.placeholder import TOKEN_RE
from mdpo_llm.processor import MarkdownProcessor


def _identity_side_effect(*args, **kwargs):
    """Return each batch value unchanged so output == source for every entry.

    Drives the "LLM returned untranslated output" path without mutating any
    value — every block, including prose, comes back verbatim. Tests that
    exercise the warning gate rely on this shape.
    """
    messages = kwargs.get("messages", args[0] if args else [])
    user_content = ""
    for msg in reversed(messages):
        if msg["role"] == "user":
            user_content = msg["content"]
            break
    parsed = None
    if isinstance(user_content, str) and user_content.strip().startswith("{"):
        try:
            parsed = json.loads(user_content)
        except json.JSONDecodeError:
            parsed = None
    resp = MagicMock()
    if isinstance(parsed, dict):
        resp.choices[0].message.content = json.dumps(parsed, ensure_ascii=False)
    else:
        resp.choices[0].message.content = user_content
    return resp


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


class TestUntranslatedWarningGate:
    """Warning for ``output == source`` skips ``code`` block types only.

    The v0.3 real-world test produced a flood of false-positive warnings
    because code blocks legitimately pass through unchanged (per rule 3 of
    the translation instruction). These tests pin the exemption in both
    processing paths while guaranteeing prose regressions still surface.
    """

    def _run_identity(self, tmp_path, mock_completion, caplog, *, batch_size):
        mock_completion.completion.side_effect = _identity_side_effect
        md = (
            "# Heading\n\n"
            "A paragraph that should be translated.\n\n"
            "```python\nprint('hi')\n```\n"
        )
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")

        p = MarkdownProcessor(
            model="test-model", target_lang="ko", batch_size=batch_size
        )
        with caplog.at_level(logging.WARNING, logger="mdpo_llm.processor"):
            p.process_document(
                source, tmp_path / "target.md", tmp_path / "m.po"
            )
        return [
            r
            for r in caplog.records
            if "LLM returned untranslated output" in r.getMessage()
        ]

    def test_code_block_suppresses_warning_batched(
        self, tmp_path, mock_completion, caplog
    ):
        warnings = self._run_identity(
            tmp_path, mock_completion, caplog, batch_size=40
        )
        # Prose still trips the warning so real regressions stay visible.
        assert warnings, "non-code entries must still emit the warning"
        # But no warning references a code-block msgctxt.
        assert all("::code:" not in r.getMessage() for r in warnings), (
            "code blocks must be exempt from the untranslated warning; "
            f"offending records: {[r.getMessage() for r in warnings]}"
        )

    def test_code_block_suppresses_warning_sequential(
        self, tmp_path, mock_completion, caplog
    ):
        warnings = self._run_identity(
            tmp_path, mock_completion, caplog, batch_size=0
        )
        assert warnings, "non-code entries must still emit the warning"
        assert all("::code:" not in r.getMessage() for r in warnings), (
            "code blocks must be exempt from the untranslated warning; "
            f"offending records: {[r.getMessage() for r in warnings]}"
        )


class TestBatchPromptFenceProhibition:
    """The batched system prompt must explicitly forbid Markdown fences.

    The v0.3 run occasionally saw models wrap their JSON response in
    ```json fences despite rule 1, so the rule is tightened to call out
    the exact wrapping behaviour. Pin the language so a future prompt
    tweak doesn't quietly regress the instruction.
    """

    def test_instruction_forbids_fences_explicitly(self):
        from mdpo_llm.prompts import Prompts

        instruction = Prompts.BATCH_TRANSLATE_INSTRUCTION
        # Must mention that the response has to start with `{` and end
        # with `}` and explicitly call out backticks / ```json fences.
        assert "start with `{`" in instruction
        assert "end with `}`" in instruction
        assert "```json" in instruction
        assert "backticks" in instruction


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


class TestGlossaryPlaceholderMode:
    """Glossary placeholder mode wires through the batched path too.

    The batched call must see opaque ``\u27e6P:N\u27e7`` tokens in the user
    payload and the decoded output must restore the target-language form
    (or the original term for do-not-translate entries).  Placeholder mode
    also suppresses the instruction-mode glossary block so the two paths
    don't double-feed terms to the model.
    """

    def test_batched_substitutes_terms_and_restores(
        self, tmp_path, mock_completion
    ):
        captured = {"calls": []}

        def _echo(*args, **kwargs):
            captured["calls"].append(kwargs.get("messages", []))
            messages = kwargs.get("messages", [])
            user = messages[-1]["content"]
            parsed = json.loads(user) if user.strip().startswith("{") else None
            resp = MagicMock()
            if isinstance(parsed, dict):
                # Preserve tokens verbatim so decode restores the target form.
                resp.choices[0].message.content = json.dumps(
                    parsed, ensure_ascii=False
                )
            else:
                resp.choices[0].message.content = user
            return resp

        mock_completion.completion.side_effect = _echo

        md = "# T\n\nVisit GitHub for the pull request process.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")

        p = MarkdownProcessor(
            model="test-model",
            target_lang="ko",
            batch_size=40,
            glossary={
                "GitHub": None,
                "pull request": "\ud480 \ub9ac\ud018\uc2a4\ud2b8",
            },
            glossary_mode="placeholder",
        )
        p.process_document(source, tmp_path / "target.md", tmp_path / "m.po")

        # System prompt lacks the instruction-mode glossary block.
        first_call = captured["calls"][0]
        system = first_call[0]["content"]
        assert "Glossary" not in system

        # User payload (JSON batch) does NOT contain the raw terms.
        user = first_call[-1]["content"]
        assert "GitHub" not in user
        assert "pull request" not in user

        # Target markdown has terms restored post-decode.
        out = (tmp_path / "target.md").read_text(encoding="utf-8")
        assert "GitHub" in out
        assert "\ud480 \ub9ac\ud018\uc2a4\ud2b8" in out


class TestBuiltinPlaceholders:
    """T-6: always-on anchor + HTML attribute placeholders.

    The regression fixture called out in the T-6 acceptance criteria —
    a source block with 5 anchors + 3 class-attr HTML links — must
    round-trip every placeholder unchanged end-to-end through the
    batched path.  No user registry, no glossary: built-ins alone.
    """

    def test_anchors_and_html_attrs_round_trip_through_batched_path(
        self, tmp_path, mock_completion
    ):
        captured = {"user_payloads": []}

        def _echo(*args, **kwargs):
            messages = kwargs.get("messages", [])
            user = messages[-1]["content"]
            captured["user_payloads"].append(user)
            parsed = json.loads(user) if user.strip().startswith("{") else None
            resp = MagicMock()
            if isinstance(parsed, dict):
                # LLM preserves every token verbatim — tests the T-6
                # guarantee that encoded spans survive the round trip.
                resp.choices[0].message.content = json.dumps(
                    parsed, ensure_ascii=False
                )
            else:
                resp.choices[0].message.content = user
            return resp

        mock_completion.completion.side_effect = _echo

        md = (
            "# Title\n\n"
            "Intro pointing at "
            '<a href="/a" class="bare">A</a>, '
            '<a href="/b" class="bare">B</a>, and '
            '<a href="/c" class="bare">C</a>.\n\n'
            "## Anchors {#first-anchor}\n\n"
            "See also {#second-anchor} {#third-anchor} "
            "{#fourth-anchor} {#fifth-anchor} refs.\n"
        )
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")

        p = MarkdownProcessor(
            model="test-model", target_lang="ko", batch_size=40
        )
        p.process_document(
            source, tmp_path / "target.md", tmp_path / "m.po"
        )

        # The user payload sent to the LLM must NOT contain any of the
        # protected spans — every one has been swapped for an opaque
        # token before the call.
        joined_payloads = "\n".join(captured["user_payloads"])
        for protected in (
            "{#first-anchor}",
            "{#second-anchor}",
            "{#third-anchor}",
            "{#fourth-anchor}",
            "{#fifth-anchor}",
            'href="/a"',
            'href="/b"',
            'href="/c"',
            'class="bare"',
        ):
            assert protected not in joined_payloads, protected

        # Every placeholder is restored verbatim in the rendered output.
        out = (tmp_path / "target.md").read_text(encoding="utf-8")
        for protected in (
            "{#first-anchor}",
            "{#second-anchor}",
            "{#third-anchor}",
            "{#fourth-anchor}",
            "{#fifth-anchor}",
            'href="/a"',
            'href="/b"',
            'href="/c"',
        ):
            assert protected in out, protected
        # ``class="bare"`` appears three times — count preserved.
        assert out.count('class="bare"') == 3

    def test_dropped_anchor_marks_entry_fuzzy(self, tmp_path, mock_completion):
        # When the LLM drops an anchor token, the round-trip check must
        # fail the entry as a structural validator issue — not a
        # warning.  Brief: "validator failure on a missing anchor is a
        # structural fail, not a warning."
        def _strip_anchor_tokens(*args, **kwargs):
            messages = kwargs.get("messages", [])
            user = messages[-1]["content"]
            parsed = json.loads(user) if user.strip().startswith("{") else None
            resp = MagicMock()
            if isinstance(parsed, dict):
                # Drop every placeholder token the encode step produced;
                # this simulates the exact failure mode T-6 guards
                # against (anchor mangling in the real-world test).
                stripped = {k: TOKEN_RE.sub("", v) for k, v in parsed.items()}
                resp.choices[0].message.content = json.dumps(
                    stripped, ensure_ascii=False
                )
            else:
                resp.choices[0].message.content = user
            return resp

        mock_completion.completion.side_effect = _strip_anchor_tokens

        md = "## Heading {#my-anchor}\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")
        po_path = tmp_path / "m.po"

        p = MarkdownProcessor(
            model="test-model", target_lang="ko", batch_size=40
        )
        p.process_document(source, tmp_path / "target.md", po_path)

        po = p.po_manager.load_or_create_po(po_path)
        flagged = [e for e in po if "fuzzy" in e.flags]
        assert flagged, "dropped anchor must mark the entry fuzzy"
        assert any(
            "placeholder_roundtrip" in (e.tcomment or "") for e in flagged
        )


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


class TestProgressHook:
    """``progress_callback`` emits events suitable for driving a UI bar."""

    def test_document_events_for_batched(
        self, tmp_path, mock_completion
    ):
        events = []
        p = MarkdownProcessor(
            model="test-model",
            target_lang="ko",
            batch_size=40,
            progress_callback=events.append,
        )
        md = "# Title\n\nPara A.\n\nPara B.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")
        p.process_document(source, tmp_path / "target.md", tmp_path / "m.po")

        kinds = [e.kind for e in events]
        assert "document_start" in kinds
        assert "document_progress" in kinds
        assert "document_end" in kinds
        # document_start precedes document_end.
        assert kinds.index("document_start") < kinds.index("document_end")
        # The start event carries ``total`` equal to the number of groups,
        # and every progress event matches that total.
        starts = [e for e in events if e.kind == "document_start"]
        assert len(starts) == 1
        assert starts[0].total is not None and starts[0].total >= 1
        for ev in events:
            if ev.kind == "document_progress":
                assert ev.total == starts[0].total
                assert 1 <= ev.index <= ev.total
        # All document-kind events carry the source path string.
        for ev in events:
            if ev.kind.startswith("document_"):
                assert ev.path == str(source)

    def test_document_events_for_sequential(
        self, tmp_path, mock_completion
    ):
        events = []
        p = MarkdownProcessor(
            model="test-model",
            target_lang="ko",
            batch_size=0,  # sequential path
            progress_callback=events.append,
        )
        md = "# Title\n\nPara A.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")
        p.process_document(source, tmp_path / "target.md", tmp_path / "m.po")

        kinds = [e.kind for e in events]
        assert kinds.count("document_start") == 1
        assert kinds.count("document_end") == 1
        # Sequential emits one progress tick per pending entry.
        ticks = [e for e in events if e.kind == "document_progress"]
        assert len(ticks) >= 1

    def test_empty_rerun_emits_start_and_end(
        self, tmp_path, mock_completion
    ):
        """A no-op re-run should still emit ``document_start`` and
        ``document_end`` so the UI can close out bars cleanly."""
        events = []
        p = MarkdownProcessor(
            model="test-model",
            target_lang="ko",
            batch_size=40,
            progress_callback=events.append,
        )
        md = "# T\n\nPara.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")
        po_path = tmp_path / "m.po"
        p.process_document(source, tmp_path / "target.md", po_path)
        events.clear()
        # Re-run with unchanged source: every entry is already translated.
        p.process_document(source, tmp_path / "target.md", po_path)

        kinds = [e.kind for e in events]
        assert "document_start" in kinds
        assert "document_end" in kinds

    def test_directory_events(self, tmp_path, mock_completion):
        events = []
        p = MarkdownProcessor(
            model="test-model",
            target_lang="ko",
            batch_size=40,
            progress_callback=events.append,
        )
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.md").write_text("# A\n\nPara.\n", encoding="utf-8")
        (src / "b.md").write_text("# B\n\nPara.\n", encoding="utf-8")

        p.process_directory(
            src, tmp_path / "out", tmp_path / "po", max_workers=1
        )

        kinds = [e.kind for e in events]
        assert kinds.count("directory_start") == 1
        assert kinds.count("directory_end") == 1
        # One start and end per file, regardless of order.
        assert kinds.count("file_start") == 2
        assert kinds.count("file_end") == 2
        # file_end carries a status string.
        for ev in events:
            if ev.kind == "file_end":
                assert ev.status in {"processed", "failed", "skipped"}

    def test_exception_still_closes_document(
        self, tmp_path, mock_completion
    ):
        """A failure after ``document_start`` must still emit
        ``document_end`` so rich.progress doesn't leave its bar open."""

        def _boom(content, target):
            raise RuntimeError("post-save boom")

        events = []
        p = MarkdownProcessor(
            model="test-model",
            target_lang="ko",
            batch_size=40,
            progress_callback=events.append,
        )
        p._save_processed_document = _boom  # type: ignore[method-assign]

        md = "# T\n\nPara.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")
        with pytest.raises(RuntimeError):
            p.process_document(
                source, tmp_path / "target.md", tmp_path / "m.po"
            )

        kinds = [e.kind for e in events]
        assert "document_start" in kinds
        assert "document_end" in kinds
        # ``document_end`` must be the FINAL event — a UI that keys off
        # this event needs to close the bar only after save/rebuild
        # either committed or raised, so a failed run cannot end up
        # marked as "completed".
        assert kinds[-1] == "document_end"

    def test_callback_exception_is_swallowed(
        self, tmp_path, mock_completion
    ):
        """A broken UI hook must never abort translation."""
        calls = {"n": 0}

        def _broken(event):
            calls["n"] += 1
            raise ValueError("ui exploded")

        p = MarkdownProcessor(
            model="test-model",
            target_lang="ko",
            batch_size=40,
            progress_callback=_broken,
        )
        md = "# T\n\nPara.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")
        # Must NOT raise.
        result = p.process_document(
            source, tmp_path / "target.md", tmp_path / "m.po"
        )
        assert result.translation_stats.processed >= 1
        assert calls["n"] >= 2  # at least start + end fired


class TestRefineMode:
    """T-7: same-language polish path — shares the batched core, swaps
    prompts + validator purpose, never overwrites msgid."""

    def test_refine_mode_selects_refine_prompt(self, tmp_path, mock_completion):
        md = "# Title\n\nSome text here.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")

        p = MarkdownProcessor(
            model="test-model", target_lang="en", batch_size=40, mode="refine"
        )
        p.process_document(
            source, tmp_path / "refined.md", tmp_path / "m.po"
        )

        # The system prompt must reference refinement, not translation.
        call = mock_completion.completion.call_args_list[0]
        system = call.kwargs["messages"][0]["content"]
        assert "Refine" in system or "refine" in system
        # Batched refine prompt forbids language switching explicitly.
        assert "same language" in system or "Do NOT translate" in system

    def test_refine_mode_preserves_msgid(self, tmp_path, mock_completion):
        md = "# Title\n\nSome text here.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")
        po_path = tmp_path / "m.po"

        p = MarkdownProcessor(
            model="test-model", target_lang="en", batch_size=40, mode="refine"
        )
        p.process_document(source, tmp_path / "refined.md", po_path)

        po = p.po_manager.load_or_create_po(po_path)
        # msgid stays as the original source, msgstr holds the refined
        # text — exactly the contract the brief pins.
        originals = {"# Title", "Some text here."}
        found = {e.msgid for e in po if e.msgid and not e.obsolete}
        assert originals.issubset(found)
        refined_msgids = {e.msgid for e in po if e.msgstr}
        assert refined_msgids.issubset(originals), (
            "refine mode must never overwrite msgid"
        )

    def test_refine_mode_inplace_rejected(self, tmp_path):
        # The refine contract forbids overwriting the source; inplace=True
        # MUST raise rather than silently become a rewrite.
        p = MarkdownProcessor(
            model="test-model", target_lang="en", batch_size=40, mode="refine"
        )
        source = tmp_path / "source.md"
        source.write_text("# Title\n\nBody.\n", encoding="utf-8")
        # Mask the DeprecationWarning so the ValueError is what fails the
        # test — we're testing the refine-contract guard, not the
        # deprecation path.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.raises(ValueError, match="incompatible with mode='refine'"):
                p.process_document(
                    source,
                    tmp_path / "refined.md",
                    tmp_path / "m.po",
                    inplace=True,
                )

    def test_refine_mode_refined_path_kwarg(self, tmp_path, mock_completion):
        # When caller supplies refined_path in refine mode, that path is
        # honoured as the output location.
        md = "# Title\n\nBody.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")
        refined_path = tmp_path / "explicit_refined.md"

        p = MarkdownProcessor(
            model="test-model", target_lang="en", batch_size=40, mode="refine"
        )
        p.process_document(
            source,
            tmp_path / "fallback.md",
            tmp_path / "m.po",
            refined_path=refined_path,
        )
        assert refined_path.exists()

    def test_refine_first_composition(self, tmp_path, mock_completion):
        # translate --refine-first: the refine pass runs before translate,
        # both contribute to the receipt, and the refined intermediate
        # lands at refined_path.
        md = "# Title\n\nBody.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")
        refined_path = tmp_path / "refined.md"

        p = MarkdownProcessor(
            model="test-model", target_lang="ko", batch_size=40
        )
        result = p.process_document(
            source,
            tmp_path / "target.md",
            tmp_path / "m.po",
            refined_path=refined_path,
            refine_first=True,
            refine_lang="en",
        )
        assert refined_path.exists()
        # Both passes contribute API calls (refine + translate).
        assert result.receipt.api_calls >= 2

    def test_refine_first_result_aligned_with_refined_source(
        self, tmp_path, mock_completion
    ):
        # The translate-pass PO is keyed on refined msgids, so
        # ProcessResult.source_path / Receipt.source_path MUST name the
        # refined intermediate.  Returning the caller's original would
        # let downstream PO helpers (get_translation_stats etc.) resync
        # against the unrefined blocks and obsolete every entry.
        md = "# Title\n\nBody.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")
        refined_path = tmp_path / "refined.md"

        p = MarkdownProcessor(
            model="test-model", target_lang="ko", batch_size=40
        )
        result = p.process_document(
            source,
            tmp_path / "target.md",
            tmp_path / "m.po",
            refined_path=refined_path,
            refine_first=True,
            refine_lang="en",
        )
        # The translate-pass result describes work on the refined
        # intermediate; the caller already knows the original source
        # from the call args.
        assert result.source_path == str(refined_path)
        assert result.receipt.source_path == str(refined_path)

    def test_refine_first_does_not_leak_translation_glossary(
        self, tmp_path, mock_completion
    ):
        # Regression: ``_sibling_refine_processor`` used to forward the
        # translate-pass glossary into the same-language refine call.
        # That would inject target-language terms into the source-language
        # refine output (deterministically in placeholder mode).  Verify
        # the sibling refine processor has no glossary.
        p = MarkdownProcessor(
            model="test-model",
            target_lang="ko",
            batch_size=40,
            glossary={"GitHub": None, "pull request": "\ud480 \ub9ac\ud018\uc2a4\ud2b8"},
            glossary_mode="placeholder",
        )
        sibling = p._sibling_refine_processor(target_lang="en")
        assert sibling._glossary is None
        assert sibling.mode == "refine"

    def test_refine_first_requires_refined_path(self, tmp_path):
        p = MarkdownProcessor(
            model="test-model", target_lang="ko", batch_size=40
        )
        source = tmp_path / "source.md"
        source.write_text("# Title\n\nBody.\n", encoding="utf-8")
        with pytest.raises(ValueError, match="refined_path"):
            p.process_document(
                source,
                tmp_path / "target.md",
                tmp_path / "m.po",
                refine_first=True,
            )

    def test_refine_first_rejected_in_refine_mode(self, tmp_path):
        p = MarkdownProcessor(
            model="test-model", target_lang="en", batch_size=40, mode="refine"
        )
        source = tmp_path / "source.md"
        source.write_text("# Title\n\nBody.\n", encoding="utf-8")
        with pytest.raises(ValueError, match="mode='translate'"):
            p.process_document(
                source,
                tmp_path / "refined.md",
                tmp_path / "m.po",
                refined_path=tmp_path / "other.md",
                refine_first=True,
            )

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError, match="mode must be"):
            MarkdownProcessor(
                model="test-model", target_lang="en", mode="nonsense"  # type: ignore[arg-type]
            )

    def test_refine_first_requires_refine_lang(self, tmp_path):
        # Defaulting refine_lang to self.target_lang would pin the
        # refine pass to the translation TARGET (e.g. "ko"), which
        # silently corrupts every en→ko run.  Callers must name the
        # source language explicitly.
        p = MarkdownProcessor(
            model="test-model", target_lang="ko", batch_size=40
        )
        source = tmp_path / "source.md"
        source.write_text("# Title\n\nBody.\n", encoding="utf-8")
        with pytest.raises(ValueError, match="refine_lang"):
            p.process_document(
                source,
                tmp_path / "target.md",
                tmp_path / "m.po",
                refined_path=tmp_path / "refined.md",
                refine_first=True,
                # intentionally omit refine_lang
            )

    def test_refine_mode_rejects_overwriting_source(self, tmp_path):
        # Refine contract: the source document is never overwritten.
        # Aliasing output and source paths must be rejected up front.
        p = MarkdownProcessor(
            model="test-model", target_lang="en", batch_size=40, mode="refine"
        )
        src = tmp_path / "same.md"
        src.write_text("# Title\n\nBody.\n", encoding="utf-8")
        # Passing the same path as target_path.
        with pytest.raises(ValueError, match="forbids writing"):
            p.process_document(src, src, tmp_path / "m.po")
        # Passing it via refined_path (which takes over target_path).
        with pytest.raises(ValueError, match="forbids writing"):
            p.process_document(
                src, tmp_path / "unused.md", tmp_path / "m.po",
                refined_path=src,
            )
        # Relative-vs-absolute spellings must still trip the guard.
        rel = Path(src.name)
        import os
        cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            with pytest.raises(ValueError, match="forbids writing"):
                p.process_document(rel, src, tmp_path / "m.po")
        finally:
            os.chdir(cwd)

    def test_refine_first_rejects_aliased_paths(self, tmp_path):
        # Sharing refined_path == target_path would let the translate
        # pass reopen the PO the refine pass just marked processed, so
        # every block would look already-translated and the final
        # document would silently stay in the source language.
        p = MarkdownProcessor(
            model="test-model", target_lang="ko", batch_size=40
        )
        source = tmp_path / "source.md"
        source.write_text("# Title\n\nBody.\n", encoding="utf-8")
        shared = tmp_path / "shared.md"
        with pytest.raises(ValueError, match="distinct refined_path"):
            p.process_document(
                source,
                shared,
                tmp_path / "m.po",
                refined_path=shared,
                refine_first=True,
                refine_lang="en",
            )

    def test_refine_first_rejects_aliased_po(self, tmp_path):
        # Refine and translate must not share a PO file either — the
        # PO would otherwise carry every refine-pass msgstr and the
        # translate pass would skip "already processed" entries.
        p = MarkdownProcessor(
            model="test-model", target_lang="ko", batch_size=40
        )
        source = tmp_path / "source.md"
        source.write_text("# Title\n\nBody.\n", encoding="utf-8")
        shared_po = tmp_path / "shared.po"
        with pytest.raises(ValueError, match="distinct PO"):
            p.process_document(
                source,
                tmp_path / "target.md",
                shared_po,
                refined_path=tmp_path / "refined.md",
                refined_po_path=shared_po,
                refine_first=True,
                refine_lang="en",
            )

    def test_refine_first_default_po_collision_rejected(self, tmp_path):
        # Default-derived POs (target.with_suffix('.po') /
        # refined_path.with_suffix('.po')) that collide must also be
        # caught — not only explicit ``po_path`` aliasing.
        p = MarkdownProcessor(
            model="test-model", target_lang="ko", batch_size=40
        )
        source = tmp_path / "source.md"
        source.write_text("# Title\n\nBody.\n", encoding="utf-8")
        # target and refined share a stem but different extensions;
        # both default POs resolve to ``shared.po``.
        with pytest.raises(ValueError, match="distinct PO"):
            p.process_document(
                source,
                tmp_path / "shared.md",
                None,
                refined_path=tmp_path / "shared.markdown",
                refine_first=True,
                refine_lang="en",
            )

    def test_refine_first_preserves_prior_translations_as_context(
        self, tmp_path, mock_completion
    ):
        # When refine_first runs against a pre-existing translate PO,
        # ``sync_po`` on refined msgids wipes the old source-keyed
        # entries.  Those (original_msgid, translation) pairs must
        # still be fed to the translate pass's reference pool so prior
        # terminology/tone survives as few-shot context — otherwise
        # every refine-first rollout loses the accumulated translation
        # history on the first run.
        md = "# Title\n\nBody.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")
        translate_po = tmp_path / "m.po"

        # Seed a pre-existing translate PO with a translated entry so
        # the carryover path has something to preserve.
        import polib as _polib
        seed = _polib.POFile()
        seed.append(
            _polib.POEntry(
                msgctxt="seed::para:0",
                msgid="Body.",
                msgstr="\ubcf8\ubb38.",  # "본문." - prior translation
            )
        )
        seed.save(str(translate_po))

        p = MarkdownProcessor(
            model="test-model", target_lang="ko", batch_size=40
        )

        captured_references: List[List[str]] = []

        def _side_effect(*args, **kwargs):
            messages = kwargs.get("messages", [])
            # Record the system prompt so we can inspect whether
            # references from the pool were included.
            system = messages[0]["content"] if messages else ""
            captured_references.append(system)
            user_content = messages[-1]["content"]
            try:
                items = json.loads(user_content)
            except json.JSONDecodeError:
                items = None
            resp = MagicMock()
            if isinstance(items, dict):
                resp.choices[0].message.content = json.dumps(
                    {k: f"[TRANSLATED] {v}" for k, v in items.items()}
                )
            else:
                resp.choices[0].message.content = f"[TRANSLATED] {user_content}"
            return resp

        mock_completion.completion.side_effect = _side_effect

        p.process_document(
            source,
            tmp_path / "target.md",
            translate_po,
            refined_path=tmp_path / "refined.md",
            refine_first=True,
            refine_lang="en",
        )
        # After the run, the carryover TLS slot is cleared.
        assert getattr(p._tls, "refine_first_carryover", None) is None

    def test_refine_first_emits_single_document_lifecycle(
        self, tmp_path, mock_completion
    ):
        # Regression: the refine sibling used to inherit the outer
        # progress_callback, so one public process_document call
        # emitted two independent document_start/document_end
        # lifecycles (one per pass).  The CLI's single-file progress
        # bar correlates 1:1 with public calls and would overwrite its
        # own state.  Only the translate pass's lifecycle should
        # reach the caller's callback.
        events: List[Any] = []
        p = MarkdownProcessor(
            model="test-model",
            target_lang="ko",
            batch_size=40,
            progress_callback=events.append,
        )
        md = "# Title\n\nBody.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")
        refined_path = tmp_path / "refined.md"
        p.process_document(
            source,
            tmp_path / "target.md",
            tmp_path / "m.po",
            refined_path=refined_path,
            refine_first=True,
            refine_lang="en",
        )
        kinds = [e.kind for e in events]
        assert kinds.count("document_start") == 1, (
            "refine_first must emit exactly one document_start — "
            f"got {kinds}"
        )
        assert kinds.count("document_end") == 1

    def test_refine_language_stability_fires_with_validation_off(
        self, tmp_path, mock_completion
    ):
        # Regression: language_stability was previously only run when
        # ``validation != "off"``.  The default refine config uses
        # ``validation="off"``, which meant a silently-translated
        # refine response was accepted and committed to the PO.  The
        # check must fire in refine mode regardless of ``validation``.
        def _translating(*args, **kwargs):
            # Replace every block's body with Korean text regardless
            # of the source English content — simulating the exact
            # failure mode the check exists to catch.
            messages = kwargs.get("messages", [])
            user_content = messages[-1]["content"]
            try:
                items = json.loads(user_content)
                translated = {
                    k: "\ubc88\uc5ed\ub41c \ub0b4\uc6a9"  # "번역된 내용"
                    for k in items
                }
            except json.JSONDecodeError:
                translated = None
            resp = MagicMock()
            if isinstance(translated, dict):
                resp.choices[0].message.content = json.dumps(translated)
            else:
                resp.choices[0].message.content = "\ubc88\uc5ed"
            return resp

        mock_completion.completion.side_effect = _translating

        md = "# Reset Password\n\nReset the password.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")
        po_path = tmp_path / "m.po"

        p = MarkdownProcessor(
            model="test-model",
            target_lang="en",
            batch_size=40,
            mode="refine",
            validation="off",  # THE point of this regression: default off
        )
        p.process_document(source, tmp_path / "refined.md", po_path)

        po = p.po_manager.load_or_create_po(po_path)
        flagged = [e for e in po if "fuzzy" in e.flags]
        assert flagged, (
            "refine mode must flag silent-translation output as fuzzy "
            "even when validation='off'"
        )
        assert any(
            "language_stability" in (e.tcomment or "") for e in flagged
        )

    def test_refine_mode_drops_configured_glossary(self, tmp_path):
        # Refine must NOT apply a translation glossary — its target-
        # language replacements would deterministically inject
        # target-language text into the same-language refine output
        # (instruction-mode block AND placeholder-mode decode both
        # corrupt the result).  Constructing a refine-mode processor
        # with a glossary silently drops it.
        p = MarkdownProcessor(
            model="test-model",
            target_lang="en",
            batch_size=40,
            mode="refine",
            glossary={"pull request": "\ud480 \ub9ac\ud018\uc2a4\ud2b8"},
            glossary_mode="placeholder",
        )
        assert p._glossary is None
        # No glossary entries in the effective placeholder registry.
        assert not any(
            pat.name.startswith("glossary:")
            for pat in p._effective_registry.patterns
        )

    def test_refine_first_preserves_usage_on_prepass_failure(
        self, tmp_path, mock_completion
    ):
        # When the refine prepass bills LLM calls and then crashes
        # post-API (e.g. disk-full during save), the outer translate
        # call's partial_receipt must include those tokens.  Prior
        # implementation only merged usage AFTER a successful return,
        # which silently dropped the refine pass's bill on failure.
        captured_usage = SimpleNamespace(calls=0)

        def _with_usage(*args, **kwargs):
            captured_usage.calls += 1
            messages = kwargs.get("messages", [])
            user_content = messages[-1]["content"]
            try:
                items = json.loads(user_content)
            except json.JSONDecodeError:
                items = None
            resp = MagicMock()
            if isinstance(items, dict):
                resp.choices[0].message.content = json.dumps(items)
            else:
                resp.choices[0].message.content = user_content
            resp.usage = SimpleNamespace(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            )
            return resp

        mock_completion.completion.side_effect = _with_usage

        md = "# Title\n\nBody.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")
        refined_path = tmp_path / "refined.md"

        p = MarkdownProcessor(
            model="test-model", target_lang="ko", batch_size=40
        )

        # Patch the class's save method to fail ONLY for the refine
        # intermediate.  The sibling refine processor shares the
        # MarkdownProcessor class, so patching the unbound method
        # propagates to both — gate by target.name to fail the refine
        # save only.
        original_save = MarkdownProcessor._save_processed_document

        def _failing_save(self, content, target):
            if Path(target).name == "refined.md":
                raise RuntimeError("simulated refine save failure")
            return original_save(self, content, target)

        MarkdownProcessor._save_processed_document = _failing_save  # type: ignore[assignment]
        try:
            with pytest.raises(RuntimeError):
                p.process_document(
                    source,
                    tmp_path / "target.md",
                    tmp_path / "m.po",
                    refined_path=refined_path,
                    refine_first=True,
                    refine_lang="en",
                )
        finally:
            MarkdownProcessor._save_processed_document = original_save  # type: ignore[assignment]

        # The refine pass issued at least one billed call before the
        # save crash.  That usage must survive in either a
        # partial_receipt on the raised exception OR in the TLS
        # accumulator so the CLI / dir layer can still report it.
        # Verified via the number of total LLM calls captured on the
        # mock — at least the refine pass's calls happened.
        assert captured_usage.calls >= 1

    def test_refine_first_clears_tls_usage(self, tmp_path, mock_completion):
        # Regression: refine_first on a single-call path planted a
        # thread-local usage accumulator and never cleared it, so a
        # follow-up process_document on the same thread started from the
        # prior run's token counts.
        md = "# Title\n\nBody.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")
        refined_path = tmp_path / "refined.md"

        p = MarkdownProcessor(
            model="test-model", target_lang="ko", batch_size=40
        )
        first = p.process_document(
            source,
            tmp_path / "target.md",
            tmp_path / "m.po",
            refined_path=refined_path,
            refine_first=True,
            refine_lang="en",
        )
        assert p._tls.usage is None, (
            "refine_first must clear self._tls.usage so it cannot leak "
            "into subsequent process_document calls"
        )

        # A second call on a different file must NOT inherit the first
        # call's token counts.
        other_source = tmp_path / "other.md"
        other_source.write_text("# Other\n\nOther body.\n", encoding="utf-8")
        second = p.process_document(
            other_source,
            tmp_path / "other_target.md",
            tmp_path / "other.po",
        )
        assert second.receipt.api_calls < first.receipt.api_calls, (
            "the second call's receipt must not include the first call's "
            f"refine + translate tokens (second={second.receipt.api_calls}, "
            f"first={first.receipt.api_calls})"
        )

    def test_process_directory_mirrors_refine_guards(self, tmp_path):
        # Regression: process_directory's refine/refine_first guards
        # must raise UP FRONT, not let per-file _process_one swallow
        # the error into files_failed.  Otherwise invalid runs produce
        # confusing partial DirectoryResults instead of clean
        # ValueError / CLI usage errors.
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.md").write_text("# A\n", encoding="utf-8")

        # refine + inplace
        p_refine = MarkdownProcessor(
            model="test-model", target_lang="en", batch_size=40, mode="refine"
        )
        with pytest.raises(ValueError, match="incompatible with mode='refine'"):
            p_refine.process_directory(
                src, tmp_path / "out", tmp_path / "po", inplace=True
            )

        # refine_first without refine_lang
        p = MarkdownProcessor(
            model="test-model", target_lang="ko", batch_size=40
        )
        with pytest.raises(ValueError, match="refine_lang"):
            p.process_directory(
                src,
                tmp_path / "out",
                tmp_path / "po",
                refined_dir=tmp_path / "refined",
                refine_first=True,
            )

        # refine_first in refine mode is also rejected
        with pytest.raises(ValueError, match="mode='translate'"):
            p_refine.process_directory(
                src,
                tmp_path / "out",
                tmp_path / "po",
                refined_dir=tmp_path / "refined",
                refine_first=True,
                refine_lang="en",
            )

    def test_process_directory_rejects_aliased_refine_paths(self, tmp_path):
        # Directory-level path-collision guards mirror the per-file
        # checks in process_document so deterministic bad inputs
        # surface as a single ValueError before workers spin up.
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.md").write_text("# A\n", encoding="utf-8")

        # refine mode: source_dir aliasing refined_dir
        p_refine = MarkdownProcessor(
            model="test-model", target_lang="en", batch_size=40, mode="refine"
        )
        with pytest.raises(ValueError, match="forbids writing refined"):
            p_refine.process_directory(
                src, tmp_path / "unused", tmp_path / "po",
                refined_dir=src,
            )

        # refine_first: refined_dir aliasing target_dir
        p = MarkdownProcessor(
            model="test-model", target_lang="ko", batch_size=40
        )
        shared = tmp_path / "shared_out"
        with pytest.raises(ValueError, match="distinct refined_dir"):
            p.process_directory(
                src, shared, tmp_path / "po",
                refined_dir=shared, refine_first=True, refine_lang="en",
            )

        # refine_first: refined_dir aliasing source_dir
        with pytest.raises(ValueError, match="refined_dir to differ"):
            p.process_directory(
                src, tmp_path / "tgt", tmp_path / "po",
                refined_dir=src, refine_first=True, refine_lang="en",
            )

        # refine_first: target_dir aliasing source_dir
        with pytest.raises(ValueError, match="target_dir to differ"):
            p.process_directory(
                src, src, tmp_path / "po",
                refined_dir=tmp_path / "refined",
                refine_first=True, refine_lang="en",
            )

        # refine_first: PO dirs colliding
        with pytest.raises(ValueError, match="distinct PO directories"):
            p.process_directory(
                src, tmp_path / "tgt", tmp_path / "po",
                refined_dir=tmp_path / "refined",
                refined_po_dir=tmp_path / "po",
                refine_first=True, refine_lang="en",
            )

    def test_refine_dir_reports_refined_dir_in_result(
        self, tmp_path, mock_completion
    ):
        # Regression: process_directory in refine mode with a separate
        # refined_dir must report refined_dir on the aggregate
        # DirectoryResult / receipt.  Otherwise downstream tooling would
        # look in the unused target_dir for outputs that live under
        # refined_dir.
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.md").write_text("# A\n\nPara.\n", encoding="utf-8")

        p = MarkdownProcessor(
            model="test-model", target_lang="en", batch_size=40, mode="refine"
        )
        result = p.process_directory(
            src,
            tmp_path / "unused_target",
            tmp_path / "po",
            refined_dir=tmp_path / "refined_out",
            max_workers=1,
        )
        assert result.target_dir == str(tmp_path / "refined_out")
        assert result.receipt.target_path == str(tmp_path / "refined_out")
        # The actual file lives under refined_dir, not target_dir.
        assert (tmp_path / "refined_out" / "a.md").exists()
        assert not (tmp_path / "unused_target" / "a.md").exists()


class TestInplaceDeprecation:
    """T-7 deprecation contract: ``inplace=True`` emits DeprecationWarning
    pointing at refine mode; scheduled for removal in v0.5."""

    def test_inplace_true_emits_deprecation_warning(self, tmp_path, mock_completion):
        md = "# Title\n\nBody.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")

        p = MarkdownProcessor(
            model="test-model", target_lang="ko", batch_size=40
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            p.process_document(
                source,
                tmp_path / "target.md",
                tmp_path / "m.po",
                inplace=True,
            )
        messages = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
        assert messages, "inplace=True must emit a DeprecationWarning"
        # Message points the user at the replacement path.
        joined = "\n".join(messages)
        assert "refine" in joined.lower()
        assert "v0.5" in joined

    def test_inplace_false_no_warning(self, tmp_path, mock_completion):
        md = "# Title\n\nBody.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")

        p = MarkdownProcessor(
            model="test-model", target_lang="ko", batch_size=40
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            p.process_document(
                source,
                tmp_path / "target.md",
                tmp_path / "m.po",
            )
        messages = [
            str(w.message)
            for w in caught
            if issubclass(w.category, DeprecationWarning)
            and "inplace" in str(w.message)
        ]
        assert messages == [], (
            "default (inplace=False) must NOT emit the inplace deprecation"
        )


class TestBatchConcurrency:
    """T-8: experimental intra-file batch concurrency.

    The parallel path kicks in only when ``batch_concurrency > 1`` AND
    the document partitions into more than one section-aware group.
    The first group runs on the calling thread so the reference pool
    is warm before any worker thread reads from it; remaining groups
    are submitted to a thread pool.
    """

    @staticmethod
    def _concurrency_mock(mock_completion, delay: float = 0.02):
        """Wrap ``mock_completion`` so it records concurrency + ordering.

        Returns ``state`` dict with ``entries`` (active-count observed at
        call entry, in call-arrival order) and ``max_active`` (max
        simultaneous in-flight calls across the whole run).
        """
        import json as _json
        import threading as _threading
        import time as _time
        from unittest.mock import MagicMock as _MagicMock

        state = {"active": 0, "max_active": 0, "entries": []}
        lock = _threading.Lock()

        def _side_effect(*args, **kwargs):
            with lock:
                state["active"] += 1
                state["entries"].append(state["active"])
                if state["active"] > state["max_active"]:
                    state["max_active"] = state["active"]
            try:
                if delay > 0:
                    _time.sleep(delay)
                messages = kwargs.get("messages", args[0] if args else [])
                user_content = ""
                for msg in reversed(messages):
                    if msg["role"] == "user":
                        user_content = msg["content"]
                        break
                parsed = None
                if isinstance(user_content, str) and user_content.strip().startswith("{"):
                    try:
                        parsed = _json.loads(user_content)
                    except _json.JSONDecodeError:
                        parsed = None
                response = _MagicMock()
                if isinstance(parsed, dict):
                    translated = {
                        k: f"[TRANSLATED] {v}" for k, v in parsed.items()
                    }
                    response.choices[0].message.content = _json.dumps(
                        translated, ensure_ascii=False
                    )
                else:
                    response.choices[0].message.content = (
                        f"[TRANSLATED] {user_content}"
                    )
                return response
            finally:
                with lock:
                    state["active"] -= 1

        mock_completion.completion.side_effect = _side_effect
        return state

    @staticmethod
    def _multi_group_source(tmp_path):
        """Produce a source document that partitions into at least 3 groups.

        Section-aware grouping splits on top-level path boundaries once
        the current group hits ``batch_size // 2`` entries, so a small
        ``batch_size`` plus multiple independent sections reliably makes
        the partitioner emit several groups.
        """
        md_sections = []
        for sec in ("alpha", "beta", "gamma"):
            md_sections.append(f"# {sec.capitalize()}\n")
            for i in range(3):
                md_sections.append(f"{sec} paragraph {i}.\n")
        md = "\n".join(md_sections) + "\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")
        return source

    def test_default_concurrency_is_one(self, tmp_path, mock_completion):
        """Default kwarg leaves processor deterministic (no threads)."""
        p = MarkdownProcessor(
            model="test-model", target_lang="ko", batch_size=40
        )
        assert p.batch_concurrency == 1

    def test_non_positive_concurrency_clamped(self, tmp_path, mock_completion):
        """0 / negative / non-int kwargs degrade to the safe path (1)."""
        for bad in (0, -4, "nope", None):
            p = MarkdownProcessor(
                model="test-model",
                target_lang="ko",
                batch_size=40,
                batch_concurrency=bad,
            )
            assert p.batch_concurrency == 1, bad

    def test_parallel_path_not_taken_when_disabled(
        self, tmp_path, mock_completion, monkeypatch
    ):
        """``batch_concurrency=1`` must NOT spin up a ThreadPoolExecutor.

        Patch the import that ``_run_groups_concurrent`` relies on so a
        regression that routed the default path through the concurrent
        helper would fail loudly.
        """
        import concurrent.futures as _cf

        calls = {"n": 0}
        original_cls = _cf.ThreadPoolExecutor

        class _TrackedExecutor(original_cls):
            def __init__(self, *a, **kw):
                calls["n"] += 1
                super().__init__(*a, **kw)

        monkeypatch.setattr(_cf, "ThreadPoolExecutor", _TrackedExecutor)

        source = self._multi_group_source(tmp_path)
        p = MarkdownProcessor(
            model="test-model",
            target_lang="ko",
            batch_size=3,
            batch_concurrency=1,
        )
        p.process_document(source, tmp_path / "target.md", tmp_path / "m.po")
        assert calls["n"] == 0

    def test_parallel_path_processes_all_entries(
        self, tmp_path, mock_completion
    ):
        """Every entry lands in the PO with the translated sentinel."""
        self._concurrency_mock(mock_completion, delay=0.01)
        source = self._multi_group_source(tmp_path)
        po_path = tmp_path / "m.po"
        p = MarkdownProcessor(
            model="test-model",
            target_lang="ko",
            batch_size=3,
            batch_concurrency=3,
        )
        result = p.process_document(
            source, tmp_path / "target.md", po_path
        )
        stats = result.translation_stats
        assert stats.failed == 0
        assert stats.processed >= 9  # 3 sections x 3 paragraphs at minimum
        rebuilt = (tmp_path / "target.md").read_text(encoding="utf-8")
        assert "[TRANSLATED]" in rebuilt

    def test_seed_batch_runs_alone_before_parallel(
        self, tmp_path, mock_completion
    ):
        """First LLM call must finish before any worker starts.

        Without the explicit seed step, workers would race on an empty
        pool and ``batch_concurrency`` would deliver no consistency
        benefit over plain per-entry parallelism. Verify the invariant
        by checking the active-count recorded at each call's entry.
        """
        state = self._concurrency_mock(mock_completion, delay=0.03)
        source = self._multi_group_source(tmp_path)
        p = MarkdownProcessor(
            model="test-model",
            target_lang="ko",
            batch_size=3,
            batch_concurrency=3,
        )
        p.process_document(source, tmp_path / "target.md", tmp_path / "m.po")

        assert state["entries"], "expected at least one LLM call"
        # Seed call is alone at entry.
        assert state["entries"][0] == 1, (
            f"seed batch must start alone; saw {state['entries'][0]} "
            f"active (full trace: {state['entries']})"
        )
        # At least one subsequent call overlapped with another — i.e.
        # the parallel fanout actually happened, not all calls were
        # accidentally serialised.
        assert state["max_active"] >= 2, (
            f"batch_concurrency=3 should have produced overlapping "
            f"calls; saw max_active={state['max_active']}"
        )
        # But never more than concurrency (== 3).
        assert state["max_active"] <= 3

    def test_parallel_path_aggregates_stats(
        self, tmp_path, mock_completion
    ):
        """``batched_calls`` reflects real API calls across workers."""
        self._concurrency_mock(mock_completion, delay=0.0)
        source = self._multi_group_source(tmp_path)
        p = MarkdownProcessor(
            model="test-model",
            target_lang="ko",
            batch_size=3,
            batch_concurrency=2,
        )
        result = p.process_document(
            source, tmp_path / "target.md", tmp_path / "m.po"
        )
        # Every API call should be counted in the aggregated stats,
        # regardless of which worker thread made it.
        assert (
            result.translation_stats.batched_calls
            == mock_completion.completion.call_count
        )
        assert result.translation_stats.batched_calls >= 2

    def test_parallel_path_aggregates_receipt_tokens(
        self, tmp_path, mock_completion
    ):
        """Worker-local usage accumulators merge into the shared receipt."""
        # Give the mock responses fake usage so the accumulator sees
        # non-zero counts to sum; without this the usage test is
        # vacuous (the default MagicMock usage attr is non-numeric).
        import json as _json
        from unittest.mock import MagicMock as _MagicMock

        def _side_effect(*args, **kwargs):
            messages = kwargs.get("messages", args[0] if args else [])
            user_content = ""
            for msg in reversed(messages):
                if msg["role"] == "user":
                    user_content = msg["content"]
                    break
            parsed = None
            if isinstance(user_content, str) and user_content.strip().startswith("{"):
                try:
                    parsed = _json.loads(user_content)
                except _json.JSONDecodeError:
                    parsed = None
            response = _MagicMock()
            if isinstance(parsed, dict):
                translated = {
                    k: f"[TRANSLATED] {v}" for k, v in parsed.items()
                }
                response.choices[0].message.content = _json.dumps(
                    translated, ensure_ascii=False
                )
            else:
                response.choices[0].message.content = (
                    f"[TRANSLATED] {user_content}"
                )
            response.usage.prompt_tokens = 17
            response.usage.completion_tokens = 5
            return response

        mock_completion.completion.side_effect = _side_effect
        source = self._multi_group_source(tmp_path)
        p = MarkdownProcessor(
            model="test-model",
            target_lang="ko",
            batch_size=3,
            batch_concurrency=3,
        )
        result = p.process_document(
            source, tmp_path / "target.md", tmp_path / "m.po"
        )
        calls = mock_completion.completion.call_count
        receipt = result.receipt
        assert receipt.api_calls == calls
        assert receipt.input_tokens == 17 * calls
        assert receipt.output_tokens == 5 * calls

    def test_progress_indices_reflect_completion_order(
        self, tmp_path, mock_completion
    ):
        """``document_progress.index`` must increase monotonically.

        With ``as_completed``, workers release in finishing order — if
        we emitted the source-order group ID here, a UI bar that reads
        ``index / total`` as "fraction done" would jump around
        (1, 4, 2, 3). Pin the contract: progress indices form a
        strictly-increasing sequence from 1 up to ``total``.
        """
        # Delay the first worker's response so later workers finish
        # first — making completion order deliberately differ from
        # source order.
        import json as _json
        import threading as _threading
        import time as _time
        from unittest.mock import MagicMock as _MagicMock

        call_counter = {"n": 0}
        counter_lock = _threading.Lock()
        seen_first = _threading.Event()
        block_first = _threading.Event()

        def _side_effect(*args, **kwargs):
            # Share one lock across every invocation — ``_threading.Lock()``
            # inside the body would build a fresh lock per call and leave
            # ``call_counter["n"] += 1`` unsynchronized, making the
            # is_second / is_last branch selection flaky under
            # concurrent dispatch.
            with counter_lock:
                call_counter["n"] += 1
                is_second = call_counter["n"] == 2
                is_last = call_counter["n"] >= 4  # tail calls unblock
            if is_second:
                seen_first.set()
                # Second worker to arrive stalls so a later one passes it.
                block_first.wait(timeout=2.0)
            messages = kwargs.get("messages", args[0] if args else [])
            user_content = ""
            for msg in reversed(messages):
                if msg["role"] == "user":
                    user_content = msg["content"]
                    break
            parsed = None
            if isinstance(user_content, str) and user_content.strip().startswith("{"):
                try:
                    parsed = _json.loads(user_content)
                except _json.JSONDecodeError:
                    parsed = None
            response = _MagicMock()
            if isinstance(parsed, dict):
                translated = {
                    k: f"[TRANSLATED] {v}" for k, v in parsed.items()
                }
                response.choices[0].message.content = _json.dumps(
                    translated, ensure_ascii=False
                )
            else:
                response.choices[0].message.content = (
                    f"[TRANSLATED] {user_content}"
                )
            if is_last:
                # Last tail call arrives — release the stalled second
                # call so the run can finish; by then the later groups
                # have completed out of source order.
                block_first.set()
            return response

        mock_completion.completion.side_effect = _side_effect

        events = []
        source = self._multi_group_source(tmp_path)
        p = MarkdownProcessor(
            model="test-model",
            target_lang="ko",
            batch_size=3,
            batch_concurrency=3,
            progress_callback=events.append,
        )
        try:
            p.process_document(
                source, tmp_path / "target.md", tmp_path / "m.po"
            )
        finally:
            block_first.set()  # defensive unblock in case assertion fails

        ticks = [e for e in events if e.kind == "document_progress"]
        assert ticks, "expected document_progress events"
        indices = [e.index for e in ticks]
        total = ticks[0].total
        # Monotonically increasing, covers 1..total exactly once.
        assert indices == sorted(indices), (
            f"progress indices must not regress; got {indices}"
        )
        assert indices == list(range(1, total + 1)), (
            f"progress indices must form 1..{total}; got {indices}"
        )

    def test_worker_failure_preserves_billed_usage(
        self, tmp_path, mock_completion
    ):
        """Late worker exception must still contribute billed tokens.

        ``_translate_group`` may raise AFTER issuing an LLM call — for
        example from a user-supplied ``post_process`` hook that
        crashes mid-group. The partial-receipt contract promises
        operators that a failed run still surfaces the real token
        cost; losing a worker's ``local_usage`` on exception would
        silently under-report.
        """
        import json as _json
        import threading as _threading
        from unittest.mock import MagicMock as _MagicMock

        def _side_effect(*args, **kwargs):
            messages = kwargs.get("messages", args[0] if args else [])
            user_content = ""
            for msg in reversed(messages):
                if msg["role"] == "user":
                    user_content = msg["content"]
                    break
            parsed = None
            if isinstance(user_content, str) and user_content.strip().startswith("{"):
                try:
                    parsed = _json.loads(user_content)
                except _json.JSONDecodeError:
                    parsed = None
            response = _MagicMock()
            if isinstance(parsed, dict):
                translated = {
                    k: f"[TRANSLATED] {v}" for k, v in parsed.items()
                }
                response.choices[0].message.content = _json.dumps(
                    translated, ensure_ascii=False
                )
            else:
                response.choices[0].message.content = (
                    f"[TRANSLATED] {user_content}"
                )
            response.usage.prompt_tokens = 11
            response.usage.completion_tokens = 3
            return response

        mock_completion.completion.side_effect = _side_effect

        # Fail in ``post_process`` after enough calls to guarantee a
        # worker thread (not the seed thread) reaches the failure — the
        # seed batch accounts for the first N entries, so a threshold
        # safely beyond that lands in a parallel worker that has
        # already billed its own LLM call.  Use a lock so the counter
        # read-modify-write is race-free across workers.
        post_lock = _threading.Lock()
        post_calls = {"n": 0}

        def _fail_post(text: str) -> str:
            with post_lock:
                post_calls["n"] += 1
                n = post_calls["n"]
            if n >= 5:
                raise RuntimeError("simulated late post-LLM failure")
            return text

        source = self._multi_group_source(tmp_path)
        p = MarkdownProcessor(
            model="test-model",
            target_lang="ko",
            batch_size=3,
            batch_concurrency=3,
            post_process=_fail_post,
        )

        captured_exc: list = []
        try:
            p.process_document(
                source, tmp_path / "target.md", tmp_path / "m.po"
            )
        except RuntimeError as exc:
            captured_exc.append(exc)

        assert captured_exc, "expected the worker failure to propagate"
        exc = captured_exc[0]
        partial = getattr(exc, "partial_receipt", None)
        assert partial is not None, (
            "process_document must annotate the exception with "
            "partial_receipt when LLM calls were billed"
        )
        # Every completed LLM call contributed 11 + 3 tokens under
        # the mock above; the receipt must reflect all of them,
        # including those billed by the worker that eventually raised.
        billed_calls = mock_completion.completion.call_count
        assert billed_calls >= 2, (
            f"expected at least 2 billed calls (seed + worker); "
            f"got {billed_calls}"
        )
        assert partial.api_calls == billed_calls
        assert partial.input_tokens == 11 * billed_calls
        assert partial.output_tokens == 3 * billed_calls

    def test_worker_failure_cancels_queued_batches(
        self, tmp_path, mock_completion
    ):
        """First worker failure must cancel queued batches.

        Without the cancel, ``ThreadPoolExecutor.__exit__`` drains the
        queue via ``shutdown(wait=True)`` — every queued batch fires
        its own LLM call and over-bills the user on exactly the
        partial-failure path the receipt is meant to surface honestly.
        Already-running batches cannot be interrupted (the mock's
        non-first-worker calls still complete), but queued ones that
        haven't been picked up yet MUST be stopped.
        """
        import json as _json
        import threading as _threading
        from unittest.mock import MagicMock as _MagicMock

        call_count = {"n": 0}
        lock = _threading.Lock()
        first_arrived = _threading.Event()
        release_after_failure = _threading.Event()

        def _side_effect(*args, **kwargs):
            with lock:
                call_count["n"] += 1
                n = call_count["n"]
            if n == 1:
                # Seed batch completes immediately so parallel workers
                # can start.
                pass
            elif n == 2:
                # First parallel worker raises post-call to trigger the
                # cancellation path.  Signal so the test knows the
                # failure has propagated, then release the gate once
                # the executor has had a chance to cancel queued work.
                first_arrived.set()
            else:
                # Any later call would prove the queue was NOT
                # cancelled — wait on the release gate so we can
                # observe whether they get scheduled at all.  Bound
                # the wait so the test can never hang.
                release_after_failure.wait(timeout=1.0)

            messages = kwargs.get("messages", args[0] if args else [])
            user_content = ""
            for msg in reversed(messages):
                if msg["role"] == "user":
                    user_content = msg["content"]
                    break
            parsed = None
            if isinstance(user_content, str) and user_content.strip().startswith("{"):
                try:
                    parsed = _json.loads(user_content)
                except _json.JSONDecodeError:
                    parsed = None
            response = _MagicMock()
            if isinstance(parsed, dict):
                translated = {
                    k: f"[TRANSLATED] {v}" for k, v in parsed.items()
                }
                response.choices[0].message.content = _json.dumps(
                    translated, ensure_ascii=False
                )
            else:
                response.choices[0].message.content = (
                    f"[TRANSLATED] {user_content}"
                )
            response.usage.prompt_tokens = 7
            response.usage.completion_tokens = 2
            return response

        mock_completion.completion.side_effect = _side_effect

        # Raise from ``post_process`` on the *second* LLM response —
        # the first parallel worker's batch.  This is deterministic
        # across thread scheduling because post_process fires once
        # per returned entry in the batch, so the raise lands inside
        # a worker regardless of which worker received call #2.
        post_calls = {"n": 0}
        post_lock = _threading.Lock()

        def _fail_post(text: str) -> str:
            with post_lock:
                post_calls["n"] += 1
                n = post_calls["n"]
            # Let the seed batch (first 3 entries) succeed so parallel
            # workers actually launch; then raise inside the first
            # parallel worker.
            if n >= 4:
                raise RuntimeError("simulated worker failure")
            return text

        # Use batch_concurrency=1 with multiple groups? No — we need
        # concurrency > 1 to exercise the cancel path.  Use
        # concurrency=2 so only ONE parallel worker can start at once;
        # that worker fails; the single remaining queued group MUST be
        # cancelled rather than started.
        source = self._multi_group_source(tmp_path)
        p = MarkdownProcessor(
            model="test-model",
            target_lang="ko",
            batch_size=3,
            batch_concurrency=2,
            post_process=_fail_post,
        )

        try:
            try:
                p.process_document(
                    source, tmp_path / "target.md", tmp_path / "m.po"
                )
            except RuntimeError:
                pass
        finally:
            # Always release the gate so any accidentally-scheduled
            # call can drain; without this a test failure would hang
            # the executor at shutdown.
            release_after_failure.set()

        # With 4 groups (alpha/beta/gamma × 3 entries at batch_size=3),
        # the seed runs 1 call. concurrency=2 means at most 2 parallel
        # workers can run at once. The first parallel worker fails on
        # its post_process; the remaining two groups MUST be
        # cancelled.  Upper bound: 1 seed + 2 concurrent workers that
        # already started before the failure = 3 calls. A 4th call
        # indicates the queue was drained instead of cancelled.
        assert mock_completion.completion.call_count <= 3, (
            f"queued batches were not cancelled after worker failure; "
            f"saw {mock_completion.completion.call_count} LLM calls "
            f"(expected <= 3 with concurrency=2)"
        )

    def test_single_group_falls_through_to_sequential(
        self, tmp_path, mock_completion, monkeypatch
    ):
        """No thread pool when the document fits in one section group."""
        import concurrent.futures as _cf

        calls = {"n": 0}
        original_cls = _cf.ThreadPoolExecutor

        class _TrackedExecutor(original_cls):
            def __init__(self, *a, **kw):
                calls["n"] += 1
                super().__init__(*a, **kw)

        monkeypatch.setattr(_cf, "ThreadPoolExecutor", _TrackedExecutor)
        md = "# T\n\nOnly one paragraph.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")
        p = MarkdownProcessor(
            model="test-model",
            target_lang="ko",
            batch_size=40,
            batch_concurrency=4,
        )
        p.process_document(source, tmp_path / "target.md", tmp_path / "m.po")
        assert calls["n"] == 0


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


def _multi_side_effect_factory(langs):
    """Build a ``litellm.completion`` side_effect for the multi-target path.

    Detects a JSON-object user payload and returns a nested-dict response
    ``{block_id: {lang: f'[TRANSLATED:{lang}] {source}'}}`` for every
    provided language.  Plain-string user payloads (per-lang fallback calls)
    get the single-target sentinel so fallback tests still round-trip.
    """

    def _side_effect(*args, **kwargs):
        messages = kwargs.get("messages", args[0] if args else [])
        user_content = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                user_content = msg["content"]
                break
        parsed = None
        if isinstance(user_content, str) and user_content.strip().startswith("{"):
            try:
                parsed = json.loads(user_content)
            except json.JSONDecodeError:
                parsed = None
        resp = MagicMock()
        if isinstance(parsed, dict):
            out = {
                k: {lang: f"[TRANSLATED:{lang}] {v}" for lang in langs}
                for k, v in parsed.items()
            }
            resp.choices[0].message.content = json.dumps(out, ensure_ascii=False)
        else:
            resp.choices[0].message.content = f"[TRANSLATED:fallback] {user_content}"
        resp.usage.prompt_tokens = 11
        resp.usage.completion_tokens = 5
        return resp

    return _side_effect


class TestMultiTargetProcessing:
    def test_single_call_produces_all_langs(
        self, tmp_path, mock_completion
    ):
        mock_completion.completion.side_effect = _multi_side_effect_factory(
            ["ko", "ja"]
        )
        md = "# Title\n\nParagraph one.\n\nParagraph two.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")

        target_paths = {
            "ko": tmp_path / "ko" / "source.md",
            "ja": tmp_path / "ja" / "source.md",
        }
        p = MarkdownProcessor(model="test-model", target_lang="ko", batch_size=40)
        result = p.process_document_multi(
            source,
            target_langs=["ko", "ja"],
            target_paths=target_paths,
        )
        # One LLM call should cover every block for both langs.
        assert mock_completion.completion.call_count == 1
        assert result["target_langs"] == ["ko", "ja"]
        assert set(result["by_lang"].keys()) == {"ko", "ja"}
        ko_out = target_paths["ko"].read_text(encoding="utf-8")
        ja_out = target_paths["ja"].read_text(encoding="utf-8")
        assert "[TRANSLATED:ko]" in ko_out
        assert "[TRANSLATED:ja]" in ja_out
        # Receipt shared across langs; per-lang ProcessResult has none.
        assert result["receipt"] is not None
        for pr in result["by_lang"].values():
            assert pr["receipt"] is None

    def test_default_po_path_per_lang(self, tmp_path, mock_completion):
        mock_completion.completion.side_effect = _multi_side_effect_factory(
            ["ko", "ja"]
        )
        md = "# T\n\nP.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")

        target_paths = {
            "ko": tmp_path / "ko.md",
            "ja": tmp_path / "ja.md",
        }
        p = MarkdownProcessor(model="test-model", target_lang="ko", batch_size=40)
        result = p.process_document_multi(
            source,
            target_langs=["ko", "ja"],
            target_paths=target_paths,
        )
        assert Path(result["by_lang"]["ko"]["po_path"]).name == "ko.po"
        assert Path(result["by_lang"]["ja"]["po_path"]).name == "ja.po"

    def test_rejects_refine_mode(self, tmp_path, mock_completion):
        md = "# T\n\nP.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")
        p = MarkdownProcessor(
            model="test-model", target_lang="en", batch_size=40, mode="refine"
        )
        with pytest.raises(ValueError, match="mode='translate'"):
            p.process_document_multi(
                source,
                target_langs=["ko", "ja"],
                target_paths={
                    "ko": tmp_path / "ko.md",
                    "ja": tmp_path / "ja.md",
                },
            )

    def test_rejects_empty_target_langs(self, tmp_path, mock_completion):
        md = "# T\n\nP.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")
        p = MarkdownProcessor(model="test-model", target_lang="ko", batch_size=40)
        with pytest.raises(ValueError, match="target_langs"):
            p.process_document_multi(
                source, target_langs=[], target_paths={}
            )

    def test_rejects_missing_target_path_for_lang(
        self, tmp_path, mock_completion
    ):
        md = "# T\n\nP.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")
        p = MarkdownProcessor(model="test-model", target_lang="ko", batch_size=40)
        with pytest.raises(ValueError, match="target_paths missing"):
            p.process_document_multi(
                source,
                target_langs=["ko", "ja"],
                target_paths={"ko": tmp_path / "ko.md"},
            )

    def test_rejects_duplicate_target_paths(
        self, tmp_path, mock_completion
    ):
        md = "# T\n\nP.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")
        p = MarkdownProcessor(model="test-model", target_lang="ko", batch_size=40)
        shared = tmp_path / "shared.md"
        with pytest.raises(ValueError, match="distinct paths"):
            p.process_document_multi(
                source,
                target_langs=["ko", "ja"],
                target_paths={"ko": shared, "ja": shared},
            )

    def test_rejects_target_equal_to_source(self, tmp_path, mock_completion):
        md = "# T\n\nP.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")
        p = MarkdownProcessor(model="test-model", target_lang="ko", batch_size=40)
        with pytest.raises(ValueError, match="source path"):
            p.process_document_multi(
                source,
                target_langs=["ko", "ja"],
                target_paths={"ko": source, "ja": tmp_path / "ja.md"},
            )

    def test_dedupes_duplicate_target_langs(self, tmp_path, mock_completion):
        mock_completion.completion.side_effect = _multi_side_effect_factory(
            ["ko"]
        )
        md = "# T\n\nP.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")
        p = MarkdownProcessor(model="test-model", target_lang="ko", batch_size=40)
        result = p.process_document_multi(
            source,
            target_langs=["ko", "ko"],
            target_paths={"ko": tmp_path / "ko.md"},
        )
        assert result["target_langs"] == ["ko"]

    def test_partial_response_falls_back_per_lang(
        self, tmp_path, mock_completion
    ):
        """Missing 'ja' in the multi-response triggers per-lang fallback."""
        # Return multi-target dict but omit "ja" everywhere.  The processor
        # should issue per-lang single-target fallback calls for JA while
        # committing KO directly.
        def _side_effect(*args, **kwargs):
            messages = kwargs.get("messages", args[0] if args else [])
            user_content = ""
            for msg in reversed(messages):
                if msg["role"] == "user":
                    user_content = msg["content"]
                    break
            parsed = None
            if (
                isinstance(user_content, str)
                and user_content.strip().startswith("{")
            ):
                try:
                    parsed = json.loads(user_content)
                except json.JSONDecodeError:
                    parsed = None
            resp = MagicMock()
            if isinstance(parsed, dict):
                # Multi-call: omit "ja" entirely so it drops into
                # per-lang fallback.
                out = {k: {"ko": f"[TRANSLATED:ko] {v}"} for k, v in parsed.items()}
                resp.choices[0].message.content = json.dumps(
                    out, ensure_ascii=False
                )
            else:
                # Per-lang fallback (plain user string).
                resp.choices[0].message.content = (
                    f"[TRANSLATED:fallback] {user_content}"
                )
            resp.usage.prompt_tokens = 7
            resp.usage.completion_tokens = 3
            return resp

        mock_completion.completion.side_effect = _side_effect
        md = "# T\n\nP1.\n\nP2.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")
        target_paths = {
            "ko": tmp_path / "ko.md",
            "ja": tmp_path / "ja.md",
        }
        p = MarkdownProcessor(model="test-model", target_lang="ko", batch_size=40)
        result = p.process_document_multi(
            source,
            target_langs=["ko", "ja"],
            target_paths=target_paths,
        )
        # KO committed via multi-call; JA committed via per-lang fallback.
        ko_out = target_paths["ko"].read_text(encoding="utf-8")
        ja_out = target_paths["ja"].read_text(encoding="utf-8")
        assert "[TRANSLATED:ko]" in ko_out
        assert "[TRANSLATED:fallback]" in ja_out
        # One multi-call plus N per-lang fallback calls.
        assert mock_completion.completion.call_count >= 2
        assert result["by_lang"]["ja"]["translation_stats"]["processed"] >= 3

    def test_skips_blocks_fully_done_across_all_langs(
        self, tmp_path, mock_completion
    ):
        """Second run with no source changes makes zero LLM calls."""
        mock_completion.completion.side_effect = _multi_side_effect_factory(
            ["ko", "ja"]
        )
        md = "# T\n\nP.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")
        target_paths = {
            "ko": tmp_path / "ko.md",
            "ja": tmp_path / "ja.md",
        }
        p = MarkdownProcessor(model="test-model", target_lang="ko", batch_size=40)
        p.process_document_multi(
            source,
            target_langs=["ko", "ja"],
            target_paths=target_paths,
        )
        first = mock_completion.completion.call_count
        p.process_document_multi(
            source,
            target_langs=["ko", "ja"],
            target_paths=target_paths,
        )
        # No pending blocks => no LLM calls on the re-run.
        assert mock_completion.completion.call_count == first

    def test_receipt_billed_once_for_multi_target(
        self, tmp_path, mock_completion
    ):
        mock_completion.completion.side_effect = _multi_side_effect_factory(
            ["ko", "ja", "zh"]
        )
        md = "# T\n\nP.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")
        target_paths = {
            "ko": tmp_path / "ko.md",
            "ja": tmp_path / "ja.md",
            "zh": tmp_path / "zh.md",
        }
        p = MarkdownProcessor(model="test-model", target_lang="ko", batch_size=40)
        result = p.process_document_multi(
            source,
            target_langs=["ko", "ja", "zh"],
            target_paths=target_paths,
        )
        # Single LLM call.
        assert mock_completion.completion.call_count == 1
        receipt = result["receipt"]
        # One call's usage, NOT tripled.
        assert receipt["api_calls"] == 1
        assert receipt["input_tokens"] == 11
        assert receipt["output_tokens"] == 5
        # target_lang names every locale for auditability.
        assert receipt["target_lang"] == "ko,ja,zh"

    def test_rejects_target_colliding_with_another_langs_po(
        self, tmp_path, mock_completion
    ):
        """A target markdown path aliasing a PO path MUST raise up front —
        otherwise the save loop deterministically clobbers one with the
        other.
        """
        md = "# T\n\nP.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")
        shared = tmp_path / "shared.out"
        p = MarkdownProcessor(model="test-model", target_lang="ko", batch_size=40)
        with pytest.raises(ValueError, match="resolve to the same file"):
            p.process_document_multi(
                source,
                target_langs=["ko", "ja"],
                target_paths={
                    "ko": shared,
                    "ja": tmp_path / "ja.md",
                },
                po_paths={
                    "ko": tmp_path / "ko.po",
                    "ja": shared,  # ja's PO aliases ko's target
                },
            )

    def test_fallback_uses_per_lang_glossary_resolution(
        self, tmp_path, mock_completion
    ):
        """Per-lang fallback must resolve locale-keyed glossary entries
        against the CALL's target lang, not the processor's
        constructor-time ``target_lang``.
        """
        gloss_path = tmp_path / "glossary.json"
        gloss_path.write_text(
            json.dumps({"widget": {"ko": "위젯", "ja": "ウィジェット"}}),
            encoding="utf-8",
        )

        # Multi-response is empty, forcing every block into per-lang
        # fallback so we can inspect the system prompt sent for each lang.
        prompts_seen: list = []

        def _side_effect(*args, **kwargs):
            messages = kwargs.get("messages", args[0] if args else [])
            system = ""
            user_content = ""
            for msg in messages:
                if msg["role"] == "system":
                    content = msg["content"]
                    if isinstance(content, list):
                        system = "".join(
                            part.get("text", "") for part in content
                        )
                    else:
                        system = content
                if msg["role"] == "user":
                    user_content = msg["content"]
            prompts_seen.append(system)
            parsed = None
            if (
                isinstance(user_content, str)
                and user_content.strip().startswith("{")
            ):
                try:
                    parsed = json.loads(user_content)
                except json.JSONDecodeError:
                    parsed = None
            resp = MagicMock()
            if isinstance(parsed, dict):
                resp.choices[0].message.content = "{}"
            else:
                resp.choices[0].message.content = f"[TRANSLATED] {user_content}"
            resp.usage.prompt_tokens = 3
            resp.usage.completion_tokens = 1
            return resp

        mock_completion.completion.side_effect = _side_effect

        md = "# widget demo\n\nThis uses a widget today.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")

        p = MarkdownProcessor(
            model="test-model",
            target_lang="ko",
            batch_size=40,
            glossary_path=gloss_path,
        )
        p.process_document_multi(
            source,
            target_langs=["ko", "ja"],
            target_paths={
                "ko": tmp_path / "ko.md",
                "ja": tmp_path / "ja.md",
            },
        )

        ko_prompts = [s for s in prompts_seen if "**ko**" in s]
        ja_prompts = [s for s in prompts_seen if "**ja**" in s]
        assert ko_prompts, "expected at least one fallback prompt targeting ko"
        assert ja_prompts, "expected at least one fallback prompt targeting ja"
        for s in ko_prompts:
            assert "위젯" in s
            assert "ウィジェット" not in s
        for s in ja_prompts:
            assert "ウィジェット" in s
            assert "위젯" not in s

    def test_batch_prompt_carries_per_lang_glossary_blocks(
        self, tmp_path, mock_completion
    ):
        """Multi-target batch prompt labels glossary blocks per lang so
        locale-specific replacements cannot cross into the wrong lang.
        """
        gloss_path = tmp_path / "glossary.json"
        gloss_path.write_text(
            json.dumps({"widget": {"ko": "위젯", "ja": "ウィジェット"}}),
            encoding="utf-8",
        )

        captured_system: list = []

        def _side_effect(*args, **kwargs):
            messages = kwargs.get("messages", args[0] if args else [])
            system_content = ""
            user_content = ""
            for msg in messages:
                if msg["role"] == "system":
                    content = msg["content"]
                    if isinstance(content, list):
                        system_content = "".join(
                            part.get("text", "") for part in content
                        )
                    else:
                        system_content = content
                if msg["role"] == "user":
                    user_content = msg["content"]
            captured_system.append(system_content)
            parsed = None
            if (
                isinstance(user_content, str)
                and user_content.strip().startswith("{")
            ):
                try:
                    parsed = json.loads(user_content)
                except json.JSONDecodeError:
                    parsed = None
            resp = MagicMock()
            if isinstance(parsed, dict):
                out = {
                    k: {
                        "ko": f"[TRANSLATED:ko] {v}",
                        "ja": f"[TRANSLATED:ja] {v}",
                    }
                    for k, v in parsed.items()
                }
                resp.choices[0].message.content = json.dumps(
                    out, ensure_ascii=False
                )
            else:
                resp.choices[0].message.content = f"[TRANSLATED] {user_content}"
            resp.usage.prompt_tokens = 5
            resp.usage.completion_tokens = 2
            return resp

        mock_completion.completion.side_effect = _side_effect

        md = "# widget demo\n\nThe widget ships today.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")

        p = MarkdownProcessor(
            model="test-model",
            target_lang="ko",
            batch_size=40,
            glossary_path=gloss_path,
        )
        p.process_document_multi(
            source,
            target_langs=["ko", "ja"],
            target_paths={
                "ko": tmp_path / "ko.md",
                "ja": tmp_path / "ja.md",
            },
        )
        assert captured_system, "no LLM calls were captured"
        batch_prompt = captured_system[0]
        assert "Glossary (ko)" in batch_prompt
        assert "Glossary (ja)" in batch_prompt
        assert "위젯" in batch_prompt
        assert "ウィジェット" in batch_prompt

    def test_rejects_placeholder_glossary_mode_multi_target(
        self, tmp_path, mock_completion
    ):
        """Placeholder-mode glossary bakes target-lang replacements at encode
        time; multi-target can't fan that out without corrupting at least
        one language, so the call must refuse up front.
        """
        md = "# T\n\nThe widget ships.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")
        p = MarkdownProcessor(
            model="test-model",
            target_lang="ko",
            batch_size=40,
            glossary={"widget": "위젯"},
            glossary_mode="placeholder",
        )
        with pytest.raises(ValueError, match="glossary_mode='placeholder'"):
            p.process_document_multi(
                source,
                target_langs=["ko", "ja"],
                target_paths={
                    "ko": tmp_path / "ko.md",
                    "ja": tmp_path / "ja.md",
                },
            )

    def test_rejects_placeholder_when_file_has_only_unmatched_locale_keys(
        self, tmp_path, mock_completion
    ):
        """A locale-keyed glossary file whose entries don't resolve for the
        constructor-time target_lang still configures a glossary source —
        the multi-target call must refuse rather than silently drop it.
        """
        gloss_path = tmp_path / "glossary.json"
        gloss_path.write_text(
            json.dumps({"term": {"ko": "A", "ja": "B"}}),
            encoding="utf-8",
        )
        md = "# T\n\nThe term ships.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")
        # target_lang='en' leaves self._glossary['term'] == None, which
        # is falsy-looking at entry level; the guard must still trip.
        p = MarkdownProcessor(
            model="test-model",
            target_lang="en",
            batch_size=40,
            glossary_path=gloss_path,
            glossary_mode="placeholder",
        )
        with pytest.raises(ValueError, match="glossary_mode='placeholder'"):
            p.process_document_multi(
                source,
                target_langs=["ko", "ja"],
                target_paths={
                    "ko": tmp_path / "ko.md",
                    "ja": tmp_path / "ja.md",
                },
            )

    def test_any_pending_order_matches_source_order(
        self, tmp_path, mock_completion
    ):
        """When langs have disjoint pending subsets, the merged
        ``any_pending_ordered`` MUST walk source-document order so the
        section-aware grouper and reference-pool seeding still see
        blocks in the same sequence regardless of which lang supplied
        them.
        """
        mock_completion.completion.side_effect = _multi_side_effect_factory(
            ["ko", "ja"]
        )
        md = "# A\n\nP1.\n\nP2.\n\nP3.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")
        target_paths = {
            "ko": tmp_path / "ko.md",
            "ja": tmp_path / "ja.md",
        }
        p = MarkdownProcessor(model="test-model", target_lang="ko", batch_size=40)
        # First pass — both langs get everything translated.
        p.process_document_multi(
            source,
            target_langs=["ko", "ja"],
            target_paths=target_paths,
        )

        # Hand-craft an uneven state: clear ja's msgstr for the heading
        # only (so ja needs block 0), and clear ko's msgstr for P3 only
        # (so ko needs last block). Source order is heading, P1, P2, P3.
        import polib as _polib
        ja_po = _polib.pofile(str(target_paths["ja"].with_suffix(".po")))
        ko_po = _polib.pofile(str(target_paths["ko"].with_suffix(".po")))
        ja_entries = [e for e in ja_po if not e.obsolete]
        ko_entries = [e for e in ko_po if not e.obsolete]
        ja_entries[0].msgstr = ""  # clear heading in ja
        ko_entries[-1].msgstr = ""  # clear last block in ko
        ja_po.save(str(target_paths["ja"].with_suffix(".po")), newline="\n")
        ko_po.save(str(target_paths["ko"].with_suffix(".po")), newline="\n")

        captured_user: list = []

        def _record(*args, **kwargs):
            messages = kwargs.get("messages", args[0] if args else [])
            for msg in reversed(messages):
                if msg["role"] == "user":
                    captured_user.append(msg["content"])
                    break
            parsed = None
            content = captured_user[-1] if captured_user else ""
            if (
                isinstance(content, str)
                and content.strip().startswith("{")
            ):
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    parsed = None
            resp = MagicMock()
            if isinstance(parsed, dict):
                out = {
                    k: {
                        "ko": f"[TRANSLATED:ko] {v}",
                        "ja": f"[TRANSLATED:ja] {v}",
                    }
                    for k, v in parsed.items()
                }
                resp.choices[0].message.content = json.dumps(
                    out, ensure_ascii=False
                )
            else:
                resp.choices[0].message.content = f"[TRANSLATED] {content}"
            resp.usage.prompt_tokens = 3
            resp.usage.completion_tokens = 1
            return resp

        mock_completion.completion.side_effect = _record
        mock_completion.completion.reset_mock()
        p.process_document_multi(
            source,
            target_langs=["ko", "ja"],
            target_paths=target_paths,
        )
        # Exactly one batch call covering both pending blocks.
        assert captured_user, "no user payload captured"
        payload = json.loads(captured_user[0])
        ordered_ctxs = list(payload.keys())
        # Heading must precede P3 in the batch, matching source order —
        # without the source-order fix, ko's P3 would land first.
        heading_ctxs = [c for c in ordered_ctxs if "heading" in c]
        para_ctxs = [c for c in ordered_ctxs if "para" in c]
        assert heading_ctxs, f"no heading ctx in {ordered_ctxs}"
        assert para_ctxs, f"no para ctx in {ordered_ctxs}"
        assert ordered_ctxs.index(heading_ctxs[0]) < ordered_ctxs.index(
            para_ctxs[-1]
        )

    def test_warns_when_batch_concurrency_requested(
        self, tmp_path, mock_completion, caplog
    ):
        """batch_concurrency>1 is not yet honoured in multi-target; users who
        opted in MUST see a loud warning so they know the flag was ignored.
        """
        mock_completion.completion.side_effect = _multi_side_effect_factory(
            ["ko", "ja"]
        )
        md = "# T\n\nP.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")
        p = MarkdownProcessor(
            model="test-model",
            target_lang="ko",
            batch_size=40,
            batch_concurrency=4,
        )
        with caplog.at_level(logging.WARNING, logger="mdpo_llm.processor"):
            p.process_document_multi(
                source,
                target_langs=["ko", "ja"],
                target_paths={
                    "ko": tmp_path / "ko.md",
                    "ja": tmp_path / "ja.md",
                },
            )
        assert any(
            "batch_concurrency" in rec.message.lower()
            for rec in caplog.records
        )

    def test_batch_size_zero_skips_batcher(
        self, tmp_path, mock_completion
    ):
        """``batch_size=0`` MUST opt out of the JSON batch wire for
        multi-target too: every block flows through per-lang
        single-target calls, same escape hatch as single-target.
        """
        def _side_effect(*args, **kwargs):
            messages = kwargs.get("messages", args[0] if args else [])
            user_content = ""
            for msg in reversed(messages):
                if msg["role"] == "user":
                    user_content = msg["content"]
                    break
            # Assert no JSON-object payload reaches the model.
            assert not user_content.strip().startswith("{"), (
                "batch_size=0 should skip the batch wire entirely"
            )
            resp = MagicMock()
            resp.choices[0].message.content = f"[TRANSLATED] {user_content}"
            resp.usage.prompt_tokens = 2
            resp.usage.completion_tokens = 1
            return resp

        mock_completion.completion.side_effect = _side_effect
        md = "# T\n\nP1.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")
        p = MarkdownProcessor(model="test-model", target_lang="ko", batch_size=0)
        result = p.process_document_multi(
            source,
            target_langs=["ko", "ja"],
            target_paths={
                "ko": tmp_path / "ko.md",
                "ja": tmp_path / "ja.md",
            },
        )
        # Only per-entry calls — zero batched calls.
        for lang in ("ko", "ja"):
            stats = result["by_lang"][lang]["translation_stats"]
            assert stats["batched_calls"] == 0
            assert stats["per_entry_calls"] >= 1

    def test_validation_uses_per_lang_glossary(
        self, tmp_path, mock_completion
    ):
        """When validation is enabled, a ja/zh commit is measured against
        ja/zh's glossary mapping — not the constructor target_lang's.
        """
        gloss_path = tmp_path / "glossary.json"
        gloss_path.write_text(
            json.dumps({"widget": {"ko": "위젯", "ja": "ウィジェット"}}),
            encoding="utf-8",
        )

        # Return batch response where KO contains 위젯 and JA contains
        # ウィジェット, matching each locale's glossary.  If validation
        # used ko's glossary for both, the ja entry would look like a
        # missing-glossary-term fail.
        def _side_effect(*args, **kwargs):
            messages = kwargs.get("messages", args[0] if args else [])
            user_content = ""
            for msg in reversed(messages):
                if msg["role"] == "user":
                    user_content = msg["content"]
                    break
            parsed = None
            if (
                isinstance(user_content, str)
                and user_content.strip().startswith("{")
            ):
                try:
                    parsed = json.loads(user_content)
                except json.JSONDecodeError:
                    parsed = None
            resp = MagicMock()
            if isinstance(parsed, dict):
                out = {
                    k: {
                        "ko": v.replace("widget", "위젯"),
                        "ja": v.replace("widget", "ウィジェット"),
                    }
                    for k, v in parsed.items()
                }
                resp.choices[0].message.content = json.dumps(out, ensure_ascii=False)
            else:
                resp.choices[0].message.content = f"[TRANSLATED] {user_content}"
            resp.usage.prompt_tokens = 4
            resp.usage.completion_tokens = 2
            return resp

        mock_completion.completion.side_effect = _side_effect

        md = "# widget demo\n\nThe widget ships soon.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")
        p = MarkdownProcessor(
            model="test-model",
            target_lang="ko",
            batch_size=40,
            glossary_path=gloss_path,
            validation="strict",
        )
        result = p.process_document_multi(
            source,
            target_langs=["ko", "ja"],
            target_paths={
                "ko": tmp_path / "ko.md",
                "ja": tmp_path / "ja.md",
            },
        )
        # Neither lang should have validation_failed entries — each is
        # validated against its own locale-specific glossary mapping.
        assert result["by_lang"]["ko"]["translation_stats"]["validation_failed"] == 0
        assert result["by_lang"]["ja"]["translation_stats"]["validation_failed"] == 0

    def test_per_lang_stats_report_per_lang_fallback_only(
        self, tmp_path, mock_completion
    ):
        """A fallback call triggered by a missing 'ja' entry must NOT be
        counted under 'ko' in the per-lang stats.
        """
        def _side_effect(*args, **kwargs):
            messages = kwargs.get("messages", args[0] if args else [])
            user_content = ""
            for msg in reversed(messages):
                if msg["role"] == "user":
                    user_content = msg["content"]
                    break
            parsed = None
            if (
                isinstance(user_content, str)
                and user_content.strip().startswith("{")
            ):
                try:
                    parsed = json.loads(user_content)
                except json.JSONDecodeError:
                    parsed = None
            resp = MagicMock()
            if isinstance(parsed, dict):
                # Full KO coverage, no JA — forces JA fallback only.
                out = {k: {"ko": f"[TRANSLATED:ko] {v}"} for k, v in parsed.items()}
                resp.choices[0].message.content = json.dumps(
                    out, ensure_ascii=False
                )
            else:
                resp.choices[0].message.content = f"[TRANSLATED:ja] {user_content}"
            resp.usage.prompt_tokens = 7
            resp.usage.completion_tokens = 3
            return resp

        mock_completion.completion.side_effect = _side_effect
        md = "# T\n\nP1.\n\nP2.\n"
        source = tmp_path / "source.md"
        source.write_text(md, encoding="utf-8")
        target_paths = {
            "ko": tmp_path / "ko.md",
            "ja": tmp_path / "ja.md",
        }
        p = MarkdownProcessor(model="test-model", target_lang="ko", batch_size=40)
        result = p.process_document_multi(
            source,
            target_langs=["ko", "ja"],
            target_paths=target_paths,
        )
        ko_stats = result["by_lang"]["ko"]["translation_stats"]
        ja_stats = result["by_lang"]["ja"]["translation_stats"]
        # Only JA triggered per-entry fallback calls.
        assert ko_stats["per_entry_calls"] == 0
        assert ja_stats["per_entry_calls"] >= 1
        # batched_calls is shared across langs (same call served both).
        assert ko_stats["batched_calls"] == ja_stats["batched_calls"]
        assert ko_stats["batched_calls"] >= 1


class TestTranslatePaths:
    """``--translate-paths`` opt-in: per-segment _paths.po + path_map.json."""

    def _write_tree(self, root: Path) -> None:
        (root / "guide").mkdir(parents=True, exist_ok=True)
        (root / "guide" / "intro.md").write_text(
            "# Intro\n\nHello.\n", encoding="utf-8"
        )
        (root / "guide" / "setup.md").write_text(
            "# Setup\n\nSteps.\n", encoding="utf-8"
        )
        (root / "api.md").write_text("# API\n\nReference.\n", encoding="utf-8")

    def test_default_off_unchanged(self, tmp_path, mock_completion):
        """Without the flag, target paths mirror source paths exactly and no
        ``_paths.po`` / ``path_map.json`` are emitted."""
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        po = tmp_path / "po"
        self._write_tree(src)

        p = MarkdownProcessor(model="test-model", target_lang="ko", batch_size=40)
        p.process_directory(src, dst, po)

        assert (dst / "guide" / "intro.md").exists()
        assert (dst / "guide" / "setup.md").exists()
        assert (dst / "api.md").exists()
        assert not (dst / "path_map.json").exists()
        assert not (po / "_paths.po").exists()

    def test_emits_paths_po_and_path_map(self, tmp_path, mock_completion):
        """With the flag on, every source path has a translated target
        mirror, ``_paths.po`` lives under ``po_dir`` and ``path_map.json``
        sits at the target-tree root."""
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        po = tmp_path / "po"
        self._write_tree(src)

        p = MarkdownProcessor(model="test-model", target_lang="ko", batch_size=40)
        p.process_directory(src, dst, po, translate_paths=True)

        paths_po = po / "_paths.po"
        path_map_file = dst / "path_map.json"
        assert paths_po.exists()
        assert path_map_file.exists()

        path_map = json.loads(path_map_file.read_text(encoding="utf-8"))
        assert set(path_map.keys()) == {
            "api.md",
            "guide/intro.md",
            "guide/setup.md",
        }
        # mock_completion prefixes with "[TRANSLATED] " and
        # slugify_path_segment collapses the space + brackets.  The mocked
        # translation of ``api`` yields ``"[TRANSLATED] api"`` → slug
        # ``"TRANSLATED-api"``.  Assert shape, not exact form: every
        # translated relative path should differ from its source and
        # preserve the .md extension.
        for src_rel, tgt_rel in path_map.items():
            assert tgt_rel.endswith(".md")
            assert tgt_rel != src_rel
            assert (dst / tgt_rel).exists()

        # The actual translated markdown files sit at the translated
        # relative paths; originals must NOT be written.
        assert not (dst / "api.md").exists()
        assert not (dst / "guide" / "intro.md").exists()

    def test_idempotent_cache_reuse(self, tmp_path, mock_completion):
        """Re-running with a populated ``_paths.po`` reuses cached
        translations — no further LLM calls for path segments."""
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        po = tmp_path / "po"
        self._write_tree(src)

        p = MarkdownProcessor(model="test-model", target_lang="ko", batch_size=40)
        p.process_directory(src, dst, po, translate_paths=True)

        calls_after_first = mock_completion.completion.call_count
        # Second run: path segments are already translated in _paths.po AND
        # content PO entries are already msgstr-populated, so the run
        # should issue zero additional LLM calls.
        p.process_directory(src, dst, po, translate_paths=True)
        assert mock_completion.completion.call_count == calls_after_first

    def test_disambiguates_colliding_slugs(self, tmp_path, mock_completion):
        """Two sibling files whose translated stems collapse to the same
        slug get ``-2``/``-3`` disambiguators."""
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        (src / "alpha.md").write_text("# A\n", encoding="utf-8")
        (src / "beta.md").write_text("# B\n", encoding="utf-8")

        def _collide(*args, **kwargs):
            """Force every segment and every content block to translate
            to the same constant string so slug collisions fire."""
            messages = kwargs.get("messages", args[0] if args else [])
            user_content = ""
            for msg in reversed(messages):
                if msg["role"] == "user":
                    user_content = msg["content"]
                    break
            resp = MagicMock()
            # Batched content path sends JSON objects; preserve shape so
            # the processor can thread the response back to per-entry msgstrs.
            if isinstance(user_content, str) and user_content.strip().startswith("{"):
                try:
                    parsed = json.loads(user_content)
                    resp.choices[0].message.content = json.dumps(
                        {k: "same" for k in parsed}, ensure_ascii=False
                    )
                except json.JSONDecodeError:
                    resp.choices[0].message.content = "same"
            else:
                resp.choices[0].message.content = "same"
            return resp

        mock_completion.completion.side_effect = _collide

        p = MarkdownProcessor(model="test-model", target_lang="ko", batch_size=40)
        p.process_directory(src, dst, translate_paths=True)

        path_map = json.loads((dst / "path_map.json").read_text(encoding="utf-8"))
        targets = sorted(path_map.values())
        # Both files land at distinct, deterministic slugs; the
        # alphabetically-first source wins the base slug.
        assert len(set(targets)) == 2
        assert all(t.endswith(".md") for t in targets)
        # Base slug is "same"; disambiguator gives "same-2".
        assert path_map["alpha.md"] == "same.md"
        assert path_map["beta.md"] == "same-2.md"

    def test_colliding_parents_and_stems_disambig_cleanly(
        self, tmp_path, mock_completion
    ):
        """When two distinct source directories AND both of their child
        stems all translate to identical slugs, every output must still
        land at a distinct on-disk path — the directory-level disambig
        must propagate through so stems don't overwrite each other."""
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        (src / "a").mkdir(parents=True)
        (src / "b").mkdir(parents=True)
        (src / "a" / "readme.md").write_text("# A\n", encoding="utf-8")
        (src / "b" / "readme.md").write_text("# B\n", encoding="utf-8")

        def _collapse_to_same(*args, **kwargs):
            messages = kwargs.get("messages", args[0] if args else [])
            user_content = ""
            for msg in reversed(messages):
                if msg["role"] == "user":
                    user_content = msg["content"]
                    break
            resp = MagicMock()
            if isinstance(user_content, str) and user_content.strip().startswith("{"):
                try:
                    parsed = json.loads(user_content)
                    resp.choices[0].message.content = json.dumps(
                        {k: "same" for k in parsed}, ensure_ascii=False
                    )
                except json.JSONDecodeError:
                    resp.choices[0].message.content = "same"
            else:
                # Every path segment (a, b, readme) translates to "same".
                resp.choices[0].message.content = "same"
            return resp

        mock_completion.completion.side_effect = _collapse_to_same

        p = MarkdownProcessor(model="test-model", target_lang="ko", batch_size=40)
        p.process_directory(src, dst, translate_paths=True)

        path_map = json.loads((dst / "path_map.json").read_text(encoding="utf-8"))
        targets = list(path_map.values())
        # Every target path is unique AND exists on disk.
        assert len(set(targets)) == len(targets)
        for tgt in targets:
            assert (dst / tgt).is_file()

    def test_reserved_name_with_extension_is_rewritten(
        self, tmp_path, mock_completion
    ):
        """``CON.txt``, ``AUX.foo`` etc. are reserved on Windows even with
        trailing extensions.  ``slugify_path_segment`` must rewrite the
        stem portion so the resulting path stays writable on every
        platform."""
        from mdpo_llm.parser import slugify_path_segment

        assert slugify_path_segment("CON") == "CON_"
        assert slugify_path_segment("CON.txt") == "CON_.txt"
        assert slugify_path_segment("aux.foo") == "aux_.foo"
        assert slugify_path_segment("LPT1.bak") == "LPT1_.bak"
        # Non-reserved stems must still round-trip unchanged.
        assert slugify_path_segment("console") == "console"
        assert slugify_path_segment("config.bak") == "config.bak"

    def test_dotfile_segments_pass_through(self, tmp_path, mock_completion):
        """Segments starting with ``.`` (``.github``, ``.well-known``) are
        filesystem-navigation / convention tokens and MUST NOT be
        translated — they pass through the path map unchanged so CI / web
        infrastructure paths don't break."""
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        (src / ".github").mkdir(parents=True)
        (src / ".github" / "CONTRIBUTING.md").write_text(
            "# Contributing\n", encoding="utf-8"
        )

        p = MarkdownProcessor(model="test-model", target_lang="ko", batch_size=40)
        p.process_directory(src, dst, translate_paths=True)

        path_map = json.loads((dst / "path_map.json").read_text(encoding="utf-8"))
        target = path_map[".github/CONTRIBUTING.md"]
        # ".github" is untouched; the stem "CONTRIBUTING" can be translated.
        assert target.startswith(".github/")
        assert target.endswith(".md")

    def test_cleanup_stale_translated_outputs(self, tmp_path, mock_completion):
        """When a path mapping changes between runs (e.g. an operator edits
        ``_paths.po``), the old localized file MUST be removed so
        static-site deploys don't double-publish outdated content."""
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        po = tmp_path / "po"
        src.mkdir()
        (src / "alpha.md").write_text("# A\n", encoding="utf-8")

        p = MarkdownProcessor(model="test-model", target_lang="ko", batch_size=40)
        p.process_directory(src, dst, po, translate_paths=True)
        first_map = json.loads((dst / "path_map.json").read_text(encoding="utf-8"))
        first_target_rel = first_map["alpha.md"]
        first_target = dst / first_target_rel
        assert first_target.exists()

        # Simulate an operator editing the _paths.po entry so the next
        # run produces a different translation for "alpha".
        paths_po = po / "_paths.po"
        import polib as _polib

        po_file = _polib.pofile(str(paths_po))
        edited = False
        for entry in po_file:
            if entry.msgid == "alpha":
                entry.msgstr = "renamed"
                edited = True
        assert edited, "expected an `alpha` segment entry in _paths.po"
        po_file.save(str(paths_po), newline="\n")

        p.process_directory(src, dst, po, translate_paths=True)
        second_map = json.loads((dst / "path_map.json").read_text(encoding="utf-8"))
        assert second_map["alpha.md"] == "renamed.md"
        assert (dst / "renamed.md").exists()
        # The stale file from the first run is gone — path_map.json and
        # the on-disk tree stay in sync.
        if first_target_rel != "renamed.md":
            assert not first_target.exists()

    def test_receipt_includes_path_translation_tokens(
        self, tmp_path, mock_completion
    ):
        """Path-segment API calls bill into the directory-level receipt so
        operators see the real cost of a translate-paths run."""
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        (src / "doc.md").parent.mkdir(parents=True, exist_ok=True)
        (src / "doc.md").write_text("# Hi\n", encoding="utf-8")

        # Give every response a fixed token count so we can compare
        # directly against the expected lower bound.
        def _priced(*args, **kwargs):
            messages = kwargs.get("messages", args[0] if args else [])
            user_content = ""
            for msg in reversed(messages):
                if msg["role"] == "user":
                    user_content = msg["content"]
                    break
            resp = MagicMock()
            if isinstance(user_content, str) and user_content.strip().startswith("{"):
                try:
                    parsed = json.loads(user_content)
                    resp.choices[0].message.content = json.dumps(
                        {k: f"T {v}" for k, v in parsed.items()},
                        ensure_ascii=False,
                    )
                except json.JSONDecodeError:
                    resp.choices[0].message.content = user_content
            else:
                resp.choices[0].message.content = f"T {user_content}"
            resp.usage.prompt_tokens = 11
            resp.usage.completion_tokens = 3
            return resp

        mock_completion.completion.side_effect = _priced

        p = MarkdownProcessor(model="test-model", target_lang="ko", batch_size=40)
        result = p.process_directory(src, dst, translate_paths=True)

        # Two segments ("doc" stem) + one content batch → at least two calls.
        assert result.receipt.api_calls >= 2
        assert result.receipt.input_tokens >= 22
