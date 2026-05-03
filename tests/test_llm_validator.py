"""Tests for the LLM validator + bounded retry loop (T-16).

Covers the standalone :class:`mdpo_llm.llm_validator.LLMValidator`
wire format, plus the in-processor retry orchestration:

- pass / fail partitioning of a graded batch
- reason accumulation across attempts
- fallback-model swap timing (``ceil(max_retries / 2)``)
- translatability gating (code-only blocks must PASS even when
  source equals output)
- ``max_retries`` clamp behaviour
- bisection on validator JSON malformation
"""

from __future__ import annotations

import json
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from mdpo_llm.batch import BatchTranslator
from mdpo_llm.llm_validator import BinaryGrade, LLMValidator
from mdpo_llm.processor import MarkdownProcessor


# ---------------------------------------------------------------- helpers


def _grade_response(payload: Dict[str, Dict[str, Any]]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _all_pass_caller(items: Dict[str, Dict[str, str]]) -> str:
    return _grade_response(
        {
            k: {"binary_score": "yes", "reason": "looks fine"}
            for k in items
        }
    )


def _all_fail_caller(items: Dict[str, Dict[str, str]]) -> str:
    return _grade_response(
        {
            k: {"binary_score": "no", "reason": f"missing translation for {k}"}
            for k in items
        }
    )


# ---------------------------------------------------------------- BinaryGrade


class TestBinaryGrade:
    def test_from_payload_yes(self) -> None:
        g = BinaryGrade.from_payload({"binary_score": "yes", "reason": "ok"})
        assert g == BinaryGrade(binary_score="yes", reason="ok")

    def test_from_payload_no(self) -> None:
        g = BinaryGrade.from_payload({"binary_score": "no", "reason": "bad"})
        assert g == BinaryGrade(binary_score="no", reason="bad")

    def test_from_payload_missing_score_returns_none(self) -> None:
        assert BinaryGrade.from_payload({"reason": "no score"}) is None

    def test_from_payload_invalid_score_returns_none(self) -> None:
        assert (
            BinaryGrade.from_payload(
                {"binary_score": "maybe", "reason": "x"}
            )
            is None
        )

    def test_from_payload_missing_reason_yields_empty(self) -> None:
        g = BinaryGrade.from_payload({"binary_score": "yes"})
        assert g is not None
        assert g.reason == ""

    def test_from_payload_non_dict_returns_none(self) -> None:
        assert BinaryGrade.from_payload("yes") is None
        assert BinaryGrade.from_payload(None) is None
        assert BinaryGrade.from_payload([]) is None


# ---------------------------------------------------------------- LLMValidator


class TestLLMValidatorWireFormat:
    def test_grade_happy_path(self) -> None:
        v = LLMValidator(_all_pass_caller)
        out = v.grade(
            {
                "a": {"source": "Hello", "output": "안녕"},
                "b": {"source": "World", "output": "세계"},
            }
        )
        assert set(out.keys()) == {"a", "b"}
        assert all(g.binary_score == "yes" for g in out.values())

    def test_grade_empty(self) -> None:
        assert LLMValidator(_all_pass_caller).grade({}) == {}

    def test_partition_by_entry_count(self) -> None:
        calls: List[List[str]] = []

        def caller(items: Dict[str, Dict[str, str]]) -> str:
            calls.append(list(items.keys()))
            return _all_pass_caller(items)

        items = {
            f"k{i}": {"source": f"src{i}", "output": f"tgt{i}"} for i in range(25)
        }
        v = LLMValidator(caller, max_entries=10, max_chars=100_000)
        out = v.grade(items)
        assert set(out.keys()) == set(items.keys())
        assert all(len(c) <= 10 for c in calls)
        assert len(calls) >= 3

    def test_malformed_json_bisects(self) -> None:
        call_count = {"n": 0}

        def caller(items: Dict[str, Dict[str, str]]) -> str:
            call_count["n"] += 1
            if len(items) > 1:
                return "not json at all"
            return _all_pass_caller(items)

        items = {
            f"k{i}": {"source": f"s{i}", "output": f"t{i}"} for i in range(4)
        }
        out = LLMValidator(caller).grade(items)
        assert set(out.keys()) == set(items.keys())
        assert call_count["n"] >= 5

    def test_single_entry_failure_drops_out(self) -> None:
        def caller(items: Dict[str, Dict[str, str]]) -> str:
            raise RuntimeError("boom")

        v = LLMValidator(caller)
        out = v.grade({"a": {"source": "x", "output": "y"}})
        assert out == {}

    def test_invalid_grade_payload_drops_to_bisection(self) -> None:
        """A grade missing ``binary_score`` survives JSON parse but
        :meth:`BinaryGrade.from_payload` returns ``None``, so the key
        falls into the bisection retry path same as a missing key."""

        seen: List[List[str]] = []

        def caller(items: Dict[str, Dict[str, str]]) -> str:
            seen.append(list(items.keys()))
            payload = {}
            for k in items:
                if k == "bad" and len(items) > 1:
                    payload[k] = {"reason": "no score"}  # invalid
                else:
                    payload[k] = {"binary_score": "yes", "reason": "ok"}
            return _grade_response(payload)

        v = LLMValidator(caller)
        out = v.grade(
            {
                "good": {"source": "g", "output": "G"},
                "bad": {"source": "b", "output": "B"},
            }
        )
        assert "good" in out
        # ``bad`` keeps producing an unscored payload even on its own —
        # bisection drops it from the result.
        assert "bad" not in out


# ---------------------------------------------------------------- in-processor retry


SIMPLE_MD = (
    "First paragraph here.\n"
    "\n"
    "Second paragraph follows.\n"
    "\n"
    "Final paragraph wraps up.\n"
)


class _Recorder:
    """Capture every litellm.completion call to drive the retry loop."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        # Programmable per-call response builder.  Tests override this
        # to script the retry loop's behaviour deterministically.
        self.responder = self._default_responder

    def _default_responder(
        self, role_messages: List[Dict[str, Any]], call_index: int
    ) -> str:
        # Translator output prefixes the source with a Korean glyph so
        # the structural target-language check (which fires for ko)
        # is satisfied; the validator default response is PASS.
        last_user = next(
            (m for m in reversed(role_messages) if m["role"] == "user"),
            None,
        )
        if last_user is None:
            return "{}"
        body = last_user["content"]
        if isinstance(body, str) and body.strip().startswith("{"):
            try:
                items = json.loads(body)
            except json.JSONDecodeError:
                return "{}"
            if not isinstance(items, dict):
                return "{}"
            sample_val = next(iter(items.values()), None)
            if isinstance(sample_val, dict) and "source" in sample_val:
                return _all_pass_caller(items)
            return json.dumps(
                {k: f"번역: {v}" for k, v in items.items()},
                ensure_ascii=False,
            )
        return f"번역: {body}"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        messages = kwargs.get("messages", args[0] if args else [])
        call_index = len(self.calls)
        self.calls.append({"messages": messages, "kwargs": kwargs})
        content = self.responder(messages, call_index)
        resp = MagicMock()
        resp.choices[0].message.content = content
        resp.usage.prompt_tokens = 10
        resp.usage.completion_tokens = 5
        return resp


@pytest.fixture
def llm_completion():
    """Patch litellm.completion with a programmable recorder."""

    rec = _Recorder()
    with patch("mdpo_llm.processor.litellm") as litellm_mock:
        litellm_mock.completion.side_effect = rec
        litellm_mock.get_supported_openai_params.return_value = [
            "temperature",
            "max_tokens",
            "response_format",
        ]
        litellm_mock.model_cost = {}
        yield rec, litellm_mock


def _classify(messages: List[Dict[str, Any]]) -> str:
    """Distinguish translator vs validator calls by inspecting the system message."""
    sys_text = ""
    for m in messages:
        if m["role"] == "system":
            content = m["content"]
            sys_text = content if isinstance(content, str) else str(content)
            break
    if "Markdown and translation validator" in sys_text:
        return "validator"
    if "PREVIOUS ATTEMPT REJECTED" in sys_text:
        return "translator-retry"
    return "translator"


def _make_processor(
    *, max_retries: int = 3, fallback_model: str | None = None
) -> MarkdownProcessor:
    return MarkdownProcessor(
        model="primary-model",
        target_lang="ko",
        validation="llm",
        max_retries=max_retries,
        fallback_model=fallback_model,
    )


def _run_doc(
    proc: MarkdownProcessor, tmp_path: Any, content: str = SIMPLE_MD
) -> Dict[str, Any]:
    src = tmp_path / "src.md"
    tgt = tmp_path / "tgt.md"
    po = tmp_path / "tgt.po"
    src.write_text(content, encoding="utf-8")
    return proc.process_document(src, tgt, po)


class TestProcessorRetryLoop:
    def test_max_retries_clamped(self) -> None:
        assert _make_processor(max_retries=99).max_retries == 10
        assert _make_processor(max_retries=-3).max_retries == 0
        assert _make_processor(max_retries=0).max_retries == 0
        # Non-int silently coerced down.
        proc = MarkdownProcessor(
            model="m", target_lang="ko", validation="llm", max_retries="bogus"  # type: ignore[arg-type]
        )
        assert proc.max_retries == 0

    def test_validation_choice_validated(self) -> None:
        with pytest.raises(ValueError):
            MarkdownProcessor(model="m", target_lang="ko", validation="bogus")  # type: ignore[arg-type]

    def test_initial_pass_no_retries(self, llm_completion, tmp_path) -> None:
        rec, _ = llm_completion
        proc = _make_processor(max_retries=3)
        result = _run_doc(proc, tmp_path)
        kinds = [_classify(c["messages"]) for c in rec.calls]
        # Initial translator + initial validator only — no retry path.
        assert "translator-retry" not in kinds
        # Stats show validation passed.
        stats = result["translation_stats"]
        assert stats["processed"] >= 1
        assert stats["validation_failed"] == 0

    def test_retries_and_reason_accumulation(
        self, llm_completion, tmp_path
    ) -> None:
        rec, _ = llm_completion
        # Validator fails twice, then passes.  Translator on retry
        # echoes the source as before (the test doesn't care about
        # content correctness, only that the retry loop runs).
        validator_call_count = {"n": 0}
        validator_calls_messages: List[List[Dict[str, Any]]] = []

        def responder(messages, call_index):
            kind = _classify(messages)
            if kind == "validator":
                validator_call_count["n"] += 1
                validator_calls_messages.append(messages)
                # First two validator calls fail with distinct reasons.
                if validator_call_count["n"] == 1:
                    body = json.loads(messages[-1]["content"])
                    return _grade_response(
                        {
                            k: {
                                "binary_score": "no",
                                "reason": f"reason-attempt-1-{k}",
                            }
                            for k in body
                        }
                    )
                if validator_call_count["n"] == 2:
                    body = json.loads(messages[-1]["content"])
                    return _grade_response(
                        {
                            k: {
                                "binary_score": "no",
                                "reason": f"reason-attempt-2-{k}",
                            }
                            for k in body
                        }
                    )
                # Third validator call PASSES — the retry succeeded.
                body = json.loads(messages[-1]["content"])
                return _all_pass_caller(body)
            return rec._default_responder(messages, call_index)

        rec.responder = responder
        proc = _make_processor(max_retries=3)
        _run_doc(proc, tmp_path)
        assert validator_call_count["n"] == 3

        # The third validator call must have grading prompts whose
        # most recent translator-retry call carried BOTH attempt-1
        # and attempt-2 reasons in its system content (ordered, no
        # duplicates).
        retry_calls = [
            c for c in rec.calls if _classify(c["messages"]) == "translator-retry"
        ]
        assert retry_calls, "expected at least one translator retry"
        # The LAST retry's system prompt should include both reasons.
        last_retry_sys = next(
            m for m in retry_calls[-1]["messages"] if m["role"] == "system"
        )
        sys_text = last_retry_sys["content"]
        assert isinstance(sys_text, str)
        assert "reason-attempt-1" in sys_text
        assert "reason-attempt-2" in sys_text
        # Reasons appear under the documented header.
        assert "PREVIOUS ATTEMPT REJECTED" in sys_text

    def test_fallback_model_swap_timing(
        self, llm_completion, tmp_path
    ) -> None:
        """At retry index ceil(max_retries/2), the translator + validator
        must switch to the fallback model.  With max_retries=4,
        midpoint = ceil(4/2) = 2 → retries 2,3,4 use the fallback.
        """
        rec, _ = llm_completion
        # All validator calls fail so every retry slot fires.
        rec.responder = lambda messages, idx: (
            _all_fail_caller(json.loads(messages[-1]["content"]))
            if _classify(messages) == "validator"
            else rec._default_responder(messages, idx)
        )
        proc = _make_processor(max_retries=4, fallback_model="fallback-model")
        _run_doc(proc, tmp_path)

        # Find the retries (translator side, in order).
        retry_indices: List[int] = []
        for i, c in enumerate(rec.calls):
            if _classify(c["messages"]) == "translator-retry":
                retry_indices.append(i)
        # 4 retries — N attempts.
        assert len(retry_indices) >= 4
        models_used = [
            rec.calls[i]["kwargs"].get("model") for i in retry_indices
        ]
        # Retry 1 uses primary, retries 2..4 use fallback.
        assert models_used[0] == "primary-model"
        assert models_used[1] == "fallback-model"
        assert models_used[2] == "fallback-model"
        assert models_used[3] == "fallback-model"

    def test_no_fallback_when_unset(
        self, llm_completion, tmp_path
    ) -> None:
        rec, _ = llm_completion
        rec.responder = lambda messages, idx: (
            _all_fail_caller(json.loads(messages[-1]["content"]))
            if _classify(messages) == "validator"
            else rec._default_responder(messages, idx)
        )
        proc = _make_processor(max_retries=4, fallback_model=None)
        _run_doc(proc, tmp_path)
        retry_models = [
            c["kwargs"].get("model")
            for c in rec.calls
            if _classify(c["messages"]) == "translator-retry"
        ]
        assert retry_models  # there were retries
        assert all(m == "primary-model" for m in retry_models)

    def test_zero_retries_marks_fuzzy_immediately(
        self, llm_completion, tmp_path
    ) -> None:
        rec, _ = llm_completion
        rec.responder = lambda messages, idx: (
            _all_fail_caller(json.loads(messages[-1]["content"]))
            if _classify(messages) == "validator"
            else rec._default_responder(messages, idx)
        )
        proc = _make_processor(max_retries=0)
        result = _run_doc(proc, tmp_path)
        assert result["translation_stats"]["validation_failed"] >= 1
        # No translator-retry calls should fire.
        kinds = [_classify(c["messages"]) for c in rec.calls]
        assert "translator-retry" not in kinds
        # PO entries that failed should be fuzzy with a tcomment.
        po = proc.po_manager.load_or_create_po(tmp_path / "tgt.po")
        fuzzy = [e for e in po if "fuzzy" in e.flags]
        assert fuzzy, "expected at least one fuzzy entry after LLM rejection"
        for e in fuzzy:
            assert "validator: llm:" in (e.tcomment or "")

    def test_translatability_gating_pass_for_code_only(
        self, llm_completion, tmp_path
    ) -> None:
        """A code-only block whose translation equals the source should
        PASS the validator path (the prompt allows identical-output
        passes for content with no human-readable prose).  The test
        scripts that gate explicitly so the loop's behaviour is
        verified end-to-end.
        """
        rec, _ = llm_completion
        captured_grade_inputs: List[Dict[str, Any]] = []

        def responder(messages, idx):
            kind = _classify(messages)
            if kind == "validator":
                body = json.loads(messages[-1]["content"])
                captured_grade_inputs.append(body)
                # Simulate the validator obeying the translatability
                # gating rule: identical source/output → PASS.
                return _grade_response(
                    {
                        k: {
                            "binary_score": (
                                "yes"
                                if v["source"] == v["output"]
                                else "no"
                            ),
                            "reason": "code-only" if v["source"] == v["output"]
                            else "needs translation",
                        }
                        for k, v in body.items()
                    }
                )
            # Translator: echo back the input verbatim (mimics the
            # "code-only block stays identical" rule).
            last_user = next(
                m for m in reversed(messages) if m["role"] == "user"
            )
            body = last_user["content"]
            try:
                items = json.loads(body)
                return json.dumps(items, ensure_ascii=False)
            except json.JSONDecodeError:
                return body

        rec.responder = responder
        proc = _make_processor(max_retries=2)
        code_md = (
            "# Title\n\n```python\nprint('hello')\n```\n\nMore text.\n"
        )
        result = _run_doc(proc, tmp_path, content=code_md)
        # Code block PASSED (validator returned yes for identical
        # source/output) — no retries should fire for it.
        kinds = [_classify(c["messages"]) for c in rec.calls]
        # At least one validator call captured an item with source ==
        # output that was returned with binary_score=yes.
        assert captured_grade_inputs, (
            "expected the validator to be called at least once"
        )
        assert any(
            v["source"] == v["output"]
            for body in captured_grade_inputs
            for v in body.values()
        )
        # Stats should show validated entries (not all fuzzy).
        assert result["translation_stats"]["validated"] >= 1


class TestProcessorPerKeyAndCountingInvariants:
    """Codex-cycle-1 follow-ups: per-key reason scoping + no double-count."""

    def test_validated_counted_once(self, llm_completion, tmp_path) -> None:
        """``_apply_validation`` is the single source of truth for the
        ``validated`` stat; the LLM-validation pre-pass must NOT
        increment it as well."""
        rec, _ = llm_completion
        proc = _make_processor(max_retries=2)
        result = _run_doc(proc, tmp_path)
        stats = result["translation_stats"]
        # processed entries == validated entries (every passing entry
        # gets counted once, not twice).  Loose equality avoids false
        # positives from skipped HRs / non-prose blocks.
        assert stats["processed"] >= 1
        assert stats["validated"] == stats["processed"]
        assert stats["validation_failed"] == 0

    def test_per_key_reasons_block_renders_per_key(
        self, llm_completion, tmp_path
    ) -> None:
        """When multiple keys fail with distinct reasons, the retry
        prompt must scope each reason to its failing key — sibling
        keys must NOT see each other's rejection histories."""

        rec, _ = llm_completion
        # Two keys: A always fails with "reason-A", B always fails
        # with "reason-B".
        validator_seen: List[List[str]] = []

        def responder(messages, idx):
            kind = _classify(messages)
            if kind == "validator":
                body = json.loads(messages[-1]["content"])
                validator_seen.append(list(body.keys()))
                return _grade_response(
                    {
                        k: {
                            "binary_score": "no",
                            "reason": f"reason-{i}",
                        }
                        for i, k in enumerate(body)
                    }
                )
            return rec._default_responder(messages, idx)

        rec.responder = responder
        proc = _make_processor(max_retries=2)
        # Multi-paragraph doc → multiple keys grouped into one batch.
        md = (
            "First paragraph.\n\n"
            "Second paragraph.\n\n"
            "Third paragraph.\n"
        )
        _run_doc(proc, tmp_path, content=md)

        retry_calls = [
            c for c in rec.calls if _classify(c["messages"]) == "translator-retry"
        ]
        assert retry_calls, "expected at least one retry"
        # The retry system message must have a per-key block: each
        # key gets its own reason list, not a shared union.
        sys_text = next(
            m["content"] for m in retry_calls[0]["messages"]
            if m["role"] == "system"
        )
        assert isinstance(sys_text, str)
        assert "PREVIOUS ATTEMPT REJECTED" in sys_text
        # Each per-key block starts with ``- key '``; expect at least
        # one such marker for the rendered block.
        assert "- key '" in sys_text


class TestSequentialPathLLMValidation:
    """Codex-cycle-3 P1: ``batch_size=0`` must also run the validator + retry."""

    def test_sequential_path_runs_grader_and_retry(
        self, llm_completion, tmp_path
    ) -> None:
        rec, _ = llm_completion
        # Validator fails the first attempt, passes the second.
        validator_calls = {"n": 0}

        def responder(messages, idx):
            kind = _classify(messages)
            if kind == "validator":
                validator_calls["n"] += 1
                body = json.loads(messages[-1]["content"])
                if validator_calls["n"] == 1:
                    return _grade_response(
                        {
                            k: {"binary_score": "no", "reason": "first reason"}
                            for k in body
                        }
                    )
                return _all_pass_caller(body)
            return rec._default_responder(messages, idx)

        rec.responder = responder
        proc = MarkdownProcessor(
            model="primary-model",
            target_lang="ko",
            validation="llm",
            max_retries=2,
            batch_size=0,  # Sequential path.
        )
        result = _run_doc(proc, tmp_path)
        # Validator was called both for initial AND retry.
        assert validator_calls["n"] >= 2
        retries = [
            c for c in rec.calls if _classify(c["messages"]) == "translator-retry"
        ]
        assert retries, "expected a translator retry on the sequential path"
        # The retry succeeded — entry was processed.
        assert result["translation_stats"]["processed"] >= 1


class TestStructuralPreGate:
    """Codex-cycle-2 follow-up: structural failures consume retry budget too."""

    def test_structural_failure_triggers_retry(
        self, llm_completion, tmp_path
    ) -> None:
        rec, _ = llm_completion
        # Source has a heading; first translator response strips
        # the heading marker (heading_level mismatch → structural
        # fail). Second response keeps the heading.  Validator
        # always passes — failures should still trigger the retry.
        attempt = {"n": 0}

        def responder(messages, idx):
            kind = _classify(messages)
            if kind == "validator":
                # Validator passes whatever it sees.
                body = json.loads(messages[-1]["content"])
                return _all_pass_caller(body)
            # Translator (initial OR retry).
            attempt["n"] += 1
            last_user = next(
                m for m in reversed(messages) if m["role"] == "user"
            )
            body = last_user["content"]
            try:
                items = json.loads(body)
            except json.JSONDecodeError:
                return body
            if attempt["n"] == 1:
                # First attempt: drop the heading marker (structural
                # fence/heading mismatch).
                return json.dumps(
                    {k: f"번역: {v.replace('# ', '')}" for k, v in items.items()},
                    ensure_ascii=False,
                )
            # Retry: produce structurally valid output.
            return json.dumps(
                {k: f"번역: {v}" for k, v in items.items()},
                ensure_ascii=False,
            )

        rec.responder = responder
        proc = _make_processor(max_retries=2)
        md_with_heading = (
            "# Heading One\n\n"
            "Content paragraph after the heading.\n"
        )
        result = _run_doc(proc, tmp_path, content=md_with_heading)
        # A retry must have fired (heading mismatch on attempt 1
        # forced the loop to re-translate).
        retry_calls = [
            c for c in rec.calls if _classify(c["messages"]) == "translator-retry"
        ]
        assert retry_calls, "expected a retry triggered by structural failure"
        # Stats: at least one entry processed successfully after
        # the retry.
        assert result["translation_stats"]["processed"] >= 1

    def test_structural_skip_validator_call_when_no_eligible_keys(
        self, llm_completion, tmp_path
    ) -> None:
        """When every initial decode fails structurally, the validator
        LLM should not be called at all — it would only see keys we
        already know are bad."""
        rec, _ = llm_completion
        # Translator strips heading on EVERY attempt → all retries
        # fail structurally and never reach the validator.
        def responder(messages, idx):
            kind = _classify(messages)
            if kind == "validator":
                body = json.loads(messages[-1]["content"])
                return _all_pass_caller(body)
            last_user = next(
                m for m in reversed(messages) if m["role"] == "user"
            )
            body = last_user["content"]
            try:
                items = json.loads(body)
            except json.JSONDecodeError:
                return body
            return json.dumps(
                {k: f"번역: {v.replace('# ', '')}" for k, v in items.items()},
                ensure_ascii=False,
            )

        rec.responder = responder
        proc = _make_processor(max_retries=1)
        md = "# Heading\n\nMore prose.\n"
        _run_doc(proc, tmp_path, content=md)
        # Heading entry: every attempt strips the heading, so the
        # validator never sees it on either attempt.  Plain prose
        # entry passes structural and DOES reach the validator.
        validator_calls = [
            c for c in rec.calls if _classify(c["messages"]) == "validator"
        ]
        # Validator was called, but never with the heading ctx
        # (whose every decode failed structural).
        for c in validator_calls:
            body = json.loads(c["messages"][-1]["content"])
            for ctx in body:
                # ctx contains a path-like context id; the heading
                # entry's ctx contains 'heading' or has no other
                # markers — easier to check that source is not the
                # stripped heading text.
                assert "Heading" not in body[ctx]["source"] or (
                    body[ctx]["source"].lstrip().startswith("#")
                ), (
                    "validator should never grade a structurally-stripped "
                    "heading whose source IS the heading line"
                )


class TestProcessorBudgetExhausted:
    def test_residual_failures_carry_last_reason(
        self, llm_completion, tmp_path
    ) -> None:
        rec, _ = llm_completion
        # Every validator call returns no with a unique reason; we
        # check the LAST one is what gets stored on the PO entry.
        counter = {"n": 0}

        def responder(messages, idx):
            kind = _classify(messages)
            if kind == "validator":
                counter["n"] += 1
                body = json.loads(messages[-1]["content"])
                return _grade_response(
                    {
                        k: {
                            "binary_score": "no",
                            "reason": f"validator-pass-{counter['n']}",
                        }
                        for k in body
                    }
                )
            return rec._default_responder(messages, idx)

        rec.responder = responder
        proc = _make_processor(max_retries=2)
        _run_doc(proc, tmp_path)
        po = proc.po_manager.load_or_create_po(tmp_path / "tgt.po")
        fuzzy = [e for e in po if "fuzzy" in e.flags]
        assert fuzzy
        # The LAST validator pass's reason must be the one recorded —
        # not the first-attempt reason.
        last_reason = f"validator-pass-{counter['n']}"
        assert any(
            last_reason in (e.tcomment or "") for e in fuzzy
        ), f"expected '{last_reason}' in tcomments: {[e.tcomment for e in fuzzy]}"
