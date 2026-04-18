"""Tests for BatchTranslator."""

import json

import pytest

from mdpo_llm.batch import BatchTranslator, MultiTargetBatchTranslator


def _ok_caller(items):
    """Simulated JSON-mode reply: round-trip with a prefix."""
    return json.dumps({k: f"T({v})" for k, v in items.items()})


def test_happy_path_3_items():
    items = {"a": "alpha", "b": "bravo", "c": "charlie"}
    out = BatchTranslator(_ok_caller).translate(items)
    assert out == {"a": "T(alpha)", "b": "T(bravo)", "c": "T(charlie)"}


def test_empty_input():
    assert BatchTranslator(_ok_caller).translate({}) == {}


def test_partition_splits_many_items():
    calls = []

    def caller(items):
        calls.append(list(items.keys()))
        return _ok_caller(items)

    items = {f"k{i}": f"v{i}" for i in range(50)}
    out = BatchTranslator(caller, max_entries=10, max_chars=10_000).translate(items)
    assert set(out.keys()) == set(items.keys())
    # At least 5 chunks (50 / 10).
    assert len(calls) >= 5
    # No chunk exceeds max_entries.
    assert all(len(c) <= 10 for c in calls)


def test_partition_char_cap():
    """Char cap forces smaller chunks than entry cap would."""

    def caller(items):
        return _ok_caller(items)

    items = {f"k{i}": "x" * 600 for i in range(10)}  # 6000 chars total
    t = BatchTranslator(caller, max_entries=40, max_chars=1000)
    out = t.translate(items)
    assert len(out) == 10


def test_malformed_json_bisects():
    call_count = {"n": 0}

    def caller(items):
        call_count["n"] += 1
        if len(items) > 1:
            return "not valid json at all"
        return _ok_caller(items)

    items = {"a": "x", "b": "y", "c": "z", "d": "w"}
    out = BatchTranslator(caller).translate(items)
    assert set(out.keys()) == set(items.keys())
    assert call_count["n"] >= 5  # one failed + bisection


def test_missing_keys_retry_subset():
    def caller(items):
        # Drop one key every time it appears.
        translated = {k: f"T({v})" for k, v in items.items() if k != "b"}
        return json.dumps(translated)

    items = {"a": "x", "b": "y", "c": "z"}
    out = BatchTranslator(caller).translate(items)
    # "b" will keep missing until single-entry bisection returns {}.
    assert "a" in out and "c" in out
    assert "b" not in out  # Caller falls back.


def test_single_entry_failure_returns_empty():
    def caller(items):
        raise RuntimeError("boom")

    out = BatchTranslator(caller).translate({"a": "x"})
    assert out == {}


def test_bisection_terminates_on_persistent_single_failure():
    calls = []

    def caller(items):
        calls.append(list(items.keys()))
        raise RuntimeError("always fails")

    out = BatchTranslator(caller).translate({f"k{i}": f"v{i}" for i in range(4)})
    assert out == {}
    # With 4 items and binary bisection to depth 1, total calls should be bounded.
    assert len(calls) < 20


def test_parse_response_direct_json():
    raw = '{"a": "T(x)"}'
    assert BatchTranslator._parse_response(raw) == {"a": "T(x)"}


def test_parse_response_fenced_json():
    raw = "```json\n{\"a\": \"T(x)\"}\n```"
    assert BatchTranslator._parse_response(raw) == {"a": "T(x)"}


def test_parse_response_prose_wrapped():
    raw = 'Here is the translation:\n{"a": "T(x)"}\nThat is all.'
    assert BatchTranslator._parse_response(raw) == {"a": "T(x)"}


def test_parse_response_rejects_array():
    raw = '[1, 2, 3]'
    assert BatchTranslator._parse_response(raw) is None


def test_parse_response_none_for_empty():
    assert BatchTranslator._parse_response("") is None
    assert BatchTranslator._parse_response(None) is None


def _multi_ok_caller(langs):
    """Return a caller that fans out {k: {lang: f'{lang}:{v}'}} for each lang."""

    def _caller(items):
        out = {k: {lang: f"{lang}:{v}" for lang in langs} for k, v in items.items()}
        return json.dumps(out)

    return _caller


class TestMultiTargetBatchTranslator:
    def test_happy_path_two_langs(self):
        items = {"a": "alpha", "b": "bravo"}
        t = MultiTargetBatchTranslator(_multi_ok_caller(["ko", "ja"]), ["ko", "ja"])
        out = t.translate(items)
        assert out == {
            "a": {"ko": "ko:alpha", "ja": "ja:alpha"},
            "b": {"ko": "ko:bravo", "ja": "ja:bravo"},
        }

    def test_empty_input(self):
        t = MultiTargetBatchTranslator(_multi_ok_caller(["ko"]), ["ko"])
        assert t.translate({}) == {}

    def test_rejects_empty_target_langs(self):
        with pytest.raises(ValueError):
            MultiTargetBatchTranslator(_multi_ok_caller(["ko"]), [])

    def test_rejects_non_string_target_langs(self):
        with pytest.raises(ValueError):
            MultiTargetBatchTranslator(_multi_ok_caller(["ko"]), [123])  # type: ignore[list-item]
        with pytest.raises(ValueError):
            MultiTargetBatchTranslator(_multi_ok_caller(["ko"]), [""])

    def test_dedupes_target_langs_preserves_order(self):
        t = MultiTargetBatchTranslator(
            _multi_ok_caller(["ko", "ja"]), ["ko", "ja", "ko"]
        )
        assert t.target_langs == ("ko", "ja")

    def test_partial_lang_coverage_kept_as_is(self):
        """Partial per-lang coverage is preserved for the caller to backfill.

        Re-billing the fully-delivered langs just to retry a missing one
        would waste tokens, so the translator keeps what the model
        actually produced and lets the caller run a targeted per-lang
        fallback for the missing lang only.
        """

        def _caller(items):
            out = {}
            for k, v in items.items():
                # Omit "ja" for key "b" — a partial delivery.
                if k == "b":
                    out[k] = {"ko": f"ko:{v}"}
                else:
                    out[k] = {"ko": f"ko:{v}", "ja": f"ja:{v}"}
            return json.dumps(out)

        t = MultiTargetBatchTranslator(_caller, ["ko", "ja"])
        out = t.translate({"a": "x", "b": "y", "c": "z"})
        assert out["a"] == {"ko": "ko:x", "ja": "ja:x"}
        assert out["c"] == {"ko": "ko:z", "ja": "ja:z"}
        # "b" is preserved with partial coverage — only "ko" came back.
        assert out["b"] == {"ko": "ko:y"}

    def test_value_not_a_dict_treated_as_missing(self):
        """A non-dict value for a key triggers the bisection fallback."""

        def _caller(items):
            out = {}
            for k, v in items.items():
                if k == "bad" and len(items) > 1:
                    out[k] = "plain string"
                else:
                    out[k] = {"ko": f"ko:{v}", "ja": f"ja:{v}"}
            return json.dumps(out)

        t = MultiTargetBatchTranslator(_caller, ["ko", "ja"])
        out = t.translate({"good": "x", "bad": "y"})
        assert out["good"] == {"ko": "ko:x", "ja": "ja:x"}
        # After bisection "bad" still returns a plain string single-entry,
        # which is non-dict, so it drops out of the result.
        assert "bad" not in out

    def test_malformed_json_bisects_down_to_singletons(self):
        call_count = {"n": 0}

        def _caller(items):
            call_count["n"] += 1
            if len(items) > 1:
                return "not valid json"
            return json.dumps(
                {k: {"ko": f"ko:{v}", "ja": f"ja:{v}"} for k, v in items.items()}
            )

        t = MultiTargetBatchTranslator(_caller, ["ko", "ja"])
        items = {"a": "x", "b": "y", "c": "z", "d": "w"}
        out = t.translate(items)
        assert set(out.keys()) == set(items.keys())
        # One failed call + bisection tree => several calls.
        assert call_count["n"] >= 5

    def test_single_entry_failure_returns_empty(self):
        def _caller(items):
            raise RuntimeError("boom")

        t = MultiTargetBatchTranslator(_caller, ["ko", "ja"])
        assert t.translate({"a": "x"}) == {}

    def test_input_partitioned_by_entry_count(self):
        calls = []

        def _caller(items):
            calls.append(list(items.keys()))
            return json.dumps(
                {k: {"ko": f"ko:{v}", "ja": f"ja:{v}"} for k, v in items.items()}
            )

        items = {f"k{i}": f"v{i}" for i in range(25)}
        t = MultiTargetBatchTranslator(
            _caller, ["ko", "ja"], max_entries=10, max_chars=10_000
        )
        out = t.translate(items)
        assert set(out.keys()) == set(items.keys())
        assert all(len(c) <= 10 for c in calls)
        assert len(calls) >= 3

    def test_extra_langs_in_response_are_dropped(self):
        """Target locales = ko; extra locales in the response are ignored."""

        def _caller(items):
            return json.dumps(
                {k: {"ko": f"ko:{v}", "ja": f"ja:{v}"} for k, v in items.items()}
            )

        t = MultiTargetBatchTranslator(_caller, ["ko"])
        out = t.translate({"a": "x"})
        assert out == {"a": {"ko": "ko:x"}}


def test_non_string_value_treated_as_missing():
    def caller(items):
        # Always drop "b" by returning a non-string for it.
        out = {}
        for k, v in items.items():
            out[k] = f"T({v})" if k != "b" else 42
        return json.dumps(out)

    result = BatchTranslator(caller).translate({"a": "x", "b": "y", "c": "z"})
    # "a" and "c" come back; "b" stays missing so the caller can fall back.
    assert result["a"] == "T(x)"
    assert result["c"] == "T(z)"
    assert "b" not in result
