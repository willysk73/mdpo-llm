"""Tests for BatchTranslator."""

import json

import pytest

from mdpo_llm.batch import BatchTranslator


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
