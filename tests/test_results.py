"""Tests for typed result dataclasses with dict-like shim."""

from mdpo_llm.results import BatchStats, Coverage, DirectoryResult, ProcessResult, Receipt


def _make_coverage():
    return Coverage(
        total_blocks=3,
        translatable_blocks=2,
        translated_blocks=1,
        fuzzy_blocks=0,
        untranslated_blocks=1,
        coverage_percentage=50.0,
        by_type={"para": {"total": 2, "translated": 1}},
    )


def test_coverage_dict_style_access():
    cov = _make_coverage()
    assert cov["total_blocks"] == 3
    assert cov.get("missing_key", "fallback") == "fallback"
    assert "total_blocks" in cov
    assert "not_a_field" not in cov


def test_coverage_attribute_access():
    cov = _make_coverage()
    assert cov.total_blocks == 3
    assert cov.translatable_blocks == 2


def test_batch_stats_defaults():
    stats = BatchStats()
    assert stats["processed"] == 0
    assert stats.failed == 0


def test_process_result_nested_dict_access():
    cov = _make_coverage()
    stats = BatchStats(processed=2, failed=0, skipped=1)
    r = ProcessResult(
        source_path="/tmp/s.md",
        target_path="/tmp/t.md",
        po_path="/tmp/m.po",
        blocks_count=3,
        coverage=cov,
        translation_stats=stats,
    )
    # Existing consumers pattern.
    assert r["translation_stats"]["processed"] == 2
    assert r.get("translation_stats", {}).get("skipped", 0) == 1
    assert r["coverage"]["total_blocks"] == 3


def test_result_to_dict_round_trips():
    r = ProcessResult(
        source_path="/tmp/s.md",
        target_path="/tmp/t.md",
        po_path="/tmp/m.po",
        blocks_count=1,
        coverage=_make_coverage(),
        translation_stats=BatchStats(processed=1),
    )
    d = r.to_dict()
    assert isinstance(d, dict)
    assert d["blocks_count"] == 1
    assert d["coverage"]["total_blocks"] == 3


def test_result_is_a_mapping():
    from collections.abc import Mapping

    r = ProcessResult(
        source_path="/tmp/s.md",
        target_path="/tmp/t.md",
        po_path="/tmp/m.po",
        blocks_count=1,
        coverage=_make_coverage(),
        translation_stats=BatchStats(processed=1),
    )
    assert isinstance(r, Mapping)
    # dict(result) should copy the field/value pairs.
    copied = dict(r)
    assert copied["blocks_count"] == 1
    assert "translation_stats" in copied


def test_result_is_json_serializable():
    """Subclassing dict keeps ``json.dumps(result)`` working for callers
    that previously serialized the plain-dict return value."""
    import json

    r = ProcessResult(
        source_path="/tmp/s.md",
        target_path="/tmp/t.md",
        po_path="/tmp/m.po",
        blocks_count=1,
        coverage=_make_coverage(),
        translation_stats=BatchStats(processed=1),
    )
    encoded = json.dumps(r.to_dict())
    assert "source_path" in encoded
    assert "blocks_count" in encoded


def test_directory_result_iter_and_len():
    d = DirectoryResult(
        source_dir="/a",
        target_dir="/b",
        po_dir=None,
        files_processed=1,
        files_failed=0,
        files_skipped=0,
        results=[],
    )
    keys = list(d)
    assert "source_dir" in keys
    assert "results" in keys
    assert len(d) == len(keys)


def _make_receipt(**overrides):
    base = dict(
        model="gpt-4o",
        target_lang="ko",
        source_path="/tmp/s.md",
        target_path="/tmp/t.md",
        po_path="/tmp/m.po",
        input_tokens=1000,
        output_tokens=500,
        total_tokens=1500,
        api_calls=3,
        duration_seconds=1.25,
        input_cost_per_1m_usd=2.5,
        output_cost_per_1m_usd=10.0,
        input_cost_usd=0.0025,
        output_cost_usd=0.005,
        total_cost_usd=0.0075,
    )
    base.update(overrides)
    return Receipt(**base)


def test_receipt_dict_and_attribute_access():
    r = _make_receipt()
    assert r["input_tokens"] == 1000
    assert r.output_tokens == 500
    assert r.total_tokens == 1500
    assert r["total_cost_usd"] == 0.0075


def test_receipt_render_includes_core_fields():
    r = _make_receipt()
    out = r.render()
    assert "Translation receipt" in out
    assert "gpt-4o" in out
    assert "ko" in out
    assert "/tmp/s.md" in out
    assert "1,000" in out  # comma-formatted input tokens
    assert "$2.50 / 1M tokens" in out
    assert "$0.007500" in out  # total cost with 6dp
    assert "1.25s" in out


def test_receipt_render_unpriced_model_shows_em_dash():
    r = _make_receipt(
        input_cost_per_1m_usd=None,
        output_cost_per_1m_usd=None,
        input_cost_usd=None,
        output_cost_usd=None,
        total_cost_usd=None,
    )
    out = r.render()
    # Every cost row degrades to the em-dash fallback.
    assert out.count("—") >= 5


def test_receipt_render_omits_missing_paths():
    r = _make_receipt(source_path=None, target_path=None, po_path=None)
    out = r.render()
    assert "Source:" not in out
    assert "Target:" not in out
    assert "PO file:" not in out


def test_receipt_to_dict_nested_in_process_result():
    cov = _make_coverage()
    r = ProcessResult(
        source_path="/tmp/s.md",
        target_path="/tmp/t.md",
        po_path="/tmp/m.po",
        blocks_count=1,
        coverage=cov,
        translation_stats=BatchStats(processed=1),
        receipt=_make_receipt(),
    )
    d = r.to_dict()
    assert d["receipt"]["model"] == "gpt-4o"
    assert d["receipt"]["total_tokens"] == 1500


def test_process_result_without_receipt_is_still_valid():
    """Legacy callers constructing ProcessResult without a receipt keep working."""
    r = ProcessResult(
        source_path="/tmp/s.md",
        target_path="/tmp/t.md",
        po_path="/tmp/m.po",
        blocks_count=1,
        coverage=_make_coverage(),
        translation_stats=BatchStats(processed=1),
    )
    assert r["receipt"] is None
    assert r.receipt is None


def test_directory_result_receipt_optional():
    d = DirectoryResult(
        source_dir="/a",
        target_dir="/b",
        po_dir=None,
        files_processed=0,
        files_failed=0,
        files_skipped=0,
        results=[],
        receipt=_make_receipt(source_path="/a", target_path="/b", po_path=None),
    )
    assert d.receipt.model == "gpt-4o"
    assert d["receipt"]["input_tokens"] == 1000
