"""Tests for typed result dataclasses with dict-like shim."""

from mdpo_llm.results import BatchStats, Coverage, DirectoryResult, ProcessResult


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
