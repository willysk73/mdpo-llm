"""Per-directory glossary cascade tests (T-11).

Covers the resolution logic exposed on :class:`MarkdownProcessor`:

    * parent-chain walk under ``source_dir``
    * ``__remove__`` sentinel unsets inherited terms
    * child level wins conflicts with parents
    * missing child level inherits from parent
    * constructor-time ``--glossary`` override sits on top of the chain
    * directory cache avoids re-reading parent glossaries for siblings
    * empty chain preserves pre-T-11 behaviour (no glossary active)

Tests exercise the resolution helpers directly instead of the full
``process_directory`` pipeline — the helpers are the public contract
for cascade semantics and keep tests fast / deterministic (no
``litellm`` mocking required).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from mdpo_llm.processor import MarkdownProcessor


def _write_glossary(directory: Path, payload: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "glossary.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _make_processor(target_lang: str = "ko") -> MarkdownProcessor:
    """Build a processor without touching the network.

    The cascade resolution methods are pure / offline — no need for a
    real LLM, mock litellm, or progress hook.  ``target_lang`` matters
    because :meth:`_effective_glossary_for_file` collapses per-locale
    dicts into a single ``{term: str | None}`` mapping for the
    processor's target locale.
    """
    return MarkdownProcessor(
        model="test-model",
        target_lang=target_lang,
        glossary_mode="placeholder",
    )


class TestParentChainWalk:
    def test_nearest_parent_merges_with_ancestor(self, tmp_path):
        source_dir = tmp_path / "docs"
        child_dir = source_dir / "api"
        _write_glossary(source_dir, {"GitHub": "GitHub"})
        _write_glossary(child_dir, {"API": "API"})
        file_path = child_dir / "file.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("# title\n", encoding="utf-8")

        proc = _make_processor()
        effective, chain = proc._effective_glossary_for_file(file_path, source_dir)

        assert effective == {"GitHub": "GitHub", "API": "API"}
        # Parent → child ordering in the debug chain.
        assert len(chain) == 2
        assert chain[0].name == "glossary.json"
        assert chain[0].parent.name == "docs"
        assert chain[1].parent.name == "api"

    def test_deep_tree_walks_all_intermediate_levels(self, tmp_path):
        source_dir = tmp_path / "docs"
        mid_dir = source_dir / "section"
        leaf_dir = mid_dir / "api"
        _write_glossary(source_dir, {"RootTerm": "rootVal"})
        _write_glossary(mid_dir, {"MidTerm": "midVal"})
        _write_glossary(leaf_dir, {"LeafTerm": "leafVal"})
        file_path = leaf_dir / "page.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("body", encoding="utf-8")

        proc = _make_processor()
        effective, chain = proc._effective_glossary_for_file(file_path, source_dir)

        assert effective == {
            "RootTerm": "rootVal",
            "MidTerm": "midVal",
            "LeafTerm": "leafVal",
        }
        # Every level above the file was loaded.
        assert len(chain) == 3


class TestRemoveSentinel:
    def test_remove_sentinel_unsets_inherited_term(self, tmp_path):
        source_dir = tmp_path / "docs"
        child_dir = source_dir / "marketing"
        _write_glossary(source_dir, {"API": "API", "Kubernetes": None})
        _write_glossary(child_dir, {"API": "__remove__"})
        file_path = child_dir / "page.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("x", encoding="utf-8")

        proc = _make_processor()
        effective, _ = proc._effective_glossary_for_file(file_path, source_dir)

        # ``API`` was inherited from the parent then unset; ``Kubernetes``
        # (do-not-translate) still flows through.
        assert effective == {"Kubernetes": None}

    def test_remove_sentinel_on_absent_parent_term_is_noop(self, tmp_path):
        source_dir = tmp_path / "docs"
        child_dir = source_dir / "child"
        _write_glossary(source_dir, {"GitHub": None})
        _write_glossary(child_dir, {"NonExistent": "__remove__"})
        file_path = child_dir / "page.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("x", encoding="utf-8")

        proc = _make_processor()
        effective, _ = proc._effective_glossary_for_file(file_path, source_dir)

        assert effective == {"GitHub": None}


class TestChildOverridesParent:
    def test_child_value_wins_when_both_define_term(self, tmp_path):
        source_dir = tmp_path / "docs"
        child_dir = source_dir / "child"
        _write_glossary(source_dir, {"API": "API"})
        _write_glossary(child_dir, {"API": "에이피아이"})
        file_path = child_dir / "page.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("x", encoding="utf-8")

        proc = _make_processor()
        effective, _ = proc._effective_glossary_for_file(file_path, source_dir)

        assert effective == {"API": "에이피아이"}

    def test_child_can_switch_mapping_to_do_not_translate(self, tmp_path):
        source_dir = tmp_path / "docs"
        child_dir = source_dir / "child"
        _write_glossary(source_dir, {"pull request": "풀 리퀘스트"})
        _write_glossary(child_dir, {"pull request": None})
        file_path = child_dir / "page.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("x", encoding="utf-8")

        proc = _make_processor()
        effective, _ = proc._effective_glossary_for_file(file_path, source_dir)

        assert effective == {"pull request": None}


class TestInheritance:
    def test_term_absent_from_child_inherits_from_parent(self, tmp_path):
        source_dir = tmp_path / "docs"
        child_dir = source_dir / "child"
        _write_glossary(source_dir, {"GitHub": None, "API": "API"})
        # Child overrides one term but not the other.
        _write_glossary(child_dir, {"GitHub": "깃허브"})
        file_path = child_dir / "page.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("x", encoding="utf-8")

        proc = _make_processor()
        effective, _ = proc._effective_glossary_for_file(file_path, source_dir)

        assert effective == {"GitHub": "깃허브", "API": "API"}

    def test_child_dir_without_glossary_inherits_parent(self, tmp_path):
        source_dir = tmp_path / "docs"
        child_dir = source_dir / "child"
        child_dir.mkdir(parents=True, exist_ok=True)
        _write_glossary(source_dir, {"API": "API"})
        file_path = child_dir / "page.md"
        file_path.write_text("x", encoding="utf-8")

        proc = _make_processor()
        effective, _ = proc._effective_glossary_for_file(file_path, source_dir)

        assert effective == {"API": "API"}


class TestCliOverride:
    def test_constructor_glossary_is_topmost(self, tmp_path):
        # Simulate ``--glossary PATH`` by passing glossary_path to the
        # constructor.  The cascade discovers in-tree levels, then the
        # constructor-time override is applied as the topmost layer.
        source_dir = tmp_path / "docs"
        _write_glossary(source_dir, {"API": "tree-api", "Shared": "tree-shared"})
        file_path = source_dir / "page.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("x", encoding="utf-8")

        cli_gloss = tmp_path / "cli_glossary.json"
        cli_gloss.write_text(
            json.dumps({"API": "cli-api", "Extra": "cli-extra"}),
            encoding="utf-8",
        )

        proc = MarkdownProcessor(
            model="test-model",
            target_lang="ko",
            glossary_path=cli_gloss,
            glossary_mode="placeholder",
        )
        effective, chain = proc._effective_glossary_for_file(
            file_path, source_dir
        )

        # CLI override supersedes the tree layer for shared term ``API``
        # while the tree-only term (``Shared``) survives.
        assert effective == {
            "API": "cli-api",
            "Shared": "tree-shared",
            "Extra": "cli-extra",
        }
        # Debug chain records both the tree layer and the synthetic
        # constructor-override marker in parent → child order.
        assert chain[-1] == Path("<constructor-override>")

    def test_cli_override_with_remove_unsets_tree_term(self, tmp_path):
        source_dir = tmp_path / "docs"
        _write_glossary(source_dir, {"API": "tree-api"})
        file_path = source_dir / "page.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("x", encoding="utf-8")

        cli_gloss = tmp_path / "cli_glossary.json"
        cli_gloss.write_text(
            json.dumps({"API": "__remove__"}), encoding="utf-8"
        )

        proc = MarkdownProcessor(
            model="test-model",
            target_lang="ko",
            glossary_path=cli_gloss,
            glossary_mode="placeholder",
        )
        effective, _ = proc._effective_glossary_for_file(file_path, source_dir)

        assert effective is None


class TestDirectoryCache:
    def test_sibling_files_reuse_cached_parent_chain(self, tmp_path):
        source_dir = tmp_path / "docs"
        child_dir = source_dir / "api"
        _write_glossary(source_dir, {"RootTerm": "root"})
        _write_glossary(child_dir, {"ChildTerm": "child"})
        first = child_dir / "a.md"
        second = child_dir / "b.md"
        first.parent.mkdir(parents=True, exist_ok=True)
        first.write_text("x", encoding="utf-8")
        second.write_text("y", encoding="utf-8")

        proc = _make_processor()
        # First call populates the caches.
        proc._effective_glossary_for_file(first, source_dir)

        original_loader = proc._load_glossary_json_raw
        with patch.object(
            proc,
            "_load_glossary_json_raw",
            wraps=original_loader,
        ) as spy:
            proc._effective_glossary_for_file(second, source_dir)

        # Sibling resolution MUST NOT re-read any glossary.json: the
        # directory chain cache already has child_dir (and its parent
        # chain through source_dir) merged from the first call.
        loaded_paths = [call.args[0] for call in spy.call_args_list]
        # cwd/glossary.json may be probed each call (depends on cwd)
        # — tolerate only that particular lookup.  Anything inside the
        # source tree means the dir cache missed.
        for p in loaded_paths:
            assert source_dir.resolve() not in p.resolve().parents, (
                f"unexpected re-read of in-tree glossary: {p}"
            )

    def test_cache_does_not_leak_across_source_dir_roots(self, tmp_path):
        """Reusing a processor across different ``source_dir`` roots.

        The cache key MUST include ``source_dir`` so a directory first
        merged as a subtree (inheriting the outer root's
        ``glossary.json``) does NOT reuse that merge when a later
        ``process_directory`` call treats it as the NEW root with no
        ancestor glossary to inherit.
        """
        outer_root = tmp_path / "outer"
        inner_root = outer_root / "inner"
        _write_glossary(outer_root, {"OuterOnly": "outer-val"})
        inner_root.mkdir(parents=True, exist_ok=True)
        file_path = inner_root / "page.md"
        file_path.write_text("x", encoding="utf-8")

        proc = _make_processor()
        # First call: outer_root is the tree root, inner inherits.
        effective_a, _ = proc._effective_glossary_for_file(
            file_path, outer_root
        )
        # Second call: inner_root is the NEW tree root — its chain
        # must NOT inherit from outer_root's glossary.json.
        effective_b, _ = proc._effective_glossary_for_file(
            file_path, inner_root
        )

        assert effective_a == {"OuterOnly": "outer-val"}
        assert effective_b is None

    def test_cached_merged_chain_survives_across_subdirs(self, tmp_path):
        source_dir = tmp_path / "docs"
        branch_a = source_dir / "a"
        branch_b = source_dir / "b"
        _write_glossary(source_dir, {"Shared": "shared"})
        _write_glossary(branch_a, {"AOnly": "a-val"})
        _write_glossary(branch_b, {"BOnly": "b-val"})

        file_a = branch_a / "page.md"
        file_b = branch_b / "page.md"
        file_a.parent.mkdir(parents=True, exist_ok=True)
        file_b.parent.mkdir(parents=True, exist_ok=True)
        file_a.write_text("x", encoding="utf-8")
        file_b.write_text("y", encoding="utf-8")

        proc = _make_processor()
        eff_a, _ = proc._effective_glossary_for_file(file_a, source_dir)
        eff_b, _ = proc._effective_glossary_for_file(file_b, source_dir)

        assert eff_a == {"Shared": "shared", "AOnly": "a-val"}
        assert eff_b == {"Shared": "shared", "BOnly": "b-val"}
        # Both paths should hit the directory cache for their respective
        # leaf dirs — the cache is populated after the first call.
        src_resolved = proc._safe_resolve(source_dir)
        assert (
            src_resolved,
            proc._safe_resolve(branch_a),
        ) in proc._glossary_dir_chain_cache
        assert (
            src_resolved,
            proc._safe_resolve(branch_b),
        ) in proc._glossary_dir_chain_cache


class TestEmptyChainBehaviour:
    def test_no_glossaries_anywhere_returns_none(self, tmp_path):
        source_dir = tmp_path / "docs"
        file_path = source_dir / "page.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("content", encoding="utf-8")

        proc = _make_processor()
        effective, chain = proc._effective_glossary_for_file(
            file_path, source_dir
        )

        assert effective is None
        # Chain is empty only if cwd also lacks a glossary.json; the
        # subdirectory walk and constructor override contribute
        # nothing.  We just assert the in-tree component is empty.
        for p in chain:
            assert source_dir.resolve() not in p.resolve().parents

    def test_empty_chain_preserves_registry_fallback(self, tmp_path):
        """Worker fall-through: no cascade => existing registry applies.

        ``_registry_for_glossary(None)`` returns a registry with no
        ``glossary:`` patterns (only built-ins + user patterns), which
        is what the constructor-time path always produced when the user
        hadn't configured a glossary.
        """
        proc = _make_processor()
        registry = proc._registry_for_glossary(None)
        pattern_names = [p.name for p in registry.patterns]
        assert not any(name.startswith("glossary:") for name in pattern_names)


class TestPerLocaleResolution:
    def test_cascade_resolves_per_locale_dict(self, tmp_path):
        source_dir = tmp_path / "docs"
        _write_glossary(
            source_dir,
            {
                "pull request": {
                    "ko": "풀 리퀘스트",
                    "ja": "プルリクエスト",
                }
            },
        )
        file_path = source_dir / "page.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("x", encoding="utf-8")

        proc_ko = _make_processor(target_lang="ko")
        proc_ja = _make_processor(target_lang="ja")

        eff_ko, _ = proc_ko._effective_glossary_for_file(file_path, source_dir)
        eff_ja, _ = proc_ja._effective_glossary_for_file(file_path, source_dir)

        assert eff_ko == {"pull request": "풀 리퀘스트"}
        assert eff_ja == {"pull request": "プルリクエスト"}

    def test_child_dict_merges_into_parent_dict_per_locale(self, tmp_path):
        """Per-locale child dict MUST NOT wipe parent locales it omits.

        A child that only refreshes the Korean value for ``API`` still
        needs the Japanese run to inherit the parent's ``ja`` entry —
        otherwise the cascade silently drops the term for locales the
        child didn't think about.
        """
        source_dir = tmp_path / "docs"
        child_dir = source_dir / "child"
        _write_glossary(
            source_dir,
            {"API": {"ko": "parent-ko", "ja": "parent-ja"}},
        )
        _write_glossary(
            child_dir,
            {"API": {"ko": "child-ko"}},
        )
        file_path = child_dir / "page.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("x", encoding="utf-8")

        proc_ko = _make_processor(target_lang="ko")
        proc_ja = _make_processor(target_lang="ja")

        eff_ko, _ = proc_ko._effective_glossary_for_file(
            file_path, source_dir
        )
        eff_ja, _ = proc_ja._effective_glossary_for_file(
            file_path, source_dir
        )

        assert eff_ko == {"API": "child-ko"}
        # ja inherits the parent's entry — child didn't override it.
        assert eff_ja == {"API": "parent-ja"}

    def test_child_scalar_replaces_parent_dict_wholesale(self, tmp_path):
        """A scalar child overrides the whole per-locale parent mapping.

        Replacing a dict with ``None`` (do-not-translate) or a string
        applies across every locale — the cascade deliberately does
        NOT try to preserve parent locales in this case, because a
        scalar and a dict are semantically different and keeping the
        parent's dict would contradict the child's intent.
        """
        source_dir = tmp_path / "docs"
        child_dir = source_dir / "child"
        _write_glossary(source_dir, {"API": {"ko": "parent-ko"}})
        _write_glossary(child_dir, {"API": None})
        file_path = child_dir / "page.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("x", encoding="utf-8")

        proc = _make_processor(target_lang="ko")
        effective, _ = proc._effective_glossary_for_file(
            file_path, source_dir
        )
        assert effective == {"API": None}


class TestRegistryWiring:
    def test_placeholder_mode_registry_carries_cascade_terms(self, tmp_path):
        source_dir = tmp_path / "docs"
        _write_glossary(source_dir, {"GitHub": None})
        file_path = source_dir / "page.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("Visit GitHub.", encoding="utf-8")

        proc = _make_processor()
        effective, _ = proc._effective_glossary_for_file(file_path, source_dir)
        registry = proc._registry_for_glossary(effective)

        names = [p.name for p in registry.patterns]
        assert "glossary:GitHub" in names

    def test_registry_cache_reuses_identical_glossaries(self, tmp_path):
        proc = _make_processor()
        gloss = {"GitHub": None, "API": "API"}
        reg_a = proc._registry_for_glossary(gloss)
        reg_b = proc._registry_for_glossary(dict(gloss))
        assert reg_a is reg_b

    def test_tls_glossary_override_takes_effect(self, tmp_path):
        """``_current_glossary`` reads the TLS override when set."""
        proc = _make_processor()
        # No constructor glossary.
        assert proc._current_glossary() is None
        proc._tls.per_file_glossary = {"GitHub": None}
        try:
            assert proc._current_glossary() == {"GitHub": None}
        finally:
            del proc._tls.per_file_glossary

    def test_tls_explicit_none_overrides_constructor_glossary(self):
        """Explicit ``None`` on TLS means ``no glossary for this file``."""
        proc = MarkdownProcessor(
            model="test-model",
            target_lang="ko",
            glossary={"API": "API"},
            glossary_mode="placeholder",
        )
        # With no TLS, constructor glossary is returned.
        assert proc._current_glossary() == {"API": "API"}
        proc._tls.per_file_glossary = None
        try:
            # TLS None overrides: current glossary is empty for this file.
            assert proc._current_glossary() is None
        finally:
            del proc._tls.per_file_glossary


class TestMalformedGlossary:
    def test_non_json_file_is_ignored_with_warning(self, tmp_path, caplog):
        source_dir = tmp_path / "docs"
        source_dir.mkdir(parents=True, exist_ok=True)
        (source_dir / "glossary.json").write_text("not json", encoding="utf-8")
        file_path = source_dir / "page.md"
        file_path.write_text("x", encoding="utf-8")

        proc = _make_processor()
        with caplog.at_level("WARNING"):
            effective, _ = proc._effective_glossary_for_file(
                file_path, source_dir
            )

        assert effective is None
        assert any(
            "glossary.json" in record.message for record in caplog.records
        )

    def test_non_object_json_is_ignored(self, tmp_path):
        source_dir = tmp_path / "docs"
        source_dir.mkdir(parents=True, exist_ok=True)
        (source_dir / "glossary.json").write_text("[1,2,3]", encoding="utf-8")
        file_path = source_dir / "page.md"
        file_path.write_text("x", encoding="utf-8")

        proc = _make_processor()
        effective, _ = proc._effective_glossary_for_file(
            file_path, source_dir
        )

        assert effective is None


class TestResolveDirChainRaw:
    def test_chain_stops_at_source_dir_root(self, tmp_path):
        """The walk MUST NOT climb above ``source_dir``.

        An unrelated ``glossary.json`` sitting outside the tree (e.g. a
        parent directory of the source tree, or cwd when it's unrelated)
        must NOT leak into the cascade for files inside the source tree.
        """
        outside = tmp_path
        source_dir = tmp_path / "docs"
        _write_glossary(outside, {"Outside": "leaked"})
        _write_glossary(source_dir, {"InTree": "kept"})
        file_path = source_dir / "page.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("x", encoding="utf-8")

        proc = _make_processor()
        raw = proc._resolve_dir_chain_raw(
            proc._safe_resolve(source_dir),
            proc._safe_resolve(source_dir),
        )

        assert raw == {"InTree": "kept"}
        assert "Outside" not in raw

    def test_file_outside_source_tree_returns_empty(self, tmp_path):
        source_dir = tmp_path / "docs"
        source_dir.mkdir()
        outside_file = tmp_path / "outside.md"
        outside_file.write_text("x", encoding="utf-8")
        proc = _make_processor()
        raw = proc._resolve_dir_chain_raw(
            proc._safe_resolve(outside_file.parent),
            proc._safe_resolve(source_dir),
        )
        assert raw == {}


class TestCacheInvalidation:
    def test_process_directory_clears_cascade_cache_between_runs(
        self, tmp_path
    ):
        """A mid-run edit to ``glossary.json`` MUST be picked up on the
        next ``process_directory`` call that reuses the same processor.

        The cascade caches are memoized on the processor instance, so
        without an explicit clear the second run would still serve the
        first run's on-disk snapshot.
        """
        source_dir = tmp_path / "docs"
        source_dir.mkdir(parents=True, exist_ok=True)
        gloss_path = source_dir / "glossary.json"
        gloss_path.write_text(
            json.dumps({"API": "old-value"}), encoding="utf-8"
        )
        # ``_effective_glossary_for_file`` populates the caches on the
        # first lookup.
        file_path = source_dir / "page.md"
        file_path.write_text("x", encoding="utf-8")
        proc = _make_processor()
        first, _ = proc._effective_glossary_for_file(file_path, source_dir)
        assert first == {"API": "old-value"}
        # Touch the on-disk glossary — simulate operator edit between
        # two ``process_directory`` runs.
        gloss_path.write_text(
            json.dumps({"API": "new-value"}), encoding="utf-8"
        )
        # Simulate what ``process_directory`` does at the start of
        # every run.
        proc._glossary_json_cache.clear()
        proc._glossary_dir_chain_cache.clear()
        proc._glossary_registry_cache.clear()
        second, _ = proc._effective_glossary_for_file(
            file_path, source_dir
        )
        assert second == {"API": "new-value"}


class TestBatchConcurrencyPropagation:
    def test_intra_file_workers_inherit_per_file_glossary(
        self, tmp_path, monkeypatch
    ):
        """``batch_concurrency>1`` spawns fresh threads — each MUST see
        the parent's per-file glossary/registry via the snapshot
        installed by ``_run_groups_concurrent``.

        Uses a stubbed ``_translate_group`` that only records the
        active glossary to avoid wiring a real LLM; the goal is to
        verify TLS propagation, not translation output.
        """
        import threading

        proc = _make_processor()
        # Install a per-file glossary on the caller thread.
        proc._tls.per_file_glossary = {"FromParent": "parent-val"}
        proc._tls.per_file_registry = proc._registry_for_glossary(
            {"FromParent": "parent-val"}
        )

        observed: list = []
        lock = threading.Lock()

        def _fake_translate_group(
            group, po_manager, pool, stats, *, inplace, usage, pool_lock=None
        ):
            with lock:
                observed.append(proc._current_glossary())

        monkeypatch.setattr(proc, "_translate_group", _fake_translate_group)

        try:
            proc._run_groups_concurrent(
                groups=[["seed"], ["a"], ["b"], ["c"]],
                po_manager=None,
                pool=None,
                stats={"processed": 0, "failed": 0, "skipped": 0},
                inplace=False,
                usage=None,
                source_path="test",
                total=4,
                concurrency=3,
            )
        finally:
            del proc._tls.per_file_glossary
            del proc._tls.per_file_registry

        # Every group (seed + 3 concurrent workers) must have seen the
        # parent's per-file glossary — never falling back to the
        # constructor's ``self._glossary`` (None here).
        assert observed == [{"FromParent": "parent-val"}] * 4


class TestCascadeApplyLevel:
    def test_apply_level_handles_all_value_kinds(self):
        merged: dict = {}
        MarkdownProcessor._apply_cascade_level(
            merged,
            {
                "a": "val-a",
                "b": None,
                "c": {"ko": "ko-c"},
            },
        )
        assert merged == {"a": "val-a", "b": None, "c": {"ko": "ko-c"}}

    def test_remove_on_empty_merged_is_noop(self):
        merged: dict = {}
        MarkdownProcessor._apply_cascade_level(merged, {"x": "__remove__"})
        assert merged == {}
