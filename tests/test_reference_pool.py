"""Tests for ReferencePool."""

import pytest
import polib

from mdpo_llm.reference_pool import ReferencePool


class TestEmptyPool:
    def test_find_similar_returns_empty(self):
        pool = ReferencePool()
        assert pool.find_similar("anything") == []

    def test_len_zero(self):
        pool = ReferencePool()
        assert len(pool) == 0


class TestAddAndFindSimilar:
    def test_add_grows_pool(self):
        pool = ReferencePool()
        pool.add("hello", "hola")
        assert len(pool) == 1
        pool.add("world", "mundo")
        assert len(pool) == 2

    def test_find_similar_returns_added_pairs(self):
        pool = ReferencePool()
        pool.add("hello world", "hola mundo")
        results = pool.find_similar("hello there")
        assert len(results) == 1
        assert results[0] == ("hello world", "hola mundo")

    def test_max_results_respected(self):
        pool = ReferencePool(max_results=2)
        for i in range(10):
            pool.add(f"sentence number {i}", f"translated {i}")
        results = pool.find_similar("sentence number 5")
        # Exact match excluded, so should return at most 2
        assert len(results) <= 2

    def test_results_sorted_by_similarity(self):
        pool = ReferencePool(max_results=10)
        pool.add("the quick brown fox", "le renard brun rapide")
        pool.add("completely different text", "texte totalement different")
        pool.add("the quick brown dog", "le chien brun rapide")

        results = pool.find_similar("the quick brown cat")
        # "the quick brown fox" and "the quick brown dog" should be more
        # similar than "completely different text"
        sources = [src for src, _ in results]
        assert sources.index("completely different text") > 0

    def test_exact_self_match_excluded(self):
        pool = ReferencePool()
        pool.add("exact text", "translation")
        pool.add("other text", "other translation")
        results = pool.find_similar("exact text")
        sources = [src for src, _ in results]
        assert "exact text" not in sources
        assert "other text" in sources


class TestSeedFromPo:
    def _make_po(self, entries):
        """Build a POFile from a list of (msgid, msgstr, flags, obsolete) tuples."""
        po = polib.POFile()
        po.metadata = {"Content-Type": "text/plain; charset=UTF-8"}
        for i, (msgid, msgstr, flags, obsolete) in enumerate(entries):
            entry = polib.POEntry(
                msgctxt=f"ctx:{i}",
                msgid=msgid,
                msgstr=msgstr,
            )
            entry.flags = list(flags) if flags else []
            entry.obsolete = obsolete
            if obsolete:
                po.append(entry)
                entry.obsolete = True
            else:
                po.append(entry)
        return po

    def test_picks_up_translated_entries(self):
        po = self._make_po([
            ("hello", "hola", [], False),
            ("world", "mundo", [], False),
        ])
        pool = ReferencePool()
        pool.seed_from_po(po)
        assert len(pool) == 2

    def test_skips_fuzzy(self):
        po = self._make_po([
            ("hello", "hola", ["fuzzy"], False),
            ("world", "mundo", [], False),
        ])
        pool = ReferencePool()
        pool.seed_from_po(po)
        assert len(pool) == 1

    def test_skips_empty_msgstr(self):
        po = self._make_po([
            ("hello", "", [], False),
            ("world", "mundo", [], False),
        ])
        pool = ReferencePool()
        pool.seed_from_po(po)
        assert len(pool) == 1

    def test_skips_obsolete(self):
        po = self._make_po([
            ("hello", "hola", [], True),
            ("world", "mundo", [], False),
        ])
        pool = ReferencePool()
        pool.seed_from_po(po)
        # Obsolete entry should be skipped
        assert len(pool) == 1
