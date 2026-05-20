"""Tests for the auto-glossary candidate extractor ``mdpo-llm suggest-glossary`` (T-23).

Covers:
  * Token extraction shape filter (single-word + 2..3-word phrases,
    code / URL / numeric stripping, stopword rejection).
  * Frequency / file-count thresholding.
  * Near-duplicate clustering via SequenceMatcher AND whole-word
    containment.
  * Default-translator behaviour (no-translator yields empty
    translations) and injectable translator stubs.
  * Output JSON schema (per-locale dict per key, sorted by canonical).
  * Refusal to overwrite ``glossary.json``.
  * Integration on a small fixture corpus driving the CLI surface.

Real LLM calls are NOT issued — every test either omits the
translator argument (default no-op path) or injects a deterministic
callable. The single test that drives the CLI mocks
``litellm.completion`` via the same pattern other CLI tests use.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence
from unittest.mock import MagicMock, patch

import pytest

from mdpo_llm.glossary_suggest import (
    AUTHORED_GLOSSARY_FILENAME,
    DEFAULT_MIN_FILES,
    DEFAULT_MIN_OCCURRENCES,
    DEFAULT_SIMILARITY_THRESHOLD,
    SUGGESTED_GLOSSARY_FILENAME,
    BulkTranslator,
    GlossaryCluster,
    GlossarySuggestion,
    TokenCandidate,
    _parse_bulk_response,
    _resolve_bulk_source,
    add_suggest_glossary_subparser,
    cluster_candidates,
    cmd_suggest_glossary,
    collect_candidates,
    extract_tokens,
    filter_by_thresholds,
    litellm_bulk_translator,
    main,
    suggest_glossary,
    write_suggested_glossary,
)


# ---------------------------------------------------------------------------
# Test helpers.
# ---------------------------------------------------------------------------


def _make_stub_translator(
    table: Dict[str, Dict[str, str]],
) -> BulkTranslator:
    """Return a deterministic :data:`BulkTranslator` driven by a lookup table.

    Sources missing from ``table`` produce an empty translations dict
    so the production code's "missing locale → empty string" fallback
    is exercised end-to-end.
    """

    def _translate(
        sources: Sequence[str], target_langs: Sequence[str]
    ) -> List[Dict[str, Any]]:
        return [
            {"source": src, "translations": dict(table.get(src, {}))}
            for src in sources
        ]

    return _translate


def _write_corpus(root: Path, files: Dict[str, str]) -> None:
    """Materialise ``{relative_path: body}`` under ``root``."""
    for rel, body in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")


# ---------------------------------------------------------------------------
# Token extraction.
# ---------------------------------------------------------------------------


class TestExtractTokens:
    def test_extracts_all_caps_acronym(self) -> None:
        tokens = extract_tokens("The WCS API is the gateway.")
        # ``API`` and ``WCS`` are accepted; ``The`` is a stopword.
        assert "WCS" in tokens
        assert "API" in tokens
        assert "The" not in tokens

    def test_extracts_camel_case(self) -> None:
        tokens = extract_tokens("GitHub and GitLab integrate with Markdown.")
        assert "GitHub" in tokens
        assert "GitLab" in tokens
        assert "Markdown" in tokens

    def test_extracts_camel_case_with_upper_prefix(self) -> None:
        # ``OAuth``, ``MdpoLLM``, and ``iOS`` carry a multi-letter
        # uppercase prefix that the original narrow CamelCase regex
        # rejected. The mixed-case predicate accepts them by checking
        # "has BOTH upper and lower" rather than fitting a single
        # case-transition pattern.
        tokens = extract_tokens(
            "OAuth and MdpoLLM ship on iOS. iPhone too.\n"
        )
        assert "OAuth" in tokens
        assert "MdpoLLM" in tokens
        assert "iOS" in tokens
        assert "iPhone" in tokens

    def test_phrase_extraction_two_and_three_words(self) -> None:
        tokens = extract_tokens(
            "The WCS API gateway connects to the WCS dashboard."
        )
        # Phrase candidates surface at every starting position.
        assert "WCS API" in tokens
        assert "API gateway" in tokens
        assert "WCS API gateway" in tokens
        assert "WCS dashboard" in tokens

    def test_skips_fenced_code(self) -> None:
        body = (
            "Use the GitHub API.\n\n"
            "```python\n"
            "import requests\n"
            "WCSGateway.connect()\n"
            "```\n\n"
            "And then call WCS again.\n"
        )
        tokens = extract_tokens(body)
        # ``WCSGateway`` lives inside the fence and must not surface.
        assert "WCSGateway" not in tokens
        assert "WCS" in tokens
        assert "GitHub" in tokens

    def test_skips_fenced_code_with_longer_closing_fence(self) -> None:
        # CommonMark allows the closing fence to be LONGER than the
        # opening fence. A backreference-based regex would miss this
        # and leak code identifiers into the candidate pool; the
        # line-walker enforces the "same char, length >=" rule.
        body = (
            "Open paragraph.\n\n"
            "```\n"
            "WCSGatewayInside.connect()\n"
            "````\n\n"  # Closing fence is 4 backticks (longer than 3 opener).
            "And then call WCS again.\n"
        )
        tokens = extract_tokens(body)
        assert "WCSGatewayInside" not in tokens
        assert "WCS" in tokens

    def test_indented_backticks_are_not_a_fence(self) -> None:
        # Per CommonMark §4.5 a fenced-code opener may be indented at
        # most three spaces; a four-or-more-space-indented line that
        # happens to start with backticks is an INDENTED code line,
        # not a fence opener. The fence stripper must not consume
        # following prose just because a backtick run happens to be
        # deeply indented inside a list item or similar construct.
        body = (
            "Pre WCS line.\n"
            "    ```not-a-fence\n"  # 4-space indent → not a fence opener.
            "Post GitHub line still surfaces.\n"
            "Trailing Anthropic mention.\n"
        )
        tokens = extract_tokens(body)
        # ``WCS`` (before the indented backtick) and the prose after
        # it both surface — the buggy version would consume all prose
        # after the indented backtick to EOF.
        assert "WCS" in tokens
        assert "GitHub" in tokens
        assert "Anthropic" in tokens

    def test_skips_fenced_code_tilde(self) -> None:
        # Tilde fences are a valid CommonMark alternative; the
        # line-walker must accept them too.
        body = (
            "Intro WCS.\n\n"
            "~~~python\n"
            "WCSTildeInside.connect()\n"
            "~~~\n\n"
            "Outro GitHub.\n"
        )
        tokens = extract_tokens(body)
        assert "WCSTildeInside" not in tokens
        assert "WCS" in tokens
        assert "GitHub" in tokens

    def test_skips_inline_code(self) -> None:
        tokens = extract_tokens("Call `WCSGateway.connect()` from the WCS SDK.")
        assert "WCSGateway" not in tokens
        assert "WCS" in tokens
        assert "SDK" in tokens

    def test_skips_urls_and_autolinks(self) -> None:
        tokens = extract_tokens(
            "Visit https://github.com/MyOrg/MyRepo and <https://example.com/Acme>."
        )
        # Tokens hidden in URL paths must not leak through.
        assert "MyOrg" not in tokens
        assert "MyRepo" not in tokens
        assert "Acme" not in tokens

    def test_skips_inline_link_labels(self) -> None:
        # ``[GitHub](url)`` is navigation chrome — the bracket body
        # must not leak into the candidate pool. Brand names that
        # only ever appear in link labels are intentionally NOT
        # counted (the operator's review pass surfaces them on the
        # next prose mention).
        tokens = extract_tokens(
            "See [GitHub Actions](https://github.com/features/actions) "
            "for details and [See the docs](#docs).\n"
            "Plain GitHub still shows up in prose.\n"
        )
        # ``Actions`` lived only inside the link label; ``GitHub`` in
        # prose still surfaces.
        assert "Actions" not in tokens
        assert "GitHub Actions" not in tokens
        assert "GitHub" in tokens

    def test_skips_reference_link_constructs(self) -> None:
        # Both the inline ``[label][ref]`` construct and the
        # line-anchored ``[ref]: url`` definition are stripped so a
        # link-heavy doc's reference table does not dominate the
        # histogram.
        tokens = extract_tokens(
            "Read [Anthropic Claude][claude-ref] and [Bedrock][].\n\n"
            "[claude-ref]: https://example.com/claude\n"
            "[Bedrock]: https://example.com/bedrock \"AWS Bedrock\"\n"
        )
        # Reference label bodies stripped; the definition line is
        # gone too (so ``"AWS"`` from the title attribute also does
        # not surface from the definition).
        assert "Anthropic" not in tokens
        assert "Claude" not in tokens
        assert "Bedrock" not in tokens
        assert "AWS" not in tokens

    def test_skips_numeric_runs(self) -> None:
        tokens = extract_tokens("The 2024 v1.2.3 release of WCS shipped 1,000 fixes.")
        assert "WCS" in tokens
        # Pure numbers / version strings never qualify under the
        # proper-noun shapes.
        assert not any(t in {"2024", "v1.2.3", "1,000"} for t in tokens)

    def test_stopwords_rejected(self) -> None:
        tokens = extract_tokens(
            "When The Anthropic team launched, This was new."
        )
        # ``Anthropic`` survives; ``When`` / ``The`` / ``This`` do not.
        assert "Anthropic" in tokens
        assert all(sw not in tokens for sw in ("When", "The", "This"))

    def test_phrase_does_not_span_sentence_boundary(self) -> None:
        # ``"WCS. GitHub"`` is a period-separated sentence boundary;
        # the two proper nouns are NOT contiguous in prose, so no
        # ``"WCS GitHub"`` phrase candidate may surface. The bug
        # this guards against was that the flat-text walker treated
        # ``"WCS"`` and ``"GitHub"`` as adjacent words and emitted
        # a bogus ``"WCS GitHub"`` phrase for the clusterer to ship
        # to the LLM.
        tokens = extract_tokens("WCS. GitHub launched. WCS. GitHub Pages.\n")
        assert "WCS GitHub" not in tokens
        # The legitimate ``"GitHub Pages"`` phrase (no boundary
        # between the two words) still surfaces.
        assert "GitHub Pages" in tokens

    def test_phrase_does_not_span_paragraph_break(self) -> None:
        # A double-newline paragraph break must also block phrase
        # extension. ``"WCS"`` and ``"GitHub"`` on consecutive
        # paragraphs are unrelated proper nouns, not a phrase.
        tokens = extract_tokens("WCS\n\nGitHub helps.\n")
        assert "WCS GitHub" not in tokens

    def test_phrase_breaks_on_non_shape_word(self) -> None:
        # ``the`` (lowercase) breaks the phrase walk between ``WCS``
        # and ``Dashboard``; no ``"WCS Dashboard"`` phrase should
        # appear because there is no consecutive run.
        tokens = extract_tokens("WCS is the Dashboard for analytics.")
        assert "WCS Dashboard" not in tokens
        assert "WCS" in tokens
        assert "Dashboard" in tokens

    def test_returns_list_with_multiplicity(self) -> None:
        # Frequency depends on multiplicity, so the return type must
        # preserve duplicates.
        tokens = extract_tokens("WCS WCS WCS API")
        assert tokens.count("WCS") >= 3


# ---------------------------------------------------------------------------
# Histogram & thresholding.
# ---------------------------------------------------------------------------


class TestCollectCandidates:
    def test_walks_markdown_only(self, tmp_path: Path) -> None:
        _write_corpus(
            tmp_path,
            {
                "a.md": "WCS API\n",
                "b.markdown": "WCS API\n",
                "c.txt": "WCS API\n",  # non-markdown, skipped.
                "d.rst": "WCS API\n",
            },
        )
        hist = collect_candidates(tmp_path)
        # Two files contribute: a.md and b.markdown.
        wcs = hist.get("WCS")
        assert wcs is not None
        assert wcs.count == 2
        assert len(wcs.files) == 2

    def test_counts_across_files(self, tmp_path: Path) -> None:
        _write_corpus(
            tmp_path,
            {
                "one.md": "WCS WCS\n",
                "two.md": "WCS\n",
                "sub/three.md": "WCS GitHub\n",
            },
        )
        hist = collect_candidates(tmp_path)
        wcs = hist["WCS"]
        assert wcs.count == 4
        assert len(wcs.files) == 3
        github = hist["GitHub"]
        assert github.count == 1
        assert len(github.files) == 1

    def test_handles_undecodable_file_gracefully(self, tmp_path: Path) -> None:
        (tmp_path / "ok.md").write_text("WCS API\n", encoding="utf-8")
        # Write invalid UTF-8 — the walker must skip it without
        # aborting the rest of the walk.
        (tmp_path / "broken.md").write_bytes(b"\xff\xfe\xfa not utf 8\n")
        hist = collect_candidates(tmp_path)
        assert "WCS" in hist


class TestFilterByThresholds:
    def test_drops_below_occurrence_threshold(self) -> None:
        cands = {
            "A": TokenCandidate("A", count=1, files=frozenset({"x.md"})),
            "B": TokenCandidate(
                "B", count=5, files=frozenset({"x.md", "y.md"})
            ),
        }
        kept = filter_by_thresholds(cands, min_occurrences=3, min_files=1)
        assert [c.text for c in kept] == ["B"]

    def test_drops_below_file_threshold(self) -> None:
        cands = {
            "A": TokenCandidate(
                "A", count=10, files=frozenset({"x.md"})
            ),
            "B": TokenCandidate(
                "B", count=10, files=frozenset({"x.md", "y.md"})
            ),
        }
        kept = filter_by_thresholds(cands, min_occurrences=1, min_files=2)
        assert [c.text for c in kept] == ["B"]

    def test_orders_by_descending_count(self) -> None:
        cands = {
            "A": TokenCandidate(
                "A", count=2, files=frozenset({"x.md", "y.md"})
            ),
            "B": TokenCandidate(
                "B", count=5, files=frozenset({"x.md", "y.md"})
            ),
            "C": TokenCandidate(
                "C", count=3, files=frozenset({"x.md", "y.md"})
            ),
        }
        kept = filter_by_thresholds(cands, min_occurrences=1, min_files=1)
        assert [c.text for c in kept] == ["B", "C", "A"]


# ---------------------------------------------------------------------------
# Clustering.
# ---------------------------------------------------------------------------


class TestClusterCandidates:
    def test_clusters_phrase_with_stem(self) -> None:
        # The whole-word containment rule should merge ``WCS``,
        # ``WCS API``, and ``WCS dashboard`` even though their
        # pairwise SequenceMatcher ratios are well below the threshold.
        cands = [
            TokenCandidate("WCS API", count=8, files=frozenset({"a.md", "b.md"})),
            TokenCandidate("WCS", count=5, files=frozenset({"a.md", "b.md"})),
            TokenCandidate(
                "WCS dashboard", count=3, files=frozenset({"c.md"})
            ),
        ]
        clusters = cluster_candidates(cands)
        assert len(clusters) == 1
        cluster = clusters[0]
        assert set(cluster.variants) == {"WCS API", "WCS", "WCS dashboard"}
        # Canonical picks the most-frequent member.
        assert cluster.canonical == "WCS API"
        assert cluster.total_count == 16
        assert "a.md" in cluster.files

    def test_does_not_merge_unrelated_terms(self) -> None:
        cands = [
            TokenCandidate(
                "GitHub", count=5, files=frozenset({"a.md", "b.md"})
            ),
            TokenCandidate(
                "GitLab", count=4, files=frozenset({"a.md", "c.md"})
            ),
            TokenCandidate(
                "Anthropic", count=3, files=frozenset({"a.md", "b.md"})
            ),
        ]
        clusters = cluster_candidates(cands)
        assert len(clusters) == 3
        canonicals = {c.canonical for c in clusters}
        assert canonicals == {"GitHub", "GitLab", "Anthropic"}

    def test_merges_close_spelling_variants(self) -> None:
        # ``"Markdownn"`` is a typo with SequenceMatcher ratio above
        # 0.85 against ``"Markdown"``; the threshold rule should
        # collapse them.
        cands = [
            TokenCandidate(
                "Markdown", count=10, files=frozenset({"a.md", "b.md"})
            ),
            TokenCandidate(
                "Markdownn", count=2, files=frozenset({"a.md", "b.md"})
            ),
        ]
        clusters = cluster_candidates(cands)
        assert len(clusters) == 1
        cluster = clusters[0]
        assert cluster.canonical == "Markdown"
        assert set(cluster.variants) == {"Markdown", "Markdownn"}

    def test_canonical_breaks_tie_by_length(self) -> None:
        # Equal counts → longer string wins; same length → lex asc.
        cands = [
            TokenCandidate(
                "WCS", count=5, files=frozenset({"a.md", "b.md"})
            ),
            TokenCandidate(
                "WCS API", count=5, files=frozenset({"a.md", "b.md"})
            ),
        ]
        clusters = cluster_candidates(cands)
        assert clusters[0].canonical == "WCS API"

    def test_generic_shared_word_does_not_collapse_unrelated_clusters(
        self,
    ) -> None:
        # ``"API"`` is a generic shared suffix of both ``"WCS API"``
        # and ``"GitHub API"``. The leading-stem cluster rule must NOT
        # fold those distinct product surfaces into one cluster, and
        # the bare ``API`` candidate must stand alone — the operator's
        # review pass needs the three rows as separate glossary
        # candidates with their own translations.
        cands = [
            TokenCandidate(
                "WCS API", count=10, files=frozenset({"a.md", "b.md"})
            ),
            TokenCandidate(
                "GitHub API", count=8, files=frozenset({"c.md", "d.md"})
            ),
            TokenCandidate(
                "API", count=4, files=frozenset({"a.md", "c.md"})
            ),
        ]
        clusters = cluster_candidates(cands)
        canonicals = {c.canonical for c in clusters}
        assert canonicals == {"WCS API", "GitHub API", "API"}

    def test_generic_term_arriving_first_does_not_absorb_phrases(
        self,
    ) -> None:
        # Regression for the arrival-order asymmetry: if the generic
        # ``"API"`` candidate has the highest count and is processed
        # first, the leading-stem rule must still refuse to absorb
        # later phrase candidates into its cluster. The previous
        # whole-word-containment rule appended ``"WCS API"`` and
        # ``"GitHub API"`` to the ``"API"`` cluster on a single-match
        # path and collapsed the three rows into one.
        cands = [
            TokenCandidate(
                "API", count=20, files=frozenset({"a.md", "b.md", "c.md"})
            ),
            TokenCandidate(
                "WCS API", count=5, files=frozenset({"a.md", "b.md"})
            ),
            TokenCandidate(
                "GitHub API", count=3, files=frozenset({"c.md", "d.md"})
            ),
        ]
        clusters = cluster_candidates(cands)
        canonicals = {c.canonical for c in clusters}
        assert canonicals == {"API", "WCS API", "GitHub API"}

    def test_bridging_stem_merges_two_phrase_clusters(self) -> None:
        # Walker arrival order: ``"WCS API"`` and ``"WCS dashboard"``
        # both have higher counts than the bare ``"WCS"`` stem, so
        # they form their own clusters BEFORE the stem is processed.
        # When ``"WCS"`` arrives last, it whole-word-contains in BOTH
        # phrase clusters; without the bridge-merge fix it would
        # silently attach to only one and leave the other split. The
        # invariant under test: a bridging candidate folds every
        # matching cluster together.
        cands = [
            TokenCandidate(
                "WCS API", count=10, files=frozenset({"a.md", "b.md"})
            ),
            TokenCandidate(
                "WCS dashboard", count=8, files=frozenset({"c.md", "d.md"})
            ),
            TokenCandidate(
                "WCS", count=4, files=frozenset({"a.md", "b.md"})
            ),
        ]
        clusters = cluster_candidates(cands)
        assert len(clusters) == 1
        variants = set(clusters[0].variants)
        assert variants == {"WCS API", "WCS dashboard", "WCS"}

    def test_cluster_order_is_deterministic(self) -> None:
        cands = [
            TokenCandidate(
                "Bravo", count=3, files=frozenset({"a.md", "b.md"})
            ),
            TokenCandidate(
                "Alpha", count=3, files=frozenset({"a.md", "b.md"})
            ),
            TokenCandidate(
                "Charlie", count=10, files=frozenset({"a.md", "b.md"})
            ),
        ]
        clusters = cluster_candidates(cands)
        # Sorted by count desc, then canonical asc.
        assert [c.canonical for c in clusters] == ["Charlie", "Alpha", "Bravo"]


# ---------------------------------------------------------------------------
# Bulk-translator parsing.
# ---------------------------------------------------------------------------


class TestParseBulkResponse:
    def test_parses_strict_json(self) -> None:
        raw = json.dumps(
            {
                "items": [
                    {
                        "source": "WCS",
                        "translations": {"ko": "더블유씨에스", "ja": "WCS"},
                    },
                    {
                        "source": "GitHub",
                        "translations": {"ko": "깃허브", "ja": "ギットハブ"},
                    },
                ]
            }
        )
        result = _parse_bulk_response(raw, ["WCS", "GitHub"])
        assert result[0]["source"] == "WCS"
        assert result[0]["translations"]["ko"] == "더블유씨에스"
        assert result[1]["translations"]["ja"] == "ギットハブ"

    def test_tolerates_json_fence(self) -> None:
        raw = (
            "```json\n"
            + json.dumps({"items": [{"source": "WCS", "translations": {"ko": "v"}}]})
            + "\n```"
        )
        result = _parse_bulk_response(raw, ["WCS"])
        assert result[0]["translations"]["ko"] == "v"

    def test_fills_missing_source_with_empty_dict(self) -> None:
        raw = json.dumps({"items": []})
        result = _parse_bulk_response(raw, ["WCS", "API"])
        assert [r["source"] for r in result] == ["WCS", "API"]
        assert all(r["translations"] == {} for r in result)

    def test_invalid_json_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            _parse_bulk_response("not json", ["WCS"])

    def test_missing_items_field_raises(self) -> None:
        with pytest.raises(ValueError):
            _parse_bulk_response("{}", ["WCS"])

    def test_normalises_numbered_source_label(self) -> None:
        # Some providers echo the user-prompt format ``[N] lang: term``
        # verbatim in the JSON response. The resolver must strip the
        # prefix and re-match the tail against the requested sources
        # so the translations still attach to the right row instead
        # of dropping silently.
        raw = json.dumps(
            {
                "items": [
                    {
                        "source": "[1] en: WCS",
                        "translations": {"ko": "WCS(ko)"},
                    },
                    {
                        "source": "[2] en: GitHub",
                        "translations": {"ko": "깃허브"},
                    },
                ]
            }
        )
        result = _parse_bulk_response(raw, ["WCS", "GitHub"])
        assert result[0]["translations"]["ko"] == "WCS(ko)"
        assert result[1]["translations"]["ko"] == "깃허브"

    def test_positional_index_fallback(self) -> None:
        # An LLM that returns ``[N]`` with a mutated term (typo /
        # paraphrase) should still resolve via the numeric index — the
        # caller's row stays attached to the right position.
        raw = json.dumps(
            {
                "items": [
                    {
                        "source": "[1] mutated text the llm changed",
                        "translations": {"ko": "fallback"},
                    }
                ]
            }
        )
        result = _parse_bulk_response(raw, ["WCS"])
        assert result[0]["translations"]["ko"] == "fallback"

    def test_drops_unresolvable_source_labels(self) -> None:
        # No exact match, no numbered prefix, no positional index →
        # the item is silently dropped (the LLM hallucinated a key the
        # caller never asked about). The requested source still
        # surfaces with an empty translation dict so the operator
        # sees the gap.
        raw = json.dumps(
            {
                "items": [
                    {
                        "source": "halucinated",
                        "translations": {"ko": "x"},
                    }
                ]
            }
        )
        result = _parse_bulk_response(raw, ["WCS"])
        assert result[0]["translations"] == {}

    def test_coerces_non_string_translation_values(self) -> None:
        raw = json.dumps(
            {
                "items": [
                    {
                        "source": "WCS",
                        "translations": {"ko": None, "ja": 42},
                    }
                ]
            }
        )
        result = _parse_bulk_response(raw, ["WCS"])
        # ``None`` collapses to ``""`` so the JSON output is always
        # string-valued; numeric values are stringified.
        assert result[0]["translations"]["ko"] == ""
        assert result[0]["translations"]["ja"] == "42"


# ---------------------------------------------------------------------------
# Source-label resolver (covers the encoding rules in one place).
# ---------------------------------------------------------------------------


class TestResolveBulkSource:
    def test_exact_match(self) -> None:
        assert _resolve_bulk_source("WCS", ["WCS", "GitHub"]) == "WCS"

    def test_numbered_prefix_with_lang(self) -> None:
        assert (
            _resolve_bulk_source("[2] en: GitHub", ["WCS", "GitHub"])
            == "GitHub"
        )

    def test_numbered_prefix_no_lang(self) -> None:
        assert (
            _resolve_bulk_source("[1] WCS", ["WCS", "GitHub"]) == "WCS"
        )

    def test_positional_when_tail_mismatch(self) -> None:
        # Tail does not exactly match any source — fall back to the
        # 1-based positional index.
        assert (
            _resolve_bulk_source(
                "[2] en: gIThub", ["WCS", "GitHub"]
            )
            == "GitHub"
        )

    def test_returns_none_for_hallucinated_key(self) -> None:
        # No exact match, no numbered prefix → cannot resolve.
        assert _resolve_bulk_source("OAuth", ["WCS", "GitHub"]) is None

    def test_returns_none_when_index_out_of_range(self) -> None:
        # Index outside the sources range is not a valid positional
        # fallback.
        assert _resolve_bulk_source("[99] en: x", ["WCS"]) is None


# ---------------------------------------------------------------------------
# End-to-end pipeline (with mocked translator).
# ---------------------------------------------------------------------------


class TestSuggestGlossary:
    def test_returns_translations_for_every_locale(self, tmp_path: Path) -> None:
        # Corpus designed so ``"WCS"``, ``"API"``, and ``"WCS API"``
        # tie at three occurrences each — the canonical tie-breaker
        # then picks the longest variant (``"WCS API"``), exercising
        # both the whole-word containment merge and the length tie-
        # break in one assertion.
        _write_corpus(
            tmp_path,
            {
                "a.md": "WCS API powers things. GitHub helps.\n",
                "b.md": "WCS API is fast. GitHub announced.\n",
                "c.md": "WCS API rocks. GitHub launched.\n",
            },
        )
        translator = _make_stub_translator(
            {
                "WCS API": {"ko": "WCS API(번역)", "ja": "WCS API(訳)"},
                "GitHub": {"ko": "깃허브", "ja": "ギットハブ"},
            }
        )
        suggestions = suggest_glossary(
            tmp_path,
            target_langs=["ko", "ja"],
            min_occurrences=2,
            min_files=2,
            translator=translator,
        )
        by_key = {s.canonical: s for s in suggestions}
        # ``WCS API`` is the canonical of the WCS cluster (longest
        # variant wins the count tie); ``GitHub`` is its own cluster.
        # ``API`` lives in its own cluster too — the leading-stem
        # rule deliberately refuses to fold ``"API"`` into the WCS
        # cluster because ``"WCS API"`` does not START with ``"API"``,
        # so a generic suffix acronym does not get collapsed onto an
        # unrelated product surface.
        assert "WCS API" in by_key
        assert "GitHub" in by_key
        wcs = by_key["WCS API"]
        assert wcs.translations["ko"] == "WCS API(번역)"
        assert wcs.translations["ja"] == "WCS API(訳)"
        # The stem ``WCS`` is a leading-word prefix of ``WCS API``
        # and is folded into the cluster as a variant. ``API`` is
        # NOT a variant — leading-stem rule.
        assert "WCS" in wcs.variants
        assert "WCS API" in wcs.variants
        assert "API" not in wcs.variants

    def test_missing_locales_render_as_empty_string(
        self, tmp_path: Path
    ) -> None:
        _write_corpus(
            tmp_path,
            {
                "a.md": "GitHub GitHub GitHub launched.\n",
                "b.md": "GitHub announced.\n",
                "c.md": "GitHub dashboard.\n",
            },
        )
        # Translator only fills ``ko`` — the verb should backfill an
        # empty ``zh-CN`` rather than dropping the key.
        translator = _make_stub_translator({"GitHub": {"ko": "깃허브"}})
        suggestions = suggest_glossary(
            tmp_path,
            target_langs=["ko", "zh-CN"],
            min_occurrences=2,
            min_files=2,
            translator=translator,
        )
        assert len(suggestions) == 1
        sugg = suggestions[0]
        assert sugg.translations["ko"] == "깃허브"
        assert sugg.translations["zh-CN"] == ""

    def test_empty_corpus_returns_empty(self, tmp_path: Path) -> None:
        # No markdown files at all → pipeline returns an empty list,
        # not an exception.
        result = suggest_glossary(
            tmp_path,
            target_langs=["ko"],
            translator=_make_stub_translator({}),
        )
        assert result == []

    def test_no_translator_yields_empty_translations(
        self, tmp_path: Path
    ) -> None:
        _write_corpus(
            tmp_path,
            {
                "a.md": "GitHub GitHub GitHub.\n",
                "b.md": "GitHub launched.\n",
                "c.md": "GitHub announced.\n",
            },
        )
        suggestions = suggest_glossary(
            tmp_path,
            target_langs=["ko", "ja"],
            min_occurrences=2,
            min_files=2,
            translator=None,
        )
        assert len(suggestions) == 1
        assert suggestions[0].translations == {"ko": "", "ja": ""}

    def test_rejects_empty_target_langs(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            suggest_glossary(
                tmp_path,
                target_langs=[],
                translator=_make_stub_translator({}),
            )

    def test_rejects_invalid_thresholds(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            suggest_glossary(
                tmp_path,
                target_langs=["ko"],
                min_occurrences=0,
                translator=_make_stub_translator({}),
            )
        with pytest.raises(ValueError):
            suggest_glossary(
                tmp_path,
                target_langs=["ko"],
                min_files=0,
                translator=_make_stub_translator({}),
            )
        with pytest.raises(ValueError):
            suggest_glossary(
                tmp_path,
                target_langs=["ko"],
                similarity_threshold=1.5,
                translator=_make_stub_translator({}),
            )


# ---------------------------------------------------------------------------
# Output writer + schema.
# ---------------------------------------------------------------------------


class TestWriteSuggestedGlossary:
    def test_emits_per_locale_dict_keyed_by_canonical(
        self, tmp_path: Path
    ) -> None:
        suggestions = [
            GlossarySuggestion(
                canonical="WCS",
                translations={"ko": "WCS(ko)", "ja": "WCS(ja)"},
                count=5,
                files=("a.md", "b.md"),
                variants=("WCS",),
            ),
            GlossarySuggestion(
                canonical="GitHub",
                translations={"ko": "깃허브", "ja": "ギットハブ"},
                count=3,
                files=("c.md",),
                variants=("GitHub",),
            ),
        ]
        output_path = tmp_path / SUGGESTED_GLOSSARY_FILENAME
        write_suggested_glossary(suggestions, output_path)
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        assert payload == {
            "WCS": {"ko": "WCS(ko)", "ja": "WCS(ja)"},
            "GitHub": {"ko": "깃허브", "ja": "ギットハブ"},
        }

    def test_refuses_to_overwrite_glossary_json(self, tmp_path: Path) -> None:
        suggestions: List[GlossarySuggestion] = []
        with pytest.raises(ValueError) as excinfo:
            write_suggested_glossary(
                suggestions, tmp_path / AUTHORED_GLOSSARY_FILENAME
            )
        assert AUTHORED_GLOSSARY_FILENAME in str(excinfo.value)

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        suggestions = [
            GlossarySuggestion(
                canonical="X",
                translations={"ko": "엑스"},
                count=1,
                files=("a.md",),
                variants=("X",),
            )
        ]
        deep = tmp_path / "fresh" / "subdir" / SUGGESTED_GLOSSARY_FILENAME
        write_suggested_glossary(suggestions, deep)
        assert deep.is_file()

    def test_empty_suggestions_writes_empty_object(self, tmp_path: Path) -> None:
        # No clusters is a valid (degenerate) run — the file is still
        # produced so the operator can see the empty result.
        output_path = tmp_path / SUGGESTED_GLOSSARY_FILENAME
        write_suggested_glossary([], output_path)
        assert output_path.is_file()
        assert json.loads(output_path.read_text(encoding="utf-8")) == {}


# ---------------------------------------------------------------------------
# Authored-glossary protection (cascade) — does NOT touch an existing
# ``glossary.json`` even when one is present.
# ---------------------------------------------------------------------------


class TestAuthoredGlossaryNotOverwritten:
    def test_existing_glossary_json_is_left_alone(self, tmp_path: Path) -> None:
        _write_corpus(
            tmp_path,
            {
                "a.md": "WCS WCS WCS API\n",
                "b.md": "WCS API GitHub GitHub\n",
                "c.md": "WCS dashboard GitHub.\n",
            },
        )
        authored = tmp_path / AUTHORED_GLOSSARY_FILENAME
        original = {"WCS": "WCS", "GitHub": None}
        authored.write_text(
            json.dumps(original, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        suggestions = suggest_glossary(
            tmp_path,
            target_langs=["ko"],
            min_occurrences=2,
            min_files=2,
            translator=_make_stub_translator({}),
        )
        write_suggested_glossary(
            suggestions, tmp_path / SUGGESTED_GLOSSARY_FILENAME
        )
        # The authored file is byte-equal to its original content; the
        # suggested file lives alongside.
        roundtrip = json.loads(authored.read_text(encoding="utf-8"))
        assert roundtrip == original
        assert (tmp_path / SUGGESTED_GLOSSARY_FILENAME).is_file()


# ---------------------------------------------------------------------------
# CLI surface.
# ---------------------------------------------------------------------------


def _stub_completion_response(payload: Dict[str, Any]) -> MagicMock:
    """Build a litellm-shaped completion response carrying a JSON payload."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = json.dumps(payload)
    return response


class TestCli:
    def test_subparser_registers_and_help_runs(self) -> None:
        import argparse

        parser = argparse.ArgumentParser(prog="mdpo-llm-test")
        sub = parser.add_subparsers(dest="command")
        add_suggest_glossary_subparser(sub)
        # Argparse raises SystemExit on --help — exit code 0 means
        # the parser accepted the registration.
        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args(["suggest-glossary", "--help"])
        assert excinfo.value.code == 0

    def test_cli_integration_with_mocked_litellm(self, tmp_path: Path) -> None:
        # Small fixture corpus the brief's test plan asks for. ``WCS``
        # appears as a standalone stem most often, so its cluster's
        # canonical is ``"WCS"`` (highest count); the WCS-prefixed
        # phrases collapse onto it via the whole-word containment
        # rule and become variants the operator sees in the cluster
        # records.
        source = tmp_path / "src"
        _write_corpus(
            source,
            {
                "intro.md": (
                    "# Introduction\n\n"
                    "The WCS API powers the WCS dashboard for ops teams.\n"
                    "GitHub launched GitHub Actions to automate WCS deploys.\n"
                ),
                "api/usage.md": (
                    "## WCS API usage\n\n"
                    "Call the WCS API from any client.\n"
                    "Authenticate via GitHub OAuth.\n"
                ),
                "guides/launch.md": (
                    "Launch the WCS dashboard from GitHub Pages.\n"
                    "The WCS API rate-limits at 1,000 reqs.\n"
                ),
            },
        )
        # Stub litellm.completion to return the bulk-translation JSON
        # keyed on the canonicals the pipeline picks (``"WCS"`` and
        # ``"GitHub"``).
        bulk_payload = {
            "items": [
                {
                    "source": "WCS",
                    "translations": {"ko": "WCS(ko)", "ja": "WCS(ja)"},
                },
                {
                    "source": "GitHub",
                    "translations": {"ko": "깃허브", "ja": "ギットハブ"},
                },
            ]
        }
        with patch(
            "mdpo_llm.glossary_suggest.litellm.completion",
            return_value=_stub_completion_response(bulk_payload),
        ):
            exit_code = main(
                [
                    "suggest-glossary",
                    str(source),
                    "--target",
                    "ko,ja",
                    "--model",
                    "gpt-4o",
                    "--min-occurrences",
                    "2",
                    "--min-files",
                    "2",
                ]
            )
        assert exit_code == 0
        output_path = source / SUGGESTED_GLOSSARY_FILENAME
        assert output_path.is_file()
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        # Both canonicals surfaced; the operator can review and
        # promote.
        assert "WCS" in payload
        assert "GitHub" in payload
        assert payload["WCS"] == {"ko": "WCS(ko)", "ja": "WCS(ja)"}
        assert payload["GitHub"] == {"ko": "깃허브", "ja": "ギットハブ"}

    def test_cli_rejects_missing_source(self, tmp_path: Path, capsys) -> None:
        missing = tmp_path / "no-such-dir"
        exit_code = main(
            [
                "suggest-glossary",
                str(missing),
                "--target",
                "ko",
                "--model",
                "gpt-4o",
            ]
        )
        assert exit_code == 2
        captured = capsys.readouterr()
        assert "does not exist" in captured.err

    def test_cli_rejects_file_source(self, tmp_path: Path, capsys) -> None:
        file_path = tmp_path / "single.md"
        file_path.write_text("WCS API\n", encoding="utf-8")
        exit_code = main(
            [
                "suggest-glossary",
                str(file_path),
                "--target",
                "ko",
                "--model",
                "gpt-4o",
            ]
        )
        assert exit_code == 2
        captured = capsys.readouterr()
        assert "not a directory" in captured.err

    def test_cli_rejects_empty_target(self, tmp_path: Path, capsys) -> None:
        source = tmp_path / "src"
        source.mkdir()
        (source / "a.md").write_text("x\n", encoding="utf-8")
        exit_code = main(
            [
                "suggest-glossary",
                str(source),
                "--target",
                ",,",
                "--model",
                "gpt-4o",
            ]
        )
        assert exit_code == 2
        captured = capsys.readouterr()
        assert "--target" in captured.err

    def test_cli_rejects_output_named_glossary_json(
        self, tmp_path: Path, capsys
    ) -> None:
        source = tmp_path / "src"
        source.mkdir()
        (source / "a.md").write_text("WCS API\n", encoding="utf-8")
        # The output path's basename is exactly ``glossary.json`` — the
        # verb hard-refuses before any LLM call.
        target = tmp_path / AUTHORED_GLOSSARY_FILENAME
        exit_code = main(
            [
                "suggest-glossary",
                str(source),
                "--target",
                "ko",
                "--model",
                "gpt-4o",
                "--output",
                str(target),
            ]
        )
        assert exit_code == 2
        captured = capsys.readouterr()
        assert AUTHORED_GLOSSARY_FILENAME in captured.err
        assert not target.exists()


# ---------------------------------------------------------------------------
# Top-level parser smoke test — ensures the verb is wired into
# ``mdpo_llm.__main__.build_parser`` without breaking sibling verbs.
# ---------------------------------------------------------------------------


class TestTopLevelWiring:
    def test_top_level_parser_lists_suggest_glossary(self) -> None:
        from mdpo_llm.__main__ import build_parser

        parser = build_parser()
        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args(["suggest-glossary", "--help"])
        assert excinfo.value.code == 0


# ---------------------------------------------------------------------------
# Default litellm-bound translator: smoke test ensures the factory
# returns a callable AND the callable wires through ``litellm.completion``.
# A full integration test against the LLM stays out of the suite (real
# calls are forbidden); we only verify the wire shape.
# ---------------------------------------------------------------------------


class TestLitellmBulkTranslator:
    def test_factory_returns_callable(self) -> None:
        translator = litellm_bulk_translator(model="gpt-4o", source_lang="en")
        assert callable(translator)

    def test_translator_invokes_litellm_completion(self) -> None:
        translator = litellm_bulk_translator(model="gpt-4o", source_lang="en")
        payload = {
            "items": [
                {"source": "WCS", "translations": {"ko": "WCS(ko)"}},
            ]
        }
        with patch(
            "mdpo_llm.glossary_suggest.litellm.completion",
            return_value=_stub_completion_response(payload),
        ) as mock_completion:
            out = translator(["WCS"], ["ko"])
        assert mock_completion.call_count == 1
        kwargs = mock_completion.call_args.kwargs
        assert kwargs["model"] == "gpt-4o"
        # The system + user messages were both sent.
        assert len(kwargs["messages"]) == 2
        assert kwargs["messages"][0]["role"] == "system"
        assert out == [
            {"source": "WCS", "translations": {"ko": "WCS(ko)"}}
        ]

    def test_translator_short_circuits_on_empty_sources(self) -> None:
        translator = litellm_bulk_translator(model="gpt-4o", source_lang="en")
        with patch(
            "mdpo_llm.glossary_suggest.litellm.completion"
        ) as mock_completion:
            assert translator([], ["ko"]) == []
        mock_completion.assert_not_called()
