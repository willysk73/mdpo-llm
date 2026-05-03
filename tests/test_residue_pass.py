"""Tests for the T-17 source-language residue post-processing pass."""

from __future__ import annotations

import logging
from typing import List, Tuple

import pytest

from mdpo_llm.residue_pass import (
    RESIDUE_FENCED_PROMPT,
    RESIDUE_INLINE_FILENAME_PROMPT,
    RESIDUE_INLINE_OTHER_PROMPT,
    RepairResult,
    ResidueSpan,
    SOURCE_LANG_PATTERNS,
    SUPPORTED_SOURCE_LANGS,
    apply_residue_pass,
    detect_residues,
    repair_block,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scripted_callable(responses: List[str]):
    """Return ``(callable, calls)`` where ``calls`` is the prompt log.

    The callable returns each ``responses[i]`` for the i-th invocation
    and raises ``IndexError`` if exceeded — so a test that expects N
    calls fails loudly when something issues N+1 (silent caller-side
    bugs would otherwise pass with a stale response replayed).
    """
    calls: List[Tuple[str, str]] = []

    def _call(system_prompt: str, user_text: str) -> str:
        calls.append((system_prompt, user_text))
        return responses[len(calls) - 1]

    return _call, calls


def _raising_callable(exc: Exception):
    """Return a callable that always raises ``exc`` and a call log."""
    calls: List[Tuple[str, str]] = []

    def _call(system_prompt: str, user_text: str) -> str:
        calls.append((system_prompt, user_text))
        raise exc

    return _call, calls


# ---------------------------------------------------------------------------
# Detection — per-language ranges
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source_lang, residue_char",
    [
        ("ko", "한"),  # Hangul syllable
        ("ja", "あ"),  # Hiragana
        ("ja", "カ"),  # Katakana
        ("zh", "汉"),  # CJK Unified Ideographs
    ],
)
def test_detect_residues_hits_each_language_range(source_lang, residue_char):
    text = f"Hello `code{residue_char}span` world"
    spans = detect_residues(text, source_lang)
    assert len(spans) == 1, spans
    assert spans[0].kind == "inline_other"
    assert residue_char in spans[0].text


def test_detect_residues_clean_output_returns_empty():
    text = "All English `var_name` and ```\nclean code\n```\nmore text."
    assert detect_residues(text, "ko") == []


def test_detect_residues_unsupported_source_language_is_no_op():
    # Russian is not in SUPPORTED_SOURCE_LANGS; even with Cyrillic in a
    # code span the residue pass declines to act.
    text = "Inline `функция` here."
    assert detect_residues(text, "ru") == []
    assert "ru" not in SUPPORTED_SOURCE_LANGS


def test_detect_residues_ignores_inline_inside_fenced_block():
    text = (
        "```python\n"
        "x = 1  # 한국어 주석\n"
        "```\n"
        "Then natural text."
    )
    spans = detect_residues(text, "ko")
    assert len(spans) == 1
    assert spans[0].kind == "fenced"


def test_detect_residues_classifies_filename_inline():
    text = "See `회원목록.md` for details."
    spans = detect_residues(text, "ko")
    assert len(spans) == 1
    assert spans[0].kind == "inline_filename"


def test_detect_residues_distinguishes_filename_from_natural_inline():
    # Spaces disqualify the filename heuristic — natural text path.
    text = "Call `함수 설명` here."
    spans = detect_residues(text, "ko")
    assert len(spans) == 1
    assert spans[0].kind == "inline_other"


def test_detect_residues_ignores_residue_outside_code_spans():
    # Source-language characters in PROSE are not the residue pass's
    # job — only code-shaped residue is in scope.  T-12 + structural
    # validators handle prose-level drift.
    text = "한국어 prose with no code spans here."
    assert detect_residues(text, "ko") == []


def test_detect_residues_returns_spans_in_document_order():
    text = "First `한1` then `한2`\n\n```\n한3\n```\n"
    spans = detect_residues(text, "ko")
    assert len(spans) == 3
    assert [s.start for s in spans] == sorted(s.start for s in spans)


# ---------------------------------------------------------------------------
# Repair — fenced code block
# ---------------------------------------------------------------------------


def test_repair_block_fenced_uses_fenced_prompt():
    text = "```python\nuser_name = '홍길동'  # 사용자 이름\n```"
    spans = detect_residues(text, "ko")
    assert spans and spans[0].kind == "fenced"
    repaired_block = (
        "```python\nuser_name = 'Hong Gil-dong'  # 사용자 이름\n```"
    )
    call, calls = _scripted_callable([repaired_block])

    result = repair_block(spans[0], "en", call)

    assert result.repaired is True
    assert result.text == repaired_block
    assert "fenced code block" in calls[0][0].lower()
    # Comments must be preserved verbatim per the prompt; we check the
    # prompt asserts that contract so prompt drift is caught here.
    assert "comments" in calls[0][0].lower()
    # Format substitution wired correctly.
    assert "en" in calls[0][0]


def test_repair_block_fenced_preserves_identifiers_in_prompt():
    """Prompt must explicitly forbid identifier renaming."""
    assert "identifiers" in RESIDUE_FENCED_PROMPT.lower()


# ---------------------------------------------------------------------------
# Repair — inline filename
# ---------------------------------------------------------------------------


def test_repair_block_inline_filename_uses_filename_prompt():
    text = "See `회원목록.md` for details."
    spans = detect_residues(text, "ko")
    assert spans[0].kind == "inline_filename"
    # LLM responds without backticks per prompt; repair_block re-wraps.
    call, calls = _scripted_callable(["MEMBER_LIST.md"])

    result = repair_block(spans[0], "en", call)

    assert result.repaired is True
    assert result.text == "`MEMBER_LIST.md`"
    # Filename prompt is distinct from the natural-text prompt.
    assert calls[0][0] == RESIDUE_INLINE_FILENAME_PROMPT
    # User text is sent WITHOUT surrounding backticks.
    assert calls[0][1] == "회원목록.md"


def test_repair_block_inline_filename_tolerates_backticked_response():
    text = "See `회원.json` here."
    spans = detect_residues(text, "ko")
    # Some models echo the backticks despite the prompt; repair_block
    # strips one pair before re-wrapping so the output is well-formed.
    call, _ = _scripted_callable(["`MEMBER.json`"])
    result = repair_block(spans[0], "en", call)
    assert result.text == "`MEMBER.json`"


# ---------------------------------------------------------------------------
# Repair — inline non-filename (natural translation)
# ---------------------------------------------------------------------------


def test_repair_block_inline_other_uses_natural_prompt():
    text = "Try `한국어 설명` here."
    spans = detect_residues(text, "ko")
    assert spans[0].kind == "inline_other"
    call, calls = _scripted_callable(["Korean description"])

    result = repair_block(spans[0], "en", call)

    assert result.repaired is True
    assert result.text == "`Korean description`"
    assert calls[0][0].startswith(RESIDUE_INLINE_OTHER_PROMPT.split("{")[0])
    # Sanity: the natural prompt is NOT the filename prompt.
    assert calls[0][0] != RESIDUE_INLINE_FILENAME_PROMPT


# ---------------------------------------------------------------------------
# Round-trip preservation (placeholder tokens)
# ---------------------------------------------------------------------------


def test_apply_residue_pass_preserves_placeholder_tokens():
    # ``⟦P:0⟧`` appears in the input msgstr (e.g. an undecoded token a
    # caller intentionally left in place); the residue pass MUST keep
    # the same token count after repair.
    text = "Use `한국어 ⟦P:0⟧` here."
    repaired_text = "Korean ⟦P:0⟧"
    call, _ = _scripted_callable([repaired_text])
    out = apply_residue_pass(text, "ko", "en", call)
    # Token preserved → repair accepted.
    assert out.count("⟦P:0⟧") == text.count("⟦P:0⟧") == 1


def test_apply_residue_pass_rejects_repair_that_introduces_token():
    # The pass-1 msgstr has no placeholder tokens; an LLM response
    # that fabricates one MUST be rejected and the original kept.
    text = "Call `한국어 함수`."
    bad_repair = "Korean function ⟦P:9⟧"  # spurious token added
    call, calls = _scripted_callable([bad_repair])
    out = apply_residue_pass(text, "ko", "en", call)
    assert out == text
    assert len(calls) == 1


def test_apply_residue_pass_rejects_repair_that_drops_token():
    text = "Use `한국어 ⟦P:0⟧` here."
    # LLM response that drops the existing token must also be rejected.
    call, _ = _scripted_callable(["Korean only"])
    out = apply_residue_pass(text, "ko", "en", call)
    assert out == text
    assert "⟦P:0⟧" in out


# ---------------------------------------------------------------------------
# Graceful failure
# ---------------------------------------------------------------------------


def test_apply_residue_pass_keeps_original_on_llm_exception(caplog):
    text = "Try `한국어 함수` here."
    call, calls = _raising_callable(RuntimeError("provider down"))
    with caplog.at_level(logging.WARNING, logger="mdpo_llm.residue_pass"):
        out = apply_residue_pass(text, "ko", "en", call)
    assert out == text  # pass-1 result wins
    assert len(calls) == 1  # one attempt was made
    assert any(
        "keeping pass-1 output" in record.message
        and "LLM call raised" in record.message
        for record in caplog.records
    )


def test_apply_residue_pass_keeps_original_on_empty_response():
    text = "Try `한국어` here."
    call, _ = _scripted_callable([""])  # whitespace / empty rejected
    out = apply_residue_pass(text, "ko", "en", call)
    assert out == text


# ---------------------------------------------------------------------------
# No-op clean output (zero LLM calls)
# ---------------------------------------------------------------------------


def test_apply_residue_pass_zero_calls_on_clean_output():
    text = "Pure English with `var_name` and no residue."
    call, calls = _scripted_callable([])
    out = apply_residue_pass(text, "ko", "en", call)
    assert out == text
    assert calls == []  # the brief's "zero LLM calls" guarantee


# ---------------------------------------------------------------------------
# Multi-span splice (offset stability)
# ---------------------------------------------------------------------------


def test_apply_residue_pass_splices_multiple_spans_in_order():
    text = "A `한1` and `한2` end."
    # Each inline span gets its own LLM response; the natural-text
    # prompt is used because the bodies don't match the filename
    # heuristic.  Replacing in reverse order keeps earlier offsets
    # valid; this test would catch an off-by-one if the splice
    # iterated forward.
    call, _ = _scripted_callable(["one", "two"])
    out = apply_residue_pass(text, "ko", "en", call)
    assert out == "A `one` and `two` end."


# ---------------------------------------------------------------------------
# Sanity: prompt registry — every prompt template is non-empty and
# references the rules tests above check for.
# ---------------------------------------------------------------------------


def test_module_level_prompt_constants_are_non_empty():
    for tmpl in (
        RESIDUE_FENCED_PROMPT,
        RESIDUE_INLINE_FILENAME_PROMPT,
        RESIDUE_INLINE_OTHER_PROMPT,
    ):
        assert isinstance(tmpl, str) and tmpl.strip()


def test_source_lang_patterns_cover_documented_languages():
    # The Decisions section names ko/ja/zh as the supported set; this
    # test pins that contract so adding a language requires updating
    # both the patterns AND the brief.
    assert set(SOURCE_LANG_PATTERNS.keys()) == {"ko", "ja", "zh"}


# ---------------------------------------------------------------------------
# Target-aware residue detection: source ranges that overlap target
# script must NOT be flagged as residue (Codex cycle-1 P1 — zh→ja
# false positive on legitimate Japanese kanji).
# ---------------------------------------------------------------------------


def test_zh_to_ja_does_not_flag_legitimate_kanji():
    # Japanese msgstr that legitimately contains kanji (e.g. 名前 =
    # "name").  Without target_lang filtering, the zh source pattern
    # (pure CJK ideographs) would mark every kanji as residue.
    text = "Use `名前` for the user."
    assert detect_residues(text, "zh", target_lang="ja") == []
    # Sanity: without target filtering the legacy behaviour still
    # flags the span (proves the new path actually changes behaviour).
    assert len(detect_residues(text, "zh")) == 1


def test_ja_to_zh_keeps_kana_as_residue_drops_shared_kanji():
    # Chinese msgstr with leftover Japanese hiragana ``あ`` is residue,
    # but the kanji ``名前`` part is legitimately Chinese script and
    # must NOT be flagged.
    text = "Try `名前あ` here."
    spans = detect_residues(text, "ja", target_lang="zh")
    assert len(spans) == 1
    # The flagged span includes the shared kanji because we report
    # the inline span as a unit; the test point is that the SPAN was
    # reported (the kana triggered detection) rather than the kanji
    # being treated as residue on its own.
    assert "あ" in spans[0].text


def test_ja_to_zh_no_residue_when_only_shared_kanji():
    # Pure-kanji span on a (ja → zh) translation must NOT be flagged.
    text = "Use `名前` for the user."
    assert detect_residues(text, "ja", target_lang="zh") == []


def test_same_source_and_target_returns_no_residue():
    # ``src == tgt`` is degenerate — the source IS the target script,
    # so by definition there is no source-language residue to detect.
    text = "한국어 `함수` 호출."
    assert detect_residues(text, "ko", target_lang="ko") == []


def test_apply_residue_pass_target_lang_filter_zero_calls():
    # End-to-end: a (zh → ja) call on legitimate Japanese kanji must
    # issue ZERO LLM calls and return the input verbatim.
    text = "Use `名前` for the user."
    call, calls = _scripted_callable([])
    out = apply_residue_pass(text, "zh", "ja", call)
    assert out == text
    assert calls == []


# ---------------------------------------------------------------------------
# Cycle 2 P2: reject repairs that don't actually remove the residue.
# Models often partially comply (``회원_LIST.md``) and the only thing
# stopping silent acceptance is a post-repair re-scan against the same
# residue pattern.
# ---------------------------------------------------------------------------


def test_apply_residue_pass_rejects_partially_repaired_filename(caplog):
    text = "See `회원목록.md` for details."
    # LLM returns a filename that still contains Korean — not a real
    # repair.  The residue pass must reject it and keep pass-1 output.
    call, _ = _scripted_callable(["회원_LIST.md"])
    with caplog.at_level(logging.WARNING, logger="mdpo_llm.residue_pass"):
        out = apply_residue_pass(text, "ko", "en", call)
    assert out == text
    assert any(
        "still contains source-language residue" in r.message
        for r in caplog.records
    )


def test_apply_residue_pass_rejects_partially_repaired_inline_other():
    text = "Try `한국어 함수` here."
    # LLM returns a partial translation that still leaves Korean.
    call, _ = _scripted_callable(["한국어 function"])
    out = apply_residue_pass(text, "ko", "en", call)
    assert out == text


def test_apply_residue_pass_accepts_fenced_repair_with_source_lang_comments():
    """Codex cycle-3 P1: the fenced prompt forbids translating
    comments, so a *correct* fenced repair on a block with
    source-language comments will (intentionally) still contain
    source-language characters in those comments.  The residue-
    not-removed re-scan must NOT reject such repairs.
    """
    text = (
        "```python\n"
        "user_name = '한국어'  # 사용자 이름\n"
        "```"
    )
    repaired_block = (
        "```python\n"
        "user_name = 'Korean'  # 사용자 이름\n"  # comment preserved
        "```"
    )
    call, _ = _scripted_callable([repaired_block])
    out = apply_residue_pass(text, "ko", "en", call)
    assert out == repaired_block


def test_apply_residue_pass_accepts_fully_repaired_output():
    """Sanity: fully-clean repairs still land on disk."""
    text = "Try `한국어 함수` here."
    call, _ = _scripted_callable(["Korean function"])
    out = apply_residue_pass(text, "ko", "en", call)
    assert out == "Try `Korean function` here."


# ---------------------------------------------------------------------------
# strip_code_spans helper (used by the processor's source-lang
# detection so intentional CJK identifiers in code spans don't get
# misread as source language).
# ---------------------------------------------------------------------------


def test_strip_code_spans_removes_inline_and_fenced():
    from mdpo_llm.residue_pass import strip_code_spans

    text = "Hello `code1` end.\n\n```\nblock_body\n```\n\nMore `c2` text."
    out = strip_code_spans(text)
    assert "code1" not in out
    assert "block_body" not in out
    assert "c2" not in out
    # Surrounding prose preserved.
    assert "Hello" in out and "More" in out and "text." in out


def test_strip_code_spans_makes_intentional_cjk_invisible_to_lang_detect():
    from mdpo_llm.language import detect_languages
    from mdpo_llm.residue_pass import strip_code_spans

    text = "See `用户.md` for details."
    # detect_languages on the raw text picks up zh from the code span.
    assert "zh" in detect_languages(text)
    # After stripping code spans the natural text is English-only.
    assert "zh" not in detect_languages(strip_code_spans(text))
