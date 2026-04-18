"""Tests for the post-translation validator."""

import pytest

from mdpo_llm.validator import validate


def test_clean_translation_passes():
    result = validate("Hello world", "안녕 세계", target_lang="ko")
    assert result.ok
    assert result.issues == []


def test_heading_level_mismatch():
    result = validate("## Section", "### 섹션", target_lang="ko")
    assert not result.ok
    assert any(i.check == "heading_level" for i in result.issues)


def test_heading_level_preserved():
    result = validate("## Section", "## 섹션", target_lang="ko")
    assert result.ok


def test_fence_count_mismatch():
    src = "```\ncode\n```\n\n```py\nmore\n```"
    # Translation lost one fence.
    tgt = "```\n코드\n```"
    result = validate(src, tgt, target_lang="ko")
    assert not result.ok
    assert any(i.check == "fence_count" for i in result.issues)


def test_glossary_do_not_translate_preserved():
    glossary = {"API": None}
    result = validate(
        "Call the API endpoint.",
        "API 엔드포인트를 호출하세요.",
        target_lang="ko",
        glossary=glossary,
    )
    assert result.ok


def test_glossary_do_not_translate_altered():
    glossary = {"API": None}
    result = validate(
        "Call the API endpoint.",
        "에이피아이 엔드포인트를 호출하세요.",
        target_lang="ko",
        glossary=glossary,
    )
    assert not result.ok
    assert any(i.check == "glossary_preserve" for i in result.issues)


def test_target_language_absent():
    # Pure English source, translation still in English — fails.
    result = validate(
        "Hello world", "Hello world (translated)", target_lang="ko"
    )
    assert not result.ok
    assert any(i.check == "target_language" for i in result.issues)


def test_target_language_present():
    result = validate("Hello world", "안녕 세계", target_lang="ko")
    assert result.ok


def test_identifier_echo_not_flagged():
    # "OpenAI API" is a legitimate technical term; an identity echo must not
    # trip the target-language check because it is not a translation attempt.
    result = validate("OpenAI API", "OpenAI API", target_lang="ko")
    assert not any(i.check == "target_language" for i in result.issues)


def test_unsupported_target_locale_skips_target_check():
    # "fr" has no regex in LANGUAGE_PATTERNS; the validator must not flag
    # every French translation as missing target characters.
    result = validate(
        "Hello world", "Bonjour le monde", target_lang="fr"
    )
    assert not any(i.check == "target_language" for i in result.issues)


def test_english_prose_echo_flagged():
    # Verbatim English echo of prose (contains the stopword "the") should
    # be flagged when the target is non-English — the model returned the
    # source untranslated.
    result = validate(
        "Reset the password",
        "Reset the password",
        target_lang="ko",
    )
    assert not result.ok
    assert any(i.check == "target_language" for i in result.issues)


def test_mixed_language_source_skips_target_check():
    # Source already contains Korean: translation only containing Korean passes.
    result = validate("Hello 안녕", "안녕하세요", target_lang="ko")
    # The `contains_any(source)` check skips target_language when source
    # already contains non-ASCII.
    assert not any(i.check == "target_language" for i in result.issues)


def test_strict_mode_checks_inline_code():
    src = "Use `foo` and `bar`."
    # Translation lost one backtick pair.
    tgt = "foo와 bar를 사용하세요."
    conservative = validate(src, tgt, target_lang="ko", mode="conservative")
    strict = validate(src, tgt, target_lang="ko", mode="strict")
    assert conservative.ok  # conservative ignores inline code count
    assert not strict.ok
    assert any(i.check == "inline_code_count" for i in strict.issues)


def test_reasons_concatenates_issues():
    result = validate("## Section", "섹션", target_lang="ko")
    assert "heading_level" in result.reasons()


def test_refine_purpose_skips_target_language_check():
    # Refine is same-language: English source, English "refined" output
    # must NOT trip the target-language check that translate-purpose
    # validation runs.
    result = validate(
        "Reset the password",
        "Reset your password",
        target_lang="en",
        purpose="refine",
    )
    assert result.ok
    assert not any(i.check == "target_language" for i in result.issues)


def test_refine_purpose_flags_language_drift():
    # Refine output must stay in the source language.  An English source
    # with a Korean output adds a script class — language_stability fail.
    result = validate(
        "Reset the password",
        "비밀번호를 재설정하세요",
        target_lang="en",
        purpose="refine",
    )
    assert not result.ok
    assert any(i.check == "language_stability" for i in result.issues)


def test_refine_language_stability_allows_script_subset():
    # Japanese source detects as both kana (``ja``) and CJK ideographs
    # (``zh``).  A refined paragraph that drops kanji in favour of pure
    # kana is a legitimate same-language rewrite and must NOT trip the
    # check — same CJK family on both sides.
    src = "こんにちは世界"  # hiragana + kanji → {'ja', 'zh'} → {cjk}
    tgt = "こんにちは"  # hiragana only → {'ja'} → {cjk}
    result = validate(src, tgt, target_lang="ja", purpose="refine")
    assert result.ok
    assert not any(i.check == "language_stability" for i in result.issues)


def test_refine_language_stability_allows_kanji_plus_kana():
    # Codex-reported regression: a kanji-only source detects as
    # ``{'zh'}``; a legitimate refinement adding kana detects as
    # ``{'ja', 'zh'}``.  Both resolve to the ``cjk`` family, so the
    # check must NOT flag this.
    src = "世界"  # kanji only → {'zh'} → {cjk}
    tgt = "こんにちは世界"  # kana + kanji → {'ja', 'zh'} → {cjk}
    result = validate(src, tgt, target_lang="ja", purpose="refine")
    assert result.ok
    assert not any(i.check == "language_stability" for i in result.issues)


def test_refine_language_stability_catches_mixed_source_translation():
    # Regression: if a mostly-English source contains a single
    # foreign-script token, a subset rule would let a fully-Korean
    # output pass because {'korean'} ⊆ {'latin','korean'}.  The
    # dominant-family rule catches this — 'latin' dominates the
    # source and the output must preserve it.
    src = "Reset the password (비밀번호)"
    tgt = "비밀번호를 재설정하세요"
    result = validate(src, tgt, target_lang="en", purpose="refine")
    assert not result.ok
    assert any(i.check == "language_stability" for i in result.issues)


def test_refine_language_stability_allows_minor_script_drop():
    # A legitimate refinement that drops a rare foreign-script token
    # while preserving the dominant Latin prose must still pass.
    src = "Use the 漢 character in names."
    tgt = "Use that character in names."
    result = validate(src, tgt, target_lang="en", purpose="refine")
    assert result.ok


def test_refine_language_stability_flags_new_script():
    # Adding a script class the source never contained (Korean hangul
    # in an English refine) fails the stability check even though the
    # source's script ("en") is also present in the output.
    result = validate(
        "Reset the password",
        "Reset the password 비밀번호",
        target_lang="en",
        purpose="refine",
    )
    assert not result.ok
    assert any(i.check == "language_stability" for i in result.issues)


def test_refine_purpose_allows_verbatim_source():
    # A clean source that the LLM returns verbatim is a valid refine
    # outcome — no checks should fail.
    result = validate(
        "Reset the password",
        "Reset the password",
        target_lang="en",
        purpose="refine",
    )
    assert result.ok


def test_refine_purpose_structural_checks_still_run():
    # Refine still enforces Markdown-structural invariants like heading
    # level preservation — only the target-language check is suppressed.
    result = validate(
        "## Overview",
        "### Overview",
        target_lang="en",
        purpose="refine",
    )
    assert not result.ok
    assert any(i.check == "heading_level" for i in result.issues)


def test_validator_rejects_unknown_purpose():
    # A typo like ``purpose="translation"`` would previously fall
    # through and silently suppress every semantic check — turn that
    # into a loud failure at the public API boundary instead.
    with pytest.raises(ValueError, match="purpose must be"):
        validate(
            "Hello",
            "Hello",
            target_lang="ko",
            purpose="translation",  # type: ignore[arg-type]
        )


def test_translate_purpose_still_runs_target_language():
    # Regression guard: translate is the default purpose and must keep
    # firing the target-language check that refine drops.
    result = validate(
        "Reset the password",
        "Reset the password",
        target_lang="ko",
    )
    assert not result.ok
    assert any(i.check == "target_language" for i in result.issues)
