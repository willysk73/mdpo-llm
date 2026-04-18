"""Tests for the post-translation validator."""

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
