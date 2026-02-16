"""Tests for language detection using BCP 47 locale strings."""

import pytest

from mdpo_llm.language import (
    LANGUAGE_PATTERNS,
    _resolve_primary,
    contains_any,
    contains_language,
    detect_languages,
)


class TestLanguagePatterns:
    def test_english_detected(self):
        assert LANGUAGE_PATTERNS["en"].search("Hello world")

    def test_chinese_detected(self):
        assert LANGUAGE_PATTERNS["zh"].search("你好世界")

    def test_japanese_detected(self):
        assert LANGUAGE_PATTERNS["ja"].search("こんにちは")

    def test_japanese_katakana_detected(self):
        assert LANGUAGE_PATTERNS["ja"].search("カタカナ")

    def test_korean_detected(self):
        assert LANGUAGE_PATTERNS["ko"].search("안녕하세요")

    def test_english_not_in_korean(self):
        assert not LANGUAGE_PATTERNS["en"].search("안녕하세요")

    def test_korean_not_in_english(self):
        assert not LANGUAGE_PATTERNS["ko"].search("Hello world")

    def test_empty_string_no_detection(self):
        for pattern in LANGUAGE_PATTERNS.values():
            assert not pattern.search("")

    def test_numbers_only_no_language(self):
        assert not LANGUAGE_PATTERNS["ko"].search("12345")
        assert not LANGUAGE_PATTERNS["zh"].search("12345")
        assert not LANGUAGE_PATTERNS["ja"].search("12345")

    def test_supported_languages(self):
        assert set(LANGUAGE_PATTERNS.keys()) == {"en", "zh", "ja", "ko"}


class TestResolvePrimary:
    def test_simple_locale(self):
        assert _resolve_primary("ko") == "ko"

    def test_region_stripped(self):
        assert _resolve_primary("zh-CN") == "zh"

    def test_case_insensitive(self):
        assert _resolve_primary("JA") == "ja"

    def test_complex_tag(self):
        assert _resolve_primary("zh-Hant-TW") == "zh"


class TestContainsLanguage:
    def test_single_language_match(self):
        assert contains_language("Hello", ["en"])

    def test_single_language_no_match(self):
        assert not contains_language("Hello", ["ko"])

    def test_multiple_languages_one_matches(self):
        assert contains_language("Hello", ["ko", "en"])

    def test_empty_languages_list(self):
        assert not contains_language("Hello", [])

    def test_region_subtag_works(self):
        assert contains_language("你好", ["zh-CN"])

    def test_unknown_locale_no_crash(self):
        assert not contains_language("Hello", ["xx"])


class TestDetectLanguages:
    def test_mixed_en_ko(self):
        result = detect_languages("Hello 안녕하세요")
        assert "en" in result
        assert "ko" in result

    def test_mixed_en_zh_ja(self):
        result = detect_languages("Hello 你好 こんにちは")
        assert "en" in result
        assert "zh" in result
        assert "ja" in result

    def test_pure_numbers(self):
        result = detect_languages("12345")
        assert result == set()

    def test_empty_string(self):
        result = detect_languages("")
        assert result == set()

    def test_subset_languages(self):
        result = detect_languages("Hello 안녕", ["en", "ko"])
        assert result == {"en", "ko"}

    def test_region_subtag(self):
        result = detect_languages("你好", ["zh-CN"])
        assert result == {"zh"}


class TestContainsAny:
    def test_english_text(self):
        assert contains_any("Hello")

    def test_korean_text(self):
        assert contains_any("안녕")

    def test_empty_string(self):
        assert not contains_any("")

    def test_numbers_only(self):
        assert not contains_any("12345")

    def test_symbols_only(self):
        assert not contains_any("!@#$%")
