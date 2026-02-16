"""Tests for language detection."""

import pytest

from mdpo_llm.language import (
    LanguageCode,
    contains_any,
    contains_language,
    detect_languages,
)


class TestLanguageCodeEnum:
    def test_english_detected(self):
        assert LanguageCode.EN.in_text("Hello world")

    def test_chinese_detected(self):
        assert LanguageCode.CN.in_text("你好世界")

    def test_japanese_detected(self):
        assert LanguageCode.JP.in_text("こんにちは")

    def test_japanese_katakana_detected(self):
        assert LanguageCode.JP.in_text("カタカナ")

    def test_korean_detected(self):
        assert LanguageCode.KO.in_text("안녕하세요")

    def test_english_not_in_korean(self):
        assert not LanguageCode.EN.in_text("안녕하세요")

    def test_korean_not_in_english(self):
        assert not LanguageCode.KO.in_text("Hello world")

    def test_empty_string_no_detection(self):
        for lang in LanguageCode:
            assert not lang.in_text("")

    def test_numbers_only_no_language(self):
        assert not LanguageCode.KO.in_text("12345")
        assert not LanguageCode.CN.in_text("12345")
        assert not LanguageCode.JP.in_text("12345")

    def test_label_attribute(self):
        assert LanguageCode.EN.label == "English"
        assert LanguageCode.CN.label == "Chinese"
        assert LanguageCode.JP.label == "Japanese"
        assert LanguageCode.KO.label == "Korean"


class TestContainsLanguage:
    def test_single_language_match(self):
        assert contains_language("Hello", [LanguageCode.EN])

    def test_single_language_no_match(self):
        assert not contains_language("Hello", [LanguageCode.KO])

    def test_multiple_languages_one_matches(self):
        assert contains_language("Hello", [LanguageCode.KO, LanguageCode.EN])

    def test_empty_languages_list(self):
        assert not contains_language("Hello", [])


class TestDetectLanguages:
    def test_mixed_en_ko(self):
        result = detect_languages("Hello 안녕하세요")
        assert LanguageCode.EN in result
        assert LanguageCode.KO in result

    def test_mixed_en_cn_jp(self):
        result = detect_languages("Hello 你好 こんにちは")
        assert LanguageCode.EN in result
        assert LanguageCode.CN in result
        assert LanguageCode.JP in result

    def test_pure_numbers(self):
        result = detect_languages("12345")
        assert result == set()

    def test_empty_string(self):
        result = detect_languages("")
        assert result == set()

    def test_subset_languages(self):
        result = detect_languages("Hello 안녕", [LanguageCode.EN, LanguageCode.KO])
        assert result == {LanguageCode.EN, LanguageCode.KO}


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
