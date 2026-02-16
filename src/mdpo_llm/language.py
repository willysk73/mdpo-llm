"""Language detection utilities using BCP 47 locale codes."""

import re
from typing import Iterable

# BCP 47 primary subtag → compiled detection regex
LANGUAGE_PATTERNS: dict[str, re.Pattern[str]] = {
    "en": re.compile(r"[A-Za-z]"),
    "zh": re.compile(r"[\u4E00-\u9FFF]"),
    "ja": re.compile(r"[\u3040-\u309F\u30A0-\u30FF\u31F0-\u31FF]"),
    "ko": re.compile(r"[\u1100-\u11FF\u3130-\u318F\uAC00-\uD7A3]"),
}


def _resolve_primary(locale: str) -> str:
    """Extract BCP 47 primary subtag: 'zh-CN' → 'zh'."""
    return locale.split("-")[0].lower()


def contains_language(text: str, langs: Iterable[str]) -> bool:
    """Return True if *text* contains any of the specified languages.

    Args:
        text: Text to check.
        langs: BCP 47 locale strings (e.g. ``["ko"]``, ``["zh-CN"]``).
    """
    for lang in langs:
        pattern = LANGUAGE_PATTERNS.get(_resolve_primary(lang))
        if pattern and pattern.search(text):
            return True
    return False


def detect_languages(
    text: str, langs: Iterable[str] | None = None
) -> set[str]:
    """Return the set of BCP 47 primary subtags detected in *text*.

    Args:
        text: Text to analyse.
        langs: Restrict detection to these locales.  Defaults to all
            supported languages.
    """
    targets = langs if langs is not None else LANGUAGE_PATTERNS.keys()
    return {
        _resolve_primary(lang)
        for lang in targets
        if LANGUAGE_PATTERNS.get(_resolve_primary(lang), re.compile("(?!x)x")).search(text)
    }


def contains_any(text: str) -> bool:
    """Return True if *text* matches any supported language pattern."""
    return contains_language(text, LANGUAGE_PATTERNS.keys())
