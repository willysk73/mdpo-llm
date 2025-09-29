import enum
import re
from typing import Iterable


class LanguageCode(enum.Enum):
    """Supported language buckets with a compiled detection pattern."""

    EN = ("English", r"[A-Za-z]")
    CN = ("Chinese", r"[\u4E00-\u9FFF]")
    JP = ("Japanese", r"[\u3040-\u309F\u30A0-\u30FF\u31F0-\u31FF]")
    KO = ("Korean", r"[\u1100-\u11FF\u3130-\u318F\uAC00-\uD7A3]")

    def __init__(self, label: str, pattern: str) -> None:
        self.label = label
        self._regex = re.compile(pattern)

    @property
    def pattern(self) -> re.Pattern[str]:
        """Return the compiled regex for this language."""
        return self._regex

    def in_text(self, text: str) -> bool:
        """Return True if this language appears in the given text."""
        return bool(self._regex.search(text))


def contains_language(text: str, languages: Iterable[LanguageCode]) -> bool:
    """Return True if `text` contains any of the specified languages."""
    return any(lang.in_text(text) for lang in languages)


def detect_languages(
    text: str, languages: Iterable[LanguageCode] = tuple(LanguageCode)
) -> set[LanguageCode]:
    """Return the set of languages detected in `text`."""
    return {lang for lang in languages if lang.in_text(text)}


def contains_any(text: str) -> bool:
    """Return True if `text` matches any known language bucket."""
    return contains_language(text, LanguageCode)
