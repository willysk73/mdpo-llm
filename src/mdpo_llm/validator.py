"""
Post-translation structural validator.

Cheap checks that flag obvious regressions against the source Markdown.
Conservative defaults; stricter checks opt-in via ``mode="strict"``.

A validator failure marks the PO entry fuzzy and records a reason line in the
entry's ``tcomment`` so a reviewer can see why it was flagged on re-run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

from .language import (
    LANGUAGE_PATTERNS,
    _resolve_primary,
    contains_language,
    detect_languages,
)
from .placeholder import PlaceholderMap, check_round_trip

Mode = Literal["conservative", "strict"]


_FENCE_RE = re.compile(r"^(```|~~~)", re.MULTILINE)
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
_WORD_RE = re.compile(r"[A-Za-z]+")

# Small set of common English function words.  Their presence in a block is a
# reliable signal that the block is prose rather than an identifier / product
# name, which lets us flag pure-English echoes (e.g. "Reset the password")
# without tripping on technical labels (e.g. "OpenAI API").
_EN_PROSE_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "of", "in", "on", "at",
    "to", "for", "with", "by", "from", "and", "or", "not", "this",
    "that", "these", "those", "it", "its", "as", "but", "if", "then",
    "so", "can", "will", "should", "would", "could", "you", "your",
    "we", "our", "they", "their",
})


@dataclass(frozen=True)
class ValidationIssue:
    check: str
    detail: str


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    issues: List[ValidationIssue]

    def reasons(self) -> str:
        return "; ".join(f"{i.check}: {i.detail}" for i in self.issues)


def _heading_level(text: str) -> Optional[int]:
    m = re.match(r"^(#{1,6})\s+", text)
    return len(m.group(1)) if m else None


def _count_fences(text: str) -> int:
    return len(_FENCE_RE.findall(text))


def _count_inline_code(text: str) -> int:
    return len(_INLINE_CODE_RE.findall(text))


def _looks_like_english_prose(text: str) -> bool:
    """Return True when ``text`` contains at least one common English
    function word, indicating it is prose rather than a technical label."""
    for token in _WORD_RE.findall(text.lower()):
        if token in _EN_PROSE_WORDS:
            return True
    return False


def validate(
    source: str,
    translation: str,
    *,
    target_lang: str,
    glossary: Optional[Dict[str, Optional[str]]] = None,
    mode: Mode = "conservative",
    placeholder_map: Optional[PlaceholderMap] = None,
    encoded_translation: Optional[str] = None,
) -> ValidationResult:
    """Validate ``translation`` against ``source``.

    Args:
        source: Original Markdown block.
        translation: Translated Markdown block, in its user-visible form
            (after placeholder decoding and any post-processing).  All
            structural checks run against this string.
        target_lang: BCP 47 locale of the translation.
        glossary: Glossary used during translation.  Terms mapped to ``None``
            are "do-not-translate" and must appear unchanged in the output.
        mode: ``"conservative"`` (default) runs cheap structural checks.
            ``"strict"`` additionally checks inline code count.
        placeholder_map: Mapping produced by
            :meth:`mdpo_llm.placeholder.PlaceholderRegistry.encode`.  When
            supplied, a ``placeholder_roundtrip`` issue is raised if any
            mapped token is missing, duplicated, or unexpected in
            ``encoded_translation`` (or, if that argument is ``None``, in
            ``translation``).
        encoded_translation: Pre-decode LLM output (still containing
            ``\u27e6P:N\u27e7`` tokens) for the round-trip check.  Structural
            checks do NOT run against this string — patterns that cover
            Markdown syntax (fenced code, inline code, headings) would
            otherwise flag correct translations as fuzzy.

    Returns:
        ``ValidationResult`` with ``ok`` and a list of ``ValidationIssue``.
    """
    issues: List[ValidationIssue] = []

    if placeholder_map is not None and placeholder_map:
        if encoded_translation is None:
            # Falling back to ``translation`` here would silently report
            # every token as missing (decode has already run), so refuse
            # to guess — external callers must thread through the raw
            # LLM output.
            raise ValueError(
                "validate(): placeholder_map requires encoded_translation "
                "(the pre-decode LLM output containing the tokens)."
            )
        reason = check_round_trip(encoded_translation, placeholder_map)
        if reason:
            issues.append(ValidationIssue("placeholder_roundtrip", reason))

    src_level = _heading_level(source)
    tgt_level = _heading_level(translation)
    if src_level != tgt_level:
        issues.append(
            ValidationIssue(
                "heading_level",
                f"source level={src_level} translation level={tgt_level}",
            )
        )

    src_fences = _count_fences(source)
    tgt_fences = _count_fences(translation)
    if src_fences != tgt_fences:
        issues.append(
            ValidationIssue(
                "fence_count",
                f"source={src_fences} translation={tgt_fences}",
            )
        )

    if glossary:
        for term, mapped in glossary.items():
            if mapped is None and term in source and term not in translation:
                issues.append(
                    ValidationIssue(
                        "glossary_preserve",
                        f'"{term}" must appear unchanged',
                    )
                )

    # Target-language presence: only fire when we have a detection pattern
    # for the target locale (otherwise the check is a false-positive machine
    # for any locale missing from LANGUAGE_PATTERNS — fr, de, es, etc.) and
    # the target is not English.  Two failure modes are flagged:
    #   1. The translation differs from the source but lacks any
    #      target-language characters (LLM did something but not a
    #      translation).
    #   2. The translation is an *exact echo* of English prose (contains a
    #      common English function word), which indicates the model returned
    #      the source untranslated.  Echoes of identifier-like labels such
    #      as "OpenAI API" are deliberately NOT flagged.
    target_primary = _resolve_primary(target_lang)
    if (
        target_primary in LANGUAGE_PATTERNS
        and target_primary != "en"
        and source.strip()
        and translation.strip()
    ):
        src_langs = detect_languages(source)
        lacks_target_chars = not contains_language(translation, [target_lang])
        differs = translation.strip() != source.strip()
        is_prose_echo = not differs and _looks_like_english_prose(source)

        if src_langs == {"en"} and lacks_target_chars and (differs or is_prose_echo):
            issues.append(
                ValidationIssue(
                    "target_language",
                    f"expected {target_lang} characters, none found",
                )
            )

    if mode == "strict":
        src_inline = _count_inline_code(source)
        tgt_inline = _count_inline_code(translation)
        if src_inline != tgt_inline:
            issues.append(
                ValidationIssue(
                    "inline_code_count",
                    f"source={src_inline} translation={tgt_inline}",
                )
            )

    return ValidationResult(ok=not issues, issues=issues)
