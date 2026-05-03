"""
Placeholder substitution framework (shared core).

Before the LLM sees a Markdown block, spans matched by registered patterns
are replaced with opaque tokens of the form ``\u27e6P:N\u27e6``.  After the
LLM replies the tokens are restored verbatim and a round-trip check
confirms every input token made it through exactly once.

T-4 ships this module with ZERO built-in patterns.  Downstream tasks
register the patterns they care about (T-5 glossary do-not-translate
terms, T-6 reference-link anchors, etc.) on a shared
:class:`PlaceholderRegistry` instance handed to
:class:`~mdpo_llm.processor.MarkdownProcessor`.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Pattern as RePattern,
    Tuple,
    Union,
)

# Marker used for pre-existing token-literals found in the source text.
# They're recorded as identity entries in the mapping so the round-trip
# check expects them in the output and numbering never collides with one.
LITERAL_PATTERN_NAME = "__literal__"

# Uses U+27E6 / U+27E7 (mathematical white brackets) because they are rare
# in technical prose yet still round-trip safely through JSON, UTF-8, and
# common Markdown renderers — unlike private-use codepoints which some
# providers strip.
TOKEN_RE = re.compile(r"\u27e6P:(\d+)\u27e7")

# T-6 built-in placeholder patterns.  Always registered on the effective
# registry the processor builds — no opt-out flag.  Each targets content
# that has been observed to mangle in real-world runs yet must be copied
# through verbatim because it forms permalinks or DOM identity:
#
#   * ``anchor`` — Kramdown / Pandoc inline attribute lists that define
#     a section / span id.  Matches the simple ``{#anchor-id}`` form AND
#     longer IAL variants such as ``{#overview .lead}`` or
#     ``{#overview key=val}``; every incoming ``#overview`` link depends
#     on the id surviving unchanged.
#   * ``html_attr`` — raw-HTML attribute pairs on an allowlist of
#     identity / link / structural attributes (``class``, ``id``,
#     ``href``, ``src``, ``srcset``, ``rel``, ``target``, ``name``,
#     ``for``, ``style``, ``type``, ``role``, ``lang``, ``dir``,
#     ``xml:lang``, ``xmlns``, ``action``, ``method``, ``width``,
#     ``height``, ``data-*``).  Matches quoted values (single or
#     double) AND HTML5 unquoted values so ``<img width=320>`` and
#     ``<a href=/docs>`` are protected too.  Compiled case-insensitively
#     because HTML attribute names are case-insensitive — ``HREF`` and
#     ``Class`` must survive just as readily as ``href`` / ``class``.
#     Attribute values that are normally user-facing prose — ``alt``,
#     ``title``, ``aria-label``, ``placeholder``, ``label``,
#     ``aria-description*`` — are deliberately NOT on the allowlist so
#     the LLM still gets to translate accessibility / tooltip text.
#
# Missing any of these tokens in the LLM output is a structural fail via
# :func:`check_round_trip` (same hard-fail path used by every other
# placeholder pattern), not a soft warning.
ANCHOR_PATTERN = re.compile(r"\{#[^\s{}][^{}]*\}")
HTML_ATTR_PATTERN = re.compile(
    r"""\b(?:class|id|href|srcset|src|rel|target|name|for|style|type|role|lang|dir|xml:lang|xmlns|action|method|width|height|data-[\w-]+)\b\s*=\s*(?:"[^"]*"|'[^']*'|[^\s"'=<>`]+)""",
    re.IGNORECASE,
)

# Matches an HTML opening tag (e.g. ``<a href="...">``, ``<img
# title="1 > 0">``); used by the ``html_attr`` predicate and the
# structural-position check to identify "inside a tag" context.  The
# ``"[^"]*"`` / ``'[^']*'`` alternatives let the pattern skip over
# quoted values that legitimately contain ``>`` (comparison text in a
# ``title``, inline SVG attribute, etc.) — a simple ``[^<>]*`` match
# would truncate at the quoted ``>`` and leave subsequent attributes
# unprotected.  Deliberately excludes closing tags (``</a>``),
# comments (``<!-- ... -->``), and XML declarations so they don't
# shift tag ordinals in the position check.
HTML_TAG_OPEN_RE = re.compile(r"""<[A-Za-z](?:"[^"]*"|'[^']*'|[^<>])*>""")


# Fence opener matches ``` or ~~~ at column ≤3, optionally after one
# or more blockquote markers (``>`` with optional trailing space) so
# fenced blocks nested inside blockquotes (``> ~~~html ...`` and
# ``>> ~~~html ...`` for multi-level quoting) aren't missed.  Closing
# fence matching below mirrors this prefix.
_FENCE_OPEN_RE = re.compile(r"^\s{0,3}(?:>\s?)*(`{3,}|~{3,})")
_BACKTICK_RUN_RE = re.compile(r"`+")
_HEADING_LINE_RE = re.compile(r"^\s{0,3}#{1,6}(?:\s|$)")
# A setext underline: runs of ``=`` (H1) or ``-`` (H2), up to 3 leading
# spaces, optional trailing whitespace.  Matches CommonMark's setext
# underline rule.
_SETEXT_UNDERLINE_RE = re.compile(r"^\s{0,3}(?:=+|-+)\s*$")
# Rough CommonMark list marker detection for "is this line inside a
# list item?" heuristics.  Bulleted (``-``, ``+``, ``*``) or ordered
# (``1.`` / ``1)``) with up to 3 leading spaces, followed by a space
# or EOL.
_LIST_MARKER_RE = re.compile(r"^\s{0,3}(?:[-+*]|\d+[.)])(?:\s|$)")


def _is_indented_code_line(lines: List[str], i: int) -> bool:
    """Heuristic: is ``lines[i]`` an indented code block line?

    CommonMark rule (relaxed for our single-block context): a line
    with 4+ leading spaces (or a tab) is an indented code block when
    it's separated from the previous paragraph by a blank line AND it
    isn't sitting inside a list-item continuation.  We default to
    "not code" on ambiguity so structurally important anchors / attrs
    in nested content still get protected — the cycle-16 regression
    from treating every 4-space-indented line as code.
    """
    line = lines[i]
    if not line.strip():
        return False
    if not (line.startswith("    ") or line.startswith("\t")):
        return False
    saw_blank = False
    for j in range(i - 1, -1, -1):
        prev = lines[j]
        if not prev.strip():
            saw_blank = True
            continue
        # Previous non-blank line found.
        if prev.startswith("    ") or prev.startswith("\t"):
            # Continuing an already-code block.
            return True
        if _LIST_MARKER_RE.match(prev):
            # Inside a list.  CommonMark distinguishes two cases by
            # whether a blank line separates the marker from the
            # indented line:
            #   * ``- item\\n    x``   → loose continuation (prose).
            #   * ``- item\\n\\n    x`` → code block nested under the
            #                              list item.
            # Only the latter is a code block; the former keeps its
            # protection so anchors / attrs in nested content still
            # get tokenized.
            return saw_blank
        if _HEADING_LINE_RE.match(prev):
            # ATX heading directly before the indented line: an
            # indented code block CAN follow a heading without a
            # blank line (headings aren't paragraphs per CommonMark,
            # so "cannot interrupt a paragraph" doesn't apply).
            return True
        # Non-blank, non-indented, non-list, non-heading prev: indented
        # line is a code block only when separated from that paragraph
        # by a blank line.
        return saw_blank
    # Top of text: treat leading indented line as code.
    return True


def _find_code_ranges(text: str) -> List[Tuple[int, int]]:
    """Return sorted ``(start, end)`` ranges covering Markdown code.

    Handles fenced blocks (``` and ``~~~`` of length ≥ 3) and inline
    backtick runs of any length — including the multi-backtick form
    ``\\`\\`see \\`foo\\` here\\`\\``` that intentionally wraps
    content containing a single backtick, and spans that wrap across
    a newline (CommonMark allows line endings inside an inline code
    span; they're treated like spaces).  Pairing is done over the
    full text rather than per line so the multi-line case is covered.
    Fence close char count must be ≥ open length.

    Ranges are inclusive of the delimiter characters on both sides so
    a position sitting on an opening / closing fence marker still
    counts as "in code" — the caller only wants to know whether the
    surrounding context is an uneditable code region.
    """
    ranges: List[Tuple[int, int]] = []
    if not text:
        return ranges

    lines = text.split("\n")
    line_offsets: List[int] = [0]
    for line in lines[:-1]:
        line_offsets.append(line_offsets[-1] + len(line) + 1)

    fenced_ranges: List[Tuple[int, int]] = []
    in_fence = False
    fence_char: Optional[str] = None
    fence_len = 0
    fence_start = 0
    for i, line in enumerate(lines):
        if in_fence:
            close_pat = (
                r"^\s{0,3}(?:>\s?)*"
                + re.escape(fence_char or "")
                + r"{"
                + str(fence_len)
                + r",}\s*$"
            )
            if re.match(close_pat, line):
                fenced_ranges.append(
                    (fence_start, line_offsets[i] + len(line))
                )
                in_fence = False
        else:
            m = _FENCE_OPEN_RE.match(line)
            if m:
                in_fence = True
                fence_char = m.group(1)[0]
                fence_len = len(m.group(1))
                fence_start = line_offsets[i]
    if in_fence:
        # Unclosed fence → treat rest of text as code so partial
        # encoding doesn't leak into whatever follows.
        fenced_ranges.append((fence_start, len(text)))

    # CommonMark indented code blocks: lines that begin with 4+ spaces
    # (or a tab) AND are not list-item continuation lines.  See
    # :func:`_is_indented_code_line` for the heuristic — defaults to
    # "not code" on ambiguity so structurally important anchors /
    # attrs in nested content still get protected.
    for i, line in enumerate(lines):
        line_offset = line_offsets[i]
        if any(s <= line_offset < e for s, e in fenced_ranges):
            continue
        if _is_indented_code_line(lines, i):
            fenced_ranges.append((line_offset, line_offset + len(line)))

    fenced_ranges.sort()
    ranges.extend(fenced_ranges)

    def in_fence_range(pos: int) -> bool:
        for s, e in fenced_ranges:
            if s <= pos < e:
                return True
        return False

    # Pair inline backtick runs over the whole text (multi-line
    # inline code spans are valid CommonMark).  Runs that fall inside
    # an already-identified fenced range (fence markers or fence body)
    # are ignored so they don't steal a closer from legitimate inline
    # code around the block.
    opened: Dict[int, Tuple[int, int]] = {}
    for m in _BACKTICK_RUN_RE.finditer(text):
        rs, re_end = m.start(), m.end()
        if in_fence_range(rs):
            continue
        rl = re_end - rs
        if rl in opened:
            o_start, _ = opened.pop(rl)
            ranges.append((o_start, re_end))
        else:
            opened[rl] = (rs, re_end)

    ranges.sort()
    return ranges


def _is_in_inline_code(text: str, position: int) -> bool:
    """True when ``position`` in ``text`` sits inside a Markdown code
    context (fenced block or inline backtick run of any length).

    Covers all three shapes the built-in placeholders need to keep
    their hands off:

    * Fenced blocks opened by 3+ backticks or tildes — both delimiter
      styles are supported because the Markdown parser used elsewhere
      in the pipeline accepts both.
    * Single-backtick inline spans (``\\`foo\\```).
    * Multi-backtick inline spans (``\\`\\`see \\`foo\\` here\\`\\```)
      that can legitimately contain a single backtick inside.
    """
    for s, e in _find_code_ranges(text):
        if s <= position < e:
            return True
    return False


def _in_quoted_value(tag_text: str, offset: int) -> bool:
    """True when ``offset`` inside ``tag_text`` falls within a quoted
    attribute value.

    Walks the tag from its opening ``<`` tracking quote state: a ``"``
    or ``'`` at top level opens a value; the matching closing quote
    closes it.  Used to reject ``HTML_ATTR_PATTERN`` matches that
    appear inside a translatable attribute's own value (e.g.
    ``title='see href="/docs"'`` — the inner ``href="/docs"`` is text
    inside ``title``, not a real attribute).
    """
    in_quote: Optional[str] = None
    for c in tag_text[:offset]:
        if in_quote is not None:
            if c == in_quote:
                in_quote = None
        elif c == '"' or c == "'":
            in_quote = c
    return in_quote is not None


def _is_inside_html_tag(text: str, start: int, end: int) -> bool:
    """Predicate used by the ``html_attr`` built-in pattern.

    Returns ``True`` when the match span ``[start, end)`` is a real
    top-level attribute pair in a ``<...>`` opening tag — i.e. inside
    a tag matched by :data:`HTML_TAG_OPEN_RE`, not nested inside a
    quoted value of another attribute, and not inside a Markdown
    backtick code span.  Three guards:

    * The quote-aware tag regex correctly spans HTML like
      ``<a title="1 > 0" href="/docs">`` so ``href`` (which follows
      the quoted ``>``) still counts as in-tag.
    * :func:`_in_quoted_value` rejects matches that land inside a
      quoted attribute value — important for markup like
      ``<a title='see href="/docs"' href="/real">`` where the inner
      ``href="/docs"`` is part of the translatable ``title`` text and
      must not be frozen as a placeholder.
    * :func:`_is_in_inline_code` rejects matches whose enclosing tag
      sits inside Markdown backticks (``Use `<a href="/docs">` to
      link``, fenced code blocks containing HTML examples).  Partial
      protection of ``href`` while leaving ``<a`` / ``>`` as prose
      would let the model rewrite the tag-like text while the
      placeholder count still balanced, so we just leave the whole
      example alone.
    """
    for m in HTML_TAG_OPEN_RE.finditer(text):
        if m.start() < start and end < m.end():
            if _is_in_inline_code(text, m.start()):
                return False
            return not _in_quoted_value(
                m.group(0), start - m.start()
            )
    return False


def _anchor_predicate(text: str, start: int, end: int) -> bool:
    """Predicate used by the ``anchor`` built-in pattern.

    Rejects ``{#...}`` spans inside Markdown backtick code — documents
    that discuss anchor syntax with literal examples
    (``Use `{#overview}` on a heading``) would otherwise have the
    example frozen as a placeholder, which partially encodes the code
    sample and can trigger spurious round-trip / position failures if
    the model rewrites the surrounding example text.  Matches the
    same guard the ``html_attr`` predicate applies.
    """
    return not _is_in_inline_code(text, start)


BUILTIN_PATTERNS: Tuple[Tuple[str, "re.Pattern[str]", Optional[Callable[[str, int, int], bool]]], ...] = (
    ("anchor", ANCHOR_PATTERN, _anchor_predicate),
    ("html_attr", HTML_ATTR_PATTERN, _is_inside_html_tag),
)


# T-14 auto source-language bracket placeholders.  Detect single-angle
# ``<source-lang-word>`` and single-brace ``{source-lang-word}`` spans
# whose content holds at least one NON-ASCII word character — typically
# CJK / Hangul / Kana / Cyrillic identifiers that the LLM would otherwise
# rewrite when asked to "translate" a path parameter or UI token.
#
# Not added to :data:`BUILTIN_PATTERNS` because T-6 built-ins are
# no-opt-out by contract, while T-14 ships with a constructor kwarg
# (``MarkdownProcessor(auto_bracket_placeholders=...)``) and a
# ``MDPO_AUTO_BRACKET_PLACEHOLDERS`` env-var override.  The registry
# caller (the processor) registers them explicitly after the glossary
# patterns using :func:`auto_bracket_predicate_factory` to compose the
# active glossary terms into the match-time predicate so glossary
# substitutions still win for any span they cover.
#
# Detection rule — the bracket content must hold at least one word
# character OUTSIDE the TARGET LANGUAGE'S primary script.  Python's
# ``re`` doesn't support character-class subtraction, so we spell it
# as ``[^\W<target-script-range>]``: NOT (non-word) AND NOT
# (target-script-range) = word-char AND not-in-target-script.
#
# Target-script gating matters because the brief says "auto-
# registration only fires for NON-target-script content": a Korean
# refine pass (``target_lang="ko"``) that hard-coded "non-ASCII" as
# the detection rule would still tokenize every ``{전송}`` / ``<다음>``
# span and never let the refiner polish them, and a Korean → Japanese
# translate pass would freeze both source-script and target-script
# brackets indiscriminately.  The default for unknown language codes
# is Latin/ASCII, which matches the repo's typical English-target
# workflow.
#
# The surrounding content class is intentionally NARROW — identifier
# shapes only (``\w``, ``-``, ``_``, ``.``).  Using the permissive
# ``[^<>]*`` / ``[^{}]*`` that the brief's shape sketches would
# match arbitrary prose / JSON whose brace run happens to hold a
# non-ASCII word char, so e.g. ``{"이름":"값"}`` and
# ``{ 상태: 한글 }`` would get frozen into opaque tokens and never
# reach the model — a regression that silently suppresses
# translation of perfectly translatable content.  Keeping the class
# tight aligns with the brief's "source-lang-word" singular wording
# and the ``{page_id}``-style URL-path-parameter example.
#
# Whitespace is deliberately EXCLUDED from the class too — a
# multi-word bracketed UI label like ``{상태 변경}`` or
# ``<следующий шаг>`` is ordinary translatable prose wrapped in
# brackets, not a source-language identifier, and since the feature
# is on by default any whitespace leniency silently suppresses those
# strings from reaching the model.  Callers who really do need to
# pin a multi-word bracketed label can register a project-specific
# user pattern via ``placeholders=PlaceholderRegistry(...)`` which
# takes priority over auto-register on exact-span ties.
#
# The surrounding lookbehind / lookahead excludes double-delimiter
# runtime templates (``{{var}}`` Mustache / Jinja, ``<<x>>``) so the
# inner single-brace / single-angle does NOT get tokenized away from
# under the template engine.
#
# HTML tag exclusion is NOT done at the regex level — an earlier
# ``(?![A-Za-z/!?])`` lookahead after ``<`` incorrectly rejected
# mixed-script identifiers that start with ASCII letters
# (``<id_게임코드>``, ``<userПример>``) that are legitimate source
# tokens.  Instead the narrow content class itself filters most
# HTML tag shapes (attribute values, slash, equals, quotes, spaces
# all fall outside ``[\w.\-]``) and the match-time predicate
# :func:`_inside_html_open_tag` rejects any auto-bracket span that
# falls INSIDE a tag body matched by :data:`HTML_TAG_OPEN_RE` —
# which covers quoted / unquoted / JSX-style attribute values in
# one check without false-positives on mixed-script IDs.
# Combining mark ranges for scripts the detection table claims to
# support (Latin, Cyrillic, Arabic, Hebrew, Greek, Thai, Devanagari,
# CJK, Tibetan, plus the adjacent Indic / Southeast-Asian blocks
# bundled together for completeness).  Python's ``\w`` excludes
# ``Mn`` / ``Mc`` / ``Me`` marks, so identifiers like ``{\u0915\u093f\u0924\u093e\u092c}``
# (Hindi ``किताब``) or ``{\u0633\u064e\u0644\u064e\u0627\u0645}``
# (Arabic ``سَلَام``) would never match the content class without
# these explicit ranges — and would silently fail the auto-bracket
# preservation contract for whole scripts the feature advertises.
_AUTO_BRACKET_MARK_RANGES = (
    r"\u0300-\u036F"  # Combining Diacritical Marks
    r"\u0483-\u0489"  # Cyrillic combining
    r"\u0591-\u05C7"  # Hebrew points
    r"\u0610-\u061A"  # Arabic extension
    r"\u064B-\u065F"  # Arabic harakat
    r"\u0670"  # Arabic superscript alef
    r"\u06D6-\u06ED"  # Arabic Quranic annotations
    r"\u0711"  # Syriac
    r"\u0730-\u074A"  # Syriac
    r"\u07A6-\u07B0"  # Thaana
    r"\u0816-\u0819\u081B-\u0823\u0825-\u0827\u0829-\u082D"  # Samaritan
    r"\u0859-\u085B"  # Mandaic
    r"\u08D3-\u0903"  # Arabic extended / Devanagari prefix
    r"\u093A-\u094F"  # Devanagari
    r"\u0951-\u0957"  # Devanagari
    r"\u0962-\u0963"  # Devanagari vowel extension
    r"\u0981-\u0983\u09BC\u09BE-\u09C4\u09C7-\u09C8\u09CB-\u09CD\u09D7\u09E2-\u09E3"  # Bengali
    r"\u0A01-\u0A03\u0A3C\u0A3E-\u0A42\u0A47-\u0A48\u0A4B-\u0A4D\u0A51"  # Gurmukhi
    r"\u0A81-\u0A83\u0ABC\u0ABE-\u0AC5\u0AC7-\u0AC9\u0ACB-\u0ACD\u0AE2-\u0AE3"  # Gujarati
    r"\u0B01-\u0B03\u0B3C\u0B3E-\u0B44\u0B47-\u0B48\u0B4B-\u0B4D\u0B56-\u0B57\u0B62-\u0B63"  # Oriya
    r"\u0B82\u0BBE-\u0BC2\u0BC6-\u0BC8\u0BCA-\u0BCD\u0BD7"  # Tamil
    r"\u0C00-\u0C04\u0C3E-\u0C44\u0C46-\u0C48\u0C4A-\u0C4D\u0C55-\u0C56\u0C62-\u0C63"  # Telugu
    r"\u0C81-\u0C83\u0CBC\u0CBE-\u0CC4\u0CC6-\u0CC8\u0CCA-\u0CCD\u0CD5-\u0CD6\u0CE2-\u0CE3"  # Kannada
    r"\u0D00-\u0D03\u0D3B-\u0D3C\u0D3E-\u0D44\u0D46-\u0D48\u0D4A-\u0D4D\u0D57\u0D62-\u0D63"  # Malayalam
    r"\u0D82-\u0D83\u0DCA\u0DCF-\u0DD4\u0DD6\u0DD8-\u0DDF\u0DF2-\u0DF3"  # Sinhala
    r"\u0E31\u0E34-\u0E3A\u0E47-\u0E4E"  # Thai
    r"\u0EB1\u0EB4-\u0EBC\u0EC8-\u0ECD"  # Lao
    r"\u0F18-\u0F19\u0F35\u0F37\u0F39\u0F3E-\u0F3F\u0F71-\u0F84\u0F86-\u0F87\u0F8D-\u0F97\u0F99-\u0FBC\u0FC6"  # Tibetan
)
_AUTO_BRACKET_CONTENT_CHARS = r"[\w.\-" + _AUTO_BRACKET_MARK_RANGES + r"]"

# BCP 47 primary-language (the substring before ``-`` / ``_``) →
# Unicode character-range string for that language's primary
# script.  Unlisted codes fall through to Latin/ASCII, which matches
# the codebase's typical English-target workflow and keeps the
# detection semantics equivalent to the earlier hard-coded
# "non-ASCII word char" rule for those callers.
# Latin covers Basic Latin + Latin-1 Supplement + Extended-A / B +
# Extended Additional so accented chars from Western-European
# languages (French ``é``, German ``Ü``, Spanish ``ñ``, Portuguese
# ``ã``, Polish ``ł``, Vietnamese tone marks, …) are classified as
# target-script for Latin-script locales.  A pure ASCII range would
# only cover English — any ``{étape}`` / ``<Überblick>`` in a
# ``fr`` / ``de`` document would otherwise auto-tokenize and never
# reach the translate / refine prompt.
_SCRIPT_RANGE_LATIN = (
    r"\u0000-\u007F"  # Basic Latin
    r"\u0080-\u00FF"  # Latin-1 Supplement
    r"\u0100-\u017F"  # Latin Extended-A
    r"\u0180-\u024F"  # Latin Extended-B
    r"\u1E00-\u1EFF"  # Latin Extended Additional
)
# Hangul-only, for Korean targets.  Hanja (shared with Chinese /
# Japanese via CJK Unified Ideographs) is NOT included — it is
# rarely used in modern Korean writing, and an ``{漢字}`` / ``{漢字ID}``
# token in a Korean-target document is almost certainly a Chinese /
# Japanese source identifier the caller wants preserved.
_SCRIPT_RANGE_HANGUL = (
    r"\u1100-\u11FF"  # Hangul Jamo
    r"\u3130-\u318F"  # Hangul Compatibility Jamo
    r"\uA960-\uA97F"  # Hangul Jamo Extended-A
    r"\uAC00-\uD7A3"  # Hangul Syllables
    r"\uD7B0-\uD7FF"  # Hangul Jamo Extended-B
)
# Japanese covers Hiragana + Katakana + Kanji (which shares the CJK
# Unified block with Chinese); ``{漢字}`` under a Japanese target is
# treated as target-script because Japanese writing routinely uses
# those same codepoints.
_SCRIPT_RANGE_JAPANESE = (
    r"\u3040-\u309F"  # Hiragana
    r"\u30A0-\u30FF"  # Katakana
    r"\u31F0-\u31FF"  # Katakana Phonetic Extensions
    r"\u3400-\u4DBF"  # CJK Unified Ideographs Extension A
    r"\u4E00-\u9FFF"  # CJK Unified Ideographs (shared Kanji)
)
# Chinese variants use Hanzi (CJK Unified) only; Hiragana /
# Katakana / Hangul content in brackets is non-target-script and
# should be preserved as a Japanese / Korean source identifier.
_SCRIPT_RANGE_CHINESE = (
    r"\u3400-\u4DBF"  # CJK Unified Ideographs Extension A
    r"\u4E00-\u9FFF"  # CJK Unified Ideographs (Hanzi)
)
_SCRIPT_RANGE_CYRILLIC = r"\u0400-\u052F"
_SCRIPT_RANGE_ARABIC = r"\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF"
_SCRIPT_RANGE_HEBREW = r"\u0590-\u05FF\uFB1D-\uFB4F"
_SCRIPT_RANGE_GREEK = r"\u0370-\u03FF"
_SCRIPT_RANGE_THAI = r"\u0E00-\u0E7F"
_SCRIPT_RANGE_DEVANAGARI = r"\u0900-\u097F"
_SCRIPT_RANGE_TIBETAN = r"\u0F00-\u0FFF"

_LANG_TO_SCRIPT_RANGE: Dict[str, str] = {}
_LANG_TO_SCRIPT_RANGE["ko"] = _SCRIPT_RANGE_HANGUL
_LANG_TO_SCRIPT_RANGE["ja"] = _SCRIPT_RANGE_JAPANESE
for _code in ("zh", "yue", "wuu", "nan", "hak"):
    _LANG_TO_SCRIPT_RANGE[_code] = _SCRIPT_RANGE_CHINESE
for _code in ("ru", "uk", "be", "bg", "sr", "mk", "ky", "tg", "kk", "mn"):
    _LANG_TO_SCRIPT_RANGE[_code] = _SCRIPT_RANGE_CYRILLIC
for _code in ("ar", "fa", "ur", "ps", "sd", "ug"):
    _LANG_TO_SCRIPT_RANGE[_code] = _SCRIPT_RANGE_ARABIC
for _code in ("he", "yi", "lad"):
    _LANG_TO_SCRIPT_RANGE[_code] = _SCRIPT_RANGE_HEBREW
for _code in ("el",):
    _LANG_TO_SCRIPT_RANGE[_code] = _SCRIPT_RANGE_GREEK
for _code in ("th",):
    _LANG_TO_SCRIPT_RANGE[_code] = _SCRIPT_RANGE_THAI
for _code in ("hi", "mr", "ne", "sa"):
    _LANG_TO_SCRIPT_RANGE[_code] = _SCRIPT_RANGE_DEVANAGARI
for _code in ("bo", "dz"):
    _LANG_TO_SCRIPT_RANGE[_code] = _SCRIPT_RANGE_TIBETAN


# BCP 47 script subtag (ISO 15924 four-letter code) -> script range.
# Takes precedence over the primary-language default so locales like
# ``sr-Latn`` (Latin-script Serbian) or ``zh-Hans`` (Simplified
# Chinese) pick the correct script instead of the language's default.
# ``Jpan`` / ``Kore`` / ``Hant`` / ``Hans`` are the superset aliases
# (Japanese writing, Korean writing, Traditional / Simplified
# Chinese) that BCP 47 uses; they map to the same ranges as
# ``ja`` / ``ko`` / ``zh`` respectively.
_SCRIPT_SUBTAG_TO_RANGE: Dict[str, str] = {
    "latn": _SCRIPT_RANGE_LATIN,
    "cyrl": _SCRIPT_RANGE_CYRILLIC,
    "hans": _SCRIPT_RANGE_CHINESE,
    "hant": _SCRIPT_RANGE_CHINESE,
    "hani": _SCRIPT_RANGE_CHINESE,
    "jpan": _SCRIPT_RANGE_JAPANESE,
    "hira": _SCRIPT_RANGE_JAPANESE,
    "kana": _SCRIPT_RANGE_JAPANESE,
    "hang": _SCRIPT_RANGE_HANGUL,
    "kore": _SCRIPT_RANGE_HANGUL,
    "arab": _SCRIPT_RANGE_ARABIC,
    "hebr": _SCRIPT_RANGE_HEBREW,
    "grek": _SCRIPT_RANGE_GREEK,
    "thai": _SCRIPT_RANGE_THAI,
    "deva": _SCRIPT_RANGE_DEVANAGARI,
    "tibt": _SCRIPT_RANGE_TIBETAN,
}


def _target_script_range(target_lang: Optional[str]) -> str:
    """Resolve a BCP 47 ``target_lang`` to its primary-script range.

    Returns the character-range string suitable for embedding inside
    a ``[^\\W...]`` regex class.  Falls back to the Latin range for
    ``None``, empty, or unrecognised codes so the default matches
    the common English-target workflow.

    A BCP 47 script subtag (ISO 15924 four-letter code, e.g.
    ``Latn``, ``Cyrl``, ``Hans``) takes precedence over the
    primary-language default — ``sr-Latn`` resolves to Latin even
    though ``sr`` alone defaults to Cyrillic, and ``zh-Hant`` /
    ``zh-Hans`` both land on Chinese because Chinese subtags are
    the same script for this purpose.  Region subtags (``en-US``,
    ``fr-CA``) don't carry script semantics and are ignored.
    """
    if not target_lang:
        return _SCRIPT_RANGE_LATIN
    normalised = target_lang.strip().lower().replace("_", "-")
    if not normalised:
        return _SCRIPT_RANGE_LATIN
    parts = normalised.split("-")
    base = parts[0]
    # Walk subtags for a 4-letter script code; the BCP 47 grammar
    # permits script after language and before region, but accept
    # any 4-letter subtag position so tools that emit a non-canonical
    # order still work.
    for sub in parts[1:]:
        if len(sub) == 4 and sub.isalpha():
            override = _SCRIPT_SUBTAG_TO_RANGE.get(sub)
            if override is not None:
                return override
    return _LANG_TO_SCRIPT_RANGE.get(base, _SCRIPT_RANGE_LATIN)


def build_auto_bracket_patterns(
    target_lang: Union[str, Iterable[str], None] = None,
) -> Tuple[Tuple[str, "re.Pattern[str]"], ...]:
    """Build T-14 auto-bracket patterns gated on ``target_lang``.

    ``target_lang`` may be a single BCP 47 string, an iterable of
    strings (multi-target runs), or ``None``.  The content "core"
    class matches word characters OUTSIDE the union of every
    target's primary-script range — so a single source encoding is
    safe to fan out to every language in the iterable:
    ``target_langs=["en", "ko"]`` keeps bare-Latin (``{game_id}``)
    AND bare-Hangul (``{전송}``) out of the auto-bracket stream, while
    cross-script content like ``{こんにちは}`` still tokenizes for
    both passes.  Single-string / ``None`` inputs retain the previous
    behaviour (one script range, unknown code → Latin).  Returns a
    tuple of ``(name, compiled_pattern)`` pairs in the same shape as
    the module-level :data:`AUTO_BRACKET_PATTERNS` Latin-target
    default.
    """
    if target_lang is None or isinstance(target_lang, str):
        script_range = _target_script_range(target_lang)
    else:
        seen = []
        for lang in target_lang:
            rng = _target_script_range(lang)
            if rng not in seen:
                seen.append(rng)
        if not seen:
            script_range = _SCRIPT_RANGE_LATIN
        else:
            script_range = "".join(seen)
    non_target_word = r"[^\W" + script_range + r"]"
    angle = re.compile(
        r"(?<!<)<"
        + _AUTO_BRACKET_CONTENT_CHARS
        + r"*"
        + non_target_word
        + r"+"
        + _AUTO_BRACKET_CONTENT_CHARS
        + r"*>(?!>)"
    )
    brace = re.compile(
        r"(?<!\{)\{"
        + _AUTO_BRACKET_CONTENT_CHARS
        + r"*"
        + non_target_word
        + r"+"
        + _AUTO_BRACKET_CONTENT_CHARS
        + r"*\}(?!\})"
    )
    return (
        ("auto_bracket_angle", angle),
        ("auto_bracket_brace", brace),
    )


AUTO_BRACKET_PATTERNS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    build_auto_bracket_patterns(None)
)
AUTO_BRACKET_ANGLE_PATTERN = AUTO_BRACKET_PATTERNS[0][1]
AUTO_BRACKET_BRACE_PATTERN = AUTO_BRACKET_PATTERNS[1][1]


# Void HTML elements — they never have a closing tag, so the
# paired-closer heuristic below would incorrectly treat them as
# unpaired "placeholders" and auto-bracket them even in real
# markup.
_VOID_HTML_TAG_NAMES = frozenset(
    {
        "area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr",
    }
)
# HTML element names that an opener like ``<name>`` should ALWAYS be
# treated as a real tag rather than a source-language placeholder.
# Markdown batching often splits ``<a> ... </a>`` across blocks, so a
# closer-search-in-same-text heuristic would tokenize an unpaired
# opener and corrupt cross-block markup on non-Latin runs.  Earlier
# revisions excluded ``title`` / ``label`` / ``button`` / ``form`` /
# ``code`` etc. on the rationale that authors might use those names
# as URL-path-parameter placeholders (``/docs/<title>``); review
# feedback at cycle 21 flagged that this freezes real HTML in
# cross-batch scenarios on ``ko`` / ``ja`` / ``ru`` / etc. runs.
# The trade-off is now resolved in favour of HTML safety: ASCII
# tag-name-shape spans match this set unconditionally.  Source-
# language placeholders ``<\uac8c\uc784\ucf54\ub4dc>``,
# ``<\u30d4\u30c3\u30c1>``, ``<\u3010\u9001\u4fe1\u3011>`` are
# unaffected — those don't match ``_TAG_NAME_EXTRACT_RE``'s
# ASCII-letter shape.  Authors who genuinely want a literal ASCII
# bracket placeholder (rare) should pin it via the glossary, which
# wins over auto-register.
_STRUCTURAL_HTML_TAG_NAMES = frozenset(
    {
        # Container / sectioning
        "div", "span", "p", "h1", "h2", "h3", "h4", "h5", "h6",
        "ul", "ol", "li", "dl", "dt", "dd",
        "table", "tr", "td", "th", "thead", "tbody", "tfoot",
        "caption", "colgroup",
        "html", "body", "head", "script", "style", "iframe",
        "article", "section", "header", "footer", "main", "nav",
        "aside", "figure", "figcaption",
        "blockquote", "pre",
        # Inline phrasing
        "strong", "em", "b", "i", "u", "s", "small", "sub", "sup",
        "ins", "del", "mark", "cite", "q", "bdi", "bdo", "ruby",
        "rb", "rp", "rt", "rtc",
        "a", "abbr", "address", "code", "dfn", "kbd", "samp", "var",
        # Forms / interactive
        "button", "datalist", "details", "dialog", "fieldset", "form",
        "label", "legend", "meter", "optgroup", "option", "output",
        "progress", "select", "summary", "template", "textarea",
        # Media / embedded
        "audio", "canvas", "map", "object", "picture", "video",
        # Document metadata that can appear inline in HTML-in-Markdown
        "noscript", "time", "title",
    }
)
_TAG_NAME_EXTRACT_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]*")


def _inside_html_open_tag(text: str, start: int, end: int) -> bool:
    """True when ``[start, end)`` belongs to an HTML opening tag.

    Covers two distinct shapes in one check:

    * STRICTLY INSIDE a tag body (``title="<\uc804\uc1a1>"``,
      ``aria-label={\uac8c\uc784\ucf54\ub4dc}``) — the outer tag is
      handled by ``html_attr`` (allowlisted attrs) or left
      translatable (user-facing attrs), so auto-bracket wedging
      itself inside would regress the existing contract.
    * SAME-SPAN as a tag that IS itself a real HTML / JSX tag
      (``<div>``, ``<MyComponent>``, ``<my-component>``) — for
      non-Latin-script targets (``ko``, ``ja``, ``ru``, ...) Latin
      content counts as non-target-script and would otherwise match
      the bare-tag span and freeze real HTML out of the translate /
      refine prompt.  A "real tag" here is ASCII tag-name-shape
      (``[A-Za-z][A-Za-z0-9\-]*`` fullmatch on the inner content,
      trimmed of a self-closing ``/`` and trailing whitespace).
      Mixed-script identifiers like ``<id_\uac8c\uc784\ucf54\ub4dc>``
      and digit-or-non-alpha-prefixed spans like ``<1\ub2e8\uacc4>``
      are NOT real HTML tags — their inner content fails the
      ASCII tag-name fullmatch — and stay protected.

    :data:`HTML_TAG_OPEN_RE` only matches tags that start with an
    ASCII letter (``<[A-Za-z]…>``), so the iteration never yields a
    span for CJK / Cyrillic / digit-prefixed bracket content, and
    the "real tag name" fullmatch on the inner text gates the same-
    span branch so mixed-script IDs keep matching.
    """
    for tag_m in HTML_TAG_OPEN_RE.finditer(text):
        if tag_m.start() < start and end <= tag_m.end():
            return True
        if tag_m.start() == start and tag_m.end() == end:
            inner = text[start + 1 : end - 1].rstrip("/").rstrip()
            # Reject only when the entire inner is a valid ASCII
            # tag-name shape.  Mixed-script inners (``<Submit\uac8c\uc784>``,
            # ``<id_\uac8c\uc784\ucf54\ub4dc>``) fail the fullmatch
            # and stay protected — those are placeholders, not
            # HTML/JSX tag names.
            if not _TAG_NAME_EXTRACT_RE.fullmatch(inner):
                continue
            # Void HTML elements (``<br>``, ``<img>``, ``<hr>``, ...)
            # reject unconditionally — they have no closer, so the
            # paired-closer heuristic below would otherwise treat
            # them as unpaired placeholders and mistakenly tokenize
            # real markup.  Structural container elements
            # (``<div>``, ``<span>``, ``<ul>``, ``<section>``, ...)
            # also reject unconditionally because batched Markdown
            # often separates the opener and closer across blocks;
            # the paired-closer check below would miss them and
            # freeze the real markup.
            inner_lower = inner.lower()
            if (
                inner_lower in _VOID_HTML_TAG_NAMES
                or inner_lower in _STRUCTURAL_HTML_TAG_NAMES
            ):
                return True
            # Every other ASCII-named span is ambiguous — could be
            # a real HTML / JSX tag pair OR a source-language
            # placeholder that happens to share a tag name
            # (``/docs/<title>``, ``<label>`` in prose,
            # ``<MyComponent>`` placeholder text, ...).  Reject only
            # when a matching ``</Name>`` closer appears AFTER this
            # opener in the same text — that is the distinguishing
            # signal of a real tag pair.  Scoping the search to
            # positions after ``end`` avoids misclassifying a prose
            # ``<Submit>`` whose document happens to mention
            # ``</Submit>`` in an earlier unrelated block.
            # HTML tag names are case-insensitive, so pair
            # ``<DIV>…</div>`` / ``<Foo>…</foo>`` correctly against
            # their lowercase / mixed-case closers.  JSX is case-
            # sensitive but also pairs (``<Foo>`` only pairs with
            # ``</Foo>``), and re.IGNORECASE preserves those pairs
            # too — the false-positive for a placeholder ``<Submit>``
            # whose document happens to contain a literal
            # ``</submit>`` elsewhere is vanishingly rare compared
            # to the common HTML case-folding shape.
            closer_re = re.compile(
                r"</\s*" + re.escape(inner) + r"\s*>",
                re.IGNORECASE,
            )
            if closer_re.search(text, end):
                return True
    return False


def auto_bracket_predicate_factory(
    glossary_terms: Optional[List[str]] = None,
) -> Callable[[str, int, int], bool]:
    """Build the match-time predicate for T-14 auto-bracket patterns.

    Composes three guards:

    * Reject matches that sit inside Markdown backtick code (inline
      or fenced) — same guard :data:`BUILTIN_PATTERNS` applies so
      documentation that illustrates bracket syntax literally (``Use
      `{한글}` here``) doesn't freeze the example.
    * Reject matches whose span sits anywhere inside an HTML opening
      tag — preserves the ``html_attr`` contract that translatable
      attributes (``title``, ``alt``, ``aria-label``, ``placeholder``,
      ``label``) keep flowing through the translate prompt, and
      covers both quoted (``title="<\uc804\uc1a1> \ubc84\ud2bc"``)
      and unquoted / JSX-style (``aria-label={\uac8c\uc784\ucf54\ub4dc}``)
      attribute value shapes in one check.  Source-language bracket
      tokens like ``<\uc804\uc1a1>`` that sit OUTSIDE any HTML tag
      (plain prose, between sibling tags) still match because
      :data:`HTML_TAG_OPEN_RE` only matches tags starting with an
      ASCII letter.
    * Reject matches whose span covers a glossary term — the
      caller-supplied glossary wins (mapped to its target-language
      form via decode, or preserved verbatim for null-entries), so
      the auto-pattern only fires when no glossary entry claims the
      span.  Terms that :meth:`MarkdownProcessor._compile_glossary_pattern`
      would itself reject (non-word-boundary terms such as ``.NET``)
      are excluded from the combined regex so they don't spuriously
      block auto-registration.

    ``glossary_terms=None`` or an empty list skips the glossary guard
    entirely — the returned predicate is then a pure inline-code +
    html-tag-body check, cheap enough to keep registered
    unconditionally.
    """
    glossary_re: Optional["re.Pattern[str]"] = None
    literal_defer_terms: List[str] = []
    if glossary_terms:
        safe_terms: List[str] = []
        for t in glossary_terms:
            if not t:
                continue
            if (t[0].isalnum() or t[0] == "_") and (
                t[-1].isalnum() or t[-1] == "_"
            ):
                safe_terms.append(t)
            else:
                # Terms like ``.NET`` / ``C++`` whose first / last
                # character isn't a word char fail ``\b`` anchors,
                # so ``_compile_glossary_pattern`` skips them and
                # the regex path misses them.  Keep them in a
                # literal-substring defer list so an explicit
                # mapping still blocks auto-bracket and the LLM
                # sees the raw term ready to apply the instruction-
                # mode mapping (cycle-20 P2 regression guard).
                literal_defer_terms.append(t)
        if safe_terms:
            # Longest-first keeps the alternation stable for overlapping
            # prefixes ("pull request" before "pull"); ``re`` alternation
            # is leftmost-first within a group so this matters for
            # ``.search`` against partial overlaps inside a bracket.
            safe_terms.sort(key=len, reverse=True)
            glossary_re = re.compile(
                r"\b(?:" + "|".join(re.escape(t) for t in safe_terms) + r")\b"
            )

    def predicate(text: str, start: int, end: int) -> bool:
        if _is_in_inline_code(text, start):
            return False
        if _inside_html_open_tag(text, start, end):
            return False
        if glossary_re is not None and glossary_re.search(text, start, end):
            return False
        if literal_defer_terms:
            span_text = text[start:end]
            for term in literal_defer_terms:
                if term in span_text:
                    return False
        return True

    return predicate


def format_token(index: int) -> str:
    """Render the Nth placeholder as the literal token string."""
    return f"\u27e6P:{index}\u27e7"


@dataclass(frozen=True)
class PlaceholderPattern:
    """A named regex whose matches become opaque tokens on ``encode``.

    ``predicate``, when supplied, receives ``(text, start, end)`` and
    must return ``True`` to keep the match as a candidate.  It lets a
    pattern restrict itself to a surrounding context that a vanilla
    regex can't express — the T-6 ``html_attr`` built-in uses this to
    only substitute attribute pairs that actually live inside a
    ``<...>`` tag, so attribute-like substrings in prose or inline
    code are left untouched.  ``None`` keeps the pre-T-6 behaviour —
    every match is a candidate.

    ``replace_builtin`` is the explicit opt-in to suppress a T-6
    built-in of the same name.  By default (``False``) a user pattern
    named ``anchor`` / ``html_attr`` layers ON TOP of the built-in so
    the "always-on / no opt-out" contract from the T-6 brief still
    protects spans the caller's regex or predicate rejects.  Callers
    who truly want to replace the default (stricter override with a
    narrower match set) set this flag explicitly so the decision is
    visible in code review.
    """

    name: str
    regex: RePattern[str]
    predicate: Optional[Callable[[str, int, int], bool]] = None
    replace_builtin: bool = False


@dataclass(frozen=True)
class Placeholder:
    """One substitution produced by :meth:`PlaceholderRegistry.encode`.

    ``replacement`` is an optional override consumed by
    :meth:`PlaceholderRegistry.decode`: when non-``None``, decode restores
    the token to this string instead of ``original``.  The glossary
    placeholder mode uses it to rewrite a matched source term to its
    target-language form on restore; other callers can leave it ``None``
    for a pure identity decode.
    """

    token: str
    original: str
    pattern_name: str
    replacement: Optional[str] = None


@dataclass
class PlaceholderMap:
    """Ordered record of substitutions made during ``encode``.

    Preserves insertion order so ``decode`` can restore tokens and so the
    round-trip check can enumerate expected tokens deterministically.
    """

    items: List[Placeholder] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __bool__(self) -> bool:
        return bool(self.items)

    def tokens(self) -> List[str]:
        return [p.token for p in self.items]


class PlaceholderRegistry:
    """Registry of named regex patterns to substitute with opaque tokens.

    An empty registry is a pass-through: ``encode`` returns the input
    unchanged with an empty :class:`PlaceholderMap`, so T-4 can ship the
    shared core without shipping any patterns.  Downstream tasks register
    the patterns they need.
    """

    def __init__(self) -> None:
        self._patterns: List[PlaceholderPattern] = []

    def register(
        self,
        name: str,
        regex: Union[str, RePattern[str]],
        *,
        flags: int = 0,
        predicate: Optional[Callable[[str, int, int], bool]] = None,
        replace_builtin: bool = False,
    ) -> None:
        """Register a pattern.

        ``regex`` may be a string (compiled with ``flags``) or an already-
        compiled ``re.Pattern`` (``flags`` is ignored in that case — bake
        flags into the compile step yourself).

        ``predicate``, when provided, filters each candidate match by its
        surrounding context (see :class:`PlaceholderPattern`).

        ``replace_builtin`` (default ``False``) is the explicit opt-in
        to suppress a T-6 built-in pattern of the same name; see
        :class:`PlaceholderPattern` for the rationale.
        """
        if isinstance(regex, re.Pattern):
            compiled = regex
        else:
            compiled = re.compile(regex, flags)
        self._patterns.append(
            PlaceholderPattern(
                name=name,
                regex=compiled,
                predicate=predicate,
                replace_builtin=replace_builtin,
            )
        )

    @property
    def patterns(self) -> Tuple[PlaceholderPattern, ...]:
        return tuple(self._patterns)

    def __len__(self) -> int:
        return len(self._patterns)

    def __bool__(self) -> bool:
        return bool(self._patterns)

    def encode(self, text: str) -> Tuple[str, PlaceholderMap]:
        """Replace every pattern match with an opaque ``\u27e6P:N\u27e7`` token.

        Returns ``(encoded_text, mapping)``.  With no patterns registered
        or no pattern / literal tokens found, returns the input unchanged
        and an empty mapping.

        Pre-existing ``\u27e6P:N\u27e7`` literals already present in ``text``
        (for example, documentation that explains the placeholder syntax)
        are recorded in the mapping as identity entries with
        ``pattern_name == "__literal__"``.  This guarantees two things:

            * the round-trip check expects them verbatim in the output
              (so the LLM must preserve them like any other token), and
            * newly generated tokens never reuse an index that a literal
              already occupies, which would otherwise cause ``decode`` to
              rewrite the literal into the wrong source span.

        Overlap resolution (applied across pattern candidates AND literal
        token spans):

            * Earliest start wins.
            * On an equal start, the longest match wins.
            * Remaining matches inside an already-chosen span are dropped.
            * Zero-width matches are ignored.
        """
        literal_matches = list(TOKEN_RE.finditer(text))
        if not self._patterns and not literal_matches:
            return text, PlaceholderMap()

        used_indices: set = {int(m.group(1)) for m in literal_matches}
        literal_spans: List[Tuple[int, int]] = [
            (m.start(), m.end()) for m in literal_matches
        ]

        def overlaps_literal(start: int, end: int) -> bool:
            for ls, le in literal_spans:
                if not (end <= ls or start >= le):
                    return True
            return False

        # Literals enter the candidate list first and CANNOT be absorbed
        # by registered patterns.  A greedy pattern that starts earlier
        # than (and spans through) a literal would otherwise swallow the
        # literal into its ``original`` text, so the round-trip check
        # would no longer require the literal to survive verbatim and
        # the model could silently drop or rewrite it.
        candidates: List[Tuple[int, int, str, str]] = []
        for m in literal_matches:
            candidates.append(
                (m.start(), m.end(), m.group(0), LITERAL_PATTERN_NAME)
            )
        for pat in self._patterns:
            for m in pat.regex.finditer(text):
                if m.start() == m.end():
                    continue
                if pat.predicate is not None and not pat.predicate(
                    text, m.start(), m.end()
                ):
                    continue
                if overlaps_literal(m.start(), m.end()):
                    continue
                candidates.append((m.start(), m.end(), m.group(0), pat.name))

        if not candidates:
            return text, PlaceholderMap()

        candidates.sort(key=lambda c: (c[0], -(c[1] - c[0])))

        chosen: List[Tuple[int, int, str, str]] = []
        cursor = 0
        for start, end, match_text, name in candidates:
            if start < cursor:
                continue
            chosen.append((start, end, match_text, name))
            cursor = end

        pieces: List[str] = []
        mapping = PlaceholderMap()
        prev = 0
        next_index = 0
        for start, end, match_text, name in chosen:
            pieces.append(text[prev:start])
            if name == LITERAL_PATTERN_NAME:
                # Preserve the literal in place — token equals its source
                # text — so encode is a structural no-op for this span.
                token = match_text
            else:
                while next_index in used_indices:
                    next_index += 1
                token = format_token(next_index)
                used_indices.add(next_index)
                next_index += 1
            mapping.items.append(
                Placeholder(token=token, original=match_text, pattern_name=name)
            )
            pieces.append(token)
            prev = end
        pieces.append(text[prev:])
        return "".join(pieces), mapping

    @staticmethod
    def decode(text: str, mapping: PlaceholderMap) -> str:
        """Restore every ``\u27e6P:N\u27e7`` token in ``text`` using ``mapping``.

        Each token is replaced with :attr:`Placeholder.replacement` when
        that field is set on the mapping entry, otherwise with
        :attr:`Placeholder.original`.  The replacement override lets
        callers map a matched source span to a different string on
        restore — the glossary placeholder path uses it to emit the
        target-language form instead of the original source term.

        Tokens absent from ``mapping`` (model hallucinated a token index)
        pass through unchanged so the round-trip check can surface them as
        ``unexpected`` rather than silently eating them.
        """
        if not mapping.items:
            return text
        lookup = {
            p.token: (p.replacement if p.replacement is not None else p.original)
            for p in mapping.items
        }

        def replace(m: "re.Match[str]") -> str:
            return lookup.get(m.group(0), m.group(0))

        return TOKEN_RE.sub(replace, text)


_PLACEHOLDER_RULE_FIELDS = ("name", "regex")


def load_placeholder_rules(path: Union[str, Path]) -> PlaceholderRegistry:
    """Load custom placeholder rules from a JSON file into a fresh
    :class:`PlaceholderRegistry`.

    The file MUST be a JSON array of rule objects, each shaped::

        {"name": "<identifier>", "regex": "<python re pattern>"}

    Both fields are required non-empty strings; any extra field is
    rejected so a typo such as ``"pattern"`` instead of ``"regex"``
    surfaces immediately rather than silently producing a no-op rule.
    Each ``regex`` is compiled with :mod:`re` eagerly so a malformed
    pattern fails the run before any LLM call.

    Empty files (``[]``) return an empty registry — a no-op equivalent
    to omitting the flag.

    Raises :class:`ValueError` with a single-line, actionable message
    on every error path (file unreadable, invalid JSON, top-level not
    a list, malformed entry, regex compile error). Callers that own a
    CLI surface should catch it and exit with the argparse convention
    code 2.
    """
    rules_path = Path(path)
    try:
        raw = rules_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(
            f"--placeholder-rules: cannot read {rules_path}: {exc}"
        ) from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"--placeholder-rules: invalid JSON in {rules_path}: {exc}"
        ) from exc

    if not isinstance(data, list):
        raise ValueError(
            "--placeholder-rules: top-level value must be a JSON array of "
            "rule objects"
        )

    allowed = set(_PLACEHOLDER_RULE_FIELDS)
    registry = PlaceholderRegistry()
    for index, entry in enumerate(data, start=1):
        if not isinstance(entry, dict):
            raise ValueError(
                f"--placeholder-rules entry {index}: must be a JSON object"
            )
        if "name" not in entry:
            raise ValueError(
                f"--placeholder-rules entry {index}: missing required "
                "field 'name'"
            )
        name = entry["name"]
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"--placeholder-rules entry {index}: 'name' must be a "
                "non-empty string"
            )
        if "regex" not in entry:
            raise ValueError(
                f"--placeholder-rules entry {index} (name={name}): "
                "missing required field 'regex'"
            )
        regex = entry["regex"]
        if not isinstance(regex, str):
            raise ValueError(
                f"--placeholder-rules entry {index} (name={name}): "
                "'regex' must be a string"
            )
        unknown = sorted(set(entry) - allowed)
        if unknown:
            raise ValueError(
                f"--placeholder-rules entry {index} (name={name}): "
                f"unknown field(s): {', '.join(unknown)}"
            )
        try:
            compiled = re.compile(regex)
        except re.error as exc:
            raise ValueError(
                f"--placeholder-rules entry {index} (name={name}): "
                f"regex compile failed: {exc}"
            ) from exc
        registry.register(name, compiled)
    return registry


def _anchor_positions(
    text: str,
    pattern: "re.Pattern[str]" = ANCHOR_PATTERN,
    predicate: Optional[Callable[[str, int, int], bool]] = None,
) -> Tuple[List[Tuple[str, int]], List[str]]:
    """Return ``(on_heading, off_heading)`` lists for anchors in ``text``.

    ``on_heading`` entries are ``(anchor, ordinal)`` pairs — the 1-based
    index of the heading line the anchor sits on.  ``off_heading`` is a
    plain list of anchor contents that appeared outside heading lines
    (span-level IAL in prose, lists, etc.).

    Both ATX headings (``#``–``######``) AND setext headings (a line of
    non-whitespace text followed by a ``===`` or ``---`` underline) are
    treated as heading lines — Pandoc / Kramdown both honour anchor
    attribute lists on setext headings, so the position check must
    too.  Without this, ``Overview {#overview}\\n========`` would
    classify its anchor as a prose anchor and let a model relocate
    it silently.
    """
    lines = text.splitlines()
    # Precompute line-offset -> line-start position so we can test
    # whether a "heading-looking" line actually lives inside a fenced
    # or indented code block.  Without that gate, a code example like
    # ``## fake`` inside ``` ``` ```…``` ``` would increment the
    # heading ordinal and cause false ``anchor_heading_drift`` reports
    # when the example text shifts between source and decoded.
    line_starts: List[int] = [0]
    for line in lines[:-1]:
        line_starts.append(line_starts[-1] + len(line) + 1)
    code_ranges = _find_code_ranges(text)

    def _line_in_code(idx: int) -> bool:
        if idx >= len(line_starts):
            return False
        pos = line_starts[idx]
        for s, e in code_ranges:
            if s <= pos < e:
                return True
        return False

    heading_ord_by_line: Dict[int, int] = {}
    ord_counter = 0
    for idx, line in enumerate(lines):
        if _line_in_code(idx):
            continue
        is_heading = bool(_HEADING_LINE_RE.match(line))
        if (
            not is_heading
            and line.strip()
            and idx + 1 < len(lines)
            and not _line_in_code(idx + 1)
            and _SETEXT_UNDERLINE_RE.match(lines[idx + 1])
        ):
            is_heading = True
        if is_heading:
            ord_counter += 1
            heading_ord_by_line[idx] = ord_counter

    def _enclosing_heading_ord(line_idx: int, match_text: str) -> Optional[int]:
        """Return the heading ordinal for the given anchor, or ``None``.

        Falls back to Kramdown / Pandoc block-IAL syntax — when a line
        contains ONLY the anchor (stripped) AND the IMMEDIATELY
        PRECEDING line is an ATX or setext heading, the anchor is
        treated as attached to that heading.  Kramdown requires the
        IAL on the directly following line; intervening blank lines
        break the association, so a ``## Title\\n\\n{#id}`` stays
        classified as off-heading.
        """
        if line_idx in heading_ord_by_line:
            return heading_ord_by_line[line_idx]
        if (
            line_idx < len(lines)
            and lines[line_idx].strip() == match_text
            and line_idx - 1 in heading_ord_by_line
        ):
            return heading_ord_by_line[line_idx - 1]
        return None

    on_heading: List[Tuple[str, int]] = []
    off_heading: List[str] = []
    for m in pattern.finditer(text):
        # When a user override predicate is supplied, defer entirely
        # to it — the override defines which spans it considers valid.
        # Default (no predicate) applies the built-in rule: skip
        # anchor syntax inside Markdown backticks so source and
        # decoded classifications agree with the encoder.
        if predicate is not None:
            if not predicate(text, m.start(), m.end()):
                continue
        elif _is_in_inline_code(text, m.start()):
            continue
        line_idx = text[: m.start()].count("\n")
        heading_ord = _enclosing_heading_ord(line_idx, m.group(0))
        if heading_ord is not None:
            on_heading.append((m.group(0), heading_ord))
        else:
            off_heading.append(m.group(0))
    return on_heading, off_heading


_TAG_NAME_RE = re.compile(r"^<([A-Za-z][\w:-]*)")


def _attr_tag_signatures(
    text: str,
    pattern: "re.Pattern[str]" = HTML_ATTR_PATTERN,
    predicate: Optional[Callable[[str, int, int], bool]] = None,
) -> List[Tuple[str, Tuple[str, ...]]]:
    """Return ``(tag_name, sorted_attrs)`` signatures for every opening
    tag that carries at least one protected attribute.

    The tag name is included so moves BETWEEN DIFFERENT ELEMENT TYPES
    that happen to share the same protected-attribute shape are still
    flagged — e.g. ``<img src="/a.png"><source src="/b.png">`` vs
    ``<img src="/b.png"><source src="/a.png">`` produces different
    signature multisets even though the raw attrs are permuted.  Tag
    names are lower-cased so ``<IMG>`` and ``<img>`` compare equal.

    Attribute matches nested inside a quoted value (``title='see
    href="/docs"'``) are filtered out by :func:`_in_quoted_value` so
    they don't leak into the tag's signature.

    Opening tags with NO protected attributes (``<strong>``,
    ``<span>``, ``<em>`` and similar accent wrappers) are skipped
    entirely — the check is about whether protected attributes stay
    attached to the right tag, not about preserving every tag in the
    document.  A translation that legitimately wraps the translated
    text in ``<strong>`` or unwraps a ``<span>`` should not be
    flagged when all the real ``href`` / ``class`` / ``id`` pairs
    still round-trip.

    Tag-ordinal positions are intentionally NOT captured here.  Cross-
    language translation legitimately reorders inline tags to fit
    target grammar (``<a href="/a">A</a> and <a href="/b">B</a>`` →
    ``<a href="/b">B</a>와 <a href="/a">A</a>``), and a structural
    check that pinned every attribute to its source ordinal would
    flag every such reorder as a regression.  Comparing the multiset
    of per-tag signatures instead lets reorders through while still
    catching the cases where an attribute crosses tag boundaries —
    e.g. two tags each with multiple attrs having ``class``-values
    swapped between them yields different signatures.
    """
    signatures: List[Tuple[str, Tuple[str, ...]]] = []
    for tag_m in HTML_TAG_OPEN_RE.finditer(text):
        # Tags that sit inside Markdown backticks are code-example
        # prose, not markup; skip to match the ``html_attr`` predicate
        # so source and decoded signatures agree in both places.
        if _is_in_inline_code(text, tag_m.start()):
            continue
        tag_text = tag_m.group(0)
        attrs: List[str] = []
        for attr_m in pattern.finditer(tag_text):
            abs_start = tag_m.start() + attr_m.start()
            abs_end = tag_m.start() + attr_m.end()
            # User override predicate takes precedence when supplied —
            # it defines the full "is a protected attr?" rule.  The
            # default (no predicate) applies the built-in quote-aware
            # filter so source and decoded signatures agree with the
            # encoder's output.
            if predicate is not None:
                if not predicate(text, abs_start, abs_end):
                    continue
            elif _in_quoted_value(tag_text, attr_m.start()):
                continue
            attrs.append(attr_m.group(0))
        if not attrs:
            continue
        name_m = _TAG_NAME_RE.match(tag_text)
        tag_name = name_m.group(1).lower() if name_m else ""
        signatures.append((tag_name, tuple(sorted(attrs))))
    return signatures


def check_structural_position(
    source: str,
    decoded: str,
    *,
    check_anchor: bool = True,
    check_html_attr: bool = True,
    anchor_pattern: "re.Pattern[str]" = ANCHOR_PATTERN,
    html_attr_pattern: "re.Pattern[str]" = HTML_ATTR_PATTERN,
    anchor_predicate: Optional[Callable[[str, int, int], bool]] = None,
    html_attr_predicate: Optional[Callable[[str, int, int], bool]] = None,
) -> Optional[str]:
    """Detect built-in placeholder tokens that moved out of context.

    :func:`check_round_trip` compares token multisets but is blind to
    placement — a model that preserves ``\u27e6P:N\u27e7`` exactly once
    but relocates it (anchor slides from a heading into the following
    paragraph, attribute pair jumps to a neighbouring tag) would
    otherwise pass validation and ``decode`` would restore the protected
    span in the wrong spot.

    The check compares structural placement of the T-6 built-in patterns
    between the original source and the decoded translation:

    * anchors on ATX heading lines are pinned by ``(content,
      heading_ordinal)`` — a multiset mismatch catches anchor-swap
      between headings even when the total anchor count is unchanged.
    * anchors off heading lines (span-level IAL in prose) are compared
      by content multiset only — prose legitimately reorders across
      languages, so ordinals aren't enforced.
    * HTML attributes inside tags are compared by the multiset of
      per-tag signatures (see :func:`_attr_tag_signatures`).  Ordinals
      are intentionally NOT pinned because cross-language translations
      legitimately reorder inline tags to fit target grammar.  The
      signature multiset still catches attributes that crossed tag
      boundaries — e.g. two tags with distinct attr sets swap attrs —
      which is the structural regression we actually care about.

    Any drift in those multisets is a structural fail.  Returns ``None``
    on success or a short human-readable reason on failure.

    The ``check_anchor`` / ``check_html_attr`` flags let a caller
    disable a sub-check entirely if the built-in concept does not
    apply (for instance, a caller who replaced ``html_attr`` with a
    pattern that covers non-HTML structures can turn the HTML sub-
    check off).  ``anchor_pattern`` / ``html_attr_pattern`` let a
    caller substitute the regex used to locate matches — useful when
    the caller overrode the corresponding built-in under the same
    name, so the structural check still runs against the exact spans
    their override tokenized and keeps the structural-safety guarantee
    for custom configurations.
    """
    problems: List[str] = []

    if check_anchor:
        src_head, src_off = _anchor_positions(
            source, anchor_pattern, anchor_predicate
        )
        dec_head, dec_off = _anchor_positions(
            decoded, anchor_pattern, anchor_predicate
        )
        if Counter(src_head) != Counter(dec_head):
            problems.append("anchor_heading_drift")
        if Counter(src_off) != Counter(dec_off):
            problems.append("anchor_offheading_count")

    if check_html_attr:
        src_sigs = _attr_tag_signatures(
            source, html_attr_pattern, html_attr_predicate
        )
        dec_sigs = _attr_tag_signatures(
            decoded, html_attr_pattern, html_attr_predicate
        )
        if Counter(src_sigs) != Counter(dec_sigs):
            problems.append("html_attr_tag_drift")

    return "; ".join(problems) if problems else None


def check_round_trip(
    text: str, mapping: PlaceholderMap
) -> Optional[str]:
    """Verify the multiset of tokens in ``mapping`` matches ``text``.

    Returns ``None`` on success or a short human-readable reason on
    failure.  Detects three failure modes:

    * ``missing`` — a mapped token appears fewer times than expected.
    * ``duplicated`` — a mapped token appears more times than expected.
    * ``unexpected`` — a token of the correct shape appears that was not
      in the mapping (model fabricated a token index).

    Multiset counts rather than a flat set are needed because pre-existing
    ``\u27e6P:N\u27e7`` literals in the source get recorded as identity
    entries in the mapping, and two copies of the same literal in the
    source legitimately require two copies in the output.
    """
    if not mapping.items:
        return None

    expected_count: Counter = Counter(p.token for p in mapping.items)
    actual_count: Counter = Counter(
        m.group(0) for m in TOKEN_RE.finditer(text)
    )

    missing: List[str] = []
    duplicated: List[str] = []
    for token, exp in expected_count.items():
        got = actual_count.get(token, 0)
        if got < exp:
            suffix = f"x{exp - got}" if exp - got > 1 else ""
            missing.append(f"{token}{suffix}")
        elif got > exp:
            duplicated.append(f"{token}x{got}")

    extras: List[str] = []
    seen: set = set()
    for token, got in actual_count.items():
        if token in expected_count:
            continue
        if token in seen:
            continue
        seen.add(token)
        extras.append(token)

    problems: List[str] = []
    if missing:
        problems.append(f"missing={','.join(missing)}")
    if duplicated:
        problems.append(f"duplicated={','.join(duplicated)}")
    if extras:
        problems.append(f"unexpected={','.join(extras)}")

    return "; ".join(problems) if problems else None
