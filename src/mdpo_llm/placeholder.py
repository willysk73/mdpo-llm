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

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Pattern as RePattern, Tuple, Union

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
