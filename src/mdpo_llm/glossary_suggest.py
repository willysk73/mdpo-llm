"""Auto-glossary candidate extraction from a source corpus (T-23).

``mdpo-llm suggest-glossary <source_dir> --target ko,ja,zh-CN`` walks a
directory of source markdown files, finds high-frequency
proper-noun-like tokens (brand names, product surfaces, acronyms),
clusters near-duplicate variants (``"WCS"`` / ``"WCS API"`` /
``"WCS dashboard"``), translates the canonical form via a bulk LLM
call, and emits a draft ``glossary.suggested.json`` the operator can
review and promote into a real ``glossary.json``.

The extraction / clustering algorithm is built around mdpo-llm's
source-corpus model:

  1. Walk the source tree for ``*.md`` files; for each file, strip the
     markdown surfaces (fenced + indented code, inline code, URLs,
     numeric runs) so token extraction sees only translatable prose.
  2. Tokenize each cleaned body into proper-noun-like word-shape
     candidates: single-word (UpperCamel, ALL_CAPS acronyms, mixed
     ``CamelCase`` like ``GitHub``) and 2- to 3-word phrases whose
     every token is itself proper-noun-like.
  3. Compute the per-token frequency histogram and the set of files
     each token appears in. Keep tokens with ``count >=
     min_occurrences`` AND ``files >= min_files``.
  4. Cluster near-duplicates via :func:`difflib.SequenceMatcher`: two
     tokens collapse into one cluster when their ratio is at or above
     ``similarity_threshold`` OR when one is a multi-word phrase that
     contains the other as a whole-word substring. Cluster picks its
     canonical form as the most-frequent variant (ties broken by
     longer string, then lexicographically) — this preserves
     ``"WCS API"`` / ``"WCS dashboard"`` as a cluster anchored on the
     most-used variant rather than collapsing them onto the shorter
     stem.
  5. Translate the canonical form of every cluster via a bulk LLM
     call. Translation is decoupled behind the :data:`BulkTranslator`
     callable so tests inject deterministic stubs; the default
     :func:`litellm_bulk_translator` routes through ``litellm`` so the
     model-string contract (OpenRouter, Anthropic, Bedrock, …) keeps
     working without a separate API client.
  6. Emit a JSON object keyed by canonical term, mapping to a
     per-locale translation dict, sorted by frequency descending then
     by canonical term ascending. The output schema is the SAME schema
     :class:`mdpo_llm.processor.MarkdownProcessor` already consumes
     via ``glossary_path=`` so the operator can promote a vetted
     ``glossary.suggested.json`` by copying it to ``glossary.json``.

The draft is written to ``<source_dir>/glossary.suggested.json`` by
default (configurable via ``--output``). The verb refuses to write to
a file whose basename is exactly ``glossary.json`` — promotion is a
manual review step by design, per the project direction.

Out of scope here:
  * Real LLM calls in tests. The translator callable is injectable
    precisely so tests pass a deterministic stub.
  * Overwriting an authored ``glossary.json``. The default output
    name is distinct (``glossary.suggested.json``); the hard refusal
    on the ``glossary.json`` basename catches the case where the
    operator passes ``--output glossary.json`` either out of habit or
    via shell expansion.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import litellm


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunables.
# ---------------------------------------------------------------------------

# Lower-cased markdown extensions the source walk recognises. Matches
# the set every other mdpo-llm verb scans by default; keeping the
# tuple explicit and module-public so library callers can extend it
# without monkey-patching the walker.
MARKDOWN_EXTENSIONS: Tuple[str, ...] = (".md", ".markdown")


# Default similarity threshold for SequenceMatcher-based clustering.
# 0.85 collapses common variant pairs (``"WCS"`` / ``"WCS API"``) via
# the whole-word substring rule but does not over-merge unrelated
# proper nouns that happen to share a few letters
# (``"GitHub"`` / ``"GitLab"`` ratio is below this floor). Operators
# who want a tighter / looser merge can override via the CLI flag.
DEFAULT_SIMILARITY_THRESHOLD = 0.85


# Default frequency / file-count thresholds. Picked low enough that a
# small fixture corpus surfaces useful candidates in tests but high
# enough that a casual single-mention of a foreign-language phrase
# does not pollute the draft. Override via ``--min-occurrences`` /
# ``--min-files``.
DEFAULT_MIN_OCCURRENCES = 3
DEFAULT_MIN_FILES = 2


# Cap on phrase length (number of whitespace-separated tokens) for
# multi-word candidates. Capped at 3: higher arities mostly surface
# fragments of full sentences with low cluster value.
_MAX_PHRASE_TOKENS = 3


# Output file basename guaranteed never to overwrite an authored
# ``glossary.json``. The walker / CLI both default to writing here so
# the operator's review pass is a one-step ``mv glossary.suggested.json
# glossary.json`` after a content check.
SUGGESTED_GLOSSARY_FILENAME = "glossary.suggested.json"


# Filename the verb hard-refuses to overwrite, regardless of whether
# it currently exists. Promotion is a manual review step by design.
AUTHORED_GLOSSARY_FILENAME = "glossary.json"


# ---------------------------------------------------------------------------
# Markdown cleaning.
# ---------------------------------------------------------------------------

# Fenced-code opening line: 3+ consecutive backticks or tildes at
# the start of a line, indented **no more than three spaces** (per
# CommonMark §4.5). A 4+-space (or tab-led) line that happens to
# start with backticks is an INDENTED code line, not a fenced-code
# opener; treating it as a fence would consume every following line
# until the next matching fence or EOF and silently drop legitimate
# prose. The closing rule is enforced by :func:`_strip_fenced_code`
# rather than a regex backreference because CommonMark allows the
# closing fence to be longer than the opening fence (opening
# ```` ``` ```` closed by ```` ```` ```` is valid); a ``\1``
# backreference would only accept exact-length closures and let the
# body of a longer-closed fence leak into the candidate pool.
_FENCE_OPEN_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")


# Inline code: balanced single- or multi-backtick run on one line. The
# ``[^\n]+?`` body refuses newlines so a stray backtick in prose does
# not span paragraphs. Lazy quantifier keeps the match tight.
_INLINE_CODE_RE = re.compile(r"`+[^`\n]+?`+")


# Indented code block: a four-space (or one-tab) leading indent at the
# start of a line marks the line as code per CommonMark. Stripping
# whole lines is safer than per-token rules — markdown's indented-code
# detection is order-sensitive, and rebuilding it just to suppress a
# few false positives would balloon the cleanup pass.
_INDENTED_CODE_RE = re.compile(r"^(?: {4,}|\t)[^\n]*$", re.MULTILINE)


# URL / autolink: anything containing ``://`` plus angle-bracketed
# autolinks. URLs frequently embed camelCase identifiers
# (``GitHubAction``, ``WCSGateway``) that would otherwise pass the
# token shape filter — the verb is meant to surface PROSE
# terminology, not URL components.
_URL_RE = re.compile(r"<[^\s<>]+>|\b\w+://[^\s<>]+")


# HTML attribute and tag bodies — markdown allows raw HTML, and
# attribute values can carry brand strings (``alt="GitHub Logo"``)
# that the operator never types in prose. Stripping whole tag bodies
# is a coarser cut than the placeholder.py HTML-attr protection used
# in the translation pipeline, but the goal here is candidate
# extraction; we err on the side of NOT surfacing strings the operator
# would have to manually filter out.
_HTML_TAG_RE = re.compile(r"<[^<>\n]+>")


# Markdown image construct: ``![alt](href)``. Stripped wholesale so
# the alt text never reaches the candidate pool — image alt strings
# are frequently filename-like or descriptive captions that would
# otherwise pollute the histogram with one-off proper-noun
# look-alikes (``"Logo"``, ``"Diagram"``, ``"Screenshot"``).
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)\n]*\)")


# Normal Markdown inline link: ``[label](href)``. Stripped wholesale
# so navigation / reference link text (``[See the docs](url)``,
# ``[mdpo-llm cleanup](#section)``) does not dominate the candidate
# pool in link-heavy docs. The negative lookbehind ``(?<!!)``
# protects images from double-matching (the :data:`_IMAGE_RE` pass
# already handles them). Brand names that only ever appear inside
# link labels (e.g. ``[GitHub](https://github.com)``) will not be
# counted; the operator gets a marginally smaller candidate pool in
# exchange for not having navigation chrome inflate the histogram.
# If the brand also appears in body prose, it still surfaces via the
# normal token pass — which is the common case.
_LINK_RE = re.compile(r"(?<!!)\[[^\]\n]*\]\([^)\n]*\)")


# Markdown reference-style link: ``[label][ref]`` and ``[label][]``.
# Stripped so a reference-link author's bracket text does not leak
# into the candidate pool either. Less common than the inline form
# but trivial to handle correctly here rather than relying on
# downstream filters.
_REF_LINK_RE = re.compile(r"\[[^\]\n]*\]\[[^\]\n]*\]")


# Markdown reference-link DEFINITIONS at the start of a line:
# ``[label]: https://… "optional title"``. The label is documentation
# metadata, not body prose; strip the whole line so the (often
# proper-noun-shaped) label slug does not surface as a candidate.
# Matched with ``re.MULTILINE`` so every reference definition in a
# multi-paragraph doc is caught.
_LINK_DEF_RE = re.compile(
    r"^[ \t]*\[[^\]\n]+\]:[ \t]+\S+(?:[ \t]+\"[^\"\n]*\")?[ \t]*$",
    re.MULTILINE,
)


# Numeric runs: pure digits, digits-with-separators (``1,000``,
# ``1.5``), version strings (``v1.2.3``). Removing them prevents
# noise like ``"2024"`` from dominating the histogram of a date-heavy
# tree.
_NUMERIC_RE = re.compile(r"\b[vV]?\d+(?:[.,]\d+)*\b")


def _strip_fenced_code(text: str) -> str:
    """Strip CommonMark fenced code blocks line-by-line.

    Per CommonMark, an opening fence is a run of 3+ consecutive
    backticks OR tildes at the start of an optionally-indented line;
    the closing fence is a run of the **same character** with
    **length >= opening's length**, on its own (optionally-indented,
    optionally-whitespace-padded) line. Unbalanced fences (no
    closing) consume to end-of-text, mirroring how renderers degrade.

    A regex with a ``\\1`` backreference cannot enforce "length >="
    correctly (it would require exact length); this line-walker
    captures both the length and the character so the rule is
    enforced precisely. The opening fence line is replaced with a
    single space so token boundaries on either side of the removed
    region are preserved; body lines are dropped entirely.
    """
    lines = text.split("\n")
    out: List[str] = []
    i = 0
    n = len(lines)
    while i < n:
        m = _FENCE_OPEN_RE.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        fence = m.group(1)
        char = fence[0]
        length = len(fence)
        # Closing fence: same character, length >= opening, indented
        # no more than three spaces (per CommonMark §4.5). Trailing
        # whitespace before EOL is allowed but no info string.
        close_re = re.compile(
            rf"^ {{0,3}}{re.escape(char)}{{{length},}}[ \t]*$"
        )
        # Replace the whole fence (opener + body + closer) with one
        # space so the candidate prose on either side stays
        # word-bounded.
        out.append(" ")
        j = i + 1
        while j < n and not close_re.match(lines[j]):
            j += 1
        # Skip the closing line (when present) too; an unbalanced
        # fence (no close) consumes to EOF.
        i = j + 1
    return "\n".join(out)


def _strip_markdown_surfaces(text: str) -> str:
    """Remove code / link / URL / HTML / numeric surfaces from ``text``.

    Order matters: fenced code first (so its body never reaches the
    inline-code or URL regex), then indented code, then markdown
    reference-link DEFINITIONS (line-anchored, so a definition that
    happens to share a slug with body prose still strips), then
    image and link constructs (so their URL halves never reach the
    URL stripper), then HTML tags, then inline code, then URLs, and
    finally numeric runs. Anything not stripped is treated as
    candidate prose by :func:`extract_tokens`.

    Returns a string with the matched regions replaced by a single
    space so token boundaries on either side of the removed region
    are preserved (otherwise ``"WCS`API`gateway"`` would collapse
    into ``"WCSgateway"`` and the candidate would never surface).
    """
    cleaned = _strip_fenced_code(text)
    cleaned = _INDENTED_CODE_RE.sub(" ", cleaned)
    cleaned = _LINK_DEF_RE.sub(" ", cleaned)
    cleaned = _IMAGE_RE.sub(" ", cleaned)
    cleaned = _LINK_RE.sub(" ", cleaned)
    cleaned = _REF_LINK_RE.sub(" ", cleaned)
    cleaned = _HTML_TAG_RE.sub(" ", cleaned)
    cleaned = _INLINE_CODE_RE.sub(" ", cleaned)
    cleaned = _URL_RE.sub(" ", cleaned)
    cleaned = _NUMERIC_RE.sub(" ", cleaned)
    return cleaned


# ---------------------------------------------------------------------------
# Token shape detection.
# ---------------------------------------------------------------------------

# A "word" for tokenization: letters (any Unicode letter) plus digits,
# joined by an optional internal hyphen / dot so ``A.I.`` and
# ``state-of-the-art`` survive. Edge punctuation is stripped by the
# caller before the shape check.
_WORD_RE = re.compile(r"[^\W_]+(?:[.\-][^\W_]+)*", re.UNICODE)


# Proper-noun shapes the candidate filter accepts. Listed explicitly
# so each shape is independently testable:
#
#   * ALL_CAPS acronym: ``WCS``, ``API``, ``HTTP``. Minimum two chars
#     so single-letter pronouns ("I", "A") do not pollute the pool.
#   * TitleCase: leading uppercase + lowercase-only tail. Matches
#     ``Markdown``, ``Korea``, ``Anthropic``. The phrase-builder uses
#     this shape as the unit of a multi-word phrase.
#   * CamelCase / mixed-case: ASCII letters and digits with at least
#     one uppercase AND at least one lowercase letter and the shape
#     does NOT already match the two narrower rules above. Captures
#     ``GitHub``, ``MacBook``, ``iPhone``, ``OAuth``, ``MdpoLLM``,
#     ``iOS``. The mixed-case predicate is evaluated in
#     :func:`_is_proper_noun_shape` after the narrower rules because
#     a narrow regex (e.g. ``^[A-Z][a-z0-9]+[A-Z]...$``) cannot accept
#     OAuth-style leading-double-uppercase shapes without false
#     positives on ALL_CAPS / TITLE_CASE words.
_ALL_CAPS_RE = re.compile(r"^[A-Z][A-Z0-9]+$")
_TITLE_CASE_RE = re.compile(r"^[A-Z][a-z0-9]+$")
_CAMEL_CASE_CHAR_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]*$")


# Common English stopwords that pass the TitleCase filter at the
# start of a sentence ("The", "This", "When"). Filtering them at the
# single-word level keeps the histogram useful; the phrase builder
# also drops them because a multi-word phrase starting with
# ``"The"`` is rarely the proper noun the operator wants. The
# phrase-continuation rule also rejects the lowercase forms (``"the"``,
# ``"is"``, ``"and"``) so noise phrases like ``"WCS is the"`` never
# materialise — the lookup is case-insensitive via
# :data:`_STOPWORDS_LOWER`.
_STOPWORDS: FrozenSet[str] = frozenset(
    {
        "A",
        "An",
        "The",
        "This",
        "That",
        "These",
        "Those",
        "It",
        "Its",
        "Is",
        "Are",
        "Was",
        "Were",
        "Be",
        "Been",
        "Being",
        "And",
        "Or",
        "But",
        "If",
        "When",
        "While",
        "Where",
        "How",
        "Why",
        "What",
        "Who",
        "Whom",
        "Whose",
        "Which",
        "Then",
        "So",
        "Because",
        "Although",
        "Though",
        "Since",
        "Until",
        "After",
        "Before",
        "During",
        "About",
        "Above",
        "Below",
        "From",
        "To",
        "For",
        "Of",
        "On",
        "In",
        "At",
        "By",
        "With",
        "Without",
        "Within",
        "Into",
        "Onto",
        "Upon",
        "Over",
        "Under",
        "We",
        "You",
        "He",
        "She",
        "They",
        "I",
        "Our",
        "Your",
        "His",
        "Her",
        "Their",
        "My",
        "Me",
        "Him",
        "Them",
        "Us",
        "All",
        "Some",
        "Any",
        "Each",
        "Every",
        "Most",
        "Many",
        "Much",
        "Few",
        "Several",
        "Other",
        "Another",
        "Such",
        "No",
        "Not",
        "Yes",
        "Only",
        "Also",
        "Just",
        "Even",
        "Still",
        "Always",
        "Never",
        "Often",
        "Sometimes",
        "Now",
        "Today",
        "Tomorrow",
        "Yesterday",
    }
)


# Lowercased view of the stopword set so the phrase-continuation rule
# can reject ``"the"`` / ``"is"`` / ``"and"`` without doubling every
# entry in :data:`_STOPWORDS`. Built once at import time.
_STOPWORDS_LOWER: FrozenSet[str] = frozenset(s.lower() for s in _STOPWORDS)


# Minimum length of a lowercase common-noun continuation token. Short
# function words (``"a"``, ``"is"``, ``"of"``, ``"to"``) would otherwise
# slip through the stopword filter when callers extend the corpus with
# tech jargon that uses two-letter abbreviations like ``"io"``; the
# floor is set conservatively low so legitimate common nouns
# (``"api"``, ``"sdk"``, when authors write them lowercase) still
# qualify.
_MIN_LOWERCASE_CONTINUATION_LEN = 3


def _is_proper_noun_shape(token: str) -> bool:
    """Return ``True`` when ``token`` looks like a proper-noun candidate.

    Accepts the three shapes documented above (ALL_CAPS, TitleCase,
    mixed-case CamelCase). Stopwords are rejected even when their
    shape matches because ``"The"`` / ``"When"`` would otherwise
    dominate any English corpus's frequency histogram.

    The mixed-case CamelCase predicate is evaluated last and only
    fires when the token has BOTH at least one uppercase AND at least
    one lowercase letter. That single rule captures every
    upper-prefix CamelCase shape (``OAuth``, ``iOS``) plus the
    classic prefix-then-suffix forms (``GitHub``, ``MacBook``,
    ``iPhone``, ``MdpoLLM``) without false positives on pure
    TitleCase / pure ALL_CAPS strings (which the narrower regexes
    already accepted).
    """
    if token in _STOPWORDS:
        return False
    if _ALL_CAPS_RE.match(token):
        return True
    if _TITLE_CASE_RE.match(token):
        return True
    if _CAMEL_CASE_CHAR_RE.match(token):
        has_upper = any(c.isupper() for c in token)
        has_lower = any(c.islower() for c in token)
        if has_upper and has_lower:
            return True
    return False


# Lowercase common-noun continuation: all-lowercase letters (Unicode
# included), at least :data:`_MIN_LOWERCASE_CONTINUATION_LEN` long,
# not a stopword. Used by the phrase walker to extend a proper-noun
# stem into a domain phrase (``"WCS dashboard"``, ``"API gateway"``)
# without admitting random lowercase prose.
def _is_lowercase_continuation(token: str) -> bool:
    """Return ``True`` when ``token`` is a lowercase common-noun continuation.

    The phrase walker uses this rule for non-leading positions only.
    Leading positions must be proper-noun-shaped because the whole
    point of the verb is to surface brand / acronym / product
    surfaces; a lowercase-leading phrase carries no proper-noun
    signal.
    """
    if len(token) < _MIN_LOWERCASE_CONTINUATION_LEN:
        return False
    if token.lower() in _STOPWORDS_LOWER:
        return False
    # All Unicode letters, no digits or punctuation. ``isalpha`` is
    # locale-independent in Python 3 and aligned with how the rest of
    # the pipeline handles non-ASCII text.
    if not token.isalpha():
        return False
    if not token.islower():
        return False
    return True


# Characters allowed between two words of the same phrase candidate.
# Strictly horizontal whitespace — a newline, period, comma, colon,
# semicolon, paren, or any other character in the gap means the words
# are NOT contiguous in prose and must not form a phrase candidate.
# Without this guard, ``"WCS. GitHub"`` produces a high-frequency
# ``"WCS GitHub"`` phrase that the clusterer would then ship to the
# LLM as a bogus canonical.
_PHRASE_GAP_CHARS = frozenset(" \t")


def extract_tokens(text: str) -> List[str]:
    """Return every proper-noun-like single-word and phrase in ``text``.

    The list preserves multiplicity (so the caller can build a
    frequency histogram by counting occurrences) and is order-stable
    relative to the cleaned text: single-word candidates appear at
    their first hit, phrase candidates follow at the position of the
    phrase's first word. Stable order matters in tests asserting
    deterministic output.

    Phrase rule: a phrase is a run of 2 to :data:`_MAX_PHRASE_TOKENS`
    words that are **contiguous in prose** (the gap between
    successive words contains only :data:`_PHRASE_GAP_CHARS` —
    horizontal whitespace), where the leading word is
    proper-noun-shaped and every following word is either
    proper-noun-shaped (so ``"WCS API"``, ``"WCS API gateway"``
    qualify) OR a lowercase common-noun continuation (so
    ``"WCS dashboard"``, ``"API gateway"`` qualify). The
    contiguity requirement prevents phrases from spanning sentence,
    paragraph, or punctuation boundaries (``"WCS. GitHub"`` no
    longer materialises as ``"WCS GitHub"``). The lowercase floor in
    :data:`_MIN_LOWERCASE_CONTINUATION_LEN` plus the stopword filter
    keep noise phrases like ``"WCS is the"`` out of the pool even
    within a single clause.
    """
    cleaned = _strip_markdown_surfaces(text)
    matches: List[Tuple[str, int, int]] = [
        (m.group(0), m.start(), m.end())
        for m in _WORD_RE.finditer(cleaned)
    ]
    tokens: List[str] = []
    # Single-word pass: proper-noun-shaped, non-stopword words.
    for word, _, _ in matches:
        if _is_proper_noun_shape(word):
            tokens.append(word)
    # Multi-word pass: 2..MAX_PHRASE_TOKENS contiguous-in-prose words.
    # The contiguity check looks at the actual characters between
    # match spans and breaks the phrase walk on the first non-
    # whitespace gap.
    n = len(matches)
    for start in range(n):
        if not _is_proper_noun_shape(matches[start][0]):
            continue
        max_end = min(start + _MAX_PHRASE_TOKENS, n)
        prev_end = matches[start][2]
        for end in range(start + 2, max_end + 1):
            nxt_text, nxt_start, nxt_end = matches[end - 1]
            gap = cleaned[prev_end:nxt_start]
            if not gap or any(c not in _PHRASE_GAP_CHARS for c in gap):
                break
            if not (
                _is_proper_noun_shape(nxt_text)
                or _is_lowercase_continuation(nxt_text)
            ):
                break
            phrase = " ".join(w for w, _, _ in matches[start:end])
            tokens.append(phrase)
            prev_end = nxt_end
    return tokens


# ---------------------------------------------------------------------------
# Data classes.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TokenCandidate:
    """Single candidate token / phrase with its corpus coverage.

    ``text`` is the verbatim form as it appears in the source corpus.
    ``count`` is the total number of occurrences across the whole
    walk. ``files`` is the lowercased-relative-path set of source
    files in which the token appears at least once.
    """

    text: str
    count: int
    files: FrozenSet[str]


@dataclass(frozen=True)
class GlossaryCluster:
    """A near-duplicate cluster of token candidates.

    ``canonical`` is the cluster's chosen representative — the variant
    with the highest count, with longer string winning ties, then
    lexicographic order. ``variants`` lists every member text
    (including the canonical) sorted by descending count then
    lexicographic ascending so the JSON payload is deterministic.
    ``total_count`` and ``files`` aggregate across every member.
    """

    canonical: str
    variants: Tuple[str, ...]
    total_count: int
    files: FrozenSet[str]


@dataclass(frozen=True)
class GlossarySuggestion:
    """One translated cluster ready to render into ``glossary.suggested.json``.

    ``translations`` maps each requested target locale to the LLM's
    best guess. Locales the LLM did not return for this entry are
    represented as empty strings — the operator's review pass spots
    them and either accepts (preserve verbatim) or fills them in.
    """

    canonical: str
    translations: Mapping[str, str]
    count: int
    files: Tuple[str, ...]
    variants: Tuple[str, ...]


# ---------------------------------------------------------------------------
# Corpus walking + histogram.
# ---------------------------------------------------------------------------


def _iter_markdown_files(source_dir: Path) -> List[Path]:
    """Return sorted markdown files under ``source_dir``.

    Sorted by relative path so JSON outputs are byte-stable across
    platforms with different walk ordering. Symlink loops are avoided
    by ``rglob``'s default refusal to descend through cycles; broken
    symlinks raise ``OSError`` on ``is_file()`` and are skipped.
    """
    files: List[Path] = []
    for path in sorted(source_dir.rglob("*"), key=lambda p: str(p)):
        try:
            if not path.is_file():
                continue
        except OSError:
            continue
        if path.suffix.lower() in MARKDOWN_EXTENSIONS:
            files.append(path)
    return files


def collect_candidates(
    source_dir: Path,
) -> Dict[str, TokenCandidate]:
    """Build a per-token frequency histogram over the source corpus.

    Returns a mapping ``{token_text: TokenCandidate}``. Files that
    fail to decode as UTF-8 are skipped silently — a single mis-encoded
    file MUST NOT abort the whole walk; the gap is recoverable by
    re-running after the operator fixes the encoding.
    """
    counts: Dict[str, int] = {}
    file_sets: Dict[str, set[str]] = {}
    for md_path in _iter_markdown_files(source_dir):
        try:
            body = md_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        rel = str(md_path.relative_to(source_dir))
        seen_in_file: set[str] = set()
        for token in extract_tokens(body):
            counts[token] = counts.get(token, 0) + 1
            seen_in_file.add(token)
        for token in seen_in_file:
            file_sets.setdefault(token, set()).add(rel)
    return {
        text: TokenCandidate(
            text=text,
            count=counts[text],
            files=frozenset(file_sets.get(text, set())),
        )
        for text in counts
    }


def filter_by_thresholds(
    candidates: Mapping[str, TokenCandidate],
    *,
    min_occurrences: int,
    min_files: int,
) -> List[TokenCandidate]:
    """Return candidates passing both occurrence and file-count gates.

    Sort order: descending count, then descending number of files,
    then descending length, then lexicographic ascending — the same
    tie-breakers the clusterer uses for the canonical pick, so
    debugging the histogram is straightforward (the top of the list
    is the next cluster's canonical).
    """
    kept = [
        c
        for c in candidates.values()
        if c.count >= min_occurrences and len(c.files) >= min_files
    ]
    kept.sort(
        key=lambda c: (
            -c.count,
            -len(c.files),
            -len(c.text),
            c.text,
        )
    )
    return kept


# ---------------------------------------------------------------------------
# Clustering.
# ---------------------------------------------------------------------------


def _similar(a: str, b: str) -> float:
    """Return the SequenceMatcher ratio between ``a`` and ``b``.

    Thin wrapper kept module-private so the clusterer's similarity
    rule is in one place — easier to swap for a fuzzy-string library
    later without touching call sites.
    """
    return SequenceMatcher(None, a, b).ratio()


def _is_leading_stem(stem: str, text: str) -> bool:
    """Return ``True`` when ``text`` begins with ``stem`` as a whole word.

    The boundary check accepts an exact match (``text == stem``) and
    any non-alphanumeric / non-underscore character immediately
    following the stem slice — so ``"WCS"`` is a leading stem of
    ``"WCS API"`` and ``"WCS-dashboard"`` but NOT of
    ``"WCSGateway"`` (no boundary).
    """
    if text == stem:
        return True
    if not text.startswith(stem):
        return False
    boundary = text[len(stem)]
    return not (boundary.isalnum() or boundary == "_")


def _can_bridge_clusters(
    cand_text: str,
    clusters: Sequence[Sequence[TokenCandidate]],
    matching_indices: Sequence[int],
) -> bool:
    """Bridging policy for a candidate matching multiple clusters.

    Returns ``True`` only when ``cand_text`` is the leading-word stem
    of at least one member in EVERY matched cluster. This blocks a
    generic shared acronym like ``"API"`` (which is a suffix word in
    both ``"WCS API"`` and ``"GitHub API"`` and therefore shouldn't
    fold those distinct product surfaces into one cluster) while
    still allowing a meaningful prefix stem like ``"WCS"`` to merge
    ``"WCS API"`` + ``"WCS dashboard"``. The asymmetric prefix-only
    rule reflects the observation that the operator's real glossary
    stems are almost always prefixes of their phrase forms
    (``WCS → WCS API``, ``GitHub → GitHub Actions``); shared suffix
    words are usually generic English nouns the LLM will translate
    contextually anyway.
    """
    for idx in matching_indices:
        cluster = clusters[idx]
        if not any(_is_leading_stem(cand_text, m.text) for m in cluster):
            return False
    return True


def _pick_canonical(members: Sequence[TokenCandidate]) -> str:
    """Return the canonical form of a cluster.

    Selection rule: most-frequent variant; ties broken by longer
    string (so ``"WCS API"`` beats ``"WCS"`` at equal counts — the
    longer form carries more semantic detail); ties at length broken
    lexicographically so the result is deterministic across runs.
    """
    best = max(
        members,
        key=lambda c: (c.count, len(c.text), tuple(-ord(ch) for ch in c.text)),
    )
    return best.text


def cluster_candidates(
    candidates: Sequence[TokenCandidate],
    *,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> List[GlossaryCluster]:
    """Group near-duplicate candidates into clusters.

    Two candidates ``a`` and ``b`` are merged into the same cluster
    when ANY of the following holds:

      * ``SequenceMatcher(None, a.text, b.text).ratio() >=
        similarity_threshold`` — captures spelling variants.
      * ``a.text`` is a **leading-word stem** of ``b.text`` (or vice
        versa) — captures ``"WCS"`` / ``"WCS API"`` / ``"WCS
        dashboard"`` where the longer form is a phrase built on the
        shorter stem. The rule is asymmetric (leading-stem only, not
        any whole-word substring) so a generic shared suffix like
        ``"API"`` does NOT merge ``"WCS API"`` and ``"GitHub API"``
        into one cluster — distinct product surfaces stay distinct
        regardless of which order the candidates arrive in.

    The walker processes candidates in the order
    :func:`filter_by_thresholds` returned them (most-frequent first),
    so an earlier cluster's canonical is locked in before a less-
    frequent variant attempts to merge. When a single candidate
    matches MULTIPLE existing clusters the bridging policy is enforced
    by :func:`_can_bridge_clusters`: a candidate folds every matched
    cluster into one ONLY when it is the leading-word stem of every
    matched cluster (``"WCS"`` arriving after ``"WCS API"`` +
    ``"WCS dashboard"`` ⇒ merge). A candidate that merely shares a
    generic acronym (``"API"`` arriving after ``"WCS API"`` +
    ``"GitHub API"``) attaches to the first matched cluster only so
    distinct product surfaces are NOT collapsed into one row.
    """
    clusters: List[List[TokenCandidate]] = []
    for cand in candidates:
        matching_indices: List[int] = []
        for idx, cluster in enumerate(clusters):
            for member in cluster:
                if (
                    _similar(cand.text, member.text) >= similarity_threshold
                    or _is_leading_stem(cand.text, member.text)
                    or _is_leading_stem(member.text, cand.text)
                ):
                    matching_indices.append(idx)
                    break
        if not matching_indices:
            clusters.append([cand])
            continue
        if len(matching_indices) == 1:
            clusters[matching_indices[0]].append(cand)
            continue
        # Multi-cluster match — bridge only when the candidate is a
        # leading-word stem of every matched cluster. Generic
        # connectors (shared suffix acronyms) fall through to the
        # "first matched cluster" attach so unrelated surfaces stay
        # separate.
        if _can_bridge_clusters(cand.text, clusters, matching_indices):
            primary = clusters[matching_indices[0]]
            primary.append(cand)
            for idx in sorted(matching_indices[1:], reverse=True):
                primary.extend(clusters.pop(idx))
        else:
            clusters[matching_indices[0]].append(cand)

    result: List[GlossaryCluster] = []
    for cluster in clusters:
        canonical = _pick_canonical(cluster)
        variants = tuple(
            sorted(
                (m.text for m in cluster),
                key=lambda t: (
                    -next(c.count for c in cluster if c.text == t),
                    t,
                ),
            )
        )
        total = sum(m.count for m in cluster)
        files: set[str] = set()
        for m in cluster:
            files.update(m.files)
        result.append(
            GlossaryCluster(
                canonical=canonical,
                variants=variants,
                total_count=total,
                files=frozenset(files),
            )
        )
    # Sort by total_count desc, then canonical asc — stable order so
    # the JSON output is byte-equal across runs of the same corpus.
    result.sort(key=lambda c: (-c.total_count, c.canonical))
    return result


# ---------------------------------------------------------------------------
# Translation.
# ---------------------------------------------------------------------------

# Signature of a bulk-translation callable. The default
# :func:`litellm_bulk_translator` matches it; tests inject simpler
# deterministic stubs (e.g. ``lambda sources, target_langs: [
#   {"source": s, "translations": {l: f"<{s}:{l}>" for l in
#   target_langs}} for s in sources]``). Returning a list of dicts
# keeps the boundary plain-JSON so a network round-trip is not
# implicit in the contract.
BulkTranslator = Callable[
    [Sequence[str], Sequence[str]],
    List[Dict[str, Any]],
]


# System prompt for the bulk-translation call. The source-language
# label is a placeholder so it can be rendered for any source language.
# The strict JSON-only instruction matters because the structured-output
# path is the fallback rather than the default — older LiteLLM installs
# and a handful of providers ignore ``response_format``, and the verb
# needs to keep working there.
_BULK_SYSTEM_PROMPT_TEMPLATE = (
    "You are a translation AI specialized in translating {source} text "
    "into other languages. Given multiple {source} source terms (brand "
    "names, product names, technical jargon), provide accurate and "
    "contextually appropriate translations in the following languages: "
    "{targets}. For each source term, provide a translation in every "
    "requested target language. Preserve casing for acronyms and brand "
    "names; localise common nouns. Respond ONLY with a JSON object of "
    "the shape "
    "{{\"items\": [{{\"source\": str, \"translations\": "
    "{{lang: str, ...}}}}, ...]}}."
)


def _bulk_system_prompt(source_lang: str, target_langs: Sequence[str]) -> str:
    """Render the bulk-translation system prompt for the given languages.

    Kept as a free function rather than a constant so library callers
    can re-render it for ad-hoc inspection (e.g. logging the exact
    prompt that was billed) without rebuilding the whole pipeline.
    """
    return _BULK_SYSTEM_PROMPT_TEMPLATE.format(
        source=source_lang,
        targets=", ".join(target_langs),
    )


def _bulk_user_prompt(
    sources: Sequence[str],
    source_lang: str,
    target_langs: Sequence[str],
) -> str:
    """Render the bulk-translation user prompt.

    The numbered ``[N]`` prefix is preserved in LLM responses verbatim,
    which the parser uses as a robustness signal (a missing index
    implies the LLM dropped an entry and the caller emits an
    empty-translation placeholder for that source).
    """
    lines: List[str] = [
        f"Translate the following {source_lang} source terms into "
        + ", ".join(target_langs)
        + ".",
        f"Target language codes: {', '.join(target_langs)}",
        "",
    ]
    for i, src in enumerate(sources, 1):
        lines.append(f"[{i}] {source_lang}: {src}")
    return "\n".join(lines)


# Matches the user-prompt format ``[N] {source_lang}: {term}`` that
# :func:`_bulk_user_prompt` renders. Some LLMs echo this prefix back
# verbatim in their JSON response ``source`` field; the resolver
# below strips the prefix so a faithful echo still maps back to the
# caller's source list.  ``[N]`` alone (no tail) is also matched so a
# positional-index-only response can fall back to the Nth source.
_BULK_SOURCE_LABEL_RE = re.compile(
    r"^\s*\[(\d+)\]\s*(?:[A-Za-z0-9_.\-]+\s*:\s*)?(.*?)\s*$"
)


def _resolve_bulk_source(
    label: str, sources: Sequence[str]
) -> Optional[str]:
    """Map an LLM-returned ``source`` field back to one of ``sources``.

    Resolution order (most specific wins):
      1. **Exact text match** — the LLM honoured the strict prompt
         and returned the bare term. The common path.
      2. **Numbered prefix, exact tail** — the LLM echoed the prompt
         format ``[N] {source_lang}: {term}``. Strip the prefix and
         re-match the tail against ``sources``. Robust against an LLM
         that volunteered the wrong index alongside a correct term.
      3. **Numbered prefix, positional fallback** — the tail did not
         match (typo, light paraphrase, dropped trailing punctuation),
         but the ``[N]`` index lies inside ``sources``. Returns
         ``sources[N-1]``. Worst-case the operator sees a
         translation attached to the right source position even when
         the LLM mutated the term itself.

    Returns ``None`` when none of the rules apply — the LLM
    hallucinated a key we never asked about. The caller drops such
    items rather than promoting them to glossary rows.
    """
    if label in sources:
        return label
    m = _BULK_SOURCE_LABEL_RE.match(label)
    if not m:
        return None
    idx_str, tail = m.group(1), m.group(2)
    if tail and tail in sources:
        return tail
    try:
        idx = int(idx_str)
    except ValueError:
        return None
    if 1 <= idx <= len(sources):
        return sources[idx - 1]
    return None


def _parse_bulk_response(
    raw: str, sources: Sequence[str]
) -> List[Dict[str, Any]]:
    """Parse the bulk-translation LLM JSON into the public list shape.

    Tolerates a ``json``-fenced response (some providers wrap the
    payload even with the strict prompt) and falls back to the raw
    text otherwise. Items the LLM dropped surface as empty
    translation dicts so the caller still emits an entry for the
    source — better to surface the gap to the operator than silently
    skip it.

    Source labels are resolved via :func:`_resolve_bulk_source` so an
    LLM that echoes the user-prompt format
    (``"[1] en: WCS"`` instead of ``"WCS"``) still maps back to the
    caller's source list. Without that normalisation the strict
    exact-match path would silently emit empty translations for every
    row in a faithful echo response.
    """
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text[3:]
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"bulk-translator LLM did not return valid JSON: {raw!r}"
        ) from exc
    if not isinstance(payload, dict) or "items" not in payload:
        raise ValueError(
            "bulk-translator JSON must be an object with an 'items' "
            f"field; got: {payload!r}"
        )
    items = payload["items"]
    if not isinstance(items, list):
        raise ValueError(
            f"bulk-translator 'items' must be a JSON array; got: {items!r}"
        )
    # Build a {source: translations} map from the LLM's output so the
    # caller's source ordering is preserved even when the LLM returned
    # items in a different sequence. Items whose ``source`` does not
    # resolve to any input are silently dropped (the LLM hallucinated
    # a key we never asked about). Later items with the same resolved
    # source overwrite earlier ones — a tie is rare and the more
    #-detailed item is conventionally returned last by chat models.
    by_source: Dict[str, Dict[str, str]] = {}
    sources_list = list(sources)
    for item in items:
        if not isinstance(item, dict):
            continue
        src_label = item.get("source")
        translations = item.get("translations")
        if not isinstance(src_label, str) or not isinstance(translations, dict):
            continue
        # Coerce all values to strings so a stray null / number is not
        # propagated into the glossary file.
        coerced = {
            str(lang): str(val) if val is not None else ""
            for lang, val in translations.items()
        }
        resolved = _resolve_bulk_source(src_label, sources_list)
        if resolved is None:
            continue
        by_source[resolved] = coerced
    return [
        {
            "source": src,
            "translations": by_source.get(src, {}),
        }
        for src in sources
    ]


def litellm_bulk_translator(
    *,
    model: str,
    source_lang: str = "en",
) -> BulkTranslator:
    """Return a :data:`BulkTranslator` that routes through ``litellm``.

    The returned callable is closure-bound to ``model`` /
    ``source_lang`` so the CLI builds it once and the cluster loop
    invokes it without re-resolving config. Tests do NOT use this
    factory — they inject a deterministic stub callable directly.
    """

    def _translate(
        sources: Sequence[str],
        target_langs: Sequence[str],
    ) -> List[Dict[str, Any]]:
        if not sources:
            return []
        system_prompt = _bulk_system_prompt(source_lang, target_langs)
        user_prompt = _bulk_user_prompt(sources, source_lang, target_langs)
        response = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw = response.choices[0].message.content or "{}"
        return _parse_bulk_response(raw, sources)

    return _translate


# ---------------------------------------------------------------------------
# End-to-end pipeline.
# ---------------------------------------------------------------------------


def suggest_glossary(
    source_dir: Path,
    *,
    target_langs: Sequence[str],
    min_occurrences: int = DEFAULT_MIN_OCCURRENCES,
    min_files: int = DEFAULT_MIN_FILES,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    translator: Optional[BulkTranslator] = None,
    source_lang: str = "en",
) -> List[GlossarySuggestion]:
    """Run the full extract → filter → cluster → translate pipeline.

    Returns the list of :class:`GlossarySuggestion` ready to render
    via :func:`write_suggested_glossary`. ``translator`` is injectable
    so tests pass deterministic stubs without monkey-patching
    ``litellm``; when ``None``, the caller is responsible for
    constructing a real translator via :func:`litellm_bulk_translator`
    before any LLM-issuing run.

    Empty / no-candidate corpora are a successful no-op: the function
    returns an empty list rather than raising. The walker handles
    missing-directory cases by returning zero candidates, which is
    surfaced to the operator via the empty output file.
    """
    if not target_langs:
        raise ValueError("target_langs must contain at least one locale")
    if min_occurrences < 1:
        raise ValueError("min_occurrences must be >= 1")
    if min_files < 1:
        raise ValueError("min_files must be >= 1")
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be in [0.0, 1.0]")

    histogram = collect_candidates(source_dir)
    candidates = filter_by_thresholds(
        histogram,
        min_occurrences=min_occurrences,
        min_files=min_files,
    )
    clusters = cluster_candidates(
        candidates,
        similarity_threshold=similarity_threshold,
    )
    if not clusters:
        return []
    sources = [c.canonical for c in clusters]
    if translator is None:
        translated: List[Dict[str, Any]] = [
            {"source": src, "translations": {}} for src in sources
        ]
    else:
        translated = translator(sources, list(target_langs))
    by_source: Dict[str, Dict[str, str]] = {
        item["source"]: dict(item.get("translations", {}))
        for item in translated
        if isinstance(item, dict) and isinstance(item.get("source"), str)
    }
    suggestions: List[GlossarySuggestion] = []
    for cluster in clusters:
        raw = by_source.get(cluster.canonical, {})
        # Ensure every requested locale is present so the operator's
        # review pass sees a stable per-row shape — missing locales
        # render as empty strings.
        translations = {
            lang: str(raw.get(lang, "")) for lang in target_langs
        }
        suggestions.append(
            GlossarySuggestion(
                canonical=cluster.canonical,
                translations=translations,
                count=cluster.total_count,
                files=tuple(sorted(cluster.files)),
                variants=cluster.variants,
            )
        )
    return suggestions


# ---------------------------------------------------------------------------
# Output rendering.
# ---------------------------------------------------------------------------


def _suggestions_to_glossary_dict(
    suggestions: Sequence[GlossarySuggestion],
) -> Dict[str, Any]:
    """Render suggestions into the ``glossary.json`` schema.

    The schema matches what :class:`mdpo_llm.processor.MarkdownProcessor`
    already consumes via ``glossary_path=``: each key maps either to a
    string (single-locale shorthand) or to a per-locale dict. The
    verb always emits the per-locale form because the corpus is
    multi-locale by construction; the operator may flatten on
    promotion.
    """
    out: Dict[str, Any] = {}
    for sugg in suggestions:
        out[sugg.canonical] = dict(sugg.translations)
    return out


def write_suggested_glossary(
    suggestions: Sequence[GlossarySuggestion],
    output_path: Path,
) -> None:
    """Write ``suggestions`` to ``output_path`` as JSON.

    Refuses (``ValueError``) when the output path's basename is
    exactly ``glossary.json`` — promotion is a manual review step by
    design. Operators who really want to write to ``glossary.json``
    can still do so by renaming the suggested file outside the verb.

    Parent directories are created on the fly when missing so the
    operator can run the verb against a fresh checkout without first
    mkdir'ing the target tree.
    """
    if output_path.name == AUTHORED_GLOSSARY_FILENAME:
        raise ValueError(
            f"refusing to write to {output_path}: "
            f"basename {AUTHORED_GLOSSARY_FILENAME!r} is reserved for "
            "the user-authored glossary. Use the default "
            f"{SUGGESTED_GLOSSARY_FILENAME!r} or an alternative name."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _suggestions_to_glossary_dict(suggestions)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False)
        + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# CLI plumbing.
# ---------------------------------------------------------------------------


def _split_targets(raw: str) -> List[str]:
    """Split a ``--target`` argument value into a list of locales.

    Accepts comma- or space-separated locales; empty entries are
    dropped so trailing commas don't surface as zero-length locale
    keys in the JSON output.
    """
    parts = re.split(r"[,\s]+", raw.strip())
    return [p for p in parts if p]


def add_suggest_glossary_subparser(
    sub: "argparse._SubParsersAction[argparse.ArgumentParser]",
) -> argparse.ArgumentParser:
    """Register the ``suggest-glossary`` subcommand on the top-level argparse.

    Kept as a public helper so :mod:`mdpo_llm.__main__` can attach the
    subparser without importing implementation symbols individually,
    matching :func:`mdpo_llm.cli_check_image.add_check_image_subparser`
    / :func:`mdpo_llm.cli_lint.add_lint_subparser`.
    """
    p = sub.add_parser(
        "suggest-glossary",
        help=(
            "Scan a source markdown tree for high-frequency proper-noun-like "
            "tokens and emit a draft glossary.suggested.json."
        ),
        description=(
            "Walk a source directory of markdown files, extract proper-"
            "noun-like single-word and short-phrase candidates, filter "
            "by occurrence / file thresholds, cluster near-duplicates "
            "via SequenceMatcher, translate each cluster's canonical "
            "form into the requested target locales via a bulk LLM "
            "call, and write a draft glossary.suggested.json the "
            "operator can review and promote into a real glossary.json. "
            "Refuses to overwrite the user-authored glossary.json "
            "filename by design — promotion is manual."
        ),
    )
    p.add_argument(
        "source_dir",
        help=(
            "Source directory of markdown files (scanned recursively). "
            "Extensions: " + " ".join(MARKDOWN_EXTENSIONS) + "."
        ),
    )
    p.add_argument(
        "--target",
        required=True,
        help=(
            "Comma-separated list of target locales (e.g. 'ko,ja,zh-CN'). "
            "Each cluster's canonical form is translated into every "
            "requested locale via a single bulk LLM call."
        ),
    )
    p.add_argument(
        "--model",
        required=True,
        help=(
            "LiteLLM model string used for the bulk translation call "
            "(e.g. 'gpt-4o', 'openrouter/openai/gpt-4o', "
            "'anthropic/claude-sonnet-4-5-20250929')."
        ),
    )
    p.add_argument(
        "--source-lang",
        default="en",
        help=(
            "BCP 47 locale of the source corpus (default: 'en'). Used to "
            "label the prompt rendered for the bulk-translation LLM call."
        ),
    )
    p.add_argument(
        "--min-occurrences",
        type=int,
        default=DEFAULT_MIN_OCCURRENCES,
        help=(
            "Minimum total occurrences across the corpus for a token / "
            f"phrase to be eligible as a candidate (default: "
            f"{DEFAULT_MIN_OCCURRENCES})."
        ),
    )
    p.add_argument(
        "--min-files",
        type=int,
        default=DEFAULT_MIN_FILES,
        help=(
            "Minimum number of distinct source files a token / phrase "
            f"must appear in (default: {DEFAULT_MIN_FILES})."
        ),
    )
    p.add_argument(
        "--similarity-threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help=(
            "SequenceMatcher ratio at or above which two candidates "
            "merge into the same cluster (default: "
            f"{DEFAULT_SIMILARITY_THRESHOLD}). The whole-word containment "
            "rule fires independently of this threshold."
        ),
    )
    p.add_argument(
        "--output",
        default=None,
        help=(
            "Output path for the draft glossary. Default: "
            f"<source_dir>/{SUGGESTED_GLOSSARY_FILENAME}. The verb "
            f"refuses to write to a path named {AUTHORED_GLOSSARY_FILENAME!r}."
        ),
    )
    p.set_defaults(func=cmd_suggest_glossary)
    return p


def _build_default_translator(model: str, source_lang: str) -> BulkTranslator:
    """Return a default :data:`BulkTranslator` for CLI runs.

    Indirection point so tests that drive ``cmd_suggest_glossary``
    can monkey-patch this single symbol to inject a fake translator
    without having to also stub ``litellm`` at import time.
    """
    return litellm_bulk_translator(model=model, source_lang=source_lang)


def cmd_suggest_glossary(args: argparse.Namespace) -> int:
    """``mdpo-llm suggest-glossary`` entry point.

    Exit code contract:
      * ``2`` — usage error (missing / non-directory source path,
        empty ``--target``, output basename equals
        ``glossary.json``, threshold values out of range).
      * ``1`` — reserved; this verb does not surface "findings" in
        the lint sense — the JSON output IS the deliverable.
      * ``0`` — successful run, including the corner case of zero
        clusters (the output file is still written, just empty).
    """
    source_dir = Path(args.source_dir)
    if not source_dir.exists():
        print(
            f"error: source directory does not exist: {source_dir}",
            file=sys.stderr,
        )
        return 2
    if not source_dir.is_dir():
        print(
            f"error: source path is not a directory: {source_dir}",
            file=sys.stderr,
        )
        return 2

    target_langs = _split_targets(args.target)
    if not target_langs:
        print(
            "error: --target must list at least one locale "
            "(comma-separated, e.g. 'ko,ja,zh-CN')",
            file=sys.stderr,
        )
        return 2

    if args.output is None:
        output_path = source_dir / SUGGESTED_GLOSSARY_FILENAME
    else:
        output_path = Path(args.output)
    if output_path.name == AUTHORED_GLOSSARY_FILENAME:
        print(
            f"error: refusing to write to {output_path}: "
            f"basename {AUTHORED_GLOSSARY_FILENAME!r} is reserved for "
            "the user-authored glossary. Use the default "
            f"{SUGGESTED_GLOSSARY_FILENAME!r} or pass --output with a "
            "different name.",
            file=sys.stderr,
        )
        return 2

    try:
        translator = _build_default_translator(args.model, args.source_lang)
    except Exception as exc:  # pragma: no cover - construction-time guard
        print(f"error: failed to build LLM translator: {exc}", file=sys.stderr)
        return 2

    try:
        suggestions = suggest_glossary(
            source_dir,
            target_langs=target_langs,
            min_occurrences=args.min_occurrences,
            min_files=args.min_files,
            similarity_threshold=args.similarity_threshold,
            translator=translator,
            source_lang=args.source_lang,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        write_suggested_glossary(suggestions, output_path)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Surface a one-line summary on stderr so the operator can see
    # whether the run produced anything useful without having to cat
    # the JSON. Stdout stays empty so callers can still pipe stderr
    # for logging without polluting downstream readers.
    print(
        f"Wrote {len(suggestions)} glossary candidate(s) to {output_path}",
        file=sys.stderr,
    )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Standalone entry point: ``python -m mdpo_llm.glossary_suggest …``.

    Mirrors the ``mdpo-llm suggest-glossary`` subcommand surface so the
    module can be driven directly without going through the top-level
    parser — convenient for ad-hoc runs and for tests that exercise the
    CLI surface in isolation.
    """
    parser = argparse.ArgumentParser(prog="mdpo-llm-suggest-glossary")
    sub = parser.add_subparsers(dest="command", required=True)
    add_suggest_glossary_subparser(sub)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
