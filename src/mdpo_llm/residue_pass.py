"""
Source-language residue post-processing pass for code spans (T-17).

The translate pass occasionally leaves source-language characters inside
fenced code blocks, inline code spans, or filename-shaped code tokens
even after T-12 prompt hardening — the LLM's monolingual prior is strong
enough that "preserve identifiers, translate user-facing strings" is not
always honoured.  This module runs *after* the structural / LLM
validators in :class:`mdpo_llm.processor.MarkdownProcessor` and
re-translates only the affected spans through specialised prompts:

* Fenced code block → preserve identifiers + comments, translate
  user-facing string literals only.
* Filename-shaped inline code → transliterate to ``UPPER_SNAKE_CASE``
  ASCII placeholders so the resulting filename stays cross-platform safe.
* Other inline code → translate to natural target text.

Best-effort: if the specialised LLM call raises or the post-repair
round-trip check rejects the output, the original pass-1 translation is
kept verbatim and a warning is logged.  Residue pass MUST NOT introduce
new ``⟦P:N⟧`` placeholder tokens or alter the existing token
counts — the caller's structural validator runs against the same text
post-pass.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Optional, Tuple

from .language import LANGUAGE_PATTERNS, _resolve_primary
from .placeholder import TOKEN_RE

logger = logging.getLogger(__name__)


# Per-source-language Unicode-range patterns used for residue detection.
# The map is keyed by BCP 47 primary subtag so callers (and tests) can
# add a language by editing one place.  ``ja`` deliberately includes the
# CJK Unified Ideographs block alongside the kana ranges because
# Japanese kanji shares the codepoint range with Chinese — false-flagging
# a CJK ideograph as Japanese rather than Chinese is fine for residue
# detection (we only care that *some* source-language character leaked
# into a code span).
SOURCE_LANG_PATTERNS: Dict[str, re.Pattern[str]] = {
    "ko": re.compile(r"[ᄀ-ᇿ㄰-㆏가-힣]"),
    "ja": re.compile(
        r"[぀-ゟ゠-ヿㇰ-ㇿ一-鿿]"
    ),
    "zh": re.compile(r"[一-鿿]"),
}

SUPPORTED_SOURCE_LANGS = frozenset(SOURCE_LANG_PATTERNS.keys())


# Kana-only subset of the Japanese pattern, used when the target
# language also uses CJK ideographs (so kanji is NOT residue but
# hiragana / katakana still are).  Keeps the (ja → zh) case useful
# instead of degenerating to a no-op while avoiding the false
# positives Codex flagged on (zh → ja).
_JA_KANA_ONLY_PATTERN = re.compile(r"[぀-ゟ゠-ヿㇰ-ㇿ]")


def _resolve_residue_pattern(
    source_lang: str, target_lang: Optional[str]
) -> Optional[re.Pattern[str]]:
    """Return the regex used to detect residue characters of
    ``source_lang`` inside text written in ``target_lang`` — or
    ``None`` when the pair has no detectable residue.

    Three pairs are special-cased to avoid false positives where the
    source language's range overlaps a script the target language
    naturally writes in:

    * ``zh → ja`` and ``zh → zh`` — ``zh``'s pattern is just CJK
      Unified Ideographs, which Japanese (kanji) and Chinese both use
      natively.  Scanning a Japanese ``msgstr`` for "zh residue" would
      flag every legitimate kanji.
    * ``ja → zh`` — ``ja``'s pattern includes CJK ideographs alongside
      hiragana / katakana.  We narrow detection to the kana-only
      subset so kanji isn't double-counted as residue.
    * ``X → X`` (any same-language pair) — the source IS the target,
      so by definition there is no source-language residue to detect.

    Unknown source / target languages fall through to the full
    :data:`SOURCE_LANG_PATTERNS` entry (for ``source_lang``) so callers
    targeting unsupported scripts (Latin English, etc.) still get the
    full per-source scan they had before this refinement.
    """
    src = _resolve_primary(source_lang)
    base = SOURCE_LANG_PATTERNS.get(src)
    if base is None:
        return None
    if target_lang is None:
        return base
    tgt = _resolve_primary(target_lang)
    if src == tgt:
        return None
    if src == "zh" and tgt in ("ja", "zh"):
        return None
    if src == "ja" and tgt == "zh":
        return _JA_KANA_ONLY_PATTERN
    return base


# Span types reported by :func:`detect_residues`.
SpanKind = Literal["fenced", "inline_filename", "inline_other"]


# Fenced code block: opening / closing ``` or ~~~ on their own line, with
# matching fence character and length.  ``re.MULTILINE`` makes ``^`` /
# ``$`` line-relative; the ``[ \t]*`` prefix tolerates the kramdown-style
# indented fences that appear inside list items.
_FENCED_RE = re.compile(
    r"(?ms)^([ \t]*)(`{3,}|~{3,})([^\n]*)\n(.*?)(?:\n)\1\2[ \t]*$"
)

# Inline code span: a single backtick-delimited run on a single line.  We
# deliberately do NOT support multi-backtick fences (`` `` etc.) here —
# the residue pass only needs to spot contaminated runs, and the rare
# multi-backtick form would risk overlapping with fenced-block matches.
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")

# Conservative filename heuristic: an inline code span looks like a
# filename when its trimmed body ends with a dot followed by 1-8
# alphanumeric characters AND contains no whitespace.  Path separators
# are allowed (``docs/intro.md`` should classify as filename); raw
# extension tokens like ``.json`` alone do NOT match because the
# leading dot is part of the extension test rather than the stem.
_FILENAME_RE = re.compile(r"^[^\s]+\.[A-Za-z0-9]{1,8}$")


# Module-level prompt templates.  Lifted from
# ``/processors/llm_refiner.py``
# with the comment-preservation rule made explicit because the T-12
# pass-1 prompt asks the LLM to translate comments — at residue-pass time
# the comments have already been processed and re-translating them risks
# round-trip drift, so they MUST be preserved verbatim.
RESIDUE_FENCED_PROMPT = """\
You are repairing a translated fenced code block. The block still contains
source-language characters that the previous translation pass missed.

Rewrite the block so that:
- Source-language text inside USER-FACING string literals (the strings end
  users see — printed messages, log labels, exception messages, UI strings)
  is translated into {target}.
- Comments (``#`` line comments, ``//`` line comments, ``/* ... */`` block
  comments, ``<!-- ... -->`` markup comments, docstring blocks) MUST be kept
  exactly as in the input — do NOT translate or rewrite comments.
- Identifiers (variable names, function names, class names, keys, type
  names), code keywords, operators, indentation, and structure are kept
  exactly as in the input — do NOT rename, reorder, or re-style anything.
- The opening and closing fence (``` or ~~~) and the info string after the
  opening fence MUST appear unchanged.

Output ONLY the rewritten code block (fences included), with no commentary,
no surrounding prose, and no Markdown headings.\
"""

RESIDUE_INLINE_FILENAME_PROMPT = """\
You are repairing a translated inline code span that names a filename or
filesystem path. The span still contains source-language characters that
the previous translation pass missed.

Rewrite the span so that source-language tokens are transliterated into
``UPPER_SNAKE_CASE`` ASCII identifiers — for example ``회원목록`` becomes
``MEMBER_LIST``. Preserve:
- Path separators (``/``).
- File extensions exactly as in the input (case included).
- Any ASCII identifier characters already present.

Do NOT translate the extension itself, do NOT introduce spaces, and do NOT
add any commentary. Output ONLY the rewritten inline span (without
surrounding backticks).\
"""

RESIDUE_INLINE_OTHER_PROMPT = """\
You are repairing a translated inline code span. The span still contains
source-language characters that the previous translation pass missed.

Translate the source-language text into natural {target} while keeping
identifier-shaped ASCII tokens (variable names, type names, API names)
exactly as in the input. Output ONLY the rewritten inline span (without
surrounding backticks), with no commentary or Markdown.\
"""


# ``(system_prompt, user_text) -> response_text``.
# The caller owns provider-specific concerns (model selection, API keys,
# usage accounting) and threads the response back as a plain string.
LLMCallable = Callable[[str, str], str]


@dataclass(frozen=True)
class ResidueSpan:
    """One residue-bearing span detected inside a translated entry.

    Offsets are character positions into the original ``text`` passed to
    :func:`detect_residues` so callers can splice repaired output back
    in without re-scanning.  ``text`` includes the span's delimiters
    (fence lines for ``fenced``, surrounding backticks for inline) so
    the LLM sees the same shape it must reproduce.
    """

    kind: SpanKind
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class RepairResult:
    """Outcome of a single :func:`repair_block` call.

    ``repaired`` is ``True`` only when the LLM returned a usable
    replacement that differs from the original AND survived the
    placeholder-token preservation check.  ``text`` is the value to
    splice back into the surrounding entry — equal to the span's
    original text when ``repaired`` is ``False``.  ``reason`` is a
    short human-readable note suitable for logging.
    """

    text: str
    repaired: bool
    reason: str


def _placeholder_token_counts(text: str) -> Counter:
    """Multiset of ``⟦P:N⟧`` token strings in ``text``.

    Used to confirm the residue pass did not introduce, drop, or
    duplicate any placeholder token.  The msgstr is normally
    post-decode (no tokens at all), so the common case is a pair of
    empty Counters; we still run the check unconditionally so a
    malicious / malformed LLM response that *injects* a fake token
    cannot slip past.
    """
    return Counter(m.group(0) for m in TOKEN_RE.finditer(text))


def _looks_like_filename(body: str) -> bool:
    """Return True when ``body`` (the text inside the backticks) shapes
    like a filename or filesystem path.

    Centralised so the public detection / repair paths apply the same
    heuristic — drift would let one path classify a span as filename
    while the other treated it as natural text and emit the wrong prompt.
    """
    return bool(_FILENAME_RE.match(body.strip()))


def strip_code_spans(text: str) -> str:
    """Return ``text`` with every fenced and inline code span removed.

    Used by callers (notably the processor's residue-pass entrypoint)
    that want to detect the source's natural language without
    code-span identifiers polluting the signal: a CJK filename
    intentionally embedded as `` `用户.md` `` in an English document
    should NOT cause :func:`detect_languages` to flag the source as
    Chinese.  Removed spans collapse to a single space so adjacent
    natural-text tokens stay visually separated.
    """
    stripped = _FENCED_RE.sub(" ", text)
    return _INLINE_CODE_RE.sub(" ", stripped)


def detect_residues(
    text: str,
    source_lang: str,
    *,
    target_lang: Optional[str] = None,
) -> List[ResidueSpan]:
    """Return every residue-bearing fenced or inline span in ``text``.

    A span is "residue-bearing" when it contains at least one character
    in the residue pattern resolved for ``(source_lang, target_lang)``
    (see :func:`_resolve_residue_pattern`).  When ``target_lang`` is
    omitted the full :data:`SOURCE_LANG_PATTERNS` entry is used —
    callers that already know they're translating into a script that
    overlaps the source range (CJK ideographs are shared between
    Japanese kanji and Chinese) MUST pass ``target_lang`` to avoid
    flagging legitimate target-script characters as residue.

    Spans are returned in document order.  Inline spans that fall
    inside a previously matched fenced span are dropped so the caller
    repairs the fenced block as a unit instead of double-repairing the
    same characters via two prompts.
    """
    pattern = _resolve_residue_pattern(source_lang, target_lang)
    if pattern is None:
        return []
    spans: List[ResidueSpan] = []

    # Fenced first so we can mask their character ranges before the
    # inline scan.  Iterating with ``finditer`` (not ``findall``) keeps
    # each match's ``start`` / ``end`` available for offset-based
    # splicing in :func:`apply_residue_pass`.
    fenced_ranges: List[tuple] = []
    for m in _FENCED_RE.finditer(text):
        body = m.group(4)
        if pattern.search(body):
            spans.append(
                ResidueSpan(
                    kind="fenced",
                    start=m.start(),
                    end=m.end(),
                    text=text[m.start() : m.end()],
                )
            )
        fenced_ranges.append((m.start(), m.end()))

    for m in _INLINE_CODE_RE.finditer(text):
        # Skip inline matches that overlap any fenced range — those
        # characters belong to the fenced block and would otherwise be
        # repaired twice (once by the fenced prompt, once standalone),
        # which the brief explicitly rules out.
        in_fence = any(
            f_start <= m.start() < f_end for f_start, f_end in fenced_ranges
        )
        if in_fence:
            continue
        body = m.group(1)
        if not pattern.search(body):
            continue
        kind: SpanKind = (
            "inline_filename" if _looks_like_filename(body) else "inline_other"
        )
        spans.append(
            ResidueSpan(
                kind=kind,
                start=m.start(),
                end=m.end(),
                text=text[m.start() : m.end()],
            )
        )

    spans.sort(key=lambda s: s.start)
    return spans


def _strip_inline_backticks(repaired: str) -> str:
    """Return ``repaired`` with at most one pair of leading / trailing
    backticks removed.

    The inline prompts ask the model to omit the surrounding backticks,
    but models occasionally include them anyway; we re-wrap below so
    accepting either shape keeps the repair path tolerant.  Only ONE
    pair is stripped — multi-backtick fences would be malformed for an
    inline span and should fail validation rather than be silently
    unwrapped.
    """
    s = repaired.strip()
    if len(s) >= 2 and s.startswith("`") and s.endswith("`"):
        return s[1:-1]
    return s


def repair_block(
    span: ResidueSpan,
    target_lang: str,
    llm_callable: LLMCallable,
    *,
    validator: Optional[Callable[[str, str], bool]] = None,
) -> RepairResult:
    """Re-translate one residue span and return the result.

    Args:
        span: One :class:`ResidueSpan` from :func:`detect_residues`.
        target_lang: BCP 47 locale to translate the residue into.
        llm_callable: Closure that takes ``(system_prompt, user_text)``
            and returns the LLM's raw response string.  Caller owns
            model selection and usage accounting — this module is
            provider-agnostic on purpose so the unit tests can drive
            the path with a plain Python function.
        validator: Optional ``(original_text, repaired_text) -> bool``.
            Called after the LLM responds; ``False`` means the repair
            failed validation (e.g. introduced a new placeholder
            token).  Defaults to a built-in placeholder-token
            preservation check when omitted.

    Returns:
        :class:`RepairResult`. ``repaired=False`` cases include LLM
        exceptions, an empty / whitespace-only response, validator
        rejection, and "no-op" responses identical to the input.  The
        caller MUST splice ``RepairResult.text`` back in either way —
        on failure that text is the original span verbatim, satisfying
        the "pass-1 result wins on residue-pass failure" contract.
    """
    if span.kind == "fenced":
        prompt = RESIDUE_FENCED_PROMPT.format(target=target_lang)
        user_input = span.text
        wrap_inline = False
    elif span.kind == "inline_filename":
        prompt = RESIDUE_INLINE_FILENAME_PROMPT
        # Strip the backticks before sending so the LLM only sees the
        # filename body — the prompt asks for output without backticks
        # and we re-wrap below.  A naive caller passing ``span.text``
        # would force the LLM to either echo the backticks (defeating
        # the prompt) or omit them inconsistently.
        user_input = _strip_inline_backticks(span.text)
        wrap_inline = True
    else:
        prompt = RESIDUE_INLINE_OTHER_PROMPT.format(target=target_lang)
        user_input = _strip_inline_backticks(span.text)
        wrap_inline = True

    try:
        raw = llm_callable(prompt, user_input)
    except Exception as exc:  # noqa: BLE001 — best-effort polish
        logger.warning(
            "residue pass: LLM call raised for %s span (%d chars): %s; "
            "keeping pass-1 output",
            span.kind,
            len(span.text),
            exc,
        )
        return RepairResult(
            text=span.text,
            repaired=False,
            reason=f"llm-error: {exc}",
        )

    if not isinstance(raw, str) or not raw.strip():
        logger.warning(
            "residue pass: LLM returned empty / non-string for %s span; "
            "keeping pass-1 output",
            span.kind,
        )
        return RepairResult(
            text=span.text,
            repaired=False,
            reason="empty-response",
        )

    if wrap_inline:
        body = _strip_inline_backticks(raw)
        candidate = f"`{body}`"
    else:
        candidate = raw.strip("\n")

    if candidate == span.text:
        return RepairResult(
            text=span.text,
            repaired=False,
            reason="no-op",
        )

    check = validator if validator is not None else _default_validator
    if not check(span.text, candidate):
        logger.warning(
            "residue pass: validator rejected repair for %s span; "
            "keeping pass-1 output",
            span.kind,
        )
        return RepairResult(
            text=span.text,
            repaired=False,
            reason="validator-rejected",
        )

    return RepairResult(text=candidate, repaired=True, reason="ok")


def _default_validator(original: str, repaired: str) -> bool:
    """Default repair validator: placeholder token multiset preserved.

    Encodes the brief's "MUST NOT introduce new ``⟦P:N⟧``
    tokens or alter existing token counts" rule.  Other structural
    invariants (fence count, heading level, language presence) are the
    surrounding processor validator's job — running them here would
    duplicate the work already done in pass 1.
    """
    return _placeholder_token_counts(original) == _placeholder_token_counts(
        repaired
    )


def apply_residue_pass(
    text: str,
    source_lang: str,
    target_lang: str,
    llm_callable: LLMCallable,
    *,
    validator: Optional[Callable[[str, str], bool]] = None,
) -> str:
    """Detect and repair every residue span in ``text``.

    Spans are spliced in REVERSE document order so each replacement
    preserves the offsets of earlier (lower-indexed) spans — a
    forward-order rewrite would invalidate every subsequent
    ``ResidueSpan.start`` after the first non-identity replacement.

    Returns the repaired text.  When detection finds no residues, the
    function returns ``text`` unchanged AND issues zero LLM calls — the
    "no-op clean output" guarantee from the brief's test plan, which
    keeps an opt-in residue pass cheap on documents that don't need it.
    """
    spans = detect_residues(text, source_lang, target_lang=target_lang)
    if not spans:
        return text
    # Issue LLM calls in DOCUMENT order so callers / mocks see a stable
    # call sequence that matches the spans they would scan in by hand.
    # Splicing then runs in REVERSE order so each replacement preserves
    # the offsets of earlier spans (a forward-splice on the first
    # non-identity replacement would invalidate every later
    # ``ResidueSpan.start``).
    residue_pattern = _resolve_residue_pattern(source_lang, target_lang)
    repairs: List[Tuple[ResidueSpan, RepairResult]] = []
    for span in spans:
        result = repair_block(
            span, target_lang, llm_callable, validator=validator
        )
        if (
            result.repaired
            and residue_pattern is not None
            and span.kind != "fenced"
            and residue_pattern.search(result.text)
        ):
            # Models often partially comply (e.g. ``회원_LIST.md`` still
            # contains Hangul) on inline spans.  Accepting that would
            # let the residue pass silently "succeed" while leaving the
            # same source-language characters on disk — defeating the
            # only job the pass has.  Reject and keep pass-1 output.
            #
            # Fenced blocks are EXEMPT from this re-scan because the
            # fenced repair prompt explicitly forbids translating
            # comments — a legitimate fenced repair on a block with
            # source-language comments will (correctly) still contain
            # source-language characters in those comments.  The
            # placeholder-token round-trip + the surrounding
            # structural validator (re-run by the processor caller)
            # remain in play for fenced spans.
            logger.warning(
                "residue pass: repair for %s span still contains "
                "source-language residue; keeping pass-1 output",
                span.kind,
            )
            result = RepairResult(
                text=span.text,
                repaired=False,
                reason="residue-not-removed",
            )
        repairs.append((span, result))
    out = text
    for span, result in reversed(repairs):
        if not result.repaired:
            continue
        out = out[: span.start] + result.text + out[span.end :]
    return out
