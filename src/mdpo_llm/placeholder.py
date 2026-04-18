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
from typing import Dict, List, Optional, Pattern as RePattern, Tuple, Union

# Marker used for pre-existing token-literals found in the source text.
# They're recorded as identity entries in the mapping so the round-trip
# check expects them in the output and numbering never collides with one.
LITERAL_PATTERN_NAME = "__literal__"

# Uses U+27E6 / U+27E7 (mathematical white brackets) because they are rare
# in technical prose yet still round-trip safely through JSON, UTF-8, and
# common Markdown renderers — unlike private-use codepoints which some
# providers strip.
TOKEN_RE = re.compile(r"\u27e6P:(\d+)\u27e7")


def format_token(index: int) -> str:
    """Render the Nth placeholder as the literal token string."""
    return f"\u27e6P:{index}\u27e7"


@dataclass(frozen=True)
class PlaceholderPattern:
    """A named regex whose matches become opaque tokens on ``encode``."""

    name: str
    regex: RePattern[str]


@dataclass(frozen=True)
class Placeholder:
    """One substitution produced by :meth:`PlaceholderRegistry.encode`."""

    token: str
    original: str
    pattern_name: str


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
    ) -> None:
        """Register a pattern.

        ``regex`` may be a string (compiled with ``flags``) or an already-
        compiled ``re.Pattern`` (``flags`` is ignored in that case — bake
        flags into the compile step yourself).
        """
        if isinstance(regex, re.Pattern):
            compiled = regex
        else:
            compiled = re.compile(regex, flags)
        self._patterns.append(PlaceholderPattern(name=name, regex=compiled))

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

        Tokens absent from ``mapping`` (model hallucinated a token index)
        pass through unchanged so the round-trip check can surface them as
        ``unexpected`` rather than silently eating them.
        """
        if not mapping.items:
            return text
        lookup = {p.token: p.original for p in mapping.items}

        def replace(m: "re.Match[str]") -> str:
            return lookup.get(m.group(0), m.group(0))

        return TOKEN_RE.sub(replace, text)


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
