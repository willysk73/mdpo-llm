"""Domain context cascade loader (T-18).

Free-text domain context — audience, tone, conventions, proper-noun
guidance — fed verbatim into translation and validation system prompts.
Glossary handles term-level substitutions; this module handles the
document-level briefing that does not fit cleanly as term pairs.

Two public entry points:

- :func:`resolve_context_chain` walks the per-directory cascade
  (``source_root`` → file directory, plus ``cwd``, plus the
  constructor-time ``--context PATH`` override) and returns the
  concatenated text. Empty / missing files at any level are silently
  skipped because most directories will not have a ``context.md`` —
  logging that would be noise.
- :func:`inject_context` splices the resolved text into a system
  prompt under the stable ``ADDITIONAL_CONTEXT_HEADER`` so callers do
  not have to re-derive the heading text.

Pure functions; the processor owns any caching. The brief calls out
that this lives alongside (not inside) :class:`MarkdownProcessor`
because glossary handles term-level substitutions while context
handles document-level briefing — a deliberately different shape.
Concatenation (parent + child) is the merge rule rather than
glossary's child-wins override: domain context is additive by nature
("game-security SDK" + "casual tone OK in tutorials" + "this section
uses formal voice"); a child writer rarely wants to *replace* the
parent's domain framing, only to extend it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

CONTEXT_FILENAME = "context.md"
ADDITIONAL_CONTEXT_HEADER = (
    "**ADDITIONAL CONTEXT (use for proper nouns, terminology, "
    "tone, audience):**"
)
# Soft cap on a single context source (per-file or CLI override). Beyond
# this size the user is paying serious token cost on every batch — the
# loader logs a warning so the bill is visible but does NOT truncate;
# the brief is explicit that users self-select brief size.
MAX_CONTEXT_BYTES = 64 * 1024


def _safe_resolve(path: Path) -> Path:
    """``Path.resolve(strict=False)`` with an ``OSError`` fallback.

    Mirrors the resolution strategy used by the glossary cascade so
    cache keys and existence probes line up byte-for-byte.
    """
    try:
        return path.resolve(strict=False)
    except OSError:
        return path.absolute()


def read_context_file(path: Path) -> str:
    """Read a context file as UTF-8 text; return ``""`` for missing / invalid.

    Both "missing file" and "empty file" return ``""`` — the brief
    treats them identically (silently skipped). Soft-caps at
    :data:`MAX_CONTEXT_BYTES`: longer files are passed through verbatim
    (the user opts into the token cost) but a warning is logged so
    operators see that the bill scales per API call.
    """
    resolved = _safe_resolve(path)
    try:
        if not resolved.is_file():
            return ""
        data = resolved.read_bytes()
    except OSError as exc:
        logger.warning(
            "failed to read context file at %s: %s; ignored.",
            resolved,
            exc,
        )
        return ""
    if not data.strip():
        return ""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        logger.warning(
            "context file at %s is not valid UTF-8 (%s); ignored.",
            resolved,
            exc,
        )
        return ""
    if len(data) > MAX_CONTEXT_BYTES:
        logger.warning(
            "context file at %s is %d bytes (>%d soft cap); passing "
            "through verbatim — token cost will scale with every API "
            "call.",
            resolved,
            len(data),
            MAX_CONTEXT_BYTES,
        )
    return text


def _walk_dirs(source_root: Path, file_dir: Path) -> List[Path]:
    """Return ``source_root`` → ``file_dir`` inclusive in parent → child order.

    Returns ``[]`` when ``file_dir`` is outside ``source_root`` so a
    sibling tree's ``context.md`` cannot leak into this run — same
    containment rule the glossary cascade uses (security smell at
    worst, surprising at best).
    """
    root = _safe_resolve(source_root)
    leaf = _safe_resolve(file_dir)
    try:
        rel = leaf.relative_to(root)
    except ValueError:
        return []
    walk = [root]
    cur = root
    for part in rel.parts:
        cur = cur / part
        walk.append(cur)
    return walk


def resolve_context_chain(
    source_path: Path,
    source_root: Optional[Path],
    cwd: Optional[Path],
    cli_override: Optional[str] = None,
) -> str:
    """Concatenate per-directory ``context.md`` files plus the CLI override.

    Order (each non-empty block is appended, blocks separated by a
    blank line):

    1. ``source_root/context.md`` … ``file_dir/context.md`` walked in
       parent → child order. Empty / missing files at any level are
       silently skipped — most directories will NOT have a
       ``context.md`` and warning on absence would be noise. When
       ``source_root`` is ``None`` the walk falls back to just the
       file's own directory, which is what single-file
       :meth:`MarkdownProcessor.process_document` callers want.
    2. ``cwd/context.md`` if ``cwd`` is non-``None`` AND not already a
       step in the tree walk (otherwise we'd double-apply). Mirrors
       the glossary cascade's "run from tree root" vs "run from an
       unrelated dir" symmetry.
    3. ``cli_override`` (the constructor-time ``--context PATH`` text)
       appended LAST so it follows / extends the cascade rather than
       replacing it. The brief deliberately makes this additive
       because domain framing is cumulative, not authoritative-from-
       one-source.

    Returns the merged string; ``""`` when nothing resolves so the
    caller can ``if context: ...``-gate the injection.
    """
    blocks: List[str] = []
    visited: set = set()

    # Self-reference guard: when the caller is translating a
    # ``context.md`` file (e.g. ``glob="**/context.md"``), the file
    # itself sits at ``file_dir / context.md`` and would otherwise be
    # read back into its own system prompt as ADDITIONAL CONTEXT —
    # and then sent again as the user payload. Track the resolved
    # source path so any cascade level that points at the same file
    # is skipped.
    source_resolved = _safe_resolve(Path(source_path))

    def _read_unless_self(path: Path) -> str:
        candidate_resolved = _safe_resolve(path)
        if candidate_resolved == source_resolved:
            return ""
        return read_context_file(path)

    file_dir = Path(source_path).parent
    if source_root is not None:
        walked = _walk_dirs(Path(source_root), file_dir)
        if not walked:
            # ``file_dir`` sits outside ``source_root`` (shouldn't
            # normally happen — process_directory globs under
            # source_root). Fall back to the file's own dir so a
            # single ``context.md`` next to the file is still picked
            # up; cwd / override semantics still apply.
            walked = [_safe_resolve(file_dir)]
        for d in walked:
            res = _safe_resolve(d)
            if res in visited:
                continue
            visited.add(res)
            text = _read_unless_self(res / CONTEXT_FILENAME)
            if text:
                blocks.append(text)
    else:
        res = _safe_resolve(file_dir)
        visited.add(res)
        text = _read_unless_self(res / CONTEXT_FILENAME)
        if text:
            blocks.append(text)

    if cwd is not None:
        cwd_resolved = _safe_resolve(Path(cwd))
        if cwd_resolved not in visited:
            text = _read_unless_self(cwd_resolved / CONTEXT_FILENAME)
            if text:
                blocks.append(text)

    if cli_override:
        blocks.append(cli_override)

    return "\n\n".join(blocks)


def inject_context(
    system_prompt: str, context_text: Optional[str]
) -> str:
    """Append the additional-context block to ``system_prompt``.

    ``context_text`` of ``None`` or ``""`` returns ``system_prompt``
    unchanged so a "no context configured" run pays no prompt-shape
    cost. Otherwise the block is appended after every existing rule
    section (instruction body, glossary block, reference-pair block,
    retry-reason block) — the brief places it last because the LLM
    reads instructions top-down and the freshest content in the
    model's working memory at the moment user-message translation
    begins should be the domain briefing.

    A trailing newline on the input prompt is preserved so the result
    keeps the same final-newline shape as the original — callers that
    rely on either form (existing single-target template ends with
    ``\\n``; batch builders strip it) get back what they put in.
    """
    if not context_text:
        return system_prompt
    block = f"{ADDITIONAL_CONTEXT_HEADER}\n{context_text}"
    if system_prompt.endswith("\n"):
        return system_prompt[:-1] + "\n\n" + block + "\n"
    return system_prompt + "\n\n" + block
