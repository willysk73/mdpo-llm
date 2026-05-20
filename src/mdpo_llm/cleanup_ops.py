"""Remove translated files whose source has been deleted since the
last ``translate-dir`` run (T-20).

This is the standalone equivalent of the in-flight stale-output cleanup
that :meth:`MarkdownProcessor.process_directory` already performs when
``translate_paths=True``. The verb is meant for the common case where
the operator has translated a tree, then deleted or moved files in the
source tree, and wants to prune the now-orphaned target outputs
without re-running translation.

Layouts supported (the same ones :mod:`mdpo_llm.processor` writes):

* **Mirror layout** — target tree mirrors source tree
  (``target_dir / src_rel == target_path``), per-document PO is at
  ``target_dir / src_rel.with_suffix(".po")``. No ``path_map.json``.
* **Translate-paths layout** — target paths are localized; the
  source-to-target mapping lives in ``target_dir / "path_map.json"``,
  per-document POs default to ``target_dir / src_rel.with_suffix(".po")``
  (source-relative, NOT translated-relative — see processor.py:3645),
  and the segment catalog lives in ``<po_root> / "_paths.po"``.
* **Separate ``--po-dir`` variant of either layout** — when supplied,
  per-document POs live at ``po_dir / src_rel.with_suffix(".po")`` and
  ``_paths.po`` (when present) at ``po_dir / "_paths.po"``.
  ``path_map.json`` is always under ``target_dir`` (it tracks the
  effective output tree).

Three deletion modes:

1. **Orphaned target file.** Source file is gone from ``source_dir``;
   the translated target survives under ``target_dir``. The cleanup
   removes the target markdown, its sibling PO (unless ``--keep-po``),
   and prunes the segment entries in ``_paths.po`` whose ``msgid`` is
   no longer used by any surviving source.
2. **Stale ``path_map.json`` entries.** A ``{src_rel: tgt_rel}`` pair
   whose source file no longer exists is removed from the map. This
   happens implicitly as part of mode 1 for every orphaned source that
   had an entry, plus a defensive sweep for entries whose target was
   never on disk.
3. **(Limitation) Renamed source detection.** 's
   ``cleanup_ops`` does not detect renames either: it can only
   distinguish "input present" from "input absent". mdpo-llm's
   ``_paths.po`` records segment translations, not file-level rename
   pairs, so a rename surfaces here as "old source absent, new source
   present" — i.e. an orphan plus a new translation pending. The
   operator must re-run ``translate-dir`` to mint the new target;
   moving target files automatically would clobber hand-edits.

The target markdown file itself is never moved or modified — only
removed when the source has truly disappeared.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import polib


logger = logging.getLogger(__name__)


# Extensions of files we consider "translated markdown artefacts" when
# walking the target tree. Limited to ``.md`` for now; 
# also covered ``.pdf`` but mdpo-llm's pipeline only writes Markdown,
# and treating an arbitrary user-deposited PDF as an orphan would risk
# deleting hand-managed attachments.
TARGET_EXTENSIONS: frozenset[str] = frozenset({".md"})


@dataclass(frozen=True)
class CleanupResult:
    """Summary of a cleanup run.

    The ``removed_*`` lists hold the paths or identifiers that were (or
    would have been, under ``--dry-run``) acted on. Lists are sorted by
    repository sort order on the contained paths / strings so the
    body of the report renders deterministically across runs and across
    platforms. ``failures`` lists the paths whose apply step actually
    raised (filesystem permission denied, locked file on Windows,
    read-only mount, …) — the CLI uses it to surface a non-zero exit so
    CI does not silently treat a partial cleanup as success.
    """

    removed_targets: List[Path] = field(default_factory=list)
    removed_pos: List[Path] = field(default_factory=list)
    removed_paths_po_entries: List[str] = field(default_factory=list)
    removed_path_map_entries: List[str] = field(default_factory=list)
    failures: List[Path] = field(default_factory=list)
    dry_run: bool = False

    def is_noop(self) -> bool:
        return not (
            self.removed_targets
            or self.removed_pos
            or self.removed_paths_po_entries
            or self.removed_path_map_entries
        )

    def to_dict(self) -> dict:
        return {
            "dry_run": self.dry_run,
            "removed_targets": [str(p) for p in self.removed_targets],
            "removed_pos": [str(p) for p in self.removed_pos],
            "removed_paths_po_entries": list(self.removed_paths_po_entries),
            "removed_path_map_entries": list(self.removed_path_map_entries),
            "failures": [str(p) for p in self.failures],
        }


class _CorruptPathMap(Exception):
    """Raised when ``path_map.json`` exists but cannot be parsed.

    Distinguishes "file genuinely absent" (mirror-layout fallback is
    safe) from "file present but unreadable" (destructive
    mirror-fallback would wrongly classify translated paths like
    ``ko/intro.md`` as orphans of source ``guide/intro.md``). The
    caller refuses to run in the second case.
    """


def _load_path_map(path_map_json_path: Path) -> dict[str, str]:
    """Load ``path_map.json`` strictly.

    Returns ``{}`` only when the file does not exist. When it DOES
    exist but cannot be parsed (OSError, UnicodeDecodeError, malformed
    JSON, non-dict root), raises :class:`_CorruptPathMap`. The caller
    refuses the destructive mirror-fallback in that case — a
    ``--translate-paths`` tree has localized targets that mirror-fallback
    would wrongly classify as orphans.
    """
    if not path_map_json_path.is_file():
        return {}
    try:
        raw = json.loads(path_map_json_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _CorruptPathMap(
            f"Could not parse {path_map_json_path}: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise _CorruptPathMap(
            f"{path_map_json_path}: top-level JSON value is not an object"
        )
    # Every row must be a non-empty ``str -> str`` mapping. Silently
    # dropping malformed rows is unsafe: in a ``--translate-paths``
    # layout a single bad row (e.g. ``{"guide/intro.md": 123}``) would
    # cause the target-walk to mirror-fallback for ``ko/intro.md`` and
    # delete a live translation. Treat any deviation as corruption so
    # the operator repairs the file before re-running.
    for k, v in raw.items():
        if (
            not isinstance(k, str)
            or not isinstance(v, str)
            or not k
            or not v
        ):
            raise _CorruptPathMap(
                f"{path_map_json_path}: malformed row {k!r} -> {v!r} "
                "(both must be non-empty strings)"
            )
        if not _is_safe_relative(k) or not _is_safe_relative(v):
            # A row with ``..`` components or absolute paths poisons the
            # reverse lookup: the on-disk target won't match the
            # advertised value, mirror-fallback kicks in, and live
            # translations get deleted. Same correctness pitfall as
            # the dotdot escape in PO-path resolution — refuse early.
            raise _CorruptPathMap(
                f"{path_map_json_path}: row {k!r} -> {v!r} contains "
                "an absolute path or '..' component (only normalised, "
                "tree-relative POSIX paths are accepted)"
            )
        if k != _canonical_relposix(k) or v != _canonical_relposix(v):
            # Non-normalised forms (``ko//intro.md``, ``ko/./intro.md``,
            # backslash-separated ``ko\\intro.md``) pass the safety
            # checks but won't match ``Path.relative_to(...).as_posix()``
            # in the target walk; the reverse lookup misses, mirror-
            # fallback kicks in, and a live translation gets deleted.
            # The operator's file is the source of truth — refuse with
            # a clear repair instruction instead of silently
            # canonicalising the read (which would let the on-disk
            # file stay inconsistent across runs).
            raise _CorruptPathMap(
                f"{path_map_json_path}: row {k!r} -> {v!r} is not in "
                "canonical POSIX form (no ``//``, no ``./`` segments, "
                "forward slashes only). Re-emit the map via "
                "translate-dir --translate-paths or fix the entry by hand."
            )
    # Reject duplicate target values: two source keys mapping to the
    # same target is structurally invalid for a translate-paths layout
    # (the target filesystem can only hold one file at that path), and
    # the reverse lookup ``{tgt: src for src, tgt in raw.items()}``
    # below would silently drop one owner. If the surviving owner is
    # the one dropped, the target walk classifies a live translation
    # as an orphan and deletes it.
    seen_targets: dict[str, str] = {}
    for k, v in raw.items():
        prior = seen_targets.get(v)
        if prior is not None:
            raise _CorruptPathMap(
                f"{path_map_json_path}: duplicate target {v!r} mapped "
                f"from both {prior!r} and {k!r}. Each target path must "
                "appear at most once."
            )
        seen_targets[v] = k
    return {k: v for k, v in raw.items()}


def _canonical_relposix(s: str) -> str:
    """Return the canonical POSIX form of ``s`` (no ``//``, no ``./``,
    no ``\\`` backslashes).

    Used to detect non-normalised path_map rows: the target walk
    compares against ``tf.relative_to(target_path).as_posix()`` which
    is always canonical, so a row in a non-canonical form would miss
    the reverse lookup and mirror-fallback would delete a live
    translation. ``PurePosixPath`` collapses ``//`` and drops ``./``
    segments while preserving case and the relative-vs-absolute
    distinction.
    """
    from pathlib import PurePosixPath

    # Normalise backslashes first so a Windows operator's ``ko\\intro.md``
    # gets compared against the same shape as ``ko/intro.md`` would.
    s_fwd = s.replace("\\", "/")
    return PurePosixPath(s_fwd).as_posix()


def _is_safe_relative(s: str) -> bool:
    """Return True when ``s`` is a normalised tree-relative POSIX path.

    Rejects anything that could escape the tree on join:
    - absolute paths (``/foo`` on POSIX, ``C:\\foo`` on Windows);
    - any ``..`` component;
    - Windows drive letters / UNC roots (``C:/x``, ``//host/share``).

    These checks mirror the runtime escape probe used for PO-path
    deletion; pulling them up to load time means the corrupt-map
    abort surfaces a clear repair message instead of a quiet skip in
    the per-target loop.
    """
    if not s:
        return False
    if "\x00" in s:
        return False
    # Reject absolute paths under either PosixPath or WindowsPath
    # semantics. ``Path`` resolves the local flavour, so an operator
    # editing path_map.json on Windows could legitimately have written
    # backslashes — normalise before the test.
    normalised = s.replace("\\", "/")
    if normalised.startswith("/"):
        return False
    # Windows drive letter (``C:``, ``C:/x``). Be precise: an ASCII
    # letter followed by ``:`` followed by ``/`` or end-of-string is a
    # drive root; a colon anywhere else in the string (POSIX file names
    # legally allow it) is fine.
    if (
        len(normalised) >= 2
        and normalised[0].isascii()
        and normalised[0].isalpha()
        and normalised[1] == ":"
        and (len(normalised) == 2 or normalised[2] == "/")
    ):
        return False
    parts = normalised.split("/")
    if any(part == ".." for part in parts):
        return False
    return True


def _looks_like_segment_catalog(po_path: Path) -> bool:
    """Return True when ``po_path`` may be a translate-paths segment catalog.

    A real segment catalog has at least one entry whose ``msgctxt``
    starts with ``"path::segment::"`` (see
    :meth:`MarkdownProcessor._translate_path_segments`). A per-document
    PO for a source file named ``_paths.md`` shares the filename but
    its entries are parser-context-keyed and never start with that
    prefix.

    A PARSE FAILURE returns ``True`` rather than ``False``: the caller
    uses this signal alongside a missing ``path_map.json`` to decide
    whether to refuse the destructive mirror-fallback. An unreadable
    catalog could still be a translate-paths catalog whose layout we
    can no longer confirm, and assuming "definitely not the catalog"
    would let the cleanup proceed and delete live localized targets.
    Only a successfully-parsed PO with ZERO ``path::segment::*`` rows
    is definitively NOT a segment catalog.
    """
    try:
        po_file = polib.pofile(str(po_path), encoding="utf-8")
    except Exception:
        return True
    for entry in po_file:
        if entry.obsolete:
            continue
        if (entry.msgctxt or "").startswith("path::segment::"):
            return True
    return False


def _segments_used_by(source_rels: set[str]) -> set[str]:
    """Return the set of raw path segments referenced by ``source_rels``.

    Mirrors the segment-collection rule in
    :meth:`MarkdownProcessor._translate_path_segments` so the cleanup's
    "is this segment still in use?" check matches what a fresh
    ``translate-dir`` run would write. Dotfile-only and navigation
    segments (``""``, ``"."``, ``".."``, ``.foo``) are filesystem
    tokens, never translated, and therefore never tracked in
    ``_paths.po`` — exclude them from the surviving set so a stray
    ``msgid == ""`` row gets correctly classified as orphan.
    """

    def _is_translatable(seg: str) -> bool:
        return bool(seg) and seg not in {".", ".."} and not seg.startswith(".")

    survivors: set[str] = set()
    for src_rel in source_rels:
        parts = Path(src_rel).parts
        if not parts:
            continue
        parent_parts = list(parts[:-1])
        stem = Path(parts[-1]).stem
        for seg in parent_parts + [stem]:
            if _is_translatable(seg):
                survivors.add(seg)
    return survivors


def _scan_source_rels(source_dir: Path) -> set[str]:
    """Return the POSIX-form relative paths of every Markdown file under
    ``source_dir``.

    POSIX form so the comparison against ``path_map.json`` keys (which
    are always POSIX) works on Windows checkouts too. Extension match
    is case-INsensitive (``.md`` and ``.MD`` both count) to mirror the
    target walk's ``tf.suffix.lower() in TARGET_EXTENSIONS`` rule —
    otherwise an uppercase source like ``README.MD`` on a
    case-sensitive filesystem would be missing from ``source_rels``
    while the target walker still considers ``README.MD`` a cleanup
    candidate, and the live translation would be deleted.
    """
    result: set[str] = set()
    for sf in source_dir.rglob("*"):
        if not sf.is_file():
            continue
        if sf.suffix.lower() not in TARGET_EXTENSIONS:
            continue
        try:
            result.add(sf.relative_to(source_dir).as_posix())
        except ValueError:
            # rglob normally yields children, but symlink loops or
            # cross-tree links could surface paths outside source_dir;
            # skip those defensively rather than crash.
            continue
    return result


def cleanup_target_tree(
    target_dir: Path | str,
    source_dir: Path | str,
    *,
    po_dir: Optional[Path | str] = None,
    dry_run: bool = False,
    keep_po: bool = False,
) -> CleanupResult:
    """Prune orphaned outputs under ``target_dir``.

    Args:
        target_dir: Directory containing translated markdown (the
            ``target_dir`` argument that was passed to ``translate-dir``).
        source_dir: Source directory the translation ran against.
            Must exist; the absence of every translation's source is
            indistinguishable from "wrong directory entirely", so we
            refuse to run rather than wipe the target.
        po_dir: Optional override for the per-document PO directory and
            ``_paths.po`` location. When omitted, the cleanup looks for
            ``_paths.po`` under ``target_dir`` and per-document POs
            alongside the source-relative path under ``target_dir`` —
            matching the defaults that ``process_directory`` uses.
        dry_run: When ``True``, classify but do not delete or rewrite
            anything. The returned :class:`CleanupResult` still lists
            what would have been removed so the body of the report
            matches what a real run would emit (the header line
            differs).
        keep_po: When ``True``, the sibling per-document PO file for
            each orphan target is preserved. The matching
            ``_paths.po`` segment entries are still pruned (their
            translations remain valid input for any new source picking
            the same segment).

    Returns:
        :class:`CleanupResult` summarising the (planned) deletions.

    Raises:
        FileNotFoundError: ``source_dir`` does not exist.
        NotADirectoryError: ``source_dir`` or ``target_dir`` exists but
            is not a directory.
    """
    target_path = Path(target_dir)
    source_path = Path(source_dir)

    if not source_path.exists():
        raise FileNotFoundError(
            f"source directory does not exist: {source_path}"
        )
    if not source_path.is_dir():
        raise NotADirectoryError(
            f"source path is not a directory: {source_path}"
        )

    # Validate ``--po-dir`` BEFORE the missing-target no-op so a
    # mistyped path doesn't silently pass on a first-run / already-
    # cleaned tree. Same rationale as the source check: a usage error
    # belongs in front of the idempotent fast-path.
    if po_dir is not None and not Path(po_dir).is_dir():
        raise NotADirectoryError(
            f"po-dir path is not a directory: {po_dir}"
        )

    # Target may legitimately be missing (translate-dir was never run, or
    # the operator already cleaned the tree). Surface a no-op rather
    # than an error — the verb is idempotent and an empty result is the
    # honest answer.
    if not target_path.exists():
        return CleanupResult(dry_run=dry_run)
    if not target_path.is_dir():
        raise NotADirectoryError(
            f"target path is not a directory: {target_path}"
        )

    # Nesting guard. The classifier compares ``tgt_rel`` (relative to
    # target_path) against ``source_rels`` (relative to source_path).
    # If target_path is an ancestor of source_path — e.g.
    # ``mdpo-llm cleanup . --source docs`` — every real source file
    # under target would surface with ``tgt_rel = "docs/foo.md"``
    # while ``source_rels = {"foo.md"}``: mismatch → the orphan
    # classifier schedules the SOURCE for deletion. The verb is
    # destructive; refuse the invocation rather than try to slice the
    # source subtree out of the walk (which has its own subtle
    # corner cases with symlinks, ``--po-dir`` overlap, etc.).
    try:
        target_resolved = target_path.resolve(strict=False)
    except OSError:
        target_resolved = target_path.absolute()
    try:
        source_resolved = source_path.resolve(strict=False)
    except OSError:
        source_resolved = source_path.absolute()
    if target_resolved == source_resolved:
        raise ValueError(
            f"refusing to run: target_dir and --source resolve to the "
            f"same path ({target_resolved}); cleanup would treat every "
            f"source file as an orphan and delete it."
        )
    try:
        source_resolved.relative_to(target_resolved)
    except ValueError:
        pass
    else:
        raise ValueError(
            f"refusing to run: --source ({source_resolved}) lies "
            f"inside target_dir ({target_resolved}); cleanup would "
            f"classify every source file as an orphan and delete it. "
            f"Run cleanup against the localized output tree, not a "
            f"parent that contains the source."
        )

    po_root = Path(po_dir) if po_dir is not None else target_path

    source_rels = _scan_source_rels(source_path)
    path_map_json_path = target_path / "path_map.json"
    paths_po_path = po_root / "_paths.po"
    try:
        path_map = _load_path_map(path_map_json_path)
    except _CorruptPathMap as exc:
        # Refuse to run rather than mirror-fallback over a known
        # ``--translate-paths`` tree: targets like ``ko/intro.md`` are
        # not their own source, and the mirror probe would wrongly
        # classify them as orphans and delete live translations. The
        # operator needs to repair / restore / delete ``path_map.json``
        # before re-running cleanup.
        raise ValueError(
            f"refusing to run with a corrupt path_map.json — repair "
            f"or delete the file and re-run. Details: {exc}"
        ) from exc

    # Layout-ambiguity guard: a ``--translate-paths`` tree is
    # identified by the segment catalog at ``po_root / "_paths.po"``.
    # When that artefact is present BUT ``path_map.json`` is missing
    # (operator-deleted, restored partially from backup, …), we have
    # no source→target reverse map and the per-target classification
    # would silently fall back to mirror mode, treating a live
    # localized target like ``ko/intro.md`` as the orphan of
    # ``guide/intro.md`` and deleting it.
    #
    # Distinguish the actual segment catalog from a plain per-document
    # PO whose source happens to be named ``_paths.md`` (the mirror
    # layout produces ``target/_paths.po`` in that case): a real
    # segment catalog has at least one entry whose ``msgctxt`` starts
    # with ``path::segment::``. A per-document PO does not — its
    # entries are content blocks keyed by parser context. Only refuse
    # to run when the file is a true segment catalog AND ``path_map.json``
    # is missing.
    if (
        paths_po_path.is_file()
        and not path_map_json_path.is_file()
        and _looks_like_segment_catalog(paths_po_path)
    ):
        raise ValueError(
            f"refusing to run: {paths_po_path} looks like a "
            f"--translate-paths segment catalog but {path_map_json_path} is "
            f"missing. Restore path_map.json (or re-run translate-dir "
            f"--translate-paths to regenerate it) before cleanup."
        )

    # ``path_map.json`` is operator-editable and may have drifted: a key
    # containing ``..`` segments or an absolute path would, when joined
    # against ``po_root``, resolve OUTSIDE the configured PO tree —
    # absolute paths fully discard the prefix. The orphan deletion phase
    # below would then happily unlink an unrelated ``.po`` somewhere
    # else on disk. Resolve both roots up front so the per-target probe
    # can refuse anything that escapes the PO tree.
    try:
        po_root_resolved = po_root.resolve(strict=False)
    except OSError:
        po_root_resolved = po_root.absolute()

    # Reverse lookup: tgt_rel → src_rel. Built once so the per-file
    # classification below is O(1) per target file.
    tgt_to_src: dict[str, str] = {
        tgt_rel: src_rel for src_rel, tgt_rel in path_map.items()
    }

    removed_targets: list[Path] = []
    removed_pos: list[Path] = []
    removed_path_map_keys: set[str] = set()
    # Reverse maps so the apply phase can correlate a failed unlink
    # back to the path_map.json key it came from. Critical in
    # translate-paths layouts: the path_map row is the ONLY pointer
    # from translated tgt_rel back to source-relative src_rel, so a
    # successful target delete combined with a failed PO delete must
    # NOT drop the row — otherwise a retry has no way to find the PO
    # path (lookup would fall back to tgt_rel and probe the wrong PO
    # location).
    target_owner: dict[Path, str] = {}
    po_owner: dict[Path, str] = {}

    def _schedule_po_removal(src_rel: str) -> None:
        """Queue the per-document PO for ``src_rel`` (if any) for deletion.

        Respects ``--keep-po`` and the ``po_root`` escape check. Idempotent
        — called from both the target-walk loop (orphan-on-disk path) and
        the stale-path_map sweep (target already gone, PO may still be).
        """
        if keep_po or not src_rel:
            return
        po_path = po_root / Path(src_rel).with_suffix(".po")
        try:
            po_resolved = po_path.resolve(strict=False)
        except OSError:
            po_resolved = po_path.absolute()
        try:
            po_resolved.relative_to(po_root_resolved)
        except ValueError:
            logger.warning(
                "Skipping PO deletion outside po-dir: %s (from key %r)",
                po_resolved,
                src_rel,
            )
            return
        if po_path.is_file() and po_path not in po_owner:
            removed_pos.append(po_path)
            po_owner[po_path] = src_rel

    # Walk every Markdown file under target_dir and decide whether it
    # belongs to a surviving source. Files whose extension is outside
    # TARGET_EXTENSIONS are left alone — the cleanup verb is scoped to
    # what mdpo-llm produces, not to arbitrary user content the
    # operator may have parked in the target tree.
    for tf in sorted(target_path.rglob("*")):
        if not tf.is_file():
            continue
        if tf.suffix.lower() not in TARGET_EXTENSIONS:
            continue
        try:
            tgt_rel_posix = tf.relative_to(target_path).as_posix()
        except ValueError:
            continue

        src_rel = tgt_to_src.get(tgt_rel_posix, tgt_rel_posix)
        if src_rel in source_rels:
            continue
        if not src_rel:
            # Defence in depth — ``_load_path_map`` already drops empty
            # keys, but a future caller might construct a path_map by
            # hand. ``Path("").with_suffix(".po")`` raises ``ValueError``,
            # which would abort the whole cleanup.
            continue

        removed_targets.append(tf)
        target_owner[tf] = src_rel
        _schedule_po_removal(src_rel)
        if src_rel in path_map:
            removed_path_map_keys.add(src_rel)

    # Mode 2 sweep: ``path_map.json`` may carry rows whose source is
    # gone AND whose translated target was never on disk in the first
    # place (operator deleted both, or the target was generated by a
    # later --translate-paths run and then the source was renamed
    # without re-running). The matching per-document PO can still be
    # on disk in either layout — same-tree (default) or separate
    # ``--po-dir`` — and an entry-only sweep would orphan that PO and
    # then drop the only mapping needed to locate it on a retry. So
    # schedule the PO too, subject to the same escape / keep-po rules
    # as the on-disk-target path.
    for src_rel in list(path_map.keys()):
        if src_rel in source_rels:
            continue
        removed_path_map_keys.add(src_rel)
        _schedule_po_removal(src_rel)

    # ``_paths.po`` segment pruning. Load BEFORE the apply step so the
    # dry-run report is identical to the real run. ``paths_po_path``
    # was resolved up front for the layout-ambiguity guard above.
    paths_po_file: Optional[polib.POFile] = None
    paths_po_entries_to_drop: list[polib.POEntry] = []
    removed_paths_po_msgctxts: list[str] = []
    if paths_po_path.is_file():
        # polib raises a mix of OSError (read failures), UnicodeDecodeError
        # (encoding drift), and bare ``Exception`` (malformed PO syntax) —
        # a corrupt segment catalog must not abort the whole cleanup, so
        # log and continue without segment pruning instead.
        try:
            paths_po_file = polib.pofile(str(paths_po_path), encoding="utf-8")
        except Exception as exc:
            logger.warning(
                "Could not parse %s; skipping segment pruning: %s",
                paths_po_path,
                exc,
            )
            paths_po_file = None
    if paths_po_file is not None:
        surviving_segments = _segments_used_by(source_rels)
        for entry in list(paths_po_file):
            if entry.obsolete:
                continue
            msgctxt = entry.msgctxt or ""
            if not msgctxt.startswith("path::segment::"):
                # Foreign rows (operator-edited or future schema
                # extensions) are out of scope — only prune entries
                # this module is sure it understands.
                continue
            if entry.msgid in surviving_segments:
                continue
            paths_po_entries_to_drop.append(entry)
            removed_paths_po_msgctxts.append(msgctxt)

    sorted_targets = sorted(removed_targets)
    sorted_pos = sorted(removed_pos)
    sorted_paths_po_msgctxts = sorted(removed_paths_po_msgctxts)
    sorted_path_map_keys = sorted(removed_path_map_keys)

    if dry_run:
        return CleanupResult(
            removed_targets=sorted_targets,
            removed_pos=sorted_pos,
            removed_paths_po_entries=sorted_paths_po_msgctxts,
            removed_path_map_entries=sorted_path_map_keys,
            dry_run=True,
        )

    # Apply deletes. Each operation is independent — a failure on one
    # row should not abort the others (the operator can re-run cleanup
    # to mop up the rest). Per-failure logs surface the path AND we
    # collect them into ``failures`` so the CLI can signal a non-zero
    # exit; CI must not treat a partial cleanup as a clean run.
    failures: list[Path] = []
    # Track src_rel keys whose target OR PO deletion failed: the
    # path_map row remains the only reverse pointer a retry needs, so
    # preserving it lets the next run resume from the same classification.
    preserve_keys: set[str] = set()
    for tf in sorted_targets:
        try:
            tf.unlink()
        except OSError:
            logger.exception("Failed to remove orphan target %s", tf)
            failures.append(tf)
            owner = target_owner.get(tf)
            if owner is not None:
                preserve_keys.add(owner)

    for po in sorted_pos:
        try:
            po.unlink()
        except OSError:
            logger.exception("Failed to remove orphan PO %s", po)
            failures.append(po)
            owner = po_owner.get(po)
            if owner is not None:
                preserve_keys.add(owner)

    effective_path_map_drops = [
        k for k in sorted_path_map_keys if k not in preserve_keys
    ]
    if effective_path_map_drops:
        for key in effective_path_map_drops:
            path_map.pop(key, None)
        try:
            path_map_json_path.parent.mkdir(parents=True, exist_ok=True)
            path_map_json_path.write_text(
                json.dumps(
                    path_map,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        except OSError:
            logger.exception(
                "Failed to update %s", path_map_json_path
            )
            failures.append(path_map_json_path)

    if paths_po_file is not None and paths_po_entries_to_drop:
        for entry in paths_po_entries_to_drop:
            try:
                paths_po_file.remove(entry)
            except ValueError:
                continue
        try:
            paths_po_file.save(str(paths_po_path))
        except OSError:
            logger.exception("Failed to update %s", paths_po_path)
            failures.append(paths_po_path)

    return CleanupResult(
        removed_targets=sorted_targets,
        removed_pos=sorted_pos,
        removed_paths_po_entries=sorted_paths_po_msgctxts,
        removed_path_map_entries=effective_path_map_drops,
        failures=sorted(failures),
        dry_run=False,
    )


def format_human_report(result: CleanupResult) -> str:
    """Render a human-readable summary of the (planned) cleanup.

    Stable across ``--dry-run`` and real runs so operators can compare
    a preview against the actual outcome line by line. Items are
    already sorted in :func:`cleanup_target_tree`.
    """
    lines: list[str] = []
    header = "DRY RUN" if result.dry_run else "CLEANUP"
    lines.append(f"=== {header} ===")
    if result.is_noop():
        lines.append("Nothing to remove. Target tree is in sync with source.")
        return "\n".join(lines)
    if result.removed_targets:
        lines.append(
            f"Orphan target files ({len(result.removed_targets)}):"
        )
        for p in result.removed_targets:
            lines.append(f"  - {p}")
    if result.removed_pos:
        lines.append(f"Orphan per-document POs ({len(result.removed_pos)}):")
        for p in result.removed_pos:
            lines.append(f"  - {p}")
    if result.removed_path_map_entries:
        lines.append(
            f"Stale path_map.json entries "
            f"({len(result.removed_path_map_entries)}):"
        )
        for key in result.removed_path_map_entries:
            lines.append(f"  - {key}")
    if result.removed_paths_po_entries:
        lines.append(
            f"Unused _paths.po segment entries "
            f"({len(result.removed_paths_po_entries)}):"
        )
        for ctx in result.removed_paths_po_entries:
            lines.append(f"  - {ctx}")
    if result.failures:
        lines.append(f"Failed apply steps ({len(result.failures)}):")
        for p in result.failures:
            lines.append(f"  - {p}")
    return "\n".join(lines)


def add_cleanup_subparser(
    sub: "argparse._SubParsersAction[argparse.ArgumentParser]",
) -> argparse.ArgumentParser:
    """Register the ``cleanup`` subcommand on the top-level argparse.

    Kept as a public helper so :mod:`mdpo_llm.__main__` can attach the
    subparser without importing the implementation symbols
    individually, mirroring the ``add_lint_subparser`` pattern from
    :mod:`mdpo_llm.cli_lint`.
    """
    p = sub.add_parser(
        "cleanup",
        help=(
            "Remove orphaned translated files (and PO siblings) from a "
            "target tree whose source has been deleted since the last "
            "translate-dir run."
        ),
        description=(
            "Walk a translated markdown tree and remove files whose "
            "source no longer exists under --source. Also prunes stale "
            "path_map.json entries and unused _paths.po segment rows. "
            "Never moves target files (they may have been hand-edited): "
            "renames surface as orphan-plus-new-translation, and the "
            "operator must re-run translate-dir to mint the new target."
        ),
    )
    p.add_argument(
        "target_dir",
        help="Translated markdown directory to clean (scanned recursively).",
    )
    p.add_argument(
        "--source",
        dest="source_dir",
        required=True,
        help=(
            "Source directory the translation ran against. Required: "
            "the absence of every translation's source would otherwise "
            "be indistinguishable from a wrong-directory invocation."
        ),
    )
    p.add_argument(
        "--po-dir",
        dest="po_dir",
        type=Path,
        default=None,
        help=(
            "Optional override for the per-document PO directory. "
            "Defaults to TARGET_DIR (matching the default layout "
            "translate-dir writes when --po-dir was not used)."
        ),
    )
    p.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help=(
            "Print what would be removed without acting. The header "
            "differs from a real run (\"DRY RUN\" vs \"CLEANUP\") but "
            "the per-section body lists match the same classification "
            "the real run would emit, so a preview / diff workflow "
            "stays predictable."
        ),
    )
    p.add_argument(
        "--keep-po",
        dest="keep_po",
        action="store_true",
        help=(
            "Remove the orphan target markdown but preserve its "
            "sibling per-document PO file. A subsequent translate-dir "
            "run can then re-emit the target from the cached "
            "translation if the source comes back."
        ),
    )
    p.add_argument(
        "--json",
        dest="json_out",
        action="store_true",
        help=(
            "Emit the cleanup summary as JSON to stdout instead of the "
            "human-readable report. Schema: "
            "{dry_run: bool, removed_targets: [str], removed_pos: [str], "
            "removed_path_map_entries: [str], "
            "removed_paths_po_entries: [str]}."
        ),
    )
    p.set_defaults(func=cmd_cleanup)
    return p


def cmd_cleanup(args: argparse.Namespace) -> int:
    """``mdpo-llm cleanup`` entry point.

    Exit code contract:
      * ``2`` — usage error (non-directory source / target / po-dir,
        or ``--source`` missing). Surfaced before any filesystem
        mutation.
      * ``1`` — one or more apply steps failed (a file could not be
        unlinked, a metadata write failed). Partial work is preserved
        on disk; re-running the verb mops up the rest.
      * ``0`` — otherwise, including successful runs that removed
        zero files (the cleanup is idempotent). A missing
        ``target_dir`` is treated as a no-op so CI pipelines that
        always invoke ``cleanup`` after ``translate-dir`` do not
        choke on the first run.
    """
    target_dir = Path(args.target_dir)
    source_dir = Path(args.source_dir)
    po_dir: Optional[Path] = getattr(args, "po_dir", None)

    # Always route through ``cleanup_target_tree`` rather than early-no-op
    # on a missing target: the library validates ``--source`` (and
    # ``--po-dir``) first, so a typo like
    # ``mdpo-llm cleanup missing-target --source wrong-dir`` still
    # surfaces as a usage error (exit 2) instead of silently succeeding.
    # The library handles a genuinely absent target as a clean no-op
    # itself, so the idempotent-CI behaviour is preserved.
    try:
        result = cleanup_target_tree(
            target_dir=target_dir,
            source_dir=source_dir,
            po_dir=po_dir,
            dry_run=getattr(args, "dry_run", False),
            keep_po=getattr(args, "keep_po", False),
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except NotADirectoryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        # Corrupt path_map.json — operator must repair / delete before
        # cleanup can safely run.
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if getattr(args, "json_out", False):
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(format_human_report(result))
    if result.failures:
        return 1
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Standalone entry point: ``python -m mdpo_llm.cleanup_ops …``.

    Mirrors the ``mdpo-llm cleanup`` subcommand surface so the module
    can be driven directly without going through the top-level parser
    — convenient for ad-hoc runs and for tests that want to exercise
    the CLI surface in isolation.
    """
    parser = argparse.ArgumentParser(prog="mdpo-llm-cleanup")
    sub = parser.add_subparsers(dest="command", required=True)
    add_cleanup_subparser(sub)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
