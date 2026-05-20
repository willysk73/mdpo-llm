"""Tests for the standalone ``mdpo-llm cleanup`` verb (T-20).

Covers the three cases the brief calls out plus the layouts the
cleanup must handle (mirror, translate-paths, separate --po-dir),
``--dry-run`` byte-stability, ``--keep-po`` preservation, and the
usage errors the CLI surfaces with exit code 2.
"""

from __future__ import annotations

import json
from pathlib import Path

import polib
import pytest

from mdpo_llm.cleanup_ops import (
    CleanupResult,
    cleanup_target_tree,
    cmd_cleanup,
    format_human_report,
    main,
)


def _write(path: Path, content: str = "x\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _write_path_map(target_dir: Path, mapping: dict[str, str]) -> None:
    """Drop a ``path_map.json`` matching what `translate-dir --translate-paths`
    would emit. Tests that exercise the ``_paths.po`` segment catalog must
    also write this file — its presence (alongside _paths.po) is what
    distinguishes a translate-paths layout from the mirror layout.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "path_map.json").write_text(
        json.dumps(mapping, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_paths_po(path: Path, entries: list[tuple[str, str]]) -> None:
    """Write a `_paths.po` with the supplied ``(msgid, msgstr)`` rows.

    ``msgctxt`` is fixed to ``"path::segment::<msgid>"`` so the rows
    look like what :meth:`MarkdownProcessor._translate_path_segments`
    actually emits.
    """
    po = polib.POFile()
    po.metadata = {"Content-Type": "text/plain; charset=UTF-8"}
    for msgid, msgstr in entries:
        po.append(
            polib.POEntry(
                msgctxt=f"path::segment::{msgid}",
                msgid=msgid,
                msgstr=msgstr,
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    po.save(str(path))


# ---------------------------------------------------------------------------
# Mirror layout (no --translate-paths, no path_map.json)
# ---------------------------------------------------------------------------


def test_orphan_detection_mirror_layout(tmp_path: Path) -> None:
    """Source file gone → target + sibling PO removed."""
    source = tmp_path / "src"
    target = tmp_path / "tgt"
    _write(source / "alive.md")
    _write(target / "alive.md")
    _write(target / "alive.po", "msgid \"\"\nmsgstr \"\"\n")
    _write(target / "orphan.md")
    _write(target / "orphan.po", "msgid \"\"\nmsgstr \"\"\n")

    result = cleanup_target_tree(target, source)

    assert result.removed_targets == [target / "orphan.md"]
    assert result.removed_pos == [target / "orphan.po"]
    assert not result.dry_run
    assert not (target / "orphan.md").exists()
    assert not (target / "orphan.po").exists()
    # Live files untouched.
    assert (target / "alive.md").exists()
    assert (target / "alive.po").exists()


def test_nested_orphan_mirror_layout(tmp_path: Path) -> None:
    """Orphans under nested dirs are detected too."""
    source = tmp_path / "src"
    target = tmp_path / "tgt"
    _write(source / "guide/intro.md")
    _write(target / "guide/intro.md")
    _write(target / "guide/intro.po", "msgid \"\"\nmsgstr \"\"\n")
    _write(target / "guide/removed.md")
    _write(target / "guide/removed.po", "msgid \"\"\nmsgstr \"\"\n")

    result = cleanup_target_tree(target, source)

    assert result.removed_targets == [target / "guide/removed.md"]
    assert result.removed_pos == [target / "guide/removed.po"]


def test_uppercase_md_source_preserved(tmp_path: Path) -> None:
    """A live ``README.MD`` source on a case-sensitive FS must not orphan its target.

    The target walker uses ``suffix.lower()`` so ``README.MD`` IS a
    cleanup candidate. If the source scan only matched lowercase
    ``.md``, the surviving source would look missing and the live
    translation would be deleted.
    """
    source = tmp_path / "src"
    target = tmp_path / "tgt"
    _write(source / "README.MD")
    _write(target / "README.MD")

    result = cleanup_target_tree(target, source)
    assert result.is_noop()
    assert (target / "README.MD").exists()


def test_non_md_files_left_alone(tmp_path: Path) -> None:
    """Anything outside ``TARGET_EXTENSIONS`` is operator content, untouched."""
    source = tmp_path / "src"
    target = tmp_path / "tgt"
    _write(source / "alive.md")
    _write(target / "alive.md")
    _write(target / "screenshot.png", "binary")
    _write(target / "data.json", "{}")

    cleanup_target_tree(target, source)

    assert (target / "screenshot.png").exists()
    assert (target / "data.json").exists()
    assert (target / "alive.md").exists()


def test_dry_run_makes_no_changes(tmp_path: Path) -> None:
    """``--dry-run`` reports planned removals but touches nothing."""
    source = tmp_path / "src"
    target = tmp_path / "tgt"
    _write(source / "alive.md")
    _write(target / "alive.md")
    _write(target / "orphan.md")
    _write(target / "orphan.po", "msgid \"\"\nmsgstr \"\"\n")

    real = cleanup_target_tree(target, source, dry_run=True)

    assert real.dry_run
    assert real.removed_targets == [target / "orphan.md"]
    assert real.removed_pos == [target / "orphan.po"]
    # Nothing actually removed.
    assert (target / "orphan.md").exists()
    assert (target / "orphan.po").exists()


def test_dry_run_matches_real_run(tmp_path: Path) -> None:
    """Same tree → same classification dry-run vs real run."""
    source = tmp_path / "src"
    target_a = tmp_path / "tgt_a"
    target_b = tmp_path / "tgt_b"
    for tgt in (target_a, target_b):
        _write(source / "alive.md")
        _write(tgt / "alive.md")
        _write(tgt / "orphan.md")
        _write(tgt / "orphan.po", "msgid \"\"\nmsgstr \"\"\n")

    dry = cleanup_target_tree(target_a, source, dry_run=True)
    real = cleanup_target_tree(target_b, source, dry_run=False)

    # Strip the tree-specific path prefixes so the two reports compare.
    def _rel(paths: list[Path], root: Path) -> list[str]:
        return [str(p.relative_to(root)) for p in paths]

    assert _rel(dry.removed_targets, target_a) == _rel(
        real.removed_targets, target_b
    )
    assert _rel(dry.removed_pos, target_a) == _rel(real.removed_pos, target_b)
    assert dry.removed_path_map_entries == real.removed_path_map_entries
    assert dry.removed_paths_po_entries == real.removed_paths_po_entries


def test_keep_po_preserves_sibling(tmp_path: Path) -> None:
    """``--keep-po`` keeps the PO so a returning source can re-emit."""
    source = tmp_path / "src"
    target = tmp_path / "tgt"
    _write(source / "alive.md")
    _write(target / "alive.md")
    _write(target / "orphan.md")
    _write(target / "orphan.po", "msgid \"\"\nmsgstr \"\"\n")

    result = cleanup_target_tree(target, source, keep_po=True)

    assert result.removed_targets == [target / "orphan.md"]
    assert result.removed_pos == []
    assert not (target / "orphan.md").exists()
    assert (target / "orphan.po").exists()


# ---------------------------------------------------------------------------
# Edge cases (empty / missing / errors)
# ---------------------------------------------------------------------------


def test_empty_target_dir_noop(tmp_path: Path) -> None:
    source = tmp_path / "src"
    target = tmp_path / "tgt"
    source.mkdir()
    target.mkdir()
    _write(source / "alive.md")

    result = cleanup_target_tree(target, source)

    assert result.is_noop()


def test_missing_target_returns_noop(tmp_path: Path) -> None:
    """Absent target tree returns a clean no-op (idempotent)."""
    source = tmp_path / "src"
    source.mkdir()
    _write(source / "alive.md")

    result = cleanup_target_tree(tmp_path / "no_such_target", source)

    assert result.is_noop()


def test_missing_source_dir_errors(tmp_path: Path) -> None:
    target = tmp_path / "tgt"
    _write(target / "alive.md")
    with pytest.raises(FileNotFoundError):
        cleanup_target_tree(target, tmp_path / "no_such_source")


def test_source_is_file_errors(tmp_path: Path) -> None:
    target = tmp_path / "tgt"
    target.mkdir()
    source_file = tmp_path / "not_a_dir.md"
    source_file.write_text("x", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        cleanup_target_tree(target, source_file)


def test_target_is_file_errors(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    target_file = tmp_path / "not_a_dir.md"
    target_file.write_text("x", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        cleanup_target_tree(target_file, source)


def test_target_same_as_source_refuses(tmp_path: Path) -> None:
    """Refuse the wipe-the-source case where target_dir == source_dir."""
    shared = tmp_path / "docs"
    _write(shared / "foo.md")
    with pytest.raises(ValueError, match="same path"):
        cleanup_target_tree(shared, shared)
    assert (shared / "foo.md").exists()


def test_source_inside_target_refuses(tmp_path: Path) -> None:
    """Refuse when target is an ancestor of source.

    ``mdpo-llm cleanup . --source docs`` would otherwise classify
    every real source file as orphaned (``docs/foo.md`` not in
    ``{"foo.md"}``) and delete the source tree.
    """
    target = tmp_path / "repo"
    source = target / "docs"
    _write(source / "foo.md")
    with pytest.raises(ValueError, match="inside target_dir"):
        cleanup_target_tree(target, source)
    assert (source / "foo.md").exists()


def test_po_dir_not_a_dir_errors(tmp_path: Path) -> None:
    source = tmp_path / "src"
    target = tmp_path / "tgt"
    source.mkdir()
    target.mkdir()
    po_dir_file = tmp_path / "po_is_a_file"
    po_dir_file.write_text("x", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        cleanup_target_tree(target, source, po_dir=po_dir_file)


# ---------------------------------------------------------------------------
# translate-paths layout (path_map.json + _paths.po)
# ---------------------------------------------------------------------------


def test_orphan_detection_translate_paths_layout(tmp_path: Path) -> None:
    """``path_map.json`` reverse lookup drives orphan detection."""
    source = tmp_path / "src"
    target = tmp_path / "tgt"
    _write(source / "guide/intro.md")
    _write(target / "ko/intro.md")
    _write(target / "guide/intro.po", "msgid \"\"\nmsgstr \"\"\n")
    _write(target / "ko/removed.md")
    _write(target / "guide/removed.po", "msgid \"\"\nmsgstr \"\"\n")
    (target / "path_map.json").write_text(
        json.dumps(
            {
                "guide/intro.md": "ko/intro.md",
                "guide/removed.md": "ko/removed.md",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    result = cleanup_target_tree(target, source)

    assert result.removed_targets == [target / "ko/removed.md"]
    assert result.removed_pos == [target / "guide/removed.po"]
    assert result.removed_path_map_entries == ["guide/removed.md"]

    # path_map.json rewritten in place with the stale entry dropped.
    new_map = json.loads((target / "path_map.json").read_text(encoding="utf-8"))
    assert new_map == {"guide/intro.md": "ko/intro.md"}


def test_paths_po_segment_pruning(tmp_path: Path) -> None:
    """Segments unused by surviving sources drop from ``_paths.po``."""
    source = tmp_path / "src"
    target = tmp_path / "tgt"
    _write(source / "guide/intro.md")
    _write(target / "guide/intro.md")
    _write_paths_po(
        target / "_paths.po",
        [
            ("guide", "guide-ko"),
            ("intro", "intro-ko"),
            # Orphan segment — no surviving source uses it.
            ("legacy", "legacy-ko"),
        ],
    )
    _write_path_map(target, {"guide/intro.md": "guide/intro.md"})

    result = cleanup_target_tree(target, source)

    assert result.removed_paths_po_entries == ["path::segment::legacy"]
    pruned = polib.pofile(str(target / "_paths.po"), encoding="utf-8")
    msgctxts = {e.msgctxt for e in pruned if not e.obsolete}
    assert "path::segment::legacy" not in msgctxts
    # Surviving segments remain.
    assert "path::segment::guide" in msgctxts
    assert "path::segment::intro" in msgctxts


def test_paths_po_preserved_when_segment_still_used(tmp_path: Path) -> None:
    """A segment shared across many sources stays after one is removed."""
    source = tmp_path / "src"
    target = tmp_path / "tgt"
    # ``guide`` appears in two sources; only one is deleted.
    _write(source / "guide/intro.md")
    _write(target / "guide/intro.md")
    _write(target / "guide/setup.md")  # orphan target
    _write_paths_po(
        target / "_paths.po",
        [
            ("guide", "guide-ko"),
            ("intro", "intro-ko"),
            ("setup", "setup-ko"),
        ],
    )
    _write_path_map(
        target,
        {
            "guide/intro.md": "guide/intro.md",
            "guide/setup.md": "guide/setup.md",
        },
    )

    result = cleanup_target_tree(target, source)

    # ``setup`` is orphaned but ``guide`` is still in use → only setup goes.
    assert result.removed_paths_po_entries == ["path::segment::setup"]


def test_po_dir_routes_per_document_po(tmp_path: Path) -> None:
    """``--po-dir`` reroutes both per-doc PO and ``_paths.po`` lookups."""
    source = tmp_path / "src"
    target = tmp_path / "tgt"
    po_dir = tmp_path / "po"
    _write(source / "alive.md")
    _write(target / "alive.md")
    _write(target / "orphan.md")
    _write(po_dir / "alive.po", "msgid \"\"\nmsgstr \"\"\n")
    _write(po_dir / "orphan.po", "msgid \"\"\nmsgstr \"\"\n")
    _write_paths_po(
        po_dir / "_paths.po",
        [
            ("alive", "alive-ko"),
            ("orphan", "orphan-ko"),
        ],
    )
    _write_path_map(
        target,
        {"alive.md": "alive.md", "orphan.md": "orphan.md"},
    )

    result = cleanup_target_tree(target, source, po_dir=po_dir)

    assert result.removed_targets == [target / "orphan.md"]
    assert result.removed_pos == [po_dir / "orphan.po"]
    assert result.removed_paths_po_entries == ["path::segment::orphan"]
    assert not (po_dir / "orphan.po").exists()
    assert (po_dir / "alive.po").exists()


def test_path_map_stale_only_no_targets(tmp_path: Path) -> None:
    """A path_map row whose source AND target are gone still gets dropped."""
    source = tmp_path / "src"
    target = tmp_path / "tgt"
    target.mkdir()
    source.mkdir()
    _write(source / "alive.md")
    _write(target / "alive.md")
    (target / "path_map.json").write_text(
        json.dumps(
            {
                "alive.md": "alive.md",
                "ghost.md": "ghost-ko.md",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    result = cleanup_target_tree(target, source)

    assert result.removed_targets == []
    assert "ghost.md" in result.removed_path_map_entries
    new_map = json.loads((target / "path_map.json").read_text(encoding="utf-8"))
    assert new_map == {"alive.md": "alive.md"}


def test_path_map_escape_via_dotdot_key_refuses(tmp_path: Path) -> None:
    """A ``..``-bearing source key triggers the corrupt-map abort.

    Before the path_map validation it would have leaked through to the
    per-target loop and forced the runtime escape check to skip the
    bogus PO. The earlier abort is the cleaner contract: any unsafe
    path_map row is structurally wrong, not just resolution-wrong.
    """
    source = tmp_path / "src"
    target = tmp_path / "tgt"
    elsewhere = tmp_path / "elsewhere"
    source.mkdir()
    target.mkdir()
    elsewhere.mkdir()
    _write(elsewhere / "victim.po", "msgid \"\"\nmsgstr \"\"\n")
    _write(target / "orphan.md")
    (target / "path_map.json").write_text(
        json.dumps({"../elsewhere/victim": "orphan.md"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="path_map.json"):
        cleanup_target_tree(target, source)
    # Bystander PO still on disk; nothing got unlinked.
    assert (elsewhere / "victim.po").exists()
    assert (target / "orphan.md").exists()


def test_path_map_escape_via_absolute_key_refuses(tmp_path: Path) -> None:
    """An absolute-path source key also trips the corrupt-map abort."""
    source = tmp_path / "src"
    target = tmp_path / "tgt"
    elsewhere = tmp_path / "elsewhere"
    source.mkdir()
    target.mkdir()
    elsewhere.mkdir()
    victim = elsewhere / "victim.po"
    _write(victim, "msgid \"\"\nmsgstr \"\"\n")
    _write(target / "orphan.md")
    (target / "path_map.json").write_text(
        json.dumps({str(elsewhere / "victim"): "orphan.md"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="path_map.json"):
        cleanup_target_tree(target, source)
    assert victim.exists()


def test_empty_key_in_path_map_refuses_to_run(tmp_path: Path) -> None:
    """A row with an empty source key is treated as a corrupt map.

    Dropping it silently would let a translate-paths tree mirror-fallback
    for the affected target and delete a live translation — the same
    correctness pitfall that justifies the broader corrupt-map abort.
    """
    source = tmp_path / "src"
    target = tmp_path / "tgt"
    _write(source / "alive.md")
    _write(target / "alive.md")
    (target / "path_map.json").write_text(
        json.dumps(
            {
                "": "orphan.md",
                "alive.md": "alive.md",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="path_map.json"):
        cleanup_target_tree(target, source)


def test_empty_value_in_path_map_refuses_to_run(tmp_path: Path) -> None:
    """An empty target value also counts as corrupt — refuse the run.

    In a translate-paths tree ``{"guide/intro.md": ""}`` would otherwise
    accept the row, lose the reverse lookup for the actual localized
    target, mirror-fallback, and delete a live translation.
    """
    source = tmp_path / "src"
    target = tmp_path / "tgt"
    _write(source / "guide/intro.md")
    _write(target / "ko/intro.md")
    (target / "path_map.json").write_text(
        json.dumps({"guide/intro.md": ""}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="path_map.json"):
        cleanup_target_tree(target, source)
    assert (target / "ko/intro.md").exists()


def test_dotdot_in_path_map_value_refuses_to_run(tmp_path: Path) -> None:
    """A ``..``-bearing target value is also corrupt — refuse the run.

    ``{"guide/intro.md": "../ko/intro.md"}`` would otherwise leave the
    on-disk target ``ko/intro.md`` unmatched, the mirror-fallback
    would classify it as orphan, and a live translation would be
    deleted.
    """
    source = tmp_path / "src"
    target = tmp_path / "tgt"
    _write(source / "guide/intro.md")
    _write(target / "ko/intro.md")
    (target / "path_map.json").write_text(
        json.dumps({"guide/intro.md": "../ko/intro.md"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="path_map.json"):
        cleanup_target_tree(target, source)
    assert (target / "ko/intro.md").exists()


def test_absolute_path_in_path_map_value_refuses_to_run(tmp_path: Path) -> None:
    """An absolute-path target value is also corrupt — refuse the run."""
    source = tmp_path / "src"
    target = tmp_path / "tgt"
    _write(source / "guide/intro.md")
    _write(target / "ko/intro.md")
    (target / "path_map.json").write_text(
        json.dumps({"guide/intro.md": "/tmp/evil/intro.md"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="path_map.json"):
        cleanup_target_tree(target, source)
    assert (target / "ko/intro.md").exists()


def test_non_canonical_path_map_refuses_to_run(tmp_path: Path) -> None:
    """Non-canonical forms (``ko//intro.md``, ``ko/./intro.md``,
    backslash-separated) miss the canonical target-walk lookup.

    Letting them through would mirror-fallback and delete the live
    localized target. Surface a clear repair message instead.
    """
    source = tmp_path / "src"
    target = tmp_path / "tgt"
    _write(source / "guide/intro.md")
    _write(target / "ko/intro.md")
    (target / "path_map.json").write_text(
        json.dumps({"guide/intro.md": "ko//intro.md"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="canonical"):
        cleanup_target_tree(target, source)
    assert (target / "ko/intro.md").exists()


def test_colon_in_filename_accepted(tmp_path: Path) -> None:
    """A POSIX filename like ``a:b.md`` is legal — must not be rejected.

    The Windows-drive check used to fire on any colon at position 2,
    which made cleanup unusable for trees containing such filenames.
    """
    source = tmp_path / "src"
    target = tmp_path / "tgt"
    _write(source / "a:b.md")
    _write(target / "a:b-ko.md")
    (target / "path_map.json").write_text(
        json.dumps({"a:b.md": "a:b-ko.md"}),
        encoding="utf-8",
    )
    # Should not raise.
    result = cleanup_target_tree(target, source)
    assert result.is_noop()


def test_duplicate_target_in_path_map_refuses_to_run(tmp_path: Path) -> None:
    """Two source keys mapping to the same target is structurally invalid.

    Reverse lookup ``{tgt: src for src, tgt in raw.items()}`` silently
    drops one owner. If the surviving source's row got dropped, the
    target walk would classify the live translation as an orphan and
    delete it.
    """
    source = tmp_path / "src"
    target = tmp_path / "tgt"
    _write(source / "alive.md")
    _write(target / "out.md")
    (target / "path_map.json").write_text(
        json.dumps(
            {
                "alive.md": "out.md",
                "ghost.md": "out.md",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate target"):
        cleanup_target_tree(target, source)
    assert (target / "out.md").exists()


def test_non_string_value_in_path_map_refuses_to_run(tmp_path: Path) -> None:
    """A non-string value in path_map.json must abort, not be silently dropped.

    Dropping the row leaves the target-walk to mirror-fallback for the
    affected source, which would delete a live localized translation
    (e.g. ``ko/intro.md`` when source is ``guide/intro.md``).
    """
    source = tmp_path / "src"
    target = tmp_path / "tgt"
    _write(source / "guide/intro.md")
    _write(target / "ko/intro.md")
    (target / "path_map.json").write_text(
        json.dumps({"guide/intro.md": 123}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="path_map.json"):
        cleanup_target_tree(target, source)
    # Live translation untouched.
    assert (target / "ko/intro.md").exists()


def test_mirror_tree_with_paths_md_source_does_not_refuse(
    tmp_path: Path,
) -> None:
    """A mirror-layout source named ``_paths.md`` produces ``target/_paths.po``.

    That sibling PO is NOT a translate-paths segment catalog — its
    entries are parser-context-keyed, not ``path::segment::*``. The
    layout-ambiguity guard must distinguish the two and let the
    cleanup proceed normally.
    """
    source = tmp_path / "src"
    target = tmp_path / "tgt"
    _write(source / "_paths.md")
    _write(target / "_paths.md")
    # Per-document PO at the same filename as the segment catalog,
    # but contents are an ordinary content-block PO (no
    # ``path::segment::*`` msgctxt).
    po = polib.POFile()
    po.metadata = {"Content-Type": "text/plain; charset=UTF-8"}
    po.append(
        polib.POEntry(
            msgctxt="paragraph-0",
            msgid="hello",
            msgstr="안녕",
        )
    )
    (target / "_paths.po").parent.mkdir(parents=True, exist_ok=True)
    po.save(str(target / "_paths.po"))

    # No raise — guard correctly skips when the file lacks segment-catalog
    # signature, and the cleanup runs (zero orphans here).
    result = cleanup_target_tree(target, source)
    assert result.is_noop()


def test_corrupt_path_map_refuses_to_run(tmp_path: Path) -> None:
    """A present-but-unparseable ``path_map.json`` must abort, not mirror-fallback.

    In a translate-paths tree the targets are localized (``ko/intro.md``)
    and the sources are ``guide/intro.md``. Mirror-fallback would
    classify the live translation as orphan and delete it. The cleanup
    refuses the destructive fallback and surfaces a usage error instead.
    """
    source = tmp_path / "src"
    target = tmp_path / "tgt"
    _write(source / "guide/intro.md")
    _write(target / "ko/intro.md")
    (target / "path_map.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="path_map.json"):
        cleanup_target_tree(target, source)
    # Live translation is still on disk.
    assert (target / "ko/intro.md").exists()


def test_paths_po_without_path_map_refuses_to_run(tmp_path: Path) -> None:
    """A ``--translate-paths`` tree missing path_map.json must NOT mirror-fallback.

    The presence of ``_paths.po`` is the signature of a translate-paths
    layout. If ``path_map.json`` is missing (operator deleted it, a
    backup was only partial, …), the cleanup has no reverse map and
    would otherwise wrongly classify live localized targets as orphans.
    Refuse to run instead.
    """
    source = tmp_path / "src"
    target = tmp_path / "tgt"
    _write(source / "guide/intro.md")
    _write(target / "ko/intro.md")
    _write_paths_po(
        target / "_paths.po",
        [("guide", "ko"), ("intro", "intro")],
    )

    with pytest.raises(ValueError, match="path_map.json"):
        cleanup_target_tree(target, source)
    # Live localized translation untouched.
    assert (target / "ko/intro.md").exists()


def test_unreadable_paths_po_without_path_map_refuses_to_run(
    tmp_path: Path,
) -> None:
    """A corrupt ``_paths.po`` + missing path_map.json is the danger case.

    If we treated unparseable as "definitely not the catalog", the
    cleanup would mirror-fallback and could delete live localized
    translations. The helper returns True on parse failure so the
    layout-ambiguity guard fires defensively.
    """
    source = tmp_path / "src"
    target = tmp_path / "tgt"
    _write(source / "guide/intro.md")
    _write(target / "ko/intro.md")
    _write(target / "_paths.po", "not a valid PO\n")

    with pytest.raises(ValueError, match="path_map.json"):
        cleanup_target_tree(target, source)
    assert (target / "ko/intro.md").exists()


def test_non_dict_path_map_refuses_to_run(tmp_path: Path) -> None:
    """A path_map.json whose root is not an object also refuses to run."""
    source = tmp_path / "src"
    target = tmp_path / "tgt"
    _write(source / "alive.md")
    _write(target / "alive.md")
    (target / "path_map.json").write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="path_map.json"):
        cleanup_target_tree(target, source)


def test_cmd_cleanup_corrupt_path_map_exits_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI surfaces a corrupt path_map.json as exit 2."""
    source = tmp_path / "src"
    target = tmp_path / "tgt"
    _write(source / "alive.md")
    _write(target / "alive.md")
    (target / "path_map.json").write_text("{not json", encoding="utf-8")

    rc = main(["cleanup", str(target), "--source", str(source)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "path_map.json" in err


def test_stale_path_map_row_also_removes_orphan_po(tmp_path: Path) -> None:
    """A path_map row whose source is gone (target also gone) must still
    schedule its sibling PO for deletion.

    Otherwise the orphan PO is stranded — after the row is dropped, no
    later cleanup can reconstruct the source-relative PO path under
    ``--po-dir`` or translated-path layouts.
    """
    source = tmp_path / "src"
    target = tmp_path / "tgt"
    po_dir = tmp_path / "po"
    source.mkdir()
    target.mkdir()
    _write(source / "alive.md")
    _write(target / "alive-ko.md")
    _write(po_dir / "alive.po", "msgid \"\"\nmsgstr \"\"\n")
    # Orphan: source AND target gone, but the PO is still on disk.
    orphan_po = _write(po_dir / "ghost.po", "msgid \"\"\nmsgstr \"\"\n")
    (target / "path_map.json").write_text(
        json.dumps(
            {
                "alive.md": "alive-ko.md",
                "ghost.md": "ghost-ko.md",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    result = cleanup_target_tree(target, source, po_dir=po_dir)

    assert orphan_po in result.removed_pos
    assert "ghost.md" in result.removed_path_map_entries
    assert not orphan_po.exists()


def test_corrupt_paths_po_does_not_abort(tmp_path: Path) -> None:
    """A garbage ``_paths.po`` is logged & skipped, not raised.

    A translate-paths layout still requires a parseable path_map.json
    alongside, so we provide one to exercise the segment-pruning skip
    rather than the layout-ambiguity guard.
    """
    source = tmp_path / "src"
    target = tmp_path / "tgt"
    _write(source / "alive.md")
    _write(target / "alive.md")
    _write(target / "_paths.po", "this is not a valid PO file\n")
    _write_path_map(target, {"alive.md": "alive.md"})

    # Should not raise; segment pruning is skipped.
    result = cleanup_target_tree(target, source)
    assert result.removed_paths_po_entries == []


# ---------------------------------------------------------------------------
# Format & CLI behaviour
# ---------------------------------------------------------------------------


def test_format_human_report_noop_message(tmp_path: Path) -> None:
    report = format_human_report(CleanupResult(dry_run=True))
    assert "DRY RUN" in report
    assert "Nothing to remove" in report


def test_format_human_report_lists_paths(tmp_path: Path) -> None:
    result = CleanupResult(
        removed_targets=[Path("a/b.md")],
        removed_pos=[Path("a/b.po")],
        removed_path_map_entries=["a/b.md"],
        removed_paths_po_entries=["path::segment::a"],
    )
    text = format_human_report(result)
    assert "CLEANUP" in text
    assert "a/b.md" in text
    assert "a/b.po" in text
    assert "path::segment::a" in text


def test_cmd_cleanup_returns_zero_on_noop(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "src"
    target = tmp_path / "tgt"
    source.mkdir()
    target.mkdir()
    rc = main(
        ["cleanup", str(target), "--source", str(source)],
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Nothing to remove" in out


def test_cmd_cleanup_json_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "src"
    target = tmp_path / "tgt"
    _write(source / "alive.md")
    _write(target / "alive.md")
    _write(target / "orphan.md")
    rc = main(
        [
            "cleanup",
            str(target),
            "--source",
            str(source),
            "--dry-run",
            "--json",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["dry_run"] is True
    assert any(p.endswith("orphan.md") for p in parsed["removed_targets"])


def test_cmd_cleanup_missing_source_exits_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "tgt"
    target.mkdir()
    rc = main(
        [
            "cleanup",
            str(target),
            "--source",
            str(tmp_path / "no_such_source"),
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "source directory does not exist" in err


def test_cmd_cleanup_missing_target_still_validates_source(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing target must NOT mask a typo in ``--source``.

    The library validates ``--source`` first, so a caller who fat-fingered
    both arguments still sees the source error (exit 2) instead of a
    silent zero-removal success that hides the bug.
    """
    rc = main(
        [
            "cleanup",
            str(tmp_path / "no_such_target"),
            "--source",
            str(tmp_path / "no_such_source"),
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "source directory does not exist" in err


def test_path_map_row_preserved_when_po_delete_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In translate-paths layout the path_map row is the only reverse
    pointer from translated tgt_rel back to source-relative src_rel.

    If we drop the row after a failed PO unlink, a retry has no way to
    find the orphan PO at ``po_dir / src_rel.with_suffix(".po")`` —
    lookup falls back to ``tgt_rel`` and probes the wrong path. The
    cleanup must therefore preserve the path_map row when ANY apply
    step for that key failed.
    """
    source = tmp_path / "src"
    target = tmp_path / "tgt"
    po_dir = tmp_path / "po"
    _write(source / "alive.md")
    _write(target / "alive-ko.md")
    _write(po_dir / "alive.po", "msgid \"\"\nmsgstr \"\"\n")
    _write(target / "orphan-ko.md")
    orphan_po = _write(po_dir / "guide/orphan.po", "msgid \"\"\nmsgstr \"\"\n")
    (target / "path_map.json").write_text(
        json.dumps(
            {
                "alive.md": "alive-ko.md",
                "guide/orphan.md": "orphan-ko.md",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    real_unlink = Path.unlink

    def _failing_unlink(self: Path, *args, **kwargs):
        if self == orphan_po:
            raise OSError("simulated permission denied")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", _failing_unlink)

    result = cleanup_target_tree(target, source, po_dir=po_dir)

    # The target deletion succeeded, the PO deletion failed → keep the
    # path_map row so a retry can resolve the PO path via src_rel.
    assert orphan_po in result.failures
    new_map = json.loads((target / "path_map.json").read_text(encoding="utf-8"))
    assert "guide/orphan.md" in new_map
    # And the reported "removed_path_map_entries" must reflect only what
    # actually got dropped, not the planned set.
    assert "guide/orphan.md" not in result.removed_path_map_entries


def test_apply_failure_surfaces_in_failures_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing unlink must surface in ``failures``, not be silently swallowed.

    Simulates a permission-denied / locked-file scenario by monkey-patching
    ``Path.unlink`` to raise on the orphan target. The cleanup must keep
    going (metadata pruning still happens) but record the failure.
    """
    source = tmp_path / "src"
    target = tmp_path / "tgt"
    _write(source / "alive.md")
    _write(target / "alive.md")
    orphan = _write(target / "orphan.md")

    real_unlink = Path.unlink

    def _failing_unlink(self: Path, *args, **kwargs):
        if self == orphan:
            raise OSError("simulated permission denied")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", _failing_unlink)

    result = cleanup_target_tree(target, source)

    assert orphan in result.failures
    # File still on disk because the unlink raised.
    assert orphan.exists()
    # Classification still reports it would have been removed — so a
    # re-run sees the same orphan and can retry.
    assert orphan in result.removed_targets


def test_cmd_cleanup_returns_one_when_apply_fails(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI surfaces apply-step failures as exit 1 (not 0)."""
    source = tmp_path / "src"
    target = tmp_path / "tgt"
    _write(source / "alive.md")
    _write(target / "alive.md")
    orphan = _write(target / "orphan.md")

    real_unlink = Path.unlink

    def _failing_unlink(self: Path, *args, **kwargs):
        if self == orphan:
            raise OSError("simulated permission denied")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", _failing_unlink)

    rc = main(
        ["cleanup", str(target), "--source", str(source)],
    )
    assert rc == 1
    out = capsys.readouterr().out
    assert "Failed apply steps" in out


def test_cmd_cleanup_missing_target_still_validates_po_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Mistyped ``--po-dir`` must surface even when target_dir is absent."""
    source = tmp_path / "src"
    source.mkdir()
    bad_po = tmp_path / "po_typo_is_file"
    bad_po.write_text("oops", encoding="utf-8")

    rc = main(
        [
            "cleanup",
            str(tmp_path / "no_such_target"),
            "--source",
            str(source),
            "--po-dir",
            str(bad_po),
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "po-dir" in err


def test_cmd_cleanup_target_is_file_exits_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "src"
    source.mkdir()
    target_file = tmp_path / "not_a_dir.md"
    target_file.write_text("x", encoding="utf-8")
    rc = main(
        ["cleanup", str(target_file), "--source", str(source)],
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "not a directory" in err
