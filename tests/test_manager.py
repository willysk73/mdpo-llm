"""Tests for POManager sync/fuzzy/obsolete logic."""

from pathlib import Path

import polib
import pytest

from mdpo_llm.manager import POManager
from mdpo_llm.parser import BlockParser


@pytest.fixture
def manager():
    return POManager(skip_types=["hr"])


@pytest.fixture
def fresh_po():
    po = polib.POFile()
    po.metadata = {"Content-Type": "text/plain; charset=UTF-8"}
    return po


@pytest.fixture
def simple_blocks():
    """Minimal block list for testing."""
    return [
        {"type": "heading", "text": "# Title", "path": ["title"], "idx_in_section": 0},
        {"type": "para", "text": "Hello world", "path": ["title"], "idx_in_section": 0},
        {"type": "hr", "text": "---", "path": ["title"], "idx_in_section": 0},
    ]


def ctx_func(block):
    return f"{'/'.join(block['path'])}::{block['type']}:{block['idx_in_section']}"


class TestLoadOrCreatePO:
    def test_create_new(self, manager, tmp_path):
        po_path = tmp_path / "new.po"
        po = manager.load_or_create_po(po_path)
        assert isinstance(po, polib.POFile)
        assert len(po) == 0

    def test_load_existing(self, manager, tmp_path):
        po_path = tmp_path / "existing.po"
        # Create a PO file with one entry
        po = polib.POFile()
        po.metadata = {"Content-Type": "text/plain; charset=UTF-8"}
        po.append(polib.POEntry(msgctxt="test::para:0", msgid="Hello", msgstr="World"))
        po.save(po_path.as_posix())

        loaded = manager.load_or_create_po(po_path)
        assert len(loaded) == 1
        assert loaded[0].msgid == "Hello"

    def test_no_path_raises(self, manager):
        with pytest.raises(ValueError):
            manager.load_or_create_po(None)


class TestSyncPO:
    def test_new_entries_created(self, manager, fresh_po, simple_blocks):
        manager.sync_po(fresh_po, simple_blocks, ctx_func)
        # heading + para + hr = 3 entries
        assert len(fresh_po) == 3

    def test_new_entries_empty_msgstr(self, manager, fresh_po, simple_blocks):
        manager.sync_po(fresh_po, simple_blocks, ctx_func)
        for entry in fresh_po:
            assert entry.msgstr == ""

    def test_changed_entry_marked_fuzzy(self, manager, fresh_po, simple_blocks):
        manager.sync_po(fresh_po, simple_blocks, ctx_func)
        # Simulate a translation
        para_entry = next(e for e in fresh_po if "para" in e.msgctxt)
        para_entry.msgstr = "Translated"

        # Change the source text
        simple_blocks[1]["text"] = "Hello changed world"
        manager.sync_po(fresh_po, simple_blocks, ctx_func)

        para_entry = next(e for e in fresh_po if "para" in e.msgctxt)
        assert "fuzzy" in para_entry.flags

    def test_unchanged_entry_not_fuzzy(self, manager, fresh_po, simple_blocks):
        manager.sync_po(fresh_po, simple_blocks, ctx_func)
        para_entry = next(e for e in fresh_po if "para" in e.msgctxt)
        para_entry.msgstr = "Translated"

        # Sync again without changes
        manager.sync_po(fresh_po, simple_blocks, ctx_func)
        para_entry = next(e for e in fresh_po if "para" in e.msgctxt)
        assert "fuzzy" not in para_entry.flags

    def test_removed_entry_deleted(self, manager, fresh_po, simple_blocks):
        manager.sync_po(fresh_po, simple_blocks, ctx_func)
        assert len(fresh_po) == 3

        # Remove one block
        reduced = simple_blocks[:2]
        manager.sync_po(fresh_po, reduced, ctx_func)
        # hr entry removed (obsolete entries are purged)
        assert len(fresh_po) == 2

    def test_skip_type_not_fuzzy_on_change(self, manager, fresh_po, simple_blocks):
        """HR is a skip type — changing its text should NOT set fuzzy."""
        manager.sync_po(fresh_po, simple_blocks, ctx_func)
        simple_blocks[2]["text"] = "***"
        manager.sync_po(fresh_po, simple_blocks, ctx_func)

        hr_entry = next(e for e in fresh_po if "hr" in e.msgctxt)
        assert "fuzzy" not in hr_entry.flags


class TestMarkEntryProcessed:
    def test_removes_fuzzy(self, manager):
        entry = polib.POEntry(msgctxt="x", msgid="hi", msgstr="hey")
        entry.flags.append("fuzzy")
        manager.mark_entry_processed(entry)
        assert "fuzzy" not in entry.flags

    def test_no_fuzzy_is_noop(self, manager):
        entry = polib.POEntry(msgctxt="x", msgid="hi", msgstr="hey")
        manager.mark_entry_processed(entry)
        assert "fuzzy" not in entry.flags


class TestGetUnprocessedEntries:
    def test_empty_msgstr_is_unprocessed(self, manager, fresh_po):
        fresh_po.append(polib.POEntry(msgctxt="a", msgid="text", msgstr=""))
        result = manager.get_unprocessed_entries(fresh_po)
        assert len(result) == 1

    def test_fuzzy_is_unprocessed(self, manager, fresh_po):
        entry = polib.POEntry(msgctxt="a", msgid="text", msgstr="translated")
        entry.flags.append("fuzzy")
        fresh_po.append(entry)
        result = manager.get_unprocessed_entries(fresh_po)
        assert len(result) == 1

    def test_processed_not_returned(self, manager, fresh_po):
        fresh_po.append(
            polib.POEntry(msgctxt="a", msgid="text", msgstr="translated")
        )
        result = manager.get_unprocessed_entries(fresh_po)
        assert len(result) == 0

    def test_no_po_raises(self, manager):
        with pytest.raises(ValueError):
            manager.get_unprocessed_entries(None)


class TestGetFuzzyEntries:
    def test_returns_fuzzy(self, manager, fresh_po):
        entry = polib.POEntry(msgctxt="a", msgid="text", msgstr="t")
        entry.flags.append("fuzzy")
        fresh_po.append(entry)
        result = manager.get_fuzzy_entries(fresh_po)
        assert len(result) == 1

    def test_non_fuzzy_excluded(self, manager, fresh_po):
        fresh_po.append(polib.POEntry(msgctxt="a", msgid="text", msgstr="t"))
        result = manager.get_fuzzy_entries(fresh_po)
        assert len(result) == 0


class TestGetProcessingStats:
    def test_all_counts(self, manager, fresh_po):
        # processed entry
        fresh_po.append(polib.POEntry(msgctxt="a", msgid="x", msgstr="y"))
        # unprocessed entry
        fresh_po.append(polib.POEntry(msgctxt="b", msgid="x", msgstr=""))
        # fuzzy entry
        fuzzy = polib.POEntry(msgctxt="c", msgid="x", msgstr="z")
        fuzzy.flags.append("fuzzy")
        fresh_po.append(fuzzy)

        stats = manager.get_processing_stats(fresh_po)
        assert stats["total"] == 3
        assert stats["processed"] == 1
        assert stats["unprocessed"] == 1
        assert stats["fuzzy"] == 1
        assert stats["obsolete"] == 0

    def test_no_po_raises(self, manager):
        with pytest.raises(ValueError):
            manager.get_processing_stats(None)


class TestRedrawContext:
    def test_creates_fresh_po(self, manager, simple_blocks):
        manager.po_file = polib.POFile()
        manager.redraw_context(simple_blocks, ctx_func)
        po = manager.po_file
        # heading + para get msgstr=text; hr gets msgstr=""
        para_entry = next(e for e in po if "para" in e.msgctxt)
        assert para_entry.msgstr == para_entry.msgid

        hr_entry = next(e for e in po if "hr" in e.msgctxt)
        assert hr_entry.msgstr == ""


class TestSavePO:
    def _nonempty_po(self):
        """An empty POFile is falsy; save_po needs a truthy one."""
        po = polib.POFile()
        po.metadata = {"Content-Type": "text/plain; charset=UTF-8"}
        po.append(polib.POEntry(msgctxt="x", msgid="hi", msgstr=""))
        return po

    def test_saves_to_disk(self, manager, tmp_path):
        po_path = tmp_path / "out.po"
        manager.save_po(self._nonempty_po(), po_path)
        assert po_path.exists()

    def test_creates_parent_dirs(self, manager, tmp_path):
        po_path = tmp_path / "sub" / "dir" / "out.po"
        manager.save_po(self._nonempty_po(), po_path)
        assert po_path.exists()

    def test_no_po_raises(self, manager, tmp_path):
        with pytest.raises(ValueError):
            manager.save_po(None, tmp_path / "out.po")

    def test_no_path_raises(self, manager):
        with pytest.raises(ValueError):
            manager.save_po(self._nonempty_po(), None)
