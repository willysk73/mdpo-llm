"""Regression tests for inplace mode preserving validator state."""

import json
from unittest.mock import MagicMock

from mdpo_llm.processor import MarkdownProcessor


def test_inplace_preserves_fuzzy_per_occurrence(tmp_path, mock_completion):
    """Duplicate translated labels must not share fuzzy state across entries."""

    # Two "# Overview" headings → translate identically.  Only the *second*
    # one's msgstr is deliberately mangled so the validator flags it.  After
    # inplace redraw, exactly one of the two entries should remain fuzzy.
    def _translate(v: str, overview_n: int):
        if v.startswith("# Overview"):
            if overview_n == 2:
                return "요약"  # heading marker dropped → validator flags fuzzy
            return "# 개요"
        if v.startswith("# Other"):
            return "# 기타"
        return f"번역 {v}"

    def _side_effect(*args, **kwargs):
        messages = kwargs.get("messages", [])
        user_content = messages[-1]["content"]
        try:
            items = json.loads(user_content)
        except json.JSONDecodeError:
            items = None

        resp = MagicMock()
        if isinstance(items, dict):
            out = {}
            n = 0
            for k, v in items.items():
                if v.startswith("# Overview"):
                    n += 1
                out[k] = _translate(v, n)
            resp.choices[0].message.content = json.dumps(out)
        else:
            resp.choices[0].message.content = f"번역 {user_content}"
        return resp

    mock_completion.completion.side_effect = _side_effect

    md = (
        "# Overview\n\nFirst body.\n\n# Other\n\nMid.\n\n# Overview\n\nSecond body.\n"
    )
    source = tmp_path / "source.md"
    source.write_text(md, encoding="utf-8")
    po_path = tmp_path / "m.po"

    from mdpo_llm.processor import MarkdownProcessor

    p = MarkdownProcessor(
        model="test-model",
        target_lang="ko",
        batch_size=40,
        validation="conservative",
    )
    p.process_document(source, tmp_path / "target.md", po_path, inplace=True)

    po = p.po_manager.load_or_create_po(po_path)
    fuzzy_entries = [e for e in po if "fuzzy" in e.flags]
    # Exactly one fuzzy entry — the second Overview — not both.
    assert len(fuzzy_entries) == 1


def test_inplace_preserves_fuzzy_from_validator(tmp_path, mock_completion):
    """Validator-flagged entries must stay fuzzy after an inplace redraw."""

    def _side_effect(*args, **kwargs):
        messages = kwargs.get("messages", [])
        user_content = messages[-1]["content"]
        try:
            items = json.loads(user_content)
        except json.JSONDecodeError:
            items = None

        resp = MagicMock()
        if isinstance(items, dict):
            # Strip heading markers → heading-level check fails.
            out = {}
            for k, v in items.items():
                stripped = v.lstrip("# ").strip() or "요약"
                out[k] = stripped
            resp.choices[0].message.content = json.dumps(out)
        else:
            resp.choices[0].message.content = "번역"
        return resp

    mock_completion.completion.side_effect = _side_effect

    md = "# Title\n\n텍스트.\n"
    source = tmp_path / "source.md"
    source.write_text(md, encoding="utf-8")
    po_path = tmp_path / "m.po"

    p = MarkdownProcessor(
        model="test-model",
        target_lang="ko",
        batch_size=40,
        validation="conservative",
    )
    p.process_document(source, tmp_path / "target.md", po_path, inplace=True)

    po = p.po_manager.load_or_create_po(po_path)
    # The heading entry should still carry the fuzzy flag after redraw.
    assert any("fuzzy" in e.flags for e in po)
    # The validator tcomment survives the redraw too.
    assert any(
        e.tcomment and "validator" in e.tcomment for e in po
    )
