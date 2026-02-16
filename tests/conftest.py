"""Shared fixtures for mdpo-llm tests."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import polib
import pytest

from mdpo_llm.manager import POManager
from mdpo_llm.parser import BlockParser


SAMPLE_MARKDOWN = """\
# Introduction

This is the first paragraph with some **bold** text.

## Getting Started

- Item one
- Item two
- Item three

### Installation

```bash
pip install mdpo-llm
```

Here is a table:

| Header 1 | Header 2 |
|----------|----------|
| Cell 1   | Cell 2   |

> This is a blockquote
> spanning two lines.

---

1. First ordered item
2. Second ordered item

Final paragraph here.
"""


@pytest.fixture
def sample_markdown():
    """Multi-block sample markdown string."""
    return SAMPLE_MARKDOWN


@pytest.fixture
def sample_lines():
    """Sample markdown split into lines (without trailing newlines)."""
    return SAMPLE_MARKDOWN.splitlines()


@pytest.fixture
def mock_completion():
    """Patch litellm.completion — returns '[TRANSLATED] {source_text}' by default."""

    def _side_effect(*args, **kwargs):
        messages = kwargs.get("messages", args[0] if args else [])
        source_text = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                source_text = msg["content"]
                break
        mock_response = MagicMock()
        mock_response.choices[0].message.content = f"[TRANSLATED] {source_text}"
        return mock_response

    with patch("mdpo_llm.processor.litellm") as mock_litellm:
        mock_litellm.completion.side_effect = _side_effect
        yield mock_litellm


@pytest.fixture
def parser():
    """BlockParser instance."""
    return BlockParser()


@pytest.fixture
def po_manager():
    """POManager with hr skipped."""
    return POManager(skip_types=["hr"])


@pytest.fixture
def tmp_paths(tmp_path):
    """Convenience bundle of temp file paths."""
    return {
        "source": tmp_path / "source.md",
        "target": tmp_path / "target.md",
        "po": tmp_path / "messages.po",
    }


@pytest.fixture
def populated_po(parser, po_manager, sample_lines):
    """Return a PO file that has been synced with the sample markdown."""
    blocks = parser.segment_markdown(sample_lines)
    po_file = polib.POFile()
    po_file.metadata = {"Content-Type": "text/plain; charset=UTF-8"}
    po_manager.sync_po(po_file, blocks, parser.context_id)
    return po_file, blocks
