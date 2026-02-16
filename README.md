# mdpo-llm

[![Python Version](https://img.shields.io/pypi/pyversions/mdpo-llm.svg)](https://pypi.org/project/mdpo-llm/)
[![PyPI Version](https://img.shields.io/pypi/v/mdpo-llm.svg)](https://pypi.org/project/mdpo-llm/)
[![License](https://img.shields.io/pypi/l/mdpo-llm.svg)](https://github.com/yourusername/mdpo-llm/blob/main/LICENSE)

Process Markdown documents with LLMs using PO files for efficient, incremental translation and refinement.

## Features

- Incremental processing — only changed content gets sent to your LLM
- Structure preservation — Markdown formatting survives the round-trip intact
- Concurrent execution — process multiple blocks in parallel
- LLM agnostic — bring any provider (OpenAI, Anthropic, local models, etc.)
- Batch processing — process entire directory trees in one call

## Why mdpo-llm?

Traditional approaches send the entire file to an LLM on every change. mdpo-llm does better:

1. **Parse** — split Markdown into semantic blocks (headings, paragraphs, code blocks, etc.)
2. **Track** — record each block's content and state in a PO file
3. **Process** — send only new or changed blocks to your LLM
4. **Reconstruct** — reassemble the document with original formatting

Only pay for the sections that actually changed.

## Installation

```bash
pip install mdpo-llm
```

## Quick Start

### 1. Implement the LLM Interface

```python
from mdpo_llm import LLMInterface, MdpoLLM

class MyLLM(LLMInterface):
    def process(self, source_text: str) -> str:
        # Call any LLM here
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Translate to Korean"},
                {"role": "user", "content": source_text},
            ],
        )
        return response.choices[0].message.content
```

### 2. Process a single document

```python
from pathlib import Path

processor = MdpoLLM(MyLLM())

result = processor.process_document(
    source_path=Path("docs/README.md"),
    target_path=Path("docs/README_ko.md"),
    po_path=Path("translations/README.po"),
    inplace=False,  # True for same-language refinement
)

print(f"Processed {result['blocks_count']} blocks")
print(f"Coverage: {result['coverage']['coverage_percentage']}%")
```

Run it again after editing the source — only the changed paragraphs get reprocessed.

### 3. Process a directory

```python
result = processor.process_directory(
    source_dir=Path("docs/"),
    target_dir=Path("docs_ko/"),
    po_dir=Path("translations/"),
    glob="**/*.md",
)

print(f"{result['files_processed']} files processed")
print(f"{result['files_skipped']} files unchanged")
```

The directory structure under `source_dir` is mirrored into `target_dir` and `po_dir`.

## API Reference

### MdpoLLM

| Method | Description |
|--------|-------------|
| `process_document(source_path, target_path, po_path, inplace=False)` | Process a single Markdown file |
| `process_directory(source_dir, target_dir, po_dir, glob="**/*.md", inplace=False)` | Process all matching files in a directory tree |
| `get_translation_stats(source_path, po_path)` | Return coverage and block statistics |
| `export_report(source_path, po_path)` | Generate a detailed text report |

### LLMInterface

Abstract base class. Implement one method:

```python
def process(self, source_text: str) -> str:
    ...
```

## Working with PO Files

PO files (GNU gettext) track the state of each content block:

- **Untranslated** — new content that needs processing
- **Translated** — completed, up-to-date translations
- **Fuzzy** — source changed since last processing; will be reprocessed on the next run
- **Obsolete** — source block was removed; cleaned up automatically

You can inspect PO files with any standard gettext tool or PO editor.

## Development

```bash
# Install with dev dependencies
uv pip install -e ".[dev]"

# Run tests
pytest tests/
```

## License

MIT
