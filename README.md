# mdpo-llm

[![Python Version](https://img.shields.io/pypi/pyversions/mdpo-llm.svg)](https://pypi.org/project/mdpo-llm/)
[![PyPI Version](https://img.shields.io/pypi/v/mdpo-llm.svg)](https://pypi.org/project/mdpo-llm/)
[![License](https://img.shields.io/pypi/l/mdpo-llm.svg)](https://github.com/yourusername/mdpo-llm/blob/main/LICENSE)

Process Markdown documents with LLMs using PO files for efficient translation and refinement workflows.

## Features

- 📝 **Incremental Processing**: Only process changed content using GNU gettext PO files
- 🌍 **Multi-language Support**: Built-in support for English, Chinese, Japanese, and Korean
- 🔄 **Translation & Refinement**: Use LLMs for both translation and document refinement
- 🏗️ **Structure Preservation**: Maintains perfect Markdown structure and formatting
- ⚡ **Concurrent Processing**: Process multiple blocks in parallel for speed
- 🔌 **LLM Agnostic**: Implement your own LLM interface for any provider

## Why mdpo-llm?

Traditional approaches to translating or refining Markdown documents with LLMs require processing the entire file every time there's a change. **mdpo-llm** solves this by:

1. **Parsing** Markdown into semantic blocks (headings, paragraphs, code blocks, etc.)
2. **Tracking** each block's content and translation state using PO files
3. **Processing** only new or changed content through your LLM
4. **Preserving** the exact document structure when reconstructing the output

This means you can make small edits to a large document and only pay for processing the changed sections!

## Installation

```bash
pip install mdpo-llm
```

## Quick Start

### 1. Implement the LLM Interface

```python
from mdpo_llm import LLMInterface, MdpoLLM, LanguageCode

class MyLLM(LLMInterface):
    def process(self, source_text: str) -> str:
        # Your LLM logic here (OpenAI, Anthropic, Google, local models, etc.)
        # Example with OpenAI:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Translate to Korean"},
                {"role": "user", "content": source_text}
            ]
        )
        return response.choices[0].message.content
```

### 2. Process Your Documents

```python
from pathlib import Path

# Initialize with your LLM
llm = MyLLM()
processor = MdpoLLM(llm)

# Process a document
result = processor.process_document(
    source_path=Path("docs/README.md"),
    target_path=Path("docs/README_ko.md"),
    po_path=Path("translations/README.po"),
    inplace=False  # Set True for refinement instead of translation
)

print(f"Processed {result['blocks_count']} blocks")
print(f"Translation coverage: {result['coverage']['coverage_percentage']}%")
```

## Advanced Usage

### Language Detection

```python
from mdpo_llm import LanguageCode

# Detect languages in text
text = "Hello 世界"
if LanguageCode.EN.in_text(text):
    print("Contains English")
if LanguageCode.CN.in_text(text):
    print("Contains Chinese")
```

### Custom Processing Configuration

```python
class CustomProcessor(MdpoLLM):
    # Skip processing certain block types
    SKIP_TYPES = ["hr", "code"]  # Don't process horizontal rules or code blocks
    
processor = CustomProcessor(llm)
```

### Working with PO Files

The PO files track translation state for each block:
- **Untranslated**: New content that needs processing
- **Translated**: Completed translations
- **Fuzzy**: Content that changed and needs re-processing
- **Obsolete**: Removed content (automatically cleaned up)

You can inspect PO files with standard gettext tools or any PO editor.

## API Reference

### Core Classes

#### `MdpoLLM` (alias for `MarkdownProcessor`)

Main processor class that orchestrates the workflow.

**Methods:**
- `process_document(source_path, target_path, po_path, inplace=False)` - Process a document
- `get_translation_stats(source_path, po_path)` - Get processing statistics
- `export_report(source_path, po_path)` - Generate a detailed report

#### `LLMInterface`

Abstract base class for LLM implementations.

**Methods to implement:**
- `process(source_text: str) -> str` - Process text and return result

#### `LanguageCode`

Enum for supported languages with detection capabilities.

**Values:**
- `LanguageCode.EN` - English
- `LanguageCode.CN` - Chinese
- `LanguageCode.JP` - Japanese  
- `LanguageCode.KO` - Korean

**Methods:**
- `in_text(text: str) -> bool` - Check if language appears in text

## Examples

### Translation Workflow

```python
from mdpo_llm import MdpoLLM, LLMInterface
from pathlib import Path

class TranslationLLM(LLMInterface):
    def __init__(self, target_language):
        self.target_language = target_language
        
    def process(self, source_text: str) -> str:
        # Your translation logic
        return translate(source_text, self.target_language)

# Setup
llm = TranslationLLM("Korean")
processor = MdpoLLM(llm)

# First run - translates everything
processor.process_document(
    Path("README.md"),
    Path("README_ko.md"),
    Path("translations/README.po")
)

# Edit README.md (e.g., fix a typo)
# ...

# Second run - only translates changed paragraphs!
processor.process_document(
    Path("README.md"),
    Path("README_ko.md"),
    Path("translations/README.po")
)
```

### Document Refinement Workflow

```python
class RefinementLLM(LLMInterface):
    def process(self, source_text: str) -> str:
        # Improve clarity, fix grammar, etc.
        return refine_text(source_text)

llm = RefinementLLM()
processor = MdpoLLM(llm)

# Refine document in-place (same language)
processor.process_document(
    Path("draft.md"),
    Path("refined.md"),
    Path("refinements/draft.po"),
    inplace=True  # Indicates same-language refinement
)
```

## Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/mdpo-llm.git
cd mdpo-llm

# Install with development dependencies
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest tests/
```

### Build Package

```bash
python -m build
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on top of [polib](https://github.com/izimobil/polib) for PO file handling
- Inspired by traditional gettext localization workflows
