"""
Example: demonstrate the ``post_process`` escape hatch.

``post_process`` is a callable applied to every LLM response before it is
stored. Use it for deterministic cleanups that don't need another LLM call:
stripping markdown fences the model wraps around its answer, normalising
whitespace, enforcing house-style rules, etc.
"""

import re
from pathlib import Path
from mdpo_llm import MdpoLLM


def strip_markdown_wrapper(text: str) -> str:
    """Remove markdown code fences that some models wrap around their output."""
    text = text.strip()
    # Strip ```markdown ... ``` wrapper
    if text.startswith("```") and text.endswith("```"):
        first_newline = text.index("\n") if "\n" in text else len(text)
        text = text[first_newline + 1 : -3].strip()
    return text


def normalize_whitespace(text: str) -> str:
    """Collapse runs of blank lines into a single blank line."""
    return re.sub(r"\n{3,}", "\n\n", text)


def combined_post_process(text: str) -> str:
    """Chain multiple cleanups together."""
    text = strip_markdown_wrapper(text)
    text = normalize_whitespace(text)
    return text


def main():
    """Demonstrate post_process in action."""

    print("=" * 60)
    print("post_process Example for mdpo-llm")
    print("=" * 60)

    # Create sample document
    sample_md = """\
# Technical Documentation

This document explains our API endpoints.

## Authentication

Use the following endpoint to authenticate:
https://api.example.com/auth

## Code Example

```python
def authenticate(token):
    # Send authentication request
    response = api.post("/auth", headers={"Token": token})
    return response.json()
```

## Important Notes

- Always use HTTPS
- Tokens expire after 24 hours
- Rate limit: 100 requests/minute
"""

    test_dir = Path("test_validation")
    test_dir.mkdir(exist_ok=True)

    source_file = test_dir / "api_docs.md"
    target_file = test_dir / "api_docs_translated.md"
    po_file = test_dir / "api_docs.po"

    source_file.write_text(sample_md, encoding="utf-8")

    # Create processor with post_process
    processor = MdpoLLM(
        model="gpt-4",
        target_lang="ko",
        post_process=combined_post_process,  # applied to every LLM response
        temperature=0.3,
    )

    print("\nProcessing with post_process enabled...")
    result = processor.process_document(
        source_path=source_file,
        target_path=target_file,
        po_path=po_file,
    )

    print(f"\nProcessing complete!")
    print(f"  - Processed: {result['translation_stats'].get('processed', 0)}")
    print(f"  - Failed: {result['translation_stats'].get('failed', 0)}")
    print(f"  - Skipped: {result['translation_stats'].get('skipped', 0)}")


if __name__ == "__main__":
    main()
