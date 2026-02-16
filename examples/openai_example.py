"""
Example: translate Markdown to Korean using OpenAI's GPT-4 via LiteLLM.
"""

from pathlib import Path
from mdpo_llm import MdpoLLM


def main():
    """Example usage."""

    # Setup paths
    source_file = Path("README.md")
    output_file = Path("README_ko.md")
    po_file = Path("translations/README.po")

    # Check if source file exists
    if not source_file.exists():
        print(f"Source file {source_file} not found. Creating example...")
        source_file.write_text("""\
# Example Document

This is a sample document for translation.

## Features

- Feature 1: Something amazing
- Feature 2: Another great thing

## Code Example

```python
def hello_world():
    print("Hello, World!")
    return True
```

## Conclusion

This document demonstrates the translation capabilities.
""")

    # Create processor — that's it. No subclassing needed.
    processor = MdpoLLM(
        model="gpt-4",               # any LiteLLM model string
        target_lang="ko",            # baked into the system prompt
        source_langs=["ko"],         # code blocks without Korean are skipped
        temperature=0.3,             # passed through to litellm.completion()
    )

    # Process document
    print(f"Processing {source_file} -> {output_file}")
    result = processor.process_document(
        source_path=source_file,
        target_path=output_file,
        po_path=po_file,
    )

    # Print results
    print(f"\nProcessing complete!")
    print(f"Statistics:")
    print(f"  - Total blocks: {result['blocks_count']}")
    print(f"  - Translated: {result['translation_stats'].get('processed', 0)}")
    print(f"  - Failed: {result['translation_stats'].get('failed', 0)}")
    print(f"  - Skipped: {result['translation_stats'].get('skipped', 0)}")
    print(f"  - Coverage: {result['coverage']['coverage_percentage']:.1f}%")

    print(f"\nFiles created:")
    print(f"  - Translation: {output_file}")
    print(f"  - PO file: {po_file}")

    # Example: Make a small edit and reprocess
    print("\nDemonstrating incremental processing...")
    print("Making a small edit to the source file...")

    content = source_file.read_text()
    content = content.replace("Something amazing", "Something truly amazing")
    source_file.write_text(content)

    print("Reprocessing (only changed blocks will be translated)...")
    result2 = processor.process_document(
        source_path=source_file,
        target_path=output_file,
        po_path=po_file,
    )

    print(f"  - Newly translated: {result2['translation_stats'].get('processed', 0)} blocks")
    print("Only the changed paragraph was retranslated!")


if __name__ == "__main__":
    main()
