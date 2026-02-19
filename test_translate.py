#!/usr/bin/env python3
"""Test script for mdpo-llm translation with real API."""

from pathlib import Path
from mdpo_llm import MdpoLLM

# Configure your model here
# For Anthropic: "anthropic/claude-sonnet-4-5-20250929"
# For OpenAI: "gpt-4" or "gpt-4o"
MODEL = "gpt-4o"
TARGET_LANG = "ko"  # Korean

processor = MdpoLLM(
    model=MODEL,
    target_lang=TARGET_LANG,
    temperature=0.3,
)

print(f"🚀 Translating test_sample.md to {TARGET_LANG} using {MODEL}\n")

result = processor.process_document(
    source_path=Path("test_sample.md"),
    target_path=Path("test_sample_ko.md"),
    # po_path defaults to test_sample_ko.po
)

print("\n✅ Translation complete!")
print(f"📊 Processed: {result['translation_stats']['processed']} blocks")
print(f"📈 Coverage: {result['coverage']['coverage_percentage']:.1f}%")
print(f"📝 Output: test_sample_ko.md")
print(f"💾 PO file: test_sample_ko.po")
print(
    "\nRun again to see incremental processing - only changed blocks will be retranslated!"
)
