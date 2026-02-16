"""
Test demonstration of mdpo-llm package functionality.
"""

from pathlib import Path
from src.mdpo_llm import MdpoLLM, MockLLMInterface, LANGUAGE_PATTERNS

def create_sample_document():
    """Create a sample markdown document for testing."""
    sample_md = """# Sample Documentation

This is a test document to demonstrate the mdpo-llm package functionality.

## Features

The package offers several key features:

- Incremental processing with PO files
- Support for multiple languages
- Preservation of Markdown structure

## Code Example

Here's a simple Python example:

```python
def greet(name):
    # This is a comment
    print(f"Hello, {name}!")
    return True
```

## Table Example

| Feature | Status | Description |
|---------|--------|-------------|
| Translation | Yes | Fully supported |
| Refinement | Yes | In-place editing |
| Concurrency | Yes | Multi-threading |

## Conclusion

This demonstrates how the package processes Markdown documents efficiently.
"""
    return sample_md

def main():
    print("=" * 60)
    print("mdpo-llm Package Test Demonstration")
    print("=" * 60)
    
    # Create test directories
    test_dir = Path("test_output")
    test_dir.mkdir(exist_ok=True)
    
    # Create sample document
    source_file = test_dir / "sample.md"
    target_file = test_dir / "sample_processed.md"
    po_file = test_dir / "sample.po"
    
    print("\n📝 Creating sample Markdown document...")
    source_file.write_text(create_sample_document(), encoding='utf-8')
    print(f"   Created: {source_file}")
    
    # Initialize with MockLLMInterface for testing
    print("\n🤖 Initializing MdpoLLM with MockLLMInterface...")
    llm = MockLLMInterface()
    processor = MdpoLLM(llm)
    print("   ✓ Processor initialized")
    
    # Process the document
    print("\n⚙️ Processing document...")
    result = processor.process_document(
        source_path=source_file,
        target_path=target_file,
        po_path=po_file,
        inplace=False
    )
    
    # Display results
    print("\n📊 Processing Results:")
    print(f"   • Total blocks parsed: {result['blocks_count']}")
    print(f"   • Processed blocks: {result['translation_stats'].get('processed', 0)}")
    print(f"   • Failed blocks: {result['translation_stats'].get('failed', 0)}")
    print(f"   • Skipped blocks: {result['translation_stats'].get('skipped', 0)}")
    
    print("\n📈 Coverage Statistics:")
    coverage = result['coverage']
    print(f"   • Total blocks: {coverage['total_blocks']}")
    print(f"   • Translatable blocks: {coverage['translatable_blocks']}")
    print(f"   • Translated blocks: {coverage['translated_blocks']}")
    print(f"   • Coverage: {coverage['coverage_percentage']:.1f}%")
    
    print("\n📁 Files Generated:")
    print(f"   • Source: {source_file}")
    print(f"   • Target: {target_file}")
    print(f"   • PO file: {po_file}")
    
    # Show PO file statistics
    print("\n📋 Checking PO file content...")
    if po_file.exists():
        po_content = po_file.read_text(encoding='utf-8')
        msgid_count = po_content.count('msgid "')
        msgstr_count = po_content.count('msgstr "[MOCK PROCESSING]')
        print(f"   • Total entries: {msgid_count}")
        print(f"   • Processed entries: {msgstr_count}")
    
    # Test language detection
    print("\n🌍 Testing Language Detection:")
    test_texts = [
        ("Hello World", "English"),
        ("你好世界", "Chinese"),
        ("こんにちは", "Japanese"),
        ("안녕하세요", "Korean"),
        ("Hello 世界", "Mixed EN+CN")
    ]
    
    for text, description in test_texts:
        detected = []
        for code, pattern in LANGUAGE_PATTERNS.items():
            if pattern.search(text):
                detected.append(code)
        print(f"   • '{text}' ({description}): {', '.join(detected) if detected else 'None'}")
    
    # Demonstrate incremental processing
    print("\n🔄 Testing Incremental Processing...")
    print("   Making a small change to the source document...")
    
    # Modify the document
    content = source_file.read_text(encoding='utf-8')
    modified_content = content.replace(
        "This is a test document", 
        "This is an updated test document"
    )
    source_file.write_text(modified_content, encoding='utf-8')
    
    # Reprocess
    print("   Reprocessing with changes...")
    result2 = processor.process_document(
        source_path=source_file,
        target_path=target_file,
        po_path=po_file,
        inplace=False
    )
    
    print(f"   • Newly processed blocks: {result2['translation_stats'].get('processed', 0)}")
    print("   ✓ Only changed blocks were reprocessed!")
    
    # Show sample of processed content
    print("\n📄 Sample of Processed Content:")
    print("   (First 500 characters)")
    print("-" * 40)
    processed_content = target_file.read_text(encoding='utf-8')[:500]
    for line in processed_content.split('\n')[:10]:
        print(f"   {line}")
    print("-" * 40)
    
    print("\n✅ Test demonstration completed successfully!")
    print(f"   All output files are in: {test_dir.absolute()}")

if __name__ == "__main__":
    main()
