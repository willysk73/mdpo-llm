"""
Example implementation using OpenAI's GPT models for translation.
"""

import os
from pathlib import Path
from mdpo_llm import LLMInterface, MdpoLLM, LanguageCode

# You'll need to install openai: pip install openai
try:
    import openai
except ImportError:
    print("Please install openai: pip install openai")
    exit(1)


class OpenAITranslator(LLMInterface):
    """OpenAI-based translator implementation."""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4", target_language: str = "Korean"):
        """Initialize OpenAI translator.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model to use (e.g., "gpt-4", "gpt-3.5-turbo")
            target_language: Target language for translation
        """
        self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.target_language = target_language
        
    def process(self, source_text: str) -> str:
        """Translate text using OpenAI."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": f"You are a professional translator. Translate the following text to {self.target_language}. "
                                   f"Preserve all Markdown formatting, code blocks, and special characters exactly as they appear."
                    },
                    {
                        "role": "user", 
                        "content": source_text
                    }
                ],
                temperature=0.3,  # Lower temperature for more consistent translations
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Translation error: {e}")
            raise


def main():
    """Example usage."""
    
    # Setup paths
    source_file = Path("README.md")
    output_file = Path("README_ko.md")
    po_file = Path("translations/README.po")
    
    # Check if source file exists
    if not source_file.exists():
        print(f"Source file {source_file} not found. Creating example...")
        source_file.write_text("""# Example Document

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
    
    # Initialize translator
    translator = OpenAITranslator(
        api_key=os.getenv("OPENAI_API_KEY"),  # Set your API key as environment variable
        model="gpt-3.5-turbo",  # Use gpt-4 for better quality
        target_language="Korean"
    )
    
    # Create processor
    processor = MdpoLLM(translator)
    
    # Process document
    print(f"Processing {source_file} -> {output_file}")
    result = processor.process_document(
        source_path=source_file,
        target_path=output_file,
        po_path=po_file,
        inplace=False
    )
    
    # Print results
    print(f"\n✅ Processing complete!")
    print(f"📊 Statistics:")
    print(f"  - Total blocks: {result['blocks_count']}")
    print(f"  - Translated: {result['translation_stats'].get('processed', 0)}")
    print(f"  - Failed: {result['translation_stats'].get('failed', 0)}")
    print(f"  - Skipped: {result['translation_stats'].get('skipped', 0)}")
    print(f"  - Coverage: {result['coverage']['coverage_percentage']:.1f}%")
    
    print(f"\n📁 Files created:")
    print(f"  - Translation: {output_file}")
    print(f"  - PO file: {po_file}")
    
    # Example: Make a small edit and reprocess
    print("\n🔄 Demonstrating incremental processing...")
    print("Making a small edit to the source file...")
    
    content = source_file.read_text()
    content = content.replace("Something amazing", "Something truly amazing")
    source_file.write_text(content)
    
    print("Reprocessing (only changed blocks will be translated)...")
    result2 = processor.process_document(
        source_path=source_file,
        target_path=output_file,
        po_path=po_file,
        inplace=False
    )
    
    print(f"  - Newly translated: {result2['translation_stats'].get('processed', 0)} blocks")
    print("✨ Only the changed paragraph was retranslated!")


if __name__ == "__main__":
    main()
