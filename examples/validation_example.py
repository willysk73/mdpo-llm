"""
Example implementation showing how to add validation and retry logic to your LLM.

This demonstrates best practices for ensuring quality in translation/refinement tasks.
"""

import time
from pathlib import Path
from typing import Optional
from mdpo_llm import LLMInterface, MdpoLLM, LanguageCode


class ValidatingLLM(LLMInterface):
    """Example LLM with built-in validation and retry logic."""
    
    def __init__(
        self, 
        target_language: str = "Korean",
        max_retries: int = 3,
        validate: bool = True
    ):
        """Initialize LLM with validation settings.
        
        Args:
            target_language: Target language for translation
            max_retries: Maximum number of retries when validation fails
            validate: Whether to perform validation
        """
        self.target_language = target_language
        self.max_retries = max_retries
        self.validate = validate
        self.total_attempts = 0
        self.validation_failures = 0
        
    def process(self, source_text: str) -> str:
        """Process text with automatic validation and retry."""
        attempt = 0
        last_error = None
        
        while attempt < self.max_retries:
            attempt += 1
            self.total_attempts += 1
            
            # Generate processed text
            processed_text = self._generate(source_text, last_error)
            
            # Validate if enabled
            if self.validate:
                is_valid, error = self._validate(source_text, processed_text)
                if is_valid:
                    return processed_text
                else:
                    self.validation_failures += 1
                    last_error = error
                    print(f"  ⚠️  Validation failed (attempt {attempt}/{self.max_retries}): {error}")
                    if attempt < self.max_retries:
                        print(f"  🔄 Retrying...")
                        time.sleep(0.5)  # Brief delay before retry
            else:
                # No validation, return immediately
                return processed_text
        
        # If we've exhausted retries, return the last attempt with a warning
        print(f"  ❌ Validation failed after {self.max_retries} attempts. Using last result.")
        return processed_text
    
    def _generate(self, source_text: str, previous_error: Optional[str] = None) -> str:
        """Generate processed text (this would call your actual LLM).
        
        Args:
            source_text: Text to process
            previous_error: Error from previous validation attempt (if any)
            
        Returns:
            Processed text
        """
        # This is where you'd call your actual LLM (OpenAI, Anthropic, etc.)
        # For demo purposes, we'll simulate different responses
        
        if previous_error and "too short" in previous_error.lower():
            # Simulate fixing a "too short" error
            return f"[VALIDATED TRANSLATION - FIXED LENGTH] {source_text}"
        elif previous_error and "missing" in previous_error.lower():
            # Simulate fixing missing content
            return f"[VALIDATED TRANSLATION - RESTORED CONTENT] {source_text}"
        else:
            # First attempt - sometimes simulate an error for demo
            import random
            if random.random() < 0.3:  # 30% chance of initial error
                return f"[TRANS] {source_text[:len(source_text)//2]}"  # Too short
            return f"[VALIDATED TRANSLATION] {source_text}"
    
    def _validate(self, source_text: str, processed_text: str) -> tuple[bool, Optional[str]]:
        """Validate the processed text meets quality standards.
        
        Args:
            source_text: Original text
            processed_text: Processed text to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Example validation rules:
        
        # 1. Check length ratio (processed shouldn't be too different from source)
        source_len = len(source_text)
        processed_len = len(processed_text)
        
        if source_len > 50:  # Only check for substantial text
            ratio = processed_len / source_len
            if ratio < 0.5:
                return False, f"Output too short (ratio: {ratio:.2f})"
            if ratio > 2.0:
                return False, f"Output too long (ratio: {ratio:.2f})"
        
        # 2. Check for markdown structure preservation
        source_headers = source_text.count('#')
        processed_headers = processed_text.count('#')
        if source_headers != processed_headers:
            return False, f"Markdown structure changed (headers: {source_headers} -> {processed_headers})"
        
        # 3. Check for code block preservation
        source_code_blocks = source_text.count('```')
        processed_code_blocks = processed_text.count('```')
        if source_code_blocks != processed_code_blocks:
            return False, f"Code blocks altered (count: {source_code_blocks} -> {processed_code_blocks})"
        
        # 4. Language-specific validation could go here
        # For example, checking if Korean text actually contains Korean characters
        if self.target_language == "Korean":
            # This is just an example - real implementation would be more sophisticated
            pass
        
        # 5. Check for critical content preservation (customize as needed)
        # For example, ensure URLs are preserved
        import re
        source_urls = re.findall(r'https?://[^\s\)]+', source_text)
        for url in source_urls:
            if url not in processed_text:
                return False, f"Missing URL: {url}"
        
        return True, None
    
    def get_stats(self):
        """Get validation statistics."""
        return {
            "total_attempts": self.total_attempts,
            "validation_failures": self.validation_failures,
            "retry_rate": f"{(self.validation_failures / max(1, self.total_attempts)) * 100:.1f}%"
        }


def main():
    """Demonstrate validation in action."""
    
    print("=" * 60)
    print("Validation Example for mdpo-llm")
    print("=" * 60)
    
    # Create sample document
    sample_md = """# Technical Documentation

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
    
    # Setup
    test_dir = Path("test_validation")
    test_dir.mkdir(exist_ok=True)
    
    source_file = test_dir / "api_docs.md"
    target_file = test_dir / "api_docs_translated.md"
    po_file = test_dir / "api_docs.po"
    
    source_file.write_text(sample_md, encoding='utf-8')
    
    # Test WITH validation
    print("\n📊 Testing WITH validation enabled:")
    print("-" * 40)
    
    llm_with_validation = ValidatingLLM(
        target_language="Korean",
        max_retries=3,
        validate=True
    )
    
    processor = MdpoLLM(llm_with_validation)
    result = processor.process_document(
        source_path=source_file,
        target_path=target_file,
        po_path=po_file,
        inplace=False
    )
    
    stats = llm_with_validation.get_stats()
    print(f"\n✅ Processing complete with validation!")
    print(f"   • Total LLM calls: {stats['total_attempts']}")
    print(f"   • Validation failures: {stats['validation_failures']}")
    print(f"   • Retry rate: {stats['retry_rate']}")
    
    # Test WITHOUT validation
    print("\n📊 Testing WITHOUT validation:")
    print("-" * 40)
    
    # Clean up for fresh test
    po_file.unlink(missing_ok=True)
    
    llm_without_validation = ValidatingLLM(
        target_language="Korean",
        max_retries=3,
        validate=False  # Disabled
    )
    
    processor = MdpoLLM(llm_without_validation)
    result = processor.process_document(
        source_path=source_file,
        target_path=target_file,
        po_path=po_file,
        inplace=False
    )
    
    stats = llm_without_validation.get_stats()
    print(f"\n✅ Processing complete without validation!")
    print(f"   • Total LLM calls: {stats['total_attempts']}")
    print(f"   • No validation performed")
    
    print("\n" + "=" * 60)
    print("💡 Key Takeaways:")
    print("=" * 60)
    print("""
1. Validation helps ensure quality but requires more LLM calls
2. Implement validation rules specific to your use case
3. Set appropriate retry limits to balance quality vs. cost
4. Track statistics to monitor validation performance
5. Consider disabling validation for drafts or testing
    """)


if __name__ == "__main__":
    main()
