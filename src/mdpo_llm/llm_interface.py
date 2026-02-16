"""
LLM interface for processing services.
"""

from abc import ABC, abstractmethod


class LLMInterface(ABC):
    """Abstract base class for LLM processing interfaces."""

    @abstractmethod
    def process(self, source_text: str) -> str:
        """Process source text to target language.

        Implementations may also accept the following optional keyword
        arguments (detected via ``inspect.signature``):

        * ``reference_pairs`` — a list of ``(source, translation)``
          tuples providing few-shot context from previously translated
          blocks in the same document.
        * ``target_lang`` — a BCP 47 locale string (e.g. ``"ko"``)
          indicating the desired output language.
        """
        pass


class MockLLMInterface(LLMInterface):
    """Mock implementation for testing and development."""

    def process(self, source_text: str, reference_pairs=None, target_lang=None) -> str:
        """Mock processing using predefined mappings.

        Args:
            source_text: The text to process.
            reference_pairs: Optional list of (source, translation) context pairs.
            target_lang: Optional BCP 47 locale string for the target language.
        """
        parts = ["[MOCK PROCESSING"]
        if target_lang:
            parts.append(f" lang={target_lang}")
        if reference_pairs:
            parts.append(f" ref={len(reference_pairs)}")
        parts.append(f"] {source_text}")
        return "".join(parts)
