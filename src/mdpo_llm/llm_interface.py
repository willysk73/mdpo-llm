"""
LLM interface for processing services.
"""

from abc import ABC, abstractmethod


class LLMInterface(ABC):
    """Abstract base class for LLM processing interfaces."""

    @abstractmethod
    def process(self, source_text: str) -> str:
        """Process source text to target language.

        Implementations may also accept an optional ``reference_pairs``
        keyword argument — a list of ``(source, translation)`` tuples
        providing few-shot context from previously translated blocks in
        the same document.  The processor detects support for this
        parameter via ``inspect.signature`` and passes it automatically
        when available.
        """
        pass


class MockLLMInterface(LLMInterface):
    """Mock implementation for testing and development."""

    def process(self, source_text: str, reference_pairs=None) -> str:
        """Mock processing using predefined mappings.

        Args:
            source_text: The text to process.
            reference_pairs: Optional list of (source, translation) context pairs.
        """
        if reference_pairs:
            return f"[MOCK PROCESSING ref={len(reference_pairs)}] {source_text}"
        return f"[MOCK PROCESSING] {source_text}"
