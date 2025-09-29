"""
LLM interface for processing services.
"""

from abc import ABC, abstractmethod


class LLMInterface(ABC):
    """Abstract base class for LLM processing interfaces."""

    @abstractmethod
    def process(self, source_text: str) -> str:
        """Process source text to target language. Accepts additional keyword arguments for extensibility."""
        pass


class MockLLMInterface(LLMInterface):
    """Mock implementation for testing and development."""

    def process(self, source_text: str) -> str:
        """Mock processing using predefined mappings."""
        # Return mock processing or create a fallback
        return f"[MOCK PROCESSING] {source_text}"
