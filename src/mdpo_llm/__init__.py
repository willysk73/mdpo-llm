"""mdpo-llm: Markdown document processing with LLMs and PO files.

A Python package for processing Markdown documents using Language Learning Models (LLMs)
with GNU gettext PO files for efficient translation and refinement workflows.
"""

from .llm_interface import LLMInterface, MockLLMInterface
from .processor import MarkdownProcessor
from .language import LanguageCode
from .reference_pool import ReferencePool

__version__ = "0.1.0"
__all__ = [
    "MdpoLLM",  # Main class alias
    "LLMInterface",
    "MockLLMInterface",
    "LanguageCode",
    "MarkdownProcessor",  # Keep for backwards compatibility
    "ReferencePool",
]

# Create main class alias for better naming
MdpoLLM = MarkdownProcessor
