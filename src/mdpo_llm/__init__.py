"""mdpo-llm: Markdown document processing with LLMs and PO files.

A Python package for processing Markdown documents using Language Learning Models (LLMs)
with GNU gettext PO files for efficient translation and refinement workflows.
"""

from .processor import MarkdownProcessor
from .prompts import Prompts
from .reference_pool import ReferencePool

__version__ = "0.2.0"
__all__ = [
    "MdpoLLM",  # Main class alias
    "Prompts",
    "MarkdownProcessor",  # Keep for backwards compatibility
    "ReferencePool",
]

# Create main class alias for better naming
MdpoLLM = MarkdownProcessor
