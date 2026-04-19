"""mdpo-llm: Markdown document processing with LLMs and PO files.

A Python package for processing Markdown documents using Language Learning Models (LLMs)
with GNU gettext PO files for efficient translation workflows.
"""

from .batch import BatchTranslator, MultiTargetBatchTranslator
from .processor import MarkdownProcessor, Mode
from .prompts import Prompts
from .reference_pool import ReferencePool

__version__ = "0.4.0"
__all__ = [
    "MdpoLLM",  # Main class alias
    "Mode",
    "Prompts",
    "MarkdownProcessor",  # Keep for backwards compatibility
    "ReferencePool",
    "BatchTranslator",
    "MultiTargetBatchTranslator",
]

# Create main class alias for better naming
MdpoLLM = MarkdownProcessor
