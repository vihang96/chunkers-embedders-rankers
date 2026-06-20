"""Chunking module for processing documents into vector-ready chunks.

This module provides intelligent chunking strategies for different document types,
with special focus on preserving hierarchical structure in markdown documents.

Requires the ``cer[chunking]`` extra (chunknorris).
"""

__all__ = []

try:
    from .markdown_chunker import CustomMarkdownChunker, MarkdownChunk
except ImportError:
    pass
else:
    __all__ += ["CustomMarkdownChunker", "MarkdownChunk"]
