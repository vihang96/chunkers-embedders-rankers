"""Markdown chunker that preserves document hierarchy.

This module parses markdown files into a syntax tree and creates chunks
while maintaining the hierarchical structure of the document.

1. Parse the markdown to create a tree structure of headers and content
2. Traverse the tree, building the header path as we go
3. For each content block, prepend the current header path
4. Split the content into chunks of appropriate size, preserving tables/codeblocks etc
5. Each chunk retains the prepended header path
"""

from dataclasses import dataclass
import logging
import pathlib
from typing import Any, Dict, List, Optional

from chunknorris.chunkers import MarkdownChunker  # type: ignore
from chunknorris.core import MarkdownLine  # type: ignore
from chunknorris.parsers import MarkdownParser  # type: ignore
from chunknorris.pipelines import BasePipeline  # type: ignore
import tiktoken

from .base import BaseChunker

logger = logging.getLogger(__name__)


@dataclass
class MarkdownChunk:
    """Represents a chunk of markdown content with metadata."""

    text: str  # The actual text content
    node_type: str  # Type of content (heading, paragraph, code, etc.)
    hierarchy_path: str  # Full path in document hierarchy (human-readable)
    hierarchy_ltree: Optional[str]  # ltree-formatted path for database
    metadata: Dict[str, Any]  # Additional metadata
    chunk_id: str  # Unique identifier for the chunk

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary for storage."""
        return {
            "text": self.text,
            "hierarchy_path": self.hierarchy_ltree or "",  # Use ltree path for storage
            "metadata": self.metadata,
            "chunk_id": self.chunk_id,
        }


class CustomMarkdownChunker(BaseChunker):
    """Chunker for markdown documents that preserves hierarchical structure."""

    def __init__(
        self,
        max_chunk_word_count: int = 2000,
        max_headers_to_use: str = "h4",
        hard_max_chunk_token_count: int = 8000,
        tiktoken_encoding: str = "cl100k_base",
    ):
        """Initialize the markdown chunker.

        Args:
                max_chunk_word_count: Maximum number of words per chunk. Soft limit.
                max_headers_to_use: Maximum number of headers to use.
                hard_max_chunk_token_count: Hard limit on the number of tokens per chunk.
                tiktoken_encoding: Encoding to use for tokenization.
        """
        self.max_chunk_word_count = max_chunk_word_count
        self.max_headers_to_use = max_headers_to_use
        self.hard_max_chunk_token_count = hard_max_chunk_token_count
        self.tiktoken_encoding = tiktoken_encoding

    def ltree_from_headers(self, headers: List[MarkdownLine]) -> str:
        """Convert a list of headers to an ltree path."""
        return ".".join([h.text for h in headers])

    def get_content_depth(self, headers: List[MarkdownLine]) -> int:
        """Get the depth of the content."""
        return len(headers)

    def chunk_markdown(self, md: str) -> List[Dict[str, Any]]:
        encoder = tiktoken.get_encoding(self.tiktoken_encoding)

        # 1) Parse Markdown -> internal tree; 2) traverse/bucket by headers
        pipeline = BasePipeline(
            parser=MarkdownParser(),
            chunker=MarkdownChunker(
                max_headers_to_use=self.max_headers_to_use,
                max_chunk_word_count=self.max_chunk_word_count,
                hard_max_chunk_token_count=self.hard_max_chunk_token_count,
                tokenizer=encoder,
            ),
        )

        # 3) & 5) get_text(prepend_headers=True) prepends the header path
        chunks = pipeline.chunk_string(md)
        markdown_chunks: List[Dict[str, Any]] = []
        for id, c in enumerate(chunks):
            text = c.get_text(prepend_headers=True)
            headers = c.headers
            markdown_chunks.append(
                MarkdownChunk(
                    text=text,
                    node_type="",
                    hierarchy_path=self.ltree_from_headers(headers),
                    hierarchy_ltree=self.ltree_from_headers(headers),
                    metadata={
                        "start_line": c.start_line,
                        "word_count": c.word_count,
                    },
                    chunk_id=f"chunk_{id}:depth_{self.get_content_depth(headers)}",
                ).to_dict()
            )
        return markdown_chunks

    def chunk_markdown_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Chunk a markdown file into smaller pieces.

        Args:
                file_path: Path to the markdown file.

        Returns:
                List of chunk dictionaries with text and metadata.
        """
        with open(file_path, "r") as f:
            content = f.read()

        return self.chunk_markdown(content)

    def chunk(self, content: str = "", file_path: str = "") -> List[Dict[str, Any]]:
        """Chunk document content into smaller pieces.

        Args:
                content: Document content as string.
                file_path: Optional path to the source file.

        Returns:
                List of chunk dictionaries with text and metadata.
        """
        if not content and not file_path:
            raise ValueError("Either content or file_path must be provided")

        if content:
            return self.chunk_markdown(content)
        else:
            file_path_path = pathlib.Path(file_path)
            if not self.supports_file_type(file_path_path.suffix):
                raise ValueError(f"File type {file_path_path.suffix} not supported")
            return self.chunk_markdown_file(file_path)

    def supports_file_type(self, file_extension: str) -> bool:
        """Check if this chunker supports the given file type.

        Args:
                file_extension: File extension (e.g., '.md', '.txt').

        Returns:
                True if the chunker can process this file type.
        """
        return file_extension.lower() in [".md", ".markdown", ".mkd", ".mdx"]
