"""Base classes and interfaces for document chunkers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseChunker(ABC):
    """Abstract base class for document chunkers."""

    @abstractmethod
    def chunk(self, content: str = "", file_path: str = "") -> List[Dict[str, Any]]:
        """Chunk document content into smaller pieces.

        Args:
                content: Document content as string.
                file_path: Optional path to the source file.

        Returns:
                List of chunk dictionaries with text and metadata.
        """
        pass

    @abstractmethod
    def supports_file_type(self, file_extension: str) -> bool:
        """Check if this chunker supports the given file type.

        Args:
                file_extension: File extension (e.g., '.md', '.txt').

        Returns:
                True if the chunker can process this file type.
        """
        pass
