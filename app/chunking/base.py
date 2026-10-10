from abc import ABC, abstractmethod

from app.ingestion.models import Document, DocumentChunk, DocumentPage


class ChunkingStrategy(ABC):
    """Base interface for document chunking strategies."""

    @abstractmethod
    def chunk(
        self, document: Document, pages: list[DocumentPage]
    ) -> list[DocumentChunk]:
        """Convert document pages into document chunks."""
        raise NotImplementedError
