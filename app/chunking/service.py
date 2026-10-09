from app.chunking.base import ChunkingStrategy
from app.ingestion.models import Document, DocumentChunk, DocumentPage


class ChunkingService:
    """Coordinates document chunking."""

    def __init__(
        self,
        fixed_size_strategy: ChunkingStrategy,
        paragraph_strategy: ChunkingStrategy,
    ) -> None:
        self.fixed_size_strategy = fixed_size_strategy
        self.paragraph_strategy = paragraph_strategy

    def chunk(
        self,
        document: Document,
        pages: list[DocumentPage],
        strategy: str = "fixed",
    ) -> list[DocumentChunk]:

        if strategy == "fixed":
            chunking_strategy = self.fixed_size_strategy

        elif strategy == "paragraph":
            chunking_strategy = self.paragraph_strategy

        else:
            raise ValueError(f"Unsupported chunking strategy: {strategy}")

        return chunking_strategy.chunk(
            document=document,
            pages=pages,
        )
