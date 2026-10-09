from app.chunking.base import ChunkingStrategy
from app.ingestion.models import Document, DocumentChunk, DocumentPage


class FixedSizeChunkingStrategy(ChunkingStrategy):
    """Split document text into fixed-size overlapping chunks."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(
        self,
        document: Document,
        pages: list[DocumentPage],
    ) -> list[DocumentChunk]:

        chunks: list[DocumentChunk] = []
        chunk_index = 0

        step = self.chunk_size - self.chunk_overlap

        for page in pages:
            text = page.text.strip()

            if not text:
                continue

            for start in range(0, len(text), step):
                chunk_text = text[start : start + self.chunk_size].strip()

                if not chunk_text:
                    continue

                chunks.append(
                    DocumentChunk(
                        text=chunk_text,
                        document_id=document.document_id,
                        study_space_id=document.study_space_id,
                        page_number=page.page_number,
                        chunk_index=chunk_index,
                    )
                )

                chunk_index += 1

                if start + self.chunk_size >= len(text):
                    break

        return chunks
