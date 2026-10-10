from app.chunking.base import ChunkingStrategy
from app.ingestion.models import Document, DocumentChunk, DocumentPage


class ParagraphChunkingStrategy(ChunkingStrategy):
    """Group paragraphs into chunks without exceeding the target size."""

    def __init__(
        self,
        chunk_size: int = 1000,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        self.chunk_size = chunk_size

    def chunk(
        self,
        document: Document,
        pages: list[DocumentPage],
    ) -> list[DocumentChunk]:

        chunks: list[DocumentChunk] = []
        chunk_index = 0

        for page in pages:
            paragraphs = [
                paragraph.strip()
                for paragraph in page.text.split("\n\n")
                if paragraph.strip()
            ]

            current_parts: list[str] = []
            current_length = 0

            for paragraph in paragraphs:
                paragraph_length = len(paragraph)

                if (
                    current_parts
                    and current_length + paragraph_length + 2 > self.chunk_size
                ):
                    chunks.append(
                        DocumentChunk(
                            text="\n\n".join(current_parts),
                            document_id=document.document_id,
                            study_space_id=document.study_space_id,
                            page_number=page.page_number,
                            chunk_index=chunk_index,
                        )
                    )

                    chunk_index += 1

                    current_parts = []
                    current_length = 0

                current_parts.append(paragraph)
                current_length += paragraph_length + 2

            if current_parts:
                chunks.append(
                    DocumentChunk(
                        text="\n\n".join(current_parts),
                        document_id=document.document_id,
                        study_space_id=document.study_space_id,
                        page_number=page.page_number,
                        chunk_index=chunk_index,
                    )
                )

                chunk_index += 1

        return chunks
