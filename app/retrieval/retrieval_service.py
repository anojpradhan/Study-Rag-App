from app.embeddings.embedding_service import EmbeddingService
from app.vectorstore.qdrant_service import QdrantService


class RetrievalService:
    """Retrieve relevant document chunks for a query."""

    def __init__(
        self, embedding_service: EmbeddingService, qdrant_service: QdrantService
    ) -> None:
        self.embedding_service = embedding_service
        self.qdrant_service = qdrant_service

    def retrieve(self, query: str, study_space_id: str, limit: int = 5) -> list[dict]:
        query_embedding = self.embedding_service.embed_text(query)
        return self.qdrant_service.search(
            query_embedding=query_embedding, study_space_id=study_space_id, limit=limit
        )
