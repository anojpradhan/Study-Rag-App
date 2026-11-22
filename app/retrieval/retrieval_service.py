from app.embeddings.embedding_service import EmbeddingService
from app.vectorstore.qdrant_service import QdrantService


class RetrievalService:
    """Retrieve relevant document chunks for a query."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        qdrant_service: QdrantService,
    ) -> None:
        self.embedding_service = embedding_service
        self.qdrant_service = qdrant_service

    def retrieve(
        self,
        query: str,
        study_space_id: str,
        limit: int = 5,
    ) -> list[dict]:
        """Retrieve semantically relevant chunks."""

        query_embedding = self.embedding_service.embed_text(query)

        return self.qdrant_service.search(
            query_embedding=query_embedding,
            study_space_id=study_space_id,
            limit=limit,
        )

    def retrieve_all(
        self,
        study_space_id: str,
        limit: int = 100,
    ) -> list[dict]:
        """Retrieve document chunks without semantic similarity filtering."""

        return self.qdrant_service.get_all_chunks(
            study_space_id=study_space_id,
            limit=limit,
        )
