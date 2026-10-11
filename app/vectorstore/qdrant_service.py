from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from app.ingestion.models import DocumentChunk


class QdrantService:
    """Store and search document embeddings in Qdrant."""

    def __init__(
        self,
        collection_name: str = "study_chunks",
        url: str = "http://localhost:6333",
    ) -> None:
        self.collection_name = collection_name

        self.client = QdrantClient(url=url)

        self._create_collection()

    def _create_collection(self) -> None:
        """Create the vector collection if it does not exist."""

        if self.client.collection_exists(
            collection_name=self.collection_name,
        ):
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=384,
                distance=models.Distance.COSINE,
            ),
        )

    def store_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> int:
        """Store chunk vectors and their metadata."""

        if len(chunks) != len(embeddings):
            raise ValueError(
                "The number of chunks must match the number of embeddings."
            )

        if not chunks:
            return 0

        # Recreate the collection if it was deleted.
        self._create_collection()
        points = []

        for chunk, embedding in zip(
            chunks,
            embeddings,
            strict=True,
        ):
            point_id = str(
                uuid5(
                    NAMESPACE_URL,
                    f"{chunk.document_id}:{chunk.page_number}:{chunk.chunk_index}",
                )
            )

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "text": chunk.text,
                        "document_id": chunk.document_id,
                        "study_space_id": chunk.study_space_id,
                        "page_number": chunk.page_number,
                        "chunk_index": chunk.chunk_index,
                    },
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

        return len(points)

    def search(
        self,
        query_embedding: list[float],
        study_space_id: str,
        limit: int = 5,
    ) -> list[dict]:
        """Search for relevant, unique chunks within one Study Space."""

        # Retrieve extra candidates because some may contain duplicate text.
        candidate_limit = min(limit * 3, 60)

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="study_space_id",
                        match=models.MatchValue(value=study_space_id),
                    )
                ]
            ),
            limit=candidate_limit,
            with_payload=True,
        ).points

        unique_results = []
        seen_texts = set()

        for result in results:
            payload = result.payload or {}
            text = payload.get("text", "")

            # Normalize whitespace and capitalization for duplicate detection.
            normalized_text = " ".join(text.casefold().split())

            if not normalized_text or normalized_text in seen_texts:
                continue

            seen_texts.add(normalized_text)

            unique_results.append(
                {
                    "score": result.score,
                    **payload,
                }
            )

            if len(unique_results) >= limit:
                break

        return unique_results
