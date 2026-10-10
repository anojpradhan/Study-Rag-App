from fastapi import APIRouter, HTTPException

from app.embeddings.embedding_service import EmbeddingService
from app.retrieval.retrieval_service import RetrievalService
from app.schemas.search import SearchRequest, SearchResponse
from app.vectorstore.qdrant_service import QdrantService

router = APIRouter(prefix="/search", tags=["Search"])

embedding_service = EmbeddingService()
qdrant_service = QdrantService()
retrieval_service = RetrievalService(
    embedding_service=embedding_service,
    qdrant_service=qdrant_service,
)


@router.post("/", response_model=SearchResponse)
def search_documents(request: SearchRequest) -> SearchResponse:
    try:
        results = retrieval_service.retrieve(
            query=request.query,
            study_space_id=request.study_space_id,
            limit=request.limit,
        )

        return SearchResponse(
            query=request.query,
            results=results,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to search document chunks",
        ) from exc
