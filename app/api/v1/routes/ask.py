from fastapi import APIRouter, HTTPException

from app.embeddings.embedding_service import EmbeddingService
from app.llm.gemini_service import GeminiService
from app.rag.rag_service import RAGService
from app.retrieval.retrieval_service import RetrievalService
from app.schemas.ask import AskRequest, AskResponse
from app.vectorstore.qdrant_service import QdrantService

router = APIRouter(prefix="/ask", tags=["RAG"])

embedding_service = EmbeddingService()
qdrant_service = QdrantService()

retrieval_service = RetrievalService(
    embedding_service=embedding_service,
    qdrant_service=qdrant_service,
)

gemini_service = GeminiService()

rag_service = RAGService(
    retrieval_service=retrieval_service,
    gemini_service=gemini_service,
)


@router.post("/", response_model=AskResponse)
def ask_question(request: AskRequest) -> AskResponse:
    try:
        result = rag_service.answer_question(
            query=request.query,
            study_space_id=request.study_space_id,
            limit=request.limit,
        )

        return AskResponse(**result)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to generate an answer from study materials.",
        ) from exc
