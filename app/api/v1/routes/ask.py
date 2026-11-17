import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user
from app.db.session import get_db
from app.embeddings.embedding_service import EmbeddingService
from app.llm.gemini_service import GeminiService
from app.models.study_space import StudySpace
from app.models.user import User
from app.rag.rag_service import RAGService
from app.retrieval.retrieval_service import RetrievalService
from app.schemas.ask import AskRequest, AskResponse
from app.services.memory_service import MemoryService
from app.vectorstore.qdrant_service import QdrantService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ask", tags=["RAG"])

embedding_service = EmbeddingService()
qdrant_service = QdrantService()

retrieval_service = RetrievalService(
    embedding_service=embedding_service,
    qdrant_service=qdrant_service,
)

gemini_service = GeminiService()
memory_service = MemoryService()

rag_service = RAGService(
    retrieval_service=retrieval_service,
    gemini_service=gemini_service,
    memory_service=memory_service,
)


@router.post("/", response_model=AskResponse)
def ask_question(
    request: AskRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AskResponse:

    study_space = (
        db.query(StudySpace)
        .filter(
            StudySpace.id == request.study_space_id,
            StudySpace.user_id == current_user.id,
        )
        .first()
    )

    if study_space is None:
        raise HTTPException(
            status_code=404,
            detail="Study Space not found.",
        )

    try:
        result = rag_service.answer_question(
            query=request.query,
            study_space_id=str(study_space.id),
            session_id=request.session_id,
            limit=request.limit,
        )

        return AskResponse(**result)

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Failed to process the RAG ask request.")

        raise HTTPException(
            status_code=500,
            detail="Failed to generate an answer from study materials.",
        ) from exc
