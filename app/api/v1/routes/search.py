import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user
from app.db.session import get_db
from app.embeddings.embedding_service import EmbeddingService
from app.models.study_space import StudySpace
from app.models.user import User
from app.retrieval.retrieval_service import RetrievalService
from app.schemas.search import SearchRequest, SearchResponse
from app.vectorstore.qdrant_service import QdrantService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["Search"])

embedding_service = EmbeddingService()
qdrant_service = QdrantService()

retrieval_service = RetrievalService(
    embedding_service=embedding_service,
    qdrant_service=qdrant_service,
)


@router.post("/", response_model=SearchResponse)
def search_documents(
    request: SearchRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SearchResponse:

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
        results = retrieval_service.retrieve(
            query=request.query,
            study_space_id=str(study_space.id),
            limit=request.limit,
        )

        return SearchResponse(
            query=request.query,
            results=results,
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Failed to search document chunks.")

        raise HTTPException(
            status_code=500,
            detail="Failed to search document chunks.",
        ) from exc
