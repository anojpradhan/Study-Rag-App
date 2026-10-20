import logging
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user
from app.chunking.fixed_size import FixedSizeChunkingStrategy
from app.chunking.paragraph import ParagraphChunkingStrategy
from app.chunking.service import ChunkingService
from app.db.session import get_db
from app.embeddings.embedding_service import EmbeddingService
from app.ingestion.document_loader import DocumentLoader
from app.ingestion.models import Document as IngestionDocument
from app.ingestion.service import IngestionService
from app.ingestion.text_cleaner import TextCleaner
from app.models.document import StudyDocument
from app.models.study_space import StudySpace
from app.models.user import User
from app.vectorstore.qdrant_service import QdrantService

logger = logging.getLogger(__name__)

router = APIRouter()

UPLOAD_DIR = Path("data/uploads")
ALLOWED_EXTENSIONS = {".pdf", ".txt"}

ingestion_service = IngestionService(
    loader=DocumentLoader(),
    cleaner=TextCleaner(),
)

chunking_service = ChunkingService(
    fixed_size_strategy=FixedSizeChunkingStrategy(
        chunk_size=1000,
        chunk_overlap=200,
    ),
    paragraph_strategy=ParagraphChunkingStrategy(
        chunk_size=1000,
    ),
)

embedding_service = EmbeddingService()
qdrant_service = QdrantService()


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    study_space_id: UUID = Form(...),
    chunking_strategy: str = Form("fixed"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # 1. Validate the requested Study Space and its ownership.
    study_space = (
        db.query(StudySpace)
        .filter(
            StudySpace.id == study_space_id,
            StudySpace.user_id == current_user.id,
        )
        .first()
    )

    if study_space is None:
        raise HTTPException(
            status_code=404,
            detail="Study Space not found.",
        )

    # 2. Validate the upload.
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    extension = Path(file.filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only PDF and TXT files are supported.",
        )

    if chunking_strategy not in {"fixed", "paragraph"}:
        raise HTTPException(
            status_code=400,
            detail="Use 'fixed' or 'paragraph' as the chunking strategy.",
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    # 3. Create a single ID shared by PostgreSQL and Qdrant.
    document_uuid = uuid4()
    document_id = str(document_uuid)
    file_path = UPLOAD_DIR / f"{document_id}{extension}"
    vectors_stored = False

    try:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(content)

        ingestion_document = IngestionDocument(
            document_id=document_id,
            study_space_id=str(study_space.id),
            file_name=file.filename,
            file_path=file_path,
        )

        # 4. Run the existing ingestion pipeline.
        pages = ingestion_service.ingest(ingestion_document)

        chunks = chunking_service.chunk(
            document=ingestion_document,
            pages=pages,
            strategy=chunking_strategy,
        )

        if not chunks:
            raise ValueError("No text chunks could be extracted from this document.")

        embeddings = embedding_service.embed_texts([chunk.text for chunk in chunks])

        stored_chunks = qdrant_service.store_chunks(
            chunks=chunks,
            embeddings=embeddings,
        )
        vectors_stored = stored_chunks > 0

        # 5. Persist document metadata in PostgreSQL.
        db_document = StudyDocument(
            id=document_uuid,
            study_space_id=study_space.id,
            file_name=file.filename,
            file_path=str(file_path),
            content_type=file.content_type,
            file_size=len(content),
            page_count=len(pages),
            chunk_count=len(chunks),
            chunking_strategy=chunking_strategy,
            status="completed",
        )

        db.add(db_document)
        db.commit()
        db.refresh(db_document)

        return {
            "document_id": document_id,
            "study_space_id": str(study_space.id),
            "filename": db_document.file_name,
            "content_type": db_document.content_type,
            "size": db_document.file_size,
            "pages": db_document.page_count,
            "chunks": db_document.chunk_count,
            "chunking_strategy": db_document.chunking_strategy,
            "status": db_document.status,
            "embedded_chunks": len(embeddings),
            "stored_chunks": stored_chunks,
            "vector_collection": "study_chunks",
        }

    except (ValueError, FileNotFoundError) as exc:
        db.rollback()

        if vectors_stored:
            try:
                qdrant_service.delete_document_chunks(document_id)
            except Exception:
                logger.exception(
                    "Failed to clean up Qdrant vectors for %s",
                    document_id,
                )

        file_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=400,
            detail=f"Document processing failed: {exc}",
        ) from exc

    except Exception as exc:
        db.rollback()

        if vectors_stored:
            try:
                qdrant_service.delete_document_chunks(document_id)
            except Exception:
                logger.exception(
                    "Failed to clean up Qdrant vectors for %s",
                    document_id,
                )

        file_path.unlink(missing_ok=True)

        logger.exception(
            "Failed to upload document %s",
            document_id,
        )

        status_code = 500 if isinstance(exc, SQLAlchemyError) else 500

        raise HTTPException(
            status_code=status_code,
            detail="Document upload failed. Please try again.",
        ) from exc
