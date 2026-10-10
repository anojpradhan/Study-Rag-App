import logging
from pathlib import Path
from uuid import uuid4

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
)

from app.chunking.fixed_size import FixedSizeChunkingStrategy
from app.chunking.paragraph import ParagraphChunkingStrategy
from app.chunking.service import ChunkingService
from app.embeddings.embedding_service import EmbeddingService
from app.ingestion.document_loader import DocumentLoader
from app.ingestion.models import Document
from app.ingestion.service import IngestionService
from app.ingestion.text_cleaner import TextCleaner
from app.vectorstore.qdrant_service import QdrantService

logger = logging.getLogger(__name__)

router = APIRouter()

UPLOAD_DIR = Path("data/uploads")
ALLOWED_EXTENSIONS = {".pdf", ".txt"}
DEFAULT_STUDY_SPACE_ID = "dev-study-space"


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
    chunking_strategy: str = Form("fixed"),
):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required",
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

    UPLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    document_id = str(uuid4())
    file_path = UPLOAD_DIR / f"{document_id}{extension}"

    file_path.write_bytes(content)

    document = Document(
        document_id=document_id,
        study_space_id=DEFAULT_STUDY_SPACE_ID,
        file_name=file.filename,
        file_path=file_path,
    )

    try:
        pages = ingestion_service.ingest(document)

        chunks = chunking_service.chunk(
            document=document,
            pages=pages,
            strategy=chunking_strategy,
        )
        embeddings = embedding_service.embed_texts([chunk.text for chunk in chunks])
        stored_chunks = qdrant_service.store_chunks(
            chunks=chunks, embeddings=embeddings
        )

    except (ValueError, FileNotFoundError) as exc:
        file_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=400,
            detail=f"Document processing failed: {exc}",
        ) from exc

    except Exception as exc:
        logger.exception(
            "Unexpected error processing document %s",
            document_id,
        )

        file_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    return {
        "document_id": document.document_id,
        "study_space_id": document.study_space_id,
        "filename": document.file_name,
        "content_type": file.content_type,
        "size": len(content),
        "pages": len(pages),
        "chunks": len(chunks),
        "chunking_strategy": chunking_strategy,
        "status": "chunked",
        "embedded_chunks": len(embeddings),
        "stored_chunks": stored_chunks,
        "vector_collection": "study_chunks",
    }
