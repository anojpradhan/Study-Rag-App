from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.ingestion.document_loader import DocumentLoader
from app.ingestion.models import Document
from app.ingestion.service import IngestionService
from app.ingestion.text_cleaner import TextCleaner

router = APIRouter()

UPLOAD_DIR = Path("data/uploads")
ALLOWED_EXTENSIONS = {".pdf", ".txt"}

DEFAULT_STUDY_SPACE_ID = "dev-study-space"

ingestion_service = IngestionService(loader=DocumentLoader(), cleaner=TextCleaner())


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    extension = Path(file.filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Not a valid file. Only PDF and TXT Files are suported for now",
        )
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    document_id = str(uuid4())
    safe_filename = f"{document_id}{extension}"
    file_path = UPLOAD_DIR / safe_filename
    content = await file.read()
    file_path.write_bytes(content)
    document = Document(
        document_id=document_id,
        study_space_id=DEFAULT_STUDY_SPACE_ID,
        file_name=file.filename,
        file_path=file_path,
    )
    try:
        pages = ingestion_service.ingest(document)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Document ingestion failed:{exc}"
        ) from exc
    return {
        "document_id": document_id,
        "study_space_id": document.study_space_id,
        "filename": file.filename,
        "content_type": file.content_type,
        "size": len(content),
        "pages": len(pages),
        "status": "uploaded",
    }
