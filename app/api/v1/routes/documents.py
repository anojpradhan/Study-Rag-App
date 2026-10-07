from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile

router = APIRouter()

UPLOAD_DIR = Path("data/uploads")
ALLOWED_EXTENSIONS = {".pdf", ".txt"}


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
    return {
        "document_id": document_id,
        "filename": file.filename,
        "content_type": file.content_type,
        "size": len(content),
        "status": "uploaded",
    }
