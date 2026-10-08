from dataclasses import dataclass
from pathlib import Path


@dataclass
class Document:
    """Application-level representation of an uploaded document."""

    document_id: str
    study_space_id: str
    file_name: str
    file_path: Path


@dataclass
class DocumentPage:
    """Extracted text from a single document page."""

    text: str
    page_number: int


@dataclass
class DocumentChunk:
    """A chunk of document text prepared for embedding and retrieval"""

    text: str
    document_id: str
    study_space_id: str
    page_number: int
    chunk_index: int
