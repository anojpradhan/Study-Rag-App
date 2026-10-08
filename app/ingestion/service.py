from app.ingestion.document_loader import DocumentLoader
from app.ingestion.models import Document, DocumentPage
from app.ingestion.text_cleaner import TextCleaner


class IngestionService:
    """Coordinates the document ingestion pipeline."""

    def __init__(
        self,
        loader: DocumentLoader,
        cleaner: TextCleaner,
    ) -> None:
        self.loader = loader
        self.cleaner = cleaner

    def ingest(self, document: Document) -> list[DocumentPage]:
        pages = self.loader.load(document.file_path)
        cleaned_pages = [self.cleaner.clean(page) for page in pages]
        return cleaned_pages
