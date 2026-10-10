from pathlib import Path

from pypdf import PdfReader

from app.ingestion.models import DocumentPage


class DocumentLoader:
    """Extract text from supported document types."""

    SUPPORTED_EXTENSIONS = {".pdf", ".txt"}

    def load(self, file_path: str | Path) -> list[DocumentPage]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")
        extension = path.suffix.lower()

        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {extension}")

        if extension == ".pdf":
            return self.__load_pdf(path)
        return self.__load_text(path)

    def __load_pdf(self, path: Path) -> list[DocumentPage]:
        reader = PdfReader(path)
        pages: list[DocumentPage] = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""

            if not text.strip():
                continue
            pages.append(DocumentPage(text=text, page_number=page_number))

        return pages

    def __load_text(self, path: Path) -> list[DocumentPage]:
        text = path.read_text(encoding="utf-8")
        return [DocumentPage(text=text, page_number=1)]
