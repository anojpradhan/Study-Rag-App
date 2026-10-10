import re

from app.ingestion.models import DocumentPage


class TextCleaner:
    """Normalize extracted document texts."""

    def clean(self, page: DocumentPage) -> DocumentPage:
        text = page.text

        # Normalize line endings
        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        # remove Trailing whitespaces from each lines
        text = "\n".join(line.rstrip() for line in text.splitlines())

        # collapse excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)

        # collapse repeated spaces and tabs
        text = re.sub(r"[ \t]+", " ", text)

        text = text.strip()

        return DocumentPage(text=text, page_number=page.page_number)
