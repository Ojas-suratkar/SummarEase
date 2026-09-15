"""Text extraction from PDF files using PyMuPDF (fitz)."""
from __future__ import annotations

import fitz  # PyMuPDF


class PdfExtractionError(RuntimeError):
    pass


def extract_text(file_stream) -> str:
    """Extract plain text from an in-memory/uploaded PDF file stream."""
    try:
        data = file_stream.read()
        with fitz.open(stream=data, filetype="pdf") as doc:
            pages = [page.get_text() for page in doc]
        return "\n".join(pages).strip()
    except Exception as exc:
        raise PdfExtractionError(f"Could not read PDF: {exc}") from exc
