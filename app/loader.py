"""Step 1 of the pipeline: read a file from disk and return plain text.

Supports PDF, TXT and Markdown. A PDF is returned page by page so that every
chunk can later be traced back to its page number.
"""
from pathlib import Path

from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}


class DocumentLoadError(Exception):
    """Raised when a file cannot be read or contains no extractable text."""


def load_document(path: Path) -> list[dict]:
    """Return a list of sections: [{"text": str, "page": int | None}, ...]."""
    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise DocumentLoadError(f"Unsupported file type '{extension}'. Use one of: {allowed}.")

    try:
        sections = _load_pdf(path) if extension == ".pdf" else _load_text(path)
    except DocumentLoadError:
        raise
    except Exception as exc:  # corrupt PDF, permission problem, ...
        raise DocumentLoadError(f"Could not read '{path.name}': {exc}") from exc

    if not sections:
        raise DocumentLoadError(
            f"'{path.name}' has no extractable text (a scanned PDF needs OCR first)."
        )
    return sections


def _load_text(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [{"text": text, "page": None}] if text.strip() else []


def _load_pdf(path: Path) -> list[dict]:
    reader = PdfReader(str(path))
    if reader.is_encrypted:
        reader.decrypt("")  # many PDFs are "encrypted" with an empty password
    sections = []
    for number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            sections.append({"text": text, "page": number})
    return sections
