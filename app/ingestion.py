"""Glue between loader, chunker and vector store: file in, vectors out."""
from pathlib import Path

from .chunker import split_text
from .loader import SUPPORTED_EXTENSIONS, DocumentLoadError, load_document


def ingest_file(path: Path, store, chunk_size: int, overlap: int) -> int:
    """Load, chunk, embed and store one file. Returns the number of chunks."""
    sections = load_document(path)
    chunks, index = [], 0
    for section in sections:
        for piece in split_text(section["text"], chunk_size, overlap):
            chunks.append({"text": piece, "page": section["page"], "chunk_index": index})
            index += 1
    if not chunks:
        raise DocumentLoadError(f"'{path.name}' produced no text chunks.")
    return store.add_chunks(path.name, chunks)


def sync_directory(directory: Path, store, chunk_size: int, overlap: int) -> list[str]:
    """Index every supported file in `directory` that is not indexed yet."""
    directory.mkdir(parents=True, exist_ok=True)
    already_indexed = set(store.list_sources())
    added = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS or path.name in already_indexed:
            continue
        try:
            ingest_file(path, store, chunk_size, overlap)
            added.append(path.name)
        except DocumentLoadError as exc:
            print(f"[skip] {exc}")
    return added
