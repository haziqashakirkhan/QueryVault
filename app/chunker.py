"""Step 2: split long text into small overlapping chunks.

Why chunk at all? An embedding model squeezes a whole text into one vector. If
the text is a 20-page document, that vector is a blurry average of every topic
in it. Small chunks give each vector one clear meaning, so retrieval is precise
and only a few relevant paragraphs are sent to the LLM.

Strategy: split into sentences/paragraphs, then pack whole sentences into a
chunk until it is about `chunk_size` characters. The last sentences of each
chunk are repeated at the start of the next one (the overlap) so an answer that
sits on a boundary is not cut in half.
"""
import re

_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _units(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Sentences/paragraphs; anything longer than a chunk is hard-split."""
    units = []
    for piece in _BOUNDARY.split(_clean(text)):
        piece = piece.strip()
        if not piece:
            continue
        if len(piece) <= chunk_size:
            units.append(piece)
        else:
            step = chunk_size - overlap
            units.extend(piece[i : i + chunk_size] for i in range(0, len(piece), step))
    return units


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    chunks: list[str] = []
    current: list[str] = []
    length = 0

    for unit in _units(text, chunk_size, overlap):
        extra = len(unit) + (1 if current else 0)
        if current and length + extra > chunk_size:
            chunks.append(" ".join(current))
            # keep the trailing sentences (up to `overlap` characters) as context
            tail: list[str] = []
            tail_len = 0
            for previous in reversed(current):
                if tail_len + len(previous) + 1 > overlap:
                    break
                tail.insert(0, previous)
                tail_len += len(previous) + 1
            current = tail
            length = sum(len(u) for u in current) + max(len(current) - 1, 0)
            extra = len(unit) + (1 if current else 0)
        current.append(unit)
        length += extra

    if current:
        chunks.append(" ".join(current))
    return chunks
