"""Steps 4 to 6: store embeddings in ChromaDB and search them.

ChromaDB keeps three things for every chunk: its text, its embedding and some
metadata (source file, page, chunk number). We create the collection with the
cosine metric, so Chroma returns `distance = 1 - cosine_similarity`:
    distance 0.0  -> identical meaning
    distance ~1.0 -> unrelated
"""
import threading
from collections import Counter

import chromadb
from chromadb.config import Settings as ChromaSettings

_BATCH = 256  # how many chunks we send to Chroma at once


class VectorStore:
    def __init__(self, path, collection_name: str, embedder):
        self.embedder = embedder
        self._lock = threading.Lock()
        client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    # ------------------------------------------------------------- writing
    def add_chunks(self, source: str, chunks: list[dict]) -> int:
        """Replace everything stored for `source` with `chunks`."""
        with self._lock:
            self.collection.delete(where={"source": source})
            for start in range(0, len(chunks), _BATCH):
                batch = chunks[start : start + _BATCH]
                texts = [c["text"] for c in batch]
                self.collection.add(
                    ids=[f"{source}::{c['chunk_index']}" for c in batch],
                    documents=texts,
                    embeddings=self.embedder.embed(texts),
                    metadatas=[
                        {
                            "source": source,
                            "chunk_index": c["chunk_index"],
                            "page": c["page"] or 0,  # Chroma metadata cannot be None
                        }
                        for c in batch
                    ],
                )
        return len(chunks)

    def delete_source(self, source: str) -> None:
        with self._lock:
            self.collection.delete(where={"source": source})

    # ------------------------------------------------------------- reading
    def count(self) -> int:
        return self.collection.count()

    def list_sources(self) -> dict[str, int]:
        """Return {file name: number of chunks}."""
        if self.count() == 0:
            return {}
        metadatas = self.collection.get(include=["metadatas"])["metadatas"]
        return dict(sorted(Counter(m["source"] for m in metadatas).items()))

    def all_chunks(self) -> list[dict]:
        if self.count() == 0:
            return []
        data = self.collection.get(include=["documents", "metadatas"])
        return [_to_chunk(doc, meta) for doc, meta in zip(data["documents"], data["metadatas"])]

    def search(self, query: str, top_k: int) -> list[dict]:
        """Embed the query and return the `top_k` nearest chunks."""
        total = self.count()
        if total == 0:
            return []
        result = self.collection.query(
            query_embeddings=[self.embedder.embed_one(query)],
            n_results=min(top_k, total),
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        rows = zip(result["documents"][0], result["metadatas"][0], result["distances"][0])
        for rank, (text, meta, distance) in enumerate(rows, start=1):
            hit = _to_chunk(text, meta)
            hit.update(rank=rank, distance=round(distance, 4), similarity=round(1 - distance, 4))
            hits.append(hit)
        return hits


def _to_chunk(text: str, meta: dict) -> dict:
    return {
        "text": text,
        "source": meta["source"],
        "page": meta.get("page") or None,
        "chunk_index": meta["chunk_index"],
    }
