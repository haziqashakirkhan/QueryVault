"""A classic keyword search (BM25), used only to demonstrate how it differs
from semantic search.

BM25 scores a chunk by counting how often the *exact words* of the question
appear in it (rare words count more). It has no idea that "car" and
"automobile" mean the same thing: that is the gap embeddings close.
"""
import math
import re
from collections import Counter

K1 = 1.5   # how quickly repeated words stop adding score
B = 0.75   # how strongly long chunks are penalised

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for",
    "from", "how", "in", "is", "it", "of", "on", "or", "that", "the", "this",
    "to", "was", "what", "when", "where", "which", "who", "why", "with",
}


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]


def keyword_search(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    query_terms = set(tokenize(query))
    if not query_terms or not chunks:
        return []

    tokenized = [tokenize(c["text"]) for c in chunks]
    doc_count = len(tokenized)
    avg_len = (sum(len(t) for t in tokenized) / doc_count) or 1.0
    doc_freq = Counter(term for tokens in tokenized for term in set(tokens))

    scored = []
    for chunk, tokens in zip(chunks, tokenized):
        freq = Counter(tokens)
        score, matched = 0.0, []
        for term in query_terms:
            if not freq[term]:
                continue
            idf = math.log(1 + (doc_count - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            norm = freq[term] + K1 * (1 - B + B * len(tokens) / avg_len)
            score += idf * freq[term] * (K1 + 1) / norm
            matched.append(term)
        if score > 0:
            scored.append({**chunk, "score": round(score, 4), "matched_terms": sorted(matched)})

    scored.sort(key=lambda c: c["score"], reverse=True)
    top = scored[:top_k]
    for rank, item in enumerate(top, start=1):
        item["rank"] = rank
    return top
