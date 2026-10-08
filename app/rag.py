"""The RAG service: retrieve first, then generate.

    question --> embed --> top-k chunks --> prompt (technique) --> LLM --> answer
"""
import json
from pathlib import Path

from .prompts import TECHNIQUES, build_messages, render_prompt


def load_json_list(path: Path) -> list:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


class RAGService:
    def __init__(self, store, llm, top_k: int, few_shot_file: Path):
        self.store = store
        self.llm = llm
        self.top_k = top_k
        self.few_shot_file = few_shot_file

    def _examples(self) -> list[dict]:
        # re-read on every call so edits to the JSON file apply without a restart
        return load_json_list(self.few_shot_file)

    def retrieve(self, question: str) -> list[dict]:
        return self.store.search(question, self.top_k)

    def generate(self, technique: str, question: str, chunks: list[dict]) -> dict:
        messages = build_messages(technique, question, chunks, self._examples())
        answer = self.llm.chat(messages)
        return {"technique": technique, "answer": answer, "prompt": render_prompt(messages)}

    def ask(self, question: str, technique: str) -> dict:
        chunks = self.retrieve(question)
        result = self.generate(technique, question, chunks)
        result["chunks"] = chunks
        return result

    def compare(self, question: str) -> dict:
        """Run all three techniques on one shared retrieval (a fair test)."""
        chunks = self.retrieve(question)
        results = {t: self.generate(t, question, chunks) for t in TECHNIQUES}
        return {"chunks": chunks, "results": results}
