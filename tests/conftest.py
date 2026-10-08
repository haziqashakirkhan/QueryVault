import hashlib
import json
import math
import re
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import BASE_DIR, Settings
from app.main import create_app


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder: no model download needed."""
    dim = 64

    def embed(self, texts):
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out

    def embed_one(self, text):
        return self.embed([text])[0]


class FakeLLM:
    model = "fake-model"
    ready = True

    def chat(self, messages, **kwargs):
        if kwargs.get("json_mode"):
            return json.dumps({"accuracy": 4, "clarity": 5, "relevance": 5, "reason": "Looks fine."})
        return "A grounded fake answer [1]."


@pytest.fixture()
def tmp_settings(tmp_path):
    docs = tmp_path / "documents"
    shutil.copytree(BASE_DIR / "data" / "documents", docs)
    return Settings(
        groq_api_key="test",
        documents_dir=docs,
        chroma_dir=tmp_path / "chroma",
        few_shot_file=BASE_DIR / "data" / "few_shot_examples.json",
        eval_questions_file=BASE_DIR / "data" / "eval_questions.json",
        eval_results_file=tmp_path / "eval_results.json",
        chunk_size=400,
        chunk_overlap=80,
        top_k=3,
    )


@pytest.fixture()
def client(tmp_settings):
    app = create_app(tmp_settings, embedder=FakeEmbedder(), llm=FakeLLM())
    with TestClient(app) as c:
        yield c
