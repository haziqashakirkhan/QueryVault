"""Central configuration.

Every tunable value lives here and is read from environment variables (or a
local .env file). Nothing else in the project hard-codes a model name, a path
or a number: change the .env file and the whole system follows.
"""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _path(env_name: str, default: str) -> Path:
    value = Path(os.getenv(env_name, default))
    return value if value.is_absolute() else BASE_DIR / value


@dataclass(frozen=True)
class Settings:
    # --- LLM (Groq) -------------------------------------------------------
    groq_api_key: str = os.getenv("GROQ_API_KEY", "").strip()
    llm_model: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    judge_model: str = os.getenv("JUDGE_MODEL", os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.1"))
    max_tokens: int = int(os.getenv("MAX_TOKENS", "500"))

    # --- Embeddings and chunking -----------------------------------------
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    top_k: int = int(os.getenv("TOP_K", "3"))

    # --- Storage ----------------------------------------------------------
    documents_dir: Path = _path("DOCUMENTS_DIR", "data/documents")
    chroma_dir: Path = _path("CHROMA_DIR", "data/chroma")
    collection_name: str = os.getenv("COLLECTION_NAME", "documents")
    few_shot_file: Path = _path("FEW_SHOT_FILE", "data/few_shot_examples.json")
    eval_questions_file: Path = _path("EVAL_QUESTIONS_FILE", "data/eval_questions.json")
    eval_results_file: Path = _path("EVAL_RESULTS_FILE", "data/eval_results.json")
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "15"))


settings = Settings()
