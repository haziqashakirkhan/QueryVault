"""FastAPI application: the HTTP layer on top of the RAG pipeline."""
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import BASE_DIR, settings as default_settings
from .evaluation import load_latest, run_evaluation
from .ingestion import ingest_file, sync_directory
from .keyword_search import keyword_search
from .llm import LLMClient, LLMError, LLMNotConfigured
from .loader import SUPPORTED_EXTENSIONS, DocumentLoadError
from .prompts import TECHNIQUES
from .rag import RAGService
from .vectorstore import VectorStore

STATIC_DIR = BASE_DIR / "static"


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    technique: Literal["zero_shot", "few_shot", "role_based"] = "role_based"


class CompareRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


def _safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._ -]", "_", Path(name or "").name).strip(" .")
    return cleaned or "document"


def _require_text(value: str, field: str) -> str:
    value = value.strip()
    if not value:
        raise HTTPException(400, f"{field} cannot be empty.")
    return value


def create_app(settings=default_settings, embedder=None, llm=None) -> FastAPI:
    """App factory. Tests pass a fake embedder/LLM; production uses the defaults."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        emb = embedder
        if emb is None:
            from .embeddings import Embedder

            emb = Embedder(settings.embedding_model)
        store = VectorStore(settings.chroma_dir, settings.collection_name, emb)
        client = llm or LLMClient(
            settings.groq_api_key, settings.llm_model, settings.temperature, settings.max_tokens
        )
        app.state.store = store
        app.state.rag = RAGService(store, client, settings.top_k, settings.few_shot_file)
        sync_directory(settings.documents_dir, store, settings.chunk_size, settings.chunk_overlap)
        yield

    app = FastAPI(title="Marginalia", lifespan=lifespan)

    def services(request: Request):
        return request.app.state.store, request.app.state.rag

    def guard_llm(call):
        try:
            return call()
        except LLMNotConfigured as exc:
            raise HTTPException(503, str(exc))
        except LLMError as exc:
            raise HTTPException(502, str(exc))

    # ---------------------------------------------------------------- info
    @app.get("/api/config")
    def get_config(request: Request):
        store, rag = services(request)
        return {
            "llm_model": settings.llm_model,
            "judge_model": settings.judge_model,
            "embedding_model": settings.embedding_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "top_k": settings.top_k,
            "llm_ready": rag.llm.ready,
            "chunk_count": store.count(),
            "accepted_types": sorted(SUPPORTED_EXTENSIONS),
            "techniques": [{"id": k, **v} for k, v in TECHNIQUES.items()],
        }

    # ----------------------------------------------------------- documents
    @app.get("/api/documents")
    def list_documents(request: Request):
        store, _ = services(request)
        return [{"name": n, "chunks": c} for n, c in store.list_sources().items()]

    @app.post("/api/documents")
    async def upload_documents(request: Request, files: list[UploadFile] = File(...)):
        store, _ = services(request)
        limit = settings.max_upload_mb * 1024 * 1024
        saved, errors = [], []
        settings.documents_dir.mkdir(parents=True, exist_ok=True)

        for upload in files:
            name = _safe_filename(upload.filename)
            if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
                errors.append({"name": name, "error": "Unsupported file type. Use PDF, TXT or Markdown."})
                continue
            data = await upload.read(limit + 1)
            if len(data) > limit:
                errors.append({"name": name, "error": f"File is larger than {settings.max_upload_mb} MB."})
                continue
            target = settings.documents_dir / name
            target.write_bytes(data)
            try:
                count = ingest_file(target, store, settings.chunk_size, settings.chunk_overlap)
                saved.append({"name": name, "chunks": count})
            except DocumentLoadError as exc:
                target.unlink(missing_ok=True)
                errors.append({"name": name, "error": str(exc)})
        return {"saved": saved, "errors": errors}

    @app.delete("/api/documents/{name}")
    def delete_document(name: str, request: Request):
        store, _ = services(request)
        name = _safe_filename(name)
        if name not in store.list_sources():
            raise HTTPException(404, "Document not found.")
        store.delete_source(name)
        (settings.documents_dir / name).unlink(missing_ok=True)
        return {"deleted": name}

    # ----------------------------------------------------- Part 2: search
    @app.post("/api/search")
    def search(body: SearchRequest, request: Request):
        store, _ = services(request)
        query = _require_text(body.query, "Query")
        top_k = body.top_k or settings.top_k
        return {
            "semantic": store.search(query, top_k),
            "keyword": keyword_search(query, store.all_chunks(), top_k),
        }

    # ------------------------------------------------- Part 1: generation
    @app.post("/api/ask")
    def ask(body: AskRequest, request: Request):
        _, rag = services(request)
        question = _require_text(body.question, "Question")
        if not request.app.state.store.count():
            raise HTTPException(400, "The library is empty. Upload a document first.")
        return guard_llm(lambda: rag.ask(question, body.technique))

    @app.post("/api/compare")
    def compare(body: CompareRequest, request: Request):
        _, rag = services(request)
        question = _require_text(body.question, "Question")
        if not request.app.state.store.count():
            raise HTTPException(400, "The library is empty. Upload a document first.")
        return guard_llm(lambda: rag.compare(question))

    @app.post("/api/evaluate")
    def evaluate(request: Request):
        store, rag = services(request)
        if not store.count():
            raise HTTPException(400, "The library is empty. Upload a document first.")
        if not rag.llm.ready:
            raise HTTPException(503, "GROQ_API_KEY is missing. Add it to .env and restart.")
        try:
            return run_evaluation(
                rag, settings.judge_model, settings.eval_questions_file, settings.eval_results_file
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        except LLMError as exc:
            raise HTTPException(502, str(exc))

    @app.get("/api/evaluate/latest")
    def latest_evaluation():
        return load_latest(settings.eval_results_file)

    # -------------------------------------------------------------- static
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
