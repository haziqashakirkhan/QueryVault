# QueriVault: AI-Powered Document Question Answering

Upload PDF, TXT or Markdown files and ask questions about them. Answers are produced with
Retrieval-Augmented Generation (RAG): sentence-transformer embeddings, a ChromaDB vector store,
and a Groq-hosted LLM. Every answer shows the passages it was based on, with similarity scores.

## How it works

```
UPLOAD   Document -> Text extraction -> Chunking -> Embeddings -> ChromaDB
ASK      Question -> Question embedding -> Semantic search -> Top 3 chunks
                  -> Prompt (technique + context + question) -> LLM -> Answer
```

1. **Load**: text is extracted from PDF (with page numbers), TXT or Markdown files.
2. **Chunk**: text is split on sentence boundaries into chunks of about 500 characters, with
   about 100 characters of overlap so an answer on a boundary is not cut in half.
3. **Embed**: each chunk becomes a 384-number vector using `sentence-transformers/all-MiniLM-L6-v2`
   (runs locally, no API call).
4. **Store**: ChromaDB keeps each chunk's text, vector and metadata (source, page, chunk index),
   using the cosine metric.
5. **Retrieve**: the question is embedded with the same model and the 3 nearest chunks are returned.
   Similarity is shown as `1 - cosine distance`.
6. **Generate**: the chunks are placed in a prompt and sent to the LLM, which answers from them.

The LLM never queries the database. Retrieved text reaches it only as text inside the prompt.

## Prompting techniques (Part 1)

All three receive the same retrieved context and the same question, so any difference in the
answers comes from the prompt alone.

| Technique | What changes |
| --- | --- |
| Zero-shot | A plain instruction, no examples |
| Few-shot | The same instruction plus worked examples from `data/few_shot_examples.json` |
| Role-based | A system message gives the model a job and rules: use only the excerpts, cite them as [1] [2], and say so when the answer is not in them |

## Setup

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # Windows: copy .env.example .env
# open .env and paste your free key from https://console.groq.com into GROQ_API_KEY

python -m uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000. The first start downloads the embedding model (about 90 MB) and
indexes the documents in `data/documents/`.

Run the server from the project root, because the code uses package imports (`app.main`). Running
`uvicorn main:app` will fail. Do not move or rename a folder that contains `.venv`; recreate the
environment instead.

Retrieval works without an API key. The key is only needed for generated answers and evaluation.

## Interface

| View | What it does |
| --- | --- |
| Workspace | Upload documents, ask a question, choose a prompt technique, read the grounded answer and the top 3 retrieved chunks with similarity scores |
| Evaluation | Compares zero-shot, few-shot and role-based on accuracy, clarity and relevance |
| Settings | Shows the active configuration read from the server |

## API

| Method | Route | Purpose |
| --- | --- | --- |
| GET | `/api/config` | Models, chunk settings, top-k, whether the LLM key is set |
| GET | `/api/documents` | Indexed documents and chunk counts |
| POST | `/api/documents` | Upload files (`multipart/form-data`, field `files`) |
| DELETE | `/api/documents/{name}` | Remove a document and its chunks |
| POST | `/api/search` | Semantic and keyword (BM25) results side by side |
| POST | `/api/ask` | Answer one question: `{"question": "...", "technique": "zero_shot \| few_shot \| role_based"}` |
| POST | `/api/compare` | Run all three techniques on one shared retrieval |
| POST | `/api/evaluate` | Five questions x three techniques, scored 1 to 5 by an LLM judge |
| GET | `/api/evaluate/latest` | Most recent saved evaluation (`data/eval_results.json`) |

Errors return a JSON `detail` message: 400 for empty input or an empty library, 502 when the
model request fails, 503 when `GROQ_API_KEY` is missing.

## Evaluation method

The same five questions in `data/eval_questions.json` are answered with each technique using one
shared retrieval. A second model call (the judge, temperature 0) scores each answer from 1 to 5
for accuracy, clarity and relevance, using the question, the retrieved context and the reference
answer. Run it with `POST /api/evaluate` and read the averages from the saved results. Edit the
JSON file to change the questions.

## Configuration

Set in `.env`; nothing is hard-coded elsewhere.

| Variable | Default in code | Meaning |
| --- | --- | --- |
| `GROQ_API_KEY` | none | Required for answers and evaluation. Never commit it |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Model that writes answers (`.env.example` sets `openai/gpt-oss-120b`) |
| `JUDGE_MODEL` | same as `LLM_MODEL` | Model that scores evaluation answers |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Turns text into vectors |
| `CHUNK_SIZE` | 500 | Characters per chunk |
| `CHUNK_OVERLAP` | 100 | Characters repeated between neighbouring chunks |
| `TOP_K` | 3 | Chunks retrieved per question |
| `TEMPERATURE` | 0.1 | Low for focused, repeatable answers |
| `MAX_TOKENS` | 500 | Maximum answer length. Raise it if answers are cut off |
| `MAX_UPLOAD_MB` | 15 | Largest accepted upload |

## Project structure

```
app/
  config.py          all settings, read from .env
  loader.py          read PDF / TXT / Markdown
  chunker.py         split text into overlapping chunks
  embeddings.py      sentence-transformer wrapper
  vectorstore.py     ChromaDB: store, search, cosine distance
  keyword_search.py  BM25, used to contrast with semantic search
  ingestion.py       load -> chunk -> embed -> store
  prompts.py         the three prompt templates
  llm.py             Groq client wrapper
  rag.py             retrieve, then generate
  evaluation.py      questions x techniques, scored by an LLM judge
  main.py            FastAPI endpoints
static/              index.html, styles.css, app.js
data/
  documents/         documents indexed at startup
  few_shot_examples.json
  eval_questions.json
tests/               automated tests
```

## Tests

```bash
python -m pytest -q
```

The tests use a small fake embedder and fake LLM, so they run quickly with no downloads or API key.

## Limitations

- If the wrong chunks are retrieved, the answer will be weak. RAG reduces hallucination but does not eliminate it.
- Chunks are a fixed size in characters, so a topic can be split across chunks.
- The embedding model is small and mainly English.
- Scanned PDFs have no text layer and are rejected (they need OCR first).

## Troubleshooting

- "GROQ_API_KEY is missing": create `.env` from `.env.example`, add the key, restart the server.
- Rate-limit messages from Groq: the client retries automatically. For long evaluation runs a
  smaller model such as `llama-3.1-8b-instant` has higher limits.
- Truncated or empty answers: raise `MAX_TOKENS`.
- To rebuild the index from scratch, stop the server and delete `data/chroma/`.