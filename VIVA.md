# Presentation and Viva Guide

## 1. The project in thirty seconds

Marginalia lets you upload documents and ask questions about them. A plain LLM would answer from
memory and might invent facts. Instead, my system **retrieves** the most relevant passages from the
uploaded documents and gives only those to the LLM, which **generates** the answer. This is
Retrieval-Augmented Generation (RAG). The page shows which passages were used, so every answer can be checked.

## 2. Architecture

```
INGESTION (once per document)
  file -> loader -> chunker -> embedding model -> ChromaDB

QUESTION TIME
  question -> embedding model -> ChromaDB top-3 chunks -> prompt template -> Groq LLM -> answer + sources
```

| Stage | File | One-line explanation |
| --- | --- | --- |
| Load | `loader.py` | Reads PDF (page by page), TXT, Markdown into text |
| Chunk | `chunker.py` | Packs whole sentences into ~500-character pieces with 100 characters of overlap |
| Embed | `embeddings.py` | `all-MiniLM-L6-v2` turns each chunk into a 384-number vector, locally |
| Store | `vectorstore.py` | ChromaDB keeps text, vector and metadata; cosine metric |
| Retrieve | `vectorstore.py` | Embeds the question, returns the 3 nearest chunks with distance |
| Prompt | `prompts.py` | Three templates: zero-shot, few-shot, role-based |
| Generate | `llm.py`, `rag.py` | Sends the prompt to Groq (Llama 3.3 70B), temperature 0.1 |
| Evaluate | `evaluation.py` | 5 questions x 3 techniques, scored by an LLM judge |

## 3. Part 2: embeddings and semantic search (25 marks)

**Checklist against the assignment**

1. Load at least five documents: five files in `data/documents/` are indexed on startup; users can add more.
2. Divide into chunks: `chunker.py`.
3. Embeddings with a sentence-transformer: `embeddings.py`.
4. Store in ChromaDB: `vectorstore.py`.
5. Embed the question: `VectorStore.search` uses the same model as for the chunks (this is essential).
6. Top three chunks: `TOP_K=3` in `.env`.
7. Show scores: the Retrieval tab shows similarity and distance for every chunk.

**Why chunk?** One vector per document is a blurry average of every topic in it. Small chunks give each
vector one clear meaning and keep the prompt short. Overlap protects a sentence that falls on a boundary.
Trade-off: small chunks are precise but lose context; large chunks keep context but blur topics.

**What is an embedding?** A list of numbers (384 here) that represents meaning. A neural network is
trained so that sentences with similar meaning get vectors pointing in similar directions.

**Distance and similarity.** I normalise vectors and use cosine. `distance = 1 - similarity`. Distance 0
means identical direction; larger means less related. I checked this in testing: ChromaDB's distance
equals `1 - dot product` of the normalised vectors.

**Semantic search vs keyword search (required explanation)**

| | Keyword search (BM25) | Semantic search (embeddings) |
| --- | --- | --- |
| Matches | Exact words | Meaning |
| "car" vs "automobile" | Miss | Match |
| Needs shared vocabulary | Yes | No |
| Strong at | Codes, names, exact terms | Paraphrases, natural questions |
| Weak at | Synonyms, rewording | Rare identifiers |
| Explainable | Very (you see the word) | Less (a number) |

The Retrieval tab proves this live: it runs both methods on the same chunks, and highlights the exact
words keyword search matched. Try a query that uses different words than the document and keyword
search returns nothing while semantic search still finds the right chunk. Production systems often combine both (hybrid search).

## 4. Part 1: prompt engineering (25 marks)

All three techniques get the **same retrieved chunks and the same question**, so the only variable is the prompt.

| Technique | What the prompt contains | Expected strength | Expected weakness |
| --- | --- | --- | --- |
| Zero-shot | One instruction, context, question | Cheapest and simplest | Format and length vary; may add outside knowledge |
| Few-shot | Instruction plus 3 worked examples from `few_shot_examples.json` (including one "not in the documents" case) | Consistent format; learns to cite and to refuse | More tokens, so slower and costlier |
| Role-based | System message: "senior technical document analyst" plus rules (use only excerpts, cite [n], say when unknown) | Focused, grounded, cites sources | Needs well-written rules |

**How I compare.** `evaluation.py` runs the five questions in `data/eval_questions.json` through every
technique. A second LLM call (the "judge", temperature 0) scores each answer 1 to 5 for accuracy,
clarity and relevance using the question, the retrieved context and a reference answer. The Prompt lab
shows the averages, highlights the winner and gives an explanation generated from the actual numbers.

**Which is best and why?** Run the evaluation and quote **your own numbers**, because results depend on
the model and run. The reasoning to present: role-based and few-shot usually beat zero-shot on accuracy
and clarity because they give the model explicit constraints (answer only from the context, be concise,
cite) or a pattern to copy; zero-shot is the baseline and tends to be wordier. Role-based costs fewer
tokens than few-shot for similar quality, which is a good practical argument. If your run shows a
different winner, explain that instead. Honest reading of your own data scores higher than a memorised answer.

**Be ready to admit the limits of the evaluation:** five questions is a small sample, and an LLM judge
can be biased (for example towards longer answers). That is why the reference answers are included and
the judge temperature is 0.

## 5. Design decisions you can defend

- **Nothing hard-coded:** models, chunk size, top-k, temperature, paths are in `.env`; questions and
  few-shot examples are JSON files; model names and technique labels in the UI come from the API.
- **Local embeddings, hosted LLM:** embeddings are free and private; Groq gives fast free-tier inference.
- **ChromaDB over FAISS:** it stores text and metadata with the vectors and persists to disk, so there is less glue code.
- **Fair comparison:** retrieval runs once per question and is shared by all three techniques.
- **Safe re-upload:** chunk ids are `filename::index`, and a file's old chunks are deleted before re-adding, so no duplicates.
- **Robust inputs:** file type and size checks, filename sanitising (no path traversal), clear errors for
  scanned PDFs, empty questions and a missing API key; the UI inserts text with `textContent`, so no HTML injection.
- **Tested:** 22 automated tests covering chunking, keyword search, loading, prompts, every endpoint and error case.

## 6. Likely viva questions

**What is RAG and why use it?** It retrieves relevant text and gives it to the LLM, so answers are grounded
in documents, can be updated by changing files (no retraining), and can show sources. It reduces hallucination.

**Why not just paste the whole document into the prompt?** Context windows are limited, long prompts cost
more and are slower, and models handle irrelevant text poorly. Retrieval sends only what matters.

**Why must the question and the chunks use the same embedding model?** Vectors from different models live in
different spaces, so distances between them are meaningless.

**What does overlap do?** It repeats the last sentences of a chunk at the start of the next, so an answer on
a boundary is not split.

**What is cosine similarity?** The cosine of the angle between two vectors; 1 means same direction. Distance is 1 minus that.

**Why normalise the embeddings?** It makes every vector length 1, so cosine similarity equals the dot
product, and results do not depend on text length.

**How does ChromaDB search quickly?** It uses an HNSW graph index for approximate nearest-neighbour search, trading a tiny accuracy loss for large speed gains.

**ChromaDB vs FAISS?** FAISS is a fast similarity-search library without built-in text and metadata storage;
ChromaDB is a small database that stores them with the vectors.

**What is top-k and how would you choose it?** The number of chunks retrieved. Too small risks missing the
answer; too large adds noise and cost. Three is a sensible start; test on your questions.

**What is temperature and why 0.1?** It controls randomness. Factual question answering needs repeatable, grounded answers, so it is low.

**What if the answer is not in the documents?** The role-based and few-shot prompts instruct the model to say the documents do not contain it.

**How does the model cite sources?** Chunks are numbered [1], [2], [3] in the prompt and the model is told to cite those numbers; the UI shows the same passages.

**What are the weaknesses?** Retrieval can miss the right chunk (then the answer is wrong or refused); chunk
boundaries can hurt; scanned PDFs need OCR; only five evaluation questions; LLM-judge bias; semantic search can miss exact identifiers.

**How would you improve it?** Hybrid search (BM25 plus embeddings), a re-ranker, better chunking by headings,
OCR, conversation memory, and a larger evaluation set.

**Why did you add keyword search to a semantic project?** Only to demonstrate the difference the assignment asks me to explain.

## 7. Suggested live demo (about five minutes)

1. Show the Library: five documents, chunk counts, and the Pipeline panel (models and settings from `.env`).
2. **Ask** tab: ask a question, show the answer, open "Show the exact prompt", then show the passages and their similarity.
3. **Retrieval** tab: search a query that paraphrases a document. Point out the highlighted matches in keyword results and how semantic search still ranks the right chunk first. Explain similarity and distance.
4. **Prompt lab**: run "Compare on one question" and read the three answers side by side.
5. Run the five-question evaluation, read the table, name the winner and explain why using the generated analysis.
6. Upload a new file and ask about it to show that nothing is hard-coded.
7. Open `config.py`, `.env.example` and `prompts.py` to show where settings and templates live.

## 8. One-minute code walkthrough order

`config.py` (settings) then `ingestion.py` (the pipeline in 15 lines) then `chunker.py` then
`vectorstore.py` (`add_chunks`, `search`) then `prompts.py` (the three templates) then `rag.py`
(retrieve, then generate) then `evaluation.py` then `main.py` (endpoints).
