import io


def test_startup_indexes_all_five_documents(client):
    docs = client.get("/api/documents").json()
    assert len(docs) == 5 and all(d["chunks"] > 0 for d in docs)


def test_search_returns_top3_with_scores(client):
    body = client.post("/api/search", json={"query": "How does ChromaDB measure distance?"}).json()
    assert len(body["semantic"]) == 3
    for hit in body["semantic"]:
        assert 0 <= hit["distance"] <= 2 and abs(hit["similarity"] + hit["distance"] - 1) < 1e-3
    distances = [h["distance"] for h in body["semantic"]]
    assert distances == sorted(distances)  # nearest first


def test_empty_query_rejected(client):
    assert client.post("/api/search", json={"query": "   "}).status_code == 400


def test_ask_all_techniques(client):
    for technique in ("zero_shot", "few_shot", "role_based"):
        r = client.post("/api/ask", json={"question": "What is RAG?", "technique": technique})
        assert r.status_code == 200
        data = r.json()
        assert data["answer"] and len(data["chunks"]) == 3 and "What is RAG?" in data["prompt"]


def test_unknown_technique_rejected(client):
    assert client.post("/api/ask", json={"question": "x", "technique": "magic"}).status_code == 422


def test_compare_shares_one_retrieval(client):
    data = client.post("/api/compare", json={"question": "What is RAG?"}).json()
    assert set(data["results"]) == {"zero_shot", "few_shot", "role_based"} and len(data["chunks"]) == 3


def test_upload_search_delete_roundtrip(client):
    files = [("files", ("notes.txt", io.BytesIO(b"Quokkas are small marsupials from Western Australia."), "text/plain"))]
    assert client.post("/api/documents", files=files).json()["saved"][0]["name"] == "notes.txt"
    top = client.post("/api/search", json={"query": "quokkas marsupials"}).json()["semantic"][0]
    assert top["source"] == "notes.txt"
    assert client.delete("/api/documents/notes.txt").status_code == 200
    assert "notes.txt" not in [d["name"] for d in client.get("/api/documents").json()]
    assert client.delete("/api/documents/notes.txt").status_code == 404


def test_upload_rejects_bad_type_and_path_traversal(client):
    files = [("files", ("../evil.exe", io.BytesIO(b"x"), "application/octet-stream"))]
    result = client.post("/api/documents", files=files).json()
    assert result["saved"] == [] and result["errors"]


def test_reupload_replaces_instead_of_duplicating(client):
    before = {d["name"]: d["chunks"] for d in client.get("/api/documents").json()}
    data = open("data/documents/rag_overview.md", "rb").read()
    client.post("/api/documents", files=[("files", ("rag_overview.md", io.BytesIO(data), "text/markdown"))])
    after = {d["name"]: d["chunks"] for d in client.get("/api/documents").json()}
    assert before == after


def test_evaluation_scores_and_picks_winner(client):
    report = client.post("/api/evaluate").json()
    assert len(report["questions"]) == 5 and report["winner"] in report["summary"]
    assert report["summary"]["role_based"]["accuracy"] == 4.0
    assert client.get("/api/evaluate/latest").json()["winner"] == report["winner"]


def test_missing_api_key_gives_clear_error(tmp_settings):
    from fastapi.testclient import TestClient
    from app.main import create_app
    from tests.conftest import FakeEmbedder
    with TestClient(create_app(tmp_settings.__class__(**{**tmp_settings.__dict__, "groq_api_key": ""}),
                               embedder=FakeEmbedder())) as c:
        r = c.post("/api/ask", json={"question": "What is RAG?", "technique": "zero_shot"})
        assert r.status_code == 503 and "GROQ_API_KEY" in r.json()["detail"]
        assert c.post("/api/search", json={"query": "rag"}).status_code == 200  # retrieval works without a key


def test_index_page_served(client):
    assert client.get("/").status_code == 200
