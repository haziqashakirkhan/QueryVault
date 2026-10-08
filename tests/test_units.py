import pytest
from pypdf import PdfWriter

from app.chunker import split_text
from app.keyword_search import keyword_search, tokenize
from app.loader import DocumentLoadError, load_document
from app.prompts import TECHNIQUES, build_messages

CHUNKS = [{"text": "alpha beta gamma", "source": "a.txt", "page": None, "chunk_index": 0}]


def test_short_text_is_one_chunk():
    assert split_text("One short sentence.", 500, 50) == ["One short sentence."]


def test_empty_text_gives_no_chunks():
    assert split_text("   \n\n ", 500, 50) == []


def test_chunks_respect_size_and_overlap():
    text = " ".join(f"Sentence number {i} is here." for i in range(60))
    chunks = split_text(text, 200, 60)
    assert len(chunks) > 3
    assert all(len(c) <= 200 + 60 for c in chunks)
    # overlap: the end of one chunk reappears at the start of the next
    assert chunks[0].split(". ")[-1][:12] in chunks[1]


def test_very_long_sentence_is_hard_split():
    chunks = split_text("x" * 1000, 300, 50)
    assert len(chunks) >= 4 and all(len(c) <= 300 for c in chunks)


def test_invalid_overlap_rejected():
    with pytest.raises(ValueError):
        split_text("text", 100, 100)


def test_tokenize_drops_stopwords():
    assert tokenize("What is the Embedding?") == ["embedding"]


def test_keyword_search_finds_exact_word_and_ignores_synonyms():
    docs = CHUNKS + [{"text": "automobile engine", "source": "b.txt", "page": None, "chunk_index": 0}]
    hits = keyword_search("alpha", docs, 3)
    assert [h["source"] for h in hits] == ["a.txt"] and hits[0]["matched_terms"] == ["alpha"]
    assert keyword_search("car", docs, 3) == []  # no shared word -> no match


def test_loader_rejects_unsupported_and_empty(tmp_path):
    bad = tmp_path / "x.docx"
    bad.write_text("hi")
    with pytest.raises(DocumentLoadError):
        load_document(bad)
    empty = tmp_path / "e.txt"
    empty.write_text("   ")
    with pytest.raises(DocumentLoadError):
        load_document(empty)


def test_blank_pdf_reports_no_text(tmp_path):
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    path = tmp_path / "blank.pdf"
    with open(path, "wb") as fh:
        writer.write(fh)
    with pytest.raises(DocumentLoadError):
        load_document(path)


def test_all_three_templates_build():
    examples = [{"context": "c", "question": "q", "answer": "a"}]
    for name in TECHNIQUES:
        messages = build_messages(name, "What?", CHUNKS, examples)
        assert messages[-1]["role"] == "user" and "What?" in messages[-1]["content"]
    assert build_messages("role_based", "q", CHUNKS, examples)[0]["role"] == "system"
    assert "Example 1" in build_messages("few_shot", "q", CHUNKS, examples)[0]["content"]
    assert "Example" not in build_messages("zero_shot", "q", CHUNKS, examples)[0]["content"]
