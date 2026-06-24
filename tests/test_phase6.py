"""
Phase 6 tests: RAG core (chunking, tokenize, BM25 RagStore, text ingest).
No network, no PDF dependency required (uses .txt/.md and direct text).

Run:
    python tests/test_phase6.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_parametizer.core import rag
from llm_parametizer.core.rag import RagStore, chunk_text, tokenize


def test_tokenize():
    assert tokenize("Hello, World! 123") == ["hello", "world", "123"]
    assert tokenize("") == []


def test_chunk_text_basic():
    text = "word " * 500  # ~2500 chars
    chunks = chunk_text(text, source="doc", chunk_size=800, overlap=150)
    assert len(chunks) >= 3
    assert all(c.doc == "doc" for c in chunks)
    assert all(c.text for c in chunks)


def test_chunk_text_empty():
    assert chunk_text("", source="x") == []


def test_chunk_overlap_guard():
    # overlap >= chunk_size should not hang or error
    chunks = chunk_text("a b c d e f g h", source="d", chunk_size=4, overlap=10)
    assert isinstance(chunks, list)


def test_chunk_indices_unique():
    chunks = chunk_text("alpha " * 300, source="doc", chunk_size=400, overlap=80)
    idxs = [c.index for c in chunks]
    assert idxs == list(range(len(chunks)))


def test_store_add_and_has_docs():
    s = RagStore()
    assert not s.has_docs
    n = s.add_text("notes", "The refund policy allows returns within 30 days of purchase.")
    assert n >= 1
    assert s.has_docs
    assert s.docs["notes"] == n


def test_store_retrieve_ranks_relevant_chunk():
    s = RagStore()
    s.add_text("policy", "The refund policy allows returns within thirty days.")
    s.add_text("recipe", "To bake bread you need flour, water, yeast and salt.")
    s.build()
    hits = s.retrieve("how do refunds and returns work", k=1)
    assert hits
    top_chunk, score = hits[0]
    assert "refund" in top_chunk.text.lower()
    # BM25 IDF can be 0 on a 2-doc corpus; relevance is gated on overlap, so the
    # score may legitimately be 0. Just confirm it's a real number.
    assert isinstance(score, float)


def test_store_retrieve_no_match_returns_empty():
    s = RagStore()
    s.add_text("doc", "apples bananas cherries")
    s.build()
    assert s.retrieve("quantum chromodynamics tensor", k=3) == []


def test_store_retrieve_empty_query():
    s = RagStore()
    s.add_text("doc", "some content here")
    assert s.retrieve("", k=3) == []


def test_store_remove_and_clear():
    s = RagStore()
    s.add_text("a", "alpha content one two three")
    s.add_text("b", "beta content four five six")
    assert s.remove_document("a")
    assert "a" not in s.docs
    assert not s.remove_document("missing")
    s.clear()
    assert not s.has_docs and s.docs == {}


def test_store_rebuilds_after_mutation():
    s = RagStore()
    s.add_text("doc", "first topic about astronomy and stars")
    s.build()
    s.add_text("doc2", "second topic about marine biology and fish")
    # No explicit build(): retrieve() must rebuild because store is dirty.
    hits = s.retrieve("fish and marine life", k=1)
    assert hits and "marine" in hits[0][0].text.lower()


def test_context_for_formats_block():
    s = RagStore()
    s.add_text("guide", "The capital of France is Paris, a major European city.")
    s.build()
    context, chunks = s.context_for("what is the capital of France", k=1)
    assert chunks
    assert "[guide #1]" in context
    assert "Paris" in context


def test_load_document_txt(tmp=None):
    d = Path(tempfile.mkdtemp())
    f = d / "note.txt"
    f.write_text("hello from a text file", encoding="utf-8")
    assert "hello" in rag.load_document(f).lower()
    assert rag.load_document(d / "missing.txt") == ""


def test_load_document_unsupported():
    d = Path(tempfile.mkdtemp())
    f = d / "data.bin"
    f.write_bytes(b"\x00\x01\x02")
    assert rag.load_document(f) == ""


def test_add_file_txt():
    d = Path(tempfile.mkdtemp())
    f = d / "doc.md"
    f.write_text("# Title\n\nThis document describes the onboarding process in detail.",
                 encoding="utf-8")
    s = RagStore()
    n = s.add_file(f)
    assert n >= 1
    assert "doc.md" in s.docs


def _run_all():
    funcs = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for fn in funcs:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL  {fn.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)
