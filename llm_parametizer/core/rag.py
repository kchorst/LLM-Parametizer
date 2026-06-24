"""
Retrieval-Augmented Generation core (BM25-only, fully offline).

Pure logic + light IO:
  - chunk_text()          split documents into overlapping character windows
  - tokenize()            simple word tokenizer for BM25
  - extract_pdf()         text from a PDF (pypdf, pdfplumber fallback) with caching
  - RagStore              holds chunks, builds a BM25 index, retrieves top-k

Semantic embeddings are intentionally out of scope here (planned Phase 6.5).
The retrieve() interface is ranker-agnostic so a second ranker can slot in later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from rank_bm25 import BM25Okapi

from . import config


# ── Tokenizing / chunking ─────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Common English stopwords removed before indexing/matching. Dropping these
# prevents spurious overlap (e.g. "and", "the") from skewing retrieval on small
# corpora and improves BM25 ranking quality.
_STOPWORDS = frozenset("""
a an and are as at be been but by do does did for from had has have how i if in
into is it its of on or that the their them then there these this to was were
what when where which who why will with you your yours we our us he she they
""".split())


def _stem(token: str) -> str:
    """Very light plural stemmer so 'refunds' matches 'refund', 'policies'→'policy'.

    Keyword retrieval has no morphology; this small normalization meaningfully
    improves recall without the weight of a full stemmer. Applied to both the
    index and the query so matching stays consistent.
    """
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    toks = _TOKEN_RE.findall((text or "").lower())
    if drop_stopwords:
        toks = [t for t in toks if t not in _STOPWORDS]
    return [_stem(t) for t in toks]


@dataclass
class Chunk:
    doc: str          # source document name
    index: int        # chunk ordinal within the document
    text: str

    @property
    def label(self) -> str:
        return f"{self.doc} #{self.index + 1}"


def chunk_text(text: str, source: str = "", chunk_size: int = 800,
               overlap: int = 150) -> list[Chunk]:
    """
    Split text into ~chunk_size-character windows that overlap by `overlap`
    characters, breaking on whitespace where possible to avoid mid-word cuts.
    """
    text = (text or "").strip()
    if not text:
        return []
    if overlap >= chunk_size:
        overlap = chunk_size // 4

    chunks: list[Chunk] = []
    start = 0
    n = len(text)
    idx = 0
    while start < n:
        end = min(start + chunk_size, n)
        # Prefer to end on a whitespace boundary (look back a little).
        if end < n:
            window = text.rfind(" ", start + chunk_size - overlap, end)
            if window != -1 and window > start:
                end = window
        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(doc=source, index=idx, text=piece))
            idx += 1
        if end >= n:
            break
        start = max(0, end - overlap)
    return chunks


# ── PDF extraction (with disk cache) ───────────────────────────────────────────

def _cache():
    try:
        import diskcache
        return diskcache.Cache(str(config.APP_DIR / "cache"))
    except Exception:
        return None


def extract_pdf(path: str | Path) -> str:
    """
    Extract text from a PDF. Tries pypdf first, falls back to pdfplumber.
    Results are cached by (path, mtime, size) so re-adding a file is instant.
    Returns "" on failure.
    """
    p = Path(path)
    try:
        stat = p.stat()
        key = f"pdf::{p.resolve()}::{int(stat.st_mtime)}::{stat.st_size}"
    except Exception:
        key = None

    cache = _cache()
    if cache is not None and key is not None:
        try:
            hit = cache.get(key)
            if isinstance(hit, str):
                return hit
        except Exception:
            pass

    text = _extract_pypdf(p)
    if not text.strip():
        text = _extract_pdfplumber(p)

    if cache is not None and key is not None and text.strip():
        try:
            cache.set(key, text)
        except Exception:
            pass
    return text


def _extract_pypdf(p: Path) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(p))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception:
        return ""


def _extract_pdfplumber(p: Path) -> str:
    try:
        import pdfplumber
        out = []
        with pdfplumber.open(str(p)) as pdf:
            for page in pdf.pages:
                out.append(page.extract_text() or "")
        return "\n".join(out)
    except Exception:
        return ""


def load_document(path: str | Path) -> str:
    """Extract text from a supported file (.pdf, .txt, .md). "" on failure."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(p)
    if suffix in (".txt", ".md", ".markdown", ".text"):
        try:
            return p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""
    return ""


# ── Store ──────────────────────────────────────────────────────────────────────

@dataclass
class RagStore:
    """
    Holds document chunks and a BM25 index. UI-free and thread-safe enough for
    the simple build-then-query pattern used by the chat path.
    """
    chunks: list[Chunk] = field(default_factory=list)
    docs: dict[str, int] = field(default_factory=dict)   # name → chunk count
    enabled: bool = False
    top_k: int = 4
    chunk_size: int = 800
    overlap: int = 150

    _bm25: BM25Okapi | None = field(default=None, repr=False)
    _dirty: bool = True

    # ── Mutation ────────────────────────────────────────────────────────────────

    def add_text(self, name: str, text: str) -> int:
        """Chunk text under a document name. Returns chunks added."""
        new = chunk_text(text, source=name, chunk_size=self.chunk_size,
                         overlap=self.overlap)
        if not new:
            return 0
        # Re-base chunk indices to be unique per doc even if added incrementally.
        base = self.docs.get(name, 0)
        for i, c in enumerate(new):
            c.index = base + i
        self.chunks.extend(new)
        self.docs[name] = base + len(new)
        self._dirty = True
        return len(new)

    def add_file(self, path: str | Path) -> int:
        text = load_document(path)
        if not text.strip():
            return 0
        return self.add_text(Path(path).name, text)

    def remove_document(self, name: str) -> bool:
        if name not in self.docs:
            return False
        self.chunks = [c for c in self.chunks if c.doc != name]
        del self.docs[name]
        self._dirty = True
        return True

    def clear(self) -> None:
        self.chunks.clear()
        self.docs.clear()
        self._bm25 = None
        self._dirty = True

    # ── Index / query ────────────────────────────────────────────────────────────

    @property
    def has_docs(self) -> bool:
        return bool(self.chunks)

    def build(self) -> None:
        """(Re)build the BM25 index from current chunks."""
        if not self.chunks:
            self._bm25 = None
            self._dirty = False
            return
        corpus = [tokenize(c.text) for c in self.chunks]
        self._bm25 = BM25Okapi(corpus)
        self._dirty = False

    def retrieve(self, query: str, k: int | None = None) -> list[tuple[Chunk, float]]:
        """
        Return up to k (chunk, score) pairs, best first; empty if nothing matches.

        Relevance is gated on lexical overlap between the query and the chunk
        rather than a strictly positive BM25 score. On a small corpus BM25's IDF
        can legitimately be 0 for a term present in one of few documents, which
        would otherwise discard a clearly relevant chunk.
        """
        if self._dirty or self._bm25 is None:
            self.build()
        if self._bm25 is None:
            return []
        q = tokenize(query)
        if not q:
            return []
        qset = set(q)
        scores = self._bm25.get_scores(q)
        k = k or self.top_k
        ranked = sorted(zip(self.chunks, scores), key=lambda cs: cs[1], reverse=True)
        out = [(c, float(s)) for c, s in ranked if qset & set(tokenize(c.text))]
        return out[:k]

    def format_context(self, chunks: list[Chunk]) -> str:
        """Render retrieved chunks as a context block for a system prompt."""
        if not chunks:
            return ""
        parts = ["Use the following document excerpts to answer the question. "
                 "If the answer is not contained in them, say so.\n"]
        for c in chunks:
            parts.append(f"[{c.label}]\n{c.text}\n")
        return "\n".join(parts).strip()

    def context_for(self, query: str, k: int | None = None) -> tuple[str, list[Chunk]]:
        """Convenience: retrieve then format. Returns (context_text, chunks)."""
        hits = self.retrieve(query, k)
        chunks = [c for c, _ in hits]
        return self.format_context(chunks), chunks
