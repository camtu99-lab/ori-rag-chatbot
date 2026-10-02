from __future__ import annotations

import os
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from .ingest import (_chunks_to_documents, chunk_faq, chunk_menu_customization, chunk_menu_items, chunk_promotions)

try:
    from unidecode import unidecode
except ImportError:  # pragma: no cover
    def unidecode(text: str) -> str:
        text = text.replace("đ", "d").replace("Đ", "D")
        decomposed = unicodedata.normalize("NFD", text)
        return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")

_NON_ALNUM = re.compile(r"[^a-z0-9\s]")
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_STOPWORDS_PATH = _PROJECT_ROOT / "data" / "stopwords_vi.txt"

def _resolve_stopwords_path() -> Path:
    configured = os.getenv("RAG_STOPWORDS_PATH", "").strip()
    return (
        Path(configured).expanduser().resolve()
        if configured
        else _DEFAULT_STOPWORDS_PATH
    )

def _normalize_token(text: str) -> str:
    folded = unidecode(text.strip().lower())
    return _NON_ALNUM.sub(" ", folded).strip()

@lru_cache(maxsize=1)
def _load_stopwords() -> frozenset[str]:
    path = _resolve_stopwords_path()
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy stopwords file: {path}. " "Có thể override bằng RAG_STOPWORDS_PATH.")

    words: set[str] = set()
    with path.open(encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            normalized = _normalize_token(line)
            if normalized and " " not in normalized:
                words.add(normalized)
    return frozenset(words)

def reload_stopwords() -> None:
    _load_stopwords.cache_clear()

def _tokenize(text: str) -> list[str]:
    folded = unidecode(text.lower())
    cleaned = _NON_ALNUM.sub(" ", folded)
    stopwords = _load_stopwords()
    return [token for token in cleaned.split() if token not in stopwords]

@lru_cache(maxsize=1)
def _build_corpus() -> list[Document]:
    all_chunks = (
        chunk_menu_items()
        + chunk_menu_customization()
        + chunk_faq()
        + chunk_promotions()
    )
    return _chunks_to_documents(all_chunks)

@lru_cache(maxsize=1)
def _get_bm25_index() -> tuple[BM25Okapi, list[Document]]:
    documents = _build_corpus()
    tokenized_corpus = [_tokenize(doc.page_content) for doc in documents]
    return BM25Okapi(tokenized_corpus), documents

def reload_sparse_index() -> None:
    reload_stopwords()
    _build_corpus.cache_clear()
    _get_bm25_index.cache_clear()

def bm25_search(query: str, k: int) -> list[tuple[Document, float]]:
    index, documents = _get_bm25_index()
    scores = index.get_scores(_tokenize(query))
    ranked = sorted(((doc, float(score)) for doc, score in zip(documents, scores)), key=lambda pair: pair[1], reverse=True)
    return [(doc, score) for doc, score in ranked[:k] if score > 0.0]
