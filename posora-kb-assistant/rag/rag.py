"""
Module truy vấn RAG.
Public API giữ nguyên: retrieve(), retrieve_context()
Pipeline bên dưới: Dense + BM25 -> RRF Fusion -> Cross-Encoder Reranker -> Top-K
"""

from __future__ import annotations

from langchain_core.documents import Document

from .hybrid_retriever import get_hybrid_retriever
from .retriever import SCORE_THRESHOLD, TOP_K

__all__ = [
    "retrieve",
    "retrieve_context",
    "TOP_K",
    "SCORE_THRESHOLD",
]

def retrieve(query: str, top_k: int = TOP_K) -> dict:
    """Retrieve context và trả cả documents thô sau reranking."""
    documents: list[Document] = (get_hybrid_retriever(top_k=top_k).invoke(query))
    context = "\n".join(f"- {doc.page_content}" for doc in documents)
    return {"context": context, "results": documents}

def retrieve_context(query: str, top_k: int = TOP_K) -> str:
    """Backward-compatible API cho chatbot.py."""
    return retrieve(query, top_k)["context"]

if __name__ == "__main__":
    import sys
    query = (" ".join(sys.argv[1:]) or "Quán có món cà phê muối không, giá bao nhiêu?")
    print(f"Query: {query}\n")
    context = retrieve_context(query)
    print("Context tìm được:\n" + (context or "(không có gì đủ liên quan)"))
