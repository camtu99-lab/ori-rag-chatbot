from __future__ import annotations

import os
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from .retriever import SCORE_THRESHOLD, TOP_K
from .reranker import get_reranker
from .sparse_retriever import bm25_search
from .vector_store import get_vector_store


DENSE_POOL_K = int(os.getenv("RAG_DENSE_POOL_K", "100"))
SPARSE_POOL_K = int(os.getenv("RAG_SPARSE_POOL_K", "100"))
STRICT_SCORE_THRESHOLD = float(os.getenv("RAG_STRICT_SCORE_THRESHOLD", "0.55"))
MIN_BM25_SCORE_FOR_SUPPORT = float(
    os.getenv("RAG_MIN_BM25_SCORE_FOR_SUPPORT", "1.0")
)
RRF_K = int(os.getenv("RAG_RRF_K", "60"))
RERANK_CANDIDATE_K = int(os.getenv("RAG_RERANK_CANDIDATE_K", "10"))


def _doc_key(doc: Document) -> tuple[str, str]:
    return str(doc.metadata.get("source", "")), doc.page_content


@dataclass(frozen=True, slots=True)
class RetrievalProfile:
    query: str
    documents: list[Document]
    reranked_with_scores: list[tuple[Document, float]]
    dense_ms: float
    sparse_ms: float
    fusion_ms: float
    rerank_ms: float
    total_ms: float
    dense_raw_count: int
    dense_filtered_count: int
    sparse_count: int
    fused_candidate_count: int
    rerank_candidate_count: int
    has_sparse_support: bool
    effective_dense_threshold: float
    best_rerank_score: float | None
    best_dense_score: float | None
    best_bm25_score: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "dense_ms": self.dense_ms,
            "sparse_ms": self.sparse_ms,
            "fusion_ms": self.fusion_ms,
            "rerank_ms": self.rerank_ms,
            "total_ms": self.total_ms,
            "dense_raw_count": self.dense_raw_count,
            "dense_filtered_count": self.dense_filtered_count,
            "sparse_count": self.sparse_count,
            "fused_candidate_count": self.fused_candidate_count,
            "rerank_candidate_count": self.rerank_candidate_count,
            "returned_count": len(self.documents),
            "has_sparse_support": self.has_sparse_support,
            "effective_dense_threshold": self.effective_dense_threshold,
            "best_rerank_score": self.best_rerank_score,
            "best_dense_score": self.best_dense_score,
            "best_bm25_score": self.best_bm25_score,
        }


class HybridRetriever(BaseRetriever):
    k: int = TOP_K
    score_threshold: float = SCORE_THRESHOLD
    strict_score_threshold: float = STRICT_SCORE_THRESHOLD
    dense_pool_k: int = DENSE_POOL_K
    sparse_pool_k: int = SPARSE_POOL_K
    rerank_candidate_k: int = RERANK_CANDIDATE_K

    def _run_pipeline(self, query: str) -> RetrievalProfile:
        total_start = time.perf_counter()
        vector_store = get_vector_store()

        start = time.perf_counter()
        dense_hits = vector_store.similarity_search_with_score(
            query,
            k=self.dense_pool_k,
        )
        dense_ms = (time.perf_counter() - start) * 1000
        best_dense_score = float(dense_hits[0][1]) if dense_hits else None

        start = time.perf_counter()
        sparse_hits = bm25_search(query, k=self.sparse_pool_k)
        sparse_ms = (time.perf_counter() - start) * 1000
        best_bm25_score = float(sparse_hits[0][1]) if sparse_hits else None

        start = time.perf_counter()

        has_sparse_support = (
            best_bm25_score is not None
            and best_bm25_score >= MIN_BM25_SCORE_FOR_SUPPORT
        )
        effective_threshold = (
            self.score_threshold
            if has_sparse_support
            else self.strict_score_threshold
        )

        dense_filtered = [
            (doc, score)
            for doc, score in dense_hits
            if score >= effective_threshold
        ]

        fused_scores: dict[tuple[str, str], float] = {}
        doc_by_key: dict[tuple[str, str], Document] = {}

        for rank, (doc, _score) in enumerate(dense_filtered, start=1):
            key = _doc_key(doc)
            fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            doc_by_key[key] = doc

        for rank, (doc, _score) in enumerate(sparse_hits, start=1):
            key = _doc_key(doc)
            fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            doc_by_key[key] = doc

        ranked_keys = sorted(
            fused_scores,
            key=fused_scores.get,
            reverse=True,
        )
        all_candidates = [doc_by_key[key] for key in ranked_keys]

        candidates = (
            all_candidates[: self.rerank_candidate_k]
            if self.rerank_candidate_k > 0
            else all_candidates
        )
        fusion_ms = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        scored = (
            get_reranker().rerank_with_scores(query, candidates, top_k=self.k)
            if candidates
            else []
        )
        rerank_ms = (time.perf_counter() - start) * 1000

        documents = [doc for doc, _score in scored]
        best_rerank_score = float(scored[0][1]) if scored else None

        return RetrievalProfile(
            query=query,
            documents=documents,
            reranked_with_scores=scored,
            dense_ms=dense_ms,
            sparse_ms=sparse_ms,
            fusion_ms=fusion_ms,
            rerank_ms=rerank_ms,
            total_ms=(time.perf_counter() - total_start) * 1000,
            dense_raw_count=len(dense_hits),
            dense_filtered_count=len(dense_filtered),
            sparse_count=len(sparse_hits),
            fused_candidate_count=len(all_candidates),
            rerank_candidate_count=len(candidates),
            has_sparse_support=has_sparse_support,
            effective_dense_threshold=effective_threshold,
            best_rerank_score=best_rerank_score,
            best_dense_score=best_dense_score,
            best_bm25_score=best_bm25_score,
        )

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Any = None,
    ) -> list[Document]:
        return self._run_pipeline(query).documents

    def profile(self, query: str) -> RetrievalProfile:
        return self._run_pipeline(query)


@lru_cache(maxsize=32)
def get_hybrid_retriever(
    top_k: int = TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    rerank_candidate_k: int = RERANK_CANDIDATE_K,
) -> BaseRetriever:
    return HybridRetriever(
        k=top_k,
        score_threshold=score_threshold,
        strict_score_threshold=STRICT_SCORE_THRESHOLD,
        dense_pool_k=DENSE_POOL_K,
        sparse_pool_k=SPARSE_POOL_K,
        rerank_candidate_k=rerank_candidate_k,
    )
