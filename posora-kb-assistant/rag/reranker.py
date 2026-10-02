from __future__ import annotations

import os
from functools import lru_cache
from typing import Sequence

from langchain_core.documents import Document


RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANKER_MAX_LENGTH = int(os.getenv("RERANKER_MAX_LENGTH", "512"))
RERANKER_BATCH_SIZE = int(os.getenv("RERANKER_BATCH_SIZE", "16"))


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = RERANKER_MODEL,
        max_length: int = RERANKER_MAX_LENGTH,
        batch_size: int = RERANKER_BATCH_SIZE,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                import torch
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RuntimeError(
                    "Thiếu dependency Cross-Encoder. "
                    "Cài bằng: pip install sentence-transformers"
                ) from exc

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device=device,
            )

        return self._model

    def rerank_with_scores(
        self,
        query: str,
        documents: Sequence[Document],
        top_k: int,
    ) -> list[tuple[Document, float]]:
        docs = list(documents)
        if not docs or top_k <= 0:
            return []

        model = self._load_model()
        pairs = [(query, doc.page_content) for doc in docs]
        scores = model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        ranked = sorted(
            (
                (doc, float(score))
                for doc, score in zip(docs, scores)
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        return ranked[:top_k]

    def rerank(
        self,
        query: str,
        documents: Sequence[Document],
        top_k: int,
    ) -> list[Document]:
        return [
            doc
            for doc, _score in self.rerank_with_scores(
                query,
                documents,
                top_k,
            )
        ]


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker()
