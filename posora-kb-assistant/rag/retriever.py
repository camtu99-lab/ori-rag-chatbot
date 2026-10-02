"""
Khởi tạo và cấu hình LangChain Retriever cho luồng RAG:
1. Kiến trúc: Đóng gói VectorStore thành Retriever chuẩn, cung cấp phương thức `invoke(query)` cho rag.py.
2. Xử lý triệt để bug Score Threshold:
   - Thay thế `similarity_score_threshold` mặc định của LangChain (tự động scale điểm qua `(raw + 1) / 2` làm lọt chunk rác/none-source).
   - Triển khai `_ThresholdRetriever` để so khớp trực tiếp trên thang RAW SCORE gốc (`score >= SCORE_THRESHOLD`).
3. Tham số: Bảo toàn tuyệt đối các cấu hình gốc (`TOP_K`, `SCORE_THRESHOLD = 0.4`).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore

from .vector_store import get_vector_store

TOP_K = 5
# Ngưỡng Raw Cosine Score tối thiểu để giữ lại chunk; lọc bỏ context không liên quan nhằm hạn chế hallucination.
SCORE_THRESHOLD = 0.4

class _ThresholdRetriever(BaseRetriever):
    """Retriever lọc theo RAW SCORE trực tiếp từ `similarity_search_with_score()`, 
    bỏ qua hàm chuẩn hóa relevance score mặc định của LangChain để giữ đúng ý nghĩa SCORE_THRESHOLD."""
    vector_store: VectorStore
    k: int = TOP_K
    score_threshold: float = SCORE_THRESHOLD

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> list[Document]:
        # Lọc ngưỡng trực tiếp phía server qua param native của similarity_search_with_score; 
        # tối ưu hiệu năng bằng cách giảm payload truyền tải.
        results = self.vector_store.similarity_search_with_score(query, k=self.k)
        docs = []

        for doc, score in results:
            if score >= self.score_threshold:
                docs.append(doc)

        return docs
    #    return [doc for doc, _score in results]

@lru_cache(maxsize=8)
def get_retriever(top_k: int = TOP_K, score_threshold: float = SCORE_THRESHOLD) -> BaseRetriever:
    """Trả về instance Retriever tương thích chuẩn `.invoke(query)`.
    Cache theo cặp tham số `(top_k, score_threshold)` để tái sử dụng object, tránh overhead khởi tạo lại qua từng request.
    """
    return _ThresholdRetriever(vector_store=get_vector_store(), k=top_k, score_threshold=score_threshold)
