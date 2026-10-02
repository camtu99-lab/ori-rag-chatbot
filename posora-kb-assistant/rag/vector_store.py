"""
Module quản trị và kết nối Vector Store tập trung:
1. Trừu tượng hóa kiến trúc (Plugin-friendly):
   - Đóng gói toàn bộ thao tác Vector DB qua interface `VectorStore` của LangChain.
   - Cho phép hoán đổi backend lưu trữ độc lập mà không làm thay đổi caller logic tại ingest.py, retriever.py hay rag.py.
2. Cấu hình Backend:
   - Sử dụng Redis Cloud (RediSearch) thông qua package chính thức `langchain-redis`.
3. Lớp tương thích ngược (`_ScoreAdjustedRedisVectorStore`):
   - Chuẩn hóa Score: Nghịch đảo Cosine Distance từ RediSearch thành Cosine Similarity (higher is better) để giữ nguyên hành vi lọc ngưỡng của Retriever cũ.
   - Chuẩn hóa Schema ID: Tự động cast `int` sang `str` cho tham số `ids`, đảm bảo tương thích 100% với luồng ingest hiện hành.
"""
from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.vectorstores import VectorStore
from langchain_redis import RedisConfig, RedisVectorStore

from .embeddings import get_embeddings

load_dotenv()  # đọc .env để lấy REDIS_URL, REDIS_INDEX_NAME

REDIS_URL = os.getenv("REDIS_URL")
REDIS_INDEX_NAME = os.getenv("REDIS_INDEX_NAME", "dmp_knowledge")

if not REDIS_URL:
    raise ValueError("Thiếu REDIS_URL trong file .env")

def _build_config() -> RedisConfig:
    """Tạo cấu hình RediSearch index đồng nhất cho cả luồng đọc và ghi, tránh xung đột schema.
    - Phép đo: Giữ nguyên `distance_metric="COSINE"` tương thích với thang điểm chuẩn của retriever.py.
    - Schema: Khai báo tường minh các trường metadata từ ingest.py (cho phép trường tùy chọn vắng mặt tùy theo loại document).
    """
    return RedisConfig(
        index_name=REDIS_INDEX_NAME,
        redis_url=REDIS_URL,
        distance_metric="COSINE",
        metadata_schema=[
            {"name": "source", "type": "tag"},
            {"name": "type", "type": "tag"},
            {"name": "name", "type": "text"},
            {"name": "category", "type": "tag"},
            {"name": "price_m", "type": "numeric"},
            {"name": "price_l", "type": "numeric"},
        ],
    )

class _ScoreAdjustedRedisVectorStore(RedisVectorStore):
    """Lớp bọc RedisVectorStore chuẩn hóa giá trị trả về nhằm tương thích ngược với ingest.py và retriever.py mà không thay đổi public interface."""

    def similarity_search_with_score(self, query: str, k: int = 4, **kwargs):
        """Chuyển đổi Cosine Distance của RediSearch (`1 - similarity`) về Raw Cosine Similarity (`1.0 - distance`).
        Đảm bảo quy ước điểm 'càng cao càng giống' tương thích với logic lọc `score >= SCORE_THRESHOLD` của retriever.py.
        """
        results = super().similarity_search_with_score(query, k=k, **kwargs)
        return [(doc, 1.0 - distance) for doc, distance in results]

    def add_documents(self, documents, **kwargs):
        """ingest.py gọi `add_documents(documents, ids=list(range(len(documents))))`
        - ids là số nguyên. Redis key bắt buộc là chuỗi, nên ép kiểu str()
        ở đây thay vì bắt ingest.py (file không được sửa) tự làm."""
        ids = kwargs.get("ids")
        if ids is not None:
            kwargs["ids"] = [str(i) for i in ids]
        return super().add_documents(documents, **kwargs)

def reset_collection() -> None:
    """Xoá sạch index Redis cũ (nếu có) - dùng trong ingest.py để đảm bảo
    dữ liệu cũ không còn sót lại mỗi lần chạy lại `python -m rag.ingest`"""
    store = _ScoreAdjustedRedisVectorStore(get_embeddings(), config=_build_config())
    if store.index.exists():
        store.index.delete(drop=True)

@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    store = _ScoreAdjustedRedisVectorStore(get_embeddings(), config=_build_config())
    return store