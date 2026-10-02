"""
- Đầu mối DUY NHẤT khởi tạo embedding model cho cả ingest.py (nạp dữ liệu) và retriever.py (truy vấn), tránh lặp lại cấu hình và đảm bảo đồng nhất model.
- Model: "intfloat/multilingual-e5-large" chạy local qua FastEmbed (miễn phí, không tốn API call).
- Bọc qua FastEmbedEmbeddings chuẩn của LangChain để tương thích trực tiếp với VectorStore và Retriever.
"""
from __future__ import annotations
from functools import lru_cache
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.embeddings import Embeddings

EMBED_MODEL = "intfloat/multilingual-e5-large"

@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """Trả về instance embedding model dùng chung cho toàn bộ project.
    Dùng @lru_cache để nạp model ONNX 1 lần duy nhất vào bộ nhớ, tránh overhead đọc đĩa ở mỗi request.
    """
    return FastEmbedEmbeddings(model_name=EMBED_MODEL)
