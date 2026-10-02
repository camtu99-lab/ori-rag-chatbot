"""
Factory (Strategy Pattern): Đầu mối duy nhất ánh xạ tên chiến lược (llm/ner/phobert) sang class tương ứng; 
các module khác chỉ cần gọi get_extractor(name) mà không cần import trực tiếp.
Cache Instance (@lru_cache): Khởi tạo mỗi extractor đúng 1 lần và tái sử dụng cho các lần gọi sau để tránh lãng phí tài nguyên.
"""
from __future__ import annotations
from functools import lru_cache
from intent.extractor_base import EntityExtractor
from intent.llm_extractor import LLMExtractor
from intent.ner_extractor import NERExtractor
from intent.phobert_ner_extractor import PhoBERTNERExtractor

_EXTRACTORS: dict[str, type[EntityExtractor]] = {
    "llm": LLMExtractor,
    "ner": NERExtractor,
    "phobert": PhoBERTNERExtractor,
}

@lru_cache(maxsize=None)
def get_extractor(name: str) -> EntityExtractor:
    """Trả về 1 instance EntityExtractor theo tên chiến lược.
    Args:
        name: "llm", "ner" hoặc "phobert" (không phân biệt hoa/thường).
    Returns:
        Instance EntityExtractor tương ứng (cache lại cho các lần gọi sau với CÙNG tên).
    Raises:
        ValueError: nếu `name` không khớp chiến lược nào đã đăng ký -liệt kê rõ các tên hợp lệ trong message lỗi, để lỗi gõ sai tên
            (vd. "llmm") bị phát hiện ngay, không âm thầm dùng nhầm extractor khác.
    """
    key = name.strip().lower()
    extractor_cls = _EXTRACTORS.get(key)
    if extractor_cls is None:
        valid = ", ".join(sorted(_EXTRACTORS))
        raise ValueError(f"Không có extractor '{name}'. Các lựa chọn hợp lệ: {valid}")
    return extractor_cls()
