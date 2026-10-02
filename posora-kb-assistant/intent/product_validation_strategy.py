"""
Strategy Pattern cho việc kiểm tra product giữa LLM Intent Extractor và PhoBERT NER.
Không hard-code tên món. Validation dựa trên product catalog thông qua find_product() của tools.py.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from tools import find_product

@dataclass(frozen=True)
class ProductValidationResult:
    matched: bool
    llm_product: str | None
    ner_product: str | None
    canonical_product: str | None = None

class ProductValidationStrategy(ABC):
    """Interface cho chiến lược validation product."""

    @abstractmethod
    def validate(self, llm_product: str | None, ner_product: str | None) -> ProductValidationResult:
        raise NotImplementedError

class CatalogProductMatchStrategy(ProductValidationStrategy):
    """So sánh 2 entity bằng cách quy về tên chuẩn (canonical name) qua catalog menu (không hardcode).
    Ví dụ: "ly bạc xỉu" và "Bạc Xỉu" đều resolve về "Bạc Xỉu" -> MATCH.
    """
    def validate(self, llm_product: str | None, ner_product: str | None) -> ProductValidationResult:
        if not llm_product:
            return ProductValidationResult(matched=True, llm_product=llm_product, ner_product=ner_product)

        if not ner_product:
            return ProductValidationResult(matched=True, llm_product=llm_product, ner_product=ner_product)

        llm_match = find_product(llm_product)
        ner_match = find_product(ner_product)

        llm_resolved = llm_match.product
        ner_resolved = ner_match.product

        if llm_resolved is None or ner_resolved is None:
            return ProductValidationResult(matched=True, llm_product=llm_product, ner_product=ner_product)

        llm_canonical = llm_resolved["name"]
        ner_canonical = ner_resolved["name"]

        matched = llm_canonical == ner_canonical

        return ProductValidationResult(matched=matched, llm_product=llm_product, ner_product=ner_product, canonical_product=ner_canonical if matched else None)

def get_product_validation_strategy() -> ProductValidationStrategy:
    """Factory trả về Strategy mặc định."""
    return CatalogProductMatchStrategy()
