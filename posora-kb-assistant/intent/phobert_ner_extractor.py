"""PhoBERT NER adapter.
Unknown/new NER labels pass through automatically. Only core entities that
need application normalization have explicit normalizers.
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Any, Callable, Iterable
from intent.extractor_base import EntityExtractor, ExtractedEntities, clean_entities
from intent.ner_extractor import (
    NERExtractor,
    _NUMBER_WORDS_RAW,
    _SIZE_WORD_MAP,
    _classify_intent,
    _fold,
    _normalize,
)
from intent.phobert_runtime import get_phobert_runtime
from tools import find_product

_FOLDED_NUMBER_WORDS: dict[str, int] = {
    _fold(_normalize(word)): value for word, value in _NUMBER_WORDS_RAW.items()
}

def _map_values(value: Any, mapper: Callable[[Any], Any]) -> Any:
    if isinstance(value, list):
        mapped = [mapper(item) for item in value]
        mapped = [item for item in mapped if item is not None]
        if not mapped:
            return None
        return mapped[0] if len(mapped) == 1 else mapped
    return mapper(value)

def _quantity(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    digits = re.search(r"\d+", text)
    if digits:
        return int(digits.group(0))
    return _FOLDED_NUMBER_WORDS.get(_fold(_normalize(text)))

def _size(value: Any) -> str | None:
    if not value:
        return None
    folded = _fold(_normalize(str(value).strip()))
    if folded in _SIZE_WORD_MAP:
        return _SIZE_WORD_MAP[folded]
    bare = re.search(r"\b(m|l)\b", folded, re.IGNORECASE)
    return bare.group(1).upper() if bare else None

def _product(value: Any) -> str | None:
    if not value:
        return None
    text = str(value).strip()
    match = find_product(text)
    return match.product["name"] if match.product is not None else text

class PhoBERTNERExtractor(EntityExtractor):
    def __init__(self, candidate_dirs: Iterable[str | Path] | None = None) -> None:
        self._runtime = get_phobert_runtime(candidate_dirs)

    def extract(self, text: str) -> ExtractedEntities:
        intent = _classify_intent(_fold(_normalize(text)))
        raw = self._runtime.extract(text)

        # All labels pass through. Only entities that need canonicalization are
        # normalized here. Newly trained labels require no code change.
        entities = dict(raw)

        if "product" in entities:
            entities["product_name"] = _map_values(entities.pop("product"), _product)

        if "quantity" in entities:
            entities["quantity"] = _map_values(entities["quantity"], _quantity)

        if "size" in entities:
            entities["size"] = _map_values(entities["size"], _size)

        product_name = entities.get("product_name")
        size = entities.get("size")
        if isinstance(product_name, str) and isinstance(size, str):
            entities["size"] = NERExtractor._validate_size(product_name, size)

        return ExtractedEntities(intent=intent, entities=clean_entities(entities))
