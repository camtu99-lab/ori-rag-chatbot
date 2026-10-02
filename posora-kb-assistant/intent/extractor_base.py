"""Common extraction contract.
The application consumes a generic entity mapping instead of a fixed list of
NER slots. This lets a trained model add labels without forcing chatbot.py to
change for every new entity.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, TypedDict

EntityMap = dict[str, Any]

class ExtractedEntities(TypedDict):
    intent: str
    entities: EntityMap

def clean_entities(entities: EntityMap) -> EntityMap:
    """Drop absent/empty values while preserving model-produced entity names."""
    cleaned: EntityMap = {}
    for key, value in entities.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, (list, tuple)) and not value:
            continue
        cleaned[str(key)] = value
    return cleaned

class EntityExtractor(ABC):
    @abstractmethod
    def extract(self, text: str) -> ExtractedEntities:
        raise NotImplementedError
