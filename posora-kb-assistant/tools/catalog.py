"""Catalog service: menu loading, size discovery and product matching."""
from __future__ import annotations
import difflib
import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Literal

try:
    from unidecode import unidecode
except ImportError:
    def unidecode(text: str) -> str:
        text = text.replace("đ", "d").replace("Đ", "D")
        decomposed = unicodedata.normalize("NFD", text)
        return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MENU_PATH = os.path.join(BASE_DIR, "data", "menu.json")

_menu_cache: list[dict] | None = None
_menu_cache_mtime: float | None = None
_PRICE_FIELD_RE = re.compile(r"^price_(\w+)$")

def _load_menu() -> list[dict]:
    global _menu_cache, _menu_cache_mtime

    mtime = os.path.getmtime(MENU_PATH)
    if _menu_cache is None or mtime != _menu_cache_mtime:
        with open(MENU_PATH, encoding="utf-8") as file:
            _menu_cache = json.load(file)
        _menu_cache_mtime = mtime
    return _menu_cache

def reload_menu() -> None:
    global _menu_cache, _menu_cache_mtime
    _menu_cache = None
    _menu_cache_mtime = None

def get_product_sizes(item: dict) -> dict[str, int | float]:
    """Derive valid sizes directly from price_* fields in menu data."""
    sizes: dict[str, int | float] = {}

    for key, value in item.items():
        match = _PRICE_FIELD_RE.match(key)
        if match and isinstance(value, (int, float)):
            sizes[match.group(1).upper()] = value
    return sizes

def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())

def _fold(text: str) -> str:
    return unidecode(text)

@dataclass
class ProductMatch:
    status: Literal[
        "exact",
        "normalized",
        "accent_insensitive",
        "fuzzy",
        "ambiguous",
        "not_found",
    ]
    product: dict | None = None
    suggestions: list[str] = field(default_factory=list)

def find_product(product_name: str) -> ProductMatch:
    menu = _load_menu()
    query = _normalize(product_name)

    if not query:
        return ProductMatch(status="not_found")

    normalized_names = {_normalize(item["name"]): item for item in menu}

    # 1. Exact
    if query in normalized_names:
        return ProductMatch(status="exact", product=normalized_names[query])

    # 2. Normalized substring
    substring_hits = [
        item
        for normalized_name, item in normalized_names.items()
        if query in normalized_name or normalized_name in query
    ]

    if len(substring_hits) == 1:
        return ProductMatch(status="normalized", product=substring_hits[0])

    if len(substring_hits) > 1:
        return ProductMatch(status="ambiguous", suggestions=[item["name"] for item in substring_hits[:5]])

    folded_names = {_fold(normalized_name): item for normalized_name, item in normalized_names.items()}
    query_folded = _fold(query)

    # 3. Accent-insensitive
    if query_folded in folded_names:
        return ProductMatch(status="accent_insensitive", product=folded_names[query_folded])

    folded_hits = [
        item
        for folded_name, item in folded_names.items()
        if query_folded in folded_name or folded_name in query_folded
    ]

    if len(folded_hits) == 1:
        return ProductMatch(status="accent_insensitive", product=folded_hits[0])

    if len(folded_hits) > 1:
        return ProductMatch(status="ambiguous", suggestions=[item["name"] for item in folded_hits[:5]])

    # 4. Fuzzy
    close = difflib.get_close_matches(query_folded, folded_names.keys(), n=5, cutoff=0.6)

    if len(close) == 1:
        return ProductMatch(status="fuzzy", product=folded_names[close[0]])

    if len(close) > 1:
        return ProductMatch(status="ambiguous", suggestions=[folded_names[name]["name"] for name in close])
    return ProductMatch(status="not_found")
