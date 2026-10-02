"""Reusable schema marker for dynamic optional NER modifiers."""
from __future__ import annotations
from typing import Any
from pydantic import Field

def entity_sink_field():
    """Pydantic field that receives unmatched NER entities from ToolArgumentBuilder."""
    return Field(
        default_factory=dict,
        description="Các tuỳ chọn chỉ xuất hiện khi người dùng nói rõ.",
        json_schema_extra={"x-entity-sink": True},
    )

EntityCustomizations = dict[str, Any]
