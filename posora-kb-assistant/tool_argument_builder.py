"""Schema-driven mapping từ entities trích xuất sang arguments của tool.
- Required slots: Tự động phân giải trực tiếp từ Pydantic/JSON schema của tool.
- Entity Sink (`x-entity-sink: true`): Tự động gom các entity chưa khớp (unmatched) vào tham số được khai báo extension này.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from tool_registry import tool_registry

@dataclass(frozen=True, slots=True)
class ToolBuildResult:
    tool_name: str | None
    arguments: dict[str, Any] | None
    missing_required: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        return self.tool_name is not None and self.arguments is not None and not self.missing_required

def _resolve_ref(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    ref = schema.get("$ref")
    if not ref:
        return schema
    if not ref.startswith("#/"):
        return schema
    node: Any = root
    for part in ref[2:].split("/"):
        node = node[part]
    return node if isinstance(node, dict) else schema

def _tool_schema(tool: Any) -> dict[str, Any]:
    args_schema = getattr(tool, "args_schema", None)
    if args_schema is not None and hasattr(args_schema, "model_json_schema"):
        return args_schema.model_json_schema()

    # LangChain BaseTool exposes ``args`` as property schemas.
    args = getattr(tool, "args", None)
    if isinstance(args, dict):
        return {"type": "object", "properties": args, "required": []}

    return {"type": "object", "properties": {}, "required": []}

def _is_entity_sink(schema: dict[str, Any]) -> bool:
    return bool(schema.get("x-entity-sink"))

def _build_object(schema: dict[str, Any], root: dict[str, Any], entities: dict[str, Any]) -> tuple[dict[str, Any], list[str], set[str]]:
    schema = _resolve_ref(schema, root)
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])

    result: dict[str, Any] = {}
    missing: list[str] = []
    consumed: set[str] = set()
    sinks: list[tuple[str, dict[str, Any]]] = []

    for name, raw_prop in properties.items():
        prop = _resolve_ref(raw_prop, root)

        if _is_entity_sink(prop):
            sinks.append((name, prop))
            continue

        value = entities.get(name)
        if value is not None:
            result[name] = value
            consumed.add(name)
            continue

        prop_type = prop.get("type")
        if prop_type == "array" and isinstance(prop.get("items"), dict):
            item_schema = _resolve_ref(prop["items"], root)
            if item_schema.get("type") == "object" or item_schema.get("properties"):
                item, item_missing, item_consumed = _build_object(item_schema, root, entities)
                if item_missing:
                    if name in required:
                        missing.extend(item_missing)
                elif item:
                    result[name] = [item]
                    consumed.update(item_consumed)
                elif name in required:
                    missing.append(name)
                continue

        if name in required:
            missing.append(name)

    extras = {
        key: value
        for key, value in entities.items()
        if key not in consumed and value is not None
    }
    if sinks and extras:
        # One sink is sufficient; multiple sinks would make routing ambiguous.
        sink_name, _ = sinks[0]
        result[sink_name] = extras
        consumed.update(extras)

    return result, missing, consumed

class ToolArgumentBuilder:
    def __init__(self, registry=tool_registry) -> None:
        self._registry = registry

    def has_tool(self, intent: str) -> bool:
        return self._registry.get(intent) is not None

    def build(self, intent: str, entities: dict[str, Any] | None = None) -> ToolBuildResult:
        tool = self._registry.get(intent)
        if tool is None:
            return ToolBuildResult(None, None)

        schema = _tool_schema(tool)
        arguments, missing, _ = _build_object(schema, schema, entities or {})
        if missing:
            return ToolBuildResult(tool.name, None, tuple(dict.fromkeys(missing)))
        return ToolBuildResult(tool.name, arguments)
