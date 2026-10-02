"""
Registry tập trung cho business tools.
tools.py vẫn là source of truth duy nhất cho định nghĩa tool.
Module này chỉ chịu trách nhiệm discovery, validation và lookup.
Không hard-code tên tool.
"""
from __future__ import annotations
from collections.abc import Iterable
from typing import Any
from tools import ALL_TOOLS

class ToolRegistry:
    """Registry tập trung, kiểm tra duplicate và tool contract."""
    def __init__(self, tools: Iterable[Any]) -> None:
        discovered = tuple(tools)

        invalid = [
            tool
            for tool in discovered
            if not getattr(tool, "name", None)
            or not callable(getattr(tool, "invoke", None))
        ]
        if invalid:
            names = [getattr(tool, "name", "<unknown>") for tool in invalid]
            raise TypeError(f"Invalid tool definitions: {names}")

        names = [tool.name for tool in discovered]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Duplicate tool names: {duplicates}")

        self._tools = discovered
        self._by_name = {tool.name: tool for tool in discovered}

    @property
    def tools(self) -> tuple[Any, ...]:
        return self._tools

    def get(self, name: str) -> Any | None:
        return self._by_name.get(name)

    def as_dict(self) -> dict[str, Any]:
        return dict(self._by_name)

tool_registry = ToolRegistry(ALL_TOOLS)

REGISTERED_TOOLS = tool_registry.tools
TOOLS_BY_NAME = tool_registry.as_dict()
