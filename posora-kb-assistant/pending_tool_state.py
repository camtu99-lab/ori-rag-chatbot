"""Session-scoped pending business-tool state.

This module stores only structured entities that the user has already supplied.
It does not know PRODUCT/SIZE/QUANTITY, tool names, menu items, or domain rules.

The chatbot decides when a tool request is incomplete/failed; this store only:
- remembers the intent/tool route,
- merges newly extracted entities into the previous entity mapping,
- isolates state by session_id,
- expires stale pending state.

This is intentionally separate from conversation history: transcript context and
business transaction state are different concerns.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any


DEFAULT_SESSION_ID = "__default__"
PENDING_TTL_SECONDS = int(os.getenv("PENDING_TOOL_TTL_SECONDS", "900"))


@dataclass(frozen=True, slots=True)
class PendingToolState:
    intent: str
    entities: dict[str, Any] = field(default_factory=dict)
    updated_at: float = field(default_factory=time.time)

    def merged(self, new_entities: dict[str, Any] | None) -> "PendingToolState":
        merged = dict(self.entities)
        for key, value in (new_entities or {}).items():
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            if isinstance(value, (list, tuple, dict)) and not value:
                continue
            merged[str(key)] = value
        return PendingToolState(
            intent=self.intent,
            entities=merged,
            updated_at=time.time(),
        )


class PendingToolStore:
    def __init__(self, ttl_seconds: int = PENDING_TTL_SECONDS) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds phải > 0")
        self._ttl_seconds = ttl_seconds
        self._states: dict[str, PendingToolState] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _key(session_id: str | None) -> str:
        normalized = (session_id or "").strip()
        return normalized or DEFAULT_SESSION_ID

    def get(self, session_id: str | None = None) -> PendingToolState | None:
        key = self._key(session_id)
        now = time.time()
        with self._lock:
            state = self._states.get(key)
            if state is None:
                return None
            if now - state.updated_at > self._ttl_seconds:
                self._states.pop(key, None)
                return None
            return state

    def put(
        self,
        *,
        intent: str,
        entities: dict[str, Any] | None,
        session_id: str | None = None,
    ) -> PendingToolState:
        state = PendingToolState(
            intent=intent,
            entities={},
        ).merged(entities)
        with self._lock:
            self._states[self._key(session_id)] = state
        return state

    def merge(
        self,
        *,
        new_entities: dict[str, Any] | None,
        session_id: str | None = None,
    ) -> PendingToolState | None:
        key = self._key(session_id)
        with self._lock:
            current = self.get(session_id)
            if current is None:
                return None
            updated = current.merged(new_entities)
            self._states[key] = updated
            return updated

    def clear(self, session_id: str | None = None) -> None:
        with self._lock:
            self._states.pop(self._key(session_id), None)


pending_tool_store = PendingToolStore()
