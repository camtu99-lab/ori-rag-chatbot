from __future__ import annotations
import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from threading import RLock
from typing import Protocol
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict
from .config import ContextConfig

class ConversationStore(Protocol):
    """Persistence boundary cho transcript hội thoại.
    ContextManager chỉ phụ thuộc interface này. Có thể đổi RAM -> Redis/DB
    mà không sửa orchestration hoặc business logic.
    """
    def get_messages(self, session_id: str) -> list[BaseMessage]: ...
    def append_messages(self, session_id: str, messages: Sequence[BaseMessage]) -> None: ...
    def replace_messages(self, session_id: str, messages: Sequence[BaseMessage]) -> None: ...
    def clear(self, session_id: str) -> None: ...

class InMemoryConversationStore:
    def __init__(self) -> None:
        self._data: dict[str, list[BaseMessage]] = defaultdict(list)
        self._lock = RLock()

    def get_messages(self, session_id: str) -> list[BaseMessage]:
        with self._lock:
            return list(self._data.get(session_id, ()))

    def append_messages(self, session_id: str, messages: Sequence[BaseMessage]) -> None:
        if not messages:
            return
        with self._lock:
            self._data[session_id].extend(messages)

    def replace_messages(self, session_id: str, messages: Sequence[BaseMessage]) -> None:
        with self._lock:
            self._data[session_id] = list(messages)

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._data.pop(session_id, None)

class RedisConversationStore:
    """Redis-backed transcript store cho multi-process/multi-instance deployment."""

    def __init__(self, redis_url: str, key_prefix: str, ttl_seconds: int) -> None:
        try:
            from redis import Redis
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Redis context store cần package 'redis'") from exc

        self._client = Redis.from_url(redis_url, decode_responses=True)
        self._key_prefix = key_prefix.rstrip(":")
        self._ttl_seconds = ttl_seconds

    def _key(self, session_id: str) -> str:
        # Không đưa raw session id vào Redis key: tránh key quá dài/ký tự lạ và giảm rò rỉ identifier khi vận hành Redis.
        digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
        return f"{self._key_prefix}:{digest}"

    def _refresh_ttl(self, key: str) -> None:
        if self._ttl_seconds > 0:
            self._client.expire(key, self._ttl_seconds)

    def get_messages(self, session_id: str) -> list[BaseMessage]:
        raw_items = self._client.lrange(self._key(session_id), 0, -1)
        if not raw_items:
            return []
        payload = [json.loads(item) for item in raw_items]
        return list(messages_from_dict(payload))

    def append_messages(self, session_id: str, messages: Sequence[BaseMessage]) -> None:
        if not messages:
            return
        key = self._key(session_id)
        serialized = [json.dumps(message_to_dict(msg), ensure_ascii=False) for msg in messages]
        pipe = self._client.pipeline(transaction=True)
        pipe.rpush(key, *serialized)
        if self._ttl_seconds > 0:
            pipe.expire(key, self._ttl_seconds)
        pipe.execute()

    def replace_messages(self, session_id: str, messages: Sequence[BaseMessage]) -> None:
        key = self._key(session_id)
        pipe = self._client.pipeline(transaction=True)
        pipe.delete(key)
        if messages:
            serialized = [json.dumps(message_to_dict(msg), ensure_ascii=False) for msg in messages]
            pipe.rpush(key, *serialized)
            if self._ttl_seconds > 0:
                pipe.expire(key, self._ttl_seconds)
        pipe.execute()

    def clear(self, session_id: str) -> None:
        self._client.delete(self._key(session_id))

def build_conversation_store(config: ContextConfig) -> ConversationStore:
    if config.store_backend == "memory":
        return InMemoryConversationStore()

    import os

    redis_url = os.getenv("CONTEXT_REDIS_URL") or os.getenv("REDIS_URL")
    if not redis_url:
        raise ValueError("CONTEXT_STORE_BACKEND=redis nhưng thiếu CONTEXT_REDIS_URL/REDIS_URL")
    return RedisConversationStore(
        redis_url=redis_url,
        key_prefix=config.redis_key_prefix,
        ttl_seconds=config.session_ttl_seconds,
    )
