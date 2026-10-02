from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Protocol

@dataclass(slots=True)
class UsageEvent:
    timestamp: float
    tokens: int = 0

@dataclass(slots=True)
class QuotaBucket:
    events: deque[UsageEvent] = field(default_factory=deque)
    pending_requests: int = 0

class QuotaStore(Protocol):
    def get_bucket(self, key: str) -> QuotaBucket: ...

class InMemoryQuotaStore:
    """Bộ nhớ cục bộ thread-safe, thiết kế backend-agnostic để dễ dàng thay thế bằng Redis khi scale multi-instance mà không cần sửa code nơi gọi."""
    def __init__(self) -> None:
        self._buckets: dict[str, QuotaBucket] = {}
        self._lock = threading.RLock()

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    def get_bucket(self, key: str) -> QuotaBucket:
        with self._lock:
            return self._buckets.setdefault(key, QuotaBucket())

    @staticmethod
    def prune(bucket: QuotaBucket, *, window_seconds: int, now: float | None = None) -> None:
        current = time.time() if now is None else now
        cutoff = current - window_seconds
        while bucket.events and bucket.events[0].timestamp <= cutoff:
            bucket.events.popleft()
