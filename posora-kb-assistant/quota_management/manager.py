from __future__ import annotations
import time
from dataclasses import dataclass
from quota_management.config import QuotaLimit, QuotaSettings
from quota_management.store import InMemoryQuotaStore, UsageEvent

class QuotaExceededError(RuntimeError):
    def __init__(self, *, resource: str, limit: int, used: int, retry_after_seconds: int):
        self.resource = resource
        self.limit = limit
        self.used = used
        self.retry_after_seconds = retry_after_seconds
        super().__init__(
            f"Quota exceeded: {resource} used={used} limit={limit}; "
            f"retry_after={retry_after_seconds}s"
        )

@dataclass(frozen=True, slots=True)
class QuotaSnapshot:
    requests: int
    tokens: int
    pending_requests: int

class QuotaReservation:
    def __init__(self, manager: "QuotaManager", key: str, limit: QuotaLimit):
        self._manager = manager
        self._key = key
        self._limit = limit
        self._closed = False

    def commit(self, tokens: int | None = None) -> None:
        if self._closed:
            return
        self._manager._commit(self._key, self._limit, max(0, tokens or 0))
        self._closed = True

    def cancel(self) -> None:
        if self._closed:
            return
        self._manager._cancel(self._key)
        self._closed = True

class QuotaManager:
    def __init__(self, settings: QuotaSettings, store: InMemoryQuotaStore | None = None) -> None:
        self.settings = settings
        self.store = store or InMemoryQuotaStore()

    def reserve(self, *, workload: str, subject: str | None = None, limit: QuotaLimit | None = None) -> QuotaReservation:
        selected = limit or self.settings.default_limit
        key = self._key(subject or self.settings.default_subject, workload)

        if not self.settings.enabled:
            return QuotaReservation(self, key, selected)

        now = time.time()
        with self.store.lock:
            bucket = self.store.get_bucket(key)
            self.store.prune(bucket, window_seconds=selected.window_seconds, now=now)

            requests = len(bucket.events) + bucket.pending_requests
            tokens = sum(event.tokens for event in bucket.events)

            if selected.max_requests is not None and requests >= selected.max_requests:
                raise self._exceeded(bucket=bucket, selected=selected, resource="requests", limit=selected.max_requests, used=requests, now=now)

            if selected.max_tokens is not None and tokens >= selected.max_tokens:
                raise self._exceeded(bucket=bucket, selected=selected, resource="tokens", limit=selected.max_tokens, used=tokens, now=now)

            bucket.pending_requests += 1

        return QuotaReservation(self, key, selected)

    def snapshot(self, *, workload: str, subject: str | None = None, limit: QuotaLimit | None = None) -> QuotaSnapshot:
        selected = limit or self.settings.default_limit
        key = self._key(subject or self.settings.default_subject, workload)
        with self.store.lock:
            bucket = self.store.get_bucket(key)
            self.store.prune(bucket, window_seconds=selected.window_seconds)
            return QuotaSnapshot(
                requests=len(bucket.events),
                tokens=sum(event.tokens for event in bucket.events),
                pending_requests=bucket.pending_requests,
            )

    def _commit(self, key: str, limit: QuotaLimit, tokens: int) -> None:
        if not self.settings.enabled:
            return
        with self.store.lock:
            bucket = self.store.get_bucket(key)
            bucket.pending_requests = max(0, bucket.pending_requests - 1)
            bucket.events.append(UsageEvent(timestamp=time.time(), tokens=tokens))
            self.store.prune(bucket, window_seconds=limit.window_seconds)

    def _cancel(self, key: str) -> None:
        if not self.settings.enabled:
            return
        with self.store.lock:
            bucket = self.store.get_bucket(key)
            bucket.pending_requests = max(0, bucket.pending_requests - 1)

    @staticmethod
    def _key(subject: str, workload: str) -> str:
        return f"{subject}:{workload}"

    @staticmethod
    def _exceeded(*, bucket, selected, resource, limit, used, now):
        if bucket.events:
            retry = max(1, int(bucket.events[0].timestamp + selected.window_seconds - now) + 1)
        else:
            retry = selected.window_seconds
        return QuotaExceededError(resource=resource, limit=limit, used=used, retry_after_seconds=retry)
