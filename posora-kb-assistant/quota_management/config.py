from __future__ import annotations
import os
from dataclasses import dataclass
from functools import lru_cache

@dataclass(frozen=True, slots=True)
class QuotaLimit:
    max_requests: int | None
    max_tokens: int | None
    window_seconds: int

@dataclass(frozen=True, slots=True)
class QuotaSettings:
    enabled: bool
    default_subject: str
    default_limit: QuotaLimit

    @classmethod
    def from_env(cls) -> "QuotaSettings":
        def optional_positive_int(name: str) -> int | None:
            raw = os.getenv(name, "").strip()
            if not raw:
                return None
            value = int(raw)
            if value <= 0:
                raise ValueError(f"{name} phải > 0")
            return value

        enabled = os.getenv("QUOTA_ENABLED", "true").strip().lower() in {
            "1", "true", "yes", "on"
        }
        window = int(os.getenv("QUOTA_WINDOW_SECONDS", "3600"))
        if window <= 0:
            raise ValueError("QUOTA_WINDOW_SECONDS phải > 0")

        return cls(
            enabled=enabled,
            default_subject=os.getenv("QUOTA_DEFAULT_SUBJECT", "default").strip() or "default",
            default_limit=QuotaLimit(
                max_requests=optional_positive_int("QUOTA_MAX_REQUESTS"),
                max_tokens=optional_positive_int("QUOTA_MAX_TOKENS"),
                window_seconds=window,
            ),
        )

@lru_cache(maxsize=1)
def get_quota_settings() -> QuotaSettings:
    return QuotaSettings.from_env()
