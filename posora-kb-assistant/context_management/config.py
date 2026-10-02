from __future__ import annotations
import os
from dataclasses import dataclass

def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.getenv(name)
    value = default if raw in (None, "") else int(raw)
    if value < minimum:
        raise ValueError(f"{name} phải >= {minimum}, nhận được {value}")
    return value

def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = os.getenv(name)
    value = default if raw in (None, "") else float(raw)
    if value <= minimum:
        raise ValueError(f"{name} phải > {minimum}, nhận được {value}")
    return value

@dataclass(frozen=True, slots=True)
class ContextConfig:
    """Cấu hình Context Management, hoàn toàn tách khỏi business domain.
    max_input_tokens là ngân sách prompt an toàn dùng chung cho mọi provider
    phía sau Gateway. Nên đặt theo model có context window nhỏ nhất trong
    fallback chain, không theo model lớn nhất.
    """
    max_input_tokens: int
    reserved_output_tokens: int
    max_history_tokens: int
    max_external_context_tokens: int
    default_session_id: str
    store_backend: str
    session_ttl_seconds: int
    redis_key_prefix: str
    token_encoding: str
    chars_per_token: float
    message_overhead_tokens: int

    @property
    def prompt_budget(self) -> int:
        return self.max_input_tokens - self.reserved_output_tokens

    @classmethod
    def from_env(cls) -> "ContextConfig":
        config = cls(
            max_input_tokens=_env_int("CONTEXT_MAX_INPUT_TOKENS", 8192, minimum=1),
            reserved_output_tokens=_env_int("CONTEXT_RESERVED_OUTPUT_TOKENS", 768, minimum=0),
            max_history_tokens=_env_int("CONTEXT_MAX_HISTORY_TOKENS", 3200, minimum=0),
            max_external_context_tokens=_env_int("CONTEXT_MAX_EXTERNAL_TOKENS", 3200, minimum=0),
            default_session_id=os.getenv("CONTEXT_DEFAULT_SESSION_ID", "default").strip() or "default",
            store_backend=os.getenv("CONTEXT_STORE_BACKEND", "memory").strip().lower(),
            session_ttl_seconds=_env_int("CONTEXT_SESSION_TTL_SECONDS", 86400, minimum=0),
            redis_key_prefix=os.getenv("CONTEXT_REDIS_KEY_PREFIX", "dmp:context").strip() or "dmp:context",
            token_encoding=os.getenv("CONTEXT_TOKEN_ENCODING", "cl100k_base").strip() or "cl100k_base",
            chars_per_token=_env_float("CONTEXT_CHARS_PER_TOKEN", 3.5, minimum=0.0),
            message_overhead_tokens=_env_int("CONTEXT_MESSAGE_OVERHEAD_TOKENS", 4, minimum=0),
        )
        if config.reserved_output_tokens >= config.max_input_tokens:
            raise ValueError("CONTEXT_RESERVED_OUTPUT_TOKENS phải nhỏ hơn CONTEXT_MAX_INPUT_TOKENS")
        if config.store_backend not in {"memory", "redis"}:
            raise ValueError("CONTEXT_STORE_BACKEND chỉ nhận 'memory' hoặc 'redis'")
        return config
