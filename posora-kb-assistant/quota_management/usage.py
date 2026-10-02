from __future__ import annotations
from typing import Any

def extract_total_tokens(response: Any) -> int | None:
    """Đọc metadata về lượng token/usage từ LangChain/Provider mà không bị phụ thuộc vào provider cố định."""
    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, dict):
        total = usage.get("total_tokens")
        if total is not None:
            return _as_non_negative_int(total)
        input_tokens = _as_non_negative_int(usage.get("input_tokens")) or 0
        output_tokens = _as_non_negative_int(usage.get("output_tokens")) or 0
        if input_tokens or output_tokens:
            return input_tokens + output_tokens

    metadata = getattr(response, "response_metadata", None)
    if isinstance(metadata, dict):
        token_usage = metadata.get("token_usage") or metadata.get("usage")
        if isinstance(token_usage, dict):
            total = token_usage.get("total_tokens")
            if total is not None:
                return _as_non_negative_int(total)
            prompt = _as_non_negative_int(
                token_usage.get("prompt_tokens") or token_usage.get("input_tokens")
            ) or 0
            completion = _as_non_negative_int(
                token_usage.get("completion_tokens") or token_usage.get("output_tokens")
            ) or 0
            if prompt or completion:
                return prompt + completion
    return None

def _as_non_negative_int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None
