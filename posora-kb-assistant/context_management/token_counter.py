from __future__ import annotations
import math
from typing import Protocol, Sequence
from langchain_core.messages import BaseMessage
from .config import ContextConfig

class TokenCounter(Protocol):
    def count_text(self, text: str) -> int: ...

    def count_messages(self, messages: Sequence[BaseMessage]) -> int: ...

    def truncate_text(self, text: str, max_tokens: int) -> str: ...

class HeuristicTokenCounter:
    """Provider-agnostic fallback khi không có tokenizer chính xác.
    Gateway có thể fallback qua nhiều model/tokenizer khác nhau nên tầng
    application không nên phụ thuộc cứng vào tokenizer của một provider.
    """
    def __init__(self, chars_per_token: float, message_overhead_tokens: int) -> None:
        self._chars_per_token = chars_per_token
        self._message_overhead_tokens = message_overhead_tokens

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        return max(1, math.ceil(len(text) / self._chars_per_token))

    def count_messages(self, messages: Sequence[BaseMessage]) -> int:
        return sum(
            self.count_text(str(message.content)) + self._message_overhead_tokens
            for message in messages
        )

    def truncate_text(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0 or not text:
            return ""
        max_chars = max(1, int(max_tokens * self._chars_per_token))
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip()

class TiktokenCounter:
    """Tokenizer chính xác hơn cho model tương thích OpenAI tokenizer.
    Nếu package/encoding không khả dụng, factory tự fallback về heuristic.
    """
    def __init__(self, encoding_name: str, message_overhead_tokens: int) -> None:
        import tiktoken
        self._encoding = tiktoken.get_encoding(encoding_name)
        self._message_overhead_tokens = message_overhead_tokens

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        return len(self._encoding.encode(text, disallowed_special=()))

    def count_messages(self, messages: Sequence[BaseMessage]) -> int:
        return sum(
            self.count_text(str(message.content)) + self._message_overhead_tokens
            for message in messages
        )

    def truncate_text(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0 or not text:
            return ""
        tokens = self._encoding.encode(text, disallowed_special=())
        if len(tokens) <= max_tokens:
            return text
        return self._encoding.decode(tokens[:max_tokens]).rstrip()

def build_token_counter(config: ContextConfig) -> TokenCounter:
    try:
        return TiktokenCounter(
            encoding_name=config.token_encoding,
            message_overhead_tokens=config.message_overhead_tokens,
        )
    except (ImportError, KeyError, ValueError):
        return HeuristicTokenCounter(
            chars_per_token=config.chars_per_token,
            message_overhead_tokens=config.message_overhead_tokens,
        )
