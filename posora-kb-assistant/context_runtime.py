from __future__ import annotations
from functools import lru_cache
from context_management import (ContextConfig, ContextManager, build_conversation_store, build_token_counter)
from prompts import SYSTEM_PROMPT

@lru_cache(maxsize=1)
def get_context_manager() -> ContextManager:
    """Composition Root khởi tạo và tiêm (inject) các dependency hạ tầng, 
    prompt nghiệp vụ vào generic package `context_management`."""

    config = ContextConfig.from_env()
    return ContextManager(system_prompt=SYSTEM_PROMPT, config=config, store=build_conversation_store(config), token_counter=build_token_counter(config))
