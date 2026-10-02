"""Tool-enabled application LLM routed exclusively through LiteLLM Gateway."""

from __future__ import annotations

import os

from langchain_core.runnables import Runnable

from llm_client import build_gateway_client, get_gateway_settings
from tools import ALL_TOOLS

_settings = get_gateway_settings()

_llm = build_gateway_client(
    model=_settings.chat_model,
    temperature=float(os.getenv("LLM_CHAT_TEMPERATURE", "0.7")),
    max_tokens=int(os.getenv("LLM_CHAT_MAX_TOKENS", "300")),
)

llm: Runnable = _llm.bind_tools(ALL_TOOLS)
