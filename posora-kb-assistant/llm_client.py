from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from quota_management import QuotaRunnable
from quota_runtime import get_quota_manager

load_dotenv()

@dataclass(frozen=True, slots=True)
class GatewaySettings:
    base_url: str
    api_key: str
    chat_model: str
    intent_model: str
    request_timeout_seconds: float

    @classmethod
    def from_env(cls) -> "GatewaySettings":
        api_key = os.getenv("LITELLM_MASTER_KEY")
        if not api_key:
            raise ValueError("Thiếu LITELLM_MASTER_KEY trong file .env")

        chat_model = os.getenv("LLM_GATEWAY_MODEL", "dmp-chat").strip()
        intent_model = os.getenv("LLM_GATEWAY_INTENT_MODEL", "dmp-intent").strip()

        return cls(
            base_url=os.getenv("LLM_GATEWAY_URL", "http://localhost:4000/v1").rstrip("/"),
            api_key=api_key,
            chat_model=chat_model,
            intent_model=intent_model,
            request_timeout_seconds=float(os.getenv("LLM_GATEWAY_TIMEOUT_SECONDS", "60")),
        )

@lru_cache(maxsize=1)
def get_gateway_settings() -> GatewaySettings:
    return GatewaySettings.from_env()

def build_gateway_client(*, model: str, temperature: float, max_tokens: int) -> QuotaRunnable:
    settings = get_gateway_settings()
    client = ChatOpenAI(
        model=model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=settings.request_timeout_seconds,
    )
    return QuotaRunnable(client, manager=get_quota_manager(), workload=model)
