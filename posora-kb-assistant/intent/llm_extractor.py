"""Intent classifier routed through the shared LiteLLM Gateway.
This extractor keeps only the core order entities needed for product
cross-validation. Detailed order modifiers are owned by PhoBERT NER.
"""
from __future__ import annotations
import json
import logging
import os
from intent.extractor_base import EntityExtractor, ExtractedEntities, clean_entities
from llm_client import build_gateway_client, get_gateway_settings

logger = logging.getLogger(__name__)

_VALID_INTENTS = (
    "add_to_cart",
    "view_cart",
    "remove_from_cart",
    "checkout",
    "unknown",
)

_SYSTEM_PROMPT = """Bạn là bộ phân loại intent cho chatbot quán cà phê.
Trả về ĐÚNG 1 JSON object, không markdown/giải thích.

Schema:
{
  "intent": "add_to_cart" | "view_cart" | "remove_from_cart" | "checkout" | "unknown",
  "product_name": string hoặc null,
  "size": string hoặc null,
  "quantity": số nguyên hoặc null
}

Không suy diễn thông tin khách không nói. product_name giữ nguyên cách khách gọi.
quantity phải là null nếu khách không nói rõ số lượng.
"""

class LLMExtractor(EntityExtractor):
    def __init__(self, model: str | None = None) -> None:
        settings = get_gateway_settings()
        self._llm = build_gateway_client(
            model=model or settings.intent_model,
            temperature=float(os.getenv("LLM_INTENT_TEMPERATURE", "0")),
            max_tokens=int(os.getenv("LLM_INTENT_MAX_TOKENS", "200")),
        )

    def extract(self, text: str) -> ExtractedEntities:
        response = self._llm.invoke([
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ])
        return self._parse_response(response.content)

    @staticmethod
    def _parse_response(content: object) -> ExtractedEntities:
        if isinstance(content, list):
            text = "".join(
                block if isinstance(block, str) else str(block.get("text", ""))
                for block in content
                if isinstance(block, (str, dict))
            )
        else:
            text = str(content)

        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip()

        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            logger.warning("LLMExtractor: invalid JSON: %r", text)
            return ExtractedEntities(intent="unknown", entities={})

        intent = data.get("intent")
        if intent not in _VALID_INTENTS:
            intent = "unknown"

        entities = {
            "product_name": data.get("product_name"),
            "size": data.get("size"),
            "quantity": data.get("quantity"),
        }

        if entities["product_name"] is not None:
            entities["product_name"] = str(entities["product_name"]).strip() or None

        if entities["size"] is not None:
            entities["size"] = str(entities["size"]).strip().upper() or None

        if entities["quantity"] is not None:
            try:
                entities["quantity"] = int(entities["quantity"])
            except (TypeError, ValueError):
                entities["quantity"] = None

        return ExtractedEntities(intent=intent, entities=clean_entities(entities))
