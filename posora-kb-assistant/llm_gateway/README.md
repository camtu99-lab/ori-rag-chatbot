# LLM Gateway response fix

The Groq GPT-OSS deployment can spend completion budget on reasoning.
For the chatbot gateway we do not expose reasoning, so the Groq deployment
sets:

include_reasoning: false
reasoning_effort: low

The gateway test also allows a larger completion budget and normalizes
LangChain string/list content.

Start:
    py -m llm_gateway.gateway

Test:
    py -m llm_gateway.test_gateway
    py -m llm_gateway.test_gateway_raw
