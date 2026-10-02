# DMP Coffee AI Assistant

**DMP Coffee AI Assistant** is a Vietnamese conversational assistant for a coffee-shop domain. The project combines Retrieval-Augmented Generation (RAG), LLM-based intent routing, PhoBERT-based entity extraction, business tool calling, conversation context management, quota control, MCP exposure, and an optional voice interface.

The current implementation is a modular AI Engineering prototype rather than a production-ready system.

## Current Capabilities

- Vietnamese text chat with the **Ori** coffee-shop assistant persona.
- LLM access routed through a **LiteLLM Gateway** using an OpenAI-compatible client.
- **LLM-based intent classification** for routing user requests.
- **PhoBERT NER** for extracting entities used by business tools.
- Product consistency validation between LLM and NER outputs.
- **Schema-driven tool argument building** from tool/Pydantic schemas.
- Business tool execution for coffee-shop operations such as cart and checkout workflows.
- Centralized **Tool Registry** shared by the chatbot and MCP adapter.
- **Retrieval-Augmented Generation (RAG)** over coffee-shop knowledge.
- Dense/sparse retrieval components, hybrid retrieval, and reranking.
- Conversation context management with configurable history/context limits.
- Session-aware chatbot API through `session_id`.
- LLM quota management through a reusable runnable wrapper.
- **MCP server** exposing registered business tools over `stdio`.
- Optional **voice chat** pipeline with microphone input, STT, chatbot processing, and TTS.
- Automated tests for context management, quota handling, tools, gateway behavior, PhoBERT runtime, sparse retrieval, MCP, and voice-related components.

## Architecture

The main text-chat flow is orchestrated by `chatbot.py`.

```text
User
  |
  v
chatbot.py
  |
  +--> LLM Intent Classifier
  |
  +--> Is the intent mapped to a business tool?
  |        |
  |        +--> No --> RAG + Context Management --> LLM
  |        |
  |        +--> Yes
  |              |
  |              +--> Tool schema already satisfied?
  |              |        |
  |              |        +--> Yes --> Execute tool
  |              |
  |              +--> PhoBERT NER
  |                       |
  |                       +--> Product validation
  |                       |
  |                       +--> ToolArgumentBuilder
  |                                  |
  |                                  +--> Execute tool
  |
  +--> Retrieve RAG context
  |
  +--> Assemble conversation context
  |
  +--> LLM + tool-result messages
  |
  v
Ori response
```

The chatbot uses fallback behavior: when intent classification, NER, product validation, or required tool-slot preparation cannot safely produce a tool call, the request continues through the RAG/LLM response path instead of forcing an invalid business action.

## Main Components

### Chatbot Orchestration

`chatbot.py` is the main composition/orchestration layer. It coordinates intent classification, NER, product validation, tool preparation and execution, RAG retrieval, context assembly, LLM generation, quota errors, and conversation recording.

Main public functions:

```python
ask(question: str, session_id: str | None = None) -> str
reset_history(session_id: str | None = None) -> None
```

The project also provides the alias:

```python
chat = ask
```

### LLM Gateway

LLM traffic is routed through a LiteLLM Gateway rather than coupling the application directly to one provider.

`llm_client.py` builds an OpenAI-compatible `ChatOpenAI` client using:

- `LLM_GATEWAY_URL`
- `LITELLM_MASTER_KEY`
- `LLM_GATEWAY_MODEL`
- `LLM_GATEWAY_INTENT_MODEL`
- `LLM_GATEWAY_TIMEOUT_SECONDS`

`llm.py` creates the main chat model and binds the registered business tools to it.

The gateway configuration is kept under:

```text
llm_gateway/
├── config.yaml
├── gateway.py
└── README.md
```

This design keeps model/provider routing outside the chatbot orchestration layer.

### Intent Classification and NER

Intent/entity processing is organized under `intent/`.

The current orchestration uses two responsibilities:

1. **LLM Intent Classifier** — determines the user's intent and produces generic entities.
2. **PhoBERT NER** — extracts structured entities needed for business-tool execution.

The extractor factory keeps extractor creation behind a common abstraction, while product validation is isolated in `product_validation_strategy.py`.

A rule-based `ner_extractor.py` also exists as an extractor implementation. Its product lookup is dynamic through the catalog/menu data rather than embedding the product list directly in the extractor.

### Tool Calling

Business tools are centralized under `tools/` and exposed through `ALL_TOOLS`.

`tool_registry.py` discovers tools from that source, validates their contracts, detects duplicate names, and provides lookup maps. Tool names are therefore not duplicated manually inside the registry.

`ToolArgumentBuilder` builds tool arguments from each tool's schema. Required fields are derived from the schema, and unmatched extracted entities can be collected by schema fields marked as an entity sink.

`tool_executor.py` is responsible for:

```text
LLM tool call
    -> resolve registered tool
    -> validate/invoke tool
    -> create ToolMessage
    -> send result back to LLM
    -> final natural-language response
```

The executor limits recursive tool interaction to prevent an unlimited tool-call loop.

### Coffee-Shop Business Tools

The `tools/` package contains the coffee-shop business layer:

```text
tools/
├── cart.py
├── catalog.py
├── customization_contract.py
├── definitions.py
├── pricing.py
└── response.py
```

Catalog operations read product information from project data rather than maintaining a duplicate product list inside the NER implementation.

### Retrieval-Augmented Generation

RAG is implemented as a dedicated package:

```text
rag/
├── embeddings.py
├── hybrid_retriever.py
├── ingest.py
├── rag.py
├── reranker.py
├── retriever.py
├── sparse_retriever.py
└── vector_store.py
```

The package separates ingestion, embedding/vector-store access, dense/sparse retrieval, hybrid retrieval, reranking, and the application-facing retrieval interface.

Knowledge sources currently stored under `data/` include:

```text
data/
├── faq.md
├── menu.json
├── menu.md
├── promotions.md
└── stopwords_vi.txt
```

The chatbot retrieves relevant context for each request and passes that context to the context-management layer before generation.

### Context Management

The active context-management implementation is located in:

```text
context_management/
├── config.py
├── manager.py
├── store.py
└── token_counter.py
```

`context_runtime.py` acts as the composition root: it loads `ContextConfig`, injects the Ori system prompt, creates the configured conversation store and token counter, and returns a cached `ContextManager`.

During a chatbot turn, retrieved RAG information is passed as a `ContextBlock`, then combined with the conversation history for the current LLM request. After generation, the final user/assistant turn is recorded through the context manager.

`session_id` is already accepted by the chatbot/context API. The exact persistence and isolation behavior depends on the configured conversation store.

> `memory.py` and `conversation_state.py` remain in the repository, but the current `chatbot.py` orchestration uses the newer `context_management` package for conversation context.

### Quota Management

Quota handling is separated into:

```text
quota_management/
├── config.py
├── manager.py
├── runnable.py
├── store.py
└── usage.py
```

`llm_client.py` wraps the gateway client with `QuotaRunnable`, while `quota_runtime.py` provides the application-level `QuotaManager`.

If a quota is exceeded, `chatbot.py` catches `QuotaExceededError` and returns a user-facing retry message instead of exposing an internal exception.

### MCP

`mcp_server.py` provides an MCP adapter using `FastMCP`.

It does not redefine business logic. Instead, it discovers the same registered tools used by the chatbot through `tool_registry.py` and exposes their underlying functions through an MCP server using `stdio` transport.

Run it with:

```bash
python mcp_server.py
```

### Voice Interface

The project contains an optional voice client:

```text
voice/
├── config.py
├── microphone.py
├── speaker.py
├── stt.py
├── text_normalizer.py
├── tts.py
├── tts_text_prep.py
└── voice_chat.py
```

`voice_main.py` provides the CLI entry point. The user presses Enter to start recording and Enter again to stop, after which the voice pipeline processes the request and returns Ori's answer.

Run it with:

```bash
python voice_main.py
```

## Project Structure

The main source layout is:

```text
Chatbot/
├── chatbot.py
├── context_runtime.py
├── conversation_state.py
├── llm.py
├── llm_client.py
├── mcp_server.py
├── memory.py
├── prompts.py
├── quota_runtime.py
├── tool_argument_builder.py
├── tool_executor.py
├── tool_registry.py
├── voice_main.py
│
├── context_management/
├── intent/
├── llm_gateway/
├── quota_management/
├── rag/
├── tools/
├── voice/
│
├── data/
│   ├── faq.md
│   ├── menu.json
│   ├── menu.md
│   ├── promotions.md
│   └── stopwords_vi.txt
│
├── ner_model/
│   └── best_phobert_ner/
│
├── test/
│   ├── gateway_llm/
│   ├── intent/
│   ├── mcp/
│   ├── test_voice/
│   └── tools/
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

Generated caches such as `__pycache__/` and `.pytest_cache/` are omitted from the structure above.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/doanminhphuc2011-debug/ai-knowledge-assistant.git
cd ai-knowledge-assistant
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Some modules also contain component-specific requirement files for their own dependencies.

### 4. Configure environment variables

Copy the provided example:

```bash
copy .env.example .env
```

On macOS/Linux:

```bash
cp .env.example .env
```

Then configure the values required by the enabled components.

Important gateway variables used directly by the current source include:

```env
LITELLM_MASTER_KEY=your_master_key
LLM_GATEWAY_URL=http://localhost:4000/v1
LLM_GATEWAY_MODEL=dmp-chat
LLM_GATEWAY_INTENT_MODEL=dmp-intent
LLM_GATEWAY_TIMEOUT_SECONDS=60
```

RAG/vector-store, quota, context, voice, and model-provider settings may require additional variables defined by their corresponding configuration modules and `.env.example`.

**Do not commit the real `.env` file or API keys.**

## Running the Project

### Start the LiteLLM Gateway

The chatbot expects the configured gateway to be available before LLM requests are made.

Use the project's LiteLLM gateway configuration under `llm_gateway/config.yaml` with your LiteLLM installation/environment.

### Prepare RAG Data

When the knowledge data or vector index needs to be initialized/rebuilt, use the ingestion module:

```bash
python -m rag.ingest
```

### Run Text Chat

```bash
python chatbot.py
```

Example:

```text
☕ Ori - Trợ lý quán cà phê DMP
Nhập 'exit' để thoát.

👤 Bạn: Cho tôi một cà phê sữa size M
Ori: ...
```

### Run Voice Chat

```bash
python voice_main.py
```

### Run MCP Server

```bash
python mcp_server.py
```

## Testing

Tests are organized by component under `test/`, including coverage for:

- Context management
- Quota manager/runnable
- Tool argument building
- Tool registry
- Cart customization
- PhoBERT runtime labels
- Sparse-retrieval stopwords
- LiteLLM gateway behavior
- MCP server
- Voice-related behavior

Run the test suite with:

```bash
pytest
```

Specific groups can also be executed independently, for example:

```bash
pytest test/gateway_llm
pytest test/mcp
pytest test/tools
```

The repository also contains intent benchmark artifacts under `test/intent/`. These should be treated separately from the old RAG evaluation/report workflow.

## Design Principles

The project currently follows several practical separation-of-concerns rules:

- **Orchestration vs. business logic** — `chatbot.py` coordinates modules; coffee-shop operations stay in `tools/`.
- **Provider abstraction** — application code communicates through the LiteLLM Gateway instead of embedding provider-specific calls throughout the project.
- **Schema-driven tool preparation** — required tool arguments are derived from tool schemas rather than duplicated in router logic.
- **Centralized tool discovery** — chatbot and MCP reuse the same registry.
- **Dynamic catalog lookup** — product data comes from the catalog/menu source rather than a duplicated product-name list in NER code.
- **Context isolation** — generic context-management code receives business prompt/infrastructure dependencies through the application composition root.
- **Fail-safe routing** — uncertain extraction or incomplete tool arguments fall back to the conversational RAG path instead of executing an unsafe tool call.
- **Modular interfaces** — intent/NER, retrieval, context, quota, voice, and tools are separated so individual implementations can evolve independently.

## Current Development Status

| Area | Status |
|---|---|
| Text chatbot orchestration | Implemented |
| Persona/System Prompt | Implemented |
| LiteLLM Gateway integration | Implemented |
| LLM intent classification | Implemented |
| PhoBERT NER integration | Implemented |
| Product validation | Implemented |
| Business Tool Calling | Implemented |
| Schema-driven tool arguments | Implemented |
| Tool Registry | Implemented |
| RAG pipeline | Implemented |
| Hybrid/sparse retrieval components | Implemented |
| Reranking component | Implemented |
| Context Management | Implemented |
| Quota Management | Implemented |
| MCP tool adapter | Implemented |
| Voice client | Implemented |
| Component tests | Implemented / ongoing |
| System-wide evaluation and tuning | Ongoing |
| Production deployment | Not yet claimed |

## Known Development Notes

- The repository is still evolving and contains some legacy modules alongside newer implementations.
- `memory.py` represents an older conversation-memory approach; the current chatbot path uses `context_management`.
- `conversation_state.py` contains pending-order state logic, while current orchestration and future session/state design should be reviewed together before treating it as a complete multi-user state solution.
- Local model files under `ner_model/` can be large and should be managed carefully in Git.
- Generated caches, secrets, local runtime data, and generated evaluation reports should not be committed.

## Roadmap

Near-term work is focused on validating and improving the modules that are already implemented rather than presenting unimplemented features as complete.

```text
1. Re-evaluate Context Management behavior
2. Re-evaluate LiteLLM Gateway routing and failure handling
3. Evaluate intent classification and PhoBERT NER
4. Evaluate tool routing and business-tool workflows
5. Re-evaluate RAG retrieval/reranking quality
6. Validate MCP integration
7. Validate voice pipeline
8. Review session/state behavior for multiple users
9. Perform end-to-end testing and quality tuning
10. Prepare deployment/documentation after the architecture is stable
```

## Security

- Keep `.env` out of version control.
- Never commit provider API keys, LiteLLM master keys, Qdrant credentials, or other secrets.
- Use `.env.example` only as a template.
- Validate tool arguments before business-tool execution.
- Keep quota and provider limits configured appropriately for the deployment environment.

## License / Purpose

This project is developed for educational and AI Engineering internship purposes.

---

**Ori — DMP Coffee AI Assistant**
