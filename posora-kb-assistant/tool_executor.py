"""
Trách nhiệm duy nhất:
    LLM -> resolve tool -> execute -> trả ToolMessage -> LLM tiếp tục.
Business logic nằm ở tools.py.
Registry nằm ở tool_registry.py.
"""
from __future__ import annotations
from typing import Any
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.runnables import Runnable
from pydantic import ValidationError
from tool_registry import TOOLS_BY_NAME
from tools import error_response

MAX_TOOL_ITERATIONS = 5

_FALLBACK_ANSWER = ("Xin lỗi, hiện tôi đang xử lý yêu cầu này chưa xong. " "Bạn vui lòng thử lại hoặc nói rõ hơn giúp tôi nhé.")

def _extract_text(content: Any) -> str:
    """Chuẩn hóa content của các LLM provider về str."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("text"):
                parts.append(str(block["text"]))
        return "".join(parts)

    return str(content)

def _execute_tool_call(tool_call: dict) -> str:
    """Execute tool an toàn và luôn trả structured JSON string."""
    tool_name = tool_call.get("name")
    tool_args = tool_call.get("args", {})

    if not tool_name:
        return error_response("invalid_tool_call", "Tool call không có tên tool.")

    tool_fn = TOOLS_BY_NAME.get(tool_name)
    if tool_fn is None:
        return error_response("unknown_tool", f"Không tìm thấy tool '{tool_name}'.")
    try:
        return tool_fn.invoke(tool_args)
    except ValidationError as exc:
        return error_response("invalid_arguments", f"Tham số truyền vào tool '{tool_name}' không hợp lệ: " f"{exc.errors()}")
    except Exception as exc:  # noqa: BLE001
        return error_response("tool_execution_error", f"Lỗi khi thực thi tool '{tool_name}': {exc}")

def generate_with_tools(llm: Runnable, messages: list[BaseMessage]) -> str:
    """Generate và xử lý tool calls cho tới khi có final response."""
    response: AIMessage = llm.invoke(messages)
    iterations = 0

    while response.tool_calls and iterations < MAX_TOOL_ITERATIONS:
        iterations += 1
        messages = messages + [response]

        for tool_call in response.tool_calls:
            result = _execute_tool_call(tool_call)
            messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))

        response = llm.invoke(messages)

    if response.tool_calls:
        return _FALLBACK_ANSWER

    return _extract_text(response.content)
