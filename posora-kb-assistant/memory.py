"""
Module quản lý trạng thái lịch sử hội thoại (Conversation Memory Manager):
1. Mục tiêu & Phân tách trách nhiệm:
   - Quản lý trạng thái đa lượt (Multi-turn State): LLM hoạt động stateless; module chịu trách nhiệm duy trì context xuyên suốt phiên tương tác.
   - Decoupling: Trừu tượng hóa việc quản lý danh sách message, chuẩn hóa format đầu vào cho LLM và sẵn sàng mở rộng backend lưu trữ (In-memory -> Redis/SQL Session) mà không sửa đổi `chatbot.py`.
2. Chức năng cốt lõi:
   - Lưu trữ & Cắt tỉa (Rolling Window): Tự động duy trì độ dài hội thoại trong giới hạn `MAX_HISTORY` để tối ưu token limit.
   - Ghép ngữ cảnh RAG: Tích hợp context truy xuất được vào đúng vị trí payload message của lượt tương tác hiện hành.
"""
from __future__ import annotations  
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from prompts import SYSTEM_PROMPT

MAX_HISTORY = 20  # số message tối đa giữ lại (không tính system prompt)

# State của phiên chat hiện tại. Luôn có SystemMessage ở vị trí đầu tiên.
conversation_history: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]

def _trim() -> None:
    """Chỉ giữ N tin nhắn gần nhất, luôn giữ system prompt ở đầu.
    Tránh context bị tràn / tốn token khi hội thoại quá dài."""
    global conversation_history
    system_msg = conversation_history[0]
    recent = conversation_history[1:][-MAX_HISTORY:]
    conversation_history = [system_msg] + recent

def append_user_message(question: str) -> None:
    """Lưu câu hỏi GỐC (không kèm context RAG) vào lịch sử rồi trim ngay -
    để lịch sử gọn, tránh phình to / lặp lại context RAG nhiều lần qua các
    lượt chat (context RAG chỉ dùng cho ĐÚNG lượt gọi hiện tại."""
    conversation_history.append(HumanMessage(content=question))
    _trim()

def append_ai_message(answer: str) -> None:
    """Lưu câu trả lời CUỐI CÙNG của AI vào lịch sử để nhớ cho lượt sau.
    Không lưu các bước tool_call trung gian vào đây (xem tool_executor.py)
    để lịch sử gọn, tránh phình to."""
    conversation_history.append(AIMessage(content=answer))

def build_augmented_message(user_input: str, context: str) -> str:
    """Ghép câu hỏi user với context RAG lấy được từ Redis."""
    if context:
        return f"{user_input}\n\n[THÔNG TIN THAM KHẢO]\n{context}"
    return f"{user_input}\n\n[THÔNG TIN THAM KHẢO]\n(không tìm thấy thông tin liên quan trong dữ liệu quán)"

def build_llm_messages(augmented_input: str) -> list[BaseMessage]:
    """Ghép lịch sử hội thoại với câu hỏi đã được chèn context RAG (Augmented Prompt) cho lượt gọi LLM hiện tại.
    Câu hỏi kèm context chỉ phục vụ payload tạm thời của request này, tuyệt đối không lưu đè vào `conversation_history` gốc.
    """
    return conversation_history[:-1] + [HumanMessage(content=augmented_input)]

def reset() -> None:
    """Đưa lịch sử hội thoại về trạng thái ban đầu (chỉ còn system prompt).
    LƯU Ý: module này KHÔNG reset giỏ hàng - giỏ hàng là state riêng do
    tools.py quản lý. chatbot.reset_history() gọi cả memory.reset() lẫn
    tools.reset_cart() để 2 module không cần phụ thuộc chéo vào nhau."""
    global conversation_history
    conversation_history = [SystemMessage(content=SYSTEM_PROMPT)]
