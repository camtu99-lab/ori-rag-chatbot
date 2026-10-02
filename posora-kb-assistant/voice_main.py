"""
CLI Entry Point thực thi Voice Chat Client tích hợp:
1. Luồng tương tác phím Enter linh hoạt (Interactive Recording):
   - Phân định UI / Core: File quản lý prompt và lần nhấn Enter thứ 1 (Start); lần nhấn Enter thứ 2 (Stop) do `Microphone.record_on_enter()` điều khiển ngầm.
   - Loại bỏ hardcode timeout: Thu âm thời lượng động phù hợp với độ dài câu nói của người dùng.
2. Quản lý Vòng đời & Cấu hình:
   - Tự động nạp cấu hình từ `VoiceConfig` (Singleton qua `.env`).
   - Duy trì Session Loop liên tục cho tới khi người dùng chủ động ngắt bằng `KeyboardInterrupt` (Ctrl+C).
3. Nguyên tắc xử lý ngoại lệ (Fail-Fast Policy):
   - Không nuốt ngoại lệ nghiệp vụ/phần cứng: Để lỗi crash tự nhiên kèm traceback chi tiết giúp dễ debug khi gặp sự cố driver âm thanh hoặc LLM/STT/TTS API.
"""
from __future__ import annotations
import logging
from voice.voice_chat import VoiceChat

# Kích hoạt logger console ở mức INFO cho standalone entry point.
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    voice_chat = VoiceChat()
    print("Voice Chat - Nhấn Enter để nói, Ctrl+C để thoát.\n")
    while True:
        try:
            input("Nhấn Enter để bắt đầu ghi âm...")
            print("Đang ghi âm... Nhấn Enter để dừng.")
            answer = voice_chat.listen_once_interactive()
            if answer:
                print(f"Ori: {answer}\n")
        except KeyboardInterrupt:
            break
