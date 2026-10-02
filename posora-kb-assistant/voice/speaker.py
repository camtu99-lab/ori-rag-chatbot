"""
voice/speaker.py
Speaker: audio bytes -> phát ra loa.

`play(audio: bytes) -> None` (API chính, giữ nguyên hoàn toàn) là API DUY
NHẤT mà kiến trúc sentence-level mới ở voice_chat.py dùng: mỗi câu/đoạn
đã được `TextToSpeech.synthesize()` tổng hợp thành audio HOÀN CHỈNH, rồi
gọi `play()` - BLOCK tới khi phát xong mới trả về - nên không cần thêm
API stream/queue nào mới ở đây: gọi `play()` tuần tự cho từng đoạn (ở
voice_chat.py) đã tự nhiên đảm bảo không phát chồng, không phát đoạn sau
trước đoạn trước.

`play_stream()` được GIỮ LẠI (backward compatibility, phòng khi có nơi
khác ngoài phạm vi 3 file đang sửa gọi tới), nhưng làm rõ trong docstring:
đây KHÔNG PHẢI real-time streaming - nó gom (buffer) toàn bộ chunk trước
rồi mới decode+phát 1 lần duy nhất (đúng những gì code này vốn đã làm).
Kiến trúc mới ở voice_chat.py KHÔNG gọi hàm này nữa.
"""
from __future__ import annotations

import io
import logging
import time
from typing import Callable, Iterator

import sounddevice as sd
import soundfile as sf

logger = logging.getLogger(__name__)


class Speaker:
    """Bọc `soundfile` (decode) + `sounddevice` (playback)."""

    def __init__(self) -> None:
        from voice.config import get_voice_config

        config = get_voice_config()

        if not config.enabled:
            raise RuntimeError(
                "Speaker được khởi tạo nhưng VOICE_ENABLED=False. "
                "Không nên khởi tạo Speaker khi Voice đang tắt."
            )

        try:
            sd.check_output_settings()
        except Exception as exc:
            logger.exception("Không tìm thấy thiết bị phát âm thanh (loa) khả dụng")
            raise RuntimeError(
                "Không tìm thấy thiết bị phát âm thanh (loa) khả dụng trên hệ thống"
            ) from exc

        logger.info("Đã khởi tạo Speaker")

    def play(self, audio: bytes) -> None:
        """Decode toàn bộ `audio` (phải là audio HOÀN CHỈNH, không phải 1
        fragment MP3 dở) và phát ra loa, BLOCK cho tới khi phát xong mới
        trả về - đây là tính chất mà kiến trúc sentence-level mới ở
        voice_chat.py dựa vào để đảm bảo các đoạn không bao giờ phát
        chồng lên nhau (đoạn kế tiếp chỉ được xử lý SAU KHI lệnh gọi
        `play()` này return)."""
        self._validate_audio(audio)

        try:
            data, sample_rate = sf.read(io.BytesIO(audio), dtype="float32")
        except Exception as exc:
            logger.exception("Không thể decode audio bytes (độ dài=%d byte)", len(audio))
            raise RuntimeError(
                "Không thể decode audio bytes - dữ liệu hỏng hoặc định dạng không được hỗ trợ"
            ) from exc

        duration_seconds = len(data) / float(sample_rate) if sample_rate else 0.0
        channels = data.shape[1] if data.ndim > 1 else 1

        logger.info(
            "Bắt đầu phát audio: %.2fs, %d Hz, %d kênh",
            duration_seconds,
            sample_rate,
            channels,
        )

        try:
            sd.play(data, sample_rate)
            sd.wait()
        except Exception as exc:
            logger.exception("Lỗi khi phát audio qua loa")
            raise RuntimeError("Không thể phát audio qua loa") from exc

        logger.info("Phát audio xong: %.2fs", duration_seconds)

    def play_stream(
        self,
        chunk_iterator: Iterator[bytes],
        on_first_play_callback: Callable[[float], None] | None = None,
    ) -> bool:
        """LEGACY - giữ lại để backward compatibility, KHÔNG dùng trong
        kiến trúc sentence-level mới (xem voice_chat.py).

        KHÔNG PHẢI real-time streaming: hàm này GOM (buffer) toàn bộ chunk
        từ `chunk_iterator` trước, rồi mới decode + phát 1 LẦN DUY NHẤT
        sau khi đã nhận hết - đúng những gì implementation này vốn đã làm
        (không đổi hành vi), chỉ làm rõ trong docstring để không ai hiểu
        nhầm đây là streaming thật."""
        buffer = bytearray()
        first_chunk_received = False
        start_time = time.perf_counter()

        for chunk in chunk_iterator:
            if chunk:
                if not first_chunk_received:
                    first_chunk_received = True
                    if on_first_play_callback:
                        on_first_play_callback(time.perf_counter() - start_time)
                buffer.extend(chunk)

        if not buffer:
            return False

        try:
            data, sample_rate = sf.read(io.BytesIO(buffer), dtype="float32")
        except Exception as exc:
            logger.exception("Không thể decode stream audio bytes: %s", exc)
            return False

        duration_seconds = len(data) / float(sample_rate) if sample_rate else 0.0
        channels = data.shape[1] if data.ndim > 1 else 1

        logger.info(
            "Bắt đầu phát stream audio (đã gom đủ buffer): %.2fs, %d Hz, %d kênh",
            duration_seconds,
            sample_rate,
            channels,
        )

        try:
            sd.play(data, sample_rate)
            sd.wait()
            logger.info("Phát audio stream xong: %.2fs", duration_seconds)
            return True
        except Exception as exc:
            logger.exception("Lỗi khi phát audio qua loa: %s", exc)
            return False

    @staticmethod
    def _validate_audio(audio: bytes) -> None:
        if not isinstance(audio, bytes):
            raise ValueError(f"audio phải là bytes, nhận được: {type(audio)!r}")
        if len(audio) == 0:
            raise ValueError("audio rỗng, không có gì để phát")
