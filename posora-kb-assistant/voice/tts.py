"""
voice/tts.py
Local Text-to-Speech using Piper.

Runtime pipeline:
    text -> prepare_tts_text -> Piper local ONNX model -> WAV bytes -> Speaker.play()

Không gọi edge-tts, websocket, API hay Internet trong lúc synthesize.
Model phải được tải sẵn về máy trước khi chạy.
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import time
import wave
from pathlib import Path
from typing import AsyncIterator

from piper import PiperVoice

from voice.config import get_voice_config
from voice.tts_text_prep import prepare_tts_text

logger = logging.getLogger(__name__)

_TTS_MODEL_ENV = "VOICE_TTS_MODEL_PATH"


class TextToSpeech:
    """Piper local TTS nhưng giữ API cũ để không phải sửa voice_chat.py."""

    def __init__(self) -> None:
        config = get_voice_config()
        if not config.enabled:
            raise RuntimeError(
                "TextToSpeech được khởi tạo nhưng VOICE_ENABLED=False."
            )

        raw_model_path = os.getenv(_TTS_MODEL_ENV, "").strip()
        if not raw_model_path:
            raise RuntimeError(
                f"Thiếu {_TTS_MODEL_ENV} trong .env. "
                "Ví dụ: VOICE_TTS_MODEL_PATH=voice/models/piper/"
                "vi_VN-vais1000-medium.onnx"
            )

        model_path = Path(raw_model_path)
        if not model_path.is_absolute():
            # Resolve theo working directory của project khi chạy voice_main.py.
            model_path = Path.cwd() / model_path

        model_path = model_path.resolve()
        config_path = Path(str(model_path) + ".json")

        if not model_path.is_file():
            raise FileNotFoundError(
                f"Không tìm thấy Piper model: {model_path}"
            )
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Không tìm thấy Piper model config: {config_path}"
            )

        self._model_path = model_path

        logger.info(
            "Đang tải local TTS model='%s'...",
            self._model_path.name,
        )
        try:
            self._voice = PiperVoice.load(str(self._model_path))
        except Exception as exc:
            logger.exception("Không thể load Piper TTS model")
            raise RuntimeError("Không thể khởi tạo Piper local TTS") from exc

        logger.info(
            "Đã khởi tạo TextToSpeech "
            "(provider=piper-local, model='%s')",
            self._model_path.name,
        )

    def synthesize(self, text: str) -> bytes:
        """Text -> WAV bytes hoàn chỉnh, tương thích Speaker.play(bytes)."""
        self._validate_text(text)

        prep_start = time.perf_counter()
        tts_text = prepare_tts_text(text)
        logger.info(
            "[TTS] prepare completed in %.2fs",
            time.perf_counter() - prep_start,
        )

        if not tts_text.strip():
            raise ValueError("Text rỗng sau bước prepare_tts_text")

        synth_start = time.perf_counter()

        try:
            buffer = io.BytesIO()
            # Piper synthesize_wav cần wave.Wave_write.
            with wave.open(buffer, "wb") as wav_file:
                self._voice.synthesize_wav(tts_text, wav_file)

            audio_bytes = buffer.getvalue()
        except Exception as exc:
            logger.exception(
                "Piper local TTS lỗi khi tổng hợp %d ký tự",
                len(tts_text),
            )
            raise RuntimeError("Không thể tổng hợp giọng nói bằng Piper") from exc

        if not audio_bytes:
            raise RuntimeError("Piper không trả về audio")

        logger.info(
            "[TTS] local provider completed in %.2fs",
            time.perf_counter() - synth_start,
        )
        logger.info("[TTS] audio bytes = %d", len(audio_bytes))
        return audio_bytes

    async def synthesize_async(self, text: str) -> bytes:
        """Giữ backward compatibility với API async cũ.

        Piper là local synchronous inference, chạy ở worker thread để
        không block event loop nếu hàm này được gọi từ async code.
        """
        return await asyncio.to_thread(self.synthesize, text)

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        """Backward-compatible stream API.

        Pipeline hiện tại không dùng hàm này. Yield 1 WAV hoàn chỉnh để
        tránh giả lập raw streaming không cần thiết.
        """
        yield await self.synthesize_async(text)

    @staticmethod
    def _validate_text(text: str) -> None:
        if not isinstance(text, str):
            raise ValueError(
                f"text phải là str, nhận được: {type(text)!r}"
            )
        if not text.strip():
            raise ValueError("text rỗng, không có gì để tổng hợp giọng nói")
