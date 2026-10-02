"""
voice/stt.py
Speech-to-Text: Audio (np.ndarray) -> Text (raw).

Pipeline:
    Audio -> [stt.py] -> raw text -> text_normalizer.py -> chatbot.ask()

Backend hiện tại dùng Hugging Face Whisper-compatible model qua Transformers.
Model ID, device và sample rate vẫn lấy từ VoiceConfig/.env; file này không
chứa business rule, tên món, size hay mapping domain.
"""
from __future__ import annotations

import logging

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from voice.config import get_voice_config

logger = logging.getLogger(__name__)


class SpeechToText:
    """Giữ nguyên public interface cũ: transcribe(audio) -> raw text."""

    def __init__(self) -> None:
        config = get_voice_config()

        if not config.enabled:
            raise RuntimeError(
                "SpeechToText được khởi tạo nhưng VOICE_ENABLED=False."
            )

        self._model_id = config.model
        self._sample_rate = config.sample_rate

        configured_device = config.device.strip().lower()
        if configured_device in {"cuda", "gpu"}:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "VOICE_DEVICE yêu cầu CUDA nhưng torch.cuda.is_available()=False"
                )
            self._device = "cuda:0"
            self._dtype = torch.float16
        elif configured_device == "cpu":
            self._device = "cpu"
            self._dtype = torch.float32
        else:
            raise ValueError(
                f"VOICE_DEVICE='{config.device}' không hỗ trợ. "
                "Hiện STT chấp nhận 'cpu' hoặc 'cuda'."
            )

        logger.info(
            "Đang tải STT model='%s' trên device='%s'...",
            self._model_id,
            self._device,
        )

        try:
            self._processor = AutoProcessor.from_pretrained(self._model_id)
            self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self._model_id,
                torch_dtype=self._dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            ).to(self._device)
            self._model.eval()
        except Exception as exc:
            logger.exception(
                "Không thể tải STT model='%s' trên device='%s'",
                self._model_id,
                self._device,
            )
            raise RuntimeError(
                f"Không thể khởi tạo STT model='{self._model_id}'"
            ) from exc

        logger.info("Đã tải xong STT model='%s'", self._model_id)

    def transcribe(self, audio: np.ndarray) -> str:
        self._validate_audio(audio)

        try:
            inputs = self._processor(
                audio,
                sampling_rate=self._sample_rate,
                return_tensors="pt",
            )

            input_features = inputs.input_features.to(
                device=self._device,
                dtype=self._dtype,
            )

            with torch.inference_mode():
                predicted_ids = self._model.generate(
                    input_features,
                    max_new_tokens=128,
                )

            text = self._processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True,
            )[0].strip()

        except Exception as exc:
            logger.exception(
                "Lỗi khi transcribe audio (%d mẫu, %.2fs)",
                audio.shape[0],
                audio.shape[0] / self._sample_rate,
            )
            raise RuntimeError("Không thể transcribe audio") from exc

        logger.info(
            "Transcribe xong: %.2fs audio, độ dài text=%d ký tự",
            audio.shape[0] / self._sample_rate,
            len(text),
        )
        return text

    @staticmethod
    def _validate_audio(audio: np.ndarray) -> None:
        if not isinstance(audio, np.ndarray):
            raise ValueError(
                f"audio phải là np.ndarray, nhận được: {type(audio)!r}"
            )
        if audio.ndim != 1:
            raise ValueError(
                f"audio phải là mảng 1 chiều (mono), nhận được ndim={audio.ndim}"
            )
        if audio.size == 0:
            raise ValueError("audio rỗng, không có gì để transcribe")
        if audio.dtype != np.float32:
            raise ValueError(
                f"audio phải có dtype=float32, nhận được: {audio.dtype}"
            )
