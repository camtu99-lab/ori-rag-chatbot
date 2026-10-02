"""Module I/O thu âm phần cứng thuần túy qua `sounddevice`: Ghi âm in-memory và trả về `np.ndarray` trực tiếp cho `stt.py`,
 không lưu file tạm và độc lập hoàn toàn với tầng xử lý nghiệp vụ/LLM."""
from __future__ import annotations

import logging

import numpy as np
import sounddevice as sd

from voice.config import get_voice_config

logger = logging.getLogger(__name__)

_CHANNELS = 1
_DTYPE = "float32"

class Microphone:
    """Bọc `sounddevice`, chỉ expose đúng 1 hành vi: thu âm N giây ->
    numpy array (mono, float32).

    Không giữ state giữa các lần gọi `record()` (không buffer, không
    cache) - mỗi lần gọi là một lượt thu âm độc lập, tự chứa. Điều này
    giúp `Microphone` dễ test độc lập: test chỉ cần mock `sounddevice`,
    không phụ thuộc `chatbot`/`stt`/bất kỳ module nào khác.
    """

    def __init__(self) -> None:
        """Đọc cấu hình từ voice.config.get_voice_config() và kiểm tra
        có microphone khả dụng hay không.

        Raises:
            RuntimeError: nếu VOICE_ENABLED=False (không nên khởi tạo
                Microphone khi Voice đang tắt), hoặc nếu không tìm thấy
                microphone khả dụng trên hệ thống (không có thiết bị
                input, hoặc thiết bị input mặc định không hỗ trợ
                sample_rate/số kênh yêu cầu).
        """
        config = get_voice_config()

        if not config.enabled:
            raise RuntimeError(
                "Microphone được khởi tạo nhưng VOICE_ENABLED=False. "
                "Không nên khởi tạo Microphone khi Voice đang tắt."
            )

        self._sample_rate = config.sample_rate

        try:
            sd.check_input_settings(
                samplerate=self._sample_rate,
                channels=_CHANNELS,
                dtype=_DTYPE,
            )
        except Exception as exc:
            logger.exception(
                "Không tìm thấy microphone khả dụng (sample_rate=%d Hz, channels=%d)",
                self._sample_rate,
                _CHANNELS,
            )
            raise RuntimeError(
                "Không tìm thấy microphone khả dụng trên hệ thống"
            ) from exc

        logger.info("Đã khởi tạo Microphone (sample_rate=%d Hz)", self._sample_rate)

    def record(self, duration: float) -> np.ndarray:
        """Thu âm từ microphone mặc định của hệ thống trong `duration`
        giây.

        Args:
            duration: thời lượng cần thu âm, tính bằng giây, phải > 0.

        Returns:
            Waveform mono, dtype float32, biên độ trong khoảng
            [-1.0, 1.0], mảng 1 chiều (đã squeeze bỏ chiều channel), thu
            ở sample rate = VOICE_SAMPLE_RATE (config.sample_rate) -
            đúng định dạng mà `SpeechToText.transcribe()` (stt.py) yêu
            cầu.

        Raises:
            ValueError: nếu `duration` không hợp lệ (không phải số,
                hoặc <= 0).
            RuntimeError: nếu quá trình thu âm thất bại (microphone bị
                rút ra giữa chừng, lỗi driver PortAudio, v.v.).
        """
        self._validate_duration(duration)

        num_frames = int(duration * self._sample_rate)
        logger.info(
            "Bắt đầu ghi âm %.2fs (%d mẫu, %d Hz)",
            duration,
            num_frames,
            self._sample_rate,
        )

        try:
            audio = sd.rec(
                num_frames,
                samplerate=self._sample_rate,
                channels=_CHANNELS,
                dtype=_DTYPE,
            )
            sd.wait()
        except Exception as exc:
            logger.exception("Lỗi khi ghi âm từ microphone (duration=%.2fs)", duration)
            raise RuntimeError("Không thể ghi âm từ microphone") from exc

        # sd.rec() trả về mảng shape (num_frames, channels); channels=1
        # nên squeeze về mảng 1 chiều (num_frames,) - đúng shape mà
        # stt.py (_validate_audio: audio.ndim != 1) yêu cầu.
        waveform = audio.reshape(-1).astype(np.float32)

        logger.info("Ghi âm xong: %d mẫu (%.2fs)", waveform.shape[0], duration)
        return waveform

    @staticmethod
    def _validate_duration(duration: float) -> None:
        """Kiểm tra `duration` hợp lệ trước khi thu âm, để lỗi (nếu có)
        báo ngay tại đây với message rõ ràng, thay vì để sounddevice ném
        ra lỗi khó hiểu (hoặc tệ hơn: thu âm 0 giây/số âm một cách im
        lặng)."""
        if not isinstance(duration, (int, float)) or isinstance(duration, bool):
            raise ValueError(
                f"duration phải là số (int/float), nhận được: {type(duration)!r}"
            )
        if duration <= 0:
            raise ValueError(f"duration phải > 0, hiện tại: {duration}")

    def record_on_enter(self) -> np.ndarray:
        """Ghi âm không cố định thời lượng bằng InputStream callback và dừng khi người dùng nhấn Enter,
          trả về mảng 1D float32 cho STT."""
        frames: list[np.ndarray] = []

        def _callback(indata: np.ndarray, frame_count: int, time_info, status) -> None:
            # Callback chạy trên AUDIO THREAD riêng của PortAudio, không
            # phải main thread - CHỈ append vào list (thao tác nhanh,
            # không block), không làm gì tốn thời gian ở đây để tránh
            # tràn buffer audio (underrun) nếu callback xử lý quá chậm.
            frames.append(indata.copy())

        logger.info("Bắt đầu ghi âm (nhấn Enter để dừng)...")
        try:
            stream = sd.InputStream(
                samplerate=self._sample_rate,
                channels=_CHANNELS,
                dtype=_DTYPE,
                callback=_callback,
            )
            with stream:
                # input() chặn MAIN THREAD tới khi người dùng nhấn Enter -
                # trong lúc đó, audio thread (callback ở trên) vẫn tiếp
                # tục chạy nền, gom frame vào `frames`. Không cần vòng lặp
                # sleep/polling thời lượng - đây chính là cách tránh giới
                # hạn cố định 5 giây của thiết kế cũ.
                input()
        except Exception as exc:
            logger.exception("Lỗi khi ghi âm từ microphone (chế độ Enter)")
            raise RuntimeError("Không thể ghi âm từ microphone") from exc

        if not frames:
            logger.info("Ghi âm xong: 0 mẫu (dừng ngay lập tức, chưa thu được gì)")
            return np.array([], dtype=np.float32)

        waveform = np.concatenate(frames, axis=0).reshape(-1).astype(np.float32)
        logger.info(
            "Ghi âm xong: %d mẫu (%.2fs)",
            waveform.shape[0],
            waveform.shape[0] / self._sample_rate,
        )
        return waveform
