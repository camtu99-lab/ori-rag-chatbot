"""Module quản lý cấu hình Voice Chat độc lập: Triển khai frozen dataclass singleton với 
cơ chế fail-fast và conditional validation theo feature flag `VOICE_ENABLED` (mặc định tắt), 
đảm bảo không làm gián đoạn Text Chat."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

_VALID_OUTPUT_FORMATS = {"mp3", "wav", "ogg"}

def _require(name: str) -> str:
    """Đọc 1 biến bắt buộc từ .env, raise ngay nếu thiếu - cùng cách
    llm.py đang raise ValueError khi thiếu GROQ_API_KEY."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        raise ValueError(f"Thiếu {name} trong file .env")
    return raw.strip()

def _parse_bool(name: str, raw: str) -> bool:
    normalized = raw.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise ValueError(
        f"{name} trong .env không hợp lệ: '{raw}'. "
        f"Chỉ chấp nhận true/false (hoặc 1/0, yes/no, on/off)."
    )

def _parse_int(name: str, raw: str) -> int:
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} trong .env phải là số nguyên, hiện tại: '{raw}'")

def _parse_float(name: str, raw: str) -> float:
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{name} trong .env phải là số, hiện tại: '{raw}'")

def _read_unvalidated(name: str, default: str) -> str:
    """Đọc 1 biến khi VOICE_ENABLED=False: không bắt buộc phải có trong
    .env, không raise nếu thiếu/sai định dạng. Voice đang tắt nên giá trị
    này không được dùng tới - chỉ cần đủ để khởi tạo VoiceConfig."""
    raw = os.getenv(name)
    return raw.strip() if raw and raw.strip() else default

def _read_int_unvalidated(name: str, default: int) -> int:
    """Như _read_unvalidated nhưng cho số nguyên: vẫn ưu tiên đọc từ .env
    nếu có, nhưng không raise nếu thiếu/sai định dạng (Voice đang tắt)."""
    raw = os.getenv(name)
    try:
        return int(raw) if raw and raw.strip() else default
    except ValueError:
        return default

def _read_float_unvalidated(name: str, default: float) -> float:
    """Như _read_int_unvalidated nhưng cho số thực."""
    raw = os.getenv(name)
    try:
        return float(raw) if raw and raw.strip() else default
    except ValueError:
        return default

@dataclass(frozen=True)
class VoiceConfig:
    """Cấu hình Voice Chat. Immutable - khởi tạo xong là dùng, không sửa lại
    giữa chừng. Chỉ chứa DỮ LIỆU cấu hình, không chứa logic gọi STT/TTS
    (logic đó thuộc về stt.py/tts.py ở các phase sau)."""

    enabled: bool
    model: str
    language: str
    sample_rate: int
    device: str
    timeout: float
    output_format: str
    debug_transcript: bool

    def __post_init__(self) -> None:
        """Validate ngay khi khởi tạo (fail-fast) - không đợi tới lúc
        stt.py/tts.py thực sự dùng giá trị này mới phát hiện ra sai.

        Voice là feature optional (VOICE_ENABLED mặc định False). Khi tắt,
        người dùng chỉ chạy Text Chat và có thể không hề khai báo
        VOICE_MODEL/VOICE_SAMPLE_RATE/... trong .env - nên KHÔNG được
        validate các cấu hình STT/TTS trong trường hợp này. Chỉ khi
        enabled=True mới bắt buộc các giá trị này hợp lệ."""
        if not self.enabled:
            return

        if self.sample_rate <= 0:
            raise ValueError(
                f"VOICE_SAMPLE_RATE phải > 0, hiện tại: {self.sample_rate}"
            )

        if self.timeout <= 0:
            raise ValueError(f"VOICE_TIMEOUT phải > 0, hiện tại: {self.timeout}")

        if self.output_format not in _VALID_OUTPUT_FORMATS:
            raise ValueError(
                f"VOICE_OUTPUT_FORMAT '{self.output_format}' không hợp lệ. "
                f"Chỉ chấp nhận: {', '.join(sorted(_VALID_OUTPUT_FORMATS))}"
            )

@lru_cache(maxsize=1)
def get_voice_config() -> VoiceConfig:
    """Singleton: đọc + validate .env đúng 1 lần cho cả process. Các module
    khác trong voice/ luôn lấy config qua hàm này, không tự đọc os.getenv
    trực tiếp - để đảm bảo chỉ có DUY NHẤT một nguồn sự thật cho config."""
    enabled_raw = os.getenv("VOICE_ENABLED")
    enabled = _parse_bool("VOICE_ENABLED", enabled_raw) if enabled_raw is not None else False

    if enabled:
        # Voice đang BẬT: bắt buộc + validate đầy đủ, đúng như trước.
        return VoiceConfig(
            enabled=enabled,
            model=_require("VOICE_MODEL"),
            language=_require("VOICE_LANGUAGE"),
            sample_rate=_parse_int("VOICE_SAMPLE_RATE", _require("VOICE_SAMPLE_RATE")),
            device=_require("VOICE_DEVICE"),
            timeout=_parse_float("VOICE_TIMEOUT", _require("VOICE_TIMEOUT")),
            output_format=_require("VOICE_OUTPUT_FORMAT").lower(),
            debug_transcript=_parse_bool(
                "VOICE_DEBUG_TRANSCRIPT", os.getenv("VOICE_DEBUG_TRANSCRIPT", "false")
            ),
        )

    # Voice đang TẮT: đây là optional feature, người dùng chỉ chạy Text
    # Chat nên không được bắt buộc phải khai báo VOICE_MODEL/VOICE_SAMPLE_RATE/...
    # Đọc mềm (không raise) chỉ để có đủ giá trị khởi tạo VoiceConfig.
    return VoiceConfig(
        enabled=enabled,
        model=_read_unvalidated("VOICE_MODEL", ""),
        language=_read_unvalidated("VOICE_LANGUAGE", ""),
        sample_rate=_read_int_unvalidated("VOICE_SAMPLE_RATE", 0),
        device=_read_unvalidated("VOICE_DEVICE", ""),
        timeout=_read_float_unvalidated("VOICE_TIMEOUT", 0.0),
        output_format=_read_unvalidated("VOICE_OUTPUT_FORMAT", "").lower(),
        debug_transcript=False,
    )
