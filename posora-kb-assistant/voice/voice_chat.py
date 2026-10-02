"""
voice/voice_chat.py
Orchestrator: nối các module thành pipeline Voice Chat hoàn chỉnh.

    Microphone -> STT -> Normalize -> chatbot.ask() -> TTS -> Speaker

THIẾT KẾ LẠI PHẦN TTS/SPEAKER (SENTENCE-LEVEL SEQUENTIAL TTS):

Kiến trúc CŨ dùng 1 thread bridge các fragment MP3 thô từ
`TextToSpeech.stream()` qua 1 `queue.Queue` sang `Speaker.play_stream()` -
nhưng `play_stream()` vẫn phải GOM HẾT fragment vào buffer rồi mới
decode+phát 1 lần (không phải streaming thật), còn timeout 4s/8s ở tầng
TTS lại dễ fail giả khi mạng chỉ chậm chứ chưa chết hẳn. Bỏ hẳn cách này.

Kiến trúc MỚI: chẻ `answer` thành các CÂU/ĐOẠN NGẮN (theo dấu xuống dòng/
./?/!, và thêm dấu phẩy nếu 1 câu vẫn quá dài), rồi xử lý TUẦN TỰ từng
đoạn:

    answer
      -> _split_into_tts_chunks() -> [đoạn 1, đoạn 2, ..., đoạn n]
      -> với mỗi đoạn i (THEO ĐÚNG THỨ TỰ):
           TextToSpeech.synthesize(đoạn i)  # audio HOÀN CHỈNH của đoạn i
           Speaker.play(audio i)            # BLOCK tới khi phát xong
      -> đoạn i+1 CHỈ bắt đầu xử lý sau khi Speaker.play(audio i) return

Vì `Speaker.play()` luôn BLOCK tới khi phát xong, và vòng lặp xử lý các
đoạn là tuần tự (for thường, không phải nhiều task chạy song song), việc
"không phát chồng / không phát đoạn sau trước đoạn trước" là hệ quả TỰ
NHIÊN của chính cấu trúc vòng lặp - không cần queue, không cần
OutputStream, không cần callback phức tạp.

TỐI ƯU (không bắt buộc, chỉ ảnh hưởng TỐC ĐỘ, không ảnh hưởng THỨ TỰ):
trong lúc `Speaker.play()` đang phát đoạn i (block), 1 thread nền tổng hợp
TRƯỚC audio của đoạn i+1 (`_synthesize_chunk` chạy trong thread riêng).
Thread nền CHỈ tổng hợp (chuẩn bị sẵn dữ liệu), KHÔNG BAO GIỜ tự ý gọi
`Speaker.play()` - việc phát audio đoạn i+1 vẫn luôn đợi đúng lượt của nó
trong vòng lặp `for` chính, sau khi đoạn i phát xong. Nhờ vậy, ngay cả khi
bỏ hẳn thread nền này (comment out), tính đúng đắn về thứ tự phát và
không phát chồng KHÔNG hề thay đổi - thread nền thuần túy là 1 tối ưu
"che" thời gian tổng hợp audio đoạn kế tiếp vào lúc đoạn hiện tại đang
phát, giảm khoảng LẶNG giữa 2 đoạn.
"""
from __future__ import annotations

import logging
import re
import threading
import time

from chatbot import ask
from voice.config import get_voice_config
from voice.microphone import Microphone
from voice.speaker import Speaker
from voice.stt import SpeechToText
from voice.text_normalizer import TextNormalizer
from voice.tts import TextToSpeech

logger = logging.getLogger(__name__)

# Ngưỡng độ dài (ký tự) - 1 câu dài hơn mức này mới cân nhắc chẻ thêm theo
# dấu phẩy, để mỗi đoạn đưa vào TTS không quá dài (đọc quá dài mới là thứ
# đang muốn tránh, KHÔNG phải để chẻ vụn mọi câu ngắn).
_MAX_CHUNK_CHARS = 100

# Ranh giới câu: dấu . ? ! được coi là kết thúc câu CHỈ KHI theo sau là
# khoảng trắng hoặc hết chuỗi - vd. "37.000" có dấu '.' nhưng theo sau là
# '0' (không phải khoảng trắng) nên KHÔNG bị coi là ranh giới câu, không
# tách nhầm số tiền. Việc đọc số tiền tự nhiên ("ba mươi bảy nghìn đồng")
# vẫn do prepare_tts_text() xử lý (gọi bên trong TextToSpeech.synthesize()
# cho TỪNG đoạn) - file này không đụng gì tới logic đọc số.
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.?!])(?=\s|$)")
_LINE_SPLIT_RE = re.compile(r"\r?\n+")
_COMMA_SPLIT_RE = re.compile(r",\s+")


def _split_into_tts_chunks(text: str) -> list[str]:
    """Chẻ `text` thành các đoạn ngắn để đưa lần lượt vào TTS: trước tiên
    theo dòng, rồi theo câu (./?/!), rồi (chỉ khi câu vẫn quá dài) theo dấu
    phẩy - không bao giờ cắt giữa từ, không thay đổi nội dung chữ (chỉ
    tách, không xóa/sửa ký tự nào)."""
    chunks: list[str] = []
    for line in _LINE_SPLIT_RE.split(text):
        line = line.strip()
        if not line:
            continue
        for sentence in _SENTENCE_BOUNDARY_RE.split(line):
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) <= _MAX_CHUNK_CHARS:
                chunks.append(sentence)
                continue
            # Câu quá dài - chẻ thêm theo dấu phẩy, gộp lại thành các đoạn
            # vừa phải (không tách rời từng mảnh phẩy nhỏ lẻ nếu không cần).
            parts = _COMMA_SPLIT_RE.split(sentence)
            buf = ""
            for idx, part in enumerate(parts):
                piece = part if idx == 0 else ", " + part
                candidate = (buf + piece) if buf else piece
                if buf and len(candidate) > _MAX_CHUNK_CHARS:
                    chunks.append(buf)
                    buf = piece
                else:
                    buf = candidate
            if buf:
                chunks.append(buf)
    if chunks:
        return chunks
    stripped = text.strip()
    return [stripped] if stripped else []


class VoiceChat:
    """Điều phối toàn bộ pipeline hội thoại."""

    def __init__(
        self,
        microphone: Microphone | None = None,
        stt: SpeechToText | None = None,
        normalizer: TextNormalizer | None = None,
        tts: TextToSpeech | None = None,
        speaker: Speaker | None = None,
    ) -> None:
        self._microphone = microphone if microphone is not None else Microphone()
        self._stt = stt if stt is not None else SpeechToText()
        self._normalizer = normalizer if normalizer is not None else TextNormalizer()
        self._tts = tts if tts is not None else TextToSpeech()
        self._speaker = speaker if speaker is not None else Speaker()
        self._debug_transcript = get_voice_config().debug_transcript

    def _debug_text(self, label: str, text: str) -> None:
        """Hiển thị transcript chỉ khi VOICE_DEBUG_TRANSCRIPT=true."""
        if not self._debug_transcript:
            return
        print(f"[VOICE][{label}] {text}")
        logger.info("[VOICE][%s] %s", label, text)

    def _synthesize_chunk(self, index: int, text: str, total: int) -> tuple[bytes, float]:
        """Tổng hợp audio HOÀN CHỈNH cho 1 đoạn, kèm log + đo thời gian.
        `TextToSpeech.synthesize()` tự gọi `prepare_tts_text()` bên trong
        (giữ nguyên logic đọc số tiền/markdown) - file này không tự xử lý
        text, chỉ chuyển thẳng đoạn gốc vào."""
        logger.info("[TTS] chunk %d/%d preparing", index + 1, total)
        t0 = time.perf_counter()
        audio = self._tts.synthesize(text)
        dt = time.perf_counter() - t0
        logger.info("[TTS] chunk %d/%d synthesized in %.2fs", index + 1, total, dt)
        return audio, dt

    def _fallback_full_answer(self, answer: str) -> None:
        """CHỈ được gọi khi CHƯA đoạn nào được phát ra loa - tổng hợp và
        phát nguyên `answer` gốc 1 lần. An toàn vì chưa có gì phát trước
        đó nên không tạo trùng lặp âm thanh."""
        logger.warning(
            "[TTS] fallback: tổng hợp nguyên answer gốc (chưa đoạn nào được phát)"
        )
        audio = self._tts.synthesize(answer)
        self._speaker.play(audio)

    def _speak_answer(self, answer: str) -> None:
        """Đọc `answer` theo kiến trúc sentence-level sequential TTS (xem
        docstring đầu file). Không sửa/trả về `answer` - chỉ dùng nó để
        tạo audio, màn hình vẫn hiển thị đúng `answer` gốc (không đổi)."""
        chunks = _split_into_tts_chunks(answer)
        if not chunks:
            return
        total = len(chunks)

        pipeline_start = time.perf_counter()
        total_synth_time = 0.0
        total_play_time = 0.0
        played_any = False
        ttfa_logged = False

        # Tổng hợp đoạn ĐẦU TIÊN - chưa có gì để prefetch trước đó nên
        # phải chờ đồng bộ (không tránh được độ trễ của riêng đoạn 1).
        try:
            current_audio, dt = self._synthesize_chunk(0, chunks[0], total)
        except Exception as exc:
            logger.error("[TTS] chunk 1/%d tổng hợp thất bại: %s", total, exc)
            self._fallback_full_answer(answer)
            return
        total_synth_time += dt

        # Bắt đầu prefetch đoạn 2 (nếu có) TRONG LÚC đoạn 1 sắp/đang phát.
        prefetch_result: dict[str, object] = {}
        prefetch_thread: threading.Thread | None = None

        def _prefetch(index: int, text: str) -> None:
            try:
                audio_, dt_ = self._synthesize_chunk(index, text, total)
                prefetch_result["audio"] = audio_
                prefetch_result["dt"] = dt_
            except Exception as exc:  # noqa: BLE001 - đọc lại ở luồng chính
                prefetch_result["error"] = exc

        if total > 1:
            prefetch_thread = threading.Thread(
                target=_prefetch, args=(1, chunks[1]), daemon=True
            )
            prefetch_thread.start()

        for i in range(total):
            if i > 0:
                # Đoạn 2 trở đi: audio đã (hoặc đang) được prefetch ở
                # thread nền khi đoạn trước phát - đợi nó xong (thường đã
                # xong sẵn vì thời gian phát 1 câu thường > thời gian TTS
                # 1 câu ngắn kế tiếp).
                assert prefetch_thread is not None
                prefetch_thread.join()
                if "error" in prefetch_result:
                    exc = prefetch_result["error"]
                    logger.error("[TTS] chunk %d/%d tổng hợp thất bại: %s", i + 1, total, exc)
                    if not played_any:
                        # Vẫn chưa phát gì -> an toàn để fallback nguyên answer.
                        self._fallback_full_answer(answer)
                        return
                    # Đã phát 1 phần rồi - KHÔNG fallback toàn bộ answer
                    # (sẽ đọc LẶP phần đã phát) - chỉ bỏ qua đúng đoạn lỗi
                    # này, các đoạn sau vẫn tiếp tục xử lý bình thường.
                    prefetch_result.clear()
                    if i + 1 < total:
                        prefetch_thread = threading.Thread(
                            target=_prefetch, args=(i + 1, chunks[i + 1]), daemon=True
                        )
                        prefetch_thread.start()
                    continue

                current_audio = prefetch_result["audio"]  # type: ignore[assignment]
                total_synth_time += prefetch_result["dt"]  # type: ignore[arg-type]
                prefetch_result = {}

                # Bắt đầu prefetch đoạn KẾ TIẾP nữa (nếu còn) trước khi
                # phát đoạn hiện tại, để nó chạy song song lúc speaker
                # đang phát đoạn hiện tại.
                if i + 1 < total:
                    prefetch_thread = threading.Thread(
                        target=_prefetch, args=(i + 1, chunks[i + 1]), daemon=True
                    )
                    prefetch_thread.start()

            if not ttfa_logged:
                logger.info(
                    "[VOICE] time to first audio: %.2fs",
                    time.perf_counter() - pipeline_start,
                )
                ttfa_logged = True

            logger.info("[SPEAKER] chunk %d/%d playback started", i + 1, total)
            play_start = time.perf_counter()
            try:
                self._speaker.play(current_audio)  # type: ignore[arg-type]
                played_any = True
            except Exception as exc:
                logger.error("[SPEAKER] chunk %d/%d playback thất bại: %s", i + 1, total, exc)
                continue
            play_dt = time.perf_counter() - play_start
            total_play_time += play_dt
            logger.info(
                "[SPEAKER] chunk %d/%d playback completed in %.2fs", i + 1, total, play_dt
            )

        total_pipeline_time = time.perf_counter() - pipeline_start
        logger.info("[TTS] total synthesis time: %.2fs", total_synth_time)
        logger.info("[SPEAKER] total playback time: %.2fs", total_play_time)
        logger.info("[VOICE] total TTS pipeline time: %.2fs", total_pipeline_time)

    def listen_once(self, duration: float) -> str:
        pipeline_start = time.perf_counter()

        logger.info("Recording...")
        step_start = time.perf_counter()
        try:
            audio = self._microphone.record(duration)
        except Exception as exc:
            logger.exception("VoiceChat lỗi ở bước Microphone.record()")
            raise RuntimeError("VoiceChat: lỗi ở bước ghi âm (Microphone)") from exc
        logger.info("Recording completed in %.2fs", time.perf_counter() - step_start)

        logger.info("Transcribing...")
        step_start = time.perf_counter()
        try:
            raw_text = self._stt.transcribe(audio)
        except Exception as exc:
            logger.exception("VoiceChat lỗi ở bước SpeechToText.transcribe()")
            raise RuntimeError("VoiceChat: lỗi ở bước nhận dạng giọng nói (STT)") from exc
        logger.info("STT completed in %.2fs", time.perf_counter() - step_start)
        self._debug_text("RAW", raw_text)

        logger.info("Normalizing...")
        step_start = time.perf_counter()
        try:
            normalized_text = self._normalizer.normalize(raw_text)
        except Exception as exc:
            logger.exception("VoiceChat lỗi ở bước TextNormalizer.normalize()")
            raise RuntimeError("VoiceChat: lỗi ở bước chuẩn hóa text") from exc
        logger.info("Normalize completed in %.2fs", time.perf_counter() - step_start)
        self._debug_text("NORMALIZED", normalized_text)

        if not normalized_text.strip():
            logger.info("Không phát hiện giọng nói, bỏ qua lượt hội thoại.")
            logger.info("Pipeline completed in %.2fs", time.perf_counter() - pipeline_start)
            return ""

        logger.info("Generating answer...")
        step_start = time.perf_counter()
        try:
            answer = ask(normalized_text)
        except Exception as exc:
            logger.exception("VoiceChat lỗi ở bước chatbot.ask()")
            raise RuntimeError("VoiceChat: lỗi ở bước sinh câu trả lời (chatbot)") from exc
        logger.info("LLM completed in %.2fs", time.perf_counter() - step_start)

        logger.info("Synthesizing and Playing (sentence-level)...")
        playback_start = time.perf_counter()
        try:
            self._speak_answer(answer)
        except Exception as exc:
            logger.exception("VoiceChat lỗi ở bước TTS/Speaker")
            raise RuntimeError("VoiceChat: lỗi ở bước tổng hợp hoặc phát âm thanh") from exc
        logger.info("[SPEAKER] playback completed in %.2fs", time.perf_counter() - playback_start)

        logger.info("Pipeline completed in %.2fs", time.perf_counter() - pipeline_start)
        return answer

    def listen_once_interactive(self) -> str:
        pipeline_start = time.perf_counter()

        step_start = time.perf_counter()
        try:
            audio = self._microphone.record_on_enter()
        except Exception as exc:
            logger.exception("VoiceChat lỗi ở bước Microphone.record_on_enter()")
            raise RuntimeError("VoiceChat: lỗi ở bước ghi âm (Microphone)") from exc
        logger.info("Recording completed in %.2fs", time.perf_counter() - step_start)

        if audio.size == 0:
            logger.info("Không thu được audio nào, bỏ qua lượt hội thoại.")
            logger.info("Pipeline completed in %.2fs", time.perf_counter() - pipeline_start)
            return ""

        logger.info("Transcribing...")
        step_start = time.perf_counter()
        try:
            raw_text = self._stt.transcribe(audio)
        except Exception as exc:
            logger.exception("VoiceChat lỗi ở bước SpeechToText.transcribe()")
            raise RuntimeError("VoiceChat: lỗi ở bước nhận dạng giọng nói (STT)") from exc
        logger.info("STT completed in %.2fs", time.perf_counter() - step_start)
        self._debug_text("RAW", raw_text)

        logger.info("Normalizing...")
        step_start = time.perf_counter()
        try:
            normalized_text = self._normalizer.normalize(raw_text)
        except Exception as exc:
            logger.exception("VoiceChat lỗi ở bước TextNormalizer.normalize()")
            raise RuntimeError("VoiceChat: lỗi ở bước chuẩn hóa text") from exc
        logger.info("Normalize completed in %.2fs", time.perf_counter() - step_start)
        self._debug_text("NORMALIZED", normalized_text)

        if not normalized_text.strip():
            logger.info("Không phát hiện giọng nói, bỏ qua lượt hội thoại.")
            logger.info("Pipeline completed in %.2fs", time.perf_counter() - pipeline_start)
            return ""

        logger.info("Generating answer...")
        step_start = time.perf_counter()
        try:
            answer = ask(normalized_text)
        except Exception as exc:
            logger.exception("VoiceChat lỗi ở bước chatbot.ask()")
            raise RuntimeError("VoiceChat: lỗi ở bước sinh câu trả lời (chatbot)") from exc
        logger.info("LLM completed in %.2fs", time.perf_counter() - step_start)

        logger.info("Synthesizing and Playing (sentence-level)...")
        playback_start = time.perf_counter()
        try:
            self._speak_answer(answer)
        except Exception as exc:
            logger.exception("VoiceChat lỗi ở bước TTS/Speaker")
            raise RuntimeError("VoiceChat: lỗi ở bước tổng hợp hoặc phát âm thanh") from exc
        logger.info("[SPEAKER] playback completed in %.2fs", time.perf_counter() - playback_start)

        logger.info("Pipeline completed in %.2fs", time.perf_counter() - pipeline_start)
        return answer
