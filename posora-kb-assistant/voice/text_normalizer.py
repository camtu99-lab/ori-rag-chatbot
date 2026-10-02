"""
voice/text_normalizer.py
Tầng NGÔN NGỮ, đứng GIỮA SpeechToText và chatbot.ask():

    Audio -> stt.py -> Text RAW -> [text_normalizer.py] -> chatbot.ask()

Trách nhiệm DUY NHẤT: làm sạch text theo quy tắc NGÔN NGỮ CHUNG của tiếng
Việt (khoảng trắng, unicode, dấu câu, filler word, chữ hoa/thường bất
thường) - KHÔNG diễn giải nghĩa, KHÔNG biết gì về nghiệp vụ quán DMP.

File này KHÔNG import và KHÔNG được phép biết tới:
- chatbot.py, memory.py, rag/*, tools.py, tool_executor.py, llm.py
- bất kỳ dữ liệu nghiệp vụ nào (menu, promotions, tên khách hàng thân
  thiết...). Nếu đổi toàn bộ dataset RAG, module này vẫn phải hoạt động
  y hệt, vì nó không hề biết dataset RAG là gì.

Vì vậy tuyệt đối KHÔNG có mapping kiểu "caphe" -> "cà phê muối" hay
"latte" -> "latte đá" trong file này - đó là kiến thức nghiệp vụ, thuộc
về RAG/prompt, không phải về ngôn ngữ.
"""
from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)


class TextNormalizer:
    """Chuẩn hóa text RAW từ STT thành text sạch, sẵn sàng đưa vào
    chatbot.ask(). Chỉ xử lý NGÔN NGỮ (whitespace, unicode, dấu câu,
    filler word, chữ hoa/thường bất thường do STT) - không diễn giải
    nghĩa, không biết gì về nghiệp vụ.

    Không có state - mỗi lần gọi `normalize()` độc lập hoàn toàn với các
    lần gọi trước, có thể tạo instance mới bất cứ đâu mà không lo tác
    dụng phụ. Các hằng số dùng chung (regex đã compile, danh sách filler
    word) là hằng số CHỈ ĐỌC ở cấp class, không phải state có thể thay đổi.
    """

    # Các từ đệm/hư từ khi nói (không mang nghĩa, không phải nghiệp vụ) -
    # đây là danh sách NGÔN NGỮ CHUNG của tiếng Việt khi nói, áp dụng cho
    # BẤT KỲ hội thoại nào, không riêng gì quán cà phê DMP. Cố tình KHÔNG
    # đưa "dạ"/"vâng"/"ừ" các từ xác nhận có/không vào đây vì chúng có thể
    # mang nghĩa (đồng ý) mà chatbot cần biết - chỉ loại các âm đệm thuần
    # túy không mang thông tin.
    _FILLER_WORDS = (
        "à",
        "ơ",
        "ê",
        "ừm",
        "ừ",
        "ờ",
        "ừa",
        "ấy",
    )

    # Compile 1 lần ở cấp class (hằng số chỉ đọc, không phải singleton hay
    # global mutable state) để tránh compile lại regex mỗi lần gọi normalize().
    _RE_CRLF = re.compile(r"\r\n?")
    _RE_WHITESPACE = re.compile(r"\s+")
    _RE_REPEATED_PUNCTUATION = re.compile(r"([!?,.;:])\1+")
    _RE_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([!?,.;:])")
    _RE_FILLER_WORDS = re.compile(
        r"\b(?:" + "|".join(re.escape(word) for word in _FILLER_WORDS) + r")\b",
        re.IGNORECASE | re.UNICODE,
    )

    def normalize(self, text: str) -> str:
        """Chuẩn hóa 1 đoạn text RAW từ STT thành text sạch.

        Thứ tự xử lý (mỗi bước chỉ làm đúng 1 việc, đều là chuẩn hóa
        NGÔN NGỮ, không có bước nào diễn giải nghĩa hay đụng dữ liệu
        nghiệp vụ):
            1. Chuẩn hóa Unicode (NFC) - STT/microphone có thể trả về
               tiếng Việt ở dạng tổ hợp dấu không nhất quán.
            2. Chuẩn hóa newline (CRLF/CR -> LF).
            3. Gộp khoảng trắng thừa (bao gồm newline) thành 1 space, trim
               2 đầu.
            4. Gộp dấu câu lặp ("!!!" -> "!") và bỏ khoảng trắng thừa
               trước dấu câu.
            5. Bỏ các từ đệm/hư từ phổ biến khi nói (à, ừm, ờ...).
            6. Nếu toàn bộ text là CHỮ HOA (artifact thường gặp của STT),
               chuyển về chữ thường.
            7. Gộp khoảng trắng thừa lần cuối (bước 5 có thể để lại 2
               space liền nhau) và trim.

        Args:
            text: text RAW nhận được từ SpeechToText.transcribe().

        Returns:
            Text đã chuẩn hóa ngôn ngữ, sẵn sàng truyền cho
            `chatbot.ask()`. Trả về chuỗi rỗng "" nếu sau khi chuẩn hóa
            không còn nội dung gì (vd. input chỉ toàn khoảng trắng/filler
            word).

        Raises:
            TypeError: nếu `text` không phải kiểu `str`.
            RuntimeError: nếu có lỗi không lường trước trong lúc xử lý.
        """
        if not isinstance(text, str):
            raise TypeError(f"text phải là str, nhận được: {type(text)!r}")

        if text == "":
            logger.debug("Input rỗng, không có gì để normalize")
            return ""

        try:
            result = self._normalize_unicode(text)
            result = self._normalize_newlines(result)
            result = self._collapse_whitespace(result)
            result = self._normalize_punctuation(result)
            result = self._remove_filler_words(result)
            result = self._normalize_shouting_case(result)
            result = self._collapse_whitespace(result)
        except Exception as exc:
            logger.exception("Lỗi không lường trước khi normalize text")
            raise RuntimeError("Không thể normalize text") from exc

        if not result:
            logger.warning(
                "Text sau khi normalize rỗng (raw: %d ký tự, có thể chỉ "
                "gồm khoảng trắng hoặc filler word)",
                len(text),
            )
        else:
            logger.debug(
                "Normalize xong: %d ký tự -> %d ký tự", len(text), len(result)
            )

        return result

    @classmethod
    def _normalize_unicode(cls, text: str) -> str:
        """Chuẩn hóa Unicode về dạng tổ hợp NFC - đảm bảo cùng 1 ký tự
        tiếng Việt (vd. 'à') luôn được biểu diễn nhất quán, bất kể STT
        trả về dạng dựng sẵn hay dạng tổ hợp base+dấu."""
        return unicodedata.normalize("NFC", text)

    @classmethod
    def _normalize_newlines(cls, text: str) -> str:
        """Chuẩn hóa mọi kiểu xuống dòng (CRLF, CR) về LF, để bước gộp
        khoảng trắng phía sau xử lý nhất quán."""
        return cls._RE_CRLF.sub("\n", text)

    @classmethod
    def _collapse_whitespace(cls, text: str) -> str:
        """Gộp mọi chuỗi khoảng trắng liên tiếp (space, tab, newline)
        thành đúng 1 space, đồng thời trim 2 đầu chuỗi."""
        return cls._RE_WHITESPACE.sub(" ", text).strip()

    @classmethod
    def _normalize_punctuation(cls, text: str) -> str:
        """Gộp dấu câu lặp lại liên tiếp ("!!!" -> "!", "??" -> "?") và
        loại bỏ khoảng trắng thừa đứng ngay trước dấu câu ("cà phê !" ->
        "cà phê!"). Đây là chuẩn hóa dấu câu thuần ngôn ngữ, không đụng
        tới nội dung chữ."""
        text = cls._RE_REPEATED_PUNCTUATION.sub(r"\1", text)
        text = cls._RE_SPACE_BEFORE_PUNCTUATION.sub(r"\1", text)
        return text

    @classmethod
    def _remove_filler_words(cls, text: str) -> str:
        """Loại bỏ các từ đệm/hư từ phổ biến khi nói tiếng Việt (à, ừm,
        ờ...) - danh sách NGÔN NGỮ CHUNG, không phải mapping nghiệp vụ."""
        return cls._RE_FILLER_WORDS.sub("", text)

    @classmethod
    def _normalize_shouting_case(cls, text: str) -> str:
        """Nếu TOÀN BỘ chữ cái trong text đều viết hoa (không có bất kỳ
        chữ thường nào) thì coi đây là artifact của STT/mic gain quá cao,
        chuyển toàn bộ về chữ thường. Text có trộn hoa/thường (vd. tên
        riêng viết hoa chữ đầu) sẽ được giữ nguyên, vì đó không phải dấu
        hiệu bất thường.

        `str.isupper()` trả về True chỉ khi có ít nhất 1 ký tự có thể
        phân biệt hoa/thường và TẤT CẢ ký tự đó đều viết hoa - nên không
        cần kiểm tra thêm "có chữ cái hay không" một cách thủ công."""
        if text.isupper():
            return text.lower()
        return text
