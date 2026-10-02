"""
voice/tts_text_prep.py
Chuẩn bị text CHO TTS (KHÔNG đụng tới `answer` gốc hiển thị màn hình).

Đây là bước đứng giữa chatbot.ask() và TextToSpeech.synthesize():
    answer (từ chatbot.ask(), giữ NGUYÊN cho màn hình)
        -> prepare_tts_text(answer) -> tts_text (CHỈ dùng để đọc)
        -> TextToSpeech.synthesize() gọi hàm này bên trong.

GIỚI HẠN VÀ NGUYÊN TẮC AN TOÀN:
- KHÔNG sửa nội dung nghiệp vụ (không đổi giá, size, quantity, tên món).
- Bỏ markdown / emoji.
- Rút gọn "đơn giá / giá mỗi ly" và "thành tiền / tổng tiền / tổng cộng" thành 1 thông tin giá
  CHỈ KHI 2 giá trị số thực sự GIỐNG NHAU (quantity = 1).
- GIỮ NGUYÊN cả hai số khi chúng KHÁC NHAU (X != Y, vd: quantity > 1 hoặc giỏ đã có món).
- Chuyển số tiền có kèm đơn vị tiền tệ sang chữ tiếng Việt.
- Bỏ câu CTA mời gọi cuối câu nếu là câu hỏi không chứa số.
"""
from __future__ import annotations
import re

# 1. MARKDOWN / EMOJI
_MD_BOLD_ITALIC_RE = re.compile(r"(\*\*\*|\*\*|\*|___|__|_)(.+?)\1")
_MD_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_MD_BULLET_RE = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
_MD_CODE_RE = re.compile(r"`([^`]*)`")
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "\u2190-\u21FF\u2300-\u23FF"
    "]+",
    flags=re.UNICODE,
)

def _strip_markdown(text: str) -> str:
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _MD_CODE_RE.sub(r"\1", text)
    text = _MD_BOLD_ITALIC_RE.sub(r"\2", text)
    text = _MD_HEADING_RE.sub("", text)
    text = _MD_BULLET_RE.sub("", text)
    return text

def _strip_emoji(text: str) -> str:
    return _EMOJI_RE.sub("", text)

# 2. RÚT GỌN KHI ĐƠN GIÁ = THÀNH TIỀN (X == Y) & GIỮ NGUYÊN KHI X != Y
_CURRENCY_UNIT = r"(?:VNĐ|VND|đồng|đ)"
_AMOUNT = r"\d{1,3}(?:[.,]\d{3})*"

# Nhận diện các cách diễn đạt:
# 1) Đơn giá / Giá mỗi ly / Giá một ly: X VNĐ
# 2) Thành tiền / Tổng tiền / Tổng cộng: Y VNĐ
_UNIT_PRICE_LABEL = r"(?:đơn\s*giá|giá\s*mỗi\s*(?:ly|cốc|phần|món)|giá\s*một\s*(?:ly|cốc|phần|món)|giá\s*1\s*(?:ly|cốc|phần|món))"
_TOTAL_PRICE_LABEL = r"(?:thành\s*tiền|tổng\s*tiền|tổng\s*cộng)"

_COLLAPSE_PRICE_RE = re.compile(
    rf"(?P<label1>{_UNIT_PRICE_LABEL})\s*(?::|\blà\b)?\s*"
    rf"(?P<amt1>{_AMOUNT})\s*(?P<unit1>{_CURRENCY_UNIT})\b(?!\w)"
    rf"(?P<sep>[,\s.\n]*?)"
    rf"(?P<label2>{_TOTAL_PRICE_LABEL})\s*(?::|\blà\b)?\s*"
    rf"(?P<amt2>{_AMOUNT})\s*(?P<unit2>{_CURRENCY_UNIT})\b(?!\w)",
    re.IGNORECASE,
)

def _collapse_duplicate_price(text: str) -> str:
    def _repl(m: re.Match) -> str:
        raw1 = m.group("amt1").replace(".", "").replace(",", "")
        raw2 = m.group("amt2").replace(".", "").replace(",", "")
        # CHỈ rút gọn khi 2 số giống hệt nhau (Quantity = 1)
        if raw1 == raw2:
            return f"giá {m.group('amt1')} {m.group('unit1')}"
        # Khác nhau (X != Y) -> Giữ nguyên toàn bộ cấu trúc ban đầu
        return m.group(0)

    return _COLLAPSE_PRICE_RE.sub(_repl, text)

# Nối vế "giá X" đứng ngay sau dấu câu hoặc xuống dòng bằng dấu phẩy
_PERIOD_BEFORE_GIA_RE = re.compile(r"[.\n]+\s*(giá\s)", re.IGNORECASE)

def _join_price_clause(text: str) -> str:
    return _PERIOD_BEFORE_GIA_RE.sub(r", \1", text)

# 3. BỎ CÂU MỜI GỌI (CTA) Ở CUỐI
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

def _strip_trailing_cta(text: str) -> str:
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s]
    while sentences:
        last = sentences[-1].strip()
        is_question = last.endswith("?")
        has_digit = any(ch.isdigit() for ch in last)
        if is_question and not has_digit:
            sentences.pop()
            continue
        break
    return " ".join(sentences)

# 4. ĐỌC TIỀN TIẾNG VIỆT
_ONES = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
_GROUP_UNITS = ["", " nghìn", " triệu", " tỷ"]

def _read_3digit_group(n: int, is_leading_group: bool) -> str:
    hundreds, rem = divmod(n, 100)
    tens, units = divmod(rem, 10)

    parts = []
    if hundreds != 0:
        parts.append(f"{_ONES[hundreds]} trăm")
    elif not is_leading_group and rem != 0:
        parts.append("không trăm")

    if tens == 0:
        if units != 0 and (hundreds != 0 or not is_leading_group):
            parts.append(f"linh {_ONES[units]}")
        elif units != 0:
            parts.append(_ONES[units])
    elif tens == 1:
        parts.append("mười")
        if units == 1:
            parts.append("một")
        elif units == 5:
            parts.append("lăm")
        elif units != 0:
            parts.append(_ONES[units])
    else:
        parts.append(f"{_ONES[tens]} mươi")
        if units == 1:
            parts.append("mốt")
        elif units == 4:
            parts.append("tư")
        elif units == 5:
            parts.append("lăm")
        elif units != 0:
            parts.append(_ONES[units])

    return " ".join(parts)

def _number_to_vietnamese_words(value: int) -> str:
    if value == 0:
        return "không"

    groups = []
    n = value
    while n > 0:
        groups.append(n % 1000)
        n //= 1000

    words = []
    for i in range(len(groups) - 1, -1, -1):
        group_value = groups[i]
        if group_value == 0:
            continue
        is_leading = i == len(groups) - 1
        words.append(_read_3digit_group(group_value, is_leading) + _GROUP_UNITS[i])

    return " ".join(words).strip()

_MONEY_RE = re.compile(
    rf"(?P<amount>{_AMOUNT})\s*(?P<unit>{_CURRENCY_UNIT})\b(?!\w)",
    re.IGNORECASE,
)

def _replace_money(text: str) -> str:
    def _repl(m: re.Match) -> str:
        raw = m.group("amount").replace(".", "").replace(",", "")
        try:
            value = int(raw)
        except ValueError:
            return m.group(0)
        return f"{_number_to_vietnamese_words(value)} đồng"

    return _MONEY_RE.sub(_repl, text)

# ENTRYPOINT
def prepare_tts_text(text: str) -> str:
    if not isinstance(text, str) or text.strip() == "":
        return text

    result = text
    result = _strip_markdown(result)
    result = _strip_emoji(result)
    result = _collapse_duplicate_price(result)
    result = _join_price_clause(result)
    result = _strip_trailing_cta(result)
    result = _replace_money(result)
    result = re.sub(r"\s+", " ", result).strip()
    return result