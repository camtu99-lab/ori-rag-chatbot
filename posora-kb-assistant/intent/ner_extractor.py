"""
-Bản chất Rule-based (không dùng Machine Learning): Sử dụng regex và lookup table thay vì mô hình deep learning/thống kê do domain hẹp 
(~40 món, size M/L, số lượng dạng số), giúp nhận diện chính xác và tránh cài thêm dependency.
-Tái sử dụng dữ liệu động (Dynamic Lookup): Không hardcode danh sách món/size trong file 
mà truy vấn trực tiếp từ data/menu.json qua tools.find_product() và tools.get_product_sizes().
-Khả năng mở rộng: Dễ dàng thay thế hoặc bổ sung mô hình NER thống kê khác (như spaCy/PhoBERT)
 qua interface EntityExtractor và extractor_factory.py mà không phải sửa logic hiện tại.
"""
from __future__ import annotations
import re
import unicodedata
from intent.extractor_base import EntityExtractor, ExtractedEntities, clean_entities
from tools.catalog import (_fold, _load_menu, _normalize, find_product, get_product_sizes)

try:
    from unidecode import unidecode
except ImportError:  # pragma: no cover - fallback giống tools.py/sparse_retriever.py
    def unidecode(text: str) -> str:
        text = text.replace("đ", "d").replace("Đ", "D")
        decomposed = unicodedata.normalize("NFD", text)
        return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")

# 1. INTENT CLASSIFICATION:
# - Khớp theo thứ tự ưu tiên (cụ thể -> chung) để tránh nhầm lẫn giữa các ý định đối lập.
# - Chuẩn hóa bỏ dấu sẵn 1 lần lúc nạp module để tối ưu hiệu năng.
# - Dùng regex word-boundary (\b...\b) thay vì 'in' để tránh khớp nhầm từ con (vd: "to" trong "tôi").
_INTENT_KEYWORDS_RAW: list[tuple[str, list[str]]] = [
    ("checkout", ["thanh toán", "tính tiền", "trả tiền", "checkout", "chốt đơn"]),
    ("remove_from_cart", ["xóa", "xoá", "bỏ", "hủy", "huỷ", "bớt", "giảm"]),
    ("view_cart", ["giỏ hàng", "đơn hàng", "tổng tiền", "hết bao nhiêu"]),
    ("add_to_cart", ["cho tôi", "thêm", "đặt", "gọi", "lấy", "order", "mua"]),
]
_INTENT_KEYWORDS: list[tuple[str, list[re.Pattern]]] = [
    (intent, [re.compile(rf"\b{re.escape(_fold(_normalize(kw)))}\b") for kw in keywords])
    for intent, keywords in _INTENT_KEYWORDS_RAW
]

def _classify_intent(text_norm: str) -> str:
    for intent, patterns in _INTENT_KEYWORDS:
        if any(p.search(text_norm) for p in patterns):
            return intent
    return "unknown"

# 2. QUANTITY - regex số (dạng số hoặc chữ số đếm tiếng Việt phổ biến).
# CHỈ cover các số đếm nhỏ thường gặp khi gọi đồ uống (1-10) - đủ dùng cho
# domain quán cà phê, không cần bộ chuyển số chữ->số tổng quát.
_NUMBER_WORDS_RAW = {
    "một": 1, "hai": 2, "ba": 3, "bốn": 4, "tư": 4, "năm": 5,
    "sáu": 6, "bảy": 7, "tám": 8, "chín": 9, "mười": 10,
}
_QUANTITY_DIGIT_RE = re.compile(r"\b(\d+)\s*(ly|phan|cai|suat|chai|o|tach)?\b")
_NUMBER_WORD_PATTERNS: list[tuple[re.Pattern, int]] = [
    (re.compile(rf"\b{re.escape(_fold(_normalize(word)))}\s*(ly|phan|cai|suat|chai|o|tach)\b"), value)
    for word, value in _NUMBER_WORDS_RAW.items()
]

def _extract_quantity(text_norm: str) -> int | None:
    # Ưu tiên số dạng chữ số ("2 ly") - rõ ràng, ít nhầm lẫn nhất.
    match = _QUANTITY_DIGIT_RE.search(text_norm)
    if match:
        return int(match.group(1))
    # Không có chữ số -> thử số đếm bằng chữ ("hai ly").
    for pattern, value in _NUMBER_WORD_PATTERNS:
        if pattern.search(text_norm):
            return value
    return None

# 3. SIZE EXTRACTION:
# - Ưu tiên nhận diện format tường minh ("size M/L"), sau đó mới ánh xạ từ mô tả ("to/lớn" -> L, "nhỏ/vừa" -> M).
# - Chỉ trích xuất ứng viên ban đầu; size thực tế sẽ được validate lại qua tools.get_product_sizes() sau khi xác định món.
_SIZE_LETTER_RE = re.compile(r"\bsize\s*([a-z]+)\b", re.IGNORECASE)
_SIZE_WORDS_RAW = {"lớn": "L", "to": "L", "nhỏ": "M", "vừa": "M"}
# Ánh xạ từ khóa mô tả (đã bỏ dấu) -> mã size chuẩn (M/L).
# Hỗ trợ cả 2 dạng: đi kèm tiền tố ("size lớn") hoặc đứng độc lập ("cỡ lớn", "ly lớn").
_SIZE_WORD_MAP: dict[str, str] = {_fold(_normalize(word)): size for word, size in _SIZE_WORDS_RAW.items()}
_SIZE_WORD_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(rf"\b{re.escape(folded)}\b"), size) for folded, size in _SIZE_WORD_MAP.items()
]

def _extract_size_candidate(text_norm: str) -> str | None:
    match = _SIZE_LETTER_RE.search(text_norm)
    if match:
        token = match.group(1).strip()
        # Token sau "size" có thể là TỪ MÔ TẢ ("size lớn") thay vì mã chữ
        # cái trực tiếp ("size L") - kiểm tra map trước, chỉ coi là mã chữ cái nếu KHÔNG khớp từ mô tả nào.
        if token in _SIZE_WORD_MAP:
            return _SIZE_WORD_MAP[token]
        return token.upper()
    for pattern, size in _SIZE_WORD_PATTERNS:
        if pattern.search(text_norm):
            return size
    # Trường hợp khách chỉ nói mỗi chữ cái size, không có từ "size" đứng
    # trước (vd. "...cỡ L", "...1 ly L") - bắt riêng chữ M/L đứng độc lập.
    bare = re.search(r"\b(m|l)\b", text_norm, re.IGNORECASE)
    if bare:
        return bare.group(1).upper()
    return None

# 4. PRODUCT EXTRACTION:
# - Quét danh sách món động từ menu.json (không hardcode) sau khi chuẩn hóa bỏ dấu.
# - Dùng chiến lược Longest Match First: Ưu tiên tên món dài/cụ thể nhất khi có nhiều món cùng khớp.
def _extract_product_name(text: str) -> str | None:
    text_folded = _fold(_normalize(text))
    menu = _load_menu()

    candidates = [
        item["name"]
        for item in menu
        if _fold(_normalize(item["name"])) in text_folded
    ]
    if not candidates:
        return None
    return max(candidates, key=len)

class NERExtractor(EntityExtractor):
    """Trích xuất intent/entity bằng regex + rule-based, không gọi LLM."""

    def extract(self, text: str) -> ExtractedEntities:
        text_norm = _fold(_normalize(text))

        intent = _classify_intent(text_norm)
        quantity = _extract_quantity(text_norm)
        product_name = _extract_product_name(text)

        size_candidate = _extract_size_candidate(text_norm)
        size = self._validate_size(product_name, size_candidate)

        return ExtractedEntities(
            intent=intent,
            entities=clean_entities({
                "product_name": product_name,
                "size": size,
                "quantity": quantity,
            }),
        )

    @staticmethod
    def _validate_size(product_name: str | None, size_candidate: str | None) -> str | None:
        """Xác thực size_candidate dựa trên danh sách size thực tế từ tools.get_product_sizes(product) (không hardcode).
        Trả về None nếu không tìm thấy món hoặc size không hợp lệ với món đó.
        """
        if size_candidate is None or product_name is None:
            return size_candidate  # chưa xác nhận được, giữ nguyên phỏng đoán ban đầu

        match = find_product(product_name)
        if match.product is None:
            return size_candidate

        valid_sizes = get_product_sizes(match.product)
        return size_candidate if size_candidate in valid_sizes else None
