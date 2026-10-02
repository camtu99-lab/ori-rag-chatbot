"""Composition Root cho hệ thống Context Management:
- Nguyên tắc phân tách: Giữ generic package `context_management` hoàn toàn độc lập, không phụ thuộc trực tiếp vào business prompt hay infrastructure cụ thể.
- Dependency Injection: Đóng vai trò điểm tập trung duy nhất để khởi tạo, cấu hình và tiêm các dependency thực thi vào luồng xử lý context.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, replace
from typing import Callable

logger = logging.getLogger(__name__)

@dataclass
class PendingOrder:
    """1 yêu cầu add_to_cart đang chờ đủ thông tin (product_name/size/
    quantity) qua nhiều lượt chat. `intent` hiện luôn là "add_to_cart" (xem
    docstring đầu file) - vẫn giữ field này để log/tương lai dễ mở rộng,
    """
    intent: str = "add_to_cart"
    product_name: str | None = None
    size: str | None = None
    quantity: int | None = None

_pending: PendingOrder | None = None


def get_pending() -> PendingOrder | None:
    """Trả về pending order hiện tại, hoặc None nếu không có."""
    return _pending

def is_pending_active() -> bool:
    return _pending is not None

def _log_pending(pending: PendingOrder | None) -> None:
    logger.info("[PENDING]")
    if pending is None:
        logger.info("(cleared)")
        return
    logger.info("intent = %s", pending.intent)
    logger.info("product = %s", pending.product_name)
    logger.info("size = %s", pending.size)
    logger.info("quantity = %s", pending.quantity)

def start_pending(
    product_name: str | None, size: str | None, quantity: int | None
) -> PendingOrder:
    """Bắt đầu 1 pending order MỚI cho add_to_cart (ghi đè pending cũ nếu
    có sẵn - gọi hàm này nghĩa là nơi gọi đã xác định đây là 1 yêu cầu
    add_to_cart MỚI, không phải tiếp tục cái cũ; việc huỷ pending cũ trước
    khi gọi hàm này là trách nhiệm của nơi gọi, xem chatbot.py)."""
    global _pending
    _pending = PendingOrder(product_name=product_name, size=size, quantity=quantity)
    _log_pending(_pending)
    return _pending

def update_pending(
    product_name: str | None = None,
    size: str | None = None,
    quantity: int | None = None,
) -> PendingOrder:
    """Merge các field MỚI trích xuất được (khác None) vào pending hiện
    có. CHỈ ghi đè field nào nơi gọi thực sự truyền vào khác None - field
    không được nhắc tới ở lượt này giữ NGUYÊN giá trị cũ, không suy diễn gì
    thêm (giá trị cũ đó vốn cũng do chính user nói ra ở lượt trước, không
    phải suy diễn mới - đúng yêu cầu "không tự động lấy thông tin mà user
    không nói rõ")."""
    global _pending
    if _pending is None:
        # Không có pending để cập nhật - coi như bắt đầu mới với đúng
        # những gì vừa trích xuất được ở lượt này.
        return start_pending(product_name, size, quantity)
    _pending = replace(
        _pending,
        product_name=product_name if product_name is not None else _pending.product_name,
        size=size if size is not None else _pending.size,
        quantity=quantity if quantity is not None else _pending.quantity,
    )
    _log_pending(_pending)
    return _pending

def clear_pending() -> None:
    """Xoá pending order hiện tại (không làm gì nếu vốn đã không có)."""
    global _pending
    if _pending is not None:
        _pending = None
        _log_pending(None)

def missing_slot_question(
    pending: PendingOrder,
    find_product: Callable[[str], object],
    get_product_sizes: Callable[[dict], dict],
) -> str:
    """Sinh câu hỏi tiếp theo dựa trên slot nào CÒN THIẾU trong `pending`.

    Nhận `find_product`/`get_product_sizes` qua tham số (thay vì tự import
    tools.py ở top-level) để module này không bắt buộc phải biết chi tiết
    tools.py implement thế nào - chỉ cần đúng chữ ký 2 hàm đã có sẵn ở đó
    (dependency injection, không phải kiến trúc mới - vẫn dùng đúng
    ProductMatch/get_product_sizes hiện có, không viết lại logic tìm món).

    Quantity KHÔNG BAO GIỜ là slot chặn câu hỏi: đã có sẵn quy tắc nghiệp
    vụ "không nói số lượng thì mặc định 1" (xem
    chatbot._build_tool_call_args) - nên hàm này chỉ xét product_name và
    size, đúng 2 slot thực sự bắt buộc phải hỏi lại khách.
    """
    product = None
    if pending.product_name:
        match = find_product(pending.product_name)
        if match.status not in ("not_found", "ambiguous"):
            product = match.product

    if product is None:
        # Chưa có product HỢP LỆ (chưa nói, hoặc nói nhưng không khớp món
        # nào/mơ hồ) -> vẫn cần hỏi lại tên món.
        if pending.size is None:
            return "Bạn muốn thêm món gì và size nào?"
        return "Bạn muốn thêm món gì?"

    # Đã có product hợp lệ -> chỉ còn thiếu size.
    sizes = sorted(get_product_sizes(product).keys())
    if len(sizes) >= 2:
        return f"Bạn muốn size {' hay '.join(sizes)}?"
    return "Bạn muốn size nào?"
