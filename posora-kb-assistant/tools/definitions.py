"""LangChain tool definitions.
This module is the adapter between tool-calling and domain services. Business logic remains in the domain modules.
"""
from __future__ import annotations
import time
from langchain_core.tools import tool
from .cart import (OrderItem, add_single_item, cart_summary, remove_item, reset_cart, update_item)
from .pricing import compute_total
from .response import success_response

@tool
def add_to_cart(items: list[OrderItem]) -> str:
    """Thêm một hoặc nhiều món vào giỏ hàng."""
    results = [
        add_single_item(
            product_name=item.product_name,
            size=item.size,
            quantity=item.quantity,
            customizations=item.customizations,
        )
        for item in items
    ]
    return success_response(results=results, cart=cart_summary())

@tool
def view_cart() -> str:
    """Xem giỏ hàng hiện tại."""
    return success_response(**cart_summary())

@tool
def remove_from_cart(product_name: str, size: str) -> str:
    """Xoá một dòng món theo tên và size.
    Nếu cùng tên+size có nhiều customization khác nhau, domain service sẽ trả ambiguous_cart_item thay vì tự chọn.
    """
    removed, error = remove_item(product_name, size)

    if error is not None:
        from .response import error_response
        return error_response(
            error["error_type"],
            error["message"],
            **{
                key: value
                for key, value in error.items()
                if key not in {"order_status", "error_type", "message"}
            },
        )
    return success_response(removed=removed.to_dict(), cart=cart_summary())

@tool
def update_cart(
    product_name: str,
    current_size: str | None = None,
    new_size: str | None = None,
    new_quantity: int | None = None,
    change_quantity: int | None = None,
) -> str:
    """Cập nhật món đã có trong giỏ."""
    result = update_item(
        product_name=product_name,
        current_size=current_size,
        new_size=new_size,
        new_quantity=new_quantity,
        change_quantity=change_quantity,
    )

    from .response import error_response

    if result.get("order_status") == "error":
        return error_response(
            result["error_type"],
            result["message"],
            **{
                key: value
                for key, value in result.items()
                if key not in {"order_status", "error_type", "message"}
            },
        )

    return success_response(
        **{
            key: value
            for key, value in result.items()
            if key != "order_status"
        }
    )

@tool
def clear_cart() -> str:
    """Xoá toàn bộ giỏ hàng."""
    reset_cart()
    return success_response(message="Đã xoá toàn bộ giỏ hàng.", cart=cart_summary())

@tool
def checkout() -> str:
    """Chốt đơn hàng hiện tại."""
    from .response import error_response

    summary = cart_summary()
    if not summary["items"]:
        return error_response("empty_cart", "Giỏ hàng đang trống, chưa có món nào để chốt đơn.")

    totals = compute_total(summary)
    receipt = success_response(
        items=summary["items"],
        item_count=summary["item_count"],
        **totals,
        checked_out_at=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    reset_cart()
    return receipt

ALL_TOOLS = [
    add_to_cart,
    view_cart,
    remove_from_cart,
    update_cart,
    clear_cart,
    checkout,
]
