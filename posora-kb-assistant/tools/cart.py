"""Shopping cart domain service.
No LangChain/MCP code here. The cart stores required order fields explicitly
and keeps all optional NER modifiers in a generic ``customizations`` mapping.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from pydantic import BaseModel, Field
from .catalog import _normalize, find_product, get_product_sizes
from .response import error_dict

EntityCustomizations = dict[str, Any]

def customization_field():
    """Schema marker used by ToolArgumentBuilder as the generic entity sink."""
    return Field(
        default_factory=dict,
        description="Các tùy chọn chỉ xuất hiện khi người dùng nói rõ.",
        json_schema_extra={"x-entity-sink": True},
    )

@dataclass
class CartLine:
    product_name: str
    size: str
    unit_price: int | float
    quantity: int
    customizations: EntityCustomizations = field(default_factory=dict)

    @property
    def line_total(self) -> int | float:
        return self.unit_price * self.quantity

    def to_dict(self) -> dict:
        return {
            "product_name": self.product_name,
            "size": self.size,
            "unit_price": self.unit_price,
            "quantity": self.quantity,
            "customizations": dict(self.customizations),
            "line_total": self.line_total,
        }

class OrderItem(BaseModel):
    product_name: str = Field(description="Tên món khách đã chọn.")
    size: str = Field(description="Size hợp lệ của món.")
    quantity: int = Field(description="Số lượng, số nguyên dương.", gt=0)
    customizations: EntityCustomizations = customization_field()

# Customizations are part of cart-line identity so two otherwise-equal drinks
# with different notes do not get merged accidentally.
_cart: dict[tuple[str, str, tuple[tuple[str, str], ...]], CartLine] = {}

def reset_cart() -> None:
    global _cart
    _cart = {}

def _freeze_customizations(customizations: EntityCustomizations | None) -> tuple[tuple[str, str], ...]:
    if not customizations:
        return ()
    return tuple(
        sorted(
            (str(key), repr(value))
            for key, value in customizations.items()
            if value is not None
        )
    )

def _cart_key(product_name: str, size: str, customizations: EntityCustomizations | None = None) -> tuple[str, str, tuple[tuple[str, str], ...]]:
    return (
        _normalize(product_name),
        size.strip().upper(),
        _freeze_customizations(customizations),
    )

def cart_summary() -> dict:
    lines = [line.to_dict() for line in _cart.values()]
    subtotal = sum(line.line_total for line in _cart.values())
    return {
        "items": lines,
        "item_count": len(lines),
        "subtotal": subtotal,
        "currency": "VND",
    }

def add_single_item(product_name: str, size: str, quantity: int, customizations: EntityCustomizations | None = None) -> dict:
    if quantity <= 0:
        return error_dict("invalid_quantity", f"Số lượng '{quantity}' không hợp lệ, phải là số nguyên dương.", requested=product_name)

    match = find_product(product_name)

    if match.status == "not_found":
        return error_dict("invalid_product", f"Không tìm thấy món '{product_name}' trong menu.", requested=product_name, suggestions=match.suggestions)

    if match.status == "ambiguous":
        return error_dict("ambiguous_product", f"'{product_name}' khớp với nhiều món khác nhau, cần khách nói rõ hơn.", requested=product_name, suggestions=match.suggestions)

    product = match.product
    sizes = get_product_sizes(product)
    size_norm = size.strip().upper()

    if size_norm not in sizes:
        return error_dict("invalid_size", f"Món '{product['name']}' không có size '{size}'.", requested=product_name, valid_sizes=sorted(sizes.keys()))

    normalized_customizations = {
        str(key): value
        for key, value in (customizations or {}).items()
        if value is not None
    }

    key = _cart_key(product["name"], size_norm, normalized_customizations)
    unit_price = sizes[size_norm]

    if key in _cart:
        _cart[key].quantity += quantity
    else:
        _cart[key] = CartLine(
            product_name=product["name"],
            size=size_norm,
            unit_price=unit_price,
            quantity=quantity,
            customizations=normalized_customizations,
        )

    return {
        "order_status": "success",
        "product_name": product["name"],
        "size": size_norm,
        "quantity_added": quantity,
        "unit_price": unit_price,
        "customizations": normalized_customizations,
        "match_type": match.status,
    }

def _matching_keys(product_name: str, size: str | None = None) -> list[tuple[str, str, tuple[tuple[str, str], ...]]]:
    norm_name = _normalize(product_name)
    size_norm = size.strip().upper() if size is not None else None

    return [
        key
        for key in _cart
        if key[0] == norm_name
        and (size_norm is None or key[1] == size_norm)
    ]

def remove_item(product_name: str, size: str) -> tuple[CartLine | None, dict | None]:
    matches = _matching_keys(product_name, size)

    if not matches:
        match = find_product(product_name)
        return None, {
            "order_status": "error",
            "error_type": "not_in_cart",
            "message": f"Không tìm thấy '{product_name}' size '{size}' trong giỏ hàng.",
            "suggestions": match.suggestions,
            "cart": cart_summary(),
        }

    if len(matches) > 1:
        return None, error_dict(
            "ambiguous_cart_item",
            (
                f"'{product_name}' size '{size}' đang có nhiều cấu hình khác nhau "
                "trong giỏ hàng, cần nói rõ món muốn xoá."
            ),
            variants=[_cart[key].to_dict() for key in matches],
            cart=cart_summary(),
        )

    return _cart.pop(matches[0]), None

def locate_for_update(
    product_name: str,
    current_size: str | None,
) -> tuple[
    tuple[str, str, tuple[tuple[str, str], ...]] | None,
    dict | None,
    dict | None,
    ]:
    match = find_product(product_name)

    if match.status in ("not_found", "ambiguous"):
        return None, None, error_dict("invalid_product", f"Không tìm thấy món '{product_name}' trong menu.", requested=product_name, suggestions=match.suggestions)

    product = match.product
    matching_keys = _matching_keys(product["name"], current_size)

    if not matching_keys:
        return None, None, error_dict("not_in_cart", f"'{product['name']}' hiện không có trong giỏ hàng với điều kiện đã nêu.", cart=cart_summary())

    if len(matching_keys) > 1:
        return None, None, error_dict("ambiguous_cart_item",
            (
                f"'{product['name']}' đang có nhiều dòng phù hợp trong giỏ hàng, "
                "cần nói rõ món muốn sửa."
            ),
            variants=[_cart[key].to_dict() for key in matching_keys],
            cart=cart_summary(),
        )
    return matching_keys[0], product, None

def update_item(
    product_name: str,
    current_size: str | None = None,
    new_size: str | None = None,
    new_quantity: int | None = None,
    change_quantity: int | None = None,
) -> dict:
    if new_size is None and new_quantity is None and change_quantity is None:
        return error_dict("no_changes_requested", "Cần cho biết size mới, số lượng mới, hoặc số lượng cần tách sang size khác.")

    if new_quantity is not None and change_quantity is not None:
        return error_dict("conflicting_parameters", "Không thể dùng đồng thời new_quantity và change_quantity.")

    if change_quantity is not None and new_size is None:
        return error_dict("invalid_parameters", "change_quantity phải đi kèm new_size.")

    if new_quantity is not None and new_quantity < 0:
        return error_dict("invalid_quantity", f"Số lượng '{new_quantity}' không hợp lệ.")

    if change_quantity is not None and change_quantity <= 0:
        return error_dict("invalid_quantity", f"change_quantity '{change_quantity}' không hợp lệ.")

    key, product, error = locate_for_update(product_name, current_size)
    if error is not None:
        return error

    line = _cart[key]

    if new_quantity == 0:
        del _cart[key]
        return {
            "order_status": "success",
            "message": f"Đã xoá '{product['name']}' khỏi giỏ hàng vì số lượng = 0.",
            "cart": cart_summary(),
        }

    if change_quantity is not None:
        if change_quantity > line.quantity:
            return error_dict(
                "invalid_quantity",
                (
                    f"'{product['name']}' size '{line.size}' chỉ có "
                    f"{line.quantity} ly, không thể tách {change_quantity} ly."
                ),
                current_quantity=line.quantity,
            )

        sizes = get_product_sizes(product)
        new_size_norm = new_size.strip().upper()

        if new_size_norm not in sizes:
            return error_dict("invalid_size", f"Món '{product['name']}' không có size '{new_size}'.", valid_sizes=sorted(sizes.keys()))

        if new_size_norm == line.size:
            return error_dict("invalid_parameters", f"'{product['name']}' đã ở size '{line.size}'.")

        remaining = line.quantity - change_quantity
        if remaining == 0:
            del _cart[key]
        else:
            line.quantity = remaining

        target_key = _cart_key(product["name"], new_size_norm, line.customizations)

        if target_key in _cart:
            _cart[target_key].quantity += change_quantity
        else:
            _cart[target_key] = CartLine(
                product_name=product["name"],
                size=new_size_norm,
                unit_price=sizes[new_size_norm],
                quantity=change_quantity,
                customizations=dict(line.customizations),
            )

        return {
            "order_status": "success",
            "message": (
                f"Đã tách {change_quantity} ly "
                f"'{product['name']}' sang size '{new_size_norm}'."
            ),
            "product_name": product["name"],
            "split_from_size": line.size,
            "split_to_size": new_size_norm,
            "quantity_moved": change_quantity,
            "remaining_in_old_size": remaining,
            "customizations": dict(line.customizations),
            "cart": cart_summary(),
        }

    target_size = line.size
    if new_size is not None:
        sizes = get_product_sizes(product)
        target_size = new_size.strip().upper()

        if target_size not in sizes:
            return error_dict("invalid_size", f"Món '{product['name']}' không có size '{new_size}'.", valid_sizes=sorted(sizes.keys()))

    target_quantity = new_quantity if new_quantity is not None else line.quantity
    target_key = _cart_key(product["name"], target_size, line.customizations)
    target_unit_price = get_product_sizes(product)[target_size]

    if target_key == key:
        line.quantity = target_quantity
    else:
        del _cart[key]
        if target_key in _cart:
            _cart[target_key].quantity += target_quantity
        else:
            _cart[target_key] = CartLine(
                product_name=product["name"],
                size=target_size,
                unit_price=target_unit_price,
                quantity=target_quantity,
                customizations=dict(line.customizations),
            )

    return {
        "order_status": "success",
        "product_name": product["name"],
        "size": target_size,
        "quantity": target_quantity,
        "unit_price": target_unit_price,
        "customizations": dict(line.customizations),
        "cart": cart_summary(),
    }
