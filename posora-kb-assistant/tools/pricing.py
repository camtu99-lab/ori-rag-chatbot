"""Pricing policies for checkout.
Các policy hiện tại trả 0 cho discount/shipping/tax vì menu hiện tại
chưa cung cấp dữ liệu tương ứng.
"""
def compute_discount(cart_summary: dict) -> int | float:
    return 0

def compute_shipping_fee(cart_summary: dict) -> int | float:
    return 0

def compute_tax(cart_summary: dict) -> int | float:
    return 0

def compute_total(cart_summary: dict) -> dict:
    subtotal = cart_summary["subtotal"]
    discount = compute_discount(cart_summary)
    shipping_fee = compute_shipping_fee(cart_summary)
    tax = compute_tax(cart_summary)

    return {
        "subtotal": subtotal,
        "discount": discount,
        "shipping_fee": shipping_fee,
        "tax": tax,
        "total": subtotal - discount + shipping_fee + tax,
        "currency": cart_summary["currency"],
    }
