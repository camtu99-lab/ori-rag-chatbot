"""Shared structured responses for business tools."""
from __future__ import annotations
import json

def error_dict(error_type: str, message: str, **extra: object) -> dict:
    payload = {"order_status": "error", "error_type": error_type, "message": message}
    payload.update(extra)
    return payload

def success_dict(**fields: object) -> dict:
    return {"order_status": "success", **fields}

def error_response(error_type: str, message: str, **extra: object) -> str:
    return json.dumps(error_dict(error_type, message, **extra), ensure_ascii=False)

def success_response(**fields: object) -> str:
    return json.dumps(success_dict(**fields), ensure_ascii=False)
