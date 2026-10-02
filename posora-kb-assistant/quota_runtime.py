from __future__ import annotations
from functools import lru_cache
from quota_management.config import get_quota_settings
from quota_management.manager import QuotaManager

@lru_cache(maxsize=1)
def get_quota_manager() -> QuotaManager:
    return QuotaManager(get_quota_settings())
