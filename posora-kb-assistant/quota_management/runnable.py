from __future__ import annotations
from typing import Any
from langchain_core.runnables import Runnable
from quota_management.manager import QuotaManager
from quota_management.usage import extract_total_tokens

class QuotaRunnable(Runnable):
    """Transparent Runnable wrapper: every actual LLM invoke consumes quota."""
    def __init__(self, runnable: Runnable, *, manager: QuotaManager, workload: str, subject: str | None = None) -> None:
        self._runnable = runnable
        self._manager = manager
        self._workload = workload
        self._subject = subject

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        reservation = self._manager.reserve(workload=self._workload, subject=self._subject)
        try:
            response = self._runnable.invoke(input, config=config, **kwargs)
        except Exception:
            reservation.cancel()
            raise

        reservation.commit(extract_total_tokens(response))
        return response

    def bind_tools(self, tools: Any, **kwargs: Any) -> "QuotaRunnable":
        bound = self._runnable.bind_tools(tools, **kwargs)
        return QuotaRunnable(
            bound,
            manager=self._manager,
            workload=self._workload,
            subject=self._subject,
        )
