"""Hermes request middleware for stable AiPASS session scoping."""

from __future__ import annotations

from typing import Any


def attach_session_header(
    *,
    request: dict[str, Any],
    provider: str = "",
    base_url: str = "",
    session_id: str = "",
    **_: Any,
) -> dict[str, Any] | None:
    """Attach the logical Hermes session only to the local AiPASS provider."""
    is_aipass = str(provider).lower() == "aipass" or "127.0.0.1:8792" in str(base_url)
    if not is_aipass or not session_id:
        return None
    updated = dict(request)
    headers = dict(updated.get("extra_headers") or {})
    headers["X-AiPASS-Session-Id"] = str(session_id)[:256]
    updated["extra_headers"] = headers
    return {"request": updated, "source": "hermes-aipass-session"}

