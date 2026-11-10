"""Abstract Transport Seam decoupling OpenAI/Anthropic bridge from upstream APIs."""

from __future__ import annotations

import os
from typing import Iterator, Protocol, runtime_checkable

from .models import Model


@runtime_checkable
class Transport(Protocol):
    def models(self, *, refresh: bool = False) -> list[Model]:
        ...

    def model(self, model_id: str) -> Model:
        ...

    def create_conversation(self, prompt: str, model_id: str) -> str:
        ...

    def send_stream(
        self,
        conversation_id: str,
        prompt: str,
        model_id: str,
        *,
        thinking_level: str | None = None,
    ) -> Iterator[str]:
        ...


def get_transport(name: str | None = None) -> Transport:
    """Factory resolving the active upstream transport."""
    selected = (name or os.environ.get("AIPASS_TRANSPORT", "")).strip().lower()

    if selected in {"discovery_engine", "discoveryengine", "enterprise", "vertex", "ge"}:
        from .discovery_engine import DiscoveryEngineTransport
        return DiscoveryEngineTransport()

    if selected in {"web", "aipass", "chat"}:
        from .client import AiPassClient
        return AiPassClient()

    if os.environ.get("AIPASS_USE_ENTERPRISE", "").lower() in {"1", "true", "yes"}:
        from .discovery_engine import DiscoveryEngineTransport
        return DiscoveryEngineTransport()

    from .client import AiPassClient
    return AiPassClient()

