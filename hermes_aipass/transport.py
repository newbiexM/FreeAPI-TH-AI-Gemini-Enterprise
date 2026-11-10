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
    """Factory resolving the active upstream transport.
    
    Defaults to Gemini Enterprise DiscoveryEngineTransport.
    Use name='web' or AIPASS_TRANSPORT='web' to force legacy web client.
    """
    selected = (name or os.environ.get("AIPASS_TRANSPORT", "")).strip().lower()

    if selected in {"web", "aipass", "chat"}:
        from .client import AiPassClient
        return AiPassClient()

    try:
        from .discovery_engine import DiscoveryEngineTransport
        return DiscoveryEngineTransport()
    except Exception:
        # Fall back to web client if Discovery Engine configuration / auth is unavailable
        from .client import AiPassClient
        return AiPassClient()
