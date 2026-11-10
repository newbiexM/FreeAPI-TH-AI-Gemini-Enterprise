"""Direct Google Discovery Engine (Vertex AI Search) transport for Gemini Enterprise."""

from __future__ import annotations

import json
import os
import uuid
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .auth import KeychainStore
from .client import ProtocolError
from .models import GEMINI_ENTERPRISE_MODELS, Model

DEFAULT_TENANT_ID = os.environ.get(
    "AIPASS_DISCOVERY_ENGINE_CID",
    "70e36ff1-c0e0-4da7-841e-017ad9c3c4eb",
)
DISCOVERY_ENGINE_ENDPOINT = (
    "https://content-us-discoveryengine.googleapis.com/v1alpha/locations/us/widget"
)
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/152.0 Safari/537.36"
)


class DiscoveryEngineTransport:
    """Direct Google Cloud Discovery Engine / Vertex AI Search transport."""

    def __init__(
        self,
        *,
        tenant_id: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.tenant_id = tenant_id or DEFAULT_TENANT_ID
        self.timeout = timeout
        self.store = KeychainStore()
        self.auth = self.store.load()

    def _headers(self, *, accept: str = "text/event-stream") -> dict[str, str]:
        cookie = getattr(self.auth, "cookie_header", "")
        return {
            "Accept": accept,
            "Accept-Language": "en-US,en;q=0.9",
            "Cookie": cookie,
            "Origin": "https://vertexaisearch.cloud.google",
            "Referer": f"https://vertexaisearch.cloud.google/us/home/cid/{self.tenant_id}",
            "User-Agent": USER_AGENT,
            "Content-Type": "application/json",
        }

    def _open(self, request: Request) -> Any:
        try:
            return urlopen(request, timeout=self.timeout)
        except HTTPError as exc:
            detail = ""
            try:
                detail = exc.read(2048).decode("utf-8", errors="replace")
            except Exception:
                pass
            raise ProtocolError(
                f"Google Discovery Engine returned HTTP {exc.code}: {detail[:400]}"
            ) from exc
        except (URLError, TimeoutError) as exc:
            raise ProtocolError(f"Google Discovery Engine request failed: {exc}") from exc

    def models(self, *, refresh: bool = False) -> list[Model]:
        return list(GEMINI_ENTERPRISE_MODELS)

    def model(self, model_id: str) -> Model:
        normalized = model_id.lower()
        for prefix in (
            "freeapi-th-ai-gemini-enterprise/",
            "freeapi/",
            "th-ai/",
            "gemini-enterprise/",
            "aipass-gemini-enterprise/",
            "gemini/",
            "aipass/",
        ):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
                break
        for m in self.models():
            if m.id == normalized:
                return m
        for m in self.models():
            if m.id == "gemini-3.8-flash":
                return m
        return self.models()[0]

    def create_conversation(self, prompt: str, model_id: str) -> str:
        """Create a direct Discovery Engine conversation session."""
        return f"ge-{uuid.uuid4().hex[:16]}"

    def send_stream(
        self,
        conversation_id: str,
        prompt: str,
        model_id: str,
        *,
        thinking_level: str | None = None,
    ) -> Iterator[str]:
        """Stream responses directly from Google Discovery Engine SSE endpoint."""
        model_obj = self.model(model_id)
        payload: dict[str, Any] = {
            "conversation": conversation_id,
            "tenant": self.tenant_id,
            "query": prompt,
            "modelConfig": {
                "model": model_obj.id,
            },
        }
        if thinking_level:
            payload["modelConfig"]["thinkingLevel"] = thinking_level

        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        req = Request(
            f"{DISCOVERY_ENGINE_ENDPOINT}/streamQuery",
            data=encoded,
            headers=self._headers(accept="text/event-stream"),
            method="POST",
        )
        response = self._open(req)
        try:
            for raw_line in response:
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if not data_str or data_str == "[DONE]":
                    break
                try:
                    event = json.loads(data_str)
                except Exception:
                    continue

                # Parse Discovery Engine answer format
                answer = event.get("answer") if isinstance(event, dict) else None
                if isinstance(answer, dict):
                    thought = answer.get("thought")
                    if thought:
                        yield f"<thought>{thought}</thought>"
                    reply = answer.get("reply") or answer.get("content")
                    if reply:
                        yield str(reply)
                elif isinstance(event, dict):
                    if "thought" in event and event["thought"]:
                        yield f"<thought>{event['thought']}</thought>"
                    if "delta" in event and event["delta"]:
                        yield str(event["delta"])
                    elif "text" in event and event["text"]:
                        yield str(event["text"])
        finally:
            if hasattr(response, "close"):
                response.close()
