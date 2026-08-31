"""Authenticated direct client for AiPASS's web application transport."""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .auth import BASE_URL, AuthError, KeychainStore, apply_set_cookie
from .codec import decode_loader_bytes, extract_stream_events, text_delta, walk
from .models import Model, extract_models

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/152.0 Safari/537.36"
)


class ProtocolError(RuntimeError):
    pass


@dataclass
class StreamResult:
    conversation_id: str
    deltas: Iterator[str]


class AiPassClient:
    def __init__(self, *, timeout: float = 60, model_cache_ttl: float = 300) -> None:
        self.timeout = timeout
        self.model_cache_ttl = model_cache_ttl
        self.store = KeychainStore()
        self.auth = self.store.load()
        self._models: list[Model] = []
        self._models_at = 0.0
        self._lock = threading.RLock()

    def _headers(self, *, accept: str = "application/json") -> dict[str, str]:
        return {
            "Accept": accept,
            "Accept-Language": "en-US,en;q=0.9",
            "Cookie": self.auth.cookie_header,
            "Origin": BASE_URL,
            "Referer": f"{BASE_URL}/chat",
            "User-Agent": USER_AGENT,
        }

    def _update_auth(self, response: Any) -> None:
        headers = response.headers.get_all("Set-Cookie") or []
        if apply_set_cookie(self.auth, headers):
            self.store.save(self.auth)

    def _open(self, request: Request, *, timeout: float | None = None):
        try:
            response = urlopen(request, timeout=timeout or self.timeout)
            self._update_auth(response)
            return response
        except HTTPError as exc:
            if exc.code == 401:
                raise AuthError("AiPASS session expired; run `hermes aipass login`") from exc
            detail = ""
            try:
                detail = exc.read(2048).decode("utf-8", errors="replace")
            except Exception:
                pass
            if exc.code == 403:
                raise ProtocolError(
                    "AiPASS denied this request (HTTP 403). The selected model may "
                    "require account access or paid credits."
                ) from exc
            raise ProtocolError(f"AiPASS returned HTTP {exc.code}: {detail[:400]}") from exc
        except (URLError, TimeoutError) as exc:
            raise ProtocolError(f"AiPASS request failed: {exc}") from exc

    def models(self, *, refresh: bool = False) -> list[Model]:
        with self._lock:
            now = time.monotonic()
            if self._models and not refresh and now - self._models_at < self.model_cache_ttl:
                return list(self._models)
            url = f"{BASE_URL}/loaders/list-models.data?_routes=routes%2Floaders%2Flist-models"
            request = Request(url, headers=self._headers())
            with self._open(request) as response:
                decoded = decode_loader_bytes(response.read())
            models = extract_models(decoded)
            if not models:
                raise ProtocolError("AiPASS model loader returned no usable conversation models")
            self._models = models
            self._models_at = now
            return list(models)

    def model(self, model_id: str) -> Model:
        for model in self.models():
            if model.id == model_id:
                return model
        available = ", ".join(model.id for model in self.models()[:12])
        raise ProtocolError(f"AiPASS model {model_id!r} is unavailable. Available models: {available}")

    @staticmethod
    def _multipart(fields: dict[str, str]) -> tuple[bytes, str]:
        boundary = f"----HermesAiPASS{uuid.uuid4().hex}"
        body = bytearray()
        for name, value in fields.items():
            body.extend(f"--{boundary}\r\n".encode())
            body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
            body.extend(value.encode("utf-8"))
            body.extend(b"\r\n")
        body.extend(f"--{boundary}--\r\n".encode())
        return bytes(body), f"multipart/form-data; boundary={boundary}"

    @staticmethod
    def _conversation_id(decoded: Any) -> str | None:
        preferred = ("conversationId", "conversation_id")
        for value in walk(decoded):
            if not isinstance(value, dict):
                continue
            for key in preferred:
                candidate = value.get(key)
                if isinstance(candidate, str) and candidate:
                    return candidate
            conversation = value.get("conversation")
            if isinstance(conversation, dict):
                candidate = conversation.get("id")
                if isinstance(candidate, str) and candidate:
                    return candidate
        return None

    def create_conversation(self, prompt: str, model_id: str) -> str:
        request_id = uuid.uuid4().hex
        body, content_type = self._multipart(
            {
                "message": prompt,
                "folderId": "",
                "modelId": model_id,
                "intent": "create-conversation",
                "clientCreateRequestId": request_id,
            }
        )
        headers = self._headers()
        headers.update({"Content-Type": content_type, "X-Remix-Request": "true"})
        request = Request(f"{BASE_URL}/chat.data", data=body, headers=headers, method="POST")
        with self._open(request) as response:
            payload = response.read()
        try:
            decoded = decode_loader_bytes(payload)
        except (ValueError, json.JSONDecodeError) as exc:
            raise ProtocolError("AiPASS create-conversation response could not be decoded") from exc
        conversation_id = self._conversation_id(decoded)
        if not conversation_id:
            raise ProtocolError("AiPASS create-conversation response omitted the conversation id")
        return conversation_id

    def send_stream(
        self,
        conversation_id: str,
        prompt: str,
        model_id: str,
        *,
        thinking_level: str | None = None,
    ) -> Iterator[str]:
        message_id = uuid.uuid4().hex
        body: dict[str, Any] = {
            "id": conversation_id,
            "messages": [
                {
                    "id": message_id,
                    "role": "user",
                    "parts": [{"type": "text", "text": prompt}],
                }
            ],
            "trigger": "submit-message",
            "messageId": message_id,
            "modelId": model_id,
        }
        if thinking_level:
            body["thinkingLevel"] = thinking_level
        encoded = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()
        headers = self._headers(accept="text/event-stream")
        headers["Content-Type"] = "application/json"
        request = Request(
            f"{BASE_URL}/actions/send-message/{conversation_id}",
            data=encoded,
            headers=headers,
            method="POST",
        )
        response = self._open(request, timeout=1800)
        try:
            saw_content = False
            for event in extract_stream_events(response):
                if event.get("type") == "error":
                    detail = event.get("errorText") or event.get("error") or "unknown stream error"
                    raise ProtocolError(f"AiPASS generation failed: {detail}")
                delta = text_delta(event)
                if delta:
                    saw_content = True
                    yield delta
            if not saw_content:
                raise ProtocolError("AiPASS stream completed without assistant content")
        finally:
            response.close()
