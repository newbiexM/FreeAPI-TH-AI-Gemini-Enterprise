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
from .accounts import Account, AccountPool, AllAccountsUnavailableError
from .codec import decode_loader_bytes, extract_stream_events, reasoning_delta, text_delta, walk
from .models import GEMINI_ENTERPRISE_MODELS, Model, extract_models

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/152.0 Safari/537.36"
)


class ProtocolError(RuntimeError):
    pass


class ContentPolicyViolationError(ProtocolError):
    def __init__(self, detail: str, code: int = 422, category: str = "SCAM_FRAUD") -> None:
        super().__init__(f"AiPASS Content Policy Violation ({category}): {detail[:200]}")
        self.detail = detail
        self.code = code
        self.category = category



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
        default_cookie = getattr(self.auth, "cookie_header", "")
        self.pool = AccountPool.from_env(default_token=default_cookie)
        self._current_account: Account | None = None
        self._models: list[Model] = []
        self._models_at = 0.0
        self._lock = threading.RLock()

    def _headers(self, *, accept: str = "application/json") -> dict[str, str]:
        cookie = self.auth.cookie_header
        try:
            if self.pool.accounts:
                self._current_account = self.pool.get_account()
                if self._current_account and self._current_account.token:
                    cookie = self._current_account.token
        except Exception:
            pass

        return {
            "Accept": accept,
            "Accept-Language": "en-US,en;q=0.9",
            "Cookie": cookie,
            "Origin": BASE_URL,
            "Referer": f"{BASE_URL}/chat",
            "User-Agent": USER_AGENT,
        }

    def _update_auth(self, response: Any) -> None:
        headers = response.headers.get_all("Set-Cookie") or []
        if apply_set_cookie(self.auth, headers):
            self.store.save(self.auth)

    def _open(self, request: Request, *, timeout: float | None = None):
        cur_acct = getattr(self, "_current_account", None)
        try:
            response = urlopen(request, timeout=timeout or self.timeout)
            self._update_auth(response)
            if cur_acct:
                cur_acct.report_ok()
            return response
        except HTTPError as exc:
            if exc.code == 401:
                if cur_acct:
                    cur_acct.report_dead("AiPASS session expired")
                raise AuthError("AiPASS session expired; run `hermes aipass login`") from exc
            if exc.code == 429:
                if cur_acct:
                    cur_acct.report_429()
                raise ProtocolError(f"AiPASS rate limited (HTTP 429). Account cooling down.") from exc
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
            if exc.code == 422:
                if "CHAT_RAI_VIOLATION" in detail or "Content Policy Violation" in detail or "SCAM_FRAUD" in detail:
                    raise ContentPolicyViolationError(detail, code=422, category="SCAM_FRAUD") from exc
                raise ProtocolError(f"AiPASS returned HTTP 422: {detail[:400]}") from exc
            raise ProtocolError(f"AiPASS returned HTTP {exc.code}: {detail[:400]}") from exc

        except (URLError, TimeoutError) as exc:
            raise ProtocolError(f"AiPASS request failed: {exc}") from exc

    def models(self, *, refresh: bool = False) -> list[Model]:
        return list(GEMINI_ENTERPRISE_MODELS)

    def model(self, model_id: str) -> Model:
        for model in self.models():
            if model.id == model_id:
                return model
        normalized = model_id.lower()
        for prefix in ("gemini-enterprise/", "aipass-gemini-enterprise/", "gemini/", "aipass/"):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
                break
        for model in self.models():
            if model.id == normalized:
                return model
        # Graceful fallback for legacy or unknown model requests (e.g. glm-5.3, gpt-6.1-sol, auto)
        for model in self.models():
            if model.id == "gemini-3.8-flash":
                return model
        return self.models()[0]

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

    @staticmethod
    def _clean_model_id(model_id: str) -> str:
        for prefix in ("gemini-enterprise/", "aipass-gemini-enterprise/", "gemini/", "aipass/"):
            if model_id.lower().startswith(prefix):
                return model_id[len(prefix):]
        return model_id

    def create_conversation(self, prompt: str, model_id: str) -> str:
        model_id = self._clean_model_id(model_id)
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
        except ValueError as exc:
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
        model_id = self._clean_model_id(model_id)
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
                r_delta = reasoning_delta(event)
                if r_delta:
                    saw_content = True
                    yield f"<thought>{r_delta}</thought>"

            if not saw_content:
                raise ProtocolError("AiPASS stream completed without assistant content")
        finally:
            response.close()
