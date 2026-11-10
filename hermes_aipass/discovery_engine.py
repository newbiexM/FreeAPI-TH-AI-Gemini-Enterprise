"""Direct Google Discovery Engine (Vertex AI Search) transport for Gemini Enterprise."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from typing import Any, Iterable, Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .auth import KeychainStore
from .client import ProtocolError
from .models import GEMINI_ENTERPRISE_MODELS, Model

DEFAULT_TENANT_ID = os.environ.get(
    "AIPASS_DISCOVERY_ENGINE_CID",
    "70e36ff1-c0e0-4da7-841e-017ad9c3c4eb",
)
DEFAULT_CSESIDX = os.environ.get(
    "AIPASS_DISCOVERY_ENGINE_CSESIDX",
    "735410566",
)
DISCOVERY_ENGINE_BASE = "https://content-us-discoveryengine.googleapis.com/v1alpha/locations/us"
AUTH_PORTAL_BASE = "https://vertexaisearch.cloud.google"
API_CLIENT_HEADER = "gemini-enterprise-classic/1.0.0"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/157.0.0.0 Safari/537.36"
)


def _b64u_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def _b64u_decode(data: str) -> bytes:
    padded = data + "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(padded)


def parse_json_stream(stream: Iterable[bytes]) -> Iterator[dict[str, Any]]:
    """Incrementally parse streaming JSON objects from raw byte chunks."""
    buffer = ""
    decoder = json.JSONDecoder()
    for raw_chunk in stream:
        if not raw_chunk:
            continue
        buffer += raw_chunk.decode("utf-8", errors="replace")
        buffer = buffer.lstrip("[\r\n, ")
        while buffer:
            try:
                obj, index = decoder.raw_decode(buffer)
                if isinstance(obj, dict):
                    yield obj
                buffer = buffer[index:].lstrip("\r\n, ]")
            except json.JSONDecodeError:
                break


class DiscoveryEngineTransport:
    """Direct Google Cloud Discovery Engine / Vertex AI Search transport."""

    def __init__(
        self,
        *,
        tenant_id: str | None = None,
        csesidx: str | None = None,
        timeout: float = 60.0,
        auth: Any | None = None,
    ) -> None:
        self.store = KeychainStore()
        self.auth = auth if auth is not None else self.store.load()
        self.tenant_id = (
            tenant_id
            or getattr(self.auth, "tenant_id", None)
            or DEFAULT_TENANT_ID
        )
        self.csesidx = (
            csesidx
            or getattr(self.auth, "csesidx", None)
            or DEFAULT_CSESIDX
        )
        self.timeout = timeout
        self._cached_jwt: tuple[str, float] | None = None

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

    def _get_jwt(self) -> str:
        now = time.time()
        if self._cached_jwt is not None:
            token, exp = self._cached_jwt
            if now < exp - 30:
                return token

        data: dict[str, Any] = {}
        response: Any = None
        for attempt in (1, 2):
            if not self.csesidx and attempt == 1:
                try:
                    from .cdp import refresh_session_from_browser
                    refreshed = refresh_session_from_browser()
                    if refreshed:
                        self.auth = refreshed
                        self.csesidx = refreshed.csesidx
                        self.tenant_id = refreshed.tenant_id or self.tenant_id
                except Exception:
                    pass

            cookie = getattr(self.auth, "cookie_header", "")
            xsrf_url = f"{AUTH_PORTAL_BASE}/auth/getoxsrf?csesidx={self.csesidx}"
            req = Request(
                xsrf_url,
                headers={
                    "Cookie": cookie,
                    "User-Agent": USER_AGENT,
                    "Referer": f"{AUTH_PORTAL_BASE}/",
                    "Accept": "*/*",
                },
            )
            try:
                resp = self._open(req)
                try:
                    raw = resp.read().decode("utf-8", errors="replace")
                    clean = raw.replace(")]}'", "").strip()
                    data = json.loads(clean)
                finally:
                    if hasattr(resp, "close"):
                        resp.close()
                break
            except ProtocolError as exc:
                if attempt == 1 and ("INVALID_CSESIDX" in str(exc) or "MISSING_COOKIES" in str(exc) or "401" in str(exc)):
                    try:
                        from .cdp import refresh_session_from_browser
                        refreshed = refresh_session_from_browser()
                        if refreshed:
                            self.auth = refreshed
                            self.csesidx = refreshed.csesidx
                            self.tenant_id = refreshed.tenant_id or self.tenant_id
                            continue
                    except Exception:
                        pass
                raise

        xsrf_token = data.get("xsrfToken")
        key_id = data.get("keyId")
        if not xsrf_token or not key_id:
            raise ProtocolError("Failed to obtain XSRF token or keyId from Google Discovery Engine")

        key_bytes = _b64u_decode(xsrf_token)
        header = {"alg": "HS256", "typ": "JWT", "kid": key_id}
        now_ts = int(now)
        payload = {
            "iss": AUTH_PORTAL_BASE,
            "aud": "https://content-us-discoveryengine.googleapis.com",
            "sub": f"csesidx/{self.csesidx}",
            "iat": now_ts,
            "exp": now_ts + 300,
            "nbf": now_ts,
        }
        h_enc = _b64u_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
        p_enc = _b64u_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
        to_sign = f"{h_enc}.{p_enc}".encode("utf-8")
        sig = hmac.new(key_bytes, to_sign, hashlib.sha256).digest()
        jwt = f"{h_enc}.{p_enc}.{_b64u_encode(sig)}"

        self._cached_jwt = (jwt, now + 270)
        return jwt

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
        response: Any = None
        for attempt in (1, 2):
            jwt = self._get_jwt()
            req_body = {
                "configId": self.tenant_id,
                "additionalParams": {"token": "-", "origin": "ORIGIN_UNSPECIFIED"},
                "createSessionRequest": {"session": {"name": "-", "displayName": ""}},
            }
            encoded = json.dumps(req_body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            req = Request(
                f"{DISCOVERY_ENGINE_BASE}/widgetCreateSession",
                data=encoded,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {jwt}",
                    "X-Goog-Api-Client": API_CLIENT_HEADER,
                    "Origin": AUTH_PORTAL_BASE,
                    "Referer": f"{AUTH_PORTAL_BASE}/",
                    "User-Agent": USER_AGENT,
                },
                method="POST",
            )
            try:
                resp = self._open(req)
                try:
                    data = json.loads(resp.read().decode("utf-8", errors="replace"))
                    session_name = data.get("session", {}).get("name")
                    if not session_name:
                        raise ProtocolError("widgetCreateSession did not return a session name")
                    return str(session_name)
                finally:
                    if hasattr(resp, "close"):
                        resp.close()
            except ProtocolError as exc:
                if attempt == 1 and ("401" in str(exc) or "403" in str(exc)):
                    self._cached_jwt = None
                    try:
                        from .cdp import refresh_session_from_browser
                        refreshed = refresh_session_from_browser()
                        if refreshed:
                            self.auth = refreshed
                            self.csesidx = refreshed.csesidx
                            self.tenant_id = refreshed.tenant_id or self.tenant_id
                            continue
                    except Exception:
                        pass
                raise
        raise ProtocolError("Failed to create Discovery Engine session")

    def send_stream(
        self,
        conversation_id: str,
        prompt: str,
        model_id: str,
        *,
        thinking_level: str | None = None,
    ) -> Iterator[str]:
        """Stream responses directly from Google Discovery Engine widgetStreamAssist endpoint."""
        response: Any = None
        for attempt in (1, 2):
            if not conversation_id or not conversation_id.startswith("collections/"):
                conversation_id = self.create_conversation(prompt, model_id)

            jwt = self._get_jwt()
            req_body = {
                "configId": self.tenant_id,
                "additionalParams": {"token": "-", "origin": "ORIGIN_UNSPECIFIED"},
                "streamAssistRequest": {
                    "session": conversation_id,
                    "query": {"parts": [{"text": prompt}]},
                    "filter": "",
                    "fileIds": [],
                    "answerGenerationMode": "NORMAL",
                    "agentsSpec": {},
                    "toolsSpec": {
                        "vertexAiSearchSpec": {
                            "dataStoreSpecs": [
                                {"dataStore": "collections/default_collection/dataStores/canva_mcp_data"},
                                {"dataStore": "collections/default_collection/dataStores/citizen-announcement_1793980447630"},
                                {"dataStore": "collections/default_collection/dataStores/google-sheets_mcp_data"},
                            ]
                        },
                        "toolRegistry": "default_tool_registry",
                    },
                    "languageCode": "en-US",
                    "userMetadata": {"timeZone": "Asia/Bangkok"},
                    "assistSkippingMode": "REQUEST_ASSIST",
                },
            }

            encoded = json.dumps(req_body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            req = Request(
                f"{DISCOVERY_ENGINE_BASE}/widgetStreamAssist",
                data=encoded,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {jwt}",
                    "X-Goog-Api-Client": API_CLIENT_HEADER,
                    "Origin": AUTH_PORTAL_BASE,
                    "Referer": f"{AUTH_PORTAL_BASE}/",
                    "User-Agent": USER_AGENT,
                },
                method="POST",
            )
            try:
                response = self._open(req)
                break
            except ProtocolError as exc:
                if attempt == 1 and ("401" in str(exc) or "403" in str(exc)):
                    self._cached_jwt = None
                    try:
                        from .cdp import refresh_session_from_browser
                        refreshed = refresh_session_from_browser()
                        if refreshed:
                            self.auth = refreshed
                            self.csesidx = refreshed.csesidx
                            self.tenant_id = refreshed.tenant_id or self.tenant_id
                    except Exception:
                        pass
                    conversation_id = ""
                    continue
                raise
        try:
            def chunk_generator() -> Iterator[bytes]:
                while True:
                    chunk = response.read(1024)
                    if not chunk:
                        break
                    yield chunk

            for obj in parse_json_stream(chunk_generator()):
                resp_obj = obj.get("streamAssistResponse", {})
                answer = resp_obj.get("answer", {})
                replies = answer.get("replies", [])
                for reply in replies:
                    if not isinstance(reply, dict):
                        continue
                    grounded = reply.get("groundedContent", {})
                    content = grounded.get("content", {})
                    text = content.get("text")
                    if text:
                        yield str(text)
        finally:
            if hasattr(response, "close"):
                response.close()
