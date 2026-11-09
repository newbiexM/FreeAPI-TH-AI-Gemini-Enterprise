"""Local OpenAI-compatible HTTP facade for the direct AiPASS transport."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from . import __version__
from .auth import AuthError
from .client import AiPassClient, ProtocolError
from .models import map_reasoning
from .prompting import (
    TOOL_SENTINEL,
    bootstrap_prompt,
    delta_prompt,
    parse_assistant,
    tool_names,
)
from .sessions import SessionStore

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8792
MAX_REQUEST_BYTES = 50 * 1024 * 1024
CONVERSATION_SEED = "Hermes CLI session"


class BridgeRuntime:
    def __init__(self, client: AiPassClient | None = None, sessions: SessionStore | None = None) -> None:
        self.client = client
        self.sessions = sessions or SessionStore()
        self.queue = threading.Lock()
        self.started_at = time.time()

    def get_client(self) -> AiPassClient:
        if self.client is None:
            self.client = AiPassClient()
        return self.client


RUNTIME = BridgeRuntime()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _error(message: str, error_type: str = "aipass_bridge_error") -> dict[str, Any]:
    return {"error": {"message": message, "type": error_type}}


def _reasoning_effort(body: dict[str, Any]) -> str | None:
    direct = body.get("reasoning_effort")
    if isinstance(direct, str):
        return direct
    reasoning = body.get("reasoning")
    if isinstance(reasoning, dict) and isinstance(reasoning.get("effort"), str):
        return reasoning["effort"]
    thinking = body.get("thinking")
    if isinstance(thinking, dict) and isinstance(thinking.get("effort"), str):
        return thinking["effort"]
    return None


def _normalize_model_id(model_id: str) -> str:
    """Accept both Hermes' bare ``-m`` value and ``provider/model`` form."""
    prefixes = (
        "gemini-enterprise/",
        "gemini/",
        "aipass-gemini-enterprise/",
        "aipass_gemini_enterprise/",
        "aipass-ge/",
        "aipass/",
    )
    lower = model_id.lower()
    for prefix in prefixes:
        if lower.startswith(prefix):
            model_id = model_id[len(prefix):]
            break
    if not model_id or model_id.lower() == "auto":
        return "gemini-3.8-flash"
    return model_id


class Handler(BaseHTTPRequestHandler):
    server_version = "HermesGeminiEnterprise/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        # Do not log headers, bodies, account data, or query strings.
        print(f"aipass-bridge: {self.command} {self.path.split('?', 1)[0]} {args[1] if len(args) > 1 else ''}")

    def _send_json(self, status: int, value: Any) -> None:
        payload = _json_bytes(value)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("Invalid Content-Length") from exc
        if length <= 0 or length > MAX_REQUEST_BYTES:
            raise ValueError("Request body is empty or too large")
        decoded = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(decoded, dict):
            raise ValueError("Request body must be a JSON object")
        return decoded

    def do_GET(self) -> None:  # noqa: N802
        try:
            if self.path == "/health":
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "version": __version__,
                        "uptime_seconds": int(time.time() - RUNTIME.started_at),
                        "auth_configured": self._auth_configured(),
                    },
                )
                return
            if self.path.split("?", 1)[0] in {"/v1/models", "/models"}:
                refresh = "refresh=1" in self.path
                models = RUNTIME.get_client().models(refresh=refresh)
                self._send_json(200, {"object": "list", "data": [model.openai_dict() for model in models]})
                return
            self._send_json(404, _error("Not found", "not_found"))
        except AuthError as exc:
            self._send_json(401, _error(str(exc), "authentication_error"))
        except (ProtocolError, ValueError) as exc:
            self._send_json(502, _error(str(exc), "upstream_error"))

    @staticmethod
    def _auth_configured() -> bool:
        try:
            RUNTIME.get_client()
            return True
        except AuthError:
            return False

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in {"/v1/chat/completions", "/chat/completions"}:
            self._send_json(404, _error("Not found", "not_found"))
            return
        try:
            body = self._body()
            messages = body.get("messages")
            if not isinstance(messages, list) or not messages or not all(isinstance(item, dict) for item in messages):
                raise ValueError("messages must be a non-empty array of objects")
            model_id = body.get("model")
            if not isinstance(model_id, str) or not model_id:
                raise ValueError("model must be a non-empty string")
            model_id = _normalize_model_id(model_id)
            if body.get("stream"):
                self._stream_completion(body, messages, model_id)
            else:
                self._regular_completion(body, messages, model_id)
        except AuthError as exc:
            self._send_json(401, _error(str(exc), "authentication_error"))
        except ValueError as exc:
            self._send_json(400, _error(str(exc), "invalid_request_error"))
        except ProtocolError as exc:
            status = 400 if "unavailable" in str(exc).lower() else 502
            self._send_json(status, _error(str(exc), "upstream_error"))
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:  # pragma: no cover - defensive server boundary
            self._send_json(500, _error(f"AiPASS bridge failed: {type(exc).__name__}"))

    def _session_id(self, messages: list[dict[str, Any]]) -> str:
        supplied = self.headers.get("X-AiPASS-Session-Id", "").strip()
        if supplied:
            return supplied[:256]
        seed = _json_bytes(messages[:2])
        return "anonymous-" + hashlib.sha256(seed).hexdigest()[:24]

    def _execute(
        self,
        body: dict[str, Any],
        messages: list[dict[str, Any]],
        model_id: str,
    ) -> tuple[str, Any, set[str], str, str]:
        client = RUNTIME.get_client()
        model = client.model(model_id)
        session_id = self._session_id(messages)
        tools = body.get("tools") if isinstance(body.get("tools"), list) else []
        allowed = tool_names(tools)
        session, tail, bootstrap = RUNTIME.sessions.prepare(session_id, messages)
        if bootstrap:
            prompt = bootstrap_prompt(messages, tools)
            # The multipart create route is protected by stricter request-body
            # filtering than the chat action route. Keep creation metadata
            # innocuous, then send the complete English agent prompt exactly
            # once through the normal message transport below.
            conversation_id = client.create_conversation(CONVERSATION_SEED, model_id)
        else:
            if not tail:
                raise ValueError("Request contains no new conversation event")
            prompt = delta_prompt(tail)
            conversation_id = session.conversation_id
        if not conversation_id:
            raise ProtocolError("AiPASS conversation could not be resolved")
        thinking = map_reasoning(_reasoning_effort(body), model)
        return conversation_id, client.send_stream(
            conversation_id, prompt, model_id, thinking_level=thinking
        ), allowed, session_id, prompt

    def _complete_state(
        self,
        *,
        session_id: str,
        conversation_id: str,
        model_id: str,
        messages: list[dict[str, Any]],
        response_message: dict[str, Any],
    ) -> None:
        RUNTIME.sessions.complete(
            session_id,
            conversation_id=conversation_id,
            model=model_id,
            request_messages=messages,
            response_message=response_message,
        )

    def _regular_completion(self, body: dict[str, Any], messages: list[dict[str, Any]], model_id: str) -> None:
        with RUNTIME.queue:
            conversation_id, deltas, allowed, session_id, _ = self._execute(body, messages, model_id)
            raw = "".join(deltas)
            message = parse_assistant(raw, allowed)
            self._complete_state(
                session_id=session_id,
                conversation_id=conversation_id,
                model_id=model_id,
                messages=messages,
                response_message=message,
            )
        completion_id = f"chatcmpl-{uuid.uuid4()}"
        self._send_json(
            200,
            {
                "id": completion_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            },
        )

    def _sse(self, value: Any) -> None:
        self.wfile.write(b"data: " + _json_bytes(value) + b"\n\n")
        self.wfile.flush()

    def _stream_completion(self, body: dict[str, Any], messages: list[dict[str, Any]], model_id: str) -> None:
        completion_id = f"chatcmpl-{uuid.uuid4()}"
        created = int(time.time())
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        raw = ""
        natural_started = False
        with RUNTIME.queue:
            try:
                conversation_id, deltas, allowed, session_id, _ = self._execute(body, messages, model_id)
                for delta in deltas:
                    raw += delta
                    stripped = raw.lstrip()
                    may_be_tool = TOOL_SENTINEL.startswith(stripped) or stripped.startswith(TOOL_SENTINEL)
                    if not natural_started and stripped and not may_be_tool:
                        natural_started = True
                        self._sse(
                            {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_id,
                                "choices": [{"index": 0, "delta": {"role": "assistant", "content": raw}, "finish_reason": None}],
                            }
                        )
                    elif natural_started:
                        self._sse(
                            {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_id,
                                "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
                            }
                        )
                message = parse_assistant(raw, allowed)
                if message.get("tool_calls"):
                    self._sse(
                        {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_id,
                            "choices": [{"index": 0, "delta": {"role": "assistant", "tool_calls": message["tool_calls"]}, "finish_reason": None}],
                        }
                    )
                elif not natural_started:
                    self._sse(
                        {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_id,
                            "choices": [{"index": 0, "delta": {"role": "assistant", "content": raw}, "finish_reason": None}],
                        }
                    )
                finish = "tool_calls" if message.get("tool_calls") else "stop"
                self._complete_state(
                    session_id=session_id,
                    conversation_id=conversation_id,
                    model_id=model_id,
                    messages=messages,
                    response_message=message,
                )
                self._sse(
                    {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                    }
                )
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except Exception as exc:
                self._sse(_error(str(exc), "stream_error"))
            finally:
                self.close_connection = True


def serve(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("Refusing to expose the credentialed AiPASS bridge beyond localhost")
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Hermes AiPASS bridge {__version__} listening on http://{host}:{port}/v1", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Hermes AiPASS local bridge")
    parser.add_argument("--host", default=os.environ.get("AIPASS_BRIDGE_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.environ.get("AIPASS_BRIDGE_PORT", DEFAULT_PORT)))
    args = parser.parse_args()
    serve(args.host, args.port)


if __name__ == "__main__":
    main()
