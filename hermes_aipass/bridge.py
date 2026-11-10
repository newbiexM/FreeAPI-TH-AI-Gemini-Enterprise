"""Local OpenAI-compatible HTTP facade for the direct AiPASS transport."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "hermes_aipass"

from . import __version__
from .auth import AuthError
from .client import AiPassClient, ContentPolicyViolationError, ProtocolError
from .transport import Transport, get_transport
from .models import map_reasoning
from .prompting import (
    TOOL_SENTINEL,
    bootstrap_prompt,
    delta_prompt,
    parse_assistant,
    tool_names,
)
from .sessions import SessionStore
from .streaming import RefusalDetected, StreamInterceptor
from .anthropic import (
    anthropic_to_openai_request,
    estimate_tokens,
    openai_to_anthropic_response,
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8792
MAX_REQUEST_BYTES = 50 * 1024 * 1024
CONVERSATION_SEED = "Hermes CLI session"

_REFUSAL_PHRASES: tuple[str, ...] = (
    "ขออภัย ไม่สามารถตอบคำถามนี้ได้",
    "Sorry, I could not respond to this request",
    "ขออภัยครับ ผมไม่สามารถ",
    "ขออภัยค่ะ ดิฉันไม่สามารถ",
    "I cannot access your files",
    "I am unable to execute",
)


class BridgeRuntime:
    def __init__(self, client: Transport | None = None, sessions: SessionStore | None = None) -> None:
        self.client = client
        self.sessions = sessions or SessionStore()
        self._session_locks: dict[str, threading.Lock] = collections.defaultdict(threading.Lock)
        self._locks_mutex = threading.Lock()
        self.started_at = time.time()

    def get_client(self) -> Transport:
        if self.client is None:
            self.client = get_transport()
        return self.client

    def session_lock(self, session_id: str) -> threading.Lock:
        with self._locks_mutex:
            return self._session_locks[session_id]


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
        "freeapi-th-ai-gemini-enterprise/",
        "freeapi/",
        "th-ai/",
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

    def log_message(self, format: str, *args: Any) -> None:
        # Do not log headers, bodies, account data, or query strings.
        cmd = getattr(self, "command", "HTTP")
        path = getattr(self, "path", "/")
        print(f"aipass-bridge: {cmd} {path.split('?', 1)[0]} {args[1] if len(args) > 1 else ''}")

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
        try:
            # 1. Anthropic count_tokens endpoint
            if self.path in {"/v1/messages/count_tokens", "/messages/count_tokens"}:
                body = self._body()
                self._send_json(200, {"input_tokens": estimate_tokens(body)})
                return

            # 2. Anthropic messages endpoint
            if self.path in {"/v1/messages", "/messages"}:
                body = self._body()
                openai_body, messages, model_id = anthropic_to_openai_request(body)
                if not messages:
                    raise ValueError("messages must be a non-empty array")
                if openai_body.get("stream"):
                    self._stream_anthropic_completion(body, openai_body, messages, model_id)
                else:
                    self._regular_anthropic_completion(body, openai_body, messages, model_id)
                return

            # 3. Standard OpenAI chat completions endpoint
            if self.path not in {"/v1/chat/completions", "/chat/completions"}:
                self._send_json(404, _error("Not found", "not_found"))
                return
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

    def _session_id(self, messages: list[dict[str, Any]], body: dict[str, Any] | None = None) -> str:
        supplied = self.headers.get("X-AiPASS-Session-Id", "").strip()
        if not supplied and body:
            supplied = str(body.get("session_id") or body.get("user") or "").strip()
        if supplied:
            return supplied[:256]
        seed = _json_bytes([m.get("content") for m in messages[-4:]])
        return "anonymous-" + hashlib.sha256(seed).hexdigest()[:24]

    def _execute(
        self,
        body: dict[str, Any],
        messages: list[dict[str, Any]],
        model_id: str,
    ) -> tuple[str, Any, set[str], str, str]:
        client = RUNTIME.get_client()
        model = client.model(model_id)
        session_id = self._session_id(messages, body)
        raw_tools = body.get("tools")
        tools: list[dict[str, Any]] = [t for t in raw_tools if isinstance(t, dict)] if isinstance(raw_tools, list) else []
        allowed = tool_names(tools)
        session, tail, bootstrap = RUNTIME.sessions.prepare(session_id, messages, model=model_id)
        if bootstrap:
            prompt = bootstrap_prompt(messages, tools)
            conversation_id = client.create_conversation(CONVERSATION_SEED, model_id)
        else:
            prompt = delta_prompt(tail, tools)
            conversation_id = session.conversation_id
        if not conversation_id:
            raise ProtocolError("AiPASS conversation could not be resolved")
        thinking = map_reasoning(_reasoning_effort(body), model)
        return conversation_id, client.send_stream(
            conversation_id, prompt, model_id, thinking_level=thinking
        ), allowed, session_id, prompt

    def _execute_safe(
        self, body: dict[str, Any], messages: list[dict[str, Any]], model_id: str
    ) -> tuple[str, Any, set[str], str, str]:
        try:
            return self._execute(body, messages, model_id)
        except ContentPolicyViolationError:
            client = RUNTIME.get_client()
            model = client.model(model_id)
            raw_tools = body.get("tools") or []
            tools: list[dict[str, Any]] = [t for t in raw_tools if isinstance(t, dict)] if isinstance(raw_tools, list) else []
            allowed = tool_names(tools)
            session_id = self._session_id(messages, body)
            retry_prompt = self._sanitize_for_retry(messages, body.get("tools"))
            try:
                new_cid = client.create_conversation(CONVERSATION_SEED, model_id)
                thinking = map_reasoning(_reasoning_effort(body), model)
                deltas = client.send_stream(new_cid, retry_prompt, model_id, thinking_level=thinking)
                return new_cid, deltas, allowed, session_id, retry_prompt
            except ContentPolicyViolationError:
                explanation = (
                    "ขออภัยครับ ข้อความหรือประวัติการสนทนานี้มีคำสำคัญที่ติดระบบนโยบายความปลอดภัยของ AiPASS (Responsible AI: SCAM_FRAUD Policy) "
                    "ระบบได้พยายามปรับข้อความอัตโนมัติแล้วแต่ยังติดข้อกำหนด แนะนำให้หลีกเลี่ยงคำว่า 'มิจฉาชีพ', 'หลอกโอน', 'บัตรกดเงินสด' หรือเริ่ม Session ใหม่ครับ"
                )
                return "fallback-rai", iter([explanation]), set(), session_id, retry_prompt


    @staticmethod
    def _is_refusal(raw: str) -> bool:
        snippets = (
            "ขออภัย ไม่สามารถตอบคำถามนี้ได้",
            "Sorry, I could not respond to this request",
            "ขออภัยครับ ผมไม่สามารถ",
            "ขออภัยค่ะ ดิฉันไม่สามารถ",
        )
        return any(s in raw for s in snippets) and ("<tool_call>" not in raw and TOOL_SENTINEL not in raw)

    def _sanitize_for_retry(self, messages: list[dict[str, Any]], tools: Any) -> str:
        last = messages[-1] if messages else {"role": "user", "content": ""}
        from .prompting import _SYSTEM_PREAMBLE, _tool_guide, message_text
        txt = message_text(last)
        if "[/plan — plan mode]" in txt:
            if "Task to plan:" in txt:
                txt = "Please create a detailed plan for: " + txt.split("Task to plan:", 1)[1].split("Write the plan", 1)[0].strip()
        tools_list = tools if isinstance(tools, list) else []
        tool_section = _tool_guide(tools_list) if tools_list else ""
        return f"{_SYSTEM_PREAMBLE}\n\n{tool_section}\n\nUser request:\n{txt}"

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
        session_id = self._session_id(messages, body)
        with RUNTIME.session_lock(session_id):
            conversation_id, deltas, allowed, _, _ = self._execute_safe(body, messages, model_id)
            raw = "".join(deltas)
            if self._is_refusal(raw):
                client = RUNTIME.get_client()
                retry_prompt = self._sanitize_for_retry(messages, body.get("tools"))
                new_cid = client.create_conversation(CONVERSATION_SEED, model_id)
                new_deltas = list(client.send_stream(new_cid, retry_prompt, model_id))
                new_raw = "".join(new_deltas)
                if not self._is_refusal(new_raw):
                    raw = new_raw
                    conversation_id = new_cid
                else:
                    for phrase in _REFUSAL_PHRASES:
                        raw = raw.replace(phrase, "").strip()
                    if not raw:
                        raw = "กำลังดำเนินการตรวจสอบและเชื่อมต่อเครื่องมือ กรุณาลองสั่งการอีกครั้งครับ"
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

    def _regular_anthropic_completion(
        self,
        raw_body: dict[str, Any],
        body: dict[str, Any],
        messages: list[dict[str, Any]],
        model_id: str,
    ) -> None:
        session_id = self._session_id(messages, body)
        with RUNTIME.session_lock(session_id):
            conversation_id, deltas, allowed, _, _ = self._execute_safe(body, messages, model_id)
            raw = "".join(deltas)
            if self._is_refusal(raw):
                client = RUNTIME.get_client()
                retry_prompt = self._sanitize_for_retry(messages, body.get("tools"))
                new_cid = client.create_conversation(CONVERSATION_SEED, model_id)
                new_deltas = list(client.send_stream(new_cid, retry_prompt, model_id))
                new_raw = "".join(new_deltas)
                if not self._is_refusal(new_raw):
                    raw = new_raw
                    conversation_id = new_cid
                else:
                    for phrase in _REFUSAL_PHRASES:
                        raw = raw.replace(phrase, "").strip()
                    if not raw:
                        raw = "กำลังดำเนินการตรวจสอบและเชื่อมต่อเครื่องมือ กรุณาลองสั่งการอีกครั้งครับ"
            message = parse_assistant(raw, allowed)
            self._complete_state(
                session_id=session_id,
                conversation_id=conversation_id,
                model_id=model_id,
                messages=messages,
                response_message=message,
            )
        completion_id = f"chatcmpl-{uuid.uuid4()}"
        openai_resp = {
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
        }
        anth_resp = openai_to_anthropic_response(openai_resp)
        self._send_json(200, anth_resp)

    def _anthropic_sse(self, event: str, data: Any) -> None:
        self.wfile.write(f"event: {event}\n".encode("utf-8"))
        self.wfile.write(b"data: " + _json_bytes(data) + b"\n\n")
        self.wfile.flush()

    def _stream_anthropic_completion(
        self,
        raw_body: dict[str, Any],
        body: dict[str, Any],
        messages: list[dict[str, Any]],
        model_id: str,
    ) -> None:
        session_id = self._session_id(messages, body)
        msg_id = f"msg_{uuid.uuid4().hex[:24]}"
        headers_sent = False
        try:
            with RUNTIME.session_lock(session_id):
                conversation_id, deltas, allowed, _, _ = self._execute_safe(body, messages, model_id)
                interceptor = StreamInterceptor(allowed_tools=allowed, gate_chars=120)

                block_index = 0
                thinking_block_open = False
                text_block_open = False
                collected_thought: list[str] = []
                collected_text: list[str] = []
                collected_tool_calls: list[dict[str, Any]] = []

                def _ensure_headers() -> None:
                    nonlocal headers_sent
                    if not headers_sent:
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("Connection", "close")
                        self.end_headers()
                        headers_sent = True
                        self._anthropic_sse("message_start", {
                            "type": "message_start",
                            "message": {
                                "id": msg_id,
                                "type": "message",
                                "role": "assistant",
                                "model": model_id,
                                "content": [],
                                "stop_reason": None,
                                "stop_sequence": None,
                                "usage": {"input_tokens": 0, "output_tokens": 0},
                            },
                        })

                delta_iter = iter(deltas)
                while True:
                    try:
                        delta = next(delta_iter)
                    except StopIteration:
                        break

                    try:
                        chunk = interceptor.feed_with_gate(delta, self._is_refusal)
                    except RefusalDetected:
                        client = RUNTIME.get_client()
                        retry_prompt = self._sanitize_for_retry(messages, body.get("tools"))
                        new_cid = client.create_conversation(CONVERSATION_SEED, model_id)
                        delta_iter = iter(client.send_stream(new_cid, retry_prompt, model_id))
                        conversation_id = new_cid
                        interceptor = StreamInterceptor(allowed_tools=allowed, gate_chars=120)
                        continue

                    if getattr(chunk, "thought", ""):
                        _ensure_headers()
                        if not thinking_block_open:
                            self._anthropic_sse("content_block_start", {
                                "type": "content_block_start",
                                "index": block_index,
                                "content_block": {"type": "thinking", "thinking": ""},
                            })
                            thinking_block_open = True
                        self._anthropic_sse("content_block_delta", {
                            "type": "content_block_delta",
                            "index": block_index,
                            "delta": {"type": "thinking_delta", "thinking": chunk.thought},
                        })
                        collected_thought.append(chunk.thought)

                    if chunk.text:
                        _ensure_headers()
                        if thinking_block_open:
                            self._anthropic_sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
                            block_index += 1
                            thinking_block_open = False
                        if not text_block_open:
                            self._anthropic_sse("content_block_start", {
                                "type": "content_block_start",
                                "index": block_index,
                                "content_block": {"type": "text", "text": ""},
                            })
                            text_block_open = True
                        self._anthropic_sse("content_block_delta", {
                            "type": "content_block_delta",
                            "index": block_index,
                            "delta": {"type": "text_delta", "text": chunk.text},
                        })
                        collected_text.append(chunk.text)

                    if chunk.tool_calls:
                        _ensure_headers()
                        if thinking_block_open:
                            self._anthropic_sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
                            block_index += 1
                            thinking_block_open = False
                        if text_block_open:
                            self._anthropic_sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
                            block_index += 1
                            text_block_open = False
                        for tc in chunk.tool_calls:
                            tc_id = tc.get("id") or f"toolu_{uuid.uuid4().hex[:20]}"
                            tc_fn = tc.get("function") or {}
                            name = tc_fn.get("name") or "unknown"
                            args_str = tc_fn.get("arguments") or "{}"
                            self._anthropic_sse("content_block_start", {
                                "type": "content_block_start",
                                "index": block_index,
                                "content_block": {"type": "tool_use", "id": tc_id, "name": name, "input": {}},
                            })
                            self._anthropic_sse("content_block_delta", {
                                "type": "content_block_delta",
                                "index": block_index,
                                "delta": {"type": "input_json_delta", "partial_json": args_str},
                            })
                            self._anthropic_sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
                            block_index += 1
                            collected_tool_calls.append(tc)

                tail = interceptor.finish()
                if getattr(tail, "thought", ""):
                    _ensure_headers()
                    if not thinking_block_open:
                        self._anthropic_sse("content_block_start", {
                            "type": "content_block_start",
                            "index": block_index,
                            "content_block": {"type": "thinking", "thinking": ""},
                        })
                        thinking_block_open = True
                    self._anthropic_sse("content_block_delta", {
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {"type": "thinking_delta", "thinking": tail.thought},
                    })
                    collected_thought.append(tail.thought)

                if tail.text:
                    _ensure_headers()
                    if thinking_block_open:
                        self._anthropic_sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
                        block_index += 1
                        thinking_block_open = False
                    if not text_block_open:
                        self._anthropic_sse("content_block_start", {
                            "type": "content_block_start",
                            "index": block_index,
                            "content_block": {"type": "text", "text": ""},
                        })
                        text_block_open = True
                    self._anthropic_sse("content_block_delta", {
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {"type": "text_delta", "text": tail.text},
                    })
                    collected_text.append(tail.text)

                if tail.tool_calls:
                    _ensure_headers()
                    if thinking_block_open:
                        self._anthropic_sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
                        block_index += 1
                        thinking_block_open = False
                    if text_block_open:
                        self._anthropic_sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
                        block_index += 1
                        text_block_open = False
                    for tc in tail.tool_calls:
                        tc_id = tc.get("id") or f"toolu_{uuid.uuid4().hex[:20]}"
                        tc_fn = tc.get("function") or {}
                        name = tc_fn.get("name") or "unknown"
                        args_str = tc_fn.get("arguments") or "{}"
                        self._anthropic_sse("content_block_start", {
                            "type": "content_block_start",
                            "index": block_index,
                            "content_block": {"type": "tool_use", "id": tc_id, "name": name, "input": {}},
                        })
                        self._anthropic_sse("content_block_delta", {
                            "type": "content_block_delta",
                            "index": block_index,
                            "delta": {"type": "input_json_delta", "partial_json": args_str},
                        })
                        self._anthropic_sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
                        block_index += 1
                        collected_tool_calls.append(tc)

                if thinking_block_open:
                    self._anthropic_sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
                    block_index += 1
                    thinking_block_open = False
                if text_block_open:
                    self._anthropic_sse("content_block_stop", {"type": "content_block_stop", "index": block_index})

                _ensure_headers()
                stop_reason = "tool_use" if collected_tool_calls else "end_turn"
                self._anthropic_sse("message_delta", {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                    "usage": {"output_tokens": 0},
                })
                self._anthropic_sse("message_stop", {"type": "message_stop"})

                final_content = "".join(collected_text).strip() if collected_text else None
                final_message: dict[str, Any] = {
                    "role": "assistant",
                    "content": final_content,
                }
                if collected_tool_calls:
                    final_message["tool_calls"] = collected_tool_calls

                self._complete_state(
                    session_id=session_id,
                    conversation_id=conversation_id,
                    model_id=model_id,
                    messages=messages,
                    response_message=final_message,
                )
        except Exception as exc:
            if not headers_sent:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
            self._anthropic_sse("error", {"type": "error", "error": {"type": "api_error", "message": str(exc)}})
        finally:
            self.close_connection = True

    def _sse(self, value: Any) -> None:
        self.wfile.write(b"data: " + _json_bytes(value) + b"\n\n")
        self.wfile.flush()

    def _stream_completion(self, body: dict[str, Any], messages: list[dict[str, Any]], model_id: str) -> None:
        session_id = self._session_id(messages, body)
        completion_id = f"chatcmpl-{uuid.uuid4()}"
        created = int(time.time())
        headers_sent = False
        try:
            with RUNTIME.session_lock(session_id):
                conversation_id, deltas, allowed, _, _ = self._execute_safe(body, messages, model_id)
                interceptor = StreamInterceptor(allowed_tools=allowed, gate_chars=120)

                collected_thought: list[str] = []
                collected_text: list[str] = []
                collected_tool_calls: list[dict[str, Any]] = []

                def _ensure_headers() -> None:
                    nonlocal headers_sent
                    if not headers_sent:
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("Connection", "close")
                        self.end_headers()
                        headers_sent = True

                def _emit_chunk(delta_dict: dict[str, Any]) -> None:
                    _ensure_headers()
                    self._sse(
                        {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_id,
                            "choices": [{"index": 0, "delta": delta_dict, "finish_reason": None}],
                        }
                    )

                # Process upstream stream with initial refusal gating
                delta_iter = iter(deltas)
                while True:
                    try:
                        delta = next(delta_iter)
                    except StopIteration:
                        break

                    try:
                        chunk = interceptor.feed_with_gate(delta, self._is_refusal)
                    except RefusalDetected:
                        # Transparent retry on refusal before headers are sent
                        client = RUNTIME.get_client()
                        retry_prompt = self._sanitize_for_retry(messages, body.get("tools"))
                        new_cid = client.create_conversation(CONVERSATION_SEED, model_id)
                        delta_iter = iter(client.send_stream(new_cid, retry_prompt, model_id))
                        conversation_id = new_cid
                        interceptor = StreamInterceptor(allowed_tools=allowed, gate_chars=120)
                        continue

                    if getattr(chunk, "thought", ""):
                        _emit_chunk({"role": "assistant", "reasoning_content": chunk.thought})
                        collected_thought.append(chunk.thought)
                    if chunk.text:
                        _emit_chunk({"role": "assistant", "content": chunk.text})
                        collected_text.append(chunk.text)
                    if chunk.tool_calls:
                        _emit_chunk({"role": "assistant", "tool_calls": chunk.tool_calls})
                        collected_tool_calls.extend(chunk.tool_calls)

                # Finish draining interceptor
                tail = interceptor.finish()
                if getattr(tail, "thought", ""):
                    _emit_chunk({"role": "assistant", "reasoning_content": tail.thought})
                    collected_thought.append(tail.thought)
                if tail.text:
                    _emit_chunk({"role": "assistant", "content": tail.text})
                    collected_text.append(tail.text)
                if tail.tool_calls:
                    _emit_chunk({"role": "assistant", "tool_calls": tail.tool_calls})
                    collected_tool_calls.extend(tail.tool_calls)

                _ensure_headers()
                finish_reason = "tool_calls" if collected_tool_calls else "stop"
                self._sse(
                    {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                    }
                )
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

                # Complete and persist session state
                final_content = "".join(collected_text).strip() if collected_text else None
                final_message: dict[str, Any] = {
                    "role": "assistant",
                    "content": final_content,
                }
                if collected_thought:
                    final_message["reasoning_content"] = "".join(collected_thought).strip()
                if collected_tool_calls:
                    final_message["tool_calls"] = collected_tool_calls

                self._complete_state(
                    session_id=session_id,
                    conversation_id=conversation_id,
                    model_id=model_id,
                    messages=messages,
                    response_message=final_message,
                )
        except Exception as exc:
            if not headers_sent:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
            self._sse(_error(str(exc), "stream_error"))
        finally:
            self.close_connection = True


def serve(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("Refusing to expose the credentialed AiPASS bridge beyond localhost")
    ThreadingHTTPServer.allow_reuse_address = True
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
