"""Anthropic Messages API compatibility adapter for Hermes Gemini Enterprise bridge."""

from __future__ import annotations

import json
from typing import Any

_CLAUDE_TO_GEMINI = {
    "claude-3-7-sonnet": "gemini-3.8-flash",
    "claude-3-5-sonnet": "gemini-3.8-flash",
    "claude-3-5-haiku": "gemini-3.5-flash-lite",
    "claude-3-opus": "gemini-4-argon",
    "claude-sonnet": "gemini-3.8-flash",
    "claude-haiku": "gemini-3.5-flash-lite",
    "claude-opus": "gemini-4-argon",
}


def map_anthropic_model(model: str) -> str:
    raw = (model or "").lower().strip()
    for prefix, target in _CLAUDE_TO_GEMINI.items():
        if raw.startswith(prefix):
            return target
    if "gemini" in raw:
        return raw
    return "gemini-3.8-flash"


def estimate_tokens(body: dict[str, Any]) -> int:
    """Rough Anthropic token estimation without upstream call (~4 chars per token)."""
    messages = body.get("messages") or []
    system = body.get("system") or ""
    tools = body.get("tools") or []

    serialized = json.dumps({"m": messages, "s": system, "t": tools}, ensure_ascii=False)
    chars = len(serialized)
    return max(1, (chars + 3) // 4)


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return ""


def anthropic_to_openai_request(
    body: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    """Convert Anthropic /v1/messages body into normalized OpenAI request format."""
    model_id = map_anthropic_model(str(body.get("model") or "gemini-3.8-flash"))
    out_messages: list[dict[str, Any]] = []

    # 1. System instructions
    system_raw = body.get("system")
    if system_raw:
        system_text = _extract_text(system_raw)
        if system_text:
            out_messages.append({"role": "system", "content": system_text})

    # 2. Messages translation
    for msg in body.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "user").lower()
        content = msg.get("content")

        if isinstance(content, str):
            out_messages.append({"role": role, "content": content})
            continue

        if isinstance(content, list):
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []

            for block in content:
                if not isinstance(block, dict):
                    continue
                b_type = block.get("type")
                if b_type == "text":
                    text_parts.append(str(block.get("text") or ""))
                elif b_type == "tool_use":
                    call_id = block.get("id") or "call_unknown"
                    name = block.get("name") or "unknown"
                    inp = block.get("input") or {}
                    tool_calls.append({
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(inp, ensure_ascii=False, separators=(",", ":")),
                        },
                    })
                elif b_type == "tool_result":
                    # Tool result translates to a tool role message in OpenAI format
                    res_content = block.get("content")
                    if isinstance(res_content, list):
                        res_text = _extract_text(res_content)
                    else:
                        res_text = str(res_content or "")
                    out_messages.append({
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id") or "call_unknown",
                        "content": res_text,
                    })

            if text_parts or tool_calls:
                msg_dict: dict[str, Any] = {
                    "role": role,
                    "content": "\n".join(text_parts) if text_parts else None,
                }
                if tool_calls:
                    msg_dict["tool_calls"] = tool_calls
                out_messages.append(msg_dict)

    # 3. Tools translation
    openai_tools: list[dict[str, Any]] = []
    for tool in body.get("tools") or []:
        if isinstance(tool, dict):
            name = tool.get("name")
            if name:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": tool.get("description") or "",
                        "parameters": tool.get("input_schema") or {},
                    },
                })

    openai_body: dict[str, Any] = {
        "model": model_id,
        "messages": out_messages,
        "stream": bool(body.get("stream")),
    }
    if openai_tools:
        openai_body["tools"] = openai_tools

    return openai_body, out_messages, model_id


def openai_to_anthropic_response(openai_resp: dict[str, Any]) -> dict[str, Any]:
    """Convert OpenAI chat.completion dictionary to Anthropic message format."""
    choices = openai_resp.get("choices") or [{}]
    choice = choices[0] if choices else {}
    msg = choice.get("message") or {}
    model_id = openai_resp.get("model") or "gemini-3.8-flash"
    completion_id = str(openai_resp.get("id") or "msg_unknown")
    if completion_id.startswith("chatcmpl-"):
        completion_id = "msg_" + completion_id[len("chatcmpl-"):]

    content_blocks: list[dict[str, Any]] = []
    if msg.get("content"):
        content_blocks.append({
            "type": "text",
            "text": str(msg["content"]),
        })

    tool_calls = msg.get("tool_calls") or []
    for tc in tool_calls:
        if isinstance(tc, dict):
            fn = tc.get("function") or {}
            args_raw = fn.get("arguments") or "{}"
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except Exception:
                args = {}
            content_blocks.append({
                "type": "tool_use",
                "id": tc.get("id") or "call_unknown",
                "name": fn.get("name") or "unknown",
                "input": args if isinstance(args, dict) else {},
            })

    stop_reason = "tool_use" if tool_calls else "end_turn"
    usage = openai_resp.get("usage") or {}

    return {
        "id": completion_id,
        "type": "message",
        "role": "assistant",
        "model": model_id,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }

