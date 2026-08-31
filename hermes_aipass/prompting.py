"""English-only bridge instructions and OpenAI tool-call conversion."""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any

TOOL_SENTINEL = "@@HERMES_TOOL_CALL@@"


def message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return "" if content is None else json.dumps(content, ensure_ascii=False)


def message_fingerprint(message: dict[str, Any]) -> str:
    canonical = json.dumps(message, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def response_fingerprint(message: dict[str, Any]) -> str:
    return message_fingerprint(message)


def _tool_guide(tools: list[dict[str, Any]]) -> str:
    if not tools:
        return ""
    compact = [
        {
            "name": tool.get("function", {}).get("name"),
            "description": tool.get("function", {}).get("description"),
            "parameters": tool.get("function", {}).get("parameters"),
        }
        for tool in tools
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
    ]
    return (
        "\n\nAvailable tools:\n"
        + json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
        + "\n\nIf a tool is required, output exactly the sentinel on the first line, "
        + "followed by one compact JSON object and no markdown:\n"
        + TOOL_SENTINEL
        + "\n{\"tool_calls\":[{\"name\":\"tool_name\",\"arguments\":{}}]}\n"
        + "If no tool is required, answer normally. Never invent a tool result."
    )


def format_message(message: dict[str, Any]) -> str:
    role = message.get("role")
    if role == "tool":
        label = message.get("name") or message.get("tool_call_id") or "unknown"
        return f"Tool result ({label}):\n{message_text(message)}"
    if role == "assistant" and message.get("tool_calls"):
        return "Previous tool request:\n" + json.dumps(
            message["tool_calls"], ensure_ascii=False, separators=(",", ":")
        )
    label = {"assistant": "Assistant", "user": "User", "system": "System"}.get(str(role), "Message")
    return f"{label}:\n{message_text(message)}"


def bootstrap_prompt(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
    systems = "\n\n".join(message_text(message) for message in messages if message.get("role") == "system")
    turns = "\n\n".join(format_message(message) for message in messages if message.get("role") != "system")
    return (
        "Act as the language-model backend for a command-line AI agent. "
        "Follow the system instructions and preserve the user's language.\n\n"
        f"System instructions:\n{systems}"
        f"{_tool_guide(tools)}\n\nConversation:\n{turns}"
    )


def delta_prompt(messages: list[dict[str, Any]]) -> str:
    return "Continue the agent conversation using these new events:\n\n" + "\n\n".join(
        format_message(message) for message in messages
    )


def tool_names(tools: list[dict[str, Any]]) -> set[str]:
    return {
        str(tool.get("function", {}).get("name"))
        for tool in tools
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
        and tool.get("function", {}).get("name")
    }


def parse_assistant(raw: str, allowed_tools: set[str]) -> dict[str, Any]:
    stripped = raw.strip()
    if not stripped.startswith(TOOL_SENTINEL):
        return {"role": "assistant", "content": stripped}
    encoded = stripped[len(TOOL_SENTINEL):].strip()
    if encoded.startswith("```json"):
        encoded = encoded[7:].strip()
    elif encoded.startswith("```"):
        encoded = encoded[3:].strip()
    if encoded.endswith("```"):
        encoded = encoded[:-3].strip()
    try:
        parsed = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed AiPASS tool envelope: {exc.msg}") from exc
    calls = parsed.get("tool_calls") if isinstance(parsed, dict) else None
    if not isinstance(calls, list) or not calls:
        raise ValueError("AiPASS tool envelope contains no tool calls")
    converted = []
    for call in calls:
        if not isinstance(call, dict):
            raise ValueError("AiPASS tool call must be an object")
        name = call.get("name")
        if not isinstance(name, str) or name not in allowed_tools:
            raise ValueError(f"AiPASS requested an unavailable tool: {name!r}")
        arguments = call.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Tool arguments for {name} are not valid JSON") from exc
        if not isinstance(arguments, dict):
            raise ValueError(f"Tool arguments for {name} must be an object")
        converted.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
                },
            }
        )
    return {"role": "assistant", "content": None, "tool_calls": converted}

