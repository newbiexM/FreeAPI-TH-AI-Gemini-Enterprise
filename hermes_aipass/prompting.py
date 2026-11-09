"""English-only bridge instructions, tool output pruning, and dual OpenAI tool-call conversion."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from typing import Any

TOOL_SENTINEL = "@@HERMES_TOOL_CALL@@"
_TOOL_CALL_REGEX = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

_HISTORICAL_TOOL_RETENTION_COUNT = 8
_HISTORICAL_TOOL_MAX_CHARS = 300
_TOOL_MAX_CHARS_CAP = 20000

_EMPTY_TOOL_RESPONSE_NUDGE = (
    "You just executed tool calls but returned an empty response. "
    "Please process the tool results above and continue with the task."
)


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


def prune_tool_content(text: str, is_recent: bool = True) -> str:
    """Cap oversized tool results and prune older results to preserve context window."""
    if len(text) > _TOOL_MAX_CHARS_CAP:
        text = (
            text[:12000]
            + f"\n[... large tool output truncated: {len(text)} chars ...]\n"
            + text[-3000:]
        )
    if not is_recent and len(text) > _HISTORICAL_TOOL_MAX_CHARS:
        text = (
            text[:200]
            + f"\n[... tool output truncated: {len(text)} chars ...]\n"
            + text[-100:]
        )
    return text


def _is_recovery_nudge(message: dict[str, Any]) -> bool:
    return bool(message.get("_empty_recovery_synthetic")) or (
        message_text(message).strip() == _EMPTY_TOOL_RESPONSE_NUDGE
    )


def _latest_user_text(messages: list[dict[str, Any]]) -> str:
    """Return the text of the last real user message."""
    for msg in reversed(messages):
        if isinstance(msg, dict) and str(msg.get("role") or "").strip().lower() == "user":
            if not _is_recovery_nudge(msg):
                return message_text(msg).strip()
    return ""


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
        "\n\nAvailable tools (OpenAI function schemas):\n"
        + json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
        + "\n\nCRITICAL INSTRUCTIONS FOR TOOLS:\n"
        + "- If you need to execute tools, emit either:\n"
        + "<tool_call>{\"name\":\"tool_name\",\"arguments\":{}}</tool_call>\n"
        + "or output the sentinel on its own line:\n"
        + TOOL_SENTINEL
        + "\n{\"tool_calls\":[{\"name\":\"tool_name\",\"arguments\":{}}]}\n"
        + "- If no tool is required, answer normally.\n"
        + "- NEVER simulate or hallucinate tool results (e.g. 'Tool result (...):'). Real tool results are provided externally by Hermes Agent in the next turn.\n"
        + "- When a tool execution returns an error or non-zero exit code, acknowledge the real error output honestly; do not claim success when an operation failed."
    )


def format_message(message: dict[str, Any], is_recent_tool: bool = True) -> str:
    role = str(message.get("role") or "unknown").strip().lower()
    if role == "tool":
        label = message.get("name") or message.get("tool_call_id") or "unknown"
        raw_text = message_text(message)
        pruned = prune_tool_content(raw_text, is_recent=is_recent_tool)
        return f"Tool result ({label}):\n{pruned}"
    if role == "assistant":
        parts = []
        text = message_text(message).strip()
        if text:
            parts.append(text)
        if message.get("tool_calls"):
            parts.append("Previous tool request:\n" + json.dumps(
                message["tool_calls"], ensure_ascii=False, separators=(",", ":")
            ))
        return "Assistant:\n" + "\n\n".join(parts) if parts else "Assistant:\n"
    label = {"assistant": "Assistant", "user": "User", "system": "System"}.get(role, "Message")
    return f"{label}:\n{message_text(message)}"


def _build_conversation_turns(messages: list[dict[str, Any]]) -> list[str]:
    non_system = [m for m in messages if m.get("role") != "system"]
    tool_indices = [i for i, m in enumerate(non_system) if str(m.get("role") or "").lower() == "tool"]
    recent_tool_set = (
        set(tool_indices[-_HISTORICAL_TOOL_RETENTION_COUNT:])
        if len(tool_indices) > _HISTORICAL_TOOL_RETENTION_COUNT
        else set(tool_indices)
    )
    return [
        format_message(m, is_recent_tool=(i in recent_tool_set))
        for i, m in enumerate(non_system)
    ]


_SYSTEM_PREAMBLE = (
    "You are operating as the language-model inference backend for Hermes Agent on the user's computer. "
    "All filesystem, terminal, and environmental interactions are executed safely and externally by Hermes Agent on the host. "
    "When asked to inspect files, execute commands, read directories, or interact with the system, do NOT refuse or claim you cannot access the computer; "
    "instead, emit the appropriate tool call so Hermes Agent can execute it on your behalf. Follow the system instructions and preserve the user's language."
)


def bootstrap_prompt(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
    systems = "\n\n".join(message_text(message) for message in messages if message.get("role") == "system")
    turns = "\n\n".join(_build_conversation_turns(messages))
    last_msg = messages[-1] if messages else {}
    last_role = str(last_msg.get("role") or "").strip().lower()

    objective_section = ""
    if last_role == "tool":
        user_request = _latest_user_text(messages)
        if user_request:
            objective_section = (
                f"\n\n### ACTIVE OBJECTIVE:\n"
                f"The user request being served is:\n\"{user_request}\"\n\n"
                "INSTRUCTION: Use the tool results ONLY to address the ACTIVE OBJECTIVE above. "
                "When tool execution is complete, answer it directly and specifically. "
                "Do NOT repeat earlier completion reports or summaries."
            )

    return (
        f"{_SYSTEM_PREAMBLE}\n\n"
        f"System instructions:\n{systems}"
        f"{_tool_guide(tools)}"
        f"{objective_section}\n\nConversation:\n{turns}"
    )


def delta_prompt(messages: list[dict[str, Any]]) -> str:
    turns = "\n\n".join(_build_conversation_turns(messages))
    return f"Continue the agent conversation using these new events:\n\n{turns}"


def tool_names(tools: list[dict[str, Any]]) -> set[str]:
    return {
        str(tool.get("function", {}).get("name"))
        for tool in tools
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
        and tool.get("function", {}).get("name")
    }


def parse_assistant(raw: str, allowed_tools: set[str]) -> dict[str, Any]:
    stripped = raw.strip()

    # 1. Dual-Format: Check for embedded <tool_call> tags first
    tool_matches = list(_TOOL_CALL_REGEX.finditer(stripped))
    if tool_matches:
        converted = []
        for match in tool_matches:
            block = match.group(1).strip()
            try:
                parsed = json.loads(block)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed <tool_call> JSON: {exc.msg}") from exc
            
            if not isinstance(parsed, dict):
                continue
            name = parsed.get("name")
            arguments = parsed.get("arguments")
            if not name and isinstance(parsed.get("function"), dict):
                fn = parsed["function"]
                name = fn.get("name")
                arguments = fn.get("arguments")

            if not isinstance(name, str) or name not in allowed_tools:
                raise ValueError(f"AiPASS requested an unavailable tool: {name!r}")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Tool arguments for {name} are not valid JSON") from exc
            if not isinstance(arguments, dict):
                arguments = {}

            call_id = parsed.get("id") or f"call_{uuid.uuid4().hex[:24]}"
            converted.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
                    },
                }
            )

        clean_content = _TOOL_CALL_REGEX.sub("", stripped).strip()
        # Suppress any hallucinated tool result simulation after tool call
        if clean_content and ("Tool result (" in clean_content or "Tool Result (" in clean_content):
            lines = []
            for line in clean_content.splitlines():
                if line.startswith("Tool result (") or line.startswith("Tool Result ("):
                    break
                lines.append(line)
            clean_content = "\n".join(lines).strip()

        return {
            "role": "assistant",
            "content": clean_content if clean_content else None,
            "tool_calls": converted,
        }

    # 2. Sentinel Format: Check for @@HERMES_TOOL_CALL@@
    if stripped.startswith(TOOL_SENTINEL):
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

    return {"role": "assistant", "content": stripped}
