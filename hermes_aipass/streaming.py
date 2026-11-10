"""Incremental stream interceptor and tolerant tool-calling parser."""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Callable, NamedTuple

STREAM_KEEP = 32

MARKER_NONE = -1
MARKER_INCOMPLETE = -2

_START_WORDS = ("TOOL_CALL", "tool_call")
_END_WORDS = ("END_TOOL_CALL", "/tool_call", "tool_call")
_SENTINEL = "@@HERMES_TOOL_CALL@@"


class StreamChunk(NamedTuple):
    text: str
    tool_calls: list[dict[str, Any]]


class RefusalDetected(Exception):
    """Raised when an initial upstream refusal is caught by the gate before bytes are flushed."""

    def __init__(self, raw: str) -> None:
        super().__init__(f"Refusal detected in upstream stream: {raw[:100]!r}")
        self.raw = raw


def _bracket_run_back(s: str, i: int, char: str) -> int:
    n = 0
    while i - n - 1 >= 0 and s[i - n - 1] == char:
        n += 1
    return n


def _bracket_run_forward(s: str, char: str) -> int:
    n = 0
    while n < len(s) and s[n] == char:
        n += 1
    return n


def find_marker(text: str, marker_type: str, final: bool = False) -> tuple[int, int]:
    """Find the earliest occurrence of a tool marker framed by brackets or sentinel.

    Returns:
        (index, length) when found
        (-1, 0) when no marker is present
        (-2, 0) when a trailing candidate is incomplete and needs more stream bytes
    """
    if marker_type == "start":
        # Check sentinel
        idx = text.find(_SENTINEL)
        if idx >= 0:
            return idx, len(_SENTINEL)
        if not final and any(_SENTINEL.startswith(text[-k:]) for k in range(1, len(_SENTINEL)) if len(text) >= k):
            return MARKER_INCOMPLETE, 0

        target_words = _START_WORDS
    else:
        target_words = _END_WORDS

    candidates: list[tuple[int, int]] = []
    
    for word in target_words:
        from_idx = 0
        while from_idx < len(text):
            found = text.find(word, from_idx)
            if found < 0:
                if not final:
                    for k in range(1, len(word)):
                        prefix = word[:k]
                        if text.endswith(prefix):
                            lead = _bracket_run_back(text, len(text) - len(prefix), "<")
                            if 1 <= lead <= 4:
                                return MARKER_INCOMPLETE, 0
                break

            lead = _bracket_run_back(text, found, "<")
            if lead < 1 or lead > 4:
                from_idx = found + len(word)
                continue

            after = text[found + len(word):]
            trail = _bracket_run_forward(after, ">")
            
            if trail > 4:
                from_idx = found + len(word)
                continue

            if trail == len(after) and not final:
                if trail < 1 or trail < 4:
                    return MARKER_INCOMPLETE, 0

            if trail >= 1:
                start_pos = found - lead
                total_len = lead + len(word) + trail
                candidates.append((start_pos, total_len))

            from_idx = found + len(word)

    if candidates:
        candidates.sort(key=lambda c: c[0])
        return candidates[0]

    if not final:
        trailing_brackets = _bracket_run_back(text, len(text), "<")
        if 1 <= trailing_brackets <= 4:
            return MARKER_INCOMPLETE, 0

    return MARKER_NONE, 0


def _strip_fences(raw: str) -> str:
    s = raw.strip()
    if s.startswith("```json"):
        s = s[7:].strip()
    elif s.startswith("```"):
        s = s[3:].strip()
    if s.endswith("```"):
        s = s[:-3].strip()
    return s


def _parse_tool_payload(raw: str, allowed_tools: set[str] | None = None) -> list[tuple[str, dict[str, Any]]]:
    """Parse JSON payload and return list of (name, arguments) tuples."""
    clean = _strip_fences(raw)
    try:
        data = json.loads(clean)
    except Exception:
        return []

    calls: list[tuple[str, dict[str, Any]]] = []
    if isinstance(data, dict):
        if "tool_calls" in data and isinstance(data["tool_calls"], list):
            for tc in data["tool_calls"]:
                if isinstance(tc, dict):
                    name = tc.get("name") or (tc.get("function", {}).get("name") if isinstance(tc.get("function"), dict) else None)
                    args = tc.get("arguments") or (tc.get("function", {}).get("arguments") if isinstance(tc.get("function"), dict) else {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {}
                    if name and (allowed_tools is None or name in allowed_tools):
                        calls.append((str(name), args if isinstance(args, dict) else {}))
            return calls

        # Single object
        name = data.get("name") or data.get("tool")
        args = data.get("arguments") or data.get("parameters")
        if not name and isinstance(data.get("function"), dict):
            fn = data["function"]
            name = fn.get("name")
            args = fn.get("arguments")

        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}

        if name and (allowed_tools is None or name in allowed_tools):
            calls.append((str(name), args if isinstance(args, dict) else {}))

    return calls


class StreamInterceptor:
    """Incremental stream interceptor that separates plain text from tool-call markers."""

    def __init__(self, allowed_tools: set[str] | None = None, gate_chars: int = 120) -> None:
        self.allowed_tools = allowed_tools
        self.gate_chars = gate_chars
        self._gated: bool = True
        self._gate_buffer: str = ""
        self._buffer: str = ""
        self._offset: int = 0
        self._call_index: int = 0
        self._pending_sep: bool = False
        self._in_tool_block: bool = False
        self._tool_start_len: int = 0
        self._tool_body_start: int = 0

    def feed_with_gate(self, chunk: str, refusal_checker: Callable[[str], bool]) -> StreamChunk:
        if self._gated:
            self._gate_buffer += chunk
            if refusal_checker(self._gate_buffer):
                raise RefusalDetected(self._gate_buffer)
            if len(self._gate_buffer) >= self.gate_chars:
                self._gated = False
                unbuffered = self._gate_buffer
                self._gate_buffer = ""
                return self.feed(unbuffered)
            # Check if a tool call start marker is already present in gate buffer
            s_idx, _ = find_marker(self._gate_buffer, "start", final=False)
            if s_idx >= 0:
                self._gated = False
                unbuffered = self._gate_buffer
                self._gate_buffer = ""
                return self.feed(unbuffered)
            return StreamChunk(text="", tool_calls=[])

        return self.feed(chunk)

    def feed(self, chunk: str) -> StreamChunk:
        self._buffer += chunk
        return self._drain(final=False)

    def finish(self) -> StreamChunk:
        if self._gated and self._gate_buffer:
            self._gated = False
            unbuffered = self._gate_buffer
            self._gate_buffer = ""
            self._buffer += unbuffered
        chunk = self._drain(final=True)
        self._offset = len(self._buffer)
        return chunk

    def _drain(self, final: bool) -> StreamChunk:
        content_parts: list[str] = []
        emitted_calls: list[dict[str, Any]] = []

        while True:
            # Swallow stray fences or whitespace right after a tool call closed
            if self._pending_sep:
                while self._offset < len(self._buffer) and self._buffer[self._offset] in " \t\r\n":
                    self._offset += 1
                if self._buffer[self._offset:].startswith("```"):
                    end_line = self._buffer.find("\n", self._offset)
                    if end_line >= 0:
                        self._offset = end_line + 1
                    elif final:
                        self._offset = len(self._buffer)
                    else:
                        break
                self._pending_sep = False

            if not self._in_tool_block:
                rest = self._buffer[self._offset:]
                start, marker_len = find_marker(rest, "start", final)

                if start == MARKER_INCOMPLETE:
                    # Incomplete candidate at tail of buffer: stop and wait for more stream bytes
                    break

                if start < 0:
                    # No start marker found
                    if final:
                        if rest:
                            content_parts.append(rest)
                            self._offset = len(self._buffer)
                        break
                    
                    if len(rest) > STREAM_KEEP:
                        cut = len(rest) - STREAM_KEEP
                        content_parts.append(rest[:cut])
                        self._offset += cut
                    break

                # Start marker found!
                if start > 0:
                    content_parts.append(rest[:start])
                    self._offset += start

                self._in_tool_block = True
                self._tool_start_len = marker_len
                self._tool_body_start = self._offset + marker_len
                self._offset = self._tool_body_start

            # We are inside a tool block: look for end marker
            body_rest = self._buffer[self._offset:]
            end_idx, end_marker_len = find_marker(body_rest, "end", final)

            if end_idx == MARKER_INCOMPLETE:
                break

            if end_idx < 0:
                if final:
                    # Unterminated block at stream completion
                    raw_payload = self._buffer[self._tool_body_start:]
                    parsed = _parse_tool_payload(raw_payload, self.allowed_tools)
                    if parsed:
                        for name, args in parsed:
                            emitted_calls.append({
                                "index": self._call_index,
                                "id": f"call_{uuid.uuid4().hex[:24]}",
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps(args, ensure_ascii=False, separators=(",", ":")),
                                },
                            })
                            self._call_index += 1
                    else:
                        content_parts.append(self._buffer[self._tool_body_start - self._tool_start_len:])
                    self._offset = len(self._buffer)
                    self._in_tool_block = False
                    break
                # Wait for more data
                break

            # End marker found!
            end_pos = self._offset + end_idx
            raw_payload = self._buffer[self._tool_body_start:end_pos]
            parsed = _parse_tool_payload(raw_payload, self.allowed_tools)

            if parsed:
                for name, args in parsed:
                    emitted_calls.append({
                        "index": self._call_index,
                        "id": f"call_{uuid.uuid4().hex[:24]}",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(args, ensure_ascii=False, separators=(",", ":")),
                        },
                    })
                    self._call_index += 1
            else:
                # Could not parse tool payload: fall back to text
                content_parts.append(self._buffer[self._tool_body_start - self._tool_start_len:end_pos + end_marker_len])

            self._offset = end_pos + end_marker_len
            self._in_tool_block = False
            self._pending_sep = True

        return StreamChunk(text="".join(content_parts), tool_calls=emitted_calls)
