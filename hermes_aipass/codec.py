"""Decode Remix/devalue loader payloads and AiPASS data streams."""

from __future__ import annotations

import json
import math
from typing import Any, Iterable, Iterator


def decode_flattened(value: Any) -> Any:
    """Hydrate the flattened reference-array shape returned by Remix loaders."""
    if not isinstance(value, list):
        return value
    memo: dict[int, Any] = {}

    def hydrate(reference: Any) -> Any:
        if not isinstance(reference, int):
            return reference
        if reference < 0:
            return {
                -1: None,
                -2: None,
                -3: math.nan,
                -4: math.inf,
                -5: -math.inf,
                -6: -0.0,
            }.get(reference)
        if reference >= len(value):
            return None
        if reference in memo:
            return memo[reference]
        raw = value[reference]
        if isinstance(raw, list):
            result: list[Any] = []
            memo[reference] = result
            result.extend(hydrate(item) for item in raw)
            return result
        if isinstance(raw, dict):
            result_dict: dict[str, Any] = {}
            memo[reference] = result_dict
            for raw_key, raw_value in raw.items():
                if isinstance(raw_key, str) and raw_key.startswith("_") and raw_key[1:].isdigit():
                    key = hydrate(int(raw_key[1:]))
                else:
                    key = raw_key
                result_dict[str(key)] = hydrate(raw_value)
            return result_dict
        memo[reference] = raw
        return raw

    return hydrate(0)


def decode_loader_bytes(payload: bytes) -> Any:
    parsed = json.loads(payload.decode("utf-8"))
    return decode_flattened(parsed)


def walk(value: Any) -> Iterator[Any]:
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


def extract_stream_events(lines: Iterable[bytes]) -> Iterator[dict[str, Any]]:
    """Parse Vercel AI UI-message SSE and its legacy data-stream variant."""
    for raw_line in lines:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line or line.startswith(":"):
            continue
        if line.startswith("data:"):
            data = line[5:].strip()
            if data == "[DONE]":
                yield {"type": "done"}
                continue
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                yield event
            continue
        # AI SDK 3/4 data stream: 0:"text delta", 3:"error"
        if len(line) > 2 and line[1] == ":":
            code, encoded = line[0], line[2:]
            try:
                decoded = json.loads(encoded)
            except json.JSONDecodeError:
                continue
            if code == "0" and isinstance(decoded, str):
                yield {"type": "text-delta", "delta": decoded}
            elif code == "3":
                yield {"type": "error", "errorText": str(decoded)}


def text_delta(event: dict[str, Any]) -> str:
    kind = event.get("type")
    if kind in {"text-delta", "reasoning-delta"}:
        delta = event.get("delta")
        return delta if isinstance(delta, str) else ""
    if kind == "text" and isinstance(event.get("text"), str):
        return event["text"]
    return ""

