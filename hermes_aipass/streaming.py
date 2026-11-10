"""Incremental stream interceptor and tolerant tool-calling parser."""

from __future__ import annotations

import re
from typing import Any, Callable, NamedTuple

STREAM_KEEP = 32

MARKER_NONE = -1
MARKER_INCOMPLETE = -2

_START_WORDS = ("TOOL_CALL", "tool_call")
_END_WORDS = ("END_TOOL_CALL", "/tool_call", "tool_call")
_SENTINEL = "@@HERMES_TOOL_CALL@@"


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

    # Scan for words framed by 1..4 '<' and 1..4 '>'
    candidates: list[tuple[int, int]] = []
    
    for word in target_words:
        from_idx = 0
        while from_idx < len(text):
            found = text.find(word, from_idx)
            if found < 0:
                # Check trailing incomplete prefix of the word framed by '<'
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
                # Touches end of buffer, could still receive more '>'
                if trail < 1 or trail < 4:
                    return MARKER_INCOMPLETE, 0

            if trail >= 1:
                start_pos = found - lead
                total_len = lead + len(word) + trail
                candidates.append((start_pos, total_len))

            from_idx = found + len(word)

    if candidates:
        # Return candidate that starts earliest in string
        candidates.sort(key=lambda c: c[0])
        return candidates[0]

    # Check if text ends in bare '<' brackets (could be start of a marker)
    if not final:
        trailing_brackets = _bracket_run_back(text, len(text), "<")
        if 1 <= trailing_brackets <= 4:
            return MARKER_INCOMPLETE, 0

    return MARKER_NONE, 0
