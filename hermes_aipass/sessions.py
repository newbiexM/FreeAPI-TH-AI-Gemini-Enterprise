"""Persistent session-to-conversation mapping without transcript storage."""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field
from typing import Any

from .paths import atomic_write_json, read_json, sessions_path
from .prompting import message_fingerprint, response_fingerprint


@dataclass
class Session:
    conversation_id: str | None = None
    model: str | None = None
    request_fingerprints: list[str] = field(default_factory=list)
    last_response_fingerprint: str | None = None


class SessionStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        raw = read_json(sessions_path(), {})
        self._sessions: dict[str, Session] = {}
        if isinstance(raw, dict):
            for key, value in raw.items():
                if not isinstance(value, dict):
                    continue
                self._sessions[str(key)] = Session(
                    conversation_id=value.get("conversation_id"),
                    model=value.get("model"),
                    request_fingerprints=list(value.get("request_fingerprints") or []),
                    last_response_fingerprint=value.get("last_response_fingerprint"),
                )

    def get(self, session_id: str) -> Session:
        with self._lock:
            return self._sessions.setdefault(session_id, Session())

    def prepare(self, session_id: str, messages: list[dict[str, Any]]) -> tuple[Session, list[dict[str, Any]], bool]:
        with self._lock:
            session = self.get(session_id)
            current = [message_fingerprint(message) for message in messages]
            previous = session.request_fingerprints
            prefix_matches = len(current) >= len(previous) and current[: len(previous)] == previous
            if not session.conversation_id or not prefix_matches:
                return session, messages, True
            tail = list(messages[len(previous):])
            if tail and tail[0].get("role") == "assistant":
                if message_fingerprint(tail[0]) == session.last_response_fingerprint:
                    tail.pop(0)
            if not tail:
                return session, [messages[-1]] if messages else [], False
            return session, tail, False

    def complete(
        self,
        session_id: str,
        *,
        conversation_id: str,
        model: str,
        request_messages: list[dict[str, Any]],
        response_message: dict[str, Any],
    ) -> None:
        with self._lock:
            session = self.get(session_id)
            session.conversation_id = conversation_id
            session.model = model
            session.request_fingerprints = [message_fingerprint(message) for message in request_messages]
            session.last_response_fingerprint = response_fingerprint(response_message)
            self._save()

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()
            self._save()

    def _save(self) -> None:
        atomic_write_json(sessions_path(), {key: asdict(value) for key, value in self._sessions.items()})
