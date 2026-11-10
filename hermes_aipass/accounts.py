"""Multi-account resilience pool for Gemini Enterprise (adapted from qwen-free-api)."""

from __future__ import annotations

import os
import threading
import time
from typing import Any


class AllAccountsUnavailableError(Exception):
    """Raised when all configured accounts are dead or cooling down."""


class Account:
    """Represents a single authenticated identity with its rate limit state."""

    def __init__(self, id: int, token: str, name: str | None = None) -> None:
        self.id = id
        self.token = token.strip()
        self.name = name or f"account-{id}"
        self._lock = threading.Lock()
        self.cooldown_until: float = 0.0
        self.consecutive_429s: int = 0
        self.dead: bool = False
        self.last_error: str = ""
        self.requests_count: int = 0
        self.rate_limit_hits: int = 0

    def is_available(self) -> bool:
        with self._lock:
            return not self.dead and time.time() >= self.cooldown_until

    def cooldown_left(self) -> float:
        with self._lock:
            left = self.cooldown_until - time.time()
            return max(0.0, left)

    def report_ok(self) -> None:
        with self._lock:
            self.consecutive_429s = 0
            self.cooldown_until = 0.0
            self.last_error = ""
            self.requests_count += 1

    def report_429(self, base_cooldown_sec: float = 60.0) -> float:
        with self._lock:
            self.consecutive_429s += 1
            self.rate_limit_hits += 1
            # Exponential cooldown: base * 2^(streak - 1), capped at 30 min (1800s)
            duration = min(1800.0, base_cooldown_sec * (2 ** (self.consecutive_429s - 1)))
            self.cooldown_until = time.time() + duration
            return duration

    def report_dead(self, reason: str) -> None:
        with self._lock:
            self.dead = True
            self.last_error = reason


class AccountPool:
    """Manages round-robin rotation, exponential cooldown, and failover across accounts."""

    def __init__(self, accounts: list[Account]) -> None:
        self.accounts = accounts
        self._cursor = 0
        self._lock = threading.Lock()

    @classmethod
    def from_tokens(cls, tokens: list[str]) -> AccountPool:
        accts = [Account(id=i + 1, token=t) for i, t in enumerate(tokens) if t.strip()]
        return cls(accts)

    @classmethod
    def from_env(cls, default_token: str | None = None) -> AccountPool:
        raw = os.environ.get("AIPASS_TOKENS") or os.environ.get("GEMINI_TOKENS") or ""
        tokens = [t.strip() for t in raw.split(",") if t.strip()]
        if not tokens and default_token:
            tokens = [default_token]
        return cls.from_tokens(tokens)

    def get_account(self, wait_timeout: float = 0.0) -> Account:
        """Return the next healthy available account using round-robin selection."""
        deadline = time.time() + wait_timeout
        while True:
            with self._lock:
                if not self.accounts:
                    raise AllAccountsUnavailableError("No accounts configured in pool")

                total = len(self.accounts)
                for _ in range(total):
                    acct = self.accounts[self._cursor % total]
                    self._cursor = (self._cursor + 1) % total
                    if acct.is_available():
                        return acct

                # Check if all accounts are dead
                all_dead = all(a.dead for a in self.accounts)
                if all_dead:
                    raise AllAccountsUnavailableError("All accounts in pool are marked dead (expired/revoked)")

            if time.time() >= deadline:
                raise AllAccountsUnavailableError("All accounts are currently in rate-limit cooldown")

            time.sleep(min(0.5, deadline - time.time()))

    def report_ok(self, token: str) -> None:
        for acct in self.accounts:
            if acct.token == token:
                acct.report_ok()
                break

    def report_429(self, token: str, base_cooldown_sec: float = 60.0) -> None:
        for acct in self.accounts:
            if acct.token == token:
                acct.report_429(base_cooldown_sec)
                break
