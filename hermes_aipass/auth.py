"""AiPASS authentication storage and validation."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from http.cookies import SimpleCookie
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .paths import atomic_write_json, metadata_path

BASE_URL = "https://de.aipass.net"
KEYCHAIN_SERVICE = "hermes-aipass-auth"
KEYCHAIN_ACCOUNT = os.environ.get("USER") or "hermes"
ESSENTIAL_COOKIES = {
    "__Secure-ai_passport_auth.session_token",
    "__aipassport_domain",
    "__aipassport_project_theme_v2",
    "aip_device_id",
    "app_lang",
}


class AuthError(RuntimeError):
    pass


@dataclass
class AuthState:
    cookies: dict[str, str]
    expires_at: str | None = None

    @property
    def cookie_header(self) -> str:
        return "; ".join(f"{key}={value}" for key, value in self.cookies.items())

    def as_json(self) -> str:
        return json.dumps(
            {"cookies": self.cookies, "expires_at": self.expires_at},
            ensure_ascii=False,
            separators=(",", ":"),
        )


class KeychainStore:
    """Store reusable web-session credentials in macOS Keychain."""

    def _require_macos(self) -> None:
        if platform.system() != "Darwin":
            raise AuthError("hermes-aipass v0.1 supports macOS Keychain only")

    def save(self, state: AuthState) -> None:
        self._require_macos()
        subprocess.run(
            [
                "security", "add-generic-password", "-U",
                "-a", KEYCHAIN_ACCOUNT,
                "-s", KEYCHAIN_SERVICE,
                "-w", state.as_json(),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        atomic_write_json(metadata_path(), {"expires_at": state.expires_at})

    def load(self) -> AuthState:
        self._require_macos()
        result = subprocess.run(
            [
                "security", "find-generic-password",
                "-a", KEYCHAIN_ACCOUNT,
                "-s", KEYCHAIN_SERVICE,
                "-w",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not result.stdout.strip():
            raise AuthError("AiPASS is not authenticated; run `hermes aipass login`")
        try:
            raw = json.loads(result.stdout)
            cookies = raw["cookies"]
            if not isinstance(cookies, dict) or not cookies:
                raise ValueError
            return AuthState(
                cookies={str(k): str(v) for k, v in cookies.items()},
                expires_at=raw.get("expires_at"),
            )
        except (ValueError, TypeError, KeyError) as exc:
            raise AuthError("AiPASS Keychain entry is invalid; run `hermes aipass login`") from exc

    def delete(self) -> None:
        self._require_macos()
        subprocess.run(
            [
                "security", "delete-generic-password",
                "-a", KEYCHAIN_ACCOUNT,
                "-s", KEYCHAIN_SERVICE,
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        metadata_path().unlink(missing_ok=True)


def state_from_cdp_cookies(cookies: Iterable[dict]) -> AuthState:
    selected: dict[str, str] = {}
    auth_expiry: float | None = None
    for item in cookies:
        name = str(item.get("name") or "")
        domain = str(item.get("domain") or "")
        if name not in ESSENTIAL_COOKIES or not domain.endswith("aipass.net"):
            continue
        value = item.get("value")
        if isinstance(value, str) and value:
            selected[name] = value
        if name == "__Secure-ai_passport_auth.session_token":
            try:
                expiry = float(item.get("expires") or 0)
                if expiry > 0:
                    auth_expiry = expiry
            except (TypeError, ValueError):
                pass
    if "__Secure-ai_passport_auth.session_token" not in selected:
        raise AuthError("AiPASS login did not produce an authentication cookie")
    expires_at = (
        datetime.fromtimestamp(auth_expiry, tz=timezone.utc).isoformat()
        if auth_expiry else None
    )
    return AuthState(selected, expires_at)


def apply_set_cookie(state: AuthState, headers: Iterable[str]) -> bool:
    changed = False
    expiry: str | None = state.expires_at
    for header in headers:
        cookie = SimpleCookie()
        try:
            cookie.load(header)
        except Exception:
            continue
        for name, morsel in cookie.items():
            if name not in ESSENTIAL_COOKIES:
                continue
            if morsel.value and state.cookies.get(name) != morsel.value:
                state.cookies[name] = morsel.value
                changed = True
            if name == "__Secure-ai_passport_auth.session_token" and morsel["expires"]:
                try:
                    parsed = parsedate_to_datetime(morsel["expires"])
                    expiry = parsed.astimezone(timezone.utc).isoformat()
                except (TypeError, ValueError):
                    pass
    if expiry != state.expires_at:
        state.expires_at = expiry
        changed = True
    return changed


def validate_auth(state: AuthState, timeout: float = 15) -> dict:
    request = Request(
        f"{BASE_URL}/api/v1/auth/get-session",
        headers={
            "Cookie": state.cookie_header,
            "User-Agent": "Mozilla/5.0 Hermes-AiPASS/0.1",
            "Accept": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
            if response.status != 200 or not isinstance(data, dict) or not data.get("session"):
                raise AuthError("AiPASS session is not valid; run `hermes aipass login`")
            return {"valid": True, "expires_at": state.expires_at}
    except HTTPError as exc:
        if exc.code in {401, 403}:
            raise AuthError("AiPASS session expired; run `hermes aipass login`") from exc
        raise AuthError(f"AiPASS authentication check failed with HTTP {exc.code}") from exc
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise AuthError(f"Could not validate AiPASS authentication: {exc}") from exc

