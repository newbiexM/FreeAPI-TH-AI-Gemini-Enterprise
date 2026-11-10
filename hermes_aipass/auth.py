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

from .paths import atomic_write_json, auth_json_path, metadata_path, read_json

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
DISCOVERY_ENGINE_COOKIES = {
    "__Host-C_OSES",
    "__Secure-C_SES",
    "OSES",
    "CSES",
    "SAPISID",
    "SSID",
    "HSID",
    "APISID",
    "SID",
}


class AuthError(RuntimeError):
    pass


@dataclass
class AuthState:
    cookies: dict[str, str]
    expires_at: str | None = None
    csesidx: str | None = None
    tenant_id: str | None = None

    @property
    def cookie_header(self) -> str:
        return "; ".join(f"{key}={value}" for key, value in self.cookies.items())

    def as_json(self) -> str:
        return json.dumps(
            {
                "cookies": self.cookies,
                "expires_at": self.expires_at,
                "csesidx": self.csesidx,
                "tenant_id": self.tenant_id,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )


class KeychainStore:
    """Store reusable web-session credentials in macOS Keychain with file fallback."""

    def save(self, state: AuthState) -> None:
        if platform.system() == "Darwin":
            try:
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
            except Exception:
                pass
        atomic_write_json(auth_json_path(), json.loads(state.as_json()))
        atomic_write_json(metadata_path(), {"expires_at": state.expires_at})

    def load(self) -> AuthState:
        if platform.system() == "Darwin":
            try:
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
                if result.returncode == 0 and result.stdout.strip():
                    raw = json.loads(result.stdout)
                    cookies = raw.get("cookies")
                    if isinstance(cookies, dict) and cookies:
                        return AuthState(
                            cookies={str(k): str(v) for k, v in cookies.items()},
                            expires_at=raw.get("expires_at"),
                            csesidx=raw.get("csesidx"),
                            tenant_id=raw.get("tenant_id"),
                        )
            except Exception:
                pass

        if auth_json_path().exists():
            try:
                raw = read_json(auth_json_path(), {})
                cookies = raw.get("cookies")
                if isinstance(cookies, dict) and cookies:
                    return AuthState(
                        cookies={str(k): str(v) for k, v in cookies.items()},
                        expires_at=raw.get("expires_at"),
                        csesidx=raw.get("csesidx"),
                        tenant_id=raw.get("tenant_id"),
                    )
            except Exception:
                pass
        raise AuthError("AiPASS is not authenticated; run `hermes aipass login`")

    def delete(self) -> None:
        if platform.system() == "Darwin":
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
        auth_json_path().unlink(missing_ok=True)
        metadata_path().unlink(missing_ok=True)


def state_from_cdp_cookies(
    cookies: Iterable[dict],
    *,
    csesidx: str | None = None,
    tenant_id: str | None = None,
) -> AuthState:
    selected: dict[str, str] = {}
    auth_expiry: float | None = None
    has_aipass_auth = False
    has_ge_auth = False

    for item in cookies:
        name = str(item.get("name") or "")
        domain = str(item.get("domain") or "").lower()
        value = item.get("value")
        if not isinstance(value, str) or not value:
            continue

        if name in ESSENTIAL_COOKIES and domain.endswith("aipass.net"):
            selected[name] = value
            if name == "__Secure-ai_passport_auth.session_token":
                has_aipass_auth = True
                try:
                    expiry = float(item.get("expires") or 0)
                    if expiry > 0:
                        auth_expiry = expiry
                except (TypeError, ValueError):
                    pass

        if name in DISCOVERY_ENGINE_COOKIES and ("google" in domain or "cloud.google" in domain):
            selected[name] = value
            has_ge_auth = True
            try:
                expiry = float(item.get("expires") or 0)
                if expiry > 0 and (auth_expiry is None or expiry > auth_expiry):
                    auth_expiry = expiry
            except (TypeError, ValueError):
                pass

    if not has_aipass_auth and not has_ge_auth:
        raise AuthError("Login did not produce an AiPASS or Gemini Enterprise authentication cookie")

    expires_at = (
        datetime.fromtimestamp(auth_expiry, tz=timezone.utc).isoformat()
        if auth_expiry else None
    )
    return AuthState(selected, expires_at, csesidx=csesidx, tenant_id=tenant_id)


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
    if any(k in state.cookies for k in ("__Host-C_OSES", "__Secure-C_SES", "OSES", "CSES")):
        return {"valid": True, "expires_at": state.expires_at, "tenant": "discovery_engine"}

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

