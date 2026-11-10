"""Minimal Chrome DevTools client used only during interactive OAuth login."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import socket
import struct
import subprocess
import time
from pathlib import Path
from urllib.error import URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

from .auth import (
    AuthError,
    AuthState,
    KeychainStore,
    state_from_cdp_cookies,
    validate_auth,
)
from .paths import chrome_profile_dir


class WebSocket:
    def __init__(self, url: str):
        if not url.startswith("ws://"):
            raise AuthError("Only local ws:// DevTools endpoints are supported")
        target = url[5:]
        host_port, path = target.split("/", 1)
        host, port_text = host_port.rsplit(":", 1)
        self.sock = socket.create_connection((host, int(port_text)), timeout=10)
        key = base64.b64encode(os.urandom(16)).decode()
        request = (
            f"GET /{path} HTTP/1.1\r\nHost: {host_port}\r\nUpgrade: websocket\r\n"
            f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
        )
        self.sock.sendall(request.encode())
        response = self._read_until(b"\r\n\r\n")
        expected = base64.b64encode(
            hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()
        ).decode()
        if b" 101 " not in response or expected.encode() not in response:
            raise AuthError("Chrome DevTools WebSocket handshake failed")
        self.next_id = 0

    def _read_until(self, marker: bytes) -> bytes:
        data = b""
        while marker not in data:
            chunk = self.sock.recv(4096)
            if not chunk:
                break
            data += chunk
        return data

    def _send_text(self, text: str) -> None:
        payload = text.encode()
        mask = os.urandom(4)
        length = len(payload)
        header = bytearray([0x81])
        if length < 126:
            header.append(0x80 | length)
        elif length < 65536:
            header.append(0x80 | 126)
            header.extend(struct.pack("!H", length))
        else:
            header.append(0x80 | 127)
            header.extend(struct.pack("!Q", length))
        header.extend(mask)
        masked = bytes(byte ^ mask[i % 4] for i, byte in enumerate(payload))
        self.sock.sendall(header + masked)

    def _recv_text(self) -> str:
        while True:
            first = self.sock.recv(2)
            if len(first) < 2:
                raise AuthError("Chrome DevTools connection closed")
            opcode = first[0] & 0x0F
            length = first[1] & 0x7F
            if length == 126:
                length = struct.unpack("!H", self._recv_exact(2))[0]
            elif length == 127:
                length = struct.unpack("!Q", self._recv_exact(8))[0]
            masked = bool(first[1] & 0x80)
            mask = self._recv_exact(4) if masked else b""
            payload = self._recv_exact(length)
            if masked:
                payload = bytes(byte ^ mask[i % 4] for i, byte in enumerate(payload))
            if opcode == 0x8:
                raise AuthError("Chrome DevTools connection closed")
            if opcode == 0x9:
                self._send_control(0xA, payload)
                continue
            if opcode == 0x1:
                return payload.decode("utf-8")

    def _recv_exact(self, count: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < count:
            chunk = self.sock.recv(count - len(chunks))
            if not chunk:
                raise AuthError("Chrome DevTools connection closed")
            chunks.extend(chunk)
        return bytes(chunks)

    def _send_control(self, opcode: int, payload: bytes) -> None:
        mask = os.urandom(4)
        header = bytes([0x80 | opcode, 0x80 | len(payload)]) + mask
        body = bytes(byte ^ mask[i % 4] for i, byte in enumerate(payload))
        self.sock.sendall(header + body)

    def call(self, method: str, params: dict | None = None) -> dict:
        self.next_id += 1
        call_id = self.next_id
        self._send_text(json.dumps({"id": call_id, "method": method, "params": params or {}}))
        while True:
            message = json.loads(self._recv_text())
            if message.get("id") != call_id:
                continue
            if "error" in message:
                raise AuthError(message["error"].get("message", "Chrome DevTools error"))
            return message.get("result") or {}

    def close(self) -> None:
        try:
            self._send_control(0x8, b"")
        except OSError:
            pass
        self.sock.close()


def _browser_binary() -> Path:
    candidates = [
        Path("/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"),
        Path.home() / "Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
        Path("/usr/bin/brave-browser"),
        Path("/usr/bin/brave"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise AuthError("Brave Browser is required for AiPASS login (Chrome is disallowed)")


def _targets(port: int) -> list[dict]:
    with urlopen(f"http://127.0.0.1:{port}/json/list", timeout=2) as response:
        return json.loads(response.read().decode())


def _new_target(port: int, url: str) -> dict:
    req = Request(f"http://127.0.0.1:{port}/json/new?{url}", method="PUT")
    with urlopen(req, timeout=2) as response:
        return json.loads(response.read().decode())


def _close_target(port: int, target_id: str) -> None:
    try:
        req = Request(f"http://127.0.0.1:{port}/json/close/{target_id}")
        with urlopen(req, timeout=2):
            pass
    except Exception:
        pass


def refresh_session_from_browser(port: int = 9222) -> AuthState | None:
    """Attempt silent non-interactive credential extraction from an already running browser."""
    try:
        pages = _targets(port)
    except Exception:
        return None

    ge_page = next(
        (
            item
            for item in pages
            if item.get("type") == "page"
            and "vertexaisearch.cloud.google" in item.get("url", "")
        ),
        None,
    )
    if not ge_page:
        return None

    page_url = ge_page.get("url", "")
    parsed = urlparse(page_url)
    csesidx = parse_qs(parsed.query).get("csesidx", [None])[0]
    cid_match = re.search(r"/cid/([^/?]+)", parsed.path)
    tenant_id = cid_match.group(1) if cid_match else None

    try:
        client = WebSocket(ge_page["webSocketDebuggerUrl"])
        cookies = client.call(
            "Network.getCookies",
            {"urls": ["https://vertexaisearch.cloud.google", "https://de.aipass.net"]},
        ).get("cookies", [])
        client.close()
    except Exception:
        return None

    if not cookies:
        return None

    state = state_from_cdp_cookies(cookies, csesidx=csesidx, tenant_id=tenant_id)
    KeychainStore().save(state)
    return state


def interactive_login(port: int = 9222, timeout: int = 300) -> dict:
    """Connect to Brave/Chrome DevTools, capture session, and save credentials."""
    running_process = None
    already_running = False
    opened_target_id: str | None = None
    try:
        pages = _targets(port)
        already_running = True
        # If no aipass page is currently open, create a background tab
        if not any(item.get("type") == "page" and "de.aipass.net" in item.get("url", "") for item in pages):
            new_tab = _new_target(port, "https://de.aipass.net/chat")
            opened_target_id = new_tab.get("id")
    except (OSError, URLError):
        pass

    if not already_running:
        profile = chrome_profile_dir()
        profile.mkdir(parents=True, exist_ok=True, mode=0o700)
        running_process = subprocess.Popen(
            [
                str(_browser_binary()),
                f"--user-data-dir={profile}",
                f"--remote-debugging-port={port}",
                "--no-first-run",
                "https://de.aipass.net/chat",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    deadline = time.monotonic() + timeout
    ge_navigated = False
    try:
        while time.monotonic() < deadline:
            if running_process is not None and running_process.poll() is not None:
                raise AuthError("Browser closed before AiPASS login completed")
            try:
                pages = _targets(port)
                # First priority: Gemini Enterprise Google page
                ge_page = next(
                    (
                        item
                        for item in pages
                        if item.get("type") == "page"
                        and "vertexaisearch.cloud.google" in item.get("url", "")
                    ),
                    None,
                )
                if ge_page:
                    page_url = ge_page.get("url", "")
                    parsed = urlparse(page_url)
                    csesidx = parse_qs(parsed.query).get("csesidx", [None])[0]
                    cid_match = re.search(r"/cid/([^/?]+)", parsed.path)
                    tenant_id = cid_match.group(1) if cid_match else None

                    client = WebSocket(ge_page["webSocketDebuggerUrl"])
                    cookies = client.call("Network.getAllCookies").get("cookies", [])
                    if not cookies:
                        cookies = client.call(
                            "Network.getCookies",
                            {"urls": ["https://de.aipass.net", "https://vertexaisearch.cloud.google"]},
                        ).get("cookies", [])
                    state = state_from_cdp_cookies(cookies, csesidx=csesidx, tenant_id=tenant_id)
                    KeychainStore().save(state)
                    validate_auth(state)
                    if not already_running:
                        client.call("Browser.close")
                    client.close()
                    return {
                        "valid": True,
                        "expires_at": state.expires_at,
                        "tenant": "discovery_engine",
                        "csesidx": csesidx,
                    }

                # Second priority: AiPass portal page
                aipass_page = next(
                    (
                        item
                        for item in pages
                        if item.get("type") == "page"
                        and "de.aipass.net" in item.get("url", "")
                    ),
                    None,
                )
                if aipass_page:
                    client = WebSocket(aipass_page["webSocketDebuggerUrl"])
                    session = client.call(
                        "Runtime.evaluate",
                        {
                            "expression": "fetch('/api/v1/auth/get-session').then(r => r.ok)",
                            "awaitPromise": True,
                            "returnByValue": True,
                        },
                    )
                    is_valid = session.get("result", {}).get("value") is True
                    if is_valid:
                        # User is logged in to TH-Ai-Pass: automatically navigate to Gemini Enterprise SSO
                        if not ge_navigated:
                            client.call(
                                "Page.navigate",
                                {"url": "https://de.aipass.net/ge/open"},
                            )
                            ge_navigated = True
                            client.close()
                            time.sleep(1.5)
                            continue
                        else:
                            # If already attempted navigation, capture whatever cookies we have
                            cookies = client.call("Network.getAllCookies").get("cookies", [])
                            if not cookies:
                                cookies = client.call(
                                    "Network.getCookies",
                                    {"urls": ["https://de.aipass.net", "https://vertexaisearch.cloud.google"]},
                                ).get("cookies", [])
                            state = state_from_cdp_cookies(cookies)
                            KeychainStore().save(state)
                            validate_auth(state)
                            if not already_running:
                                client.call("Browser.close")
                            client.close()
                            return {"valid": True, "expires_at": state.expires_at}
                    client.close()
            except (OSError, ValueError, AuthError):
                pass
            time.sleep(1)
        raise AuthError("Timed out waiting for AiPASS login")
    finally:
        if opened_target_id:
            _close_target(port, opened_target_id)
        if running_process is not None:
            try:
                running_process.terminate()
            except OSError:
                pass
