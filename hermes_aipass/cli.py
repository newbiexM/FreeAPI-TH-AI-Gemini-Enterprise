"""Native ``hermes aipass`` command tree and macOS daemon management."""

from __future__ import annotations

import argparse
import json
import os
import platform
import plistlib
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from . import __version__
from .auth import AuthError, KeychainStore, validate_auth
from .cdp import interactive_login
from .client import AiPassClient, ProtocolError
from .paths import hermes_home, launch_agent_path, logs_dir, runtime_dir, sessions_path
from .sessions import SessionStore

LABEL = "com.hermes.aipass"
DEFAULT_URL = "http://127.0.0.1:8792"


def register_cli(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="aipass_command")

    setup = sub.add_parser("setup", help="Configure the provider and bridge daemon")
    setup.add_argument("--set-default", action="store_true", help="Make AiPASS the default Hermes provider")

    login = sub.add_parser("login", help="Authenticate through a temporary Chrome window")
    login.add_argument("--timeout", type=int, default=300)

    sub.add_parser("logout", help="Delete AiPASS credentials and conversation mappings")
    sub.add_parser("status", help="Show redacted auth and daemon status")
    models = sub.add_parser("models", help="List available conversation models")
    models.add_argument("--refresh", action="store_true")
    sub.add_parser("start", help="Start the local bridge daemon")
    sub.add_parser("stop", help="Stop the local bridge daemon")
    sub.add_parser("restart", help="Restart the local bridge daemon")
    sub.add_parser("doctor", help="Run configuration and upstream compatibility checks")
    parser.set_defaults(func=dispatch)


def _plugin_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _domain() -> str:
    return f"gui/{os.getuid()}"


def _service_target() -> str:
    return f"{_domain()}/{LABEL}"


def _run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=check, capture_output=True, text=True)


def _health() -> dict | None:
    try:
        with urlopen(f"{DEFAULT_URL}/health", timeout=2) as response:
            data = json.loads(response.read().decode())
            return data if isinstance(data, dict) else None
    except (OSError, URLError, ValueError):
        return None


def _write_launch_agent() -> Path:
    if platform.system() != "Darwin":
        raise RuntimeError("hermes-aipass v0.1 daemon setup supports macOS only")
    path = launch_agent_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    root = _plugin_root()
    log = logs_dir() / "aipass-bridge.log"
    error_log = logs_dir() / "aipass-bridge-error.log"
    environment = {
        "HERMES_HOME": str(hermes_home()),
        "PYTHONPATH": str(root),
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
    }
    payload = {
        "Label": LABEL,
        "ProgramArguments": [sys.executable, "-m", "hermes_aipass.bridge"],
        "EnvironmentVariables": environment,
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ThrottleInterval": 5,
        "StandardOutPath": str(log),
        "StandardErrorPath": str(error_log),
        "WorkingDirectory": str(root),
    }
    with path.open("wb") as handle:
        plistlib.dump(payload, handle, sort_keys=True)
    path.chmod(0o600)
    return path


def _bootout() -> None:
    _run(["launchctl", "bootout", _service_target()], check=False)


def _start() -> None:
    path = launch_agent_path()
    if not path.exists():
        path = _write_launch_agent()
    if _health():
        return
    _run(["launchctl", "bootstrap", _domain(), str(path)], check=False)
    result = _run(["launchctl", "kickstart", "-k", _service_target()], check=False)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if _health():
            return
        time.sleep(0.2)
    message = (result.stderr or result.stdout).strip()
    detail = f": {message}" if message else ""
    raise RuntimeError(f"Could not start AiPASS bridge{detail}")


def _configure_provider(set_default: bool) -> None:
    hermes = shutil.which("hermes")
    if not hermes:
        raise RuntimeError("Hermes executable was not found in PATH")
    settings = [
        ("providers.aipass.api", f"{DEFAULT_URL}/v1"),
        ("providers.aipass.transport", "chat_completions"),
        ("providers.aipass.default_model", "gpt-5.6-sol"),
        ("providers.aipass.discover_models", "true"),
        ("providers.aipass.context_length", "128000"),
    ]
    if set_default:
        settings.extend(
            [
                ("model.default", "gpt-5.6-sol"),
                ("model.provider", "aipass"),
                ("model.base_url", f"{DEFAULT_URL}/v1"),
                ("model.context_length", "128000"),
                ("streaming.enabled", "true"),
            ]
        )
    for key, value in settings:
        result = _run([hermes, "config", "set", key, value], check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Could not set Hermes config {key}: {(result.stderr or result.stdout).strip()}")


def _setup(args: argparse.Namespace) -> int:
    runtime_dir()
    _configure_provider(bool(args.set_default))
    path = _write_launch_agent()
    _bootout()
    _start()
    print(f"Configured AiPASS provider and installed {path}")
    if not _auth_present():
        print("Next: hermes aipass login")
    return 0


def _auth_present() -> bool:
    try:
        KeychainStore().load()
        return True
    except AuthError:
        return False


def _login(args: argparse.Namespace) -> int:
    print("Complete AiPASS sign-in in the Chrome window. It will close automatically.")
    result = interactive_login(timeout=args.timeout)
    print(f"AiPASS login succeeded. Session expiry: {result.get('expires_at') or 'server-managed'}")
    if launch_agent_path().exists():
        _bootout()
        _start()
    return 0


def _logout() -> int:
    KeychainStore().delete()
    SessionStore().clear()
    print("Deleted AiPASS credentials and conversation mappings.")
    return 0


def _status() -> int:
    health = _health()
    print(f"Plugin version: {__version__}")
    print(f"Bridge: {'running' if health else 'stopped'}")
    if health:
        print(f"Bridge version: {health.get('version', 'unknown')}")
    try:
        state = KeychainStore().load()
        status = validate_auth(state)
        print("Authentication: valid")
        print(f"Session expiry: {status.get('expires_at') or 'server-managed'}")
    except AuthError as exc:
        print(f"Authentication: invalid ({exc})")
        return 1
    return 0 if health else 1


def _models(args: argparse.Namespace) -> int:
    models = AiPassClient().models(refresh=bool(args.refresh))
    for model in models:
        levels = f" [{'/'.join(model.thinking_levels)}]" if model.thinking_levels else ""
        print(f"{model.id:<44} {model.display_name}{levels}")
    return 0


def _doctor() -> int:
    checks: list[tuple[str, bool, str]] = []
    checks.append(("platform", platform.system() == "Darwin", platform.system()))
    checks.append(("plugin manifest", (_plugin_root() / "plugin.yaml").exists(), str(_plugin_root())))
    checks.append(("LaunchAgent", launch_agent_path().exists(), str(launch_agent_path())))
    health = _health()
    checks.append(("bridge", bool(health), DEFAULT_URL))
    try:
        state = KeychainStore().load()
        validate_auth(state)
        checks.append(("authentication", True, state.expires_at or "server-managed"))
        models = AiPassClient().models(refresh=True)
        checks.append(("model protocol", bool(models), f"{len(models)} conversation models"))
    except (AuthError, ProtocolError) as exc:
        checks.append(("authentication/upstream", False, str(exc)))
    failed = False
    for name, ok, detail in checks:
        print(f"{'✓' if ok else '✗'} {name}: {detail}")
        failed = failed or not ok
    return 1 if failed else 0


def dispatch(args: argparse.Namespace) -> int:
    command = getattr(args, "aipass_command", None)
    try:
        if command == "setup":
            return _setup(args)
        if command == "login":
            return _login(args)
        if command == "logout":
            return _logout()
        if command == "status":
            return _status()
        if command == "models":
            return _models(args)
        if command == "start":
            _start()
            print("AiPASS bridge started.")
            return 0
        if command == "stop":
            _bootout()
            print("AiPASS bridge stopped.")
            return 0
        if command == "restart":
            _bootout()
            _start()
            print("AiPASS bridge restarted.")
            return 0
        if command == "doctor":
            return _doctor()
        print("Usage: hermes aipass <setup|login|logout|status|models|start|stop|restart|doctor>")
        return 2
    except (AuthError, ProtocolError, RuntimeError) as exc:
        print(f"AiPASS error: {exc}", file=sys.stderr)
        return 1
