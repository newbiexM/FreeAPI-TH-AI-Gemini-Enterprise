"""Runtime paths and atomic private-state helpers."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes").expanduser()


def runtime_dir() -> Path:
    path = hermes_home() / "aipass"
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        path.chmod(0o700)
    except OSError:
        pass
    return path


def logs_dir() -> Path:
    path = hermes_home() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def sessions_path() -> Path:
    return runtime_dir() / "sessions.json"


def metadata_path() -> Path:
    return runtime_dir() / "auth-metadata.json"


def auth_json_path() -> Path:
    return runtime_dir() / "auth.json"


def chrome_profile_dir() -> Path:
    return runtime_dir() / "chrome-profile"


def launch_agent_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / "com.hermes.aipass.plist"


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return default


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp, 0o600)
        os.replace(temp, path)
    finally:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass

