from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping


def git_commit(path: str | os.PathLike[str] | None = None) -> str | None:

    cwd = str(Path(path or os.getcwd()).resolve())
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return value.decode().strip() or None


def config_sha256(config_path: str | os.PathLike[str] | None) -> str | None:
    if not config_path:
        return None
    path = Path(config_path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_result_json(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> None:

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, output)
