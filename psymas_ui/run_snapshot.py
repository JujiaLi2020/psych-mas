"""Zip snapshot of a completed PsyMAS detect run (integrated DB + session inputs)."""

from __future__ import annotations

import io
import json
import math
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SNAPSHOT_SCHEMA = "1.0"
SQLITE_ENTRY = "psymas_run.sqlite"
MANIFEST_ENTRY = "manifest.json"
INPUTS_ENTRY = "session_inputs.json"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    return value


def _json_text(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        default=str,
        allow_nan=False,
        indent=indent,
    )


def pack_snapshot(*, store_path: Path, manifest: dict[str, Any], session_inputs: dict[str, Any]) -> bytes:
    store_path = Path(store_path)
    if not store_path.exists():
        raise FileNotFoundError(f"Run store not found: {store_path}")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            MANIFEST_ENTRY,
            _json_text(manifest, indent=2),
        )
        zf.writestr(
            INPUTS_ENTRY,
            _json_text(session_inputs),
        )
        zf.write(store_path, SQLITE_ENTRY)
    return buf.getvalue()


def unpack_snapshot(data: bytes, *, store_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = set(zf.namelist())
        if MANIFEST_ENTRY not in names or SQLITE_ENTRY not in names:
            raise ValueError("Invalid snapshot: missing manifest.json or psymas_run.sqlite.")
        manifest = json.loads(zf.read(MANIFEST_ENTRY).decode("utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("Invalid snapshot: manifest must be a JSON object.")
        session_inputs: dict[str, Any] = {}
        if INPUTS_ENTRY in names:
            raw_inputs = json.loads(zf.read(INPUTS_ENTRY).decode("utf-8"))
            if isinstance(raw_inputs, dict):
                session_inputs = raw_inputs
        store_path = Path(store_path)
        store_path.parent.mkdir(parents=True, exist_ok=True)
        store_path.write_bytes(zf.read(SQLITE_ENTRY))
    return manifest, session_inputs
