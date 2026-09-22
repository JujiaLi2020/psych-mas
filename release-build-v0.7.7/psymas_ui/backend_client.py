"""HTTP client and status helpers for the PsyMAS backend service."""

import os

import requests
import streamlit as st
from dotenv import load_dotenv


load_dotenv()


def normalize_backend_url(raw: str | None) -> str:
    """Normalize local, Railway internal, and public backend URLs."""
    if not raw or not str(raw).strip():
        return "http://localhost:8000"
    url = str(raw).strip().rstrip("/")
    if "://" in url:
        lowered = url.lower()
        if lowered.startswith("http://") and (
            ".up.railway.app" in lowered or lowered.endswith("railway.app")
        ):
            url = "https://" + url.split("://", 1)[1]
        return url
    host = url.split("/")[0]
    if "localhost" in host or host.startswith("127.0.0.1") or ".railway.internal" in host:
        return "http://" + url
    return "https://" + url


BACKEND_URL = normalize_backend_url(os.getenv("PSYMAS_BACKEND_URL"))

_HTTP = requests.Session()
_HTTP.trust_env = False


def backend_get(path: str, **kwargs):
    return _HTTP.get(f"{BACKEND_URL}{path}", **kwargs)


def backend_post(path: str, **kwargs):
    return _HTTP.post(f"{BACKEND_URL}{path}", **kwargs)


def _one_line_ellipsis(text: str | None, max_len: int = 52) -> str:
    if not text:
        return ""
    compact = " ".join(str(text).split())
    if len(compact) <= max_len:
        return compact
    return compact[: max(1, max_len - 1)] + "…"


def backend_status_summary() -> tuple[bool, str, str]:
    """Return backend reachability and current-session Detect status."""
    try:
        response = backend_get("/health", timeout=3)
        response.raise_for_status()
        if response.json().get("status") != "ok":
            return False, "backend health error", 'Backend /health did not return {"status":"ok"}.'
    except Exception:
        return (
            False,
            "unreachable",
            f"Cannot reach {BACKEND_URL}/health. Set PSYMAS_BACKEND_URL and ensure the API process is running.",
        )

    detect_status = st.session_state.get("detect_job_status", "pending")
    if detect_status == "running":
        return True, "running", "Detect job is running; status updates on the Preparation page."
    if detect_status == "done":
        return True, "idle (last run ok)", "Last Detect finished successfully in this browser session."
    if detect_status == "error":
        error = (st.session_state.get("detect_job_error") or "").strip() or "Unknown error"
        return False, f"last run error — {_one_line_ellipsis(error, 44)}", f"Last Detect failed: {error}"
    return True, "idle", "Backend /health OK. No Detect failure recorded in this session."


def detect_run_id_replica_hint(detect_error: str | None) -> bool:
    """Identify errors caused by requests reaching different API replicas."""
    if not detect_error:
        return False
    lowered = detect_error.lower()
    return any(
        phrase in lowered
        for phrase in (
            "each container has its own disk",
            "multiple railway replicas",
            "redis_url for shared",
        )
    )
