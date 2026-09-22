"""HTTP client and status helpers for the PsyMAS backend service."""

import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
import streamlit as st
from dotenv import load_dotenv


load_dotenv()


def normalize_backend_url(raw: str | None) -> str:
    """Normalize local, Railway internal, and public backend URLs."""
    if not raw or not str(raw).strip():
        # The local compose stack publishes the container's port 8000 as host port 9000.
        return "http://localhost:9000"
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


def _windows_creation_flags() -> int:
    if os.name != "nt":
        return 0
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)


def _local_docker_executable() -> str | None:
    candidates = [
        shutil.which("docker"),
        os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "Docker", "Docker", "resources", "bin", "docker.exe"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Docker", "Docker", "resources", "bin", "docker.exe"),
    ]
    return next((candidate for candidate in candidates if candidate and Path(candidate).exists()), None)


def _local_docker_desktop_executable() -> str | None:
    candidates = [
        os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "Docker", "Docker", "Docker Desktop.exe"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Docker", "Docker", "Docker Desktop.exe"),
    ]
    return next((candidate for candidate in candidates if candidate and Path(candidate).exists()), None)


def _start_local_docker_and_backend() -> str | None:
    """Start the local Windows stack once, without blocking Streamlit startup."""
    if os.name != "nt" or os.getenv("PSYMAS_AUTO_START_DOCKER", "1").strip().lower() in {"0", "false", "no"}:
        return None
    parsed = urlparse(BACKEND_URL)
    if parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        return None
    if st.session_state.get("_psymas_local_stack_start_attempted"):
        return st.session_state.get("_psymas_local_stack_start_note")
    st.session_state["_psymas_local_stack_start_attempted"] = True

    docker = _local_docker_executable()
    desktop = _local_docker_desktop_executable()
    if not docker:
        return "Docker CLI was not found; install Docker Desktop or set PSYMAS_AUTO_START_DOCKER=0."

    def run_quiet(args: list[str], timeout: float = 3) -> bool:
        try:
            result = subprocess.run(
                args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                check=False,
            )
            return result.returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False

    if run_quiet([docker, "info"]):
        note = "Docker Desktop is ready; starting the PsyMAS backend service."
    else:
        if not desktop:
            return "Docker Desktop was not found; start it manually or install Docker Desktop."
        try:
            subprocess.Popen(
                [desktop],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                creationflags=_windows_creation_flags(),
            )
        except OSError:
            return "Docker Desktop could not be started automatically; start it manually."
        note = "Docker Desktop is starting automatically; the backend will connect when it is ready."

    repo_root = Path(__file__).resolve().parents[1]
    compose_file = Path(os.getenv("PSYMAS_COMPOSE_FILE", str(repo_root / "docker-compose.yml")))
    env_file = repo_root / ".env"

    def start_backend_when_ready() -> None:
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if run_quiet([docker, "info"], timeout=5):
                command = [docker, "compose", "-p", "psymas-desktop"]
                if env_file.exists():
                    command.extend(["--env-file", str(env_file)])
                command.extend(["-f", str(compose_file), "up", "-d", "backend"])
                run_quiet(command, timeout=120)
                return
            time.sleep(3)

    if compose_file.exists():
        threading.Thread(target=start_backend_when_ready, daemon=True, name="psymas-docker-start").start()
    st.session_state["_psymas_local_stack_start_note"] = note
    return note


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
        startup_note = _start_local_docker_and_backend()
        if startup_note:
            return False, "starting Docker Desktop", startup_note
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
