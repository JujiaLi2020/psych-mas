"""Command-line launcher for the PsyMAS workbench."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path


VERSION = "0.7.6"
COMPOSE_PROJECT = "psymas-cli"


def _runtime_root() -> Path:
    """Return the source checkout or installation prefix containing PsyMAS assets."""
    override = os.environ.get("PSYMAS_HOME")
    candidates = [
        Path(override).expanduser() if override else None,
        Path(__file__).resolve().parent,
        Path(sys.prefix),
    ]
    for candidate in candidates:
        if candidate and (candidate / "config" / "rulebook_index.csv").is_file():
            return candidate.resolve()
    raise SystemExit(
        "PsyMAS resource files were not found. Reinstall psych-mas, or set "
        "PSYMAS_HOME to the directory containing config/ and data/."
    )


def _prepare_runtime() -> Path:
    root = _runtime_root()
    os.environ.setdefault("PSYMAS_HOME", str(root))
    os.chdir(root)
    return root


def _run_ui(args: argparse.Namespace) -> int:
    _prepare_runtime()
    if args.backend_url:
        os.environ["PSYMAS_BACKEND_URL"] = args.backend_url

    from streamlit.web import cli as streamlit_cli

    ui_file = Path(__file__).resolve().with_name("ui.py")
    sys.argv = [
        "streamlit",
        "run",
        str(ui_file),
        "--server.port",
        str(args.port),
        "--server.address",
        args.host,
        "--server.headless",
        "true" if args.no_browser else "false",
    ]
    return int(streamlit_cli.main() or 0)


def _run_backend(args: argparse.Namespace) -> int:
    _prepare_runtime()
    import uvicorn

    uvicorn.run(
        "backend_service:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


def _doctor(_: argparse.Namespace) -> int:
    problems: list[str] = []
    try:
        root = _runtime_root()
        print(f"[ok] PsyMAS {VERSION}")
        print(f"[ok] Python {sys.version.split()[0]}")
        print(f"[ok] Resources: {root}")
    except SystemExit as exc:
        print(f"[error] {exc}")
        problems.append("resources")

    if shutil.which("R") or shutil.which("Rscript"):
        print("[ok] R runtime found")
    else:
        print("[warn] R was not found; use Docker for the analysis backend")

    if shutil.which("docker"):
        print("[ok] Docker found")
    else:
        print("[warn] Docker was not found")
    return 1 if problems else 0


def _user_runtime_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "PsyMAS"


def _compose_file() -> Path:
    compose = _runtime_root() / "docker-compose.release.yml"
    if not compose.is_file():
        raise SystemExit("The packaged Docker service definition is missing. Reinstall psych-mas.")
    return compose


def _ensure_runtime_env() -> tuple[Path, Path]:
    runtime = _user_runtime_dir()
    data_dir = runtime / "data" / "output"
    runtime.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    env_file = runtime / ".env"
    values: dict[str, str] = {}
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line and not line.lstrip().startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip()
    values["PSYMAS_IMAGE_TAG"] = VERSION
    values["PSYMAS_DATA_DIR"] = data_dir.resolve().as_posix()
    values["PSYMAS_UI_PORT"] = os.environ.get("PSYMAS_UI_PORT", values.get("PSYMAS_UI_PORT", "8501"))
    values.setdefault("OPENROUTER_API_KEY", "")
    env_file.write_text("".join(f"{key}={value}\n" for key, value in values.items()), encoding="utf-8")
    return env_file, data_dir


def _docker_compose(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    if not shutil.which("docker"):
        raise SystemExit("Docker is required for the unified PsyMAS stack. Install and start Docker Desktop first.")
    env_file, _ = _ensure_runtime_env()
    command = [
        "docker", "compose", "-p", COMPOSE_PROJECT,
        "--env-file", str(env_file), "-f", str(_compose_file()), *args,
    ]
    try:
        return subprocess.run(command, check=check)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Docker Compose failed with exit code {exc.returncode}.") from exc


def _wait_for_ui(url: str, timeout: int) -> bool:
    deadline = time.monotonic() + timeout
    health_url = url.rstrip("/") + "/_stcore/health"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=4) as response:
                if response.status == 200:
                    return True
        except Exception:
            time.sleep(2)
    return False


def _start_stack(args: argparse.Namespace) -> int:
    try:
        subprocess.run(["docker", "info"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise SystemExit("Docker is not ready. Start Docker Desktop and run `psymas start` again.") from exc
    os.environ["PSYMAS_UI_PORT"] = str(args.port)
    if not args.no_pull:
        print(f"Downloading PsyMAS {VERSION} container image…")
        _docker_compose("pull")
    print("Starting PsyMAS UI and analysis backend…")
    _docker_compose("up", "-d")
    url = f"http://{args.host}:{args.port}"
    if not _wait_for_ui(url, args.timeout):
        print("PsyMAS is still starting. Run `psymas logs` to inspect the services.", file=sys.stderr)
        return 1
    print(f"PsyMAS is ready at {url}")
    if not args.no_browser:
        webbrowser.open(url)
    return 0


def _stop_stack(_: argparse.Namespace) -> int:
    _docker_compose("down")
    print("PsyMAS stopped. Local assessment and review data were preserved.")
    return 0


def _restart_stack(args: argparse.Namespace) -> int:
    _docker_compose("restart")
    url = f"http://{args.host}:{args.port}"
    if not _wait_for_ui(url, args.timeout):
        print("PsyMAS is restarting. Run `psymas logs` to follow progress.", file=sys.stderr)
        return 1
    print(f"PsyMAS is ready at {url}")
    if not args.no_browser:
        webbrowser.open(url)
    return 0


def _stack_status(_: argparse.Namespace) -> int:
    result = _docker_compose("ps", check=False)
    return int(result.returncode)


def _stack_logs(args: argparse.Namespace) -> int:
    command = ["logs", "--tail", str(args.tail)]
    if args.follow:
        command.append("--follow")
    result = _docker_compose(*command, check=False)
    return int(result.returncode)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="psymas",
        description="Launch and diagnose the PsyMAS psychometric forensics workbench.",
    )
    parser.add_argument("--version", action="version", version=f"PsyMAS {VERSION}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ui_parser = subparsers.add_parser("ui", help="start the Streamlit user interface")
    ui_parser.add_argument("--host", default="127.0.0.1")
    ui_parser.add_argument("--port", type=int, default=8501)
    ui_parser.add_argument("--backend-url", help="analysis backend URL, e.g. http://localhost:9000")
    ui_parser.add_argument("--no-browser", action="store_true", help="do not open a browser")
    ui_parser.set_defaults(handler=_run_ui)

    backend_parser = subparsers.add_parser("backend", help="start the FastAPI analysis backend")
    backend_parser.add_argument("--host", default="127.0.0.1")
    backend_parser.add_argument("--port", type=int, default=9000)
    backend_parser.add_argument("--reload", action="store_true", help="reload after source changes")
    backend_parser.set_defaults(handler=_run_backend)

    doctor_parser = subparsers.add_parser("doctor", help="check the local runtime")
    doctor_parser.set_defaults(handler=_doctor)

    start_parser = subparsers.add_parser("start", help="start the complete UI + backend Docker stack")
    start_parser.add_argument("--host", default="localhost")
    start_parser.add_argument("--port", type=int, default=8501)
    start_parser.add_argument("--timeout", type=int, default=180)
    start_parser.add_argument("--no-browser", action="store_true")
    start_parser.add_argument("--no-pull", action="store_true", help="use the locally cached image")
    start_parser.set_defaults(handler=_start_stack)

    stop_parser = subparsers.add_parser("stop", help="stop the complete Docker stack")
    stop_parser.set_defaults(handler=_stop_stack)

    restart_parser = subparsers.add_parser("restart", help="restart the complete Docker stack")
    restart_parser.add_argument("--host", default="localhost")
    restart_parser.add_argument("--port", type=int, default=8501)
    restart_parser.add_argument("--timeout", type=int, default=180)
    restart_parser.add_argument("--no-browser", action="store_true")
    restart_parser.set_defaults(handler=_restart_stack)

    status_parser = subparsers.add_parser("status", help="show Docker stack status")
    status_parser.set_defaults(handler=_stack_status)

    logs_parser = subparsers.add_parser("logs", help="show Docker stack logs")
    logs_parser.add_argument("--tail", type=int, default=200)
    logs_parser.add_argument("--follow", "-f", action="store_true")
    logs_parser.set_defaults(handler=_stack_logs)
    return parser


def main() -> int:
    args = _parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
