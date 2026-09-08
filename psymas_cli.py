"""Command-line launcher for the PsyMAS workbench."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


VERSION = "0.7.5"


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
    return parser


def main() -> int:
    args = _parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
