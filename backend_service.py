import json
import multiprocessing
import os
import uuid
from typing import Any, Dict, Literal

from dotenv import load_dotenv

from fastapi import FastAPI
from pydantic import BaseModel

from graph import FORENSIC_SPECIALIST_AGENT_ORDER

try:
    import redis as redis_lib
except ImportError:  # pragma: no cover
    redis_lib = None

JobStatus = Literal["pending", "running", "done", "error"]


class DetectRequest(BaseModel):
    responses: list[dict]
    rt_data: list[dict] = []
    itemtype: str = "2PL"
    compromised_items: list[int] = []
    model_settings: dict = {}
    psi_data: list[dict] | None = None
    # Which aberrance functions the UI selected on Preparation
    aberrance_functions: list[str] = []


class IrtRequest(BaseModel):
    responses: list[dict]
    rt_data: list[dict] = []
    itemtype: str = "2PL"
    model_settings: dict = {}


# Job registry (in-process). When REDIS_URL is unset, each Gunicorn worker has its own JOBS;
# use PSYMAS_GUNICORN_WORKERS=1 or results will not round-trip for /detect.
#
# When REDIS_URL is set, job payloads are stored in Redis so any worker/replica can read them.
#
# On Linux (Docker/Railway), rpy2/R is not fork-safe. We use the "spawn" start method.
_mp_ctx = multiprocessing.get_context("spawn")
_manager: Any = None
JOBS: Dict[str, Dict[str, Any]] = {}
PROCS: Dict[str, multiprocessing.Process] = {}
_redis_client: Any = None


def _job_key(run_id: str) -> str:
    return f"psymas:detect:{run_id}"


def _job_ttl_sec() -> int:
    try:
        return int(os.getenv("PSYMAS_JOB_TTL_SEC", "604800"))
    except ValueError:
        return 604800


def _redis_url_from_env() -> str | None:
    for key in ("REDIS_URL", "RAILWAY_REDIS_URL", "RAILWAY_REDIS_PRIVATE_URL"):
        v = os.getenv(key, "").strip()
        if v:
            return v
    return None


def _redis_jobs_enabled() -> bool:
    return bool(redis_lib and _redis_url_from_env())


def _get_redis() -> Any:
    global _redis_client
    if _redis_client is None:
        if redis_lib is None:
            raise RuntimeError("redis package is not installed")
        url = _redis_url_from_env()
        if not url:
            raise RuntimeError("REDIS_URL is not set")
        _redis_client = redis_lib.from_url(url, decode_responses=True, socket_connect_timeout=5)
    return _redis_client


def _json_safe_dumps(obj: Any) -> str:
    def _default(o: Any) -> Any:
        if hasattr(o, "item"):
            try:
                return o.item()
            except Exception:
                return str(o)
        if isinstance(o, bytes):
            return o.decode("utf-8", errors="replace")
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    return json.dumps(obj, default=_default)


class MemoryJobSink:
    """Writes detect job fields into a Manager.dict() (same process tree only)."""

    __slots__ = ("_job",)

    def __init__(self, job: Any) -> None:
        self._job = job

    def patch(self, updates: Dict[str, Any]) -> None:
        for k, v in updates.items():
            if k == "node_states" and isinstance(v, dict):
                ns = dict(self._job.get("node_states") or {})
                ns.update(v)
                self._job["node_states"] = ns
            else:
                self._job[k] = v


class RedisJobSink:
    """Persists job state to Redis for multi-worker / multi-replica APIs."""

    __slots__ = ("_r", "_rid")

    def __init__(self, r: Any, run_id: str) -> None:
        self._r = r
        self._rid = run_id

    def _load(self) -> Dict[str, Any]:
        raw = self._r.get(_job_key(self._rid))
        if not raw:
            return {}
        return json.loads(raw)

    def patch(self, updates: Dict[str, Any]) -> None:
        job = self._load()
        if not job:
            return
        for k, v in updates.items():
            if k == "node_states" and isinstance(v, dict):
                ns = dict(job.get("node_states") or {})
                ns.update(v)
                job["node_states"] = ns
            else:
                job[k] = v
        self._r.set(_job_key(self._rid), _json_safe_dumps(job), ex=_job_ttl_sec())


def _update_node_states(sink: Any, updates: dict) -> None:
    sink.patch({"node_states": updates})


def _set_node(sink: Any, name: str, state: str) -> None:
    _update_node_states(sink, {name: state})


app = FastAPI(title="PsyMAS LangGraph Backend")


def _run_detect_job(job_id: str, req: DetectRequest, sink: Any) -> None:
    """Worker process: run forensic workflow and persist status via *sink*."""
    try:
        sink.patch({"status": "running", "progress": 5})
        _initial_nodes = [
            "router",
            *FORENSIC_SPECIALIST_AGENT_ORDER,
            "synthesizer",
            "reporter",
        ]
        _update_node_states(sink, {k: "pending" for k in _initial_nodes})

        use_stub = os.getenv("PSYMAS_STUB_LANGGRAPH", "").lower() in ("1", "true", "yes")
        if use_stub:
            psi_data = req.psi_data or []
            sink.patch(
                {
                    "irt_status": "skipped" if psi_data else "stub",
                    "progress": 35,
                }
            )
            result: Dict[str, Any] = {
                "responses": req.responses,
                "rt_data": req.rt_data,
                "theta": 0.0,
                "latency_flags": [],
                "next_step": "done",
                "model_settings": req.model_settings,
                "is_verified": True,
                "psi_data": psi_data,
                "compromised_items": req.compromised_items,
                "aberrance_functions": req.aberrance_functions or [],
                "flags": {
                    "stub": {
                        "note": "Stubbed LangGraph run (PSYMAS_STUB_LANGGRAPH=1); no R/aberrance executed."
                    }
                },
                "final_report": (
                    "This is a stubbed detect run for UI development. "
                    "LangGraph agents and R-based aberrance analysis were not executed."
                ),
                "reporter_brief": "Stub run — no specialist outputs to summarize.",
            }
            _update_node_states(sink, {k: "done" for k in _initial_nodes})
            sink.patch({"status": "done", "progress": 100, "result": result})
            return

        try:
            from rpy2.robjects import numpy2ri

            numpy2ri.activate()
        except Exception:
            pass

        psi_data = req.psi_data or []
        if not psi_data:
            sink.patch(
                {
                    "irt_status": "missing",
                    "status": "error",
                    "error": "Missing psi_data. Generate item parameters (ψ) on Preparation page and retry.",
                    "progress": 0,
                }
            )
            return

        sink.patch({"psi_data": psi_data, "irt_status": "provided", "progress": 35})

        forensic_state = {
            "responses": req.responses,
            "rt_data": req.rt_data,
            "theta": 0.0,
            "latency_flags": [],
            "next_step": "start",
            "model_settings": req.model_settings,
            "is_verified": True,
            "psi_data": psi_data,
            "compromised_items": req.compromised_items,
            "aberrance_functions": req.aberrance_functions or [],
            "flags": {},
            "final_report": "",
        }

        result: Dict[str, Any] = dict(forensic_state)
        result.setdefault("flags", {})

        def _merge_fragment(fragment: dict) -> None:
            if not isinstance(fragment, dict):
                return
            for key, value in fragment.items():
                if key == "flags" and isinstance(value, dict):
                    result["flags"] = {**(result.get("flags") or {}), **value}
                else:
                    result[key] = value

        _set_node(sink, "router", "done")

        from graph import (
            ac_agent,
            as_agent,
            cp_agent,
            forensic_reporter,
            manager_synthesizer,
            nm_agent,
            pk_agent,
            pm_agent,
            rg_agent,
            tt_agent,
        )

        _agent_fn = {
            "nm_agent": nm_agent,
            "pm_agent": pm_agent,
            "ac_agent": ac_agent,
            "as_agent": as_agent,
            "pk_agent": pk_agent,
            "rg_agent": rg_agent,
            "cp_agent": cp_agent,
            "tt_agent": tt_agent,
        }
        agent_steps = [(n, _agent_fn[n]) for n in FORENSIC_SPECIALIST_AGENT_ORDER]

        for idx, (name, fn) in enumerate(agent_steps, start=1):
            _set_node(sink, name, "running")
            sink.patch({"progress": 35 + int(55 * (idx - 1) / max(1, len(agent_steps) + 1))})
            try:
                frag = fn(dict(result))
            except Exception as e:
                _set_node(sink, name, "error")
                sink.patch({"status": "error", "error": f"{name} failed: {e}"})
                return
            _merge_fragment(frag or {})
            _set_node(sink, name, "done")
            sink.patch({"progress": 35 + int(55 * idx / max(1, len(agent_steps) + 1))})

        _set_node(sink, "synthesizer", "running")
        try:
            synth = manager_synthesizer(dict(result))
        except Exception as e:
            _set_node(sink, "synthesizer", "error")
            sink.patch({"status": "error", "error": f"synthesizer failed: {e}"})
            return
        _merge_fragment(synth or {})
        _set_node(sink, "synthesizer", "done")

        _set_node(sink, "reporter", "running")
        try:
            rep = forensic_reporter(dict(result))
        except Exception as e:
            _set_node(sink, "reporter", "error")
            sink.patch({"status": "error", "error": f"reporter failed: {e}"})
            return
        _merge_fragment(rep or {})
        _set_node(sink, "reporter", "done")

        sink.patch({"status": "done", "progress": 100, "result": result})
    except Exception as e:
        try:
            sink.patch({"status": "error", "error": str(e), "progress": 0})
        except Exception:
            pass


def _child_detect_entry(
    *,
    run_id: str,
    req_dict: dict,
    jobs_proxy: Any | None = None,
    use_redis: bool = False,
) -> None:
    """Run in subprocess so R/rpy2 runs in this process's main thread (avoids Windows errors)."""
    import backend_service

    req = DetectRequest(**req_dict)
    if use_redis:
        r = backend_service._get_redis()
        sink = RedisJobSink(r, run_id)
        backend_service._run_detect_job(run_id, req, sink)
    else:
        backend_service.JOBS = jobs_proxy
        job = jobs_proxy[run_id]
        sink = MemoryJobSink(job)
        backend_service._run_detect_job(run_id, req, sink)


@app.on_event("startup")
def _startup_manager() -> None:
    global _manager, JOBS, _redis_client
    load_dotenv()
    _redis_client = None
    if _redis_jobs_enabled():
        _get_redis().ping()
        JOBS = {}
        _manager = None
    else:
        _manager = _mp_ctx.Manager()
        JOBS = _manager.dict()


def _get_job_record(run_id: str) -> Dict[str, Any] | None:
    if _redis_jobs_enabled():
        raw = _get_redis().get(_job_key(run_id))
        if not raw:
            return None
        return json.loads(raw)
    j = JOBS.get(run_id)
    if j is None:
        return None
    return dict(j)


@app.post("/detect")
def start_detect(req: DetectRequest) -> dict:
    load_dotenv()
    run_id = str(uuid.uuid4())
    if _redis_jobs_enabled():
        initial = {
            "status": "pending",
            "progress": 0,
            "irt_status": "pending",
            "node_states": {},
            "error": None,
            "result": None,
        }
        _get_redis().set(_job_key(run_id), _json_safe_dumps(initial), ex=_job_ttl_sec())
        p = _mp_ctx.Process(
            target=_child_detect_entry,
            kwargs={"run_id": run_id, "req_dict": req.model_dump(), "use_redis": True},
            daemon=True,
        )
        p.start()
        PROCS[run_id] = p
        return {"run_id": run_id}

    assert _manager is not None
    job_ref = _manager.dict()
    job_ref.update(
        {
            "status": "pending",
            "progress": 0,
            "irt_status": "pending",
            "node_states": {},
            "error": None,
        }
    )
    JOBS[run_id] = job_ref
    p = _mp_ctx.Process(
        target=_child_detect_entry,
        kwargs={"run_id": run_id, "req_dict": req.model_dump(), "jobs_proxy": JOBS, "use_redis": False},
        daemon=True,
    )
    p.start()
    PROCS[run_id] = p
    return {"run_id": run_id}


@app.post("/irt")
def run_irt(req: IrtRequest) -> dict:
    """Run IRT in the backend (Docker/Linux) and return ψ item parameters."""
    load_dotenv()
    try:
        from rpy2.robjects import numpy2ri, pandas2ri

        numpy2ri.activate()
        pandas2ri.activate()
    except Exception:
        pass
    from graph import irt_agent

    irt_state = {
        "responses": req.responses,
        "rt_data": req.rt_data,
        "theta": 0.0,
        "latency_flags": [],
        "next_step": "start",
        "model_settings": {**(req.model_settings or {}), "itemtype": req.itemtype},
        "is_verified": True,
    }
    result = irt_agent(irt_state) or {}
    item_params = result.get("item_params") or []
    if not item_params:
        return {
            "status": "error",
            "error": result.get("icc_error") or "IRT returned no item parameters.",
            "result": result,
        }
    return {"status": "done", "result": result}


@app.get("/detect/{run_id}/status")
def get_status(run_id: str) -> dict:
    proc = PROCS.get(run_id)
    job = JOBS.get(run_id) if not _redis_jobs_enabled() else None

    if proc is not None and not proc.is_alive():
        if _redis_jobs_enabled():
            rec = _get_job_record(run_id)
            if rec and rec.get("status") in (None, "pending", "running"):
                RedisJobSink(_get_redis(), run_id).patch(
                    {
                        "status": "error",
                        "error": "Detection worker crashed or exited unexpectedly.",
                    }
                )
        elif job is not None and job.get("status") in (None, "pending", "running"):
            job["status"] = "error"
            job["error"] = "Detection worker crashed or exited unexpectedly."
        PROCS.pop(run_id, None)

    job_dict = _get_job_record(run_id)
    if not job_dict:
        return {"status": "unknown", "progress": 0, "node_states": {}, "error": "run_id not found"}
    return {
        "status": job_dict.get("status", "pending"),
        "progress": job_dict.get("progress", 0),
        "irt_status": job_dict.get("irt_status", "pending"),
        "node_states": dict(job_dict.get("node_states") or {}),
        "error": job_dict.get("error"),
    }


@app.get("/detect/{run_id}/result")
def get_result(run_id: str) -> dict:
    job_dict = _get_job_record(run_id)
    if not job_dict or job_dict.get("status") != "done":
        return {"status": job_dict.get("status") if job_dict else "unknown", "result": None}
    return {"status": "done", "result": job_dict.get("result")}


@app.get("/health")
def health() -> dict:
    load_dotenv()
    return {"status": "ok", "shared_detect_jobs": _redis_jobs_enabled()}
