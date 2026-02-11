import multiprocessing
import os
import uuid
from typing import Any, Dict, Literal

from dotenv import load_dotenv

from fastapi import FastAPI
from pydantic import BaseModel


JobStatus = Literal["pending", "running", "done", "error"]


class DetectRequest(BaseModel):
    responses: list[dict]
    rt_data: list[dict] = []
    itemtype: str = "2PL"
    compromised_items: list[int] = []
    model_settings: dict = {}
    psi_data: list[dict] | None = None


# Job registry. When using subprocess (Windows R fix), JOBS is a Manager().dict()
# and each job value is a Manager().dict() so the worker can update status/result.
_manager: Any = None
JOBS: Dict[str, Dict[str, Any]] = {}
# Track worker processes so we can detect crashes (e.g., R/rpy2 segfaults on Windows).
PROCS: Dict[str, multiprocessing.Process] = {}


app = FastAPI(title="PsyMAS LangGraph Backend")


def _run_detect_job(job_id: str, req: DetectRequest) -> None:
    """Worker process: run IRT + forensic_workflow and update JOBS[job_id].

    When the env var PSYMAS_STUB_LANGGRAPH=1, this function runs in a
    lightweight stub mode that *skips* R / LangGraph entirely and simulates
    agent progress. This is mainly for local Windows development where rpy2/R
    are unstable; production (Railway) should run with the real workflow.
    """
    try:
        job = JOBS[job_id]
        job["status"] = "running"
        job["progress"] = 5
        job["node_states"] = {
            k: "pending"
            for k in [
                "router",
                "pm_agent",
                "nm_agent",
                "ac_agent",
                "as_agent",
                "rg_agent",
                "cp_agent",
                "tt_agent",
                "pk_agent",
                "synthesizer",
            ]
        }

        # Optional stub mode for local Windows development (skips R + LangGraph).
        use_stub = os.getenv("PSYMAS_STUB_LANGGRAPH", "").lower() in ("1", "true", "yes")
        if use_stub:
            psi_data = req.psi_data or []
            job["psi_data"] = psi_data
            job["irt_status"] = "skipped" if psi_data else "stub"
            job["progress"] = 35

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
                "flags": {
                    "stub": {
                        "note": "Stubbed LangGraph run (PSYMAS_STUB_LANGGRAPH=1); no R/aberrance executed."
                    }
                },
                "final_report": (
                    "This is a stubbed detect run for UI development. "
                    "LangGraph agents and R-based aberrance analysis were not executed."
                ),
            }

            # Mark all nodes as done so the UI graph turns green.
            node_states = job["node_states"]
            for node_name in node_states.keys():
                node_states[node_name] = "done"

            job["status"] = "done"
            job["progress"] = 100
            job["result"] = result
            return

        # ---- Real mode: IRT (ψ generation) + aberrance workflow ----
        # Import here so stub mode can avoid importing heavy R / LangGraph stack.
        from graph import forensic_workflow, irt_agent

        # Step 1: IRT (ψ generation)
        # If ψ was already generated in Preparation, reuse it to keep parameters identical.
        psi_data = req.psi_data or []
        if psi_data:
            job["psi_data"] = psi_data
            job["irt_status"] = "skipped"
            job["progress"] = 35
        else:
            irt_state = {
                "responses": req.responses,
                "rt_data": req.rt_data,
                "theta": 0.0,
                "latency_flags": [],
                "next_step": "start",
                "model_settings": {**req.model_settings, "itemtype": req.itemtype},
                "is_verified": True,
            }
            irt_result = irt_agent(irt_state)
            psi_data = irt_result.get("item_params") or []
            job["psi_data"] = psi_data
            job["irt_status"] = "done" if psi_data else "error"
            job["progress"] = 35

        # Step 2: Aberrance detection workflow
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
            "flags": {},
            "final_report": "",
        }

        result: Dict[str, Any] = dict(forensic_state)
        result.setdefault("flags", {})
        node_states = job["node_states"]
        done: set[str] = set()
        total_nodes = len(node_states)

        def _merge_fragment(fragment: dict) -> None:
            if not isinstance(fragment, dict):
                return
            for key, value in fragment.items():
                if key == "flags" and isinstance(value, dict):
                    result["flags"] = {**(result.get("flags") or {}), **value}
                else:
                    result[key] = value

        def _mark_done(node_name: str) -> None:
            if node_name in done:
                return
            done.add(node_name)
            if node_name in node_states:
                node_states[node_name] = "done"
            job["progress"] = 35 + int(60 * len(done) / max(1, total_nodes))

        try:
            # Prefer streaming so we can update node states as they finish.
            stream_ok = False
            try:
                for chunk in forensic_workflow.stream(forensic_state, stream_mode="updates"):
                    stream_ok = True
                    if not isinstance(chunk, dict):
                        continue
                    if any(k in chunk for k in ("flags", "final_report")):
                        _merge_fragment(chunk)
                        continue
                    for node_name, fragment in chunk.items():
                        if isinstance(fragment, dict):
                            _merge_fragment(fragment)
                        _mark_done(node_name)
            except TypeError:
                for chunk in forensic_workflow.stream(forensic_state):
                    stream_ok = True
                    if not isinstance(chunk, dict):
                        continue
                    if any(k in chunk for k in ("flags", "final_report")):
                        _merge_fragment(chunk)
                        continue
                    for node_name, fragment in chunk.items():
                        if isinstance(fragment, dict):
                            _merge_fragment(fragment)
                        _mark_done(node_name)

            if not stream_ok:
                # Fallback: no streaming, single invoke
                result = forensic_workflow.invoke(forensic_state)
                for node_name in node_states:
                    node_states[node_name] = "done"

            # Ensure any missing nodes are done
            for node_name in node_states:
                if node_name not in done:
                    node_states[node_name] = "done"

            job["status"] = "done"
            job["progress"] = 100
            job["result"] = result
        except Exception as e:
            job["status"] = "error"
            job["error"] = str(e)
    except Exception as e:
        job = JOBS.get(job_id)
        if job is not None:
            job["status"] = "error"
            job["error"] = str(e)
            job["progress"] = 0
        else:
            JOBS[job_id] = {"status": "error", "error": str(e), "progress": 0}


def _child_detect_entry(run_id: str, req_dict: dict, jobs_proxy: Any) -> None:
    """Run in subprocess so R/rpy2 runs in this process's main thread (avoids Windows errors)."""
    import backend_service

    backend_service.JOBS = jobs_proxy
    req = DetectRequest(**req_dict)
    backend_service._run_detect_job(run_id, req)


@app.on_event("startup")
def _startup_manager() -> None:
    global _manager, JOBS
    _manager = multiprocessing.Manager()
    JOBS = _manager.dict()


@app.post("/detect")
def start_detect(req: DetectRequest) -> dict:
    load_dotenv()
    run_id = str(uuid.uuid4())
    # Per-job dict must be a manager.dict() so the worker subprocess's updates are visible.
    job_ref = _manager.dict()
    job_ref.update({
        "status": "pending",
        "progress": 0,
        "irt_status": "pending",
        "node_states": {},
        "error": None,
    })
    JOBS[run_id] = job_ref
    p = multiprocessing.Process(
        target=_child_detect_entry,
        args=(run_id, req.model_dump(), JOBS),
        daemon=True,
    )
    p.start()
    # Remember the worker so /status can see if it crashed.
    PROCS[run_id] = p
    return {"run_id": run_id}


@app.get("/detect/{run_id}/status")
def get_status(run_id: str) -> dict:
    job = JOBS.get(run_id)
    # If the worker process died unexpectedly and the job never updated its status,
    # surface this as an error so the UI doesn't stay "running" forever.
    proc = PROCS.get(run_id)
    if proc is not None and not proc.is_alive():
        if job and job.get("status") in (None, "pending", "running"):
            job["status"] = "error"
            job["error"] = "Detection worker crashed or exited unexpectedly."
        # Optionally remove from registry to avoid repeated checks.
        PROCS.pop(run_id, None)

    if not job:
        return {"status": "unknown", "progress": 0, "node_states": {}, "error": "run_id not found"}
    return {
        "status": job.get("status", "pending"),
        "progress": job.get("progress", 0),
        "irt_status": job.get("irt_status", "pending"),
        "node_states": job.get("node_states", {}),
        "error": job.get("error"),
    }


@app.get("/detect/{run_id}/result")
def get_result(run_id: str) -> dict:
    job = JOBS.get(run_id)
    if not job or job.get("status") != "done":
        return {"status": job.get("status") if job else "unknown", "result": None}
    return {"status": "done", "result": job.get("result")}


@app.get("/health")
def health() -> dict:
    """Lightweight backend/graph health check for the UI."""
    # We deliberately avoid touching R or running a workflow here; this just confirms
    # that the FastAPI backend is up and importable.
    return {"status": "ok"}

