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


class IrtRequest(BaseModel):
    responses: list[dict]
    rt_data: list[dict] = []
    itemtype: str = "2PL"
    model_settings: dict = {}


# Job registry. When using subprocess (Windows R fix), JOBS is a Manager().dict()
# and each job value is a Manager().dict() so the worker can update status/result.
#
# IMPORTANT: On Linux (Docker/Railway), rpy2/R is not fork-safe. We MUST use the
# "spawn" start method for worker processes to avoid memory corruption / crashes.
_mp_ctx = multiprocessing.get_context("spawn")
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
        _initial_nodes = [
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

        def _update_node_states(updates: dict) -> dict:
            # IMPORTANT: job is a Manager().dict(); nested dicts don't propagate if mutated in-place.
            # Always reassign job["node_states"] with a fresh dict copy.
            ns = dict(job.get("node_states") or {})
            ns.update(updates or {})
            job["node_states"] = ns
            return ns

        def _set_node(name: str, state: str) -> None:
            _update_node_states({name: state})

        _update_node_states({k: "pending" for k in _initial_nodes})

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
            _update_node_states({k: "done" for k in _initial_nodes})

            job["status"] = "done"
            job["progress"] = 100
            job["result"] = result
            return

        # ---- Real mode: Aberrance workflow (ψ must be provided by Preparation) ----
        # Import here so stub mode can avoid importing heavy R / LangGraph stack.
        from graph import forensic_workflow

        # We do NOT estimate IRT parameters in the backend.
        # The UI is responsible for generating ψ in Preparation and passing it here.
        psi_data = req.psi_data or []
        if not psi_data:
            job["irt_status"] = "missing"
            job["status"] = "error"
            job["error"] = "Missing psi_data. Generate item parameters (ψ) on Preparation page and retry."
            job["progress"] = 0
            return

        job["psi_data"] = psi_data
        job["irt_status"] = "provided"
        job["progress"] = 35

        # Aberrance detection workflow (run SEQUENTIALLY to avoid concurrent rpy2/R calls)
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

        def _merge_fragment(fragment: dict) -> None:
            if not isinstance(fragment, dict):
                return
            for key, value in fragment.items():
                if key == "flags" and isinstance(value, dict):
                    result["flags"] = {**(result.get("flags") or {}), **value}
                else:
                    result[key] = value

        # Mark router as done (we always run all agents in backend mode)
        _set_node("router", "done")

        # Import agent functions (avoid LangGraph fan-out concurrency).
        from graph import (
            ac_agent,
            as_agent,
            cp_agent,
            manager_synthesizer,
            nm_agent,
            pk_agent,
            pm_agent,
            rg_agent,
            tt_agent,
        )

        try:
            # Execute each agent sequentially. Each returns a partial state dict.
            agent_steps = [
                ("pm_agent", pm_agent),
                ("nm_agent", nm_agent),
                ("ac_agent", ac_agent),
                ("as_agent", as_agent),
                ("rg_agent", rg_agent),
                ("cp_agent", cp_agent),
                ("tt_agent", tt_agent),
                ("pk_agent", pk_agent),
            ]

            for idx, (name, fn) in enumerate(agent_steps, start=1):
                _set_node(name, "running")
                job["progress"] = 35 + int(55 * (idx - 1) / max(1, len(agent_steps) + 1))
                try:
                    frag = fn(dict(result))
                except Exception as e:
                    _set_node(name, "error")
                    job["status"] = "error"
                    job["error"] = f"{name} failed: {e}"
                    return
                _merge_fragment(frag or {})
                _set_node(name, "done")
                job["progress"] = 35 + int(55 * idx / max(1, len(agent_steps) + 1))

            # Synthesizer (LLM)
            _set_node("synthesizer", "running")
            try:
                synth = manager_synthesizer(dict(result))
            except Exception as e:
                _set_node("synthesizer", "error")
                job["status"] = "error"
                job["error"] = f"synthesizer failed: {e}"
                return
            _merge_fragment(synth or {})
            _set_node("synthesizer", "done")

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
    _manager = _mp_ctx.Manager()
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
    p = _mp_ctx.Process(
        target=_child_detect_entry,
        args=(run_id, req.model_dump(), JOBS),
        daemon=True,
    )
    p.start()
    # Remember the worker so /status can see if it crashed.
    PROCS[run_id] = p
    return {"run_id": run_id}


@app.post("/irt")
def run_irt(req: IrtRequest) -> dict:
    """Run IRT in the backend (Docker/Linux) and return ψ item parameters.

    This is used by the Streamlit Preparation page so Windows clients do not need
    local R/rpy2 for ψ estimation.
    """
    load_dotenv()
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
        return {"status": "error", "error": result.get("icc_error") or "IRT returned no item parameters.", "result": result}
    return {"status": "done", "result": result}


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
        "node_states": dict(job.get("node_states") or {}),
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

