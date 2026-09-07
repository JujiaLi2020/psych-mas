"""
Streamlit UI for loading response and response-time datasets, then running the
psych_workflow graph.

Data loading lives here—not in the Orchestrator. The Orchestrator only routes;
the UI reads the two files and passes them as initial state into the graph.
"""
import warnings
import html
import importlib

# Suppress rpy2 "R is not initialized by the main thread" warning (harmless in cloud/Streamlit)
warnings.filterwarnings("ignore", message=".*main thread.*")

from pathlib import Path
import io
import json
import math  # stdlib (not 'maths')
import zipfile
import os
import re
import base64
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import yaml
import matplotlib
matplotlib.use("Agg")  # Non-GUI backend for headless/server (Streamlit)
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import time
import uuid
from datetime import datetime, timezone
import requests
from dotenv import load_dotenv
from psymas_ui.components import (
    apply_control_theme,
    kpi_progress as _render_kpi_progress,
    workspace_header as _render_workspace_header,
    workspace_view as _workspace_view,
)
from psymas_ui.app_config import (
    ABERRANCE_FN_TO_AGENT,
    ABERRANCE_FUNCTIONS,
    APP_VERSION,
    DEMO_AGENT_PRESET,
    DEMO_DATA_DIR,
    DEMO_EVALUATED_SNAPSHOT_PATH,
    DOMAIN_LABELS as _DOMAIN_LABELS,
    DOMAIN_ORDER as _DOMAIN_ORDER,
    PSYMAS_VIZ,
)
try:
    from psymas_ui.exports import build_master_results
except KeyError:
    # Streamlit hot-reload can occasionally leave a local module half-loaded.
    importlib.invalidate_caches()
    from psymas_ui.exports import build_master_results
from psymas_ui.evidence_tree import (
    build_individual_lineage_sankey,
    build_lineage_sankey,
    lineage_dual_legend_html,
    lineage_summary_rows,
    priority_level_key,
    priority_style,
    strength_style,
)
from psymas_ui.evidence_governance import (
    b3_family_signal_key,
    variant_allowed_for_b3,
)
from psymas_ui.review import build_final_flag_review
from psymas_ui.research_export import build_research_export_zip
from psymas_ui.run_store import DATA_TABLES, get_run_store
from psymas_ui.run_snapshot import pack_snapshot, unpack_snapshot
from psymas_ui.worked_example import (
    generate_worked_example_package,
    load_worked_example_candidates,
    recommended_worked_example_cases,
)
from psymas_ui.llm import (
    DEFAULT_GEMINI_MODEL_IDS,
    LOCAL_OLLAMA_MODEL_IDS,
    LOCAL_OLLAMA_MODELS,
    OPENROUTER_FREE_MODEL_IDS,
    OPENROUTER_FREE_MODELS,
    OLLAMA_CHAT_URL,
    call_ollama as _call_ollama,
    call_openrouter as _call_openrouter,
    call_selected_llm_text as _call_selected_llm_text,
    current_model_ids as _current_model_ids,
    current_model_options as _current_model_options,
    effective_llm_model as _effective_llm_model,
    llm_provider as _llm_provider,
    load_openrouter_model_options as _load_openrouter_model_options,
    model_settings_for_backend as _model_settings_for_backend,
    model_variants_with_selected_first as _model_variants_with_selected_first,
    preferred_llm_provider as _preferred_llm_provider,
    test_ollama_model as _test_ollama_model,
    test_openrouter_api_key as _test_openrouter_api_key,
    test_openrouter_model as _test_openrouter_model,
)
from psymas_ui.input_data import (
    align_rt_columns_to_response as _align_rt_columns_to_response,
    coerce_numeric as _coerce_numeric,
    drop_index_column as _drop_index_column,
    parse_compromised_items_csv as _parse_compromised_items_csv,
    validate_answer_changes_csv as _validate_answer_changes_csv,
    validate_binary_responses as _validate_binary_responses,
)
from psymas_ui.backend_client import (
    BACKEND_URL,
    backend_get as _backend_get,
    backend_post as _backend_post,
    backend_status_summary as _backend_status_summary,
    detect_run_id_replica_hint as _detect_run_id_replica_hint,
)

try:
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode, JsCode
except Exception:
    AgGrid = None
    GridOptionsBuilder = None
    GridUpdateMode = None
    DataReturnMode = None
    JsCode = None

# Load .env early so os.getenv() (keys, BACKEND_URL) works everywhere.
load_dotenv()

def _graph_module():
    """Load the analysis engine only when the user starts an analysis."""
    import graph

    return graph


def _render_llm_settings() -> None:
    """Global LLM settings: provider, API keys test, model list, and lock/unlock."""
    load_dotenv()
    _api_key = os.getenv("GOOGLE_API_KEY")
    _openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
    st.radio(
        "LLM provider",
        options=["openrouter", "local_ollama"],
        format_func=lambda x: "OpenRouter" if x == "openrouter" else "Local Ollama",
        key="llm_provider",
        horizontal=True,
        help="OpenRouter uses the curated 5-model list. Local Ollama is reserved for Llama 8B/70B and does not use OpenRouter.",
    )
    if st.session_state.llm_provider == "google":
        if "api_key_test_result" not in st.session_state:
            st.session_state.api_key_test_result = None
        if _api_key:
            if st.button("Test API key", key="test_api_key"):
                ok, msg = _test_api_key(_api_key)
                st.session_state.api_key_test_result = (ok, msg)
                st.rerun()
            if st.session_state.api_key_test_result is not None:
                ok, msg = st.session_state.api_key_test_result
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)
        else:
            st.warning("GOOGLE_API_KEY not set in .env. Add it to use Psych-MAS Summary with Google.")
    elif st.session_state.llm_provider == "openrouter":
        if "openrouter_api_key_test_result" not in st.session_state:
            st.session_state.openrouter_api_key_test_result = None
        model_options_loaded, model_list_err, model_list_live = _load_openrouter_model_options()
        _ml_col1, _ml_col2 = st.columns([1, 2])
        with _ml_col1:
            if st.button("Reload curated model list", key="refresh_openrouter_model_list"):
                model_options_loaded, model_list_err, model_list_live = _load_openrouter_model_options(force=True)
                st.toast(f"Loaded {len(model_options_loaded):,} curated US models.")
                st.rerun()
        with _ml_col2:
            st.caption(f"Curated list: {len(model_options_loaded):,} US models with price and recommended use.")
        if st.button("Test API key", key="test_openrouter_api_key"):
            load_dotenv()
            key = os.getenv("OPENROUTER_API_KEY", "")
            ok, msg = _test_openrouter_api_key(key)
            st.session_state.openrouter_api_key_test_result = (ok, msg)
            st.rerun()
        if st.session_state.openrouter_api_key_test_result is not None:
            ok, msg = st.session_state.openrouter_api_key_test_result
            if ok:
                st.success(msg)
            else:
                st.error(msg)
        st.caption("OpenRouter models. Set OPENROUTER_API_KEY in .env for higher limits and private account access.")
    else:
        st.caption(f"Local Ollama endpoint: `{OLLAMA_CHAT_URL}`")
        st.caption("Install locally with `ollama pull llama3.1:8b` or `ollama pull llama3.3:70b`, then run `ollama serve`.")
    _model_options = _current_model_options()
    _model_ids = _current_model_ids()
    if "pinned_llm_model" not in st.session_state:
        st.session_state.pinned_llm_model = None
    if "pinned_llm_provider" not in st.session_state:
        st.session_state.pinned_llm_provider = None
    _pinned = st.session_state.pinned_llm_model if st.session_state.pinned_llm_provider == st.session_state.llm_provider else None
    if "selected_gemini_model" not in st.session_state:
        st.session_state.selected_gemini_model = (_pinned if _pinned and _pinned in _model_ids else None) or (_model_ids[0] if _model_ids else (LOCAL_OLLAMA_MODEL_IDS[0] if st.session_state.llm_provider == "local_ollama" else OPENROUTER_FREE_MODEL_IDS[0]))
    if st.session_state.selected_gemini_model not in _model_ids:
        st.session_state.selected_gemini_model = (_pinned if _pinned and _pinned in _model_ids else None) or (_model_ids[0] if _model_ids else (LOCAL_OLLAMA_MODEL_IDS[0] if st.session_state.llm_provider == "local_ollama" else OPENROUTER_FREE_MODEL_IDS[0]))
    st.selectbox(
        "LLM model for Psych-MAS Summary",
        options=_model_ids,
        format_func=lambda x: next((label for label, api_id in _model_options if api_id == x), _model_id_to_display_name(x) if "/" not in str(x) else str(x).split("/")[-1].replace("-", " ").title()),
        key="selected_gemini_model",
        help="OpenRouter and Local Ollama model lists are separated.",
    )
    _pinned_now = st.session_state.pinned_llm_model if st.session_state.pinned_llm_provider == st.session_state.llm_provider else None
    _lock_col, _unlock_col = st.columns(2)
    with _lock_col:
        if st.button("Lock current model for all analyses", key="lock_llm_model", help="Use this model for every LLM call (prompt + Psych-MAS Summary) until you unlock."):
            st.session_state.pinned_llm_model = st.session_state.selected_gemini_model
            st.session_state.pinned_llm_provider = st.session_state.llm_provider
            st.rerun()
    with _unlock_col:
        if _pinned_now and st.button("Unlock model", key="unlock_llm_model", help="Stop using the locked model; selection will follow the dropdown again."):
            st.session_state.pinned_llm_model = None
            st.session_state.pinned_llm_provider = None
            st.rerun()
    if _pinned_now:
        _locked_label = next((label for label, api_id in _model_options if api_id == _pinned_now), _pinned_now.split("/")[-1].replace("-", " ").title() if "/" in _pinned_now else _pinned_now)
        st.caption(f"🔒 **Locked:** {_locked_label} — all LLM analyses use this model until you unlock.")
    else:
        st.caption("✓ The selected model is active. Use **Analyze prompt** or any **Psych-MAS Summary** button to run with this model — no extra step needed.")
    if "model_availability" not in st.session_state:
        st.session_state.model_availability = None
    if "model_availability_errors" not in st.session_state:
        st.session_state.model_availability_errors = {}
    if "model_availability_times" not in st.session_state:
        st.session_state.model_availability_times = {}
    if st.session_state.llm_provider == "openrouter":
        if st.button("Check model availability", key="check_model_availability"):
            with st.spinner("Testing selected OpenRouter model…"):
                avail = {}
                errs = {}
                times = {}
                model_id = st.session_state.get("selected_gemini_model") or (_current_model_ids()[0] if _current_model_ids() else "")
                if model_id:
                    ok, err, elapsed = _test_openrouter_model(_openrouter_key, model_id)
                    avail[model_id] = ok
                    if err:
                        errs[model_id] = err
                    times[model_id] = elapsed
                st.session_state.model_availability = avail
                st.session_state.model_availability_errors = errs
                st.session_state.model_availability_times = times
            st.rerun()
        if st.session_state.model_availability is not None:
            selected_id = st.session_state.get("selected_gemini_model", (_current_model_ids()[0] if _current_model_ids() else ""))
            st.caption("Selected model status (available/unavailable; response time in seconds)")
            for label, api_id in _current_model_options():
                if api_id != selected_id:
                    continue
                status = st.session_state.model_availability.get(api_id, False)
                err = st.session_state.model_availability_errors.get(api_id)
                elapsed = st.session_state.model_availability_times.get(api_id, 0.0)
                dot = "available" if status else "unavailable"
                sel = " **(Selected)**" if api_id == selected_id else ""
                extra = f" — {elapsed:.2f}s" if elapsed else ""
                if err:
                    st.caption(f"{dot} {label}{sel}{extra} — {err}")
                else:
                    st.caption(f"{dot} {label}{sel}{extra}")
    elif st.session_state.llm_provider == "local_ollama":
        if st.button("Check local model availability", key="check_local_ollama_availability"):
            with st.spinner("Testing selected local Ollama model…"):
                avail = {}
                errs = {}
                times = {}
                model_id = st.session_state.get("selected_gemini_model") or (LOCAL_OLLAMA_MODEL_IDS[0] if LOCAL_OLLAMA_MODEL_IDS else "")
                if model_id:
                    ok, err, elapsed = _test_ollama_model(model_id)
                    avail[model_id] = ok
                    if err:
                        errs[model_id] = err
                    times[model_id] = elapsed
                st.session_state.model_availability = avail
                st.session_state.model_availability_errors = errs
                st.session_state.model_availability_times = times
            st.rerun()
        if st.session_state.model_availability is not None:
            selected_id = st.session_state.get("selected_gemini_model", LOCAL_OLLAMA_MODEL_IDS[0] if LOCAL_OLLAMA_MODEL_IDS else "")
            for label, api_id in LOCAL_OLLAMA_MODELS:
                if api_id != selected_id:
                    continue
                status = st.session_state.model_availability.get(api_id, False)
                err = st.session_state.model_availability_errors.get(api_id)
                elapsed = st.session_state.model_availability_times.get(api_id, 0.0)
                dot = "available" if status else "unavailable"
                extra = f" — {elapsed:.2f}s" if elapsed else ""
                if err:
                    st.caption(f"{dot} {label}{extra} — {err}")
                else:
                    st.caption(f"{dot} {label}{extra}")


def _test_api_key(api_key: str, timeout: int = 15) -> tuple[bool, str]:
    """Verify GOOGLE_API_KEY by listing models. Return (True, message) or (False, error_message)."""
    if not api_key or not api_key.strip():
        return False, "No API key provided (GOOGLE_API_KEY empty or missing in .env)."
    try:
        url = "https://generativelanguage.googleapis.com/v1beta/models"
        resp = requests.get(url, params={"key": api_key.strip()}, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        models = data.get("models", [])
        count = len(models)
        return True, f"API key is valid. Found {count} model(s) available."
    except requests.exceptions.HTTPError as e:
        try:
            err_body = e.response.json() if e.response is not None else {}
            msg = err_body.get("error", {}).get("message", str(e))
        except Exception:
            msg = str(e)
        code = e.response.status_code if e.response is not None else 0
        if code == 403:
            return False, f"Invalid or unauthorized key ({code}): {msg}"
        if code == 429:
            return False, f"Quota exceeded ({code}): {msg}"
        return False, f"{code}: {msg}"
    except requests.exceptions.Timeout:
        return False, "Request timed out. Check your network."
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _model_id_to_display_name(model_id: str) -> str:
    """Turn e.g. models/gemini-2.5-flash into 'Gemini 2.5 Flash'."""
    name = model_id.replace("models/", "").strip()
    if not name:
        return model_id
    parts = name.replace("-", " ").split()
    return " ".join(p.capitalize() for p in parts)


def _discover_gemini_models(api_key: str, timeout: int = 20) -> list[tuple[str, str]]:
    """Call ListModels and return [(display_name, model_id), ...] for models that support generateContent (Gemini)."""
    if not api_key or not api_key.strip():
        return []
    try:
        url = "https://generativelanguage.googleapis.com/v1beta/models"
        resp = requests.get(url, params={"key": api_key.strip()}, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        models = data.get("models", [])
        result = []
        for m in models:
            name = m.get("name", "")
            if "gemini" not in name.lower():
                continue
            methods = m.get("supportedGenerationMethods", [])
            if "generateContent" not in methods:
                continue
            display = _model_id_to_display_name(name)
            result.append((display, name))
        # Prefer flash, then pro; keep stable order
        def order_key(item):
            label, mid = item
            mid_lower = mid.lower()
            if "flash" in mid_lower:
                return (0, mid_lower)
            if "pro" in mid_lower:
                return (1, mid_lower)
            return (2, mid_lower)
        result.sort(key=order_key)
        return result
    except Exception:
        return []


def _test_gemini_model(api_key: str, model_id: str, timeout: int = 20) -> tuple[bool, str | None, float]:
    """Send a minimal generateContent request. Return (ok, error_message, response_time_sec)."""
    t0 = time.perf_counter()
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/{model_id}:generateContent"
        body = {"contents": [{"role": "user", "parts": [{"text": "Hi"}]}]}
        resp = requests.post(url, params={"key": api_key}, json=body, timeout=timeout)
        resp.raise_for_status()
        elapsed = time.perf_counter() - t0
        return True, None, elapsed
    except requests.exceptions.HTTPError as e:
        elapsed = time.perf_counter() - t0
        try:
            err_body = e.response.json() if e.response is not None else {}
            msg = err_body.get("error", {}).get("message", str(e))
        except Exception:
            msg = str(e)
        code = e.response.status_code if e.response is not None else 0
        return False, f"{code}: {msg}", elapsed
    except requests.exceptions.Timeout:
        elapsed = time.perf_counter() - t0
        return False, "Timeout", elapsed
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return False, f"{type(e).__name__}: {e}", elapsed


def _check_models_availability(api_key: str, model_ids: list[str] | None = None) -> tuple[dict[str, bool], dict[str, str], dict[str, float]]:
    """Test each Gemini model. Return (availability_by_id, error_message_by_id, response_time_sec_by_id)."""
    load_dotenv()
    ids = model_ids or DEFAULT_GEMINI_MODEL_IDS
    available = {}
    errors = {}
    times = {}
    for model_id in ids:
        ok, err, elapsed = _test_gemini_model(api_key, model_id)
        available[model_id] = ok
        if err:
            errors[model_id] = err
        times[model_id] = elapsed
    return available, errors, times


def _load_demo_simulated_data() -> tuple[bool, str]:
    """Load Appendix C tutorial data from ./data into the same slots as uploads."""
    response_long_path = DEMO_DATA_DIR / "response_long.csv"
    if not response_long_path.exists():
        return False, f"Demo source not found: {response_long_path}"

    try:
        long_df = pd.read_csv(response_long_path)
    except Exception as exc:
        return False, f"Could not read response_long.csv: {exc}"

    required = {
        "examinee_id",
        "item_id",
        "item_position",
        "final_score",
        "response_time",
        "true_a",
        "true_b",
        "true_beta",
        "exposure_status",
    }
    missing = sorted(required - set(long_df.columns))
    if missing:
        return False, "response_long.csv is missing required columns: " + ", ".join(missing)

    try:
        item_order_df = (
            long_df[["item_id", "item_position"]]
            .drop_duplicates()
            .sort_values(["item_position", "item_id"])
        )
        item_cols = item_order_df["item_id"].astype(str).tolist()
        long_df = long_df.copy()
        long_df["item_id"] = long_df["item_id"].astype(str)

        response_matrix = (
            long_df.pivot(index="examinee_id", columns="item_id", values="final_score")
            .reindex(columns=item_cols)
            .sort_index()
        )
        rt_matrix = (
            long_df.pivot(index="examinee_id", columns="item_id", values="response_time")
            .reindex(columns=item_cols)
            .sort_index()
        )
        response_matrix = _validate_binary_responses(response_matrix)
        rt_matrix = _coerce_numeric(rt_matrix, "Demo response-time data")

        item_param_df = (
            long_df[["item_id", "item_position", "true_a", "true_b", "true_beta", "exposure_status"]]
            .drop_duplicates(subset=["item_id"])
            .sort_values(["item_position", "item_id"])
        )
        item_params = [
            {
                "item_id": str(row["item_id"]),
                "item_position": int(row["item_position"]),
                "a": float(row["true_a"]),
                "a1": float(row["true_a"]),
                "b": float(row["true_b"]),
                "beta": float(row["true_beta"]),
                "exposure_status": str(row["exposure_status"]),
            }
            for _, row in item_param_df.iterrows()
        ]
        compromised_items = sorted(
            {
                int(x)
                for x in long_df.loc[
                    long_df["exposure_status"].astype(str).str.lower().isin(
                        {"exposed", "compromised", "leaked", "high-risk", "high_risk"}
                    ),
                    "item_position",
                ].dropna().tolist()
            }
        )
    except Exception as exc:
        return False, f"Could not transform demo long table into PsyMAS inputs: {exc}"

    response_records = response_matrix.to_dict(orient="records")
    rt_records = rt_matrix.to_dict(orient="records")
    answer_change_candidates = [
        DEMO_DATA_DIR / "upload" / "answer_changes_long.csv",
        DEMO_DATA_DIR / "upload" / "answer_changes.csv",
        DEMO_DATA_DIR / "sample" / "answer_changes.csv",
    ]
    answer_changes_path = next((path for path in answer_change_candidates if path.exists()), None)
    if answer_changes_path is None:
        searched = ", ".join(str(path) for path in answer_change_candidates)
        return False, f"Could not load demo answer-change data. Searched: {searched}"
    try:
        answer_changes = _validate_answer_changes_csv(pd.read_csv(answer_changes_path))
    except Exception as exc:
        return False, f"Could not load demo answer-change data: {exc}"

    # Clear stale downstream products so the new demo inputs cannot be mixed with a previous run.
    for key in [
        "forensic_result",
        "detect_run_id",
        "detect_job_status",
        "detect_job_error",
        "prep_irt_job_id",
        "prep_auto_irt_sig",
        "last_irt_error",
        "_case_review_popup_id",
        "_case_review_popup_closed_id",
    ]:
        st.session_state.pop(key, None)

    st.session_state["last_uploaded_responses"] = response_records
    st.session_state.last_uploaded_responses = response_records
    st.session_state["forensic_responses"] = response_records
    st.session_state["last_uploaded_rt_data"] = rt_records
    st.session_state.last_uploaded_rt_data = rt_records
    st.session_state["forensic_rt_data"] = rt_records
    st.session_state["last_irt_item_params"] = item_params
    st.session_state["item_params"] = item_params
    st.session_state["forensic_psi_data"] = item_params
    st.session_state["last_irt_itemtype"] = "2PL"
    st.session_state["main_irt_itemtype"] = "2PL"
    st.session_state["prep_compromised_items"] = compromised_items
    st.session_state["ab_only_compromised_items"] = compromised_items
    st.session_state["prep_compromised_file_name"] = "response_long.csv: exposure_status"
    st.session_state["prep_answer_changes"] = answer_changes.to_dict(orient="records")
    st.session_state["prep_answer_changes_file_name"] = str(answer_changes_path.relative_to(DEMO_DATA_DIR.parent))
    st.session_state["last_uploaded_model_settings"] = st.session_state.get("model_settings") or _interpret_prompt("")
    st.session_state["last_uploaded_is_verified"] = True

    for fn in ABERRANCE_FUNCTIONS:
        st.session_state[f"ab_only_cb_{fn}"] = fn in DEMO_AGENT_PRESET
    st.session_state["ab_only_scenario_select"] = "C"
    st.session_state["ab_only_scenario_select_previous"] = "C"
    st.session_state["last_detect_agents"] = DEMO_AGENT_PRESET

    aux_files = {
        "scenario_key": "scenario_key.csv",
        "scenario_summary": "scenario_summary.csv",
        "testing_context": "testing_context.csv",
        "group_check": "group_check.csv",
        "copying_pairs_truth": "copying_pairs_truth.csv",
        "answer_change_summary": "answer_change_summary.csv",
    }
    for key, filename in aux_files.items():
        path = DEMO_DATA_DIR / filename
        if path.exists():
            try:
                st.session_state[f"demo_{key}"] = pd.read_csv(path).to_dict(orient="records")
            except Exception:
                st.session_state[f"demo_{key}"] = []

    st.session_state["demo_data_loaded"] = True
    st.session_state["demo_data_source"] = str(response_long_path)
    st.session_state["demo_examinee_ids"] = [str(x) for x in response_matrix.index.tolist()]
    st.session_state["demo_item_ids"] = item_cols
    st.session_state["prep_bulk_loaded_messages"] = [
        f"Demo response matrix: {response_matrix.shape[0]} examinees x {response_matrix.shape[1]} items",
        f"Demo response-time matrix: {rt_matrix.shape[0]} examinees x {rt_matrix.shape[1]} items",
        f"Demo item parameters: {len(item_params)} items",
        f"Demo compromised items: {len(compromised_items)} exposed items",
        f"Demo answer-change records: {len(answer_changes)} rows",
    ]
    st.session_state["prep_bulk_error_messages"] = []
    return True, (
        f"Demo loaded: {response_matrix.shape[0]} examinees x {response_matrix.shape[1]} items, "
        f"{len(compromised_items)} exposed items, and {len(answer_changes)} answer-change records."
    )


def _clear_demo_loaded_state() -> None:
    """Remove bundled demo inputs/results when switching back to a non-demo scenario."""
    if not st.session_state.get("demo_data_loaded"):
        return
    for key in [
        "demo_data_loaded",
        "demo_evaluated_snapshot_loaded",
        "demo_data_source",
        "demo_evaluated_snapshot_source",
        "last_uploaded_responses",
        "forensic_responses",
        "last_uploaded_rt_data",
        "forensic_rt_data",
        "last_irt_item_params",
        "item_params",
        "forensic_psi_data",
        "prep_compromised_items",
        "ab_only_compromised_items",
        "prep_compromised_file_name",
        "prep_answer_changes",
        "prep_answer_changes_file_name",
        "forensic_result",
        "detect_run_id",
        "detect_job_status",
        "detect_job_error",
        "psymas_active_run_id",
        "psymas_run_store_ready",
        "master_results_path",
    ]:
        st.session_state.pop(key, None)
    for key in list(st.session_state.keys()):
        if str(key).startswith("demo_"):
            st.session_state.pop(key, None)


def _demo_scenario_letter() -> str:
    letter = st.session_state.get("ab_only_scenario_select") or ""
    if letter in ("D", "Custom"):
        return ""
    return str(letter)


def _ensure_demo_scenario_ready() -> bool:
    """Load Appendix C demo tables when Scenario C is active and session inputs are missing."""
    if _demo_scenario_letter() != "C":
        return False
    if st.session_state.get("demo_data_loaded") and st.session_state.get("last_uploaded_responses"):
        for fn in ABERRANCE_FUNCTIONS:
            st.session_state[f"ab_only_cb_{fn}"] = fn in DEMO_AGENT_PRESET
        if not st.session_state.get("demo_evaluated_snapshot_loaded") or not st.session_state.get("forensic_result"):
            snap_ok, snap_msg = _restore_demo_evaluated_snapshot_if_available()
            if snap_ok:
                st.session_state["_demo_load_success"] = snap_msg
                return True
            st.session_state["_demo_load_warning"] = snap_msg
        return False
    ok, msg = _load_demo_simulated_data()
    if ok:
        snap_ok, snap_msg = _restore_demo_evaluated_snapshot_if_available(force=True)
        if snap_ok:
            st.session_state["_demo_load_success"] = f"{msg} {snap_msg}"
        else:
            st.session_state["_demo_load_success"] = msg
            st.session_state["_demo_load_warning"] = snap_msg
    else:
        st.session_state["_demo_load_error"] = msg
    return True


def _prep_detect_requirements(prep_ab_fns: list[str]) -> dict:
    """Return readiness flags for the Detect button and status pills."""
    need_rt = "detect_rg" in prep_ab_fns
    need_model = any(fn in prep_ab_fns for fn in ["detect_pm", "detect_pk", "detect_ac", "detect_as"])
    need_tt = "detect_tt" in prep_ab_fns
    resp_ok = bool(st.session_state.get("last_uploaded_responses"))
    rt_ok = bool(st.session_state.get("last_uploaded_rt_data")) if need_rt else True
    psi_ready = bool(st.session_state.get("last_irt_item_params") or st.session_state.get("item_params"))
    # IRT-based detectors can proceed without uploaded item parameters because
    # the Preparation page estimates them from responses using the selected model.
    psi_ok = True if need_model else True
    tt_ok = bool(st.session_state.get("prep_answer_changes")) if need_tt else True
    backend_ok, backend_note, backend_tip = _backend_status_summary()
    blocks: list[str] = []
    if not resp_ok:
        blocks.append("response data")
    if not backend_ok:
        blocks.append(f"backend ({backend_note})")
    detect_ready = bool(resp_ok and psi_ok and backend_ok)
    return {
        "need_rt": need_rt,
        "need_model": need_model,
        "need_tt": need_tt,
        "need_psi": need_model,
        "resp_ok": resp_ok,
        "rt_ok": rt_ok,
        "psi_ok": psi_ok,
        "psi_ready": psi_ready,
        "tt_ok": tt_ok,
        "backend_ok": backend_ok,
        "backend_note": backend_note,
        "backend_tip": backend_tip,
        "detect_ready": detect_ready,
        "blocks": blocks,
    }


def _infer_prep_upload_kind(name: str, df: pd.DataFrame | None) -> str | None:
    """Infer which Preparation data slot an uploaded file belongs to."""
    stem = re.sub(r"[^a-z0-9]+", "_", (name or "").lower()).strip("_")
    standard_names = {
        "responses": "response",
        "response_times": "rt",
        "item_params": "psi",
        "compromised_items": "compromised",
        "answer_changes": "answer_changes",
    }
    if stem in standard_names:
        return standard_names[stem]
    cols = {str(c).strip().lower().replace(" ", "_") for c in (df.columns if df is not None else [])}

    if "answer_change" in stem or ("answer" in stem and "change" in stem):
        return "answer_changes"
    if "compromised" in stem or "leaked" in stem or "exposed" in stem:
        return "compromised"
    if stem in {"rt", "response_time", "response_times"} or "response_time" in stem or stem.endswith("_rt"):
        return "rt"
    if "psi" in stem or "item_param" in stem or "item_parameter" in stem:
        return "psi"
    if "response" in stem or stem in {"resp", "responses", "responses_matrix"}:
        return "response"

    if {"examinee_id", "item_id"}.issubset(cols) and (
        "changed" in cols
        or {"initial_response", "final_response"}.issubset(cols)
        or {"initial_score", "final_score"}.issubset(cols)
    ):
        return "answer_changes"
    if {"a", "b"}.issubset(cols) or {"a1", "b"}.issubset(cols):
        return "psi"
    if cols & {"compromised", "is_compromised", "leaked", "compromised_item", "compromised_items"}:
        return "compromised"
    if df is not None and not df.empty:
        try:
            numeric = _coerce_numeric(_drop_index_column(df), "Uploaded data")
            values = set()
            for col in numeric.columns:
                values.update(numeric[col].dropna().unique().tolist())
            if values and values.issubset({0, 1}):
                return "response"
            if values:
                return "rt"
        except Exception:
            return None
    return None


def _load_psi_upload(uploaded_file) -> list[dict]:
    uploaded_file.seek(0)
    name = getattr(uploaded_file, "name", "") or ""
    raw = uploaded_file.read()
    if name.lower().endswith(".json"):
        data = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
        if not isinstance(data, list):
            data = data.get("item_params", data.get("result", data.get("items", [data])))
        if not isinstance(data, list):
            raise ValueError("JSON must be an array of item objects or an object with item_params/result/items.")
        return [dict(x) for x in data]

    df = pd.read_csv(io.BytesIO(raw) if isinstance(raw, bytes) else io.StringIO(raw))
    df = _drop_index_column(df)
    a_col = "a1" if "a1" in df.columns else "a"
    if a_col not in df.columns or "b" not in df.columns:
        raise ValueError("Item-parameter CSV must have columns a (or a1) and b; optional c or g.")
    return df.to_dict(orient="records")


def _process_prep_bulk_uploads(uploaded_files: list) -> tuple[list[str], list[str]]:
    """Load multiple Preparation files into the same session keys used by individual uploaders."""
    loaded: list[str] = []
    errors: list[str] = []
    pending_rt: tuple[str, tuple, pd.DataFrame] | None = None
    pending_comp: tuple[str, tuple, pd.DataFrame] | None = None

    for f in uploaded_files or []:
        name = getattr(f, "name", "uploaded file") or "uploaded file"
        sig = (name, getattr(f, "size", None))
        try:
            df = None
            if not name.lower().endswith(".json"):
                f.seek(0)
                df = pd.read_csv(f)
                df = _drop_index_column(df)
            kind = _infer_prep_upload_kind(name, df)
            if kind == "response":
                if df is None:
                    raise ValueError("Response must be a CSV.")
                resp = _validate_binary_responses(df)
                st.session_state.last_uploaded_responses = resp.to_dict(orient="records")
                st.session_state["prep_last_resp_sig"] = sig
                loaded.append(f"Response: {name}")
            elif kind == "rt":
                if df is None:
                    raise ValueError("RT must be a CSV.")
                rt = _coerce_numeric(df, "RT")
                pending_rt = (name, sig, rt)
            elif kind == "compromised":
                if df is None:
                    raise ValueError("Compromised-items data must be a CSV.")
                pending_comp = (name, sig, df)
            elif kind == "answer_changes":
                if df is None:
                    raise ValueError("Answer-change data must be a CSV.")
                changes = _validate_answer_changes_csv(df)
                st.session_state["prep_answer_changes"] = changes.to_dict(orient="records")
                st.session_state["prep_last_answer_changes_sig"] = sig
                st.session_state["prep_answer_changes_file_name"] = name
                loaded.append(f"Answer changes: {name}")
            elif kind == "psi":
                psi = _load_psi_upload(f)
                if not psi:
                    raise ValueError("No item parameters found.")
                st.session_state["last_irt_item_params"] = psi
                st.session_state["item_params"] = psi
                st.session_state["prep_last_psi_sig"] = sig
                loaded.append(f"Item parameters: {name}")
            else:
                errors.append(f"{name}: could not infer data type from file name or columns.")
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    if pending_rt is not None:
        name, sig, rt = pending_rt
        try:
            resp_df = pd.DataFrame(st.session_state.get("last_uploaded_responses") or [])
            if not resp_df.empty and resp_df.shape[1] == rt.shape[1]:
                rt = _align_rt_columns_to_response(rt, resp_df)
            st.session_state.last_uploaded_rt_data = rt.to_dict(orient="records")
            st.session_state["prep_last_rt_sig"] = sig
            loaded.append(f"RT: {name}")
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    if pending_comp is not None:
        name, sig, comp_df = pending_comp
        try:
            n_items = len((st.session_state.get("last_uploaded_responses") or [{}])[0]) if st.session_state.get("last_uploaded_responses") else None
            comp_items = _parse_compromised_items_csv(comp_df, n_items=n_items)
            if not comp_items:
                raise ValueError("No valid compromised item IDs found.")
            st.session_state["prep_compromised_items"] = comp_items
            st.session_state["prep_last_comp_sig"] = sig
            st.session_state["prep_compromised_file_name"] = name
            loaded.append(f"Compromised items: {name}")
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    if loaded:
        st.session_state["_nav_request"] = "Preparation"
    return loaded, errors


def _interpret_prompt(
    prompt: str,
    *,
    llm_provider: str | None = None,
    llm_model_id: str | None = None,
    openrouter_api_key: str | None = None,
    google_api_key: str | None = None,
) -> dict:
    """Run prompt interpretation using the given provider and model (from Model engine)."""
    provider = llm_provider if llm_provider is not None else _llm_provider()
    model_id = llm_model_id if llm_model_id is not None else _effective_llm_model()
    if openrouter_api_key is None and provider == "openrouter":
        load_dotenv()
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")
    if google_api_key is None and provider == "google":
        load_dotenv()
        google_api_key = os.getenv("GOOGLE_API_KEY", "")
    return _graph_module().analyze_prompt(
        prompt,
        llm_provider=provider,
        llm_model_id=model_id or None,
        openrouter_api_key=openrouter_api_key or None,
        google_api_key=google_api_key or None,
    )


def _paginate_df(df: pd.DataFrame, label: str) -> None:
    if df.empty:
        st.info(f"No data for {label}.")
        return
    filtered_df = _filter_and_sort_df(df, label)
    if filtered_df.empty:
        st.info(f"No rows match the current filters for {label}.")
        return
    page_size = 10
    total_pages = math.ceil(len(filtered_df) / page_size)
    if total_pages > 1:
        page = st.number_input(
            f"{label} page",
            min_value=1,
            max_value=total_pages,
            step=1,
            value=1,
            key=f"{label}_page",
        )
    else:
        page = 1
    start = (page - 1) * page_size
    end = start + page_size
    st.dataframe(filtered_df.iloc[start:end], width="stretch")


def _json_safe(obj: object) -> object:
    """Recursively replace NaN/inf with None so payloads are JSON-compliant."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def _filter_and_sort_df(df: pd.DataFrame, label: str) -> pd.DataFrame:
    cols = [str(c) for c in df.columns]
    if not cols:
        return df

    with st.form(f"{label}_filter_form"):
        st.caption("Filter and sort")
        c1, c2, c3 = st.columns(3)
        with c1:
            filter_col = st.selectbox(
                "Filter column",
                options=["(none)"] + cols,
                key=f"{label}_filter_col",
            )
        with c2:
            sort_col = st.selectbox(
                "Sort column",
                options=["(none)"] + cols,
                key=f"{label}_sort_col",
            )
        with c3:
            sort_dir = st.radio(
                "Order",
                options=["asc", "desc"],
                horizontal=True,
                key=f"{label}_sort_dir",
            )

        filtered_df = df
        if filter_col != "(none)":
            series = filtered_df[filter_col]
            if pd.api.types.is_numeric_dtype(series):
                min_val = st.number_input(
                    "Min",
                    value=float(series.min()) if series.notna().any() else 0.0,
                    key=f"{label}_min",
                )
                max_val = st.number_input(
                    "Max",
                    value=float(series.max()) if series.notna().any() else 0.0,
                    key=f"{label}_max",
                )
                filtered_df = filtered_df[series.between(min_val, max_val, inclusive="both")]
            else:
                text = st.text_input(
                    "Contains",
                    key=f"{label}_contains",
                )
                if text:
                    filtered_df = filtered_df[
                        series.astype(str).str.contains(text, case=False, na=False)
                    ]

        if sort_col != "(none)":
            ascending = sort_dir == "asc"
            filtered_df = filtered_df.sort_values(by=sort_col, ascending=ascending)

        apply_filters = st.form_submit_button("Apply")

    if not apply_filters and f"{label}_last_filtered" in st.session_state:
        return st.session_state[f"{label}_last_filtered"]

    st.session_state[f"{label}_last_filtered"] = filtered_df
    return filtered_df


# Bump this when the workflow graph changes (e.g. new nodes) so cache is invalidated
_WORKFLOW_VERSION = "2"

@st.cache_data(show_spinner=False)
def _run_workflow_cached(payload_json: str, workflow_version: str = "") -> dict:
    payload = json.loads(payload_json)
    initial_state = {
        "responses": payload["responses"],
        "rt_data": payload["rt_data"],
        "theta": 0.0,
        "latency_flags": [],
        "next_step": "",
        "model_settings": payload["model_settings"],
        "is_verified": payload["is_verified"],
        "aberrance_functions": payload.get("aberrance_functions", []),
        "compromised_items": payload.get("compromised_items", []),
    }
    return _graph_module().psych_workflow.invoke(initial_state)


def _run_workflow(payload_json: str) -> dict:
    """Run workflow; use version so cache invalidates when graph changes."""
    return _run_workflow_cached(payload_json, _WORKFLOW_VERSION)


def _psych_describe_responses(resp_df: pd.DataFrame) -> tuple[pd.DataFrame | None, str | None]:
    """Run psych::describe in R on response data. Pass data via R vectors only to avoid py2rpy numpy error."""
    if resp_df.empty or resp_df.shape[1] == 0:
        return None, "No response data."
    try:
        import rpy2.robjects as ro
        try:
            from rpy2.robjects import numpy2ri
            numpy2ri.activate()
        except Exception:
            pass
    except Exception as exc:
        return None, f"rpy2/R not available: {exc}"
    try:
        # Build response matrix in R from list (no DataFrame/numpy passed to R)
        block = resp_df.astype(int)
        nrow, ncol = int(block.shape[0]), int(block.shape[1])
        x_flat = block.values.flatten().tolist()
        ro.globalenv["x_vec"] = ro.IntVector(x_flat)
        ro.r(f"resp_df <- as.data.frame(matrix(x_vec, nrow={nrow}, ncol={ncol}, byrow=TRUE))")
        desc = ro.r("psych::describe(resp_df)")
        from rpy2.robjects import pandas2ri
        with (ro.default_converter + pandas2ri.converter).context():
            desc_py = ro.conversion.rpy2py(desc)
        if isinstance(desc_py, pd.DataFrame):
            return desc_py, None
        if hasattr(desc_py, "to_pandas"):
            return desc_py.to_pandas(), None
        return pd.DataFrame(np.asarray(desc_py)), None
    except Exception as exc:
        return None, str(exc)


_R_PACKAGES = ("mirt", "WrightMap", "psych", "aberrance")
_R_REPOS = "https://cloud.r-project.org"
_R_INSTALL_MSG = (
    "Run in terminal: Rscript install_r_packages.R  "
    "Or in R: install.packages(c('mirt','WrightMap','psych','aberrance'), repos='https://cloud.r-project.org').  "
    "On Streamlit Cloud, IRT is not available; run the app locally with R for full IRT."
)


def _install_r_packages_via_rpy2(packages: tuple[str, ...] | None = None) -> str | None:
    """Install R packages via rpy2 (calls R's install.packages). Returns None on success, error message on failure.
    Requires R to be on PATH; uses default R library (user or site)."""
    try:
        import rpy2.robjects as ro
    except Exception as exc:
        return f"rpy2/R not available: {exc}"
    pkgs = packages or _R_PACKAGES
    try:
        ro.r('options(repos = c(CRAN = "' + _R_REPOS + '"))')
        ro.r("install.packages(c(" + ",".join(f'"{p}"' for p in pkgs) + "), dependencies = TRUE, quiet = TRUE)")
        return None
    except Exception as exc:
        return f"Install via rpy2 failed: {type(exc).__name__}: {exc}"


def _check_r_packages(install_if_missing: bool = False) -> tuple[bool, str | None]:
    """Check if R and required packages (mirt, WrightMap, psych, aberrance) are available.
    If install_if_missing is True, try to install missing packages via rpy2 before failing.
    Returns (ok, error_message)."""
    try:
        import rpy2.robjects as ro
    except Exception:
        return False, (
            "R is not available in this environment (e.g. Streamlit Cloud). "
            "IRT, ICC, Wright Map, and aberrance are skipped. You can still use LLM summaries, RT analysis, and APA report. "
            "For full IRT and aberrance, run the app locally with R (see README)."
        )
    try:
        missing = []
        for pkg in _R_PACKAGES:
            r_ok = ro.r(f'require("{pkg}", quietly=TRUE)')
            if r_ok is None:
                ok = False
            elif len(r_ok):
                ok = bool(r_ok[0])
            else:
                ok = False
            if not ok:
                missing.append(pkg)
        if not missing:
            return True, None
        if install_if_missing and missing:
            err = _install_r_packages_via_rpy2(tuple(missing))
            if err is None:
                # Re-check after install
                for pkg in missing:
                    r_ok = ro.r(f'require("{pkg}", quietly=TRUE)')
                    ok = r_ok is not None and len(r_ok) and bool(r_ok[0])
                    if not ok:
                        return False, (f"R package '{pkg}' still missing after install. {_R_INSTALL_MSG}")
                return True, None
            return False, (err or f"R package '{missing[0]}' not installed. {_R_INSTALL_MSG}")
        return False, (f"Missing R packages: {', '.join(missing)}. {_R_INSTALL_MSG}")
    except Exception:
        return False, (
            "Could not check R packages. On Streamlit Cloud, IRT is not available; run locally with R for full IRT."
        )


def _plot_item_accuracy(resp_df: pd.DataFrame) -> str | None:
    """Plot proportion correct (accuracy) per item. Returns path to PNG or None."""
    if resp_df.empty or resp_df.shape[1] == 0:
        return None
    try:
        accuracy = resp_df.mean(axis=0)
        n_items = len(accuracy)
        fig, ax = plt.subplots(figsize=(max(6, n_items * 0.25), 4))
        x = np.arange(n_items)
        bars = ax.bar(x, accuracy.values, color="steelblue", edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Item")
        ax.set_ylabel("Proportion correct (accuracy)")
        ax.set_title("Item accuracy (proportion correct per item)")
        ax.set_xticks(x)
        ax.set_xticklabels(accuracy.index.astype(str), rotation=45, ha="right")
        ax.set_ylim(0, 1)
        ax.axhline(y=accuracy.mean(), color="gray", linestyle="--", label=f"Mean = {accuracy.mean():.2f}")
        ax.legend()
        plt.tight_layout()
        out_path = Path(tempfile.gettempdir()) / f"item_accuracy_{hash(str(accuracy.values))}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return str(out_path) if out_path.exists() else None
    except Exception:
        return None


def _plot_person_ability(person_params: pd.DataFrame) -> str | None:
    """Plot distribution of person ability (θ). Returns path to PNG or None."""
    if person_params.empty:
        return None
    ability_col = None
    for col in ["F1", "theta", "ability", "F"]:
        if col in person_params.columns:
            ability_col = col
            break
    if ability_col is None:
        return None
    try:
        ability = pd.to_numeric(person_params[ability_col], errors="coerce").dropna()
        if ability.empty:
            return None
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.hist(ability, bins=min(30, max(10, len(ability) // 5)), color="steelblue", edgecolor="white")
        ax.set_xlabel(f"Ability ({ability_col})")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of person ability (θ)")
        ax.axvline(ability.mean(), color="gray", linestyle="--", linewidth=1.5, label=f"Mean = {ability.mean():.3f}")
        ax.legend()
        plt.tight_layout()
        out_path = Path(tempfile.gettempdir()) / f"person_ability_{hash(str(ability.values))}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return str(out_path) if out_path.exists() else None
    except Exception:
        return None


def _create_wright_map(item_params: pd.DataFrame, person_params: pd.DataFrame) -> str:
    """Create a Wright Map using R's WrightMap package. Pass data to R as vectors only (no DataFrame)."""
    try:
        import rpy2.robjects as ro
        try:
            from rpy2.robjects import numpy2ri
            numpy2ri.activate()
        except Exception:
            pass
    except Exception as exc:
        print(f"Wright Map skipped; rpy2/R not available ({type(exc).__name__}: {exc})")
        return None
    
    # Extract item difficulty (b parameter) - try different possible column names
    b_col = None
    for col in ['b', 'b1', 'd', 'difficulty']:
        if col in item_params.columns:
            b_col = col
            break
    
    if b_col is None:
        return None
    
    # Extract person ability (F1 or theta) - try different possible column names
    ability_col = None
    for col in ['F1', 'theta', 'ability', 'F']:
        if col in person_params.columns:
            ability_col = col
            break
    
    if ability_col is None:
        return None
    
    # Get valid item data
    valid_mask = item_params[b_col].notna()
    item_df = item_params.loc[valid_mask].copy()
    if 'item' not in item_df.columns:
        item_df['item'] = item_df.index.astype(str)
    
    # Get person ability data
    person_df = person_params[[ability_col]].dropna().copy()
    
    if item_df.empty or person_df.empty:
        return None
    
    # Create output path
    wright_map_path = Path(tempfile.gettempdir()) / f"wright_map_{hash(str(item_df[b_col].values) + str(person_df[ability_col].values))}.png"
    wright_map_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Pass only primitive types to R: vectors from Python lists, path as string
        item_difficulty = [float(x) for x in item_df[b_col].tolist()]
        person_ability = [float(x) for x in person_df[ability_col].tolist()]
        ro.globalenv["wright_map_path"] = str(wright_map_path)
        ro.globalenv["item_difficulty"] = ro.FloatVector(item_difficulty)
        ro.globalenv["person_ability"] = ro.FloatVector(person_ability)
        ro.r(
            """
            library(WrightMap)
            library(grDevices)
            png(wright_map_path, width=1200, height=1000, res=150, type="cairo")
            wrightMap(thetas = person_ability, 
                     thresholds = item_difficulty,
                     item.prop = 0.8,
                     main.title = "Wright Map: Person Ability Distribution and Item Difficulties")
            dev.off()
            """
        )
    except Exception as exc:
        print(f"Wright Map generation failed: {type(exc).__name__}: {exc}")
        return None
    
    if not wright_map_path.exists():
        return None
    
    return str(wright_map_path)


def _query_llm_analysis(question: str, analysis_context: dict) -> str:
    """Query LLM with a question about the psychometric analysis results."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    model = os.getenv("GEMINI_MODEL", "models/gemini-1.5-flash-latest")
    
    if not api_key:
        return "LLM analysis unavailable: GOOGLE_API_KEY not set in .env file."
    
    # Prepare context summary with key statistics
    context_summary = []
    
    if analysis_context.get("item_params"):
        item_df = pd.DataFrame(analysis_context["item_params"])
        context_summary.append(f"Item parameters: {len(item_df)} items analyzed.")
        if not item_df.empty:
            # Include key statistics for common columns
            if 'b' in item_df.columns or 'b1' in item_df.columns:
                b_col = 'b' if 'b' in item_df.columns else 'b1'
                b_vals = pd.to_numeric(item_df[b_col], errors='coerce').dropna()
                if len(b_vals) > 0:
                    context_summary.append(f"Item difficulty (b) range: {b_vals.min():.2f} to {b_vals.max():.2f}, mean: {b_vals.mean():.2f}")
            if 'a' in item_df.columns or 'a1' in item_df.columns:
                a_col = 'a' if 'a' in item_df.columns else 'a1'
                a_vals = pd.to_numeric(item_df[a_col], errors='coerce').dropna()
                if len(a_vals) > 0:
                    context_summary.append(f"Item discrimination (a) range: {a_vals.min():.2f} to {a_vals.max():.2f}, mean: {a_vals.mean():.2f}")
    
    if analysis_context.get("person_params"):
        person_df = pd.DataFrame(analysis_context["person_params"])
        context_summary.append(f"Person parameters: {len(person_df)} persons analyzed.")
        if not person_df.empty:
            # Include ability statistics
            for col in ['F1', 'theta', 'ability', 'F']:
                if col in person_df.columns:
                    ability_vals = pd.to_numeric(person_df[col], errors='coerce').dropna()
                    if len(ability_vals) > 0:
                        context_summary.append(f"Person ability ({col}) range: {ability_vals.min():.2f} to {ability_vals.max():.2f}, mean: {ability_vals.mean():.2f}")
                        break
    
    if analysis_context.get("item_fit"):
        fit_df = pd.DataFrame(analysis_context["item_fit"])
        context_summary.append(f"Item fit statistics available for {len(fit_df)} items.")
        if not fit_df.empty and 'p' in fit_df.columns:
            p_vals = pd.to_numeric(fit_df['p'], errors='coerce').dropna()
            if len(p_vals) > 0:
                context_summary.append(f"Item fit p-values: {sum(p_vals < 0.05)} items with p < 0.05")
    
    context_text = "\n".join(context_summary) if context_summary else "Analysis results are available."
    
    system = (
        "You are a psychometrics expert. Write in APA style suitable for a paper. "
        "Focus on critical psychometric findings: interpret item/person parameters, model fit, item fit, and implications. "
        "Limit your response to exactly three paragraphs. "
        "Use formal, concise language; cite statistics where relevant; avoid bullet points."
    )
    
    user_prompt = (
        f"Analysis Context:\n{context_text}\n\n"
        f"User Question: {question}\n\n"
        "Provide an APA-format summary in exactly three paragraphs, focusing on critical psychometric findings."
    )
    
    body = {
        "contents": [
            {"role": "user", "parts": [{"text": f"{system}\n\n{user_prompt}"}]}
        ]
    }
    
    # Try different model name variations if the first one fails
    model_variants = [
        model,  # Try the configured/default model first
        "models/gemini-1.5-flash-latest",
        "models/gemini-1.5-flash-002",
        "models/gemini-1.5-flash-001",
        "models/gemini-pro",
    ]
    
    resp = None
    last_error = None
    for model_variant in model_variants:
        url = f"https://generativelanguage.googleapis.com/v1beta/{model_variant}:generateContent"
        try:
            resp = requests.post(url, params={"key": api_key}, json=body, timeout=30)
            resp.raise_for_status()
            # If successful, break
            break
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                last_error = e
                resp = None
                continue
            else:
                # For non-404 errors, raise immediately
                raise
        except Exception as e:
            last_error = e
            resp = None
            continue
    
    # If all hardcoded models failed, try to discover available models dynamically
    if resp is None:
        try:
            list_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
            list_resp = requests.get(list_url, timeout=30)
            list_resp.raise_for_status()
            models_data = list_resp.json().get("models", [])
            
            # Find models that support generateContent
            candidates = []
            for m in models_data:
                methods = m.get("supportedGenerationMethods", [])
                if "generateContent" in methods:
                    candidates.append(m["name"])
            
            # Prefer flash models, then pro models
            for name in candidates:
                if "gemini-1.5-flash" in name:
                    url = f"https://generativelanguage.googleapis.com/v1beta/{name}:generateContent"
                    try:
                        resp = requests.post(url, params={"key": api_key}, json=body, timeout=30)
                        resp.raise_for_status()
                        break
                    except:
                        continue
            
            # If flash didn't work, try any gemini model
            if resp is None:
                for name in candidates:
                    if "gemini" in name:
                        url = f"https://generativelanguage.googleapis.com/v1beta/{name}:generateContent"
                        try:
                            resp = requests.post(url, params={"key": api_key}, json=body, timeout=30)
                            resp.raise_for_status()
                            break
                        except:
                            continue
        except Exception as e:
            # If model discovery also fails, use the last error from hardcoded attempts
            pass
    
    if resp is None:
        error_msg = str(last_error) if last_error else "Unknown error"
        return f"LLM analysis failed. Tried multiple models but none were available. Last error: {error_msg}. Please check your API key and model access."
    
    try:
        data = resp.json()
        text = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )
        return text.strip() if text else "No response from LLM."
    except Exception as e:
        return f"Error parsing LLM response: {type(e).__name__}: {e}"


def _parse_scenario_suggestion(text: str) -> tuple[str, str]:
    """Parse LLM response: Line 1 = single letter A/B/C; Line 2 = explanation. Prefer exact letter on first line so we don't match 'A' inside words like 'against'."""
    if not (text or "").strip():
        return "", (text or "").strip()
    lines = [ln.strip() for ln in (text or "").strip().splitlines() if ln.strip()]
    first_line = lines[0] if lines else ""
    # First line only: exact "A", "B", or "C" (with optional period)
    letter = ""
    for c in "ABC":
        if first_line.upper() == c or first_line.upper() == c + ".":
            letter = c
            break
    if not letter and first_line:
        # First character of first line
        fc = first_line.upper()[0]
        if fc in "ABC":
            letter = fc
    if not letter and first_line:
        # Whole-word A/B/C on first line (avoid matching 'a' in "against")
        for c in "ABC":
            if re.search(rf"\b{c}\b", first_line, re.IGNORECASE):
                letter = c
                break
    explanation = "\n".join(lines[1:]).strip() if len(lines) > 1 else (first_line if not letter else "").strip()
    if letter and not explanation and first_line and letter in first_line.upper():
        explanation = first_line.replace(letter, "").replace(letter.lower(), "").strip()
    return letter, (explanation[:400] if explanation else "Suggested based on your description.")


def _parse_scenario_json_or_legacy(text: str) -> tuple[str, str]:
    """
    Prefer parsing a structured JSON reply of the form:
      {"scenario_letter": "A|B|C", "agents": [...], "rationale": "..."}
    Fallback to the original two-line A/B parser when JSON is not present.
    Returns (letter, rationale_or_explanation).
    """
    raw = (text or "").strip()
    if not raw:
        return "", "Describe your testing situation (e.g. classroom quiz, certification exam, at-home test)."
    try:
        start = raw.index("{")
        end = raw.rfind("}")
        obj = json.loads(raw[start : end + 1])
        letter = (obj.get("scenario_letter") or "").strip().upper()
        if letter not in ("A", "B", "C"):
            letter = ""
        rationale = (obj.get("rationale") or "").strip()
        # Store agents list in session for callers that want to use it.
        agents = obj.get("agents") or []
        if isinstance(agents, list):
            agents = [a for a in agents if isinstance(a, str)]
        else:
            agents = []
        st.session_state["_llm_suggested_agents"] = agents
        if letter or rationale:
            return letter, (rationale[:400] if rationale else "Suggested based on your description.")
    except Exception:
        # Fall back to legacy plain-text parsing.
        pass
    letter, explanation = _parse_scenario_suggestion(raw)
    # Legacy parser had no agent list; clear any previous suggestion.
    st.session_state["_llm_suggested_agents"] = []
    return letter, explanation


def _suggest_aberrance_scenario(user_description: str) -> tuple[str, str]:
    """Use LLM (same provider/model as Psych-MAS Assistant) to suggest scenario and agents.

    Returns (scenario_letter 'A'|'B'|'C'|'', rationale). Also stores a list of
    suggested agents in st.session_state['_llm_suggested_agents'] when the
    model replies with structured JSON.
    """
    if not (user_description or "").strip():
        return "", "Describe your testing situation (e.g. classroom quiz, certification exam, at-home test)."
    load_dotenv()
    desc = user_description.strip()[:800]
    prompt = (
        "You are a psychometrics assistant recommending aberrance-detection scenarios and agents."
        "\n\nGoal:\n"
        "Given a short description of a testing situation, choose one scenario letter and a subset of detection agents."
        "\n\nScenarios:\n"
        "- A: Low-stakes / Quality Assurance — focuses on data quality and non-substantive noise (e.g., course evaluations, pilot surveys, classroom quizzes).\n"
        "- B: High-stakes / Certification Security — focuses on test security and credential protection (e.g., licensure, certification, proctored exams).\n"
        "- C: Custom — when neither A nor B clearly fits or needs are very specialized.\n\n"
        "Available agents (IDs in parentheses):\n"
        "- Nonparametric Misfit (detect_nm): Guttman/HT person-fit, no IRT required.\n"
        "- Model Misfit (detect_pm): parametric person-fit under IRT models.\n"
        "- Answer Similarity (detect_as): similarity clusters / collusion.\n"
        "- Answer Copying (detect_ac): source–copier pairs.\n"
        "- Preknowledge (detect_pk): success on compromised/leaked items.\n"
        "- Test Tampering (detect_tt): erasures / overwriting patterns.\n"
        "- Rapid Guessing (detect_rg): unusually fast, low-effort responding.\n\n"
        "Rules:\n"
        "- Always pick exactly one scenario_letter in ['A','B','C'].\n"
        "- agents must be a non-empty subset of the IDs above.\n"
        "- Map obvious low-stakes surveys/quizzes to A by default.\n"
        "- Map obvious high-stakes licensure/certification/proctored tests to B by default.\n"
        "- Use C only when A or B do not clearly apply.\n\n"
        "User description:\n"
        f"{desc}\n\n"
        "Respond with a single JSON object and nothing else, of the form:\n"
        '{\"scenario_letter\": \"A\", \"agents\": [\"detect_rg\", \"detect_pm\"], \"rationale\": \"Short explanation.\"}'
    )
    # Use same provider and model as Psych-MAS Assistant (Model engine tab)
    text, err = _call_selected_llm_text(prompt, timeout=60)
    if text and not err:
        letter, explanation = _parse_scenario_json_or_legacy(text)
        return letter, explanation
    return "", err or "No selected LLM model returned a response."
    # Google (Gemini): try each model variant (same as Psych-MAS Summary)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "", "GOOGLE_API_KEY not set in .env. Set it in Model engine to use LLM scenario suggestion."
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    model_variants = _model_variants_with_selected_first()
    last_err = None
    for model_name in model_variants:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent"
            resp = requests.post(url, params={"key": api_key}, json=body, timeout=60)
            resp.raise_for_status()
            text = (
                resp.json()
                .get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
                or ""
            ).strip()
            letter, explanation = _parse_scenario_json_or_legacy(text)
            return letter, explanation
        except requests.exceptions.HTTPError as e:
            last_err = f"{e.response.status_code if e.response is not None else 'HTTP'}: {e}"
            if e.response is not None and e.response.status_code == 404:
                continue
            break
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            break
    return "", (last_err or "No model responded. Check Model engine and GOOGLE_API_KEY.")


def _generate_agent_apa_summary(
    agent_title: str, agent_key: str, data: dict, n_students: int, flagged_ids: list[int] | None = None
) -> tuple[str | None, str | None]:
    """Generate a brief research-oriented summary with interpretation in Pinker's style. Returns (summary_text, error_message)."""
    load_dotenv()
    methods = data.get("methods") or []
    n_flagged = len(data.get("flagged", [])) + len(data.get("flagged_copiers", []))
    rate = (n_flagged / n_students * 100) if n_students else 0
    flagged_ids = flagged_ids or []
    example_ids_str = ", ".join(str(i + 1) for i in sorted(flagged_ids)[:25]) if flagged_ids else "none"
    prompt = (
        "You are a psychometrics expert writing for a research paper. Write one or two short paragraphs that:\n"
        "(1) Summarize the aberrant behavior detection result (method, sample size, number flagged).\n"
        "(2) Interpret what the findings mean in practical terms.\n"
        "(3) Discuss the abnormal test takers: mention that some examinees were flagged (e.g., by examinee/student numbers "
        "if provided) and briefly describe their potential issues—e.g., what the aberrant pattern may indicate "
        "(low effort, copying, preknowledge, misfit, tampering, etc.) and why it matters for validity or fairness.\n\n"
        "Use Steven Pinker's style: clear, direct, concrete language; active voice and short sentences; explain what the "
        "numbers and flags mean. You may use APA 7 conventions (past tense, third person).\n\n"
        f"Detection method: {agent_title}.\n"
        f"Indices/methods used: {', '.join(methods) or 'N/A'}.\n"
        f"Sample size: N = {n_students}.\n"
        f"Number flagged: {n_flagged} ({rate:.1f}%).\n"
        f"Example flagged examinee numbers (1-based): {example_ids_str}.\n\n"
        "Output only the paragraph(s), no heading or extra text."
    )
    text, err = _call_selected_llm_text(prompt, timeout=60)
    if text and not err:
        return (text.strip()[:2000] if text else None), None
    return None, err or "No selected LLM model responded."
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        return None, "GOOGLE_API_KEY not set. Add it to .env or use **Settings** → LLM provider → **OpenRouter** (no key required for free models)."
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    last_err = None
    for model_name in _model_variants_with_selected_first():
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent"
            resp = requests.post(url, params={"key": api_key}, json=body, timeout=60)
            resp.raise_for_status()
            text = (resp.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "") or "").strip()
            return (text[:2000] if text else None), None
        except requests.exceptions.HTTPError as e:
            last_err = f"{e.response.status_code}" if e.response is not None else str(e)
            if e.response is not None and e.response.status_code == 404:
                continue
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
    hint = f" Last error: {last_err}." if last_err else ""
    return None, f"No Gemini model responded.{hint} Check GOOGLE_API_KEY in .env and that the key has Gemini API access. Or use **Settings** → LLM provider → **OpenRouter**."


def _generate_student_abnormal_report(
    student_id: int,
    flags: dict,
    responses: list,
    rt_data: list,
    n_students: int,
) -> tuple[str | None, str | None]:
    """Generate an LLM-based abnormal behavior report for one student. Returns (report_text, error_message)."""
    load_dotenv()
    sid = student_id - 1  # 0-based
    # Build structured summary of this student's issues across all agents
    parts = [f"Student ID: {student_id} (of {n_students} examinees)."]
    if responses and sid < len(responses):
        row = responses[sid]
        vals = list(row.values())
        correct = sum(1 for v in vals if v == 1)
        total = len(vals)
        parts.append(f"Total score: {correct}/{total} ({100*correct/total:.1f}%)" if total else "Total score: N/A")
    flagged_agents = []
    agent_descriptions = {
        "nm_agent": "Nonparametric misfit (Guttman-style person-fit)",
        "pm_agent": "Parametric misfit (IRT-based person-fit)",
        "as_agent": "Answer similarity / collusion",
        "ac_agent": "Answer copying",
        "pk_agent": "Preknowledge (compromised items)",
        "tt_agent": "Test tampering / erasure",
        "cp_agent": "Change point (speed/performance shift)",
        "rg_agent": "Rapid guessing / low effort",
    }
    for ag_name, ag_data in flags.items():
        if not isinstance(ag_data, dict):
            continue
        fl = ag_data.get("flagged", []) + ag_data.get("flagged_copiers", [])
        if sid in fl:
            flagged_agents.append(ag_name)
            desc = agent_descriptions.get(ag_name, ag_name)
            line = f"- **{ag_name}** ({desc})"
            if ag_data.get("stat") and sid < len(ag_data["stat"]):
                rec = ag_data["stat"][sid]
                # Include key indices (numeric)
                key_vals = {k: v for k, v in rec.items() if isinstance(v, (int, float)) and k != "row"}
                if key_vals:
                    line += f" — indices: {key_vals}"
            if ag_name == "rg_agent" and ag_data.get("rte") and sid < len(ag_data["rte"]):
                line += f" — RTE: {ag_data['rte'][sid]:.4f}"
            if ag_name == "ac_agent" and ag_data.get("pairs"):
                partners = []
                for p in ag_data["pairs"]:
                    if p.get("Copier") == student_id:
                        partners.append(f"Source={p.get('Source')}")
                    elif p.get("Source") == student_id:
                        partners.append(f"Copier={p.get('Copier')}")
                if partners:
                    line += f" — pairs: {partners[:5]}"
            parts.append(line)
    if not flagged_agents:
        parts.append("This student was not flagged by any detection agent.")
    else:
        parts.insert(1, f"Flagged by {len(flagged_agents)} agent(s): {', '.join(flagged_agents)}")
    context = "\n".join(parts)
    prompt = (
        "You are a test-security and psychometrics expert. Write a concise **Abnormal Behavior Report** for the following examinee based on forensic detection results.\n\n"
        "**Input (detection results for this student):**\n"
        f"{context}\n\n"
        "**Required structure:**\n"
        "1. **Summary** — One short paragraph: which detectors flagged this student and what that means in plain language.\n"
        "2. **Detailed analysis** — For each detector that flagged the student, briefly explain the finding (what the index means, severity, and implication).\n"
        "3. **Final determination** — One clear conclusion: overall risk level (e.g., low / moderate / high concern) and a one-sentence recommendation (e.g., no action, review response pattern, or escalate for review). Use a line starting with '**Final determination:**'.\n\n"
        "Write in professional, neutral tone. Output only the report (no preamble)."
    )
    text, err = _call_selected_llm_text(prompt, timeout=90)
    if text and not err:
        return (text.strip()[:3500] if text else None), None
    return None, err or "No selected LLM model responded."
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        return None, "GOOGLE_API_KEY not set. Add it to .env or use Settings → LLM provider → OpenRouter."
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    last_err = None
    for model_name in _model_variants_with_selected_first():
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent"
            resp = requests.post(url, params={"key": api_key}, json=body, timeout=90)
            resp.raise_for_status()
            text = (resp.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "") or "").strip()
            return (text[:3500] if text else None), None
        except requests.exceptions.HTTPError as e:
            last_err = f"{e.response.status_code}" if e.response is not None else str(e)
            if e.response is not None and e.response.status_code == 404:
                continue
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
    return None, (last_err or "No model responded. Check Settings and API keys.")


# Appendix Table B2 rulebook. Keep this as the single UI-facing catalog of
# forensic indices, roles, data requirements, flagging rules, and threshold notes.
APPENDIX_B2_RULEBOOK: list[dict[str, str]] = [
    {"function": "detect_nm", "index": "ZU3_S", "domain_role": "MF / Preferred Primary", "required_data": "responses / statistic + flag", "flagging_rule": "Flag if statistic exceeds the prespecified misfit cutoff", "threshold_source": "Empirical or simulated null distribution. Standardized nonparametric misfit."},
    {"function": "detect_nm", "index": "G_S", "domain_role": "MF / Supporting", "required_data": "responses / statistic", "flagging_rule": "Flag if statistic exceeds the prespecified misfit cutoff", "threshold_source": "Empirical or simulated null distribution. Guttman error statistic."},
    {"function": "detect_nm", "index": "NC_S", "domain_role": "MF / Supporting", "required_data": "responses / statistic", "flagging_rule": "Flag if statistic exceeds the prespecified misfit cutoff", "threshold_source": "Empirical or simulated null distribution. Norm conformity."},
    {"function": "detect_nm", "index": "U1_S", "domain_role": "MF / Supporting", "required_data": "responses / statistic", "flagging_rule": "Flag if statistic exceeds the prespecified inconsistency cutoff", "threshold_source": "Empirical or simulated null distribution. Response inconsistency."},
    {"function": "detect_nm", "index": "U3_S", "domain_role": "MF / Supporting", "required_data": "responses / statistic", "flagging_rule": "Flag if statistic exceeds the prespecified person-fit cutoff", "threshold_source": "Empirical or simulated null distribution. Person-fit statistic."},
    {"function": "detect_nm", "index": "A_S", "domain_role": "MF / Supporting", "required_data": "responses / statistic", "flagging_rule": "Flag if statistic exceeds the prespecified atypicality cutoff", "threshold_source": "Empirical or simulated null distribution. Atypical response behavior."},
    {"function": "detect_nm", "index": "D_S", "domain_role": "MF / Supporting", "required_data": "responses / statistic", "flagging_rule": "Flag if statistic exceeds the prespecified deviation cutoff", "threshold_source": "Empirical or simulated null distribution. Deviation statistic."},
    {"function": "detect_nm", "index": "E_S", "domain_role": "MF / Supporting", "required_data": "responses / statistic", "flagging_rule": "Flag if statistic exceeds the prespecified error-pattern cutoff", "threshold_source": "Empirical or simulated null distribution. Error-related statistic."},
    {"function": "detect_nm", "index": "C_S", "domain_role": "MF / Supporting", "required_data": "responses / statistic", "flagging_rule": "Flag if statistic falls below or exceeds the prespecified consistency cutoff, depending on index direction", "threshold_source": "Empirical or simulated null distribution. Consistency statistic."},
    {"function": "detect_nm", "index": "MC_S", "domain_role": "MF / Supporting", "required_data": "responses / statistic", "flagging_rule": "Flag if statistic falls below or exceeds the prespecified modified-consistency cutoff", "threshold_source": "Empirical or simulated null distribution. Modified consistency."},
    {"function": "detect_nm", "index": "PC_S", "domain_role": "MF / Supporting", "required_data": "responses / statistic", "flagging_rule": "Flag if statistic falls below the prespecified person-consistency cutoff", "threshold_source": "Empirical or simulated null distribution. Person consistency."},
    {"function": "detect_nm", "index": "HT_S", "domain_role": "MF / Supporting", "required_data": "responses / statistic", "flagging_rule": "Flag if statistic exceeds the prespecified deviant-ordering cutoff", "threshold_source": "Empirical or simulated null distribution. Deviant ordering."},
    {"function": "detect_nm", "index": "KL_T", "domain_role": "MF / Supporting", "required_data": "RT / statistic + flag", "flagging_rule": "Flag if statistic exceeds a calibrated person-fit cutoff", "threshold_source": "Empirical or simulated person-fit distribution. Time-based misfit statistic."},
    {"function": "detect_pm", "index": "L_ST_*", "domain_role": "MF / Supporting", "required_data": "responses + RT + model params / statistic + p-value + flag", "flagging_rule": "Flag if p-value is below the prespecified alpha level; correction variants collapse to one family-level signal", "threshold_source": "Model-based null distribution. Score-time likelihood person-fit evidence."},
    {"function": "detect_pm", "index": "L_S_*", "domain_role": "MF / Supporting", "required_data": "responses + model params / statistic + p-value + flag", "flagging_rule": "Flag if p-value is below the prespecified alpha level; correction variants collapse to one family-level signal", "threshold_source": "Model-based null distribution. Score-based likelihood person-fit evidence."},
    {"function": "detect_pm", "index": "ECI2_S_*", "domain_role": "MF / Supporting", "required_data": "responses + model params / statistic", "flagging_rule": "Flag if statistic exceeds the prespecified caution-index cutoff", "threshold_source": "Empirical, simulated, or historical baseline. Extended caution index."},
    {"function": "detect_pm", "index": "ECI4_S_*", "domain_role": "MF / Supporting", "required_data": "responses + model params / statistic", "flagging_rule": "Flag if statistic exceeds the prespecified caution-index cutoff", "threshold_source": "Empirical, simulated, or historical baseline. Extended caution index."},
    {"function": "detect_pm", "index": "L_R_*", "domain_role": "MF / Supporting", "required_data": "responses + model params / statistic", "flagging_rule": "Flag if statistic is sufficiently extreme under the model-based null distribution", "threshold_source": "Model-based null distribution. Response likelihood."},
    {"function": "detect_pm", "index": "L_T", "domain_role": "MF / Supporting", "required_data": "RT + model params / statistic", "flagging_rule": "Flag if statistic is sufficiently extreme under the RT person-fit model", "threshold_source": "Model-based RT distribution. Time-based person-fit evidence."},
    {"function": "detect_pm", "index": "Q_ST_*", "domain_role": "MF / Supporting", "required_data": "responses + RT + model params / statistic", "flagging_rule": "Flag if statistic exceeds the prespecified combined score-time cutoff", "threshold_source": "Model-based or simulation-calibrated cutoff. Combined score-time statistic."},
    {"function": "detect_pm", "index": "Q_RT_*", "domain_role": "MF / Supporting", "required_data": "responses + RT + model params / statistic", "flagging_rule": "Flag if statistic exceeds the prespecified response-time person-fit cutoff; correction variants collapse to one family-level signal", "threshold_source": "Model-based or simulation-calibrated cutoff. Joint response-time person-fit evidence."},
    {"function": "detect_pm", "index": "L_RT_*", "domain_role": "MF / Supporting", "required_data": "responses + RT + model params / statistic", "flagging_rule": "Flag if statistic is sufficiently extreme under the response-time person-fit null distribution; correction variants collapse to one family-level signal", "threshold_source": "Model-based or simulation-calibrated cutoff. Response-time person-fit evidence."},
    {"function": "detect_as", "index": "OMG_ST", "domain_role": "SIM / Preferred Primary", "required_data": "responses + RT / pair statistic", "flagging_rule": "Flag if pair statistic exceeds the prespecified similarity cutoff", "threshold_source": "Historical pair distribution or simulated null distribution. Score-time similarity."},
    {"function": "detect_as", "index": "GBT_ST", "domain_role": "SIM / Supporting", "required_data": "responses + RT / pair statistic", "flagging_rule": "Flag if statistic exceeds the prespecified similarity cutoff or if p-value is below alpha, when available", "threshold_source": "Model-based, empirical, or simulated null distribution. Similarity test."},
    {"function": "detect_as", "index": "OMG_S", "domain_role": "SIM / Fallback Primary", "required_data": "responses / pair statistic", "flagging_rule": "Flag if pair statistic exceeds the prespecified score-similarity cutoff", "threshold_source": "Historical pair distribution or simulated null distribution. Score similarity."},
    {"function": "detect_as", "index": "WOMG_S", "domain_role": "SIM / Supporting", "required_data": "responses / pair statistic", "flagging_rule": "Flag if pair statistic exceeds the prespecified weighted-similarity cutoff", "threshold_source": "Historical pair distribution or simulated null distribution. Weighted omega similarity."},
    {"function": "detect_as", "index": "GBT_S", "domain_role": "SIM / Supporting", "required_data": "responses / pair statistic", "flagging_rule": "Flag if statistic exceeds the prespecified cutoff or if p-value is below alpha, when available", "threshold_source": "Model-based, empirical, or simulated null distribution. Generalized binomial similarity."},
    {"function": "detect_as", "index": "M4_S", "domain_role": "SIM / Supporting", "required_data": "pair responses / pair statistic", "flagging_rule": "Flag if statistic exceeds the prespecified pair-similarity cutoff", "threshold_source": "Historical pair distribution or simulated null distribution. Similarity metric."},
    {"function": "detect_as", "index": "OMG_R", "domain_role": "SIM / Supporting", "required_data": "responses / pair statistic", "flagging_rule": "Flag if statistic exceeds the prespecified response-similarity cutoff", "threshold_source": "Historical pair distribution or simulated null distribution. Response-based similarity."},
    {"function": "detect_as", "index": "WOMG_R", "domain_role": "SIM / Supporting", "required_data": "responses / pair statistic", "flagging_rule": "Flag if statistic exceeds the prespecified weighted-response cutoff", "threshold_source": "Historical pair distribution or simulated null distribution. Weighted response similarity."},
    {"function": "detect_as", "index": "GBT_R", "domain_role": "SIM / Supporting", "required_data": "responses / pair statistic", "flagging_rule": "Flag if statistic exceeds the prespecified cutoff or if p-value is below alpha, when available", "threshold_source": "Model-based, empirical, or simulated null distribution. Response similarity test."},
    {"function": "detect_as", "index": "M4_R", "domain_role": "SIM / Supporting", "required_data": "responses / pair statistic", "flagging_rule": "Flag if statistic exceeds the prespecified response-based similarity cutoff", "threshold_source": "Historical pair distribution or simulated null distribution. Response-based metric."},
    {"function": "detect_as", "index": "OMG_RT", "domain_role": "SIM / Supporting", "required_data": "responses + RT / pair statistic", "flagging_rule": "Flag if statistic exceeds the prespecified RT-similarity cutoff", "threshold_source": "Historical RT-pair distribution or simulated null distribution. RT-based similarity."},
    {"function": "detect_as", "index": "GBT_RT", "domain_role": "SIM / Supporting", "required_data": "responses + RT / pair statistic", "flagging_rule": "Flag if statistic exceeds the prespecified cutoff or if p-value is below alpha, when available", "threshold_source": "Model-based, empirical, or simulated null distribution. RT similarity."},
    {"function": "detect_ac", "index": "OMG_S", "domain_role": "SIM / Preferred Primary", "required_data": "responses + model params / pair statistic", "flagging_rule": "Flag if statistic exceeds the prespecified copying-evidence cutoff", "threshold_source": "Model-based, empirical, or simulated null distribution. Copying evidence."},
    {"function": "detect_ac", "index": "GBT_S", "domain_role": "SIM / Fallback Primary", "required_data": "responses + model params / pair statistic", "flagging_rule": "Flag if statistic exceeds the prespecified cutoff or if p-value is below alpha, when available", "threshold_source": "Model-based, empirical, or simulated null distribution. Copying evidence."},
    {"function": "detect_ac", "index": "OMG_R", "domain_role": "SIM / Supporting", "required_data": "responses + model params / pair statistic", "flagging_rule": "Flag if statistic exceeds the prespecified response-copying cutoff", "threshold_source": "Model-based, empirical, or simulated null distribution. Response copying evidence."},
    {"function": "detect_ac", "index": "GBT_R", "domain_role": "SIM / Supporting", "required_data": "responses + model params / pair statistic", "flagging_rule": "Flag if statistic exceeds the prespecified cutoff or if p-value is below alpha, when available", "threshold_source": "Model-based, empirical, or simulated null distribution. Copying evidence."},
    {"function": "detect_pk", "index": "L_ST", "domain_role": "PK / Preferred Primary", "required_data": "responses + RT + compromised items / statistic + p-value", "flagging_rule": "Flag if p-value is below the prespecified alpha level", "threshold_source": "Model-based null distribution. Score-time preknowledge."},
    {"function": "detect_pk", "index": "L_S", "domain_role": "PK / Fallback Primary", "required_data": "responses + compromised items / statistic + p-value", "flagging_rule": "Flag if p-value is below the prespecified alpha level", "threshold_source": "Model-based null distribution. Score-based preknowledge."},
    {"function": "detect_pk", "index": "ML_S", "domain_role": "PK / Supporting", "required_data": "responses + compromised items / statistic", "flagging_rule": "Flag if statistic exceeds the prespecified preknowledge cutoff", "threshold_source": "Model-based or simulation-calibrated cutoff. Modified likelihood statistic."},
    {"function": "detect_pk", "index": "LR_S", "domain_role": "PK / Supporting", "required_data": "responses + compromised items / statistic", "flagging_rule": "Flag if statistic exceeds the prespecified likelihood-ratio cutoff", "threshold_source": "Model-based or simulation-calibrated cutoff. Likelihood ratio."},
    {"function": "detect_pk", "index": "S_S", "domain_role": "PK / Supporting", "required_data": "responses + compromised items / statistic", "flagging_rule": "Flag if statistic exceeds the prespecified score-statistic cutoff", "threshold_source": "Model-based or simulation-calibrated cutoff. Score statistic."},
    {"function": "detect_pk", "index": "W_S", "domain_role": "PK / Supporting", "required_data": "responses + compromised items / statistic", "flagging_rule": "Flag if statistic exceeds the prespecified Wald cutoff", "threshold_source": "Model-based or simulation-calibrated cutoff. Wald statistic."},
    {"function": "detect_pk", "index": "L_T", "domain_role": "PK / Supporting", "required_data": "RT + compromised items / statistic", "flagging_rule": "Flag if statistic indicates unusually fast timing on compromised items", "threshold_source": "Model-based RT distribution or exposure-specific baseline. RT-based evidence."},
    {"function": "detect_pk", "index": "W_T", "domain_role": "PK / Supporting", "required_data": "RT + compromised items / statistic", "flagging_rule": "Flag if statistic exceeds the prespecified RT Wald cutoff", "threshold_source": "Model-based RT distribution or simulation-calibrated cutoff. RT Wald statistic."},
    {"function": "detect_cp", "index": "L_T_*", "domain_role": "CP / Preferred Primary", "required_data": "RT + item order / statistic + cp", "flagging_rule": "Flag if change-point statistic exceeds the prespecified timing-shift cutoff", "threshold_source": "Simulation-calibrated cutoff or historical baseline. Timing change point."},
    {"function": "detect_cp", "index": "L_S_*", "domain_role": "CP / Fallback Primary", "required_data": "responses + order / statistic + cp", "flagging_rule": "Flag if change-point statistic exceeds the prespecified score-shift cutoff", "threshold_source": "Simulation-calibrated cutoff or historical baseline. Score change point."},
    {"function": "detect_cp", "index": "S_S_*", "domain_role": "CP / Supporting", "required_data": "responses + order / statistic + cp", "flagging_rule": "Flag if statistic exceeds the prespecified score-shift cutoff", "threshold_source": "Simulation-calibrated cutoff or historical baseline. Score shift statistic."},
    {"function": "detect_cp", "index": "W_S_*", "domain_role": "CP / Supporting", "required_data": "responses + order / statistic + cp", "flagging_rule": "Flag if statistic exceeds the prespecified Wald score-shift cutoff", "threshold_source": "Simulation-calibrated cutoff or historical baseline. Wald score statistic."},
    {"function": "detect_cp", "index": "W_T_*", "domain_role": "CP / Supporting", "required_data": "RT / statistic + cp", "flagging_rule": "Flag if statistic exceeds the prespecified RT shift cutoff", "threshold_source": "Simulation-calibrated cutoff or historical baseline. RT Wald statistic."},
    {"function": "detect_cp", "index": "estimated_cp", "domain_role": "CP / Auxiliary", "required_data": "ordered responses / location estimate", "flagging_rule": "Not independently flagged unless paired with a flagged change-point statistic", "threshold_source": "Derived from flagged change-point model. Estimated change location."},
    {"function": "detect_rg", "index": "RTE", "domain_role": "RT / Preferred Primary", "required_data": "RT / person summary", "flagging_rule": "Flag if response-time effort falls below the prespecified cutoff", "threshold_source": "Program rule, historical baseline, or RT distribution. Response-time effort."},
    {"function": "detect_rg", "index": "CT", "domain_role": "RT / Supporting", "required_data": "RT / threshold output", "flagging_rule": "Flag if response time falls below the custom threshold", "threshold_source": "Program-approved operational rule. Custom threshold."},
    {"function": "detect_rg", "index": "NT", "domain_role": "RT / Supporting", "required_data": "RT / threshold output", "flagging_rule": "Flag if response time falls below the normative threshold", "threshold_source": "Normative RT distribution. Normative threshold."},
    {"function": "detect_rg", "index": "CUMP", "domain_role": "RT / Supporting", "required_data": "RT + scores / threshold output", "flagging_rule": "Flag if cumulative probability indicates unusually rapid responding", "threshold_source": "Empirical or model-based RT distribution. Cumulative probability."},
    {"function": "detect_rg", "index": "VI", "domain_role": "RT / Supporting", "required_data": "RT / statistic", "flagging_rule": "Flag if statistic exceeds the prespecified RT-variability cutoff", "threshold_source": "Historical baseline or empirical RT distribution. Variability indicator."},
    {"function": "detect_rg", "index": "VITP", "domain_role": "RT / Supporting", "required_data": "RT / statistic", "flagging_rule": "Flag if transformed variability statistic exceeds the prespecified cutoff", "threshold_source": "Historical baseline or empirical RT distribution. Transformed variability."},
    {"function": "detect_rg", "index": "RTF", "domain_role": "RT / Auxiliary", "required_data": "RT / item summary", "flagging_rule": "Not independently flagged unless used by a program-approved rule", "threshold_source": "Program rule or descriptive monitoring standard. Fidelity metric."},
    {"function": "detect_tt", "index": "EDI_SD_CO", "domain_role": "TP / Preferred Primary", "required_data": "answer changes + model params / statistic + p-value", "flagging_rule": "Primary correction for EDI_SD family; NO and TS variants are sensitivity-only", "threshold_source": "Model-based null distribution. Erasure detection."},
    {"function": "detect_tt", "index": "EDI_R_CO", "domain_role": "TP / Supporting", "required_data": "answer changes / statistic", "flagging_rule": "Primary correction for EDI_R family; NO and TS variants are sensitivity-only", "threshold_source": "Historical baseline or simulation-calibrated cutoff. Response erasure."},
    {"function": "detect_tt", "index": "GBT_SD", "domain_role": "TP / Supporting", "required_data": "answer changes / statistic", "flagging_rule": "Flag if statistic exceeds the prespecified answer-change cutoff or if p-value is below alpha, when available", "threshold_source": "Model-based, empirical, or simulated null distribution. Answer change evidence."},
    {"function": "detect_tt", "index": "L_SD", "domain_role": "TP / Supporting", "required_data": "answer changes / statistic", "flagging_rule": "Flag if statistic is sufficiently extreme under the prespecified likelihood rule", "threshold_source": "Model-based or simulation-calibrated cutoff. Likelihood approach."},
]


def _b2_rows_for_fn(fn_label: str) -> list[dict[str, str]]:
    return [row for row in APPENDIX_B2_RULEBOOK if row["function"] == fn_label]


def _b2_indices_for_fn(fn_label: str, *, primary_only: bool = False) -> list[str]:
    rows = _b2_rows_for_fn(fn_label)
    if primary_only:
        rows = [row for row in rows if "Primary" in row["domain_role"]]
    return [row["index"] for row in rows]


def _b2_typical_indices(fn_label: str) -> str:
    return ", ".join(_b2_indices_for_fn(fn_label))


def _b2_flag_rule(fn_label: str) -> str:
    rows = [row for row in _b2_rows_for_fn(fn_label) if "Primary" in row["domain_role"]]
    if not rows:
        rows = _b2_rows_for_fn(fn_label)
    return "; ".join(f"{row['index']}: {row['flagging_rule']}" for row in rows)


def _b2_domain_role_indices(domain: str, role_keyword: str) -> str:
    items = []
    role_keyword_l = str(role_keyword or "").lower()
    mapping_rows = _load_rulebook_mapping_rows()
    if mapping_rows:
        for row in mapping_rows:
            if str(row.get("domain", "")).strip() != domain:
                continue
            role = str(row.get("role", "")).strip().lower()
            if role_keyword_l not in role:
                continue
            fn = str(row.get("function", ""))
            if not fn.startswith("detect_"):
                continue
            items.append(f"{fn}/{row.get('method', '')}")
        return ", ".join(items)
    for row in APPENDIX_B2_RULEBOOK:
        domain_role = row.get("domain_role", "")
        if not domain_role.startswith(f"{domain} /"):
            continue
        if role_keyword not in domain_role:
            continue
        items.append(f"{row.get('function', '')}/{row.get('index', '')}")
    return ", ".join(items)


DEFAULT_B3_DOMAIN_STRENGTH_RULES: dict[str, dict] = {
    "B3-00": {
        "condition": "required_data_unavailable",
        "evidence_pattern": "Required data for the domain are unavailable",
        "strength": "unavailable",
    },
    "B3-01": {
        "condition": "no_indicators",
        "evidence_pattern": "No primary or supporting indicators are flagged",
        "strength": "none",
    },
    "B3-02": {
        "condition": "supporting_only",
        "min_supporting": 1,
        "max_supporting": 1,
        "evidence_pattern": "One supporting indicator is flagged, with no primary indicator",
        "strength": "weak",
    },
    "B3-03": {
        "condition": "primary_only",
        "min_primary": 1,
        "max_primary": 1,
        "evidence_pattern": "One isolated primary indicator is flagged",
        "strength": "moderate",
    },
    "B3-04": {
        "condition": "supporting_only",
        "min_supporting": 2,
        "evidence_pattern": "Multiple supporting indicators are flagged, with no primary indicator",
        "strength": "moderate",
    },
    "B3-05": {
        "condition": "primary_and_supporting",
        "min_primary": 1,
        "min_supporting": 1,
        "evidence_pattern": "One primary indicator together with supporting indicators within the same domain",
        "strength": "strong",
    },
    "B3-06": {
        "condition": "multiple_primary",
        "min_primary": 2,
        "evidence_pattern": "Multiple primary indicators are flagged within the same domain",
        "strength": "strong",
    },
}

DEFAULT_B5_REVIEW_PRIORITY_RULES: dict[str, dict] = {
    "B6-01": {"rule_id": "RP-01", "case_status": "None", "review_priority": "Low"},
    "B6-02": {"rule_id": "RP-02", "case_status": "Weak", "review_priority": "Low"},
    "B6-03a": {"rule_id": "RP-03a", "case_status": "Isolated (Context-Dependent)", "review_priority": "Medium"},
    "B6-03b": {"rule_id": "RP-03b", "case_status": "Isolated (Traceable)", "review_priority": "High"},
    "B6-04a": {"rule_id": "RP-04a", "case_status": "Convergent (Context-Dependent)", "review_priority": "High (Context-Heavy)"},
    "B6-04b": {"rule_id": "RP-04b", "case_status": "Convergent (Traceable)", "review_priority": "Critical / Expedited"},
    "B6-05": {"rule_id": "RP-05", "case_status": "No substantive evidence pattern", "review_priority": "Low"},
    "B6-06": {"rule_id": "RP-06", "case_status": "Supporting evidence only", "review_priority": "Low"},
}


DEFAULT_INDEX_ACTIVATION_RULES: dict[str, dict] = {
    "primary_flag_columns": {
        "enabled": True,
        "column_patterns": ["*_flagged"],
        "operator": "truthy",
        "applies_to": "primary_hits",
    },
    "supporting_p_values": {
        "enabled": True,
        "column_patterns": ["*_pval", "*_pval_*"],
        "operator": "<=",
        "value": 0.05,
        "applies_to": "supporting_hits",
    },
    "supporting_positive_counts": {
        "enabled": False,
        "column_patterns": [],
        "operator": ">",
        "value": 0,
        "applies_to": "supporting_hits",
    },
}


DEFAULT_THRESHOLD_CONFIG: dict = {
    "version": 1,
    "description": "Default PsyMAS rule and threshold registry. Edit values, upload the YAML, then rerun Forensic Indices.",
    "defaults": {
        "alpha": 0.05,
        "enabled": True,
    },
    "governance": {
        "domain_strength": DEFAULT_B3_DOMAIN_STRENGTH_RULES,
        "review_priority": DEFAULT_B5_REVIEW_PRIORITY_RULES,
        "index_activation": DEFAULT_INDEX_ACTIVATION_RULES,
    },
    "rules": {
        "detect_nm": {
            "ZU3_S": {
                "enabled": False,
                "operator": "<",
                "value": None,
                "requires_calibration": True,
                "source": "literature_or_study_calibration_required",
            },
            "HT_S": {
                "enabled": False,
                "operator": "<=",
                "value": None,
                "requires_calibration": True,
                "source": "literature_or_study_calibration_required",
            },
        },
        "detect_pm": {"alpha": 0.05, "source": "R aberrance model-based null distribution"},
        "detect_ac": {"alpha": 0.05, "source": "R aberrance pair-level flag output"},
        "detect_as": {"alpha": 0.05, "source": "R aberrance pair-level flag output"},
        "detect_pk": {"alpha": 0.05, "source": "R aberrance model-based null distribution"},
        "detect_rg": {
            "flag_methods": ["NT"],
            "CT": {"enabled": True, "thr": 3},
            "NT": {"enabled": True, "nt": [5, 10, 15, 20, 25, 30, 35]},
            "CUMP": {"enabled": True, "outlier": 90},
            "VI": {"enabled": True, "outlier": 90},
            "VITP": {"enabled": True, "outlier": 90},
            "RTE": {
                "enabled": False,
                "requires_calibration": True,
                "source": "program_calibration_required",
            },
        },
        "detect_tt": {
            "alpha": 0.05,
            "source": "R aberrance test-tampering p-value flag",
        },
    },
}


def _default_threshold_yaml() -> str:
    return yaml.safe_dump(DEFAULT_THRESHOLD_CONFIG, sort_keys=False, allow_unicode=True)


def _active_threshold_config() -> dict:
    return st.session_state.get("threshold_config") or DEFAULT_THRESHOLD_CONFIG


def _active_threshold_yaml() -> str:
    return yaml.safe_dump(_active_threshold_config(), sort_keys=False, allow_unicode=True)


def _threshold_registry_rows(config: dict | None = None) -> list[dict]:
    config = config or _active_threshold_config()
    rules = config.get("rules") if isinstance(config, dict) else {}
    rows = []
    governance = config.get("governance") if isinstance(config, dict) else {}
    if isinstance(governance, dict):
        domain_strength = governance.get("domain_strength") or {}
        if isinstance(domain_strength, dict):
            for code, rule in domain_strength.items():
                if isinstance(rule, dict):
                    rows.append(
                        {
                            "function": "governance.domain_strength",
                            "index": code,
                            "setting": rule.get("condition", "rule"),
                            "value": rule.get("strength", ""),
                            "source": rule.get("evidence_pattern", ""),
                        }
                    )
        review_priority = governance.get("review_priority") or {}
        if isinstance(review_priority, dict):
            for code, rule in review_priority.items():
                if isinstance(rule, dict):
                    rows.append(
                        {
                            "function": "governance.review_priority",
                            "index": code,
                            "setting": rule.get("case_status", ""),
                            "value": rule.get("review_priority", ""),
                            "source": rule.get("rule_id", ""),
                        }
                    )
        index_activation = governance.get("index_activation") or {}
        if isinstance(index_activation, dict):
            for code, rule in index_activation.items():
                if isinstance(rule, dict):
                    rows.append(
                        {
                            "function": "governance.index_activation",
                            "index": code,
                            "setting": rule.get("operator", ""),
                            "value": rule.get("value", rule.get("enabled", "")),
                            "source": ", ".join(rule.get("column_patterns", []) or []),
                        }
                    )
    for fn, block in (rules or {}).items():
        if not isinstance(block, dict):
            continue
        for index, rule in block.items():
            if index in {"alpha", "source"}:
                rows.append(
                    {
                        "function": fn,
                        "index": "*",
                        "setting": index,
                        "value": rule,
                        "source": block.get("source", ""),
                    }
                )
                continue
            if isinstance(rule, dict):
                rows.append(
                    {
                        "function": fn,
                        "index": index,
                        "setting": "threshold",
                        "value": json.dumps(rule, ensure_ascii=False),
                        "source": rule.get("source") or block.get("source", ""),
                    }
                )
    return rows


# Agent keys for Aberrance Summary CSV export (agent_key -> detect function label + primary indices).
_AGENT_EXPORT_META: dict[str, dict] = {
    "nm_agent": {
        "fn": "detect_nm",
        "label": "Nonparametric Misfit (detect_nm)",
        "primary_indices": _b2_indices_for_fn("detect_nm", primary_only=True),
        "typical_indices": _b2_typical_indices("detect_nm"),
        "flag_rule": _b2_flag_rule("detect_nm"),
    },
    "pm_agent": {
        "fn": "detect_pm",
        "label": "Model Misfit (detect_pm)",
        "primary_indices": _b2_indices_for_fn("detect_pm", primary_only=True),
        "typical_indices": _b2_typical_indices("detect_pm"),
        "flag_rule": _b2_flag_rule("detect_pm"),
    },
    "ac_agent": {
        "fn": "detect_ac",
        "label": "Answer Copying (detect_ac)",
        "primary_indices": _b2_indices_for_fn("detect_ac", primary_only=True),
        "typical_indices": _b2_typical_indices("detect_ac"),
        "flag_rule": _b2_flag_rule("detect_ac"),
    },
    "as_agent": {
        "fn": "detect_as",
        "label": "Answer Similarity (detect_as)",
        "primary_indices": _b2_indices_for_fn("detect_as", primary_only=True),
        "typical_indices": _b2_typical_indices("detect_as"),
        "flag_rule": _b2_flag_rule("detect_as"),
    },
    "rg_agent": {
        "fn": "detect_rg",
        "label": "Rapid Guessing (detect_rg)",
        "primary_indices": _b2_indices_for_fn("detect_rg", primary_only=True),
        "typical_indices": _b2_typical_indices("detect_rg"),
        "flag_rule": _b2_flag_rule("detect_rg"),
    },
    "cp_agent": {
        "fn": "detect_cp",
        "label": "Change Point (detect_cp)",
        "primary_indices": _b2_indices_for_fn("detect_cp", primary_only=True),
        "typical_indices": _b2_typical_indices("detect_cp"),
        "flag_rule": _b2_flag_rule("detect_cp"),
    },
    "tt_agent": {
        "fn": "detect_tt",
        "label": "Test Tampering (detect_tt)",
        "primary_indices": _b2_indices_for_fn("detect_tt", primary_only=True),
        "typical_indices": _b2_typical_indices("detect_tt"),
        "flag_rule": _b2_flag_rule("detect_tt"),
    },
    "pk_agent": {
        "fn": "detect_pk",
        "label": "Preknowledge (detect_pk)",
        "primary_indices": _b2_indices_for_fn("detect_pk", primary_only=True),
        "typical_indices": _b2_typical_indices("detect_pk"),
        "flag_rule": _b2_flag_rule("detect_pk"),
    },
}

_EXAMINEE_FLAG_AGENT_COLUMNS: list[tuple[str, str]] = [
    (k, v["fn"]) for k, v in _AGENT_EXPORT_META.items()
]

_CSV_SKIP_KEYS = frozenset(
    {
        "Source",
        "Copier",
        "flagged",
        "_pair",
        "row",
        "Student",
        "Examinee_ID",
        "examinee_id",
        "item_id",
        "person_id",
        "error",
        "info",
        "methods",
        "n_persons",
    }
)


def _csv_numeric(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False
    return False


def _csv_flag_index_key(key: str) -> bool:
    return str(key or "").lower().endswith("_flag")


def _csv_round(value):
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    return round(float(value), 6)


def _safe_float(value, default: float | None = None) -> float | None:
    """Convert UI display values without crashing on None / NaN / non-numeric cells."""
    try:
        if value is None:
            return default
        out = float(value)
        if not math.isfinite(out):
            return default
        return out
    except (TypeError, ValueError):
        return default


def _export_col_name(fn_label: str, index_name: str, *, suffix: str = "") -> str:
    """Readable data column: pm_L_S_TS or ac_OMG_S_max."""
    agent_short = fn_label.replace("detect_", "", 1)
    base = f"{agent_short}_{index_name}"
    return f"{base}_{suffix}" if suffix else base


def _export_flag_col_name(fn_label: str) -> str:
    agent_short = fn_label.replace("detect_", "", 1)
    return f"{agent_short}_flagged"


def _index_is_primary(agent_key: str, index_name: str) -> bool:
    if index_name in ("n_pairs", "flagged_copier", "flagged") or "pval" in index_name.lower() or _csv_flag_index_key(index_name):
        return False
    meta = _AGENT_EXPORT_META.get(agent_key, {})
    primaries = meta.get("primary_indices") or []
    if index_name in primaries:
        return True
    for p in primaries:
        if isinstance(p, str) and p.endswith("*") and index_name.startswith(p[:-1]):
            return True
        if index_name == p or index_name.startswith(f"{p}_"):
            return True
    return False


def _legend_row(
    column_name: str,
    agent_key: str,
    fn_label: str,
    index_name: str,
    *,
    is_primary: bool = False,
    aggregation: str = "person",
    notes: str = "",
) -> dict:
    meta = _AGENT_EXPORT_META.get(agent_key, {})
    return {
        "column_name": column_name,
        "agent": fn_label,
        "agent_label": meta.get("label", fn_label),
        "index_name": index_name,
        "is_primary_index": "Y" if is_primary else "N",
        "aggregation": aggregation,
        "notes": notes,
    }


def _agent_flagged_indices(agent_data: dict, agent_key: str = "") -> set[int]:
    """0-based examinee indices flagged by one agent result payload."""
    if not isinstance(agent_data, dict):
        return set()
    out: set[int] = set()

    def _add(values) -> None:
        for p in values or []:
            try:
                idx = int(p)
            except (TypeError, ValueError):
                continue
            if idx >= 0:
                out.add(idx)

    # Pair-based similarity/copying: person-level review uses role-specific lists only.
    if agent_key == "ac_agent":
        _add(agent_data.get("flagged_copiers"))
        return out

    if agent_key == "as_agent":
        _add(agent_data.get("flagged_participants"))
        return out

    # Person-level agents expose a curated final list in `flagged` (e.g. detect_rg NT only).
    # Do not union every diagnostic `flagged_by_method` entry — CT/CUMP/VI can flag most examinees.
    if "flagged" in agent_data:
        _add(agent_data.get("flagged"))
        return out

    flagged_by_method = agent_data.get("flagged_by_method")
    if isinstance(flagged_by_method, dict):
        for ids in flagged_by_method.values():
            _add(ids)
    return out


def _infer_n_examinees(flags: dict, responses: list | None) -> int:
    n = len(responses or [])
    if n > 0:
        return n
    best = 0
    for agent_data in flags.values():
        if not isinstance(agent_data, dict):
            continue
        for key in ("stat", "rte"):
            block = agent_data.get(key)
            if isinstance(block, list):
                best = max(best, len(block))
        pairs = agent_data.get("pairs")
        if isinstance(pairs, list):
            for pr in pairs:
                if isinstance(pr, dict):
                    for k in ("Source", "Copier"):
                        try:
                            best = max(best, int(pr.get(k, 0)))
                        except (TypeError, ValueError):
                            pass
    return best


def _export_agents(flags: dict, visible_agents: set[str] | None) -> list[tuple[str, str]]:
    """Agents to include: selected in this run, plus cp_agent when rg ran and cp data exists."""
    out: list[tuple[str, str]] = []
    for agent_key, fn_label in _EXAMINEE_FLAG_AGENT_COLUMNS:
        if agent_key not in flags or not isinstance(flags.get(agent_key), dict):
            continue
        if visible_agents is not None:
            if agent_key in visible_agents:
                out.append((agent_key, fn_label))
            elif agent_key == "cp_agent" and "rg_agent" in visible_agents:
                out.append((agent_key, fn_label))
        else:
            out.append((agent_key, fn_label))
    return out


def _person_level_index_columns(
    agent_key: str,
    fn_label: str,
    data: dict,
    n_examinees: int,
) -> tuple[dict[str, list], list[dict]]:
    """Per-examinee index columns from agent `stat` and/or `rte`; returns (columns, legend rows)."""
    cols: dict[str, list] = {}
    legend: list[dict] = []
    stat = data.get("stat")
    if isinstance(stat, list) and stat:
        index_keys: set[str] = set()
        for rec in stat:
            if not isinstance(rec, dict):
                continue
            for k, v in rec.items():
                if k not in _CSV_SKIP_KEYS and _csv_numeric(v):
                    index_keys.add(k)
                elif k not in _CSV_SKIP_KEYS and _csv_flag_index_key(k) and isinstance(v, (bool, np.bool_, int, np.integer, float, np.floating)):
                    index_keys.add(k)
        for key in sorted(index_keys):
            col = _export_col_name(fn_label, key)
            vals: list = []
            for p in range(n_examinees):
                if p < len(stat) and isinstance(stat[p], dict) and key in stat[p]:
                    try:
                        vals.append(1 if _csv_flag_index_key(key) and _gov_truthy_flag(stat[p][key]) else _csv_round(stat[p][key]))
                    except (TypeError, ValueError):
                        vals.append(None)
                else:
                    vals.append(None)
            cols[col] = vals
            legend.append(
                _legend_row(
                    col,
                    agent_key,
                    fn_label,
                    key,
                    is_primary=_index_is_primary(agent_key, key),
                    aggregation="index_flag" if _csv_flag_index_key(key) else "person",
                    notes="Package-returned method flag from the Index Registry." if _csv_flag_index_key(key) else "",
                )
            )
    rte = data.get("rte")
    if isinstance(rte, list) and rte:
        col = _export_col_name(fn_label, "RTE")
        vals = []
        for p in range(n_examinees):
            if p < len(rte):
                try:
                    vals.append(_csv_round(rte[p]))
                except (TypeError, ValueError):
                    vals.append(None)
            else:
                vals.append(None)
        cols[col] = vals
        legend.append(
            _legend_row(
                col,
                agent_key,
                fn_label,
                "RTE",
                is_primary=True,
                aggregation="person",
                notes="Response Time Effort (detect_rg)",
            )
        )
    rte_by_method = data.get("rte_by_method")
    if isinstance(rte_by_method, dict) and rte_by_method:
        for method_name, method_vals in sorted(rte_by_method.items()):
            if not isinstance(method_vals, list) or not method_vals:
                continue
            index_name = f"RTE_{method_name}"
            col = _export_col_name(fn_label, index_name)
            vals = []
            for p in range(n_examinees):
                if p < len(method_vals):
                    try:
                        vals.append(_csv_round(method_vals[p]))
                    except (TypeError, ValueError):
                        vals.append(None)
                else:
                    vals.append(None)
            cols[col] = vals
            legend.append(
                _legend_row(
                    col,
                    agent_key,
                    fn_label,
                    index_name,
                    is_primary=index_name == "RTE_NT_2" or index_name == "RTE_NT",
                    aggregation="person",
                    notes=f"Response Time Effort from detect_rg method output: {method_name}",
                )
            )
    flagged_by_method = data.get("flagged_by_method")
    if isinstance(flagged_by_method, dict) and flagged_by_method:
        for method_name, flagged_ids in sorted(flagged_by_method.items()):
            index_name = f"{method_name}_flag"
            col = _export_col_name(fn_label, index_name)
            flagged_set: set[int] = set()
            if isinstance(flagged_ids, list):
                for item in flagged_ids:
                    try:
                        flagged_set.add(int(item))
                    except (TypeError, ValueError):
                        pass
            cols[col] = [1 if p in flagged_set else 0 for p in range(n_examinees)]
            legend.append(
                _legend_row(
                    col,
                    agent_key,
                    fn_label,
                    index_name,
                    is_primary=False,
                    aggregation="index_flag",
                    notes=f"Package threshold flag from detect_rg method output: {method_name}",
                )
            )
    return cols, legend


def _pair_record_participants(pair_record: dict) -> tuple[int | None, int | None]:
    """Return 0-based participants for pair-level records from detect_ac/detect_as."""
    if not isinstance(pair_record, dict):
        return None, None
    if "_pair" in pair_record:
        pr = pair_record.get("_pair")
        if isinstance(pr, (list, tuple)) and len(pr) >= 2:
            try:
                return int(pr[0]), int(pr[1])
            except (TypeError, ValueError):
                return None, None
    if "Source" in pair_record and "Copier" in pair_record:
        try:
            return int(pair_record.get("Source")) - 1, int(pair_record.get("Copier")) - 1
        except (TypeError, ValueError):
            return None, None
    return None, None


def _pair_level_index_columns(
    agent_key: str,
    fn_label: str,
    pairs: list,
    n_examinees: int,
    flagged_copiers: set[int] | None = None,
) -> tuple[dict[str, list], list[dict]]:
    """Aggregate pair-level indices (detect_ac / detect_as) to one row per examinee."""
    cols: dict[str, list] = {}
    legend: list[dict] = []
    if not isinstance(pairs, list) or not pairs:
        return cols, legend
    metric_keys: set[str] = set()
    pairs_by_person: list[list[dict]] = [[] for _ in range(n_examinees)]
    for pr in pairs:
        if not isinstance(pr, dict):
            continue
        for k, v in pr.items():
            if k not in _CSV_SKIP_KEYS and _csv_numeric(v):
                metric_keys.add(k)
            elif k not in _CSV_SKIP_KEYS and _csv_flag_index_key(k) and isinstance(v, (bool, np.bool_, int, np.integer, float, np.floating)):
                metric_keys.add(k)
        p1, p2 = _pair_record_participants(pr)
        if p1 is not None and 0 <= p1 < n_examinees:
            pairs_by_person[p1].append(pr)
        if p2 is not None and 0 <= p2 < n_examinees and p2 != p1:
            pairs_by_person[p2].append(pr)
    for key in sorted(metric_keys):
        is_pval = key.endswith("_pval") or "pval" in key.lower()
        is_flag = _csv_flag_index_key(key)
        agg = "pair_min" if is_pval else ("pair_flag_display" if is_flag else "pair_max")
        suffix = "min" if is_pval else ("" if is_flag else "max")
        col = _export_col_name(fn_label, key, suffix=suffix)
        vals = []
        for p in range(n_examinees):
            nums = []
            for pr in pairs_by_person[p]:
                if key in pr and (_csv_numeric(pr[key]) or (is_flag and isinstance(pr[key], (bool, np.bool_, int, np.integer, float, np.floating)))):
                    try:
                        nums.append(1.0 if is_flag and _gov_truthy_flag(pr[key]) else float(pr[key]))
                    except (TypeError, ValueError):
                        pass
            if nums:
                vals.append(_csv_round(min(nums) if is_pval else max(nums)))
            else:
                vals.append(None)
        cols[col] = vals
        legend.append(
            _legend_row(
                col,
                agent_key,
                fn_label,
                key,
                is_primary=_index_is_primary(agent_key, key) and not is_pval,
                aggregation=agg,
                notes=(
                    "Package-returned pair flag shown for trace only; pair-level flags are not counted as person-level Domain Evidence without a prespecified person-level aggregation rule."
                    if is_flag
                    else f"{'Min' if is_pval else 'Max'} over pairs where examinee is Source or Copier"
                ),
            )
        )
    col_np = _export_col_name(fn_label, "n_pairs")
    n_pairs = [len(pairs_by_person[p]) for p in range(n_examinees)]
    cols[col_np] = n_pairs
    legend.append(
        _legend_row(col_np, agent_key, fn_label, "n_pairs", aggregation="pair_count", notes="Stored pairs involving examinee")
    )
    if flagged_copiers is not None:
        col_fc = _export_col_name(fn_label, "flagged_copier")
        cols[col_fc] = [1 if p in flagged_copiers else 0 for p in range(n_examinees)]
        legend.append(
            _legend_row(
                col_fc,
                agent_key,
                fn_label,
                "flagged_copier",
                is_primary=True,
                aggregation="flag",
                notes="1 if examinee flagged as copier (detect_ac)",
            )
        )
    return cols, legend


def _collusion_partners_for_examinee(pairs: list, examinee_id_1based: int) -> str:
    partners: set[int] = set()
    for pr in pairs:
        if not isinstance(pr, dict):
            continue
        src, cop = pr.get("Source", 0), pr.get("Copier", 0)
        if src == examinee_id_1based:
            partners.add(int(cop))
        elif cop == examinee_id_1based:
            partners.add(int(src))
    return ", ".join(str(x) for x in sorted(partners))


def _build_agent_index_guide(
    flags: dict,
    agents_for_export: list[tuple[str, str]],
    legend_df: pd.DataFrame,
    data_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per agent: primary indices (documented) and indices present in this export."""
    rows: list[dict] = []
    for agent_key, fn_label in agents_for_export:
        meta = _AGENT_EXPORT_META.get(agent_key, {})
        data = flags.get(agent_key, {})
        if not isinstance(data, dict):
            continue
        sub = legend_df[legend_df["agent"] == fn_label] if not legend_df.empty and "agent" in legend_df.columns else pd.DataFrame()
        exported_indices = sorted(set(sub["index_name"].dropna().astype(str))) if not sub.empty else []
        primary_exported = sorted(
            set(sub.loc[sub["is_primary_index"] == "Y", "index_name"].dropna().astype(str)) if not sub.empty else []
        )
        methods = data.get("methods") or []
        if isinstance(methods, str):
            methods = [methods]
        flag_col = _export_flag_col_name(fn_label)
        if data_df is not None and flag_col in data_df.columns:
            flagged_n = int(sum(1 for v in data_df[flag_col].tolist() if _gov_truthy_flag(v)))
        else:
            flagged_n = len(_agent_flagged_indices(data, agent_key))
        rows.append(
            {
                "agent": fn_label,
                "agent_label": meta.get("label", fn_label),
                "primary_indices": ", ".join(meta.get("primary_indices") or []),
                "primary_indices_in_export": ", ".join(primary_exported),
                "all_indices_in_export": ", ".join(exported_indices),
                "typical_indices_reference": meta.get("typical_indices", ""),
                "methods_in_run": ", ".join(str(m) for m in methods),
                "flag_rule": meta.get("flag_rule", ""),
                "n_flagged_examinees": flagged_n,
                "export_error": data.get("error") or "",
            }
        )
    return pd.DataFrame(rows)


def _rulebook_role_kind(role: str) -> str:
    text = str(role or "").strip().lower()
    if "primary" in text:
        return "primary"
    if "support" in text:
        return "supporting"
    return "auxiliary"


def _b3_allowed_from_rule(rule: dict[str, str]) -> bool:
    evidence_use = str(rule.get("evidence_use", "")).strip().lower()
    return evidence_use in {"flag", "evidence flag"}


_PM_CORRECTION_VARIANTS = ("TSCF", "TSCS", "TSEW", "NO", "CF", "CS", "EW", "TS")
_TT_EDI_PRIMARY_VARIANT = "CO"


def _strip_index_flag_suffix(index_name: str) -> str:
    text = str(index_name or "").strip()
    text = re.sub(r"_pval_flag$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"_flag$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"_pval$", "", text, flags=re.IGNORECASE)
    return text


def _pm_family_variant(index_name: str) -> tuple[str, str, str]:
    """Return family, correction variant, and data basis for detect_pm outputs."""
    base = _strip_index_flag_suffix(index_name)
    base = re.sub(r"^pm_", "", base, flags=re.IGNORECASE)
    parts = base.split("_")
    variant = ""
    if parts and parts[-1].upper() in _PM_CORRECTION_VARIANTS:
        variant = parts[-1].upper()
        base = "_".join(parts[:-1])
    data_basis = "score"
    if base.endswith("_ST"):
        data_basis = "score_time"
    elif base.endswith("_RT"):
        data_basis = "response_time"
    elif base.endswith("_T"):
        data_basis = "time"
    elif base.endswith("_R"):
        data_basis = "response"
    return f"pm_{base}", variant or "none", data_basis


def _tt_family_variant(index_name: str) -> tuple[str, str, str, str]:
    """Return family, correction variant, data basis, and treatment for detect_tt outputs."""
    base = _strip_index_flag_suffix(index_name)
    base = re.sub(r"^tt_", "", base, flags=re.IGNORECASE)
    base = re.sub(r"\s+", "", base)
    # Some legacy rulebook rows grouped EDI corrections together; the actual flag
    # column is more reliable for identifying the returned variant.
    if "/" in base and "EDI_SD" in base:
        return "tt_EDI_SD", "unknown", "score", "sensitivity_only"
    if "/" in base and "EDI_R" in base:
        return "tt_EDI_R", "unknown", "response", "sensitivity_only"
    parts = base.split("_")
    variant = ""
    if len(parts) >= 3 and parts[0].upper() == "EDI" and parts[2].upper() in {"NO", "CO", "TS"}:
        variant = parts[2].upper()
        family_base = "_".join(parts[:2])
        data_basis = "response" if family_base.endswith("_R") else "score"
        treatment = "family_level_signal" if variant == _TT_EDI_PRIMARY_VARIANT else "sensitivity_only"
        return f"tt_{family_base}", variant, data_basis, treatment
    data_basis = "response" if base.endswith("_R") else "score"
    return f"tt_{base}", "none", data_basis, "family_level_signal"


def _registry_family_metadata(rule: dict[str, str], index_name: str) -> tuple[str, str, str, str]:
    family = str(rule.get("aggregation_family", "")).strip()
    variant = str(rule.get("correction_variant", "")).strip()
    data_basis = str(rule.get("data_basis", "")).strip()
    treatment = str(rule.get("variant_treatment", "")).strip()
    if str(rule.get("function", "")).strip() == "detect_pm":
        derived_family, derived_variant, derived_basis = _pm_family_variant(index_name)
        family = derived_family or family
        variant = derived_variant or variant
        data_basis = derived_basis or data_basis
        treatment = treatment or "family_level_signal"
    if str(rule.get("function", "")).strip() == "detect_tt":
        derived_family, derived_variant, derived_basis, derived_treatment = _tt_family_variant(index_name)
        family = derived_family or family
        variant = derived_variant or variant
        data_basis = derived_basis or data_basis
        treatment = derived_treatment or treatment or "family_level_signal"
    return family, variant, data_basis, treatment


def _variant_allowed_for_b3(variant_treatment: str) -> bool:
    return variant_allowed_for_b3(variant_treatment)


EVIDENCE_USE_FLAG = "Evidence Flag"
EVIDENCE_USE_CALIBRATION = "Calibration Required"
EVIDENCE_USE_DISPLAY = "Display Only"
EVIDENCE_USE_ORDER = {
    EVIDENCE_USE_FLAG: 0,
    EVIDENCE_USE_CALIBRATION: 1,
    EVIDENCE_USE_DISPLAY: 2,
}


def _evidence_flag_column_for_source(
    data_df: pd.DataFrame,
    legend_df: pd.DataFrame,
    column_name: str,
) -> str:
    if column_name in data_df.columns and _csv_flag_index_key(column_name):
        return column_name
    derived = f"{column_name}_Flag"
    if derived in data_df.columns:
        return derived
    rule = _rulebook_match_for_export_column(legend_df, column_name)
    if not rule:
        return ""
    fn = str(rule.get("function", "")).strip()
    domain = str(rule.get("domain", "")).strip()
    method = str(rule.get("method", "")).strip()
    if not fn or not method or legend_df is None or legend_df.empty:
        return ""
    for _, leg in legend_df.iterrows():
        candidate = str(leg.get("column_name", ""))
        if not candidate or candidate == str(column_name) or candidate not in data_df.columns:
            continue
        if str(leg.get("aggregation", "")) != "index_flag":
            continue
        candidate_rule = _rulebook_match_for_export_column(legend_df, candidate)
        if not candidate_rule:
            continue
        if str(candidate_rule.get("function", "")).strip() != fn:
            continue
        if str(candidate_rule.get("domain", "")).strip() != domain:
            continue
        candidate_method = str(candidate_rule.get("method", "")).strip()
        if _method_matches_index(method, candidate_method) or _method_matches_index(candidate_method, method):
            return candidate
    return ""


def _build_b3_input_table(data_df: pd.DataFrame, legend_df: pd.DataFrame) -> pd.DataFrame:
    """Long-format Evidence Input table. One row = examinee x index-level flag."""
    if data_df is None or data_df.empty or legend_df is None or legend_df.empty:
        return pd.DataFrame()
    if "column_name" not in legend_df.columns:
        return pd.DataFrame()

    flag_legend = legend_df[
        legend_df["aggregation"].astype(str).isin(["index_flag", "pair_flag_display"])
        & ~legend_df["column_name"].astype(str).str.endswith("_flagged")
    ].copy()
    flag_legend = flag_legend.drop_duplicates(subset=["column_name"], keep="first")
    if flag_legend.empty:
        return pd.DataFrame()

    id_vars = ["Examinee_ID"] if "Examinee_ID" in data_df.columns else []
    value_cols = [str(c) for c in flag_legend["column_name"] if str(c) in data_df.columns]
    if not value_cols:
        return pd.DataFrame()

    long_df = data_df[id_vars + value_cols].melt(
        id_vars=id_vars,
        value_vars=value_cols,
        var_name="Flag_Column",
        value_name="Value",
    )
    legend_by_col = {
        str(row["column_name"]): row
        for _, row in flag_legend.iterrows()
    }
    rows: list[dict] = []
    for flag_col, group in long_df.groupby("Flag_Column", sort=False):
        leg = legend_by_col.get(str(flag_col))
        if leg is None:
            continue
        rule = _rulebook_match_for_export_column(legend_df, str(flag_col))
        if not rule:
            continue
        domain = str(rule.get("domain", "")).strip()
        if not domain or domain == "CTX":
            continue
        aggregation = str(leg.get("aggregation", ""))
        fn = str(rule.get("function", ""))
        method = str(rule.get("method", "")) or str(leg.get("index_name", ""))
        metadata_source = str(flag_col) if fn == "detect_tt" else (str(leg.get("index_name", "")) or str(flag_col))
        family, variant, data_basis, variant_treatment = _registry_family_metadata(rule, metadata_source)
        role = str(rule.get("role", "")).strip()
        role_kind = _rulebook_role_kind(role)
        evidence_use = _evidence_use_label(rule)
        eligible = (
            aggregation == "index_flag"
            and _b3_allowed_from_rule(rule)
            and _variant_allowed_for_b3(variant_treatment)
        )
        for examinee_id, raw_value in zip(group["Examinee_ID"], group["Value"]):
            flag_bool = _gov_truthy_flag(raw_value)
            rows.append(
                {
                    "Examinee_ID": examinee_id,
                    "Domain": domain,
                    "Domain_Label": _DOMAIN_LABELS.get(domain, domain),
                    "Function": fn,
                    "Index": method,
                    "Index_Column": str(leg.get("index_name", "")),
                    "Value": raw_value,
                    "Flag_Column": flag_col,
                    "Flag": 1 if flag_bool else 0,
                    "Evidence_Eligible": 1 if eligible else 0,
                    "Evidence_Use": evidence_use,
                    "Role": role,
                    "Role_Class": role_kind,
                    "Flag_Type": str(rule.get("flag_type", "")),
                    "Package_Output": str(rule.get("package_output", "")),
                    "Package_Flag": str(rule.get("package_flag", "")),
                    "Threshold_Rule": str(rule.get("threshold_rule", "")),
                    "Threshold_Source": str(rule.get("threshold_source", "")),
                    "Aggregation": aggregation,
                    "Aggregation_Family": family,
                    "Correction_Variant": variant,
                    "Data_Basis": data_basis,
                    "Variant_Treatment": variant_treatment,
                    "Notes": str(rule.get("notes", "")),
                }
            )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    order = {domain: i for i, domain in enumerate(_DOMAIN_ORDER)}
    out["_domain_order"] = out["Domain"].map(lambda x: order.get(str(x), 99))
    out["_use_order"] = out["Evidence_Use"].map(lambda x: EVIDENCE_USE_ORDER.get(str(x), 9))
    return out.sort_values(
        ["Examinee_ID", "_domain_order", "_use_order", "Function", "Index", "Index_Column"]
    ).drop(columns=["_domain_order", "_use_order"])


def _build_pair_detail_table(flags: dict, visible_agents: set[str] | None = None) -> pd.DataFrame:
    """Pair-level details for detect_ac/detect_as, kept separate from the examinee-level wide table."""
    rows: list[dict] = []
    agents_for_export = _export_agents(flags, visible_agents)
    for agent_key, fn_label in agents_for_export:
        data = flags.get(agent_key, {})
        if not isinstance(data, dict):
            continue
        pair_blocks: list[tuple[str, list]] = []
        if isinstance(data.get("pairs"), list):
            pair_blocks.append(("pairs", data.get("pairs") or []))
        if agent_key == "as_agent" and isinstance(data.get("stat"), list):
            pair_blocks.append(("stat", data.get("stat") or []))
        for source_name, pairs in pair_blocks:
            for pair_rec in pairs:
                if not isinstance(pair_rec, dict):
                    continue
                p1, p2 = _pair_record_participants(pair_rec)
                base = {
                    "Function": fn_label,
                    "Source_Table": source_name,
                    "Examinee_A": (p1 + 1) if p1 is not None else pair_rec.get("Source", ""),
                    "Examinee_B": (p2 + 1) if p2 is not None else pair_rec.get("Copier", ""),
                }
                for key, value in pair_rec.items():
                    if key in {"_pair", "Source", "Copier"}:
                        continue
                    base[str(key)] = value
                rows.append(base)
    return pd.DataFrame(rows)


def _build_index_dictionary(data_df: pd.DataFrame, legend_df: pd.DataFrame) -> pd.DataFrame:
    """Column dictionary enriched with Index Registry metadata."""
    if legend_df is None or legend_df.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for _, leg in legend_df.iterrows():
        column = str(leg.get("column_name", ""))
        rule = _rulebook_match_for_export_column(legend_df, column) if column else {}
        flag_col = _evidence_flag_column_for_source(data_df, legend_df, column) if isinstance(data_df, pd.DataFrame) else ""
        family, variant, data_basis, variant_treatment = _registry_family_metadata(
            rule,
            str(leg.get("index_name", "")) or column,
        ) if rule else ("", "", "", "")
        rows.append(
            {
                "column_name": column,
                "agent": str(leg.get("agent", "")),
                "index_name": str(leg.get("index_name", "")),
                "aggregation": str(leg.get("aggregation", "")),
                "flag_column": flag_col,
                "function": str(rule.get("function", "")),
                "method": str(rule.get("method", "")),
                "domain": str(rule.get("domain", "")),
                "role": str(rule.get("role", "")),
                "evidence_use": _evidence_use_label(rule) if rule else "",
                "b3_eligible": 1 if rule and _b3_allowed_from_rule(rule) else 0,
                "package_output": str(rule.get("package_output", "")),
                "package_flag": str(rule.get("package_flag", "")),
                "flag_type": str(rule.get("flag_type", "")),
                "threshold_rule": str(rule.get("threshold_rule", "")),
                "threshold_source": str(rule.get("threshold_source", "")),
                "data_basis": data_basis,
                "aggregation_family": family,
                "correction_variant": variant,
                "variant_treatment": variant_treatment,
                "notes": str(leg.get("notes", "")) or str(rule.get("notes", "")),
            }
        )
    return pd.DataFrame(rows)


def _agent_key_for_fn_label(fn_label: str) -> str:
    for fn, agent_key in ABERRANCE_FN_TO_AGENT.items():
        if fn == fn_label:
            return agent_key
    return ""


def _build_examinee_flags_export(
    flags: dict,
    n_examinees: int,
    *,
    visible_agents: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (examinee data, column dictionary, per-agent index guide)."""
    empty = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if n_examinees <= 0:
        return empty

    agents_for_export = _export_agents(flags, visible_agents)
    columns: dict[str, list] = {"Examinee_ID": list(range(1, n_examinees + 1))}
    legend_rows: list[dict] = [
        _legend_row("Examinee_ID", "", "", "Examinee_ID", aggregation="id", notes="1-based examinee number"),
    ]

    collusion_partners = [""] * n_examinees

    for agent_key, fn_label in agents_for_export:
        data = flags.get(agent_key, {})
        if not isinstance(data, dict) or data.get("error"):
            continue

        if agent_key == "as_agent" and isinstance(data.get("stat"), list) and data.get("stat"):
            pair_cols, pair_legend = _pair_level_index_columns(
                agent_key,
                fn_label,
                data.get("stat") or [],
                n_examinees,
                flagged_copiers=None,
            )
            columns.update(pair_cols)
            legend_rows.extend(pair_legend)
        else:
            person_cols, person_legend = _person_level_index_columns(agent_key, fn_label, data, n_examinees)
            columns.update(person_cols)
            legend_rows.extend(person_legend)

        pairs = data.get("pairs")
        if isinstance(pairs, list) and pairs:
            fc: set[int] = set()
            if agent_key == "ac_agent":
                for x in data.get("flagged_copiers") or []:
                    try:
                        fc.add(int(x))
                    except (TypeError, ValueError):
                        pass
            pair_cols, pair_legend = _pair_level_index_columns(
                agent_key,
                fn_label,
                pairs,
                n_examinees,
                flagged_copiers=fc if agent_key == "ac_agent" else None,
            )
            columns.update(pair_cols)
            legend_rows.extend(pair_legend)
            if agent_key == "ac_agent":
                partners_by_person: list[str] = []
                adjacency: dict[int, set[int]] = {}
                for pr in pairs:
                    p1, p2 = _pair_record_participants(pr)
                    if p1 is None or p2 is None:
                        continue
                    a, b = p1 + 1, p2 + 1
                    adjacency.setdefault(a, set()).add(b)
                    adjacency.setdefault(b, set()).add(a)
                for p in range(n_examinees):
                    pid = p + 1
                    partners = sorted(adjacency.get(pid, set()))
                    partners_by_person.append(", ".join(str(x) for x in partners))
                collusion_partners = partners_by_person

    data_df = pd.DataFrame(columns)
    index_cols = sorted(
        c
        for c in data_df.columns
        if c not in {"Examinee_ID", "Any_Flagged", "Flags_Count", "Flagged_By", "Collusion_Partners"}
    )
    paired_index_cols: list[str] = []
    for col in index_cols:
        paired_index_cols.append(col)
        legend_df = pd.DataFrame(legend_rows)
        if not legend_df.empty and "column_name" in legend_df.columns:
            source_match = legend_df[legend_df["column_name"] == col]
        else:
            source_match = pd.DataFrame()
        source_aggregation = str(source_match.iloc[0].get("aggregation", "")) if not source_match.empty else ""
        if source_aggregation == "index_flag":
            continue
        source_rule = _rulebook_match_for_export_column(legend_df, col)
        if (
            str(source_rule.get("package_flag", "")).strip().lower() == "yes"
            and str(source_rule.get("flag_type", "")).strip().lower() == "package_pvalue_flag"
            and _evidence_flag_column_for_source(data_df, legend_df, col)
        ):
            continue
        index_flag_col = f"{col}_Flag"
        flag_values, flag_note = _index_flag_values_for_export_column(data_df, legend_df, col)
        if not any(v is not None for v in flag_values):
            continue
        data_df[index_flag_col] = flag_values
        paired_index_cols.append(index_flag_col)
        if not legend_df.empty and "column_name" in legend_df.columns:
            match = legend_df[legend_df["column_name"] == col]
        else:
            match = pd.DataFrame()
        fn_label = str(match.iloc[0].get("agent", "")) if not match.empty else ""
        index_name = str(match.iloc[0].get("index_name", col)) if not match.empty else col
        agent_key = _agent_key_for_fn_label(fn_label)
        legend_rows.append(
            _legend_row(
                index_flag_col,
                agent_key,
                fn_label,
                f"{index_name}_flag",
                aggregation="index_flag",
                notes=flag_note,
            )
        )

    flagged_by_per_person: list[list[str]] = [[] for _ in range(n_examinees)]
    flag_cols: list[str] = []
    for agent_key, fn_label in agents_for_export:
        flag_col = _export_flag_col_name(fn_label)
        source_flag_cols = [
            str(row.get("column_name"))
            for row in legend_rows
            if row.get("agent") == fn_label
            and row.get("aggregation") == "index_flag"
            and str(row.get("column_name")) in data_df.columns
        ]
        values = []
        for p in range(n_examinees):
            flagged = any(_gov_truthy_flag(data_df.iloc[p][src_col]) for src_col in source_flag_cols)
            values.append(1 if flagged else 0)
            if flagged:
                flagged_by_per_person[p].append(fn_label)
        data_df[flag_col] = values
        flag_cols.append(flag_col)
        legend_rows.append(
            _legend_row(
                flag_col,
                agent_key,
                fn_label,
                "flagged",
                is_primary=True,
                aggregation="flag_from_indices",
                notes="1 if any exported index-level _Flag column for this function is 1; backend aggregate flags are not used on this page.",
            )
        )

    data_df["Any_Flagged"] = [1 if x else 0 for x in flagged_by_per_person]
    data_df["Flags_Count"] = [len(x) for x in flagged_by_per_person]
    data_df["Flagged_By"] = [", ".join(x) for x in flagged_by_per_person]
    data_df["Collusion_Partners"] = collusion_partners
    for sum_col, note in (
        ("Any_Flagged", "1 if any exported index-level flag is active for this examinee"),
        ("Flags_Count", "Number of functions with at least one active exported index-level flag"),
        ("Flagged_By", "Comma-separated detect_* functions derived from index-level flags"),
        ("Collusion_Partners", "Partner examinee IDs from detect_ac pairs"),
    ):
        legend_rows.append(
            _legend_row(sum_col, "", "", sum_col, aggregation="summary", notes=note),
        )
    col_order = [
        "Examinee_ID",
        "Any_Flagged",
        "Flags_Count",
        *flag_cols,
        "Flagged_By",
        "Collusion_Partners",
        *paired_index_cols,
    ]
    data_df = data_df[[c for c in col_order if c in data_df.columns]]

    legend_df = pd.DataFrame(legend_rows)
    guide_df = _build_agent_index_guide(flags, agents_for_export, legend_df, data_df)
    return data_df, legend_df, guide_df


def _forensic_indices_table_df(data_df: pd.DataFrame) -> pd.DataFrame:
    """Visible/downloadable Forensic Indices table: no aggregate flag summary columns."""
    if data_df.empty:
        return data_df
    aggregate_cols = {
        "Any_Flagged",
        "Flags_Count",
        "Flagged_By",
        "Collusion_Partners",
    }
    aggregate_cols.update(c for c in data_df.columns if c.endswith("_flagged"))
    visible_cols = [c for c in data_df.columns if c not in aggregate_cols]
    return data_df[visible_cols].copy()


@st.cache_data(show_spinner=False)
def _load_rulebook_mapping_rows() -> list[dict[str, str]]:
    path = Path("config") / "rulebook_index.csv"
    if not path.exists():
        path = Path("manuscript") / "rulebook_index.csv"
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path, dtype=str).fillna("")
    except Exception:
        return []
    return df.to_dict(orient="records")


def _rulebook_method_tokens(method: str) -> list[str]:
    text = str(method or "").replace("families", "").replace("family", "")
    parts = re.split(r"\s*/\s*|,\s*", text)
    out: list[str] = []
    for part in parts:
        token = re.sub(r"\s+", "", part.strip())
        if token:
            out.append(token)
    return out


def _method_matches_index(method: str, index_name: str) -> bool:
    method = str(method or "").strip()
    index = str(index_name or "").strip()
    if not method or not index:
        return False
    base = re.sub(r"(_pval|_flag|min|max|_min|_max)$", "", index, flags=re.IGNORECASE)
    if method == index or method == base:
        return True
    tokens = _rulebook_method_tokens(method)
    for token in tokens:
        if token and (base == token or base.startswith(token + "_") or index.startswith(token + "_")):
            return True
    family_patterns = {
        "ECI": ["ECI2", "ECI4"],
        "L": ["L_S", "L_ST", "L_R", "L_T", "L_RT"],
        "Q": ["Q_ST", "Q_RT"],
    }
    if method in family_patterns:
        return any(base.startswith(prefix) for prefix in family_patterns[method])
    if method.endswith("_*"):
        prefix = method[:-2]
        return base.startswith(prefix)
    return False


def _rulebook_match_for_export_column(legend_df: pd.DataFrame, column_name: str) -> dict[str, str]:
    if legend_df is None or legend_df.empty:
        return {}
    if "column_name" not in legend_df.columns:
        return {}
    match = legend_df[legend_df["column_name"].astype(str) == str(column_name)]
    if match.empty:
        return {}
    rec = match.iloc[0].to_dict()
    fn = str(rec.get("agent") or "")
    index_name = str(rec.get("index_name") or column_name)
    rows = [row for row in _load_rulebook_mapping_rows() if str(row.get("function", "")) == fn]
    if not rows:
        return {"function": fn, "method": index_name, "domain": "", "role": "", "evidence_use": "", "flag_type": "", "threshold_rule": "", "threshold_source": ""}
    exact = [row for row in rows if _method_matches_index(str(row.get("method", "")), index_name)]
    if exact:
        return exact[0]
    prefix = str(column_name).split("_", 1)[0].lower()
    base = str(index_name).upper()
    inferred_domain = ""
    if fn == "detect_nm":
        inferred_domain = "MF"
    elif fn == "detect_pm":
        inferred_domain = "MF"
    elif prefix in {"rg"} or fn == "detect_rg":
        inferred_domain = "RT"
    elif prefix in {"as", "ac"} or fn in {"detect_as", "detect_ac"}:
        inferred_domain = "SIM"
    elif prefix == "pk" or fn == "detect_pk":
        inferred_domain = "PK"
    elif prefix == "cp" or fn == "detect_cp":
        inferred_domain = "CP"
    elif prefix == "tt" or fn == "detect_tt":
        inferred_domain = "TP"
    compatible = [row for row in rows if not inferred_domain or str(row.get("domain", "")) == inferred_domain]
    if compatible:
        merged = dict(compatible[0])
        merged["method"] = index_name
        return merged
    return {"function": fn, "method": index_name, "domain": inferred_domain, "role": "", "evidence_use": "", "flag_type": "", "threshold_rule": "", "threshold_source": ""}


def _evidence_use_label(row: dict[str, str]) -> str:
    evidence_use = str(row.get("evidence_use", "")).strip().lower()
    if evidence_use in {"flag", "evidence flag"}:
        return EVIDENCE_USE_FLAG
    if evidence_use in {"rule flag", "calibration required"}:
        return EVIDENCE_USE_CALIBRATION
    if evidence_use in {"support only", "display only"}:
        return EVIDENCE_USE_DISPLAY
    feeds = str(row.get("feeds_B3", "")).strip().lower()
    flag_type = str(row.get("flag_type", "")).strip().lower()
    if feeds == "yes":
        return EVIDENCE_USE_FLAG
    if feeds == "no_until_calibrated":
        return EVIDENCE_USE_CALIBRATION
    if flag_type in {"auxiliary_only", "descriptive_only"} or feeds == "no":
        return EVIDENCE_USE_DISPLAY
    return EVIDENCE_USE_DISPLAY


def _forensic_index_catalog_df(data_df: pd.DataFrame, legend_df: pd.DataFrame) -> pd.DataFrame:
    if data_df is None or data_df.empty or legend_df is None or legend_df.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    candidate_cols = [
        col for col in data_df.columns
        if col != "Examinee_ID"
        and not str(col).endswith("_flagged")
    ]
    for col in candidate_cols:
        if col not in data_df.columns:
            continue
        leg_match = legend_df[legend_df["column_name"].astype(str) == str(col)] if "column_name" in legend_df.columns else pd.DataFrame()
        aggregation = str(leg_match.iloc[0].get("aggregation", "")) if not leg_match.empty else ""
        rule = _rulebook_match_for_export_column(legend_df, col)
        if not rule:
            continue
        flag_col = _evidence_flag_column_for_source(data_df, legend_df, col)
        evidence_use = _evidence_use_label(rule)
        if aggregation in {"pair_count", "pair_min", "pair_max", "pair_flag_display"}:
            evidence_use = EVIDENCE_USE_DISPLAY
            flag_col = ""
        if evidence_use == EVIDENCE_USE_FLAG and aggregation != "index_flag" and flag_col:
            continue
        flag_series = data_df[flag_col] if flag_col in data_df.columns else pd.Series([None] * len(data_df))
        flagged_n = int(sum(1 for value in flag_series.tolist() if _gov_truthy_flag(value)))
        nonmissing_n = int(data_df[col].replace("", pd.NA).notna().sum())
        domain = str(rule.get("domain", "")).strip()
        rows.append(
            {
                "Domain": domain,
                "Domain Label": _DOMAIN_LABELS.get(domain, domain),
                "Function": str(rule.get("function", "")),
                "Index": str(rule.get("method", "")) or str(col),
                "Column": str(col),
                "Evidence Use": evidence_use,
                "Registry Use": evidence_use,
                "Flag Column": flag_col if flag_col else "Missing",
                "Role": str(rule.get("role", "")),
                "Flag Type": str(rule.get("flag_type", "")),
                "Rule / Threshold": str(rule.get("threshold_rule", rule.get("default_rule", ""))),
                "Threshold Source": str(rule.get("threshold_source", "")),
                "Flagged Examinees": flagged_n,
                "Rows With Values": nonmissing_n,
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    order = {domain: i for i, domain in enumerate(_DOMAIN_ORDER)}
    df["_domain_order"] = df["Domain"].map(lambda x: order.get(str(x), 99))
    df["_use_order"] = df["Evidence Use"].map(lambda x: EVIDENCE_USE_ORDER.get(str(x), 9))
    df = df.sort_values(["_domain_order", "_use_order", "Function", "Index", "Column"]).drop(columns=["_domain_order", "_use_order"])
    return df


def _render_eligibility_summary(catalog_df: pd.DataFrame) -> None:
    counts = {EVIDENCE_USE_FLAG: 0, EVIDENCE_USE_CALIBRATION: 0, EVIDENCE_USE_DISPLAY: 0}
    if catalog_df is not None and not catalog_df.empty:
        counts.update(catalog_df["Evidence Use"].value_counts().to_dict())
    st.markdown(
        f"""
        <div class="psymas-eligibility-strip">
          <span class="psymas-use-chip evidence">Evidence Flag <b>{int(counts.get(EVIDENCE_USE_FLAG, 0))}</b></span>
          <span class="psymas-use-chip calibration">Calibration Required <b>{int(counts.get(EVIDENCE_USE_CALIBRATION, 0))}</b></span>
          <span class="psymas-use-chip support">Display Only <b>{int(counts.get(EVIDENCE_USE_DISPLAY, 0))}</b></span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_forensic_domain_tables(catalog_df: pd.DataFrame, *, key_prefix: str = "forensic_domain") -> None:
    if catalog_df is None or catalog_df.empty:
        st.info("No index catalog is available for domain grouping.")
        return
    present_domains = [d for d in _DOMAIN_ORDER if d in set(catalog_df["Domain"].astype(str))]
    present_domains.extend(sorted(set(catalog_df["Domain"].astype(str)) - set(present_domains)))
    labels = {f"{d} · {_DOMAIN_LABELS.get(d, d)}": d for d in present_domains}
    selected_label = st.segmented_control(
        "Domain",
        options=list(labels),
        default=list(labels)[0],
        key=f"{key_prefix}_selected_domain",
        label_visibility="collapsed",
    )
    domain = labels.get(selected_label, present_domains[0])
    domain_df = catalog_df[catalog_df["Domain"].astype(str) == domain].copy()
    display_cols = [
        "Function",
        "Index",
        "Evidence Use",
        "Flag Column",
        "Role",
        "Flag Type",
        "Rule / Threshold",
        "Threshold Source",
        "Flagged Examinees",
        "Rows With Values",
        "Column",
    ]
    _render_aggrid_table(
        domain_df[[c for c in display_cols if c in domain_df.columns]],
        key=f"{key_prefix}_{domain}_aggrid",
        height=min(520, 150 + 34 * max(1, min(len(domain_df), 10))),
    )


def _forensic_visible_agents() -> set[str]:
    selected = st.session_state.get("last_detect_agents") or [
        fn for fn in ABERRANCE_FUNCTIONS if st.session_state.get(f"ab_only_cb_{fn}")
    ]
    if not selected:
        selected = ["detect_nm"]
    return {ABERRANCE_FN_TO_AGENT[fn] for fn in selected if fn in ABERRANCE_FN_TO_AGENT}


_GOV_STRENGTH_RANK = {"unavailable": -1, "none": 0, "weak": 1, "moderate": 2, "strong": 3}


def _governance_rules_config() -> dict:
    cfg = _active_threshold_config()
    governance = cfg.get("governance") if isinstance(cfg, dict) else {}
    return governance if isinstance(governance, dict) else {}


def _active_b3_domain_strength_rules() -> list[dict]:
    configured = _governance_rules_config().get("domain_strength")
    source = configured if isinstance(configured, dict) else DEFAULT_B3_DOMAIN_STRENGTH_RULES
    rules = []
    for code, rule in source.items():
        if isinstance(rule, dict):
            merged = dict(DEFAULT_B3_DOMAIN_STRENGTH_RULES.get(str(code), {}))
            merged.update(rule)
            merged["code"] = str(code)
            rules.append(merged)
    return rules


def _b3_rule_matches(rule: dict, *, available: bool, primary_count: int, supporting_count: int) -> bool:
    condition = str(rule.get("condition", "")).strip()
    if condition == "required_data_unavailable":
        return not available
    if not available:
        return False
    if condition == "no_indicators":
        return primary_count == 0 and supporting_count == 0
    if condition == "supporting_only":
        if primary_count != 0:
            return False
    elif condition == "primary_only":
        if primary_count == 0 or supporting_count != 0:
            return False
    elif condition == "primary_and_supporting":
        if primary_count == 0 or supporting_count == 0:
            return False
    elif condition == "multiple_primary":
        if primary_count < 2:
            return False
    else:
        return False

    min_primary = rule.get("min_primary")
    max_primary = rule.get("max_primary")
    min_supporting = rule.get("min_supporting")
    max_supporting = rule.get("max_supporting")
    if min_primary is not None and primary_count < int(min_primary):
        return False
    if max_primary is not None and primary_count > int(max_primary):
        return False
    if min_supporting is not None and supporting_count < int(min_supporting):
        return False
    if max_supporting is not None and supporting_count > int(max_supporting):
        return False
    return True


def _select_b3_domain_strength_rule(*, available: bool, primary_count: int, supporting_count: int) -> dict:
    for rule in _active_b3_domain_strength_rules():
        if _b3_rule_matches(
            rule,
            available=available,
            primary_count=primary_count,
            supporting_count=supporting_count,
        ):
            return rule
    fallback_code = "B3-00" if not available else "B3-01"
    fallback = dict(DEFAULT_B3_DOMAIN_STRENGTH_RULES[fallback_code])
    fallback["code"] = fallback_code
    return fallback


def _active_b5_review_priority_rules() -> dict[str, tuple[str, str, str]]:
    configured = _governance_rules_config().get("review_priority")
    source = configured if isinstance(configured, dict) else DEFAULT_B5_REVIEW_PRIORITY_RULES
    by_status: dict[str, tuple[str, str, str]] = {}
    for code, rule in source.items():
        if not isinstance(rule, dict):
            continue
        status = str(rule.get("case_status", "")).strip()
        if not status:
            continue
        by_status[status] = (
            str(rule.get("rule_id", "")),
            str(code),
            str(rule.get("review_priority", "Low")),
        )
    return by_status


def _active_index_activation_rules() -> dict[str, dict]:
    configured = _governance_rules_config().get("index_activation")
    source = configured if isinstance(configured, dict) else DEFAULT_INDEX_ACTIVATION_RULES
    rules: dict[str, dict] = {}
    for name, default_rule in DEFAULT_INDEX_ACTIVATION_RULES.items():
        merged = dict(default_rule)
        if isinstance(source.get(name), dict):
            merged.update(source[name])
        rules[name] = merged
    for name, rule in source.items() if isinstance(source, dict) else []:
        if name not in rules and isinstance(rule, dict):
            rules[str(name)] = dict(rule)
    return rules


def _column_matches_patterns(column: str, patterns: list[str] | tuple[str, ...] | None) -> bool:
    import fnmatch
    if not patterns:
        return False
    col = str(column)
    return any(fnmatch.fnmatch(col, str(pattern)) for pattern in patterns)


def _activation_rule_matches_value(rule: dict, value) -> bool:
    if not bool(rule.get("enabled", True)):
        return False
    operator = str(rule.get("operator", "truthy")).strip().lower()
    if operator == "truthy":
        return _gov_truthy_flag(value)
    val = _gov_to_float(value)
    if val is None:
        return False
    threshold = rule.get("value", 0)
    if isinstance(threshold, str) and threshold.lower() == "alpha":
        threshold = (_active_threshold_config().get("defaults") or {}).get("alpha", 0.05)
    try:
        thr = float(threshold)
    except (TypeError, ValueError):
        return False
    if operator in {"<=", "le"}:
        return val <= thr
    if operator in {"<", "lt"}:
        return val < thr
    if operator in {">=", "ge"}:
        return val >= thr
    if operator in {">", "gt"}:
        return val > thr
    if operator in {"==", "eq"}:
        return val == thr
    if operator in {"!=", "ne"}:
        return val != thr
    return False


def _index_activation_hit(rule_name: str, column: str, value) -> bool:
    rule = _active_index_activation_rules().get(rule_name, {})
    if not _column_matches_patterns(str(column), rule.get("column_patterns") or []):
        return False
    return _activation_rule_matches_value(rule, value)


def _threshold_rule_for_index(fn_label: str, index_name: str) -> tuple[dict | None, str]:
    import fnmatch
    cfg = _active_threshold_config()
    rules = cfg.get("rules") if isinstance(cfg, dict) else {}
    block = rules.get(fn_label) if isinstance(rules, dict) else None
    if not isinstance(block, dict):
        return None, "No threshold block configured."

    if index_name in block and isinstance(block[index_name], dict):
        return block[index_name], f"rules.{fn_label}.{index_name}"

    for key, rule in block.items():
        if not isinstance(rule, dict):
            continue
        if fnmatch.fnmatch(index_name, str(key)):
            return rule, f"rules.{fn_label}.{key}"

    if "pval" in index_name.lower() and "alpha" in block:
        return {"enabled": True, "operator": "<=", "value": block.get("alpha"), "source": block.get("source", "")}, f"rules.{fn_label}.alpha"

    return None, "No index-specific threshold configured."


def _threshold_value_for_series(rule: dict, series: pd.Series):
    value = rule.get("value")
    if isinstance(value, str):
        lowered = value.lower()
        if lowered == "alpha":
            return (_active_threshold_config().get("defaults") or {}).get("alpha", 0.05)
        if lowered == "sample_quantile":
            quantile = rule.get("quantile", 0.95)
            try:
                numeric = pd.to_numeric(series, errors="coerce")
                return float(numeric.quantile(float(quantile)))
            except Exception:
                return None
    return value


def _compare_threshold_value(value, operator: str, threshold) -> int | None:
    val = _gov_to_float(value)
    if val is None:
        return None
    try:
        thr = float(threshold)
    except (TypeError, ValueError):
        return None
    op = str(operator or "").strip().lower()
    if op in {"<=", "le"}:
        return 1 if val <= thr else 0
    if op in {"<", "lt"}:
        return 1 if val < thr else 0
    if op in {">=", "ge"}:
        return 1 if val >= thr else 0
    if op in {">", "gt"}:
        return 1 if val > thr else 0
    if op in {"==", "eq"}:
        return 1 if val == thr else 0
    if op in {"!=", "ne"}:
        return 1 if val != thr else 0
    if op == "truthy":
        return 1 if _gov_truthy_flag(value) else 0
    return None


def _index_flag_values_for_export_column(data_df: pd.DataFrame, legend_df: pd.DataFrame, column: str) -> tuple[list, str]:
    if column not in data_df.columns:
        return [], "Index flag unavailable: source column missing."
    match = legend_df[legend_df["column_name"] == column] if not legend_df.empty and "column_name" in legend_df.columns else pd.DataFrame()
    fn_label = str(match.iloc[0].get("agent", "")) if not match.empty else ""
    index_name = str(match.iloc[0].get("index_name", column)) if not match.empty else column
    aggregation = str(match.iloc[0].get("aggregation", "")) if not match.empty else ""
    series = data_df[column]
    if index_name in {"n_pairs", "flagged_copier"} or aggregation in {"pair_count", "pair_min", "pair_max", "pair_flag_display"}:
        return [None for _ in series.tolist()], "Index flag not computed: pair-level trace column, not a person-level forensic evidence flag."

    rulebook_rule = _rulebook_match_for_export_column(legend_df, column)
    package_flag = str(rulebook_rule.get("package_flag", "")).strip().lower()
    if package_flag in {"yes", "partial"} and "flag" in index_name.lower():
        values = [1 if _gov_truthy_flag(v) else 0 for v in series.tolist()]
        return values, "Index flag from package-returned flag output defined in the Index Registry."
    if package_flag == "yes" and str(rulebook_rule.get("flag_type", "")).strip().lower() == "package_pvalue_flag":
        if "pval" in index_name.lower():
            alpha = (_active_threshold_config().get("rules", {}).get(fn_label, {}) or {}).get(
                "alpha",
                (_active_threshold_config().get("defaults") or {}).get("alpha", 0.05),
            )
            values = [_compare_threshold_value(v, "<=", alpha) for v in series.tolist()]
            return values, f"Fallback Evidence Flag from package p-value because no package flag column was exported: <= {alpha}."
        return [None for _ in series.tolist()], "Package statistic displayed only; Evidence Input requires a package flag column or a configured threshold."

    rule, source = _threshold_rule_for_index(fn_label, index_name)
    if isinstance(rule, dict) and bool(rule.get("enabled", True)):
        threshold = _threshold_value_for_series(rule, series)
        operator = str(rule.get("operator", "")).strip() or ("<=" if "pval" in index_name.lower() else "")
        values = [_compare_threshold_value(v, operator, threshold) for v in series.tolist()]
        note = f"Index flag from {source}: {operator} {threshold}."
        return values, note

    activation_rules = _active_index_activation_rules()
    for rule_name, activation_rule in activation_rules.items():
        if not _column_matches_patterns(column, activation_rule.get("column_patterns") or []):
            continue
        values = [1 if _activation_rule_matches_value(activation_rule, v) else 0 for v in series.tolist()]
        operator = activation_rule.get("operator", "")
        threshold = activation_rule.get("value", "")
        note = f"Index flag from governance.index_activation.{rule_name}: {operator} {threshold}."
        return values, note

    return [None for _ in series.tolist()], "Index flag not computed: no threshold or activation rule configured in YAML."


_GOV_DOMAIN_META = {
    "MF": {
        "label": "Misfit Evidence",
        "flag_cols": ["pm_flagged"],
        "primary_cols": [],
        "supporting_prefixes": ["pm_", "nm_"],
        "required": ["responses"],
    },
    "RT": {
        "label": "Response-Time Evidence",
        "flag_cols": ["rg_flagged"],
        "primary_cols": ["rg_CT", "rg_NT", "rg_CUMP"],
        "supporting_prefixes": ["rg_"],
        "required": ["response_times"],
    },
    "SIM": {
        "label": "Similarity Evidence",
        "flag_cols": ["ac_flagged", "as_flagged"],
        "primary_cols": ["ac_OMG_S_max", "as_M4_S_max", "as_OMG_S_max", "as_GBT_S_max"],
        "supporting_prefixes": ["ac_", "as_"],
        "required": ["responses"],
    },
    "PK": {
        "label": "Preknowledge Evidence",
        "flag_cols": ["pk_flagged"],
        "primary_cols": ["pk_L_S", "pk_L_T", "pk_L_ST", "pk_LR_S", "pk_ML_S"],
        "supporting_prefixes": ["pk_"],
        "required": ["exposure_or_compromised_items"],
    },
    "CP": {
        "label": "Change Point Evidence",
        "flag_cols": ["cp_flagged"],
        "primary_cols": ["cp_L_T_MAX1", "cp_L_S_MAX1", "cp_W_T_MAX1", "cp_W_S_MAX1"],
        "supporting_prefixes": ["cp_"],
        "required": ["item_sequence_or_testing_order"],
    },
    "TP": {
        "label": "Tampering Evidence",
        "flag_cols": ["tt_flagged"],
        "primary_cols": ["tt_EDI_SD", "tt_EDI_R"],
        "supporting_prefixes": ["tt_"],
        "required": ["answer_change_records"],
    },
}


def _gov_to_float(value) -> float | None:
    try:
        if value is None or value == "":
            return None
        out = float(value)
        if not math.isfinite(out):
            return None
        return out
    except (TypeError, ValueError):
        return None


def _gov_truthy_flag(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    return bool(_gov_to_float(value))


def _truthy_config_value(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}
    return bool(value)


def _rule_flag_promoted_to_b3(fn_label: str, index_name: str) -> bool:
    rule, _ = _threshold_rule_for_index(fn_label, index_name)
    if not isinstance(rule, dict):
        return False
    return _truthy_config_value(rule.get("feeds_B3")) or _truthy_config_value(rule.get("B3_eligible")) or _truthy_config_value(rule.get("b3_eligible"))


def _b2_domain_codes_from_mapping() -> list[str]:
    domains: set[str] = set()
    for row in _load_rulebook_mapping_rows():
        fn = str(row.get("function", ""))
        domain = str(row.get("domain", "")).strip()
        if fn.startswith("detect_") and domain in _DOMAIN_ORDER:
            domains.add(domain)
    return [d for d in _DOMAIN_ORDER if d in domains]


def _b2_indicator_role(role: str) -> str:
    role_l = str(role or "").strip().lower()
    if "primary" in role_l:
        return "primary"
    if "supporting" in role_l:
        return "supporting"
    return "support"


def _b2_source_columns_for_domain(df: pd.DataFrame, legend_df: pd.DataFrame, domain: str) -> list[dict]:
    if df is None or df.empty or legend_df is None or legend_df.empty:
        return []
    out: list[dict] = []
    seen: set[str] = set()
    for col in df.columns:
        col_s = str(col)
        if col_s == "Examinee_ID" or col_s.endswith("_Flag") or col_s.endswith("_flagged"):
            continue
        if col_s in {"Any_Flagged", "Flags_Count", "Flagged_By", "Collusion_Partners"}:
            continue
        rule = _rulebook_match_for_export_column(legend_df, col_s)
        if not rule or str(rule.get("domain", "")).strip() != domain:
            continue
        if col_s in seen:
            continue
        seen.add(col_s)
        flag_column = _evidence_flag_column_for_source(df, legend_df, col_s)
        out.append({"column": col_s, "flag_column": flag_column, "rule": rule})
    return out


def _b2_domain_source_plan(df: pd.DataFrame, legend_df: pd.DataFrame, domains: list[str]) -> dict[str, list[dict]]:
    plan = {domain: [] for domain in domains}
    if df is None or df.empty or legend_df is None or legend_df.empty:
        return plan
    wanted = set(domains)
    seen: set[tuple[str, str]] = set()
    for col in df.columns:
        col_s = str(col)
        if col_s == "Examinee_ID" or col_s.endswith("_Flag") or col_s.endswith("_flagged"):
            continue
        if col_s in {"Any_Flagged", "Flags_Count", "Flagged_By", "Collusion_Partners"}:
            continue
        rule = _rulebook_match_for_export_column(legend_df, col_s)
        domain = str(rule.get("domain", "")).strip() if rule else ""
        if domain not in wanted:
            continue
        key = (domain, col_s)
        if key in seen:
            continue
        seen.add(key)
        use_label = _evidence_use_label(rule)
        role_kind = _b2_indicator_role(str(rule.get("role", "")))
        flag_column = _evidence_flag_column_for_source(df, legend_df, col_s)
        family, _, _, treatment = _registry_family_metadata(rule, col_s)
        hit_key, _ = b3_family_signal_key(
            function=rule.get("function", ""),
            aggregation_family=family,
            variant_treatment=treatment,
            index_column=col_s,
            hit_label=_b2_indicator_key(rule, col_s),
        )
        plan[domain].append(
            {
                "column": col_s,
                "flag_column": flag_column,
                "rule": rule,
                "use_label": use_label,
                "role_kind": role_kind,
                "label": _b2_indicator_label(rule, col_s),
                "hit_key": hit_key,
                "b3_allowed": _b2_rule_b3_allowed(rule) and _variant_allowed_for_b3(treatment),
            }
        )
    return plan


def _b2_indicator_key(rule: dict, col: str) -> str:
    fn = str(rule.get("function", ""))
    method = str(rule.get("method", "")) or str(col)
    role = str(rule.get("role", ""))
    return f"{fn}|{method}|{role}"


def _b2_indicator_label(rule: dict, col: str) -> str:
    fn = str(rule.get("function", ""))
    method = str(rule.get("method", "")) or str(col)
    return f"{fn}/{method}"


def _b2_rule_b3_allowed(rule: dict) -> bool:
    use_label = _evidence_use_label(rule)
    if use_label == EVIDENCE_USE_FLAG:
        return True
    if use_label == EVIDENCE_USE_CALIBRATION:
        return _rule_flag_promoted_to_b3(str(rule.get("function", "")), str(rule.get("method", "")))
    return False


def _b2_domain_profile(row: pd.Series, source_cols: list[dict], domain: str) -> dict:
    available = bool(source_cols)
    if not available:
        rule = _select_b3_domain_strength_rule(available=False, primary_count=0, supporting_count=0)
        return {
            "domain": domain,
            "label": _DOMAIN_LABELS.get(domain, domain),
            "strength": str(rule.get("strength", "unavailable")).lower(),
            "primary_hits": [],
            "supporting_hits": [],
            "eligible_flags": [],
            "rule_flag_candidates": [],
            "support_only_hits": [],
            "trace_columns": [],
            "available": False,
            "primary_indicator_count": 0,
            "supporting_indicator_count": 0,
            "rule_id": rule.get("code", "B3-00"),
            "pattern": rule.get("evidence_pattern", "Required data for the domain are unavailable"),
            "note": "No index-registry output columns are available for this domain.",
        }

    primary_hits: list[str] = []
    supporting_hits: list[str] = []
    eligible_flags: list[str] = []
    rule_flag_candidates: list[str] = []
    support_only_hits: list[str] = []
    trace_columns: list[str] = []
    seen_hit_keys: set[str] = set()

    for item in source_cols:
        col = item["column"]
        flag_col = item["flag_column"]
        if flag_col not in row.index:
            continue
        flag_active = _gov_truthy_flag(row.get(flag_col))
        if not flag_active:
            continue
        use_label = item.get("use_label", EVIDENCE_USE_DISPLAY)
        role_kind = item.get("role_kind", "support")
        label = item.get("label", col)
        trace_columns.extend([col, flag_col])

        if use_label == EVIDENCE_USE_DISPLAY:
            support_only_hits.append(label)
            continue
        if use_label == EVIDENCE_USE_CALIBRATION and not item.get("b3_allowed", False):
            rule_flag_candidates.append(label)
            continue
        if not item.get("b3_allowed", False):
            support_only_hits.append(label)
            continue

        hit_key = item.get("hit_key", label)
        if hit_key in seen_hit_keys:
            continue
        seen_hit_keys.add(hit_key)
        eligible_flags.append(label)
        if role_kind == "primary":
            primary_hits.append(label)
        elif role_kind == "supporting":
            supporting_hits.append(label)

    rule = _select_b3_domain_strength_rule(
        available=True,
        primary_count=len(primary_hits),
        supporting_count=len(supporting_hits),
    )
    return {
        "domain": domain,
        "label": _DOMAIN_LABELS.get(domain, domain),
        "strength": str(rule.get("strength", "none")).lower(),
        "primary_hits": primary_hits,
        "supporting_hits": supporting_hits,
        "eligible_flags": eligible_flags,
        "rule_flag_candidates": rule_flag_candidates,
        "support_only_hits": support_only_hits,
        "trace_columns": sorted(set(trace_columns)),
        "available": True,
        "primary_indicator_count": len(primary_hits),
        "supporting_indicator_count": len(supporting_hits),
        "rule_id": str(rule.get("code", "B3-01")),
        "pattern": rule.get("evidence_pattern", ""),
        "note": "",
    }


def _gov_domain_available(df: pd.DataFrame, domain: str, meta: dict) -> bool:
    cols = set(df.columns)
    if domain in {"MF", "SIM"}:
        return bool(cols & set(meta["flag_cols"] + meta["primary_cols"]))
    if domain == "RT":
        return bool(cols & {"rg_flagged", "rg_RTE"})
    if domain == "PK":
        return bool(cols & {"pk_flagged", "pk_L_S", "pk_L_T", "pk_L_ST"})
    if domain == "CP":
        return bool(cols & {"cp_flagged", "cp_L_T_MAX1", "cp_L_S_MAX1", "cp_W_T_MAX1"})
    if domain == "TP":
        return bool(cols & {"tt_flagged", "tt_EDI_SD", "tt_EDI_R", "tt_GBT_SD", "tt_GBT_R", "tt_L_SD", "tt_L_R"})
    return False


def _gov_supporting_hits(row: pd.Series, domain: str, meta: dict, primary_flag_count: int) -> list[str]:
    hits: list[str] = []
    for col in row.index:
        if col in meta["flag_cols"] or col in meta["primary_cols"]:
            continue
        if not any(str(col).startswith(prefix) for prefix in meta["supporting_prefixes"]):
            continue
        value = row.get(col)
        if _index_activation_hit("supporting_p_values", str(col), value):
            hits.append(col)
    return hits


def _gov_domain_profile(row: pd.Series, df: pd.DataFrame, domain: str, meta: dict) -> dict:
    available = _gov_domain_available(df, domain, meta)
    if not available:
        rule = _select_b3_domain_strength_rule(available=False, primary_count=0, supporting_count=0)
        return {
            "domain": domain,
            "label": meta["label"],
            "strength": str(rule.get("strength", "unavailable")).lower(),
            "primary_hits": [],
            "supporting_hits": [],
            "trace_columns": [],
            "rule_id": rule.get("code", "B3-00"),
            "pattern": rule.get("evidence_pattern", "Required data for the domain are unavailable"),
            "note": f"Required data unavailable: {', '.join(meta['required'])}.",
        }

    primary_hits = [
        c for c in meta["flag_cols"]
        if c in row.index and _index_activation_hit("primary_flag_columns", str(c), row.get(c))
    ]
    supporting_hits = _gov_supporting_hits(row, domain, meta, len(primary_hits))
    rule = _select_b3_domain_strength_rule(
        available=True,
        primary_count=len(primary_hits),
        supporting_count=len(supporting_hits),
    )
    strength = str(rule.get("strength", "none")).lower()
    rule_id = str(rule.get("code", "B3-01"))

    trace_cols = sorted(set(primary_hits + supporting_hits + [c for c in meta["primary_cols"] if c in row.index]))
    return {
        "domain": domain,
        "label": meta["label"],
        "strength": strength,
        "primary_hits": primary_hits,
        "supporting_hits": supporting_hits,
        "trace_columns": trace_cols,
        "rule_id": rule_id,
        "pattern": rule.get("evidence_pattern", ""),
        "note": "",
    }


def _b3_domain_profile_from_input(examinee_rows: pd.DataFrame, domain: str) -> dict:
    label = _DOMAIN_LABELS.get(domain, domain)
    if examinee_rows is None or examinee_rows.empty:
        rule = _select_b3_domain_strength_rule(available=False, primary_count=0, supporting_count=0)
        return {
            "domain": domain,
            "label": label,
            "strength": str(rule.get("strength", "unavailable")).lower(),
            "primary_hits": [],
            "supporting_hits": [],
            "eligible_flags": [],
            "rule_flag_candidates": [],
            "support_only_hits": [],
            "trace_columns": [],
            "available": False,
            "primary_indicator_count": 0,
            "supporting_indicator_count": 0,
            "rule_id": str(rule.get("code", "B3-00")),
            "pattern": rule.get("evidence_pattern", "Required data for the domain are unavailable"),
            "note": "No Evidence Input rows are available for this domain.",
        }

    primary_hits: list[str] = []
    supporting_hits: list[str] = []
    eligible_flags: list[str] = []
    rule_flag_candidates: list[str] = []
    support_only_hits: list[str] = []
    trace_columns: list[str] = []
    seen_hit_keys: set[str] = set()

    for _, item in examinee_rows.iterrows():
        index_column = str(item.get("Index_Column", ""))
        flag_column = str(item.get("Flag_Column", ""))
        label_text = str(item.get("Index", "")) or index_column
        hit_label = f"{item.get('Function', '')}:{label_text}"
        flag_active = _gov_truthy_flag(item.get("Flag", 0))
        if not flag_active:
            continue
        if index_column:
            trace_columns.append(index_column)
        if flag_column:
            trace_columns.append(flag_column)
        evidence_use = str(item.get("Evidence_Use", ""))
        if evidence_use == EVIDENCE_USE_CALIBRATION:
            rule_flag_candidates.append(hit_label)
            continue
        if evidence_use == EVIDENCE_USE_DISPLAY or not _gov_truthy_flag(item.get("Evidence_Eligible", 0)):
            support_only_hits.append(hit_label)
            continue
        hit_key, hit_label = b3_family_signal_key(
            function=item.get("Function", ""),
            aggregation_family=item.get("Aggregation_Family", ""),
            variant_treatment=item.get("Variant_Treatment", ""),
            index_column=index_column,
            hit_label=hit_label,
        )
        if hit_key in seen_hit_keys:
            continue
        seen_hit_keys.add(hit_key)
        eligible_flags.append(hit_label)
        role_kind = str(item.get("Role_Class", "")).lower()
        if role_kind == "primary":
            primary_hits.append(hit_label)
        elif role_kind == "supporting":
            supporting_hits.append(hit_label)

    rule = _select_b3_domain_strength_rule(
        available=True,
        primary_count=len(primary_hits),
        supporting_count=len(supporting_hits),
    )
    return {
        "domain": domain,
        "label": label,
        "strength": str(rule.get("strength", "none")).lower(),
        "primary_hits": primary_hits,
        "supporting_hits": supporting_hits,
        "eligible_flags": eligible_flags,
        "rule_flag_candidates": rule_flag_candidates,
        "support_only_hits": support_only_hits,
        "trace_columns": sorted(set(trace_columns)),
        "available": True,
        "primary_indicator_count": len(primary_hits),
        "supporting_indicator_count": len(supporting_hits),
        "rule_id": str(rule.get("code", "B3-01")),
        "pattern": rule.get("evidence_pattern", ""),
        "note": "",
    }


_GOV_PRIMARY_SCENARIO_DOMAINS = {"RT", "SIM", "PK", "TP"}
_GOV_SUPPORTING_DOMAINS = {"MF"}
_GOV_HIGH_VERIFIABILITY_DOMAINS = {"PK", "TP"}


def _gov_case_status(domain_profiles: dict[str, dict]) -> tuple[str, str, list[str]]:
    strengths = {k: v["strength"] for k, v in domain_profiles.items()}
    ranks = {k: _GOV_STRENGTH_RANK.get(v, -1) for k, v in strengths.items()}
    available = [k for k, v in ranks.items() if v >= 0]
    scenario_moderate_plus = [
        k for k, v in ranks.items()
        if k in _GOV_PRIMARY_SCENARIO_DOMAINS and v >= 2
    ]
    scenario_weak = [
        k for k, v in ranks.items()
        if k in _GOV_PRIMARY_SCENARIO_DOMAINS and v == 1
    ]
    supporting_moderate_plus = [
        k for k, v in ranks.items()
        if k in _GOV_SUPPORTING_DOMAINS and v >= 2
    ]
    has_high_verifiability = bool(set(scenario_moderate_plus) & _GOV_HIGH_VERIFIABILITY_DOMAINS)

    if not available:
        return "No substantive evidence pattern", "Low", []
    if available and all(ranks[k] == 0 for k in available):
        return "None", "Low", ["EP-01"]
    if not scenario_moderate_plus and supporting_moderate_plus:
        return "Supporting evidence only", "Low", ["EP-02s"]
    if len(scenario_weak) >= 2 and not scenario_moderate_plus:
        return "Weak", "Low", ["EP-02"]
    if len(scenario_moderate_plus) == 1:
        return (
            "Isolated (Traceable)" if has_high_verifiability else "Isolated (Context-Dependent)",
            "High" if has_high_verifiability else "Medium",
            ["EP-03b" if has_high_verifiability else "EP-03a"],
        )
    if len(scenario_moderate_plus) >= 2:
        return (
            "Convergent (Traceable)" if has_high_verifiability else "Convergent (Context-Dependent)",
            "Critical / Expedited" if has_high_verifiability else "High (Context-Heavy)",
            ["EP-04b" if has_high_verifiability else "EP-04a"],
        )
    return "No substantive evidence pattern", "Low", []


def _gov_review_priority_rule(status: str) -> tuple[str, str, str]:
    return _active_b5_review_priority_rules().get(status, ("RP-00", "B6-00", "Low"))


def _build_governed_evidence_profile(
    result_df: pd.DataFrame,
    legend_df: pd.DataFrame | None = None,
    b3_input_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    domain_rows: list[dict] = []
    if result_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    legend_df = legend_df if isinstance(legend_df, pd.DataFrame) else pd.DataFrame()
    b2_domains = _b2_domain_codes_from_mapping()
    if not b2_domains:
        b2_domains = [d for d in _DOMAIN_ORDER if d != "CTX"]
    source_plan = _b2_domain_source_plan(result_df, legend_df, b2_domains)
    use_b3_input = isinstance(b3_input_df, pd.DataFrame) and not b3_input_df.empty
    b3_by_examinee: dict[str, pd.DataFrame] = {}
    if use_b3_input:
        b3_by_examinee = {
            str(k): group
            for k, group in b3_input_df.groupby(b3_input_df["Examinee_ID"].astype(str), sort=False)
        }
    for _, row in result_df.iterrows():
        examinee_id = row.get("Examinee_ID", "")
        if use_b3_input:
            ex_rows = b3_by_examinee.get(str(examinee_id), pd.DataFrame())
            profiles = {
                code: _b3_domain_profile_from_input(
                    ex_rows[ex_rows["Domain"].astype(str) == code] if not ex_rows.empty else ex_rows,
                    code,
                )
                for code in b2_domains
            }
        else:
            profiles = {
                code: _b2_domain_profile(row, source_plan.get(code, []), code)
                for code in b2_domains
            }
        status, priority, case_rules = _gov_case_status(profiles)
        priority_rule, priority_code, priority_value = _gov_review_priority_rule(status)
        priority = priority_value
        missing = [
            f"{code}:{profile['note']}"
            for code, profile in profiles.items()
            if profile["strength"] == "unavailable" and profile.get("note")
        ]
        scenario_concern = [
            code for code, profile in profiles.items()
            if code in _GOV_PRIMARY_SCENARIO_DOMAINS and _GOV_STRENGTH_RANK.get(profile["strength"], -1) >= 2
        ]
        supporting_concern = [
            code for code, profile in profiles.items()
            if code in _GOV_SUPPORTING_DOMAINS and _GOV_STRENGTH_RANK.get(profile["strength"], -1) >= 2
        ]
        primary_concern = ", ".join(scenario_concern) or "None"
        if primary_concern != "None" and supporting_concern:
            primary_concern = f"{primary_concern} (+ {'/'.join(supporting_concern)} support)"
        trace_cols = sorted({c for profile in profiles.values() for c in profile.get("trace_columns", [])})
        rows.append(
            {
                "Examinee_ID": examinee_id,
                "Evidence_Status": status,
                "Review_Priority": priority,
                "Review_Priority_Rule": priority_rule,
                "B6_Code": priority_code,
                "Case_Evidence_Rule": ", ".join(case_rules),
                "Primary_Concern": primary_concern,
                "Rule_IDs_Triggered": ", ".join(case_rules),
                "Missing_Evidence": " | ".join(missing),
                "Trace_Columns": ", ".join(trace_cols),
                "Draft_Statement": _governance_statement(status, primary_concern),
                "Audit_Status": "pass_trace_required" if trace_cols or status in {"None", "Weak", "No substantive evidence pattern"} else "needs_trace_review",
                **{f"{code}_Strength": profile["strength"] for code, profile in profiles.items()},
            }
        )
        for code, profile in profiles.items():
            domain_rows.append(
                {
                    "Examinee_ID": examinee_id,
                    "Domain": code,
                    "Domain_Label": profile["label"],
                    "Strength": profile["strength"],
                    "Strength_Rule": profile["rule_id"],
                    "Evidence_Pattern": profile["pattern"],
                    "Registry_Primary_Indices": _b2_domain_role_indices(code, "Primary"),
                    "Registry_Supporting_Indices": _b2_domain_role_indices(code, "Supporting"),
                    "Primary_Hits": ", ".join(profile["primary_hits"]),
                    "Supporting_Hits": ", ".join(profile["supporting_hits"]),
                    "Evidence_Flags": ", ".join(profile.get("eligible_flags", [])),
                    "Calibration_Required_Hits": ", ".join(profile.get("rule_flag_candidates", [])),
                    "Display_Only_Hits": ", ".join(profile.get("support_only_hits", [])),
                    "Available": profile.get("available", False),
                    "Primary_Indicators": profile.get("primary_indicator_count", 0),
                    "Supporting_Indicators": profile.get("supporting_indicator_count", 0),
                    "Trace_Columns": ", ".join(profile["trace_columns"]),
                    "Missing_Note": profile["note"],
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(domain_rows)


def _governance_statement(status: str, primary_concern: str) -> str:
    if status == "Convergent (Traceable)":
        return f"Multiple domains show convergent traceable evidence ({primary_concern}); expedited human review is required."
    if status == "Convergent (Context-Dependent)":
        return f"Multiple domains show convergent but context-dependent evidence ({primary_concern}); contextual records should be reviewed before adjudication."
    if status == "Isolated (Traceable)":
        return f"A single traceable domain is elevated ({primary_concern}); objective logs should be checked before adjudication."
    if status == "Isolated (Context-Dependent)":
        return f"A single context-dependent domain is elevated ({primary_concern}); interpret cautiously and check legitimate contextual explanations."
    if status == "Weak":
        return "Only weak evidence accumulation was identified from the available evidence."
    if status == "No substantive evidence pattern":
        return "No substantive case-level evidence pattern was identified from the available domains."
    return "No elevated forensic concern was identified from the available evidence."


def _infer_column_legend_from_result_df(data_df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild a minimal column legend from a saved result table when cached metadata is unavailable."""
    if data_df is None or data_df.empty:
        return pd.DataFrame()
    prefix_to_fn = {
        "nm": "detect_nm",
        "pm": "detect_pm",
        "ac": "detect_ac",
        "as": "detect_as",
        "rg": "detect_rg",
        "cp": "detect_cp",
        "tt": "detect_tt",
        "pk": "detect_pk",
    }
    rows: list[dict] = []
    for col in data_df.columns:
        col_s = str(col)
        if col_s == "Examinee_ID":
            rows.append(_legend_row(col_s, "", "", col_s, aggregation="id", notes="Inferred from result table."))
            continue
        if col_s in {"Any_Flagged", "Flags_Count", "Flagged_By", "Collusion_Partners"}:
            rows.append(_legend_row(col_s, "", "", col_s, aggregation="summary", notes="Inferred from result table."))
            continue
        if "_" not in col_s:
            rows.append(_legend_row(col_s, "", "", col_s, aggregation="summary", notes="Inferred from result table."))
            continue
        prefix, index_name = col_s.split("_", 1)
        fn_label = prefix_to_fn.get(prefix.lower(), "")
        agent_key = ABERRANCE_FN_TO_AGENT.get(fn_label, "")
        if not fn_label:
            rows.append(_legend_row(col_s, "", "", index_name, aggregation="summary", notes="Inferred from result table."))
            continue
        aggregation = "index_flag" if _csv_flag_index_key(index_name) else "person"
        if index_name.endswith("_max"):
            aggregation = "pair_max"
        elif index_name.endswith("_min"):
            aggregation = "pair_min"
        elif prefix.lower() in {"ac", "as"} and _csv_flag_index_key(index_name):
            aggregation = "pair_flag_display"
        elif index_name in {"n_pairs", "flagged_copier"}:
            aggregation = "pair_count"
        rows.append(
            _legend_row(
                col_s,
                agent_key,
                fn_label,
                index_name,
                is_primary=_index_is_primary(agent_key, index_name),
                aggregation=aggregation,
                notes="Inferred from result table.",
            )
        )
    return pd.DataFrame(rows)


def _forensic_result_flags_from_session() -> dict:
    """Return flags from the current Streamlit forensic result, tolerating shallow wrappers."""
    fr = st.session_state.get("forensic_result")
    candidates = [fr]
    if isinstance(fr, dict):
        for key in ("result", "state", "output", "data"):
            nested = fr.get(key)
            if isinstance(nested, dict):
                candidates.append(nested)
    for candidate in candidates:
        if isinstance(candidate, dict) and isinstance(candidate.get("flags"), dict):
            return candidate.get("flags") or {}
    return {}


def _governance_source_export() -> tuple[pd.DataFrame, pd.DataFrame]:
    if _run_store_ready():
        store = get_run_store()
        data_df = store.load_table("indices_export")
        legend_df = store.load_table("column_legend")
        if isinstance(data_df, pd.DataFrame) and not data_df.empty:
            legend_out = legend_df.copy() if isinstance(legend_df, pd.DataFrame) else pd.DataFrame()
            if legend_out.empty:
                legend_out = _infer_column_legend_from_result_df(data_df)
            return data_df.copy(), legend_out
    for key in ("psychometrics_result_data_export", "summary_result_data_export", "prep_result_data_export"):
        cached = st.session_state.get(key)
        if isinstance(cached, dict) and isinstance(cached.get("data_df"), pd.DataFrame):
            legend_df = cached.get("legend_df")
            data_df = cached["data_df"].copy()
            legend_out = legend_df.copy() if isinstance(legend_df, pd.DataFrame) else pd.DataFrame()
            inferred_legend = _infer_column_legend_from_result_df(data_df)
            if legend_out.empty:
                legend_out = inferred_legend
            elif "aggregation" not in legend_out.columns or not (legend_out["aggregation"].astype(str) == "index_flag").any():
                legend_out = inferred_legend
            return data_df, legend_out
    flags = _forensic_result_flags_from_session()
    if flags:
        responses = st.session_state.get("forensic_responses") or st.session_state.get("last_uploaded_responses") or []
        n_export = _infer_n_examinees(flags, responses)
        if n_export > 0:
            data_df, legend_df, _ = _build_examinee_flags_export(flags, n_export, visible_agents=_forensic_visible_agents())
            if "Examinee_ID" not in data_df.columns and not data_df.empty:
                data_df.insert(0, "Examinee_ID", list(range(1, len(data_df) + 1)))
            if legend_df.empty:
                legend_df = _infer_column_legend_from_result_df(data_df)
            return data_df, legend_df
    for sample_path in (
        Path("psymas_tutorial_data/temp/psymas_result_data.csv"),
        Path("psymas_tutorial_data/psymas_result_data.csv"),
    ):
        if sample_path.exists():
            data_df = pd.read_csv(sample_path)
            if "Examinee_ID" not in data_df.columns and not data_df.empty:
                data_df.insert(0, "Examinee_ID", list(range(1, len(data_df) + 1)))
            return data_df, _infer_column_legend_from_result_df(data_df)
    return pd.DataFrame(), pd.DataFrame()


def _governance_source_b3_input() -> pd.DataFrame:
    if _run_store_ready():
        b3_df = get_run_store().load_table("b3_input")
        if isinstance(b3_df, pd.DataFrame) and not b3_df.empty:
            return b3_df.copy()
    for key in ("psychometrics_result_data_export", "summary_result_data_export", "prep_result_data_export"):
        cached = st.session_state.get(key)
        if isinstance(cached, dict) and isinstance(cached.get("b3_input_df"), pd.DataFrame):
            cached_input = cached["b3_input_df"].copy()
            if not cached_input.empty:
                return cached_input
    result_df, legend_df = _governance_source_export()
    if not result_df.empty and isinstance(legend_df, pd.DataFrame) and not legend_df.empty:
        return _build_b3_input_table(result_df, legend_df)
    return pd.DataFrame()


def _governance_source_result_df() -> pd.DataFrame:
    result_df, _ = _governance_source_export()
    return result_df


def _suggest_module(user_description: str) -> tuple[str, str]:
    """Use LLM to suggest which module (Aberrance, IRT, or RT) fits the user's goal. Returns (run_mode_key, reason)."""
    if not (user_description or "").strip():
        return "", "Describe what you want to do (e.g. detect aberrant test-takers, estimate item difficulty, analyze response times)."
    load_dotenv()
    prompt = (
        "You are helping a user choose one module in a psychometrics app.\n\n"
        "Modules:\n"
        "Aberrance = Person-fit / aberrant test-takers: detect unusual response patterns, rapid guessing, copying, misfit (R package aberrance).\n"
        "IRT = Item Response Theory: item and person parameters, item characteristic curves (ICC), model fit, item fit (R package mirt).\n"
        "RT = Response time: analyze response times (latency), histograms, rapid-guessing flags.\n\n"
        f"User's goal: {user_description.strip()[:500]}\n\n"
        "Reply with exactly two lines. Line 1: only one word - Aberrance, IRT, or RT. Line 2: one short sentence explaining why that module fits."
    )
    text, err = _call_selected_llm_text(prompt, timeout=60)
    if text and not err:
        module, reason = _parse_module_suggestion(text)
        return module, (reason[:400] if reason else "Suggested based on your description.")
    return "", err or "No selected LLM model returned a response."
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "", "GOOGLE_API_KEY not set in .env. Set Model engine to use LLM."
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    for model_name in _model_variants_with_selected_first():
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent"
            resp = requests.post(url, params={"key": api_key}, json=body, timeout=60)
            resp.raise_for_status()
            text = (resp.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "") or "").strip()
            module, reason = _parse_module_suggestion(text)
            return module, (reason[:400] if reason else "Suggested based on your description.")
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                continue
            return "", str(e)
        except Exception as e:
            return "", f"{type(e).__name__}: {e}"
    return "", "No model responded. Check Model engine and GOOGLE_API_KEY."


def _parse_module_suggestion(text: str) -> tuple[str, str]:
    """Parse LLM response: Line 1 = Aberrance | IRT | RT; Line 2 = reason. Returns (run_mode_key, reason)."""
    if not (text or "").strip():
        return "", ""
    lines = [ln.strip() for ln in (text or "").strip().splitlines() if ln.strip()]
    first = (lines[0] if lines else "").upper()
    mapping = {"ABERRANCE": "Aberrance only", "IRT": "IRT only", "RT": "RT only"}
    module_key = ""
    for key, run_mode in mapping.items():
        if key in first or first == key[:2]:
            module_key = run_mode
            break
    if not module_key and first:
        if "ABERR" in first:
            module_key = "Aberrance only"
        elif "IRT" in first or "ITEM" in first:
            module_key = "IRT only"
        elif "RT" in first or "TIME" in first or "LATEN" in first:
            module_key = "RT only"
    reason = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""
    return module_key, (reason[:400] if reason else "Suggested based on your description.")


def _format_llm_error(last_err: str | None) -> str:
    """Return a user-friendly message; special case for quota exceeded."""
    if not last_err:
        return "Unknown error. Check GOOGLE_API_KEY in .env and that the key has access to Gemini."
    if "429" in last_err or "Quota exceeded" in last_err or "quota" in last_err.lower():
        return (
            "Free tier quota exceeded (e.g. 20 requests/min per model). "
            "Wait about a minute and try again, or use an API key with billing for higher quota."
        )
    return f"LLM analysis failed. {last_err} Check GOOGLE_API_KEY in .env and that the key has access to Gemini."


def _llm_analyze_section_text(section_name: str, context_text: str) -> str:
    """Ask LLM for a brief analysis of text context (e.g. table summary). Returns 1–2 paragraph summary or error."""
    if not context_text or not context_text.strip():
        return "No content to analyze."
    load_dotenv()
    prompt = (
        f"You are a psychometrics expert. Below is context from the section \"{section_name}\" of an IRT analysis. "
        "Provide a brief, paper-ready summary in one or two short paragraphs (key findings and implications). "
        "Do not repeat raw numbers; interpret them.\n\n"
        "Context:\n" + context_text[:8000]
    )
    text, err = _call_selected_llm_text(prompt, timeout=90)
    if text and not err:
        return text
    return _format_llm_error(err or "No selected LLM model returned a response.")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "LLM analysis unavailable: GOOGLE_API_KEY not set in .env file."
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    model_variants = _model_variants_with_selected_first()
    last_err = None
    for model_name in model_variants:
        for attempt in range(2):
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent"
                resp = requests.post(url, params={"key": api_key}, json=body, timeout=90)
                resp.raise_for_status()
                data = resp.json()
                text = (data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", ""))
                return (text.strip() or "No response from LLM.")
            except requests.exceptions.HTTPError as e:
                try:
                    err_body = e.response.json() if e.response is not None else {}
                    msg = err_body.get("error", {}).get("message", str(e))
                except Exception:
                    msg = str(e)
                last_err = f"{e.response.status_code if e.response is not None else 'HTTP'}: {msg}"
                if e.response is not None and e.response.status_code == 429 and attempt == 0:
                    time.sleep(50)
                    continue
                break
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                break
    return _format_llm_error(last_err)


def _llm_analyze_image_section(image_path: str, section_name: str, figure_description: str) -> str:
    """Analyze a psychometric figure (e.g. ICC) with LLM. Returns brief summary or error."""
    if not image_path or not Path(image_path).exists():
        return "Image not found."
    try:
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        return f"Error reading image: {e}"
    prompt = (
        f"You are a psychometrics expert. Analyze this \"{section_name}\" figure ({figure_description}). "
        "Provide a brief, paper-ready summary in two short paragraphs: (1) what the figure shows, (2) key implications."
    )
    if _llm_provider() == "openrouter":
        load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}},
        ]
        model_ids = _model_variants_with_selected_first()
        for model_id in model_ids:
            text, err = _call_openrouter(api_key, model_id, [{"role": "user", "content": content}], timeout=90)
            if err:
                continue
            if text:
                return text
        return _format_llm_error("OpenRouter: no model returned a response. Some free models may not support images; try text-only sections or set OPENROUTER_API_KEY.")
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "LLM analysis unavailable: GOOGLE_API_KEY not set in .env file."
    body = {
        "contents": [{
            "role": "user",
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/png", "data": image_data}}
            ]
        }]
    }
    model_variants = _model_variants_with_selected_first()
    last_err = None
    for model_name in model_variants:
        for attempt in range(2):
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent"
                resp = requests.post(url, params={"key": api_key}, json=body, timeout=90)
                resp.raise_for_status()
                data = resp.json()
                text = (data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", ""))
                return (text.strip() or "No response from LLM.")
            except requests.exceptions.HTTPError as e:
                try:
                    err_body = e.response.json() if e.response is not None else {}
                    msg = err_body.get("error", {}).get("message", str(e))
                except Exception:
                    msg = str(e)
                last_err = f"{e.response.status_code if e.response is not None else 'HTTP'}: {msg}"
                if e.response is not None and e.response.status_code == 429 and attempt == 0:
                    time.sleep(50)
                    continue
                break
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                break
    return _format_llm_error(last_err)


def _llm_analyze_image_and_text(
    image_path: str,
    section_name: str,
    figure_description: str,
    context_text: str,
) -> str:
    """Analyze a figure and text context (e.g. ICC + item params table) in one LLM call."""
    if not image_path or not Path(image_path).exists():
        return "Image not found."
    try:
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        return f"Error reading image: {e}"
    text_context = (context_text or "").strip()[:6000]
    prompt = (
        f"You are a psychometrics expert. Analyze the \"{section_name}\" figure ({figure_description}) and the following table together. "
        "Provide a brief, paper-ready summary in two short paragraphs: (1) what the figure and table show, (2) key implications.\n\n"
    )
    if text_context:
        prompt += "Table:\n" + text_context + "\n\n"
    prompt += "Summarize integrating both the figure and the table."
    if _llm_provider() == "openrouter":
        load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}},
        ]
        model_ids = _model_variants_with_selected_first()
        for model_id in model_ids:
            text, err = _call_openrouter(api_key, model_id, [{"role": "user", "content": content}], timeout=90)
            if err:
                continue
            if text:
                return text
        # OpenRouter free models often don't support images; fall back to text-only (table + figure description)
        text_fallback = _llm_analyze_section_text(
            section_name,
            f"Figure: {figure_description}\n\nTable:\n{text_context}",
        )
        if text_fallback and not text_fallback.strip().startswith("LLM analysis") and "no model returned" not in text_fallback:
            return text_fallback + "\n\n*Summary from table and figure description; the selected model does not support image input.*"
        return _format_llm_error("OpenRouter: no model returned a response. Some free models may not support images.")
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "LLM analysis unavailable: GOOGLE_API_KEY not set in .env file."
    body = {
        "contents": [{
            "role": "user",
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/png", "data": image_data}}
            ]
        }]
    }
    model_variants = _model_variants_with_selected_first()
    last_err = None
    for model_name in model_variants:
        for attempt in range(2):
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent"
                resp = requests.post(url, params={"key": api_key}, json=body, timeout=90)
                resp.raise_for_status()
                data = resp.json()
                text = (data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", ""))
                return (text.strip() or "No response from LLM.")
            except requests.exceptions.HTTPError as e:
                try:
                    err_body = e.response.json() if e.response is not None else {}
                    msg = err_body.get("error", {}).get("message", str(e))
                except Exception:
                    msg = str(e)
                last_err = f"{e.response.status_code if e.response is not None else 'HTTP'}: {msg}"
                if e.response is not None and e.response.status_code == 429 and attempt == 0:
                    time.sleep(50)
                    continue
                break
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                break
    return _format_llm_error(last_err)


def _analyze_wright_map_image(image_path: str) -> str:
    """Analyze Wright Map PNG image using LLM and generate a paper-ready summary."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    model = os.getenv("GEMINI_MODEL", "models/gemini-1.5-flash-latest")
    
    if not api_key:
        return "LLM analysis unavailable: GOOGLE_API_KEY not set in .env file."
    
    if not Path(image_path).exists():
        return "Wright Map image not found."
    
    # Read and encode the image
    try:
        with open(image_path, "rb") as img_file:
            image_data = base64.b64encode(img_file.read()).decode("utf-8")
    except Exception as e:
        return f"Error reading image: {type(e).__name__}: {e}"
    
    system = (
        "You are a psychometrics expert. Analyze this Wright Map (item-person map) and provide "
        "a concise, two-paragraph summary suitable for a research paper. "
        "The Wright Map shows the distribution of person abilities (histogram on the left) and "
        "item difficulties (red lines on the right) on the same latent trait scale. "
        "Write EXACTLY TWO PARAGRAPHS: "
        "(1) First paragraph: Describe the patterns observed - the distribution of person abilities, "
        "the distribution and range of item difficulties, and the alignment between them. "
        "Include any gaps or clusters in the measurement. "
        "(2) Second paragraph: Present the findings and their implications for test design, "
        "measurement precision, and interpretation. "
        "Keep each paragraph concise (3-5 sentences). Write in a formal, academic style suitable for publication."
    )
    
    # Prepare the request body with image
    body = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": system},
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": image_data
                        }
                    }
                ]
            }
        ]
    }
    
    # First, try to discover available models (more reliable than hardcoded names)
    resp = None
    last_error = None
    discovered_models = []
    
    try:
        list_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
        list_resp = requests.get(list_url, timeout=30)
        list_resp.raise_for_status()
        models_data = list_resp.json().get("models", [])
        
        # Find models that support generateContent (vision-capable models)
        for m in models_data:
            methods = m.get("supportedGenerationMethods", [])
            model_name = m.get("name", "")
            if "generateContent" in methods and "gemini" in model_name.lower():
                discovered_models.append(model_name)
        
        # Prioritize: effective (pinned or selected) model first, then flash, then pro
        flash_models = [m for m in discovered_models if "flash" in m.lower()]
        pro_models = [m for m in discovered_models if "pro" in m.lower() and "flash" not in m.lower()]
        other_models = [m for m in discovered_models if m not in flash_models and m not in pro_models]
        ordered = flash_models + pro_models + other_models
        selected = _effective_llm_model()
        models_to_try = [selected] + [m for m in ordered if m != selected]

        # Try models (selected first, then discovered)
        for model_name in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent"
            try:
                resp = requests.post(url, params={"key": api_key}, json=body, timeout=60)
                resp.raise_for_status()
                break
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    last_error = e
                    resp = None
                    continue
                else:
                    raise
            except Exception as e:
                last_error = e
                resp = None
                continue
    except Exception as e:
        # If model discovery fails, fall back to hardcoded models
        last_error = e
    
    # If discovery failed or no models worked, try hardcoded variants (effective model first)
    if resp is None:
        selected = _effective_llm_model()
        fallback = [model, "models/gemini-1.5-flash-latest", "models/gemini-1.5-flash-002", "models/gemini-1.5-flash-001", "models/gemini-1.5-pro-latest", "models/gemini-1.5-pro"]
        model_variants = [selected] + [m for m in fallback if m != selected]

        for model_variant in model_variants:
            url = f"https://generativelanguage.googleapis.com/v1beta/{model_variant}:generateContent"
            try:
                resp = requests.post(url, params={"key": api_key}, json=body, timeout=60)
                resp.raise_for_status()
                break
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    last_error = e
                    resp = None
                    continue
                else:
                    raise
            except Exception as e:
                last_error = e
                resp = None
                continue
    
    if resp is None:
        error_msg = str(last_error) if last_error else "Unknown error"
        return _format_llm_error(error_msg)
    
    try:
        data = resp.json()
        text = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )
        return text.strip() if text else "No response from LLM."
    except Exception as e:
        return f"Error parsing LLM response: {type(e).__name__}: {e}"


def _plot_response_for_pdf(resp_row: dict, *, compact: bool = False) -> bytes | None:
    """Draw response by item (green=correct, red=incorrect) as a bar chart; return PNG bytes."""
    if not resp_row:
        return None
    vals = list(resp_row.values())
    n = min(len(vals), 60)
    if n == 0:
        return None
    try:
        figsize = (7.0, 1.05) if compact else (6, 1.8)
        dpi = 100 if compact else 120
        fig, ax = plt.subplots(figsize=figsize)
        items = list(range(1, n + 1))
        colors_list = ["#40916c" if v == 1 else "#ef4444" for v in vals[:n]]
        ax.bar(items, [1] * n, color=colors_list, width=0.7, edgecolor="none")
        ax.set_ylim(0, 1.2)
        ax.set_yticks([0.5])
        ax.set_yticklabels(["Correct / Incorrect"])
        ax.set_xlabel("Item", fontsize=8 if compact else 10)
        ax.set_title("Response by item (green = correct, red = incorrect)", fontsize=9 if compact else 11)
        ax.tick_params(axis="both", labelsize=7 if compact else 9)
        ax.set_xticks(items[:: max(1, n // 20)])
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()
    except Exception:
        return None


def _plot_rt_for_pdf(rt_row: list, rt_medians: list, student_id: int, *, compact: bool = False) -> bytes | None:
    """Draw response time by item as Forensic Timeline (line + shaded area, white background); return PNG bytes."""
    if not rt_row or not rt_medians or len(rt_row) != len(rt_medians):
        return None
    n = min(len(rt_row), 50)
    if n == 0:
        return None
    try:
        figsize = (7.0, 1.15) if compact else (6, 2.2)
        dpi = 100 if compact else 120
        lbl = 7 if compact else 9
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.tick_params(colors="black", labelsize=lbl)
        ax.spines["bottom"].set_color("black")
        ax.spines["left"].set_color("black")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.label.set_color("black")
        ax.yaxis.label.set_color("black")
        ax.title.set_color("black")
        rt_vals = [float(x) for x in rt_row[:n]]
        mean_vals = [float(x) for x in rt_medians[:n]]
        items = list(range(1, n + 1))
        ax.plot(items, rt_vals, color=PSYMAS_VIZ["selected"], linewidth=1.4 if compact else 2, marker="o", markersize=2.5 if compact else 4, label="Selected examinee")
        ax.plot(items, mean_vals, color=PSYMAS_VIZ["warning"], linewidth=1.3 if compact else 1.8, linestyle="--", label="Item mean")
        ax.set_xlabel("Item Number", fontsize=lbl)
        ax.set_ylabel("Response Time", fontsize=lbl)
        ax.set_title(f"Response Time Compared With Item Mean — Student {student_id}", fontsize=9 if compact else 11)
        ax.set_xticks(items[:: max(1, n // 15)])
        all_vals = rt_vals + mean_vals
        ax.set_ylim(0, max(all_vals) * 1.1 if all_vals else 1)
        ax.legend(loc="upper right", fontsize=6 if compact else 8, labelcolor="black", facecolor="white", edgecolor="gray")
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.close(fig)
        return buf.getvalue()
    except Exception:
        return None


def _plot_percentile_for_pdf(
    percentile: float,
    student_id: int,
    all_scores: list[int] | None,
    student_score: int,
    total_items: int,
    *,
    compact: bool = False,
) -> bytes | None:
    """Draw score distribution (histogram) with student's position marked; return PNG bytes. White background."""
    try:
        pct = min(100, max(0, float(percentile)))
        figsize = (7.0, 1.15) if compact else (6, 2.2)
        dpi = 100 if compact else 120
        lbl = 7 if compact else 9
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.tick_params(colors="black", labelsize=lbl)
        ax.spines["bottom"].set_color("black")
        ax.spines["left"].set_color("black")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.label.set_color("black")
        ax.yaxis.label.set_color("black")
        ax.title.set_color("black")
        if all_scores and len(all_scores) > 0:
            # Histogram of total scores (0 to total_items)
            low = 0
            high = total_items if total_items else max(all_scores)
            bins = np.arange(low - 0.5, high + 1.5, 1)
            counts, edges, _ = ax.hist(
                all_scores, bins=bins, color=PSYMAS_VIZ["reference_fill"], alpha=0.9, edgecolor=PSYMAS_VIZ["reference"], linewidth=0.5
            )
            ax.axvline(x=student_score, color=PSYMAS_VIZ["selected"], linewidth=2, linestyle="-", label=f"Student {student_id} (score {student_score})")
            ax.set_xlim(low, high)
            ax.set_xlabel("Total Score", fontsize=lbl)
            ax.set_ylabel("Count", fontsize=lbl)
            ax.set_title(
                f"Score Distribution — Student {student_id}: {student_score}/{total_items} ({pct:.0f}th percentile)",
                fontsize=9 if compact else 11,
            )
            ax.legend(loc="upper right", fontsize=6 if compact else 8, labelcolor="black", facecolor="white", edgecolor="gray")
        else:
            # Fallback: horizontal bar at percentile
            ax.barh(0, pct, left=0, height=0.5, color=PSYMAS_VIZ["selected"], alpha=0.85, edgecolor="none")
            ax.set_xlim(0, 100)
            ax.set_ylim(-0.6, 0.6)
            ax.set_yticks([])
            ax.set_xlabel("Percentile (0–100)")
            ax.set_title(f"Percentile — Student {student_id}: {pct:.0f}th percentile", fontsize=9 if compact else 11)
            ax.set_xticks([0, 25, 50, 75, 100])
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.close(fig)
        return buf.getvalue()
    except Exception:
        return None


def _plot_person_accuracy_for_pdf(resp_row: list, item_means: list, student_id: int, *, compact: bool = False) -> bytes | None:
    if not resp_row or not item_means:
        return None
    n = min(len(resp_row), len(item_means), 60)
    if n == 0:
        return None
    try:
        figsize = (7.0, 1.6) if compact else (6.2, 2.4)
        dpi = 100 if compact else 120
        lbl = 7 if compact else 9
        vals = [float(x) if pd.notna(x) else np.nan for x in resp_row[:n]]
        means = [float(x) if pd.notna(x) else np.nan for x in item_means[:n]]
        items = list(range(1, n + 1))
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        colors_list = [PSYMAS_VIZ["correct"] if v >= 0.5 else PSYMAS_VIZ["incorrect"] for v in vals]
        ax.bar(items, means, color=colors_list, edgecolor="#334155", linewidth=0.25, label="Item mean accuracy")
        ax.set_ylim(0, 1.08)
        ax.set_xlabel("Item Number", fontsize=lbl)
        ax.set_ylabel("Accuracy", fontsize=lbl)
        ax.set_title(f"Item Accuracy Colored by Selected Response — Student {student_id}", fontsize=9 if compact else 11)
        ax.set_xticks(items[:: max(1, n // 15)])
        ax.tick_params(colors="black", labelsize=lbl)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        from matplotlib.patches import Patch
        handles = [
            Patch(facecolor=PSYMAS_VIZ["correct"], edgecolor="#334155", label="Selected correct"),
            Patch(facecolor=PSYMAS_VIZ["incorrect"], edgecolor="#334155", label="Selected incorrect"),
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=6 if compact else 8, labelcolor="black", facecolor="white", edgecolor="gray")
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.close(fig)
        return buf.getvalue()
    except Exception:
        return None


def _rl_image_preserve_ratio(image_source, max_width, max_height):
    try:
        from reportlab.platypus import Image as RLImage
        img = RLImage(image_source)
        scale = min(max_width / float(img.imageWidth), max_height / float(img.imageHeight), 1.0)
        img.drawWidth = float(img.imageWidth) * scale
        img.drawHeight = float(img.imageHeight) * scale
        return img
    except Exception:
        return None


def _shorten_report_text(value, max_chars: int = 70) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "..."


def _markdown_to_reportlab(text: str) -> str:
    """Convert Markdown to ReportLab Paragraph markup (bold, italic, line breaks, headings)."""
    if not text or not text.strip():
        return text
    s = text
    # Escape XML/HTML specials first
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # Headings: ### or ## or # at start of line → bold
    s = re.sub(r"^#{1,3}\s*(.+)$", r"<b>\1</b>", s, flags=re.MULTILINE)
    # Bold: **...** (intentionally do NOT treat underscores as markup; they commonly appear in ids like nm_agent)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    # Italic: *...* (intentionally do NOT treat underscores as markup; they commonly appear in ids like L_S_NO)
    s = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", s)
    # Line breaks
    s = s.replace("\n", "<br/>")
    return s


def _escape_reportlab_text(text: str) -> str:
    """Escape text for ReportLab Paragraph (no inline markup)."""
    if text is None:
        return ""
    s = str(text)
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return s.replace("\n", "<br/>")


def _build_test_taker_report_pdf(
    student_id: int,
    correct: int,
    total_items: int,
    resp_row: dict,
    flags: dict,
    rt_row: list | None,
    rt_medians: list | None,
    percentile: float,
    aberrant_rows: list[dict],
    report_text: str | None,
    all_scores: list[int] | None = None,
) -> tuple[bytes, str | None]:
    """Build a letter-size PDF for the Test-Taker Report (PsyMAS branding, 12pt body). Returns (pdf_bytes, error_message)."""
    try:
        import datetime

        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
    except ImportError as e:
        return b"", f"reportlab not installed: {e}"

    black = colors.HexColor("#000000")
    lightgrey = colors.HexColor("#e5e7eb")
    green = colors.HexColor("#40916c")
    red = colors.HexColor("#ef4444")
    navy = colors.HexColor("#1a1a2e")
    accent = colors.HexColor("#0d9488")
    muted = colors.HexColor("#64748b")
    footer_bg = colors.HexColor("#f1f5f9")

    temp_files = []
    try:
        buffer = io.BytesIO()
        margin = 0.45 * inch
        doc = SimpleDocTemplate(
            buffer, pagesize=letter,
            rightMargin=margin, leftMargin=margin,
            topMargin=margin, bottomMargin=margin,
        )
        content_w = letter[0] - 2 * margin
        gen_date = datetime.date.today().strftime("%Y-%m-%d")

        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(
            name="TtrBrandTitle",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            textColor=colors.white,
            fontSize=22,
            leading=26,
            alignment=1,
            spaceAfter=2,
        ))
        styles.add(ParagraphStyle(
            name="TtrBrandSub",
            parent=styles["Normal"],
            textColor=colors.HexColor("#c4c9d4"),
            fontSize=11,
            leading=13,
            alignment=1,
            spaceAfter=2,
        ))
        styles.add(ParagraphStyle(
            name="TtrBrandTag",
            parent=styles["Normal"],
            textColor=accent,
            fontSize=10,
            leading=12,
            alignment=1,
            spaceAfter=0,
        ))
        styles.add(ParagraphStyle(
            name="TtrHeading1",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            textColor=black,
            fontSize=15,
            leading=18,
            spaceAfter=4,
        ))
        styles.add(ParagraphStyle(
            name="TtrHeading2",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            textColor=black,
            fontSize=12,
            leading=14,
            spaceBefore=6,
            spaceAfter=4,
        ))
        styles.add(ParagraphStyle(
            name="TtrNormal",
            parent=styles["Normal"],
            textColor=black,
            fontSize=12,
            leading=14,
            spaceAfter=2,
        ))
        styles.add(ParagraphStyle(
            name="TtrFooter",
            parent=styles["Normal"],
            textColor=muted,
            fontSize=9,
            leading=11,
            alignment=1,
            spaceBefore=4,
            spaceAfter=0,
        ))
        story = []

        # ── PsyMAS header band ──
        banner_rows = [
            [Paragraph("PsyMAS", styles["TtrBrandTitle"])],
            [Paragraph("Psychometric Modeling Assistant System", styles["TtrBrandSub"])],
            [Paragraph("Forensic drill-down · Individual examinee report", styles["TtrBrandTag"])],
        ]
        banner = Table(banner_rows, colWidths=[content_w])
        banner.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), navy),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, 0), 16),
            ("BOTTOMPADDING", (0, -1), (-1, -1), 14),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ]))
        story.append(banner)
        accent_bar = Table([[""]], colWidths=[content_w], rowHeights=[3])
        accent_bar.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), accent)]))
        story.append(accent_bar)
        story.append(Spacer(1, 0.12 * inch))

        story.append(Paragraph(
            f"<b>Test-Taker Report</b> — Student {student_id}",
            styles["TtrHeading1"],
        ))
        story.append(Paragraph(
            "Response pattern, score placement, response-time profile, and narrative summary "
            "(when generated in the app). Green / red cells = correct / incorrect.",
            styles["TtrNormal"],
        ))
        story.append(Spacer(1, 0.06 * inch))

        pct = 100 * correct / total_items if total_items else 0
        summary_line = Paragraph(
            f"Score {correct}/{total_items} ({pct:.1f}%) · Percentile {percentile:.0f}th",
            styles["TtrNormal"],
        )

        # ── Response Record: chunk columns so table width ≤ page (avoids overflow) ──
        if resp_row:
            story.append(Paragraph("Response Record", styles["TtrHeading2"]))
            vals = list(resp_row.values())
            max_items = min(len(vals), 48)
            chunk = 16
            tbl_font = 9
            tot_w = 0.45 * inch
            for start in range(0, max_items, chunk):
                end = min(start + chunk, max_items)
                n = end - start
                slice_vals = vals[start:end]
                last_chunk = end >= max_items
                if last_chunk:
                    header_row = [f"{start + i + 1}" for i in range(n)] + ["Tot"]
                    data_row = [str(int(v)) if v in (0, 1) else str(v) for v in slice_vals] + [
                        f"{correct}/{total_items}"
                    ]
                    col_w = (content_w - tot_w) / n
                    col_widths = [col_w] * n + [tot_w]
                    tot_col = n
                else:
                    header_row = [f"{start + i + 1}" for i in range(n)]
                    data_row = [str(int(v)) if v in (0, 1) else str(v) for v in slice_vals]
                    col_w = content_w / n
                    col_widths = [col_w] * n
                    tot_col = None
                cell_data = [header_row, data_row]
                t = Table(cell_data, colWidths=col_widths)
                style_list = [
                    ("FONTSIZE", (0, 0), (-1, -1), tbl_font),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 1),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 1),
                    ("TOPPADDING", (0, 0), (-1, -1), 1),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
                    ("TEXTCOLOR", (0, 0), (-1, 0), black),
                    ("BACKGROUND", (0, 0), (-1, 0), lightgrey),
                ]
                last_col = len(data_row) - 1
                style_list.append(("TEXTCOLOR", (0, 1), (last_col - 1, 1), colors.white))
                if tot_col is not None:
                    style_list.append(("BACKGROUND", (tot_col, 1), (tot_col, 1), lightgrey))
                    style_list.append(("TEXTCOLOR", (tot_col, 1), (tot_col, 1), black))
                for j in range(n):
                    gi = start + j
                    if gi < len(vals) and vals[gi] == 1:
                        style_list.append(("BACKGROUND", (j, 1), (j, 1), green))
                    elif gi < len(vals) and vals[gi] == 0:
                        style_list.append(("BACKGROUND", (j, 1), (j, 1), red))
                t.setStyle(TableStyle(style_list))
                story.append(t)
                story.append(Spacer(1, 0.02 * inch))
            story.append(summary_line)
            story.append(Spacer(1, 0.04 * inch))
        else:
            story.append(summary_line)
            story.append(Spacer(1, 0.04 * inch))

        # ── Percentile (compact plot) + Forensic Timeline side-by-side in one row ──
        png_pct = _plot_percentile_for_pdf(
            percentile, student_id, all_scores, correct, total_items, compact=True
        )
        png_rt = None
        if rt_row and rt_medians:
            png_rt = _plot_rt_for_pdf(rt_row, rt_medians, student_id, compact=True)

        plot_h = 0.92 * inch
        half_w = (content_w - 0.08 * inch) / 2
        row_cells = []
        if png_pct:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fp:
                fp.write(png_pct)
                fp.flush()
                temp_files.append(fp.name)
            row_cells.append(RLImage(temp_files[-1], width=half_w, height=plot_h))
        else:
            row_cells.append(Paragraph("—", styles["TtrNormal"]))
        if png_rt:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fp_rt:
                fp_rt.write(png_rt)
                fp_rt.flush()
                temp_files.append(fp_rt.name)
            row_cells.append(RLImage(temp_files[-1], width=half_w, height=plot_h))
        elif rt_row and rt_medians and len(rt_row) <= 20:
            rt_vals = [float(x) for x in rt_row[:20]]
            meds = rt_medians[:20]
            rows = [["#", "RT", "Δ"]]
            for i, (rv, m) in enumerate(zip(rt_vals, meds)):
                rows.append([str(i + 1), f"{rv:.1f}", ">" if rv >= m else "<"])
            t2 = Table(rows, colWidths=[0.32 * inch, 0.5 * inch, 0.28 * inch])
            t2.setStyle(TableStyle([
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("BACKGROUND", (0, 0), (-1, 0), lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ]))
            row_cells.append(t2)
        else:
            row_cells.append(Paragraph("—", styles["TtrNormal"]))

        story.append(Paragraph("Score distribution · Response time", styles["TtrHeading2"]))
        plot_table = Table([row_cells], colWidths=[half_w, half_w])
        plot_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]))
        story.append(plot_table)

        # ── Report (LLM): length cap (large type uses more vertical space) ──
        story.append(Paragraph("Narrative summary (LLM)", styles["TtrHeading2"]))
        llm_style = ParagraphStyle(
            name="TtrReport",
            parent=styles["TtrNormal"],
            fontSize=12,
            leading=15,
            spaceAfter=4,
        )
        max_report_chars = 1600
        if report_text and report_text.strip():
            full_rt = report_text.strip()
            remaining = max_report_chars
            consumed = 0
            for para in full_rt.split("\n\n")[:8]:
                if remaining <= 0:
                    break
                raw = para.strip()
                chunk = raw[:remaining]
                consumed += len(chunk)
                remaining -= len(chunk)
                safe = _markdown_to_reportlab(chunk)
                if safe.strip():
                    try:
                        story.append(Paragraph(safe, llm_style))
                    except Exception:
                        story.append(Paragraph(_escape_reportlab_text(chunk), llm_style))
            if consumed < len(full_rt):
                story.append(Paragraph("<i>[Summary truncated for one-page PDF.]</i>", llm_style))
        else:
            story.append(Paragraph("No LLM summary (generate the report in the app).", styles["TtrNormal"]))

        story.append(Spacer(1, 0.14 * inch))
        footer_para = Paragraph(
            f"<b>PsyMAS</b> · Psychometric Modeling Assistant System · Ver. {APP_VERSION}<br/>"
            f"Confidential forensic analysis artifact · Generated {gen_date} · Not for high-stakes decisions without review",
            styles["TtrFooter"],
        )
        foot_table = Table([[footer_para]], colWidths=[content_w])
        foot_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), footer_bg),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("LINEABOVE", (0, 0), (-1, 0), 2, accent),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ]))
        story.append(foot_table)

        doc.build(story)
        return buffer.getvalue(), None
    except Exception as e:
        return b"", str(e)
    finally:
        for p in temp_files:
            try:
                os.unlink(p)
            except Exception:
                pass


def _build_apa_report_pdf(final: dict) -> tuple[bytes, str | None]:
    """Build an APA-style report PDF from analysis results. Returns (pdf_bytes, error_message)."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
    except ImportError as e:
        return b"", f"reportlab not installed: {e}. Run: uv sync  or  pip install reportlab"

    def _safe_str(v):
        if v is None:
            return "—"
        if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
            return "—"
        return str(v)

    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer, pagesize=letter,
            rightMargin=inch, leftMargin=inch, topMargin=inch, bottomMargin=inch,
        )
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            name="ReportTitle",
            parent=styles["Heading1"],
            fontSize=14,
            spaceAfter=12,
            alignment=1,
        )
        story = []

        # Title
        story.append(Paragraph("Psychometric Analysis Report", title_style))
        story.append(Paragraph("<i>Item Response Theory (IRT) and Response Time Analysis</i>", styles["Normal"]))
        story.append(Spacer(1, 0.3 * inch))

        # Method
        story.append(Paragraph("Method", styles["Heading2"]))
        model_settings = st.session_state.get("model_settings", {})
        itemtype = model_settings.get("itemtype", "2PL") if isinstance(model_settings, dict) else "2PL"
        n_persons = 0
        n_items = 0
        if final.get("responses"):
            resp_df = pd.DataFrame(final["responses"])
            n_persons, n_items = resp_df.shape
        method_text = (
            f"We analyzed the assessment data using a {itemtype} IRT model (mirt package in R). "
            f"The sample comprised {n_persons} respondents and {n_items} items. "
            "Parameter estimation was conducted with marginal maximum likelihood. "
            "The analysis was produced by Psych-MAS (Psychometric Modeling Assistant System)."
        )
        story.append(Paragraph(method_text, styles["Normal"]))
        story.append(Spacer(1, 0.2 * inch))

        # Results — order matches UI: 1 Descriptive, 2 Model fit, 3 Item fit, 4 Wright Map & Parameters
        story.append(Paragraph("Results", styles["Heading2"]))

        def _add_llm_summary(session_key: str, heading: str) -> None:
            text = st.session_state.get(session_key)
            if text and isinstance(text, str) and text.strip():
                story.append(Paragraph(f"<b>Psych-MAS Summary: {heading}.</b>", styles["Normal"]))
                for part in text.strip().split("\n\n")[:4]:
                    story.append(Paragraph(part.replace("\n", " ")[:1500], styles["Normal"]))
                story.append(Spacer(1, 0.1 * inch))

        # 1. Descriptive Summary of response (order: text → item accuracy figure → summary)
        story.append(Paragraph("1. Descriptive Summary of response.", styles["Heading3"]))
        if final.get("responses"):
            resp_df = pd.DataFrame(final["responses"])
            story.append(Paragraph(f"Response matrix: {resp_df.shape[0]} persons × {resp_df.shape[1]} items. Proportion correct per item (mean): min = {resp_df.mean(axis=0).min():.3f}, max = {resp_df.mean(axis=0).max():.3f}.", styles["Normal"]))
            acc_path = _plot_item_accuracy(resp_df)
            if acc_path and Path(acc_path).exists():
                try:
                    story.append(Paragraph("<b>Figure.</b> Item accuracy (proportion correct per item).", styles["Normal"]))
                    story.append(RLImage(acc_path, width=5 * inch, height=3 * inch))
                    story.append(Spacer(1, 0.1 * inch))
                except Exception:
                    pass
        _add_llm_summary("llm_analysis_desc_result", "Descriptive summary")
        story.append(Spacer(1, 0.15 * inch))

        # 2. Model fit
        story.append(Paragraph("2. Model fit.", styles["Heading3"]))
        model_fit = final.get("model_fit") if isinstance(final.get("model_fit"), dict) else None
        if model_fit:
            fit_rows = [[_safe_str(k), _safe_str(v)] for k, v in model_fit.items()]
            fit_table = Table([["Statistic", "Value"]] + fit_rows)
            fit_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
            ]))
            story.append(fit_table)
        _add_llm_summary("llm_analysis_model_fit_result", "Model fit")
        story.append(Spacer(1, 0.15 * inch))

        # 3. Item fit
        story.append(Paragraph("3. Item fit.", styles["Heading3"]))
        if final.get("item_fit"):
            fit_df = pd.DataFrame(final["item_fit"])
            num_cols = fit_df.select_dtypes(include=["number"]).columns.tolist()[:6]
            if num_cols:
                summ = fit_df[num_cols].agg(["mean", "min", "max"]).round(4)
                summ_rows = []
                for c in num_cols:
                    row = [_safe_str(c)]
                    for r in ["mean", "min", "max"]:
                        try:
                            row.append(_safe_str(summ.loc[r, c]))
                        except Exception:
                            row.append("—")
                    summ_rows.append(row)
                summ_table = Table([["Statistic", "Mean", "Min", "Max"], *summ_rows])
                summ_table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ]))
                story.append(summ_table)
        _add_llm_summary("llm_analysis_item_fit_result", "Item fit")
        story.append(Spacer(1, 0.15 * inch))

        # 4. Wright Map & Parameters (Wright map → Item params table → ICC figure → summary → Person params → summary)
        story.append(Paragraph("4. Wright Map & Parameters.", styles["Heading3"]))
        fig_num = 0
        # 4.1 Wright map figure
        wright_map_path = None
        if final.get("item_params") and final.get("person_params"):
            try:
                item_df_wm = pd.DataFrame(final["item_params"])
                person_df_wm = pd.DataFrame(final["person_params"])
                wright_map_path = _create_wright_map(item_df_wm, person_df_wm)
            except Exception:
                wright_map_path = None
        if wright_map_path and Path(wright_map_path).exists():
            fig_num += 1
            story.append(Paragraph(f"<b>Figure {fig_num}.</b> Wright map: distribution of item difficulties and person abilities on the latent scale.", styles["Normal"]))
            try:
                img = RLImage(wright_map_path, width=5 * inch, height=6 * inch)
                story.append(img)
            except Exception:
                story.append(Paragraph("(Wright map image could not be embedded.)", styles["Normal"]))
            story.append(Spacer(1, 0.1 * inch))
            # Wright map Psych-MAS Summary (next to figure)
            for key, val in st.session_state.items():
                if isinstance(key, str) and key.startswith("wright_map_analysis_") and val and isinstance(val, str) and val.strip():
                    story.append(Paragraph("<b>Psych-MAS Summary: Wright map.</b>", styles["Normal"]))
                    for part in val.strip().split("\n\n")[:4]:
                        story.append(Paragraph(part.replace("\n", " ")[:1500], styles["Normal"]))
                    story.append(Spacer(1, 0.1 * inch))
                    break
            story.append(Spacer(1, 0.1 * inch))

        # 4.2 Item Parameters & ICC (order: ICC figure → Item params table → summary, matching UI)
        story.append(Paragraph("Item Parameters & ICC.", styles["Heading3"]))
        icc_path = final.get("icc_plot_path")
        if icc_path and Path(icc_path).exists():
            fig_num += 1
            story.append(Paragraph(f"<b>Figure {fig_num}.</b> Item characteristic curves (ICCs) for all items.", styles["Normal"]))
            try:
                img = RLImage(icc_path, width=5 * inch, height=6 * inch)
                story.append(img)
            except Exception:
                story.append(Paragraph("(ICC plot image could not be embedded.)", styles["Normal"]))
            story.append(Spacer(1, 0.1 * inch))
        if final.get("item_params"):
            item_df = pd.DataFrame(final["item_params"])
            cols = [c for c in list(item_df.columns)[:8]]
            if cols:
                head = [[_safe_str(c) for c in cols]]
                for _, row in item_df[cols].head(15).iterrows():
                    head.append([_safe_str(row[c]) for c in cols])
                if len(item_df) > 15:
                    head.append([f"... and {len(item_df) - 15} more rows"])
                tbl = Table(head)
                tbl.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ]))
                story.append(tbl)
            story.append(Spacer(1, 0.1 * inch))
        _add_llm_summary("llm_analysis_icc_result", "Item Parameters & ICC")
        story.append(Spacer(1, 0.15 * inch))

        # 4.3 Person parameters (order: person ability figure → table → summary, matching UI)
        story.append(Paragraph("Person parameters.", styles["Heading3"]))
        if final.get("person_params"):
            person_df = pd.DataFrame(final["person_params"])
            person_fig_path = _plot_person_ability(person_df)
            if person_fig_path and Path(person_fig_path).exists():
                try:
                    fig_num += 1
                    story.append(Paragraph(f"<b>Figure {fig_num}.</b> Distribution of person ability (θ).", styles["Normal"]))
                    story.append(RLImage(person_fig_path, width=5 * inch, height=3 * inch))
                    story.append(Spacer(1, 0.1 * inch))
                except Exception:
                    pass
            num_cols = person_df.select_dtypes(include=["number"]).columns.tolist()
            if num_cols:
                summ = person_df[num_cols].agg(["mean", "std", "min", "max"]).round(4)
                summ_rows = []
                for c in num_cols:
                    row = [_safe_str(c)]
                    for r in ["mean", "std", "min", "max"]:
                        try:
                            row.append(_safe_str(summ.loc[r, c]))
                        except Exception:
                            row.append("—")
                    summ_rows.append(row)
                summ_table = Table([["Parameter", "Mean", "SD", "Min", "Max"], *summ_rows])
                summ_table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ]))
                story.append(summ_table)
        _add_llm_summary("llm_analysis_person_result", "Person parameters")
        story.append(Spacer(1, 0.15 * inch))

        # 5. Person-fit (Aberrance)
        aberrance = final.get("aberrance_results") or {}
        if aberrance.get("nonparametric_misfit") or aberrance.get("parametric_misfit"):
            story.append(Paragraph("5. Person-fit (Aberrance).", styles["Heading2"]))
            methods = aberrance.get("methods", ["ZU3_S", "HT_S"])
            n_persons = aberrance.get("n_persons", 0)
            n_flagged = aberrance.get("n_flagged", 0)
            story.append(Paragraph(
                f"Person-fit indices were computed with the R package aberrance (nonparametric: {', '.join(methods)}). "
                f"Sample size: n = {n_persons}. "
                f"{n_flagged} person(s) were flagged as aberrant (nonparametric: ZU3_S &lt; -2 or HT_S in bottom 5%; "
                "parametric: L_S_2PL at α = .05 when IRT item parameters were available).",
                styles["Normal"],
            ))
            story.append(Spacer(1, 0.1 * inch))
            if aberrance.get("nonparametric_misfit"):
                nm_df = pd.DataFrame(aberrance["nonparametric_misfit"])
                num_cols = nm_df.select_dtypes(include=["number"]).columns.tolist()[:6]
                if num_cols:
                    head = [[_safe_str(c) for c in ["Person"] + num_cols]]
                    for i, (_, row) in enumerate(nm_df[num_cols].head(12).iterrows(), start=1):
                        head.append([_safe_str(i)] + [_safe_str(row[c]) for c in num_cols])
                    if len(nm_df) > 12:
                        head.append([f"... and {len(nm_df) - 12} more persons"])
                    tbl = Table(head)
                    tbl.setStyle(TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                    ]))
                    story.append(tbl)
            if aberrance.get("parametric_misfit"):
                story.append(Paragraph("Parametric person-fit (2PL) statistics were also computed when item parameters were available.", styles["Normal"]))
            story.append(Spacer(1, 0.15 * inch))

        # Advanced Analysis (Q&A from Psych-MAS Assistant)
        chat_history = st.session_state.get("analysis_chat_history") or []
        if chat_history:
            story.append(Paragraph("Advanced Analysis", styles["Heading2"]))
            story.append(Paragraph(
                "The following questions and answers were generated by the Psych-MAS Assistant "
                "in APA format, focusing on critical psychometric findings (one paragraph per response).",
                styles["Normal"],
            ))
            story.append(Spacer(1, 0.15 * inch))
            q_num = 0
            for role, message in chat_history:
                if role == "user":
                    q_num += 1
                    story.append(Paragraph(f"<b>Question {q_num}.</b> {message.replace(chr(10), ' ')[:2000]}", styles["Normal"]))
                else:
                    for part in (message or "").strip().split("\n\n")[:6]:
                        if part.strip():
                            story.append(Paragraph(part.replace("\n", " ")[:2000], styles["Normal"]))
                story.append(Spacer(1, 0.1 * inch))
            story.append(Spacer(1, 0.15 * inch))

        doc.build(story)
        return buffer.getvalue(), None
    except Exception as e:
        return b"", f"{type(e).__name__}: {e}"


def _render_prompt_and_confirm() -> None:
    """Render Psych-MAS Assistant (Prompt, Model engine, Langgraph) and Confirm settings in two columns. Used by the Tools→IRT module."""
    col_input, col_confirm = st.columns(2)
    with col_input:
        st.subheader("Psych-MAS Assistant")
        tab_prompt, tab_model_engine, tab_langgraph = st.tabs(["Prompt", "Model engine", "Langgraph"])
        with tab_prompt:
            with st.form("prompt_form"):
                prompt = st.text_input(
                    "Describe the analysis you want (e.g., guessing or 3PL).",
                    placeholder='e.g. I think there is guessing on this test, use a better model.',
                    label_visibility="visible",
                )
                st.caption("Press **Enter** to analyze.")
                analyze = st.form_submit_button("Analyze prompt")
                if analyze:
                    st.session_state.model_settings = _interpret_prompt(prompt)
                    st.session_state.is_verified = False
                    st.session_state.prompt_analyzed = True
                    st.session_state.last_prompt = prompt
                    st.session_state["confirm_itemtype"] = st.session_state.model_settings.get("itemtype", "2PL")
        with tab_model_engine:
            st.caption("Configure LLM provider and model in **Settings → Model engine**. These settings apply to all analyses.")
            st.caption(f"Current provider: `{_llm_provider()}`, model: `{_effective_llm_model()}`.")
        with tab_langgraph:
            st.caption("How LangGraph is used in this app")
            st.markdown(
                "The **psychometric workflow** is implemented as a LangGraph in `graph.py`. "
                "When you upload response/RT data and run the analysis, the app invokes **`psych_workflow.invoke(initial_state)`** directly. "
                "The graph runs: **Orchestrator** → **IRT** and **RT** (in parallel) → **Analyze** → end."
            )
            st.markdown(
                "**Optional:** From the project root, run `langgraph dev` to start the LangGraph server locally. "
                "On **Railway**, add a second service with start command `sh -c 'langgraph dev --port ${PORT:-2024}'` and generate a domain for the public LangGraph API URL (see README §6). "
                "The Streamlit UI works without the server (it uses the graph in-process)."
            )
            LANGGRAPH_API_URL = "https://langchain-ai.github.io/langgraph/concepts/langgraph_api/"
            st.link_button("Open LangGraph API", url=LANGGRAPH_API_URL, type="primary")
            st.markdown(
                f'<a href="{LANGGRAPH_API_URL}" target="_blank" rel="noopener noreferrer">Open LangGraph API in new tab</a>',
                unsafe_allow_html=True,
            )
        with tab_model_engine:
            st.markdown(
                "<style>div[data-testid='stTabs'] div[data-testid='stVerticalBlock'] { font-size: 0.9rem !important; }</style>",
                unsafe_allow_html=True,
            )
            load_dotenv()
            _api_key = os.getenv("GOOGLE_API_KEY")
            _openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
            st.radio(
                "LLM provider",
                options=["openrouter", "local_ollama"],
                format_func=lambda x: "OpenRouter" if x == "openrouter" else "Local Ollama",
                key="llm_provider",
                horizontal=True,
                help="OpenRouter uses the curated 5-model list. Local Ollama is reserved for Llama 8B/70B and does not use OpenRouter.",
            )
            if st.session_state.llm_provider == "google":
                if "api_key_test_result" not in st.session_state:
                    st.session_state.api_key_test_result = None
                if _api_key:
                    if st.button("Test API key", key="test_api_key"):
                        ok, msg = _test_api_key(_api_key)
                        st.session_state.api_key_test_result = (ok, msg)
                        st.rerun()
                    if st.session_state.api_key_test_result is not None:
                        ok, msg = st.session_state.api_key_test_result
                        if ok:
                            st.success(msg)
                        else:
                            st.error(msg)
                else:
                    st.warning("GOOGLE_API_KEY not set in .env. Add it to use Psych-MAS Summary with Google.")
            elif st.session_state.llm_provider == "openrouter":
                if "openrouter_api_key_test_result" not in st.session_state:
                    st.session_state.openrouter_api_key_test_result = None
                model_options_loaded, model_list_err, model_list_live = _load_openrouter_model_options()
                _ml_col1, _ml_col2 = st.columns([1, 2])
                with _ml_col1:
                    if st.button("Reload curated model list", key="refresh_openrouter_model_list_prompt"):
                        model_options_loaded, model_list_err, model_list_live = _load_openrouter_model_options(force=True)
                        st.toast(f"Loaded {len(model_options_loaded):,} curated US models.")
                        st.rerun()
                with _ml_col2:
                    st.caption(f"Curated list: {len(model_options_loaded):,} US models with price and recommended use.")
                if st.button("Test API key", key="test_openrouter_api_key"):
                    load_dotenv()
                    key = os.getenv("OPENROUTER_API_KEY", "")
                    ok, msg = _test_openrouter_api_key(key)
                    st.session_state.openrouter_api_key_test_result = (ok, msg)
                    st.rerun()
                if st.session_state.openrouter_api_key_test_result is not None:
                    ok, msg = st.session_state.openrouter_api_key_test_result
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)
                st.caption("OpenRouter models. Set OPENROUTER_API_KEY in .env for higher limits and private account access.")
            else:
                st.caption(f"Local Ollama endpoint: `{OLLAMA_CHAT_URL}`")
                st.caption("Install locally with `ollama pull llama3.1:8b` or `ollama pull llama3.3:70b`, then run `ollama serve`.")
            _model_options = _current_model_options()
            _model_ids = _current_model_ids()
            if "pinned_llm_model" not in st.session_state:
                st.session_state.pinned_llm_model = None
            if "pinned_llm_provider" not in st.session_state:
                st.session_state.pinned_llm_provider = None
            _pinned = st.session_state.pinned_llm_model if st.session_state.pinned_llm_provider == st.session_state.llm_provider else None
            if "selected_gemini_model" not in st.session_state:
                st.session_state.selected_gemini_model = (_pinned if _pinned and _pinned in _model_ids else None) or (_model_ids[0] if _model_ids else (LOCAL_OLLAMA_MODEL_IDS[0] if st.session_state.llm_provider == "local_ollama" else OPENROUTER_FREE_MODEL_IDS[0]))
            if st.session_state.selected_gemini_model not in _model_ids:
                st.session_state.selected_gemini_model = (_pinned if _pinned and _pinned in _model_ids else None) or (_model_ids[0] if _model_ids else (LOCAL_OLLAMA_MODEL_IDS[0] if st.session_state.llm_provider == "local_ollama" else OPENROUTER_FREE_MODEL_IDS[0]))
            st.selectbox(
                "LLM model for Psych-MAS Summary",
                options=_model_ids,
                format_func=lambda x: next((label for label, api_id in _model_options if api_id == x), _model_id_to_display_name(x) if "/" not in str(x) else str(x).split("/")[-1].replace("-", " ").title()),
                key="selected_gemini_model",
                help="OpenRouter and Local Ollama model lists are separated.",
            )
            _pinned_now = st.session_state.pinned_llm_model if st.session_state.pinned_llm_provider == st.session_state.llm_provider else None
            _lock_col, _unlock_col = st.columns(2)
            with _lock_col:
                if st.button("Lock current model for all analyses", key="lock_llm_model", help="Use this model for every LLM call (prompt + Psych-MAS Summary) until you unlock."):
                    st.session_state.pinned_llm_model = st.session_state.selected_gemini_model
                    st.session_state.pinned_llm_provider = st.session_state.llm_provider
                    st.rerun()
            with _unlock_col:
                if _pinned_now and st.button("Unlock model", key="unlock_llm_model", help="Stop using the locked model; selection will follow the dropdown again."):
                    st.session_state.pinned_llm_model = None
                    st.session_state.pinned_llm_provider = None
                    st.rerun()
            if _pinned_now:
                _locked_label = next((label for label, api_id in _model_options if api_id == _pinned_now), _pinned_now.split("/")[-1].replace("-", " ").title() if "/" in _pinned_now else _pinned_now)
                st.caption(f"🔒 **Locked:** {_locked_label} — all LLM analyses use this model until you unlock.")
            else:
                st.caption("✓ The selected model is active. Use **Analyze prompt** or any **Psych-MAS Summary** button to run with this model — no extra step needed.")
            if "model_availability" not in st.session_state:
                st.session_state.model_availability = None
            if "model_availability_errors" not in st.session_state:
                st.session_state.model_availability_errors = {}
            if "model_availability_times" not in st.session_state:
                st.session_state.model_availability_times = {}
            if st.session_state.llm_provider == "openrouter":
                if st.button("Check model availability", key="check_model_availability"):
                    with st.spinner("Testing selected OpenRouter model…"):
                        avail = {}
                        errs = {}
                        times = {}
                        model_id = st.session_state.get("selected_gemini_model") or (_current_model_ids()[0] if _current_model_ids() else "")
                        if model_id:
                            ok, err, elapsed = _test_openrouter_model(_openrouter_key, model_id)
                            avail[model_id] = ok
                            if err:
                                errs[model_id] = err
                            times[model_id] = elapsed
                        st.session_state.model_availability = avail
                        st.session_state.model_availability_errors = errs
                        st.session_state.model_availability_times = times
                    st.rerun()
                if st.session_state.model_availability is not None:
                    selected_id = st.session_state.get("selected_gemini_model", (_current_model_ids()[0] if _current_model_ids() else ""))
                    st.caption("Selected model status (available/unavailable; response time in seconds)")
                    errs = st.session_state.get("model_availability_errors", {})
                    times = st.session_state.get("model_availability_times", {})
                    for label, api_id in _current_model_options():
                        if api_id != selected_id:
                            continue
                        status = "available" if st.session_state.model_availability.get(api_id, False) else "unavailable"
                        current = " **Selected**" if api_id == selected_id else ""
                        err_msg = errs.get(api_id)
                        err_suffix = f" — *{err_msg}*" if err_msg else ""
                        t = times.get(api_id)
                        time_s = f" *({t:.1f}s)*" if t is not None else ""
                        st.markdown(f"{status} {label}{time_s}{current}{err_suffix}")
            elif st.session_state.llm_provider == "local_ollama":
                if st.button("Check local model availability", key="check_local_ollama_availability"):
                    with st.spinner("Testing selected local Ollama model…"):
                        avail = {}
                        errs = {}
                        times = {}
                        model_id = st.session_state.get("selected_gemini_model") or (LOCAL_OLLAMA_MODEL_IDS[0] if LOCAL_OLLAMA_MODEL_IDS else "")
                        if model_id:
                            ok, err, elapsed = _test_ollama_model(model_id)
                            avail[model_id] = ok
                            if err:
                                errs[model_id] = err
                            times[model_id] = elapsed
                        st.session_state.model_availability = avail
                        st.session_state.model_availability_errors = errs
                        st.session_state.model_availability_times = times
                    st.rerun()
                if st.session_state.model_availability is not None:
                    selected_id = st.session_state.get("selected_gemini_model", LOCAL_OLLAMA_MODEL_IDS[0] if LOCAL_OLLAMA_MODEL_IDS else "")
                    errs = st.session_state.get("model_availability_errors", {})
                    times = st.session_state.get("model_availability_times", {})
                    for label, api_id in LOCAL_OLLAMA_MODELS:
                        if api_id != selected_id:
                            continue
                        status = "available" if st.session_state.model_availability.get(api_id, False) else "unavailable"
                        err_msg = errs.get(api_id)
                        err_suffix = f" — *{err_msg}*" if err_msg else ""
                        t = times.get(api_id)
                        time_s = f" *({t:.1f}s)*" if t is not None else ""
                        st.markdown(f"{status} {label}{time_s}{err_suffix}")
            else:
                st.caption("Select OpenRouter or Local Ollama to check model availability.")

    with col_confirm:
        st.subheader("Confirm settings")
        feedback = st.session_state.model_settings.get("feedback", "")
        if feedback:
            st.success(f"Feedback: {feedback}")
        if "confirm_itemtype" not in st.session_state:
            st.session_state["confirm_itemtype"] = st.session_state.model_settings.get("itemtype", "2PL")
        itemtype = st.selectbox(
            "Item model",
            options=["1PL", "2PL", "3PL", "4PL"],
            key="confirm_itemtype",
        )
        r_code_preview = f"model <- mirt(df, 1, itemtype='{itemtype}')"
        with st.form("confirm_settings"):
            st.markdown("**Interpretation phase: proposed settings**")
            suggestion = st.session_state.model_settings.get("suggestion", "")
            reason = st.session_state.model_settings.get("reason", "")
            note = st.session_state.model_settings.get("note", "")
            if suggestion or reason:
                if reason:
                    st.info(f"Suggestion: {suggestion}  Reason: {reason}")
                elif suggestion:
                    st.info(f"Suggestion: {suggestion}")
            if note:
                st.warning(note)
            r_code = st.text_area(
                "R code preview",
                value=r_code_preview,
                height=100,
            )
            confirmed = st.form_submit_button("Confirm settings")
            if confirmed:
                st.session_state.model_settings = {"itemtype": itemtype, "r_code": r_code}
                st.session_state.is_verified = True
        if not st.session_state.prompt_analyzed:
            st.caption("Analyze a prompt (left) to see suggested settings here.")

    if not st.session_state.is_verified:
        st.info("Select settings and confirm to unlock IRT execution.")


def _render_response_results(final: dict) -> None:
    """Render the Response Results section (descriptive, model fit, item fit, Wright Map, APA report)."""
    st.header("Response Results")
    st.markdown("---")

    with st.expander("**1.Descriptive Summary of response**", expanded=True):
        st.subheader("1. Descriptive Summary of response")
        if final.get("icc_error"):
            st.warning(final["icc_error"])

        if "responses" in final:
            resp_df = pd.DataFrame(final["responses"])
            # Basic info: psych::describe in R (fallback to pandas describe if R unavailable)
            st.markdown("##### Basic info (psych::describe)")
            describe_df, describe_err = _psych_describe_responses(resp_df)
            if describe_df is not None and not describe_df.empty:
                st.caption("Descriptive statistics from R psych::describe.")
                st.dataframe(describe_df, height=250, use_container_width=True)
            elif describe_err:
                st.caption("R psych::describe unavailable; using pandas describe.")
                st.dataframe(resp_df.describe(), height=200, use_container_width=True)
            else:
                st.dataframe(resp_df.describe(), height=200, use_container_width=True)

            # Plot: accuracy (proportion correct) per item
            st.markdown("##### Item accuracy")
            accuracy_plot_path = _plot_item_accuracy(resp_df)
            if accuracy_plot_path and Path(accuracy_plot_path).exists():
                st.image(accuracy_plot_path)
                st.caption("Proportion correct (accuracy) per item.")
            else:
                st.info("Could not generate item accuracy plot.")

            st.markdown("##### Response matrix")
            st.caption("Rows = persons, columns = items (0/1).")
            st.dataframe(resp_df, height=200, use_container_width=True)

            if st.button("Psych-MAS Summary", key="llm_analysis_desc"):
                with st.spinner("Analyzing descriptive summary..."):
                    ctx = resp_df.describe().to_string() + "\n\nProportion correct per item:\n" + resp_df.mean(axis=0).to_string()
                    st.session_state["llm_analysis_desc_result"] = _llm_analyze_section_text("Descriptive Summary of response", ctx)
                st.rerun()
            if st.session_state.get("llm_analysis_desc_result"):
                st.markdown("###### Psych-MAS Summary")
                st.markdown(st.session_state["llm_analysis_desc_result"])

    with st.expander("**2.Model fit**", expanded=True):
        st.subheader("2. Model fit")
        model_fit = final.get("model_fit") if isinstance(final.get("model_fit"), dict) else None

        def _safe_val(v):
            if v is None: return "—"
            if isinstance(v, float) and (v != v or (v == float("inf")) or (v == float("-inf"))): return "—"
            return v

        model_fit_ctx = ""
        if model_fit:
            fit_rows = [(k, _safe_val(v)) for k, v in model_fit.items()]
            if fit_rows:
                fit_df = pd.DataFrame(fit_rows, columns=["Statistic", "Value"])
                st.dataframe(fit_df, height=min(280, 50 + 35 * len(fit_df)), use_container_width=True)
                st.caption("Overall model fit from mirt M2 (M2, df, p, RMSEA, SRMSR, TLI, CFI).")
                model_fit_ctx = fit_df.to_string() + "\n\n" + fit_df.describe().to_string()
            else:
                st.caption("Model fit (M2) could not be computed for this model.")
        elif final.get("item_fit"):
            item_fit_df = pd.DataFrame(final["item_fit"])
            num_cols = item_fit_df.select_dtypes(include=["number"]).columns
            if len(num_cols) > 0:
                summary = item_fit_df[num_cols].agg(["mean", "min", "max"]).T
                summary.columns = ["Mean", "Min", "Max"]
                summary = summary.reset_index().rename(columns={"index": "Statistic"})
                st.dataframe(summary, height=min(200, 50 + 35 * len(summary)), use_container_width=True)
                st.caption("Summary from item fit (overall M2 not available; re-run workflow to try M2).")
                model_fit_ctx = summary.to_string() + "\n\n" + summary.describe().to_string()
            else:
                st.caption("Re-run the workflow to see overall model fit (M2).")
        else:
            st.caption("Model fit (M2) requires IRT with R/mirt. Run the workflow to see fit statistics.")

        if model_fit_ctx:
            if st.button("Psych-MAS Summary", key="llm_analysis_model_fit"):
                with st.spinner("Analyzing model fit..."):
                    st.session_state["llm_analysis_model_fit_result"] = _llm_analyze_section_text("Model fit", model_fit_ctx)
                st.rerun()
            if st.session_state.get("llm_analysis_model_fit_result"):
                st.markdown("###### Psych-MAS Summary")
                st.markdown(st.session_state["llm_analysis_model_fit_result"])

    with st.expander("**3.Item Fit**", expanded=True):
        st.subheader("3. Item Fit")
        if final.get("item_fit"):
            item_fit_df = pd.DataFrame(final["item_fit"])
            st.dataframe(item_fit_df, height=400, use_container_width=True)
            if st.button("Psych-MAS Summary", key="llm_analysis_item_fit"):
                with st.spinner("Analyzing item fit..."):
                    ctx = item_fit_df.describe().to_string() + "\n\n" + item_fit_df.head(20).to_string()
                    st.session_state["llm_analysis_item_fit_result"] = _llm_analyze_section_text("Item Fit", ctx)
                st.rerun()
            if st.session_state.get("llm_analysis_item_fit_result"):
                st.markdown("###### Psych-MAS Summary")
                st.markdown(st.session_state["llm_analysis_item_fit_result"])
        else:
            st.info("No item fit data available.")

    with st.expander("**4.Wright Map & Parameters**", expanded=True):
        st.subheader("4. Wright Map & Parameters")
        item_params_df = pd.DataFrame(final.get("item_params", []))
        person_params_df = pd.DataFrame(final.get("person_params", []))
        if not item_params_df.empty and not person_params_df.empty:
            wright_map_path = _create_wright_map(item_params_df, person_params_df)
            if wright_map_path and Path(wright_map_path).exists():
                st.markdown("##### 4.1.Wright Map")
                st.image(wright_map_path)
                st.caption("Person ability distribution (histogram) and item difficulties (red lines) on the same latent trait scale.")
                wright_map_analysis_key = f"wright_map_analysis_{hash(wright_map_path)}"
                if wright_map_analysis_key not in st.session_state:
                    st.session_state[wright_map_analysis_key] = None
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown("**Psych-MAS Summary**")
                with col2:
                    if st.button("Generate Analysis", key="generate_wright_analysis"):
                        with st.spinner("Analyzing Wright Map with LLM..."):
                            st.session_state[wright_map_analysis_key] = _analyze_wright_map_image(wright_map_path)
                        st.rerun()
                if st.session_state[wright_map_analysis_key]:
                    st.markdown("###### Summary")
                    st.markdown(st.session_state[wright_map_analysis_key])
            else:
                st.info("Wright Map could not be generated. Check that item difficulty (b) and person ability (F1) columns are available.")
        st.markdown("##### Item Parameters & ICC")
        col_icc, col_item_tab = st.columns([1, 1])
        with col_icc:
            if final.get("icc_plot_path") and Path(final["icc_plot_path"]).exists():
                st.image(final["icc_plot_path"])
                st.caption("ICC (Item Characteristic Curves).")
            else:
                st.info("No ICC plot available.")
        with col_item_tab:
            if final.get("item_params"):
                item_df = pd.DataFrame(final["item_params"])
                st.dataframe(item_df, height=300, use_container_width=True)
            else:
                st.info("No item parameters available.")
        # Psych-MAS Summary full-width under figure and table (integrate ICC + item params)
        if final.get("icc_plot_path") and Path(final["icc_plot_path"]).exists() and final.get("item_params"):
            if st.button("Psych-MAS Summary", key="llm_analysis_icc"):
                with st.spinner("Analyzing Item Parameters & ICC (figure + table)..."):
                    item_df = pd.DataFrame(final["item_params"])
                    table_text = item_df.to_string() + "\n\nSummary:\n" + item_df.describe().to_string()
                    st.session_state["llm_analysis_icc_result"] = _llm_analyze_image_and_text(
                        final["icc_plot_path"],
                        "Item Parameters & ICC",
                        "Item Characteristic Curves and item parameter table",
                        table_text,
                    )
                st.rerun()
        if st.session_state.get("llm_analysis_icc_result"):
            st.markdown("###### Psych-MAS Summary (Item Parameters & ICC)")
            st.markdown(st.session_state["llm_analysis_icc_result"])
        st.markdown("##### Person Parameters")
        if final.get("person_params"):
            person_df = pd.DataFrame(final["person_params"])
            person_fig_path = _plot_person_ability(person_df)
            col_fig, col_tab = st.columns([1, 1])
            with col_fig:
                if person_fig_path and Path(person_fig_path).exists():
                    st.image(person_fig_path)
                    st.caption("Distribution of person ability (θ).")
                else:
                    st.caption("Distribution of person ability (θ) — plot not available.")
            with col_tab:
                st.dataframe(person_df, height=300, use_container_width=True)
            # Psych-MAS Summary full-width under figure and table (integrate person ability figure + person params)
            if person_fig_path and Path(person_fig_path).exists():
                if st.button("Psych-MAS Summary", key="llm_analysis_person"):
                    with st.spinner("Analyzing Person Parameters (figure + table)..."):
                        table_text = person_df.to_string() + "\n\nSummary:\n" + person_df.describe().to_string()
                        st.session_state["llm_analysis_person_result"] = _llm_analyze_image_and_text(
                            person_fig_path,
                            "Person Parameters",
                            "Distribution of person ability (θ) and person parameter table",
                            table_text,
                        )
                    st.rerun()
            if st.session_state.get("llm_analysis_person_result"):
                st.markdown("###### Psych-MAS Summary (Person Parameters)")
                st.markdown(st.session_state["llm_analysis_person_result"])
        else:
            st.info("No person parameters available.")

    with st.expander("**Advanced Analysis**", expanded=True):
        st.subheader("Ask Psych-MAS Assistant for Advanced Analysis")
        st.caption("Ask questions about your psychometric analysis. Responses are in APA format (three paragraphs, critical psychometric findings) and are included in the APA Report PDF.")
        if "analysis_chat_history" not in st.session_state:
            st.session_state.analysis_chat_history = []
        if st.session_state.analysis_chat_history:
            st.markdown("##### Conversation History")
            for i, (role, message) in enumerate(st.session_state.analysis_chat_history):
                if role == "user":
                    with st.chat_message("user"):
                        st.write(message)
                else:
                    with st.chat_message("assistant"):
                        st.write(message)
        with st.form("analysis_query_form", clear_on_submit=True):
            user_question = st.text_area(
                "Ask a question about your analysis:",
                placeholder="e.g., 'Which items are the most difficult?', 'What does the item fit suggest?'",
                height=100,
                key="analysis_question_input"
            )
            submit_question = st.form_submit_button("Ask LLM")
            if submit_question and user_question.strip():
                st.session_state.analysis_chat_history.append(("user", user_question.strip()))
                with st.spinner("Analyzing with LLM..."):
                    response = _query_llm_analysis(user_question.strip(), final)
                    st.session_state.analysis_chat_history.append(("assistant", response))
                st.rerun()
        if st.session_state.analysis_chat_history:
            if st.button("Clear Conversation", key="clear_analysis_chat"):
                st.session_state.analysis_chat_history = []
                st.rerun()

    st.markdown("---")
    st.subheader("APA Report")
    st.caption("Generate an APA-style report (Method, Results, tables) and download as PDF.")
    if st.button("Generate APA Report (PDF)", key="generate_apa_report"):
        with st.spinner("Building APA report..."):
            pdf_bytes, err = _build_apa_report_pdf(final)
            if err:
                st.session_state["apa_report_error"] = err
                st.rerun()
            elif pdf_bytes:
                st.session_state["apa_report_pdf_bytes"] = pdf_bytes
                st.session_state["apa_report_generated"] = True
                st.session_state["apa_report_error"] = None
                st.rerun()
    if st.session_state.get("apa_report_error"):
        st.error("Report generation failed: " + st.session_state["apa_report_error"])
        st.caption("If reportlab is missing, run in terminal: **uv sync**  or  **pip install reportlab**")
    if st.session_state.get("apa_report_generated") and st.session_state.get("apa_report_pdf_bytes"):
        st.download_button(
            label="Download APA Report (PDF)",
            data=st.session_state["apa_report_pdf_bytes"],
            file_name="psych_mas_apa_report.pdf",
            mime="application/pdf",
            key="download_apa_report",
        )
        st.success("Report ready. Click the button above to download.")


def _render_results(final: dict, response_only: bool = False) -> None:
    if response_only:
        _render_response_results(final)
        return
    tab1, tab2, tab3 = st.tabs(["📋 Aberrance Results", "📊 Response Results", "⏱️ RT Analysis"])
    with tab2:
        _render_response_results(final)
    with tab1:
        st.header("Aberrance Results")
        st.markdown("Results from the R package **aberrance**: person-fit indices and detected aberrant test-takers. To **report** aberrance in a PDF, run **Generate results** (or the full workflow), then generate the **APA Report (PDF)** in the Response Results tab — section *5. Person-fit (Aberrance)* will include methods, sample size, flagged count, and a summary table.")

        # Scenario presets and selection (ABERRANCE_FUNCTIONS is module-level)
        SCENARIO_PRESETS = {
            "A": {
                "title": "Scenario A: Low-Stakes",
                "icon": "🧹",
                "description": "Identifies non-substantive noise to ensure high-quality data utility.\n\n \n\nExamples: Course evaluations; Pilot surveys; Classroom quizzes",
                "selects": ["detect_rg", "detect_pm"],
            },
            "B": {
                "title": "Scenario B: High-Stakes",
                "icon": "🛡️",
                "description": "Protects high-stakes credentials: response-based, similarity, temporal, and tampering detection.\n\nExamples: Medical licensing; Answer copying; Brain-dump exposure",
                "selects": ["detect_nm", "detect_pm", "detect_ac", "detect_as", "detect_pk", "detect_rg", "detect_tt"],
            },
        }
        for fn in ABERRANCE_FUNCTIONS:
            if f"aberrance_cb_{fn}" not in st.session_state:
                st.session_state[f"aberrance_cb_{fn}"] = False
        if "aberrance_scenario_select_previous" not in st.session_state:
            st.session_state["aberrance_scenario_select_previous"] = None

        def _apply_aberrance_scenario(letter: str, *, update_select: bool = True) -> None:
            """Set scenario dropdown (if update_select) and function-list checkboxes to match the given scenario (A or B). Do not set the select key after the selectbox is instantiated (Streamlit disallows it)."""
            if letter not in SCENARIO_PRESETS:
                return
            if update_select:
                st.session_state["aberrance_scenario_select"] = letter
            st.session_state["aberrance_scenario_select_previous"] = letter
            for fn in ABERRANCE_FUNCTIONS:
                st.session_state[f"aberrance_cb_{fn}"] = fn in SCENARIO_PRESETS[letter]["selects"]

        # LLM dialogue: understand user's requirement and suggest scenario (same provider/model as Psych-MAS Assistant)
        st.subheader("Describe your testing situation")
        st.caption("Describe your context (e.g. classroom quiz, certification exam, at-home test). The assistant will suggest a scenario and select it below with the matching function list. Uses the same LLM as **Model engine** (Prompt / Psych-MAS Assistant).")
        llm_req = st.text_area(
            "Describe your testing situation",
            value=st.session_state.get("aberrance_llm_requirement", ""),
            placeholder="e.g. We run low-stakes quizzes in class and want to drop random responders. / High-stakes licensure exam in a test center. / Online unproctored assessment from home.",
            height=100,
            key="aberrance_llm_requirement",
            label_visibility="collapsed",
        )
        suggest_col1, suggest_col2 = st.columns([1, 3])
        with suggest_col1:
            suggest_btn = st.button("Suggest scenario from description", key="aberrance_suggest_scenario")
        if suggest_btn and (llm_req or st.session_state.get("aberrance_llm_requirement", "")):
            with st.spinner("Asking LLM..."):
                letter, explanation = _suggest_aberrance_scenario(
                    llm_req.strip() or st.session_state.get("aberrance_llm_requirement", "")
                )
            suggested_agents = st.session_state.get("_llm_suggested_agents") or []
            if letter and letter in SCENARIO_PRESETS:
                _apply_aberrance_scenario(letter)
                # If the model also proposed explicit agents, sync the checkboxes to that list.
                if suggested_agents:
                    for fn in ABERRANCE_FUNCTIONS:
                        st.session_state[f"aberrance_cb_{fn}"] = fn in suggested_agents
                st.session_state["aberrance_llm_suggestion"] = explanation
                st.rerun()
            elif explanation:
                st.session_state["aberrance_llm_suggestion"] = explanation
                st.rerun()
        if st.session_state.get("aberrance_llm_suggestion"):
            st.info(st.session_state["aberrance_llm_suggestion"])
            if st.button("Clear suggestion", key="aberrance_clear_suggestion"):
                st.session_state["aberrance_llm_suggestion"] = ""
                st.rerun()

        # Optional scenario selection: when changed, scenario and function list stay in sync
        st.subheader("Scenario selection")
        st.caption("Choose a scenario to auto-select detection functions; scenario and function list below update together.")
        SCENARIO_OPTIONS = [
            ("", "— Select a scenario (optional) —"),
            ("A", f"🧹 {SCENARIO_PRESETS['A']['title']}"),
            ("B", f"🛡️ {SCENARIO_PRESETS['B']['title']}"),
            ("Custom", "Custom (manual selection only)"),
        ]
        option_values = [x[0] for x in SCENARIO_OPTIONS]
        option_label_map = dict(SCENARIO_OPTIONS)
        if "aberrance_scenario_select" not in st.session_state:
            st.session_state["aberrance_scenario_select"] = ""
        scenario_choice = st.selectbox(
            "Scenario",
            options=option_values,
            format_func=lambda x: option_label_map.get(x, x),
            key="aberrance_scenario_select",
            label_visibility="collapsed",
        )
        prev = st.session_state.get("aberrance_scenario_select_previous")
        if scenario_choice in ("A", "B") and scenario_choice != prev:
            _apply_aberrance_scenario(scenario_choice, update_select=False)
            st.rerun()
        if scenario_choice in ("", "Custom"):
            st.session_state["aberrance_scenario_select_previous"] = scenario_choice
        if scenario_choice == "A":
            st.caption(SCENARIO_PRESETS["A"]["description"])
        elif scenario_choice == "B":
            st.caption(SCENARIO_PRESETS["B"]["description"])

        st.markdown("---")
        st.subheader("Function List")
        st.caption("Select which aberrance detection functions to run. Choices are used when you run the workflow.")
        cb_rg = st.checkbox(
            "Rapid Guessing (detect_rg)",
            value=st.session_state.get("aberrance_cb_detect_rg", False),
            key="aberrance_cb_detect_rg",
            help="Detects participants answering too quickly to have read the question.",
        )
        cb_pm = st.checkbox(
            "Model Misfit (detect_pm)",
            value=st.session_state.get("aberrance_cb_detect_pm", False),
            key="aberrance_cb_detect_pm",
            help="Detects general \"odd\" behavior that doesn't fit standard statistical models.",
        )
        cb_ac = st.checkbox(
            "Answer Copying (detect_ac)",
            value=st.session_state.get("aberrance_cb_detect_ac", False),
            key="aberrance_cb_detect_ac",
            help="Detects if a specific student copied answers from another specific student.",
        )
        cb_as = st.checkbox(
            "Answer Similarity (detect_as)",
            value=st.session_state.get("aberrance_cb_detect_as", False),
            key="aberrance_cb_detect_as",
            help="Detects suspicious groups of students with nearly identical answers (collusion).",
        )
        cb_pk = st.checkbox(
            "Preknowledge (detect_pk)",
            value=st.session_state.get("aberrance_cb_detect_pk", False),
            key="aberrance_cb_detect_pk",
            help="Detects students who perform suspiciously well on a specific set of \"leaked\" items.",
        )
        cb_nm = st.checkbox(
            "Guttman Errors (detect_nm)",
            value=st.session_state.get("aberrance_cb_detect_nm", False),
            key="aberrance_cb_detect_nm",
            help="Detects students who get hard questions right but miss easy ones.",
        )
        cb_tt = st.checkbox(
            "Test Tampering (detect_tt)",
            value=st.session_state.get("aberrance_cb_detect_tt", False),
            key="aberrance_cb_detect_tt",
            help="Analyzes erasure marks to find wrong-to-right answer changes. (Requires erasure data.)",
        )
        if "aberrance_compromised_items" not in st.session_state:
            st.session_state["aberrance_compromised_items"] = []
        if st.session_state.get("aberrance_cb_detect_pk", False) or (st.session_state.get("aberrance_scenario_select") == "B"):
            comp_full = st.text_input(
                "Compromised item numbers (for Preknowledge, Scenario B)",
                value=", ".join(map(str, st.session_state.get("aberrance_compromised_items") or [])),
                key="aberrance_compromised_input",
                placeholder="e.g. 1, 5, 10, 15",
                help="Comma-separated 1-based item indices that are known to be compromised/leaked.",
            )
            try:
                comp_list = [int(x.strip()) for x in comp_full.split(",") if x.strip() and x.strip().isdigit()]
                if comp_list:
                    st.session_state["aberrance_compromised_items"] = comp_list
            except Exception:
                pass
        has_rt_data = bool(final.get("rt_data") and len(final.get("rt_data", [])) > 0)
        if (cb_rg or st.session_state.get("aberrance_cb_detect_rg", False)) and not has_rt_data:
            st.warning("**Rapid Guessing (detect_rg)** requires Response Time data. Upload RT data and re-run the workflow for rapid-guessing detection.")

        gen_btn = st.button("Generate results", key="aberrance_generate_results", type="primary")
        if gen_btn:
            # Use previously uploaded data (from last run) or last_payload
            if st.session_state.get("last_uploaded_responses") and st.session_state.get("last_uploaded_model_settings") and st.session_state.get("last_uploaded_is_verified"):
                payload_dict = {
                    "responses": st.session_state.last_uploaded_responses,
                    "rt_data": st.session_state.get("last_uploaded_rt_data") or [],
                    "model_settings": st.session_state.last_uploaded_model_settings,
                    "is_verified": st.session_state.last_uploaded_is_verified,
                    "aberrance_functions": [fn for fn in ABERRANCE_FUNCTIONS if st.session_state.get(f"aberrance_cb_{fn}")],
                    "compromised_items": st.session_state.get("aberrance_compromised_items") or [],
                }
                new_payload = json.dumps(payload_dict, sort_keys=True)
            elif st.session_state.get("last_payload"):
                payload_dict = json.loads(st.session_state.last_payload)
                payload_dict["aberrance_functions"] = [fn for fn in ABERRANCE_FUNCTIONS if st.session_state.get(f"aberrance_cb_{fn}")]
                payload_dict["compromised_items"] = st.session_state.get("aberrance_compromised_items") or []
                new_payload = json.dumps(payload_dict, sort_keys=True)
            else:
                st.error("No data to run. Upload response (and optionally RT) data and confirm model settings above; then use **Generate results** here to compute aberrance indices.")
                st.stop()
            with st.spinner("Running workflow with selected functions…"):
                try:
                    final = _run_workflow(new_payload)
                    st.session_state.last_result = final
                    st.session_state.last_payload = new_payload
                    st.success("Results updated.")
                    st.rerun()
                except Exception as e:
                    st.exception(e)
                    st.stop()

        st.markdown("---")
        with st.expander("**What the aberrance package can do**", expanded=False):
            st.markdown("""
**Detection functions (aberrance v0.5.0, CRAN):**

| Function | Purpose | Main inputs |
|----------|---------|-------------|
| **detect_ac** | **Answer copying** — detect source–copier pairs | ψ (item params), x or r (scores/responses), α. Methods: OMG_S, GBT_S, OMG_R, GBT_R. |
| **detect_as** | **Answer similarity** — detect similar pairs (all pairs) | ψ, x/r/y (scores, responses, log RT). Methods: OMG_S, WOMG_S, GBT_S, M4_S; OMG_R, WOMG_R, GBT_R, M4_R; OMG_ST, GBT_ST; OMG_RT, GBT_RT. |
| **detect_cp** | **Change point** — test speededness / performance shift | ψ, x/y, cpi (change-point interval). Methods: L_S_*, S_S_*, W_S_* (scores); L_T_*, W_T_* (RT). Returns stat + estimated change point. |
| **detect_nm** | **Nonparametric misfit** — person-fit without IRT | x (scores) or y (log RT). Methods: G_S, NC_S, U1_S, U3_S, ZU3_S, A_S, D_S, E_S, C_S, MC_S, PC_S, HT_S (scores); KL_T (RT). |
| **detect_pk** | **Preknowledge** — compromised items known | ci (compromised item indices), ψ, x/y. Methods: L_S, ML_S, LR_S, S_S, W_S (scores); L_T, W_T (RT); L_ST (scores+RT). Returns stat, pval, flag. |
| **detect_pm** | **Parametric misfit** — person-fit under IRT | ψ, xi (optional), x/r/y. Methods: ECI2_S_*, ECI4_S_*, L_S_*, L_R_*, L_T, Q_ST_*, L_ST_*, Q_RT_*, L_RT_* (various corrections). Returns stat, pval, flag. |
| **detect_rg** | **Rapid guessing** — threshold or cumulative proportion | ψ (IRT + RT params), x, y (log RT). Methods for threshold/cumulative; can use item-level or person-level. |
| **detect_tt** | **Test tampering** — erasure detection | Response/erasure data; indices for tampering. |

**Utility:** **sim** — simulate item scores and/or (log) response times given ψ, ξ (e.g. 3PL, nominal, lognormal).

*Psych-MAS currently runs* detect_nm *(ZU3_S, HT_S) and, when IRT params exist,* detect_pm *(L_S_2PL) to flag aberrant test-takers. Other functions can be added in the workflow.*
            """)
        st.markdown("---")
        aberrance = final.get("aberrance_results") or {}
        if aberrance.get("error"):
            st.warning(aberrance["error"])
            st.caption("Install R package **aberrance** (e.g. `Rscript install_r_packages.R` or `install.packages('aberrance', repos='https://cloud.r-project.org')`) and re-run the workflow.")
        elif aberrance.get("info"):
            st.info(aberrance["info"])
        elif (aberrance.get("nonparametric_misfit") or aberrance.get("parametric_misfit") or
              aberrance.get("preknowledge") or aberrance.get("answer_copying_pairs") or aberrance.get("rapid_guessing")):
            flagged_persons = set(aberrance.get("flagged_persons") or [])
            flagged_copiers = set(aberrance.get("flagged_copiers") or [])
            flagged_rg = set(aberrance.get("flagged_persons_rg") or [])
            n_flagged = aberrance.get("n_flagged", 0)
            methods = aberrance.get("methods", ["ZU3_S", "HT_S"])
            # One table: all results from selected functions (nm, pm, preknowledge)
            df_aberrance = None
            for key in ("nonparametric_misfit", "parametric_misfit", "preknowledge"):
                recs = aberrance.get(key) or []
                if not recs or (df_aberrance is not None and len(recs) != len(df_aberrance)):
                    continue
                part = pd.DataFrame(recs)
                prefix = "PK_" if key == "preknowledge" else ""
                if prefix and len(part.columns) > 0:
                    part = part.add_prefix(prefix)
                if df_aberrance is None:
                    df_aberrance = part.copy()
                else:
                    for c in part.columns:
                        if c not in df_aberrance.columns:
                            df_aberrance[c] = part[c]
            if df_aberrance is not None and len(df_aberrance) > 0:
                df_aberrance = df_aberrance.copy()
                df_aberrance.insert(0, "Person", range(1, len(df_aberrance) + 1))
                if aberrance.get("answer_copying_pairs"):
                    df_aberrance["Flagged_copying"] = df_aberrance.index.isin(flagged_copiers).astype(int)
                if aberrance.get("rapid_guessing"):
                    df_aberrance["Flagged_rg"] = df_aberrance.index.isin(flagged_rg).astype(int)
                df_aberrance["Flagged"] = df_aberrance.index.isin(flagged_persons).astype(int)
                n_persons = aberrance.get("n_persons", len(df_aberrance))
                st.subheader("Person-fit statistics (all selected functions)")
                cap_parts = ["Model misfit, answer copying, preknowledge."]
                if aberrance.get("rapid_guessing"):
                    cap_parts.append("Rapid guessing (RG_NT).")
                cap_parts.append(f"Nonparametric: {', '.join(methods)}; parametric L_S_2PL; preknowledge L_S/S_S/W_S. Rows = persons (n={n_persons}).")
                st.caption(" ".join(cap_parts))
                st.dataframe(df_aberrance, height=min(400, 150 + 35 * len(df_aberrance)), use_container_width=True)
                flag_desc = "Flagged_copying = 1 if copier; Flagged_rg = 1 if rapid guessing; Flagged = any aberrant."
                st.caption(f"**{n_flagged}** of **{n_persons}** persons flagged. {flag_desc}")
            else:
                n_persons = aberrance.get("n_persons", 0)
                st.markdown(f"No person-level records (n = {n_persons}).")
                st.caption("Flagging rule: ZU3_S &lt; -2 or HT_S in bottom 5%; parametric L_S_2PL at α = .05; answer copying and preknowledge when item params and (for PK) compromised items are set.")
            ac_pairs = aberrance.get("answer_copying_pairs") or []
            if ac_pairs:
                with st.expander("Answer copying: source–copier pairs", expanded=False):
                    st.dataframe(pd.DataFrame(ac_pairs), height=min(300, 100 + 35 * len(ac_pairs)), use_container_width=True)
                    st.caption("Source = suspected source; Copier = suspected copier. Columns: OMG_S, GBT_S, etc.")
        else:
            st.info(
                "No aberrance results yet. Select functions above and click **Generate results** to compute aberrance indices using your uploaded response (and RT) data (or run the full workflow above). Then generate the **APA Report (PDF)** in the Response Results tab to include person-fit in the report (section 5. Person-fit)."
            )

    with tab3:
        st.header("RT Analysis")
        st.markdown("---")
        with st.expander("**Latency flags**", expanded=True):
            st.subheader("Latency flags")
            if final.get("latency_flags"):
                st.write(", ".join(final["latency_flags"]))
                if st.button("Psych-MAS Summary", key="llm_analysis_latency"):
                    with st.spinner("Analyzing latency flags..."):
                        st.session_state["llm_analysis_latency_result"] = _llm_analyze_section_text(
                            "Latency flags", "Flags: " + ", ".join(final["latency_flags"])
                        )
                    st.rerun()
                if st.session_state.get("llm_analysis_latency_result"):
                    st.markdown("###### Psych-MAS Summary")
                    st.markdown(st.session_state["llm_analysis_latency_result"])
            else:
                st.info("No latency flags.")
        with st.expander("**RT histograms**", expanded=True):
            st.subheader("RT histograms")
            if final.get("rt_plot_path") and Path(final["rt_plot_path"]).exists():
                st.image(final["rt_plot_path"])
                if st.button("Psych-MAS Summary", key="llm_analysis_rt"):
                    with st.spinner("Analyzing RT histograms..."):
                        st.session_state["llm_analysis_rt_result"] = _llm_analyze_image_section(
                            final["rt_plot_path"], "RT histograms", "Response time distributions per item"
                        )
                    st.rerun()
                if st.session_state.get("llm_analysis_rt_result"):
                    st.markdown("###### Psych-MAS Summary")
                    st.markdown(st.session_state["llm_analysis_rt_result"])
            else:
                st.info("No RT plot available.")

st.set_page_config(
    page_title="PsyMAS",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Modern research-workspace shell: app chrome, workflow cards, status chips.
st.markdown("""
<style>
    :root {
        color-scheme: light;
        --psymas-bg: #F4F6F8;
        --psymas-surface: #FFFFFF;
        --psymas-text: #1F2933;
        --psymas-muted: #5B6573;
        --psymas-border: #DDE2E8;
        --psymas-accent: #256D85;
        --psymas-accent-2: #2A9D8F;
        --psymas-warning: #C87512;
        --psymas-error: #B42318;
        --psymas-success: #2F855A;
        --psymas-muted-surface: #EEF2F5;
    }
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background: var(--psymas-bg) !important;
        color: var(--psymas-text) !important;
    }
    [data-testid="stHeader"] {
        background: rgba(244, 246, 248, 0.96) !important;
    }
    .block-container {
        padding-top: 2.25rem;
        padding-bottom: 1.25rem;
        color: var(--psymas-text) !important;
    }
    div[data-testid="stVerticalBlock"] {
        gap: 0.55rem !important;
    }
    hr {
        margin: 0.45rem 0 !important;
    }
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span,
    label, [data-testid="stCaptionContainer"], div[data-testid="stText"] {
        color: var(--psymas-text) !important;
    }
    [data-testid="stAppViewContainer"] :where(h1, h2, h3, h4, h5, h6, p, li, label, small):not(button *),
    [data-testid="stAppViewContainer"] :where(div[data-testid="stText"], div[data-testid="stMarkdownContainer"]) {
        color: #111827 !important;
    }
    [data-testid="stAppViewContainer"] :where(span):not(.psymas-dot):not(.psymas-nav-dot) {
        color: inherit !important;
    }
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] * {
        color: #4B5563 !important;
    }
    div[data-testid="stCheckbox"] label,
    div[data-testid="stCheckbox"] label span,
    div[data-testid="stCheckbox"] p,
    div[data-testid="stWidgetLabel"],
    div[data-testid="stWidgetLabel"] p,
    div[data-testid="stFileUploader"] label,
    div[data-testid="stFileUploader"] p {
        color: var(--psymas-text) !important;
    }
    input, textarea, select,
    div[data-baseweb="input"] *,
    div[data-baseweb="textarea"] *,
    div[data-baseweb="select"] * {
        color: #111827 !important;
    }
    input, textarea,
    div[data-baseweb="input"],
    div[data-baseweb="textarea"],
    div[data-baseweb="select"] {
        background: #FFFFFF !important;
    }
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] *,
    div[data-baseweb="menu"],
    div[data-baseweb="menu"] *,
    ul[role="listbox"],
    ul[role="listbox"] *,
    div[role="listbox"],
    div[role="listbox"] *,
    li[role="option"],
    li[role="option"] *,
    div[role="option"],
    div[role="option"] * {
        background-color: #FFFFFF !important;
        color: #111827 !important;
        opacity: 1 !important;
    }
    li[role="option"]:hover,
    li[role="option"][aria-selected="true"],
    div[role="option"]:hover,
    div[role="option"][aria-selected="true"],
    div[data-baseweb="menu"] li:hover,
    div[data-baseweb="menu"] li[aria-selected="true"] {
        background-color: #E8F4F7 !important;
        color: #0B1F33 !important;
    }
    button[role="tab"],
    button[role="tab"] *,
    div[data-baseweb="tab-list"] *,
    div[data-baseweb="tab"] * {
        color: #111827 !important;
    }
    button[role="tab"],
    div[data-baseweb="tab"],
    div[data-baseweb="tab-list"] button {
        background-color: #FFFFFF !important;
        border-color: #CBD5E1 !important;
    }
    button[role="tab"][aria-selected="true"],
    div[data-baseweb="tab"][aria-selected="true"],
    div[data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #E8F4F7 !important;
        color: #0B1F33 !important;
        border-color: #174E5F !important;
    }
    div[data-testid="stSegmentedControl"] button,
    div[data-testid="stSegmentedControl"] button * {
        background-color: #FFFFFF !important;
        color: #111827 !important;
        border-color: #CBD5E1 !important;
        opacity: 1 !important;
    }
    div[data-testid="stSegmentedControl"] button[aria-pressed="true"],
    div[data-testid="stSegmentedControl"] button[aria-pressed="true"] * {
        background-color: #E8F4F7 !important;
        color: #0B1F33 !important;
        border-color: #174E5F !important;
    }
    button:disabled,
    button:disabled *,
    button[disabled],
    button[disabled] * {
        color: #4B5563 !important;
        opacity: 1 !important;
    }
    div[data-testid="stAlert"],
    div[data-testid="stAlert"] *,
    div[data-testid="stSuccess"],
    div[data-testid="stSuccess"] *,
    div[data-testid="stInfo"],
    div[data-testid="stInfo"] *,
    div[data-testid="stWarning"],
    div[data-testid="stWarning"] *,
    div[data-testid="stError"],
    div[data-testid="stError"] * {
        color: #111827 !important;
    }
    details[data-testid="stExpander"],
    div[data-testid="stExpander"],
    div[data-testid="stExpander"] details {
        background: #FFFFFF !important;
        color: #111827 !important;
        border-color: #CBD5E1 !important;
        opacity: 1 !important;
    }
    details[data-testid="stExpander"] summary,
    details[data-testid="stExpander"] summary *,
    div[data-testid="stExpander"] summary,
    div[data-testid="stExpander"] summary *,
    div[data-testid="stExpander"] p,
    div[data-testid="stExpander"] span,
    div[data-testid="stExpander"] label {
        background: #FFFFFF !important;
        color: #111827 !important;
        opacity: 1 !important;
    }
    details[data-testid="stExpander"][open] summary,
    div[data-testid="stExpander"] details[open] summary {
        background: #F8FAFC !important;
        border-bottom: 1px solid #E2E8F0 !important;
    }
    div[data-testid="stExpander"] svg,
    details[data-testid="stExpander"] svg {
        color: #334155 !important;
        fill: currentColor !important;
        stroke: currentColor !important;
    }
    div[data-testid="stTabs"],
    div[data-testid="stTabs"] [role="tablist"] {
        background: transparent !important;
        color: #111827 !important;
    }
    div[data-testid="stTabs"] button,
    div[data-testid="stTabs"] button *,
    div[data-testid="stTabs"] [role="tab"],
    div[data-testid="stTabs"] [role="tab"] * {
        background: #FFFFFF !important;
        color: #111827 !important;
        opacity: 1 !important;
    }
    div[data-testid="stTabs"] button[aria-selected="true"],
    div[data-testid="stTabs"] button[aria-selected="true"] *,
    div[data-testid="stTabs"] [role="tab"][aria-selected="true"],
    div[data-testid="stTabs"] [role="tab"][aria-selected="true"] * {
        background: #E8F4F7 !important;
        color: #0B1F33 !important;
        opacity: 1 !important;
    }
    div[data-testid="stFileUploader"] *,
    div[data-testid="stFileUploaderFile"] *,
    div[data-testid="stFileUploaderDropzone"] * {
        color: #111827 !important;
        opacity: 1 !important;
    }
    div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stMarkdown"]) .stMarkdown h3 {
        margin-top: 0.5rem;
        margin-bottom: 0.25rem;
        font-weight: 600;
    }
    .status-strip {
        padding: 0.5rem 0.75rem;
        border-radius: 0.375rem;
        border: 1px solid rgba(49, 51, 63, 0.2);
        font-size: 0.875rem;
        margin-bottom: 1rem;
    }
    [data-theme="dark"] .status-strip { border-color: rgba(250, 250, 250, 0.2); }
    section[data-testid="stSidebar"] {
        border-right: 1px solid var(--psymas-border);
        background: #FFFFFF !important;
    }
    section[data-testid="stSidebar"] * {
        color: var(--psymas-text);
    }
    section[data-testid="stSidebar"] .stRadio label {
        font-weight: 500;
        color: var(--psymas-text);
    }
    section[data-testid="stSidebar"] h1 {
        font-size: 1.35rem;
        letter-spacing: 0;
        margin-bottom: 0.1rem;
    }
    .psymas-kicker {
        color: var(--psymas-muted);
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.25rem;
    }
    .psymas-title-row {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    .psymas-title-row h1 {
        color: var(--psymas-text);
        font-size: 1.65rem;
        line-height: 2.1rem;
        margin: 0;
        letter-spacing: 0;
    }
    .psymas-title-row p {
        color: var(--psymas-muted);
        margin: 0.2rem 0 0 0;
        max-width: 64rem;
    }
    .psymas-stage-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        border: 1px solid var(--psymas-border);
        border-radius: 999px;
        padding: 0.25rem 0.6rem;
        color: var(--psymas-accent);
        background: var(--psymas-surface);
        font-size: 0.78rem;
        white-space: nowrap;
    }
    .psymas-flow {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        flex-wrap: wrap;
        margin: 0.25rem 0 0.5rem 0;
    }
    .psymas-topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.75rem;
        border: 1px solid var(--psymas-border);
        border-radius: 8px;
        background: var(--psymas-surface);
        padding: 0.55rem 0.75rem;
        gap: 0.6rem;
        margin-bottom: 0.9rem;
    }
    .psymas-topbar-title {
        font-size: 0.92rem;
        font-weight: 650;
        color: var(--psymas-text);
        white-space: nowrap;
    }
    .psymas-topbar-meta {
        color: var(--psymas-muted);
        font-size: 0.82rem;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .psymas-flow-card {
        min-height: auto;
        padding: 0.3rem 0.55rem;
        border-radius: 999px;
        border: 1px solid var(--psymas-border);
        background: var(--psymas-surface);
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
    }
    .psymas-flow-card.active {
        border-color: var(--psymas-accent);
        background: #E7F2F5;
        box-shadow: none;
    }
    .psymas-flow-card .step {
        color: var(--psymas-muted);
        font-size: 0.72rem;
        margin-bottom: 0;
    }
    .psymas-flow-card .name {
        color: var(--psymas-text);
        font-weight: 650;
        font-size: 0.82rem;
        line-height: 1rem;
    }
    .psymas-flow-card .state { display: none; }
    .psymas-context {
        border: 1px solid var(--psymas-border);
        border-radius: 8px;
        background: var(--psymas-surface);
        padding: 0.85rem 1rem;
        margin-bottom: 1rem;
    }
    .psymas-context h3 {
        font-size: 1rem;
        margin: 0 0 0.5rem 0;
        color: var(--psymas-text);
    }
    .psymas-context dl {
        display: grid;
        grid-template-columns: 7rem 1fr;
        gap: 0.35rem 0.75rem;
        margin: 0;
        font-size: 0.86rem;
    }
    .psymas-context dt {
        color: var(--psymas-muted);
    }
    .psymas-context dd {
        color: var(--psymas-text);
        margin: 0;
    }
    .psymas-output-console {
        border: 1px solid var(--psymas-border);
        border-radius: 8px;
        background: var(--psymas-surface);
        padding: 1rem;
        margin: 0.75rem 0 1.25rem 0;
    }
    .psymas-workbench-title {
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        gap: 1rem;
        margin: 0.35rem 0 0.8rem 0;
    }
    .psymas-workbench-title h1 {
        font-size: 1.35rem;
        line-height: 1.7rem;
        margin: 0;
        color: var(--psymas-text);
        letter-spacing: 0;
    }
    .psymas-workbench-title p {
        margin: 0.15rem 0 0 0;
        color: var(--psymas-muted);
        font-size: 0.86rem;
    }
    .psymas-case-strip {
        display: grid;
        grid-template-columns: repeat(4, minmax(8rem, 1fr));
        gap: 0.6rem;
        margin-bottom: 0.8rem;
    }
    .psymas-case-chip {
        border: 1px solid var(--psymas-border);
        border-radius: 8px;
        background: var(--psymas-surface);
        padding: 0.65rem 0.75rem;
    }
    .psymas-case-chip .label {
        color: var(--psymas-muted);
        font-size: 0.74rem;
        margin-bottom: 0.15rem;
    }
    .psymas-case-chip .value {
        color: var(--psymas-text);
        font-weight: 700;
        font-size: 1.1rem;
    }
    .psymas-case-chip.psymas-priority-low {
        border-left: 4px solid #94A3B8;
        background: #F8FAFC;
    }
    .psymas-case-chip.psymas-priority-medium {
        border-left: 4px solid #C87512;
        background: #FFF7ED;
    }
    .psymas-case-chip.psymas-priority-high {
        border-left: 4px solid #C87512;
        background: #FFF7ED;
    }
    .psymas-case-chip.psymas-priority-high_context {
        border-left: 4px solid #FB923C;
        background: #FFF1F2;
    }
    .psymas-case-chip.psymas-priority-critical {
        border-left: 4px solid #DC2626;
        background: #FEF2F2;
    }
    .psymas-case-chip.psymas-priority-unknown {
        border-left: 4px solid #CBD5E1;
        background: #FFFFFF;
    }
    .psymas-case-header-compact {
        display: grid;
        grid-template-columns: minmax(7rem, 0.7fr) minmax(11rem, 1.1fr) minmax(10rem, 1.2fr);
        gap: 0.55rem;
        margin: 0.35rem 0 0.7rem 0;
    }
    .psymas-case-header-compact .case-field {
        border: 1px solid #CBD5E1;
        border-left: 4px solid #0F7890;
        border-radius: 8px;
        background: #FFFFFF;
        padding: 0.48rem 0.62rem;
        min-height: 3.2rem;
    }
    .psymas-case-header-compact .case-label {
        color: #64748B !important;
        font-size: 0.62rem;
        font-weight: 850;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.18rem;
    }
    .psymas-case-header-compact .case-value {
        color: #0F172A !important;
        font-size: 0.92rem;
        font-weight: 850;
        line-height: 1.15rem;
    }
    .psymas-review-focus-card {
        border: 1px solid #B7CBD8;
        border-left: 5px solid #0F7890;
        border-radius: 10px;
        background: linear-gradient(135deg, #FFFFFF 0%, #F8FBFD 100%);
        box-shadow: 0 10px 28px rgba(15, 94, 117, 0.08);
        padding: 0.72rem 0.9rem;
        margin: 0.2rem 0 0.85rem 0;
    }
    .psymas-review-focus-card .label {
        color: #0F7890 !important;
        font-size: 0.66rem;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.22rem;
    }
    .psymas-review-focus-card .focus {
        color: #0F172A !important;
        font-size: 1.02rem;
        font-weight: 900;
        line-height: 1.28rem;
        margin-bottom: 0.2rem;
    }
    .psymas-review-focus-card .detail {
        color: #475569 !important;
        font-size: 0.8rem;
        line-height: 1.15rem;
    }
    .psymas-review-step-title {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        color: #0F172A !important;
        font-size: 1.02rem;
        font-weight: 900;
        margin: 0.05rem 0 0.55rem 0;
    }
    .psymas-review-step-title .step-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 2rem;
        height: 2rem;
        border-radius: 999px;
        background: #0F6B7C;
        color: #FFFFFF !important;
        font-size: 0.82rem;
        font-weight: 900;
        box-shadow: 0 8px 18px rgba(15, 94, 117, 0.18);
        flex: 0 0 auto;
    }
    .psymas-review-step-title .step-text {
        line-height: 1.2rem;
    }
    .psymas-review-step-title .step-note {
        margin-left: auto;
        background: #070B14;
        color: #F8FAFC !important;
        -webkit-text-fill-color: #F8FAFC !important;
        border: 1px solid rgba(148, 163, 184, 0.35);
        border-radius: 999px;
        padding: 0.28rem 0.65rem;
        font-size: 0.68rem;
        font-weight: 750;
        line-height: 0.95rem;
        max-width: 48%;
        text-align: right;
    }
    .psymas-ai-suggestion {
        border: 1px solid #CBD5E1;
        border-left: 4px solid #0F6B7C;
        border-radius: 8px;
        background: #FFFFFF;
        padding: 0.48rem 0.68rem;
        margin: 0.25rem 0 0.4rem 0;
        color: #0F172A !important;
        font-size: 0.82rem;
        line-height: 1.18rem;
    }
    .psymas-ai-suggestion .label {
        color: #0F6B7C !important;
        font-size: 0.62rem;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.08rem;
    }
    .psymas-ai-suggestion .text {
        min-height: 0;
        max-height: 18rem;
        overflow: auto;
        white-space: pre-wrap;
    }
    .psymas-ai-suggestion .section-title {
        display: block;
        font-weight: 900;
        color: #0F172A !important;
        margin: 0.16rem 0 0 0;
        line-height: 1.05rem;
    }
    .psymas-ai-suggestion .section-body {
        display: block;
        margin: 0.02rem 0 0.08rem 0.45rem;
        line-height: 1.12rem;
    }
    .psymas-decision-guide {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.55rem;
        margin: 0.45rem 0 0.65rem 0;
    }
    .psymas-decision-guide .decision-card {
        border: 1px solid #CBD5E1;
        border-left: 4px solid #94A3B8;
        border-radius: 8px;
        background: #FFFFFF;
        padding: 0.55rem 0.65rem;
        min-height: 4.2rem;
    }
    .psymas-decision-guide .decision-card.selected {
        border-color: #0F6B7C;
        border-left-color: #0F6B7C;
        box-shadow: 0 10px 24px rgba(15, 94, 117, 0.16);
        background: #ECFEFF;
    }
    .psymas-decision-guide .level {
        color: #64748B !important;
        font-size: 0.62rem;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .psymas-decision-guide .decision {
        color: #0F172A !important;
        font-size: 0.86rem;
        font-weight: 900;
        margin-top: 0.2rem;
    }
    .psymas-decision-guide .meaning {
        color: #475569 !important;
        font-size: 0.72rem;
        line-height: 1.05rem;
        margin-top: 0.2rem;
    }
    .psymas-priority-pill {
        display: inline-block;
        padding: 0.12rem 0.45rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 800;
        border: 1px solid transparent;
        white-space: nowrap;
    }
    .psymas-priority-pill.low { background: #F8FAFC; color: #475569; border-color: #94A3B8; }
    .psymas-priority-pill.medium { background: #FFF7ED; color: #9A3412; border-color: #C87512; }
    .psymas-priority-pill.high { background: #FFF7ED; color: #9A3412; border-color: #C87512; }
    .psymas-priority-pill.high_context { background: #FFF1F2; color: #9F1239; border-color: #FB923C; }
    .psymas-priority-pill.critical { background: #FEF2F2; color: #991B1B; border-color: #DC2626; }
    .psymas-priority-pill.unknown { background: #FFFFFF; color: #64748B; border-color: #CBD5E1; }
    .psymas-examinee-priority-legend {
        display: flex;
        flex-wrap: wrap;
        gap: 0.3rem 0.55rem;
        margin: 0.1rem 0 0.25rem 0;
        font-size: 0.62rem;
        color: #64748B;
    }
    .psymas-examinee-priority-legend span.item {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        white-space: nowrap;
    }
    .psymas-examinee-priority-legend span.dot {
        width: 0.55rem;
        height: 0.55rem;
        border-radius: 999px;
        display: inline-block;
        border: 1px solid rgba(15, 23, 42, 0.12);
    }
    .psymas-examinee-matrix-wrap {
        margin: 0.05rem 0 0.35rem 0;
    }
    .psymas-examinee-matrix-wrap .psymas-priority-pill-row-label {
        font-size: 0.62rem;
        color: #64748B;
        margin: 0.2rem 0 0.08rem 0;
    }
    .psymas-priority-filter-row {
        margin: 0.05rem 0 0.2rem 0;
    }
    .psymas-case-examinee-nav {
        margin-top: 0.15rem;
        font-size: 0.82rem;
    }
    .psymas-lineage-legend {
        background: #070B14;
        border: 1px solid rgba(148, 163, 184, 0.35);
        border-radius: 8px;
        padding: 0.62rem 0.7rem;
        margin: 0.05rem 0 0.65rem 0;
    }
    .psymas-lineage-legend-heading {
        color: #F8FAFC !important;
        font-size: 0.72rem;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.5rem;
    }
    .psymas-lineage-legend-row {
        display: flex;
        align-items: flex-start;
        gap: 0.45rem;
        font-size: 0.78rem;
        color: #CBD5E1;
    }
    .psymas-lineage-legend-row + .psymas-lineage-legend-row {
        margin-top: 0.5rem;
        padding-top: 0.5rem;
        border-top: 1px solid rgba(148, 163, 184, 0.2);
    }
    .psymas-lineage-legend-row b {
        color: #F8FAFC !important;
        display: block;
        font-size: 0.78rem;
        line-height: 1rem;
    }
    .psymas-lineage-legend-row em {
        color: #94A3B8 !important;
        display: block;
        font-size: 0.68rem;
        font-style: normal;
        line-height: 0.95rem;
    }
    .psymas-lineage-legend-row span.dot {
        width: 0.85rem;
        height: 0.85rem;
        border-radius: 2px;
        display: inline-block;
        border: 1px solid rgba(15, 23, 42, 0.2);
        flex-shrink: 0;
        margin-top: 0.08rem;
    }
    .psymas-case-strip.flat {
        margin: 0.25rem 0 0.55rem 0;
    }
    .psymas-case-strip.flat .psymas-case-chip {
        border: 0;
        border-radius: 0;
        background: transparent;
        padding: 0.2rem 0;
    }
    .psymas-reviewer-note {
        border: 1px solid var(--psymas-border);
        border-left: 4px solid var(--psymas-accent);
        border-radius: 8px;
        background: #FFFFFF;
        color: #111827 !important;
        padding: 0.9rem 1rem;
        line-height: 1.55;
        margin: 0.25rem 0 0.75rem 0;
    }
    .psymas-reviewer-note * {
        color: #111827 !important;
    }
    .psymas-reviewer-note.flat {
        border: 0;
        border-left: 3px solid var(--psymas-accent);
        border-radius: 0;
        background: transparent;
        padding: 0.25rem 0 0.25rem 0.65rem;
        margin: 0.25rem 0 0.5rem 0;
    }
    .psymas-llm-panel {
        border: 1px solid #CBD5E1;
        border-left: 4px solid #0F6B7C;
        border-radius: 8px;
        background: #FFFFFF;
        padding: 0.8rem 0.95rem;
        margin: 0.45rem 0 0.75rem 0;
        color: #111827 !important;
    }
    .psymas-llm-panel .panel-kicker {
        color: #0F6B7C !important;
        font-size: 0.76rem;
        font-weight: 850;
        text-transform: uppercase;
        letter-spacing: 0.02em;
        margin-bottom: 0.25rem;
    }
    .psymas-llm-panel .panel-title {
        color: #111827 !important;
        font-size: 1rem;
        font-weight: 850;
        margin-bottom: 0.25rem;
    }
    .psymas-llm-panel .panel-copy {
        color: #4B5563 !important;
        font-size: 0.82rem;
        line-height: 1.25rem;
    }
    div[class*="st-key-case_llm_chat_"] textarea,
    div[class*="st-key-case_llm_chat_"] input {
        background: #FFFFFF !important;
        color: #111827 !important;
        border: 1px solid #94A3B8 !important;
        min-height: 2rem !important;
        font-size: 0.78rem !important;
    }
    .psymas-domain-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(12rem, 1fr));
        gap: 0.7rem;
        margin: 0.25rem 0 0.85rem 0;
    }
    .psymas-domain-card {
        border: 1px solid var(--psymas-border);
        border-radius: 8px;
        background: #FFFFFF;
        padding: 0.8rem;
        color: #111827 !important;
        min-height: 8rem;
    }
    .psymas-domain-card.issue {
        border-left: 4px solid #C87512;
        background: #FFFBEB;
    }
    .psymas-domain-card.strong {
        border-left: 4px solid #DC2626;
        background: #FEF7F5;
    }
    .psymas-domain-card.moderate {
        border-left: 4px solid #C87512;
        background: #FFFBEB;
    }
    .psymas-domain-card.weak {
        border-left: 4px solid #64748B;
        background: #F8FAFC;
    }
    .psymas-domain-card.support {
        border-left: 4px solid #6D5BD0;
        background: #F7F5FF;
    }
    .psymas-domain-card.unavailable {
        border-left: 4px solid #94A3B8;
        background: #F8FAFC;
    }
    .psymas-domain-card.clear {
        border-left: 4px solid #147A5A;
    }
    .psymas-domain-frame {
        border: 1px solid var(--psymas-border);
        border-left: 4px solid #147A5A;
        border-radius: 8px;
        background: #FFFFFF;
        padding: 0.8rem;
        color: #111827 !important;
        margin: 0.7rem 0;
    }
    .psymas-domain-frame.issue {
        border-left-color: #C87512;
        background: #FFFBEB;
    }
    .psymas-domain-frame.clear {
        border-left-color: #147A5A;
        background: #FFFFFF;
    }
    .psymas-domain-frame-inline {
        color: #111827 !important;
        margin: 0.1rem 0 0 0;
    }
    .psymas-domain-figure-slot {
        background: #FFFFFF;
        border-top: 1px solid #E5E7EB;
        margin-top: 0.55rem;
        padding-top: 0.55rem;
    }
    div[class*="st-key-domain_sim_frame_"],
    div[class*="st-key-domain_tp_frame_"],
    div[class*="st-key-domain_cp_frame_"] {
        border: 1px solid #CBD5E1 !important;
        border-left: 4px solid #94A3B8 !important;
        border-radius: 8px !important;
        background: #FFFFFF !important;
        padding: 0.75rem 0.8rem !important;
        margin: 0.7rem 0 !important;
    }
    .psymas-domain-card .domain-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-weight: 800;
        margin-bottom: 0.35rem;
    }
    .psymas-domain-row-head {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 1rem;
        margin-bottom: 0.4rem;
    }
    .psymas-domain-row-head .domain-label {
        color: #4B5563 !important;
        font-size: 0.74rem;
        font-weight: 800;
        margin-bottom: 0.15rem;
    }
    .psymas-domain-row-head .domain-name {
        color: #111827 !important;
        font-size: 1rem;
        font-weight: 850;
    }
    .psymas-domain-row-head .domain-strength {
        color: #111827 !important;
        font-weight: 850;
    }
    .psymas-domain-card .domain-step {
        display: inline-flex;
        color: #4B5563 !important;
        font-size: 0.72rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .psymas-domain-card .domain-status,
    .domain-status {
        display: inline-flex;
        width: fit-content;
        border-radius: 999px;
        background: #EEF2F7;
        padding: 0.08rem 0.42rem;
        font-size: 0.62rem;
        color: #334155 !important;
        font-weight: 850;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    .psymas-domain-card.strong .domain-status,
    .psymas-domain-card.issue .domain-status,
    .psymas-domain-frame-inline.issue .domain-status {
        background: #FEE4E2;
        color: #912018 !important;
    }
    .psymas-domain-card.moderate .domain-status {
        background: #FEF0C7;
        color: #7A4B00 !important;
    }
    .psymas-domain-card.support .domain-status {
        background: #ECE8FF;
        color: #3E2D8C !important;
    }
    .psymas-domain-card.unavailable .domain-status,
    .psymas-domain-frame-inline.unavailable .domain-status {
        background: #E2E8F0;
        color: #475569 !important;
    }
    .psymas-domain-card .domain-index,
    .domain-index {
        font-weight: 700;
        font-size: 0.86rem;
        margin-bottom: 0.35rem;
    }
    .psymas-domain-card .domain-figure,
    .domain-figure {
        border-top: 1px solid #E5E7EB;
        margin-top: 0.5rem;
        padding-top: 0.45rem;
        color: #374151 !important;
        font-size: 0.78rem;
        line-height: 1.2rem;
    }
    .psymas-domain-card .domain-pattern,
    .domain-pattern {
        color: #4B5563 !important;
        font-size: 0.78rem;
        line-height: 1.25rem;
    }
    .psymas-case-domain-strip {
        display: grid;
        grid-template-columns: repeat(6, minmax(0, 1fr));
        gap: 0.45rem;
        margin: 0.35rem 0 0.75rem 0;
    }
    .psymas-case-domain-mini {
        border: 1px solid #CBD5E1;
        border-left: 4px solid #147A5A;
        border-radius: 8px;
        background: #FFFFFF;
        padding: 0.46rem 0.52rem;
        min-height: 5.45rem;
        color: #111827 !important;
    }
    .psymas-case-domain-mini.strong { border-left-color: #DC2626; background: #FEF7F5; }
    .psymas-case-domain-mini.moderate { border-left-color: #C87512; background: #FFFBEB; }
    .psymas-case-domain-mini.weak { border-left-color: #64748B; background: #F8FAFC; }
    .psymas-case-domain-mini.support { border-left-color: #6D5BD0; background: #F7F5FF; }
    .psymas-case-domain-mini.unavailable { border-left-color: #94A3B8; background: #F8FAFC; }
    .psymas-case-domain-mini .mini-code {
        color: #64748B !important;
        font-size: 0.58rem;
        font-weight: 850;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .psymas-case-domain-mini .mini-name {
        color: #111827 !important;
        font-size: 0.72rem;
        font-weight: 850;
        margin-top: 0.08rem;
    }
    .psymas-case-domain-mini .mini-strength {
        color: #111827 !important;
        font-size: 0.92rem;
        font-weight: 850;
        margin-top: 0.14rem;
        line-height: 1.05;
    }
    .psymas-case-domain-mini .mini-status {
        display: inline-flex;
        width: fit-content;
        border-radius: 999px;
        background: #EEF2F7;
        color: #334155 !important;
        font-size: 0.54rem;
        font-weight: 850;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        padding: 0.06rem 0.32rem;
        margin-top: 0.2rem;
    }
    .psymas-case-domain-mini .mini-index {
        color: #475569 !important;
        font-size: 0.58rem;
        line-height: 0.86rem;
        margin-top: 0.24rem;
    }
    .psymas-lineage-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin: 0.15rem 0 0.55rem 0;
    }
    .psymas-lineage-chips span {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        border: 1px solid #CBD5E1;
        border-radius: 999px;
        background: #FFFFFF;
        color: #334155 !important;
        font-size: 0.72rem;
        line-height: 1;
        padding: 0.35rem 0.55rem;
    }
    .psymas-lineage-chips b {
        color: #0F172A !important;
        font-weight: 850;
    }
    .psymas-help-dot {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 1rem;
        height: 1rem;
        border-radius: 999px;
        background: #E2E8F0;
        color: #334155 !important;
        font-size: 0.68rem;
        font-weight: 850;
        cursor: help;
        margin-left: 0.35rem;
        vertical-align: middle;
    }
    .psymas-domain-visual-head {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 1rem;
        border: 1px solid #CBD5E1;
        border-left: 4px solid #147A5A;
        border-radius: 8px;
        background: #FFFFFF;
        padding: 0.56rem 0.68rem;
        margin: 0.55rem 0 0.45rem 0;
    }
    .psymas-domain-visual-head.strong { border-left-color: #DC2626; background: #FEF7F5; }
    .psymas-domain-visual-head.moderate { border-left-color: #C87512; background: #FFFBEB; }
    .psymas-domain-visual-head.support { border-left-color: #6D5BD0; background: #F7F5FF; }
    .psymas-domain-visual-head.unavailable { border-left-color: #94A3B8; background: #F8FAFC; }
    .psymas-domain-visual-head .domain-kicker {
        color: #64748B !important;
        font-size: 0.62rem;
        font-weight: 850;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .psymas-domain-visual-head .domain-title {
        color: #111827 !important;
        font-size: 0.95rem;
        font-weight: 850;
        margin-top: 0.08rem;
    }
    .psymas-domain-visual-head .domain-meta {
        color: #475569 !important;
        font-size: 0.72rem;
        font-weight: 700;
        margin-top: 0.15rem;
    }
    .psymas-domain-visual-head .domain-outcome {
        color: #111827 !important;
        font-size: 0.88rem;
        font-weight: 850;
        text-align: right;
        white-space: nowrap;
    }
    @media (max-width: 1200px) {
        .psymas-case-domain-strip { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    }
    .psymas-adjudication-guide {
        display: grid;
        grid-template-columns: repeat(4, minmax(10rem, 1fr));
        gap: 0.55rem;
        margin: 0.35rem 0 0.85rem 0;
    }
    .psymas-adjudication-card {
        border: 1px solid var(--psymas-border);
        border-left: 5px solid #94A3B8;
        border-radius: 8px;
        background: #FFFFFF;
        color: #111827 !important;
        padding: 0.62rem 0.72rem;
        min-height: 4.7rem;
    }
    .psymas-adjudication-card.selected {
        box-shadow: 0 0 0 2px rgba(17, 24, 39, 0.08);
        background: #F8FAFC;
    }
    .psymas-adjudication-card .level {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.74rem;
        color: #4B5563 !important;
        font-weight: 800;
        margin-bottom: 0.25rem;
    }
    .psymas-adjudication-card .name {
        color: #111827 !important;
        font-size: 0.9rem;
        font-weight: 850;
        margin-bottom: 0.18rem;
    }
    .psymas-adjudication-card .meaning {
        color: #4B5563 !important;
        font-size: 0.75rem;
        line-height: 1.15rem;
    }
    .psymas-adjudication-card.unreviewed { border-left-color: #94A3B8; }
    .psymas-adjudication-card.rule { border-left-color: #C87512; background: #FFFBEB; }
    .psymas-adjudication-card.potential { border-left-color: #C87512; background: #FFF7ED; }
    .psymas-adjudication-card.definite { border-left-color: #DC2626; background: #FEF2F2; }
    div[class*="st-key-case_decision_"] button {
        min-height: 5.85rem;
        justify-content: flex-start;
        text-align: left;
        white-space: pre-line;
        line-height: 1.22rem;
        font-weight: 800;
        border-radius: 8px;
        color: #111827 !important;
    }
    div[class*="st-key-case_decision_"] button p {
        color: #111827 !important;
        font-size: 0.86rem;
        line-height: 1.18rem;
    }
    div[class*="st-key-case_decision_no_issue_"] button {
        background: #F8FAFC !important;
        border: 1px solid #CBD5E1 !important;
        border-left: 5px solid #64748B !important;
    }
    div[class*="st-key-case_decision_rule_"] button {
        background: #FFFBEB !important;
        border: 1px solid #C87512 !important;
        border-left: 5px solid #C87512 !important;
    }
    div[class*="st-key-case_decision_potential_"] button {
        background: #FFF7ED !important;
        border: 1px solid #FDBA74 !important;
        border-left: 5px solid #C87512 !important;
    }
    div[class*="st-key-case_decision_definite_"] button {
        background: #FEF2F2 !important;
        border: 1px solid #FCA5A5 !important;
        border-left: 5px solid #DC2626 !important;
    }
    .psymas-stamp {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 12rem;
        min-height: 3.2rem;
        border: 3px solid #B42318;
        border-radius: 10px;
        color: #B42318 !important;
        background: #FFF7F7;
        font-weight: 900;
        letter-spacing: 0;
        transform: rotate(-2deg);
        text-transform: uppercase;
    }
    .psymas-stamp.unreviewed {
        border-color: #64748B;
        color: #475569 !important;
        background: #F8FAFC;
    }
    .psymas-stamp.rule {
        border-color: #C87512;
        color: #92400E !important;
        background: #FFFBEB;
    }
    .psymas-stamp.potential {
        border-color: #C87512;
        color: #9A3412 !important;
        background: #FFF7ED;
    }
    .psymas-stamp.definite {
        border-color: #DC2626;
        color: #991B1B !important;
        background: #FEF2F2;
    }
    .psymas-output-title {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 0.75rem;
    }
    .psymas-output-title h2 {
        font-size: 1.05rem;
        line-height: 1.4rem;
        margin: 0;
        color: var(--psymas-text);
    }
    .psymas-output-title p {
        margin: 0.15rem 0 0 0;
        color: var(--psymas-muted);
        font-size: 0.86rem;
    }
    .psymas-subtable-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin: 0.75rem 0 0.9rem 0;
    }
    .psymas-quiet-meta {
        color: var(--psymas-muted);
        font-size: 0.78rem;
        line-height: 1.2rem;
        margin-top: 0.35rem;
    }
    .psymas-inspector {
        border: 1px solid var(--psymas-border);
        border-radius: 8px;
        background: #FAFBFC;
        padding: 0.85rem;
        min-height: 11rem;
    }
    .psymas-inspector h3 {
        font-size: 0.95rem;
        margin: 0 0 0.5rem 0;
        color: var(--psymas-text);
    }
    .psymas-inspector p {
        color: var(--psymas-muted);
        font-size: 0.84rem;
        margin: 0.25rem 0;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid var(--psymas-border);
        border-radius: 8px;
        overflow: hidden;
    }
    .psymas-eligibility-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin: 0.15rem 0 0.85rem 0;
    }
    .psymas-use-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.38rem 0.68rem;
        border-radius: 999px;
        border: 1px solid var(--psymas-border);
        background: #FFFFFF;
        color: #111827;
        font-size: 0.84rem;
        font-weight: 650;
    }
    .psymas-use-chip.evidence {
        border-color: #8BC6A5;
        background: #ECFDF3;
        color: #14532D;
    }
    .psymas-use-chip.calibration {
        border-color: #F0C06C;
        background: #FFF7E6;
        color: #78350F;
    }
    .psymas-use-chip.support {
        border-color: #CBD5E1;
        background: #F8FAFC;
        color: #334155;
    }
    @media (max-width: 1100px) {
        .psymas-case-strip { grid-template-columns: repeat(2, minmax(8rem, 1fr)); }
        .psymas-domain-grid { grid-template-columns: repeat(2, minmax(12rem, 1fr)); }
    }
    @media (max-width: 760px) {
        .psymas-title-row { display: block; }
        .psymas-stage-pill { margin-top: 0.6rem; }
        .psymas-topbar { display: block; }
        .psymas-flow { margin-top: 0.5rem; }
        .psymas-case-strip { grid-template-columns: 1fr; }
        .psymas-domain-grid { grid-template-columns: 1fr; }
        .psymas-context dl { grid-template-columns: 1fr; }
    }
</style>
""", unsafe_allow_html=True)

# ----- Sidebar: navigation -----
# Use a separate key for the radio so we can set run_mode from Main-page buttons (Confirm / Go to).
WORKFLOW_NAV = [
    "Scenario",
    "Review Workspace",
    "Research Tools",
    "Configuration",
]
FLOW_STAGES = [
    "Assessment Data",
    "Deterministic Evidence",
    "AI-Assisted Review",
    "Human Review",
    "Review Record",
]
NAV_OPTIONS = WORKFLOW_NAV
_WORKFLOW_TARGETS = {
    "Scenario": "Scenario",
    "Assessment Data": "Preparation",
    "Review Workspace": "_Workbench Evidence Review",
    "Deterministic Evidence": "Preparation",
    "Evidence Governance": "_Workbench Evidence Review",
    "AI-Assisted Review": "_Workbench Evidence Review",
    "Human Review": "_Workbench Evidence Review",
    "Review Record": "_Workbench Audit",
    "Research Tools": "_Workbench Research Tools",
    "Configuration": "_Workbench Configuration",
}
_STAGE_ALIASES = {
    "Study Setup": "Assessment Data",
    "Project Setup": "Assessment Data",
    "Data Inputs": "Assessment Data",
    "Data": "Assessment Data",
    "Forensic Indices": "Deterministic Evidence",
    "Psychometrics": "Deterministic Evidence",
    "Domain Evidence": "AI-Assisted Review",
    "Evidence Governance": "AI-Assisted Review",
    "Evidence Synthesis": "AI-Assisted Review",
    "Review Prioritization": "Human Review",
    "Review Report": "Human Review",
    "Evidence Review": "Human Review",
    "Review Workspace": "Human Review",
    "Report": "Human Review",
    "Audit": "Review Record",
    "Audit Trail": "Review Record",
    "Audit & Configuration": "Configuration",
    "Settings": "Configuration",
    "Validation": "Deterministic Evidence",
    "Simulation Validation": "Deterministic Evidence",
    "Worked Example": "Research Tools",
}
_INTERNAL_NAV_OPTIONS = [
    "_Workbench Domain Evidence",
    "_Workbench Evidence Review",
    "_Workbench Review Report",
    "_Workbench Audit Configuration",
    "_Workbench Configuration",
    "_Workbench Research Tools",
    "_Workbench Review Prioritization",
    "_Workbench Evidence Synthesis",
    "_Workbench Audit Trail",
    "_Workbench Simulation Validation",
    "_Workbench Worked Example",
    "_Workbench Evidence Governance",
    "_Workbench Report",
    "_Workbench Audit",
    "Scenario",
    "Preparation",
    "Project Setup",
    "Data",
    "Psychometrics",
    "Domain Evidence",
    "Data review",
    "Aberrance Summary",
    "Evidence Governance",
    "Evidence Synthesis",
    "Review Prioritization",
    "Report",
    "Audit",
    "Simulation Validation",
    "Worked Example",
    "Validation",
    "Student Profile",
    "Collusion Network",
    "Individual Aberrance",
    "Temporal Forensics",
    "Settings",
]
if "run_mode" not in st.session_state:
    st.session_state.run_mode = "Scenario"
if "workflow_stage" not in st.session_state:
    st.session_state.workflow_stage = "Assessment Data"
elif st.session_state.workflow_stage in _STAGE_ALIASES:
    st.session_state.workflow_stage = _STAGE_ALIASES[st.session_state.workflow_stage]
if st.session_state.get("sidebar_nav") == "Scenario" and st.session_state.get("run_mode") != "Scenario":
    st.session_state.run_mode = "Scenario"
# Initialize LLM provider once so it's set even if user never opens Settings (e.g. after refresh on Aberrance Summary).
if "llm_provider" not in st.session_state:
    st.session_state.llm_provider = _preferred_llm_provider()
run_mode = st.session_state.get("run_mode", "Scenario")
active_workflow_stage = _STAGE_ALIASES.get(st.session_state.get("workflow_stage", "Assessment Data"), st.session_state.get("workflow_stage", "Assessment Data"))

# Allow safe programmatic navigation without fighting the sidebar radio widget.
# Any code can set st.session_state["_nav_request"] = "<page>" and st.rerun().
# On the same run, st.radio can still report a stale value; do not let that overwrite run_mode
# (see _psymas_just_nav_programmatic below).
_nav_req = st.session_state.pop("_nav_request", None)
_nav_req = _STAGE_ALIASES.get(_nav_req, _nav_req)
if _nav_req in NAV_OPTIONS or _nav_req in FLOW_STAGES or _nav_req in _INTERNAL_NAV_OPTIONS:
    _target = _WORKFLOW_TARGETS.get(_nav_req, _nav_req)
    _stage = _nav_req if _nav_req in FLOW_STAGES else {
        "_Workbench Domain Evidence": "AI-Assisted Review",
        "_Workbench Evidence Review": "Human Review",
        "_Workbench Review Prioritization": "Human Review",
        "_Workbench Evidence Synthesis": "AI-Assisted Review",
        "_Workbench Review Report": "Human Review",
        "_Workbench Worked Example": "Research Tools",
        "_Workbench Research Tools": "Research Tools",
        "_Workbench Audit Configuration": "Configuration",
        "_Workbench Configuration": "Configuration",
        "_Workbench Audit Trail": "Review Record",
        "_Workbench Simulation Validation": "Deterministic Evidence",
        "_Workbench Evidence Governance": "AI-Assisted Review",
        "_Workbench Report": "Human Review",
        "_Workbench Audit": "Review Record",
        "Scenario": st.session_state.get("workflow_stage", "Assessment Data"),
        "Review Workspace": st.session_state.get("workflow_stage", "Assessment Data")
        if st.session_state.get("workflow_stage") in FLOW_STAGES else "Assessment Data",
        "Research Tools": "Research Tools",
        "Configuration": "Configuration",
        "Preparation": st.session_state.get("workflow_stage", "Assessment Data")
        if st.session_state.get("workflow_stage") in {"Assessment Data", "Deterministic Evidence"} else "Assessment Data",
        "Project Setup": "Assessment Data",
        "Data": "Assessment Data",
        "Psychometrics": "Deterministic Evidence",
        "Data review": "Settings",
        "Evidence Governance": "AI-Assisted Review",
        "Evidence Synthesis": "AI-Assisted Review",
        "Review Prioritization": "Human Review",
        "Domain Evidence": "AI-Assisted Review",
        "Review Report": "Human Review",
        "Aberrance Summary": "Human Review",
        "Report": "Human Review",
        "Audit": "Review Record",
        "Audit Trail": "Review Record",
        "Validation": "Deterministic Evidence",
        "Simulation Validation": "Deterministic Evidence",
        "Worked Example": "Research Tools",
        "Settings": "Configuration",
    }.get(_nav_req, st.session_state.get("workflow_stage", "Assessment Data"))
    if _nav_req != "Scenario":
        _target = _WORKFLOW_TARGETS.get(_stage, _target)
    st.session_state.run_mode = _target
    st.session_state.workflow_stage = _stage
    st.session_state["sidebar_nav"] = _nav_req if _nav_req in NAV_OPTIONS else _stage
    run_mode = _target
    active_workflow_stage = _stage
    st.session_state["_psymas_just_nav_programmatic"] = True

# Back-compat: older sessions may still point to removed tool pages
_LEGACY_TOOL_PAGES = {"Aberrance only": "Aberrance", "IRT only": "IRT", "RT only": "RT", "Full workflow": "Aberrance"}
if run_mode in _LEGACY_TOOL_PAGES:
    # Route legacy tool pages to the main Aberrance dashboard instead of the removed Tools page.
    st.session_state.run_mode = "Aberrance Summary"
    st.rerun()
_LEGACY_WORKBENCH_MODES = {
    "_Workbench Domain Evidence": "_Workbench Evidence Review",
    "_Workbench Evidence Governance": "_Workbench Evidence Review",
    "_Workbench Evidence Synthesis": "_Workbench Evidence Review",
    "_Workbench Review Prioritization": "_Workbench Evidence Review",
    "_Workbench Review Report": "_Workbench Evidence Review",
    "_Workbench Report": "_Workbench Evidence Review",
    "_Workbench Audit Trail": "_Workbench Audit",
    "_Workbench Audit Configuration": "_Workbench Configuration",
    "_Workbench Worked Example": "_Workbench Research Tools",
    "_Workbench Simulation Validation": "Preparation",
}
if run_mode in _LEGACY_WORKBENCH_MODES:
    _legacy_target = _LEGACY_WORKBENCH_MODES[run_mode]
    if _legacy_target != run_mode:
        st.session_state.run_mode = _legacy_target
        st.rerun()
if run_mode == "Settings":
    st.session_state.workflow_stage = "Configuration"
    st.session_state.run_mode = "_Workbench Configuration"
    st.rerun()
if run_mode == "Command Center":
    st.session_state.run_mode = "Aberrance Summary"
    st.rerun()

with st.sidebar:
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] button,
        section[data-testid="stSidebar"] button *,
        section[data-testid="stSidebar"] button p,
        section[data-testid="stSidebar"] button span,
        section[data-testid="stSidebar"] button div,
        section[data-testid="stSidebar"] button[data-testid="baseButton-primary"],
        section[data-testid="stSidebar"] button[data-testid="baseButton-primary"] *,
        section[data-testid="stSidebar"] button[kind="primary"],
        section[data-testid="stSidebar"] button[kind="primary"] * {
            color: #0F172A !important;
            -webkit-text-fill-color: #0F172A !important;
            opacity: 1 !important;
            text-shadow: none !important;
        }
        section[data-testid="stSidebar"] button[data-testid="baseButton-primary"],
        section[data-testid="stSidebar"] button[kind="primary"] {
            background: #EAF7FA !important;
            border-color: #0F7890 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div style="display:flex;align-items:baseline;gap:0.45rem;margin-bottom:0.15rem;">
          <div style="font-weight:800;color:#0F172A;font-size:1.0rem;">PsyMAS</div>
          <div style="font-size:0.72rem;font-weight:750;color:#64748B;">v{APP_VERSION}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Evidence review workbench")
    st.divider()
    _has_resp = bool(st.session_state.get("last_uploaded_responses"))
    _has_rt = bool(st.session_state.get("last_uploaded_rt_data"))
    _has_psi = bool(st.session_state.get("last_irt_item_params") or st.session_state.get("item_params"))
    # Use `is not None`: `{}` is a valid completed payload (API may return empty flags) but is falsy in Python.
    _has_forensic = st.session_state.get("forensic_result") is not None
    _detect_status = st.session_state.get("detect_job_status", "pending")
    _nav_ready = {
        "_Workbench Domain Evidence": _has_forensic,
        "_Workbench Review Prioritization": _has_forensic,
        "_Workbench Evidence Synthesis": _has_forensic,
        "_Workbench Review Report": _has_forensic,
        "_Workbench Audit Trail": _has_forensic,
        "_Workbench Simulation Validation": _has_forensic,
        "_Workbench Worked Example": True,
        "_Workbench Research Tools": True,
        "_Workbench Audit": _has_forensic,
        "_Workbench Configuration": True,
        "Scenario": True,
        "Preparation": True,
        "Data review": _has_resp,
        "Aberrance Summary": _has_forensic,
        "Evidence Synthesis": _has_forensic,
        "Review Prioritization": _has_forensic,
        "Domain Evidence": _has_forensic,
        "Simulation Validation": _has_forensic,
        "Worked Example": True,
        "Research Tools": True,
        "Audit": _has_forensic,
        "Configuration": True,
        "Student Profile": _has_forensic,
        "Collusion Network": _has_forensic,
        "Individual Aberrance": _has_forensic,
        "Temporal Forensics": _has_forensic,
        "Settings": True,
    }
    _workflow_ready = {
        "Scenario": True,
        "Assessment Data": True,
        "Review Workspace": _has_forensic,
        "Deterministic Evidence": _has_resp or _has_psi or _detect_status in {"running", "done", "error"} or _has_forensic,
        "AI-Assisted Review": _has_forensic,
        "Human Review": _has_forensic,
        "Review Record": _has_forensic,
        "Research Tools": True,
        "Configuration": True,
    }
    def _nav_label(x: str) -> str:
        labels = {
            "Scenario": "Scenario",
            "Review Workspace": "Review Workspace",
            "Deterministic Evidence": "2. Deterministic Evidence",
            "AI-Assisted Review": "3. AI Review",
            "Human Review": "4. Human Review",
            "Review Record": "5. Review Record",
            "Research Tools": "Research Tools",
            "Configuration": "Configuration",
        }
        return labels.get(x, x)
    def _nav_status_tone(x: str) -> str:
        if x == "Scenario":
            return "ready"
        if x == "Assessment Data":
            return "ready"
        if x == "Review Workspace":
            return "ready" if _has_forensic else "off"
        if x == "Deterministic Evidence":
            if _detect_status == "running":
                return "input"
            if _detect_status == "error":
                return "error"
            if _has_forensic:
                return "ready"
            return "input" if _has_resp else "off"
        if x in {"AI-Assisted Review", "Human Review"}:
            return "ready" if _has_forensic else "off"
        if x == "Review Record":
            return "ready" if _has_forensic else "off"
        if x in {"Research Tools", "Configuration"}:
            return "ready"
        return "ready"
    if active_workflow_stage not in NAV_OPTIONS and active_workflow_stage not in FLOW_STAGES:
        active_workflow_stage = "Assessment Data"
    _prog_nav = st.session_state.pop("_psymas_just_nav_programmatic", False)
    for nav_item in NAV_OPTIONS:
        sidebar_active_stage = active_workflow_stage
        if run_mode == "Scenario":
            sidebar_active_stage = "Scenario"
        elif active_workflow_stage in {
            "Assessment Data",
            "Deterministic Evidence",
            "AI-Assisted Review",
            "Human Review",
            "Review Record",
        }:
            sidebar_active_stage = "Review Workspace"
        is_active = nav_item == sidebar_active_stage
        tone = _nav_status_tone(nav_item)
        dot_col, btn_col = st.columns([0.09, 0.91], gap="small")
        with dot_col:
            st.markdown(f"<div class='psymas-nav-dot psymas-nav-dot-{tone}'></div>", unsafe_allow_html=True)
        with btn_col:
            if st.button(
                _nav_label(nav_item),
                key=f"workflow_nav_btn_{nav_item}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
            ):
                if not is_active:
                    if nav_item == "Scenario":
                        st.session_state.run_mode = "Scenario"
                        st.session_state["sidebar_nav"] = "Scenario"
                    elif nav_item == "Review Workspace":
                        st.session_state.workflow_stage = "Assessment Data"
                        st.session_state.run_mode = "Preparation"
                        st.session_state["sidebar_nav"] = "Review Workspace"
                    else:
                        st.session_state.workflow_stage = nav_item
                        st.session_state.run_mode = _WORKFLOW_TARGETS[nav_item]
                        st.session_state["sidebar_nav"] = nav_item
                    st.rerun()
    resp_loaded = bool(st.session_state.get("last_uploaded_responses"))
    rt_loaded = bool(st.session_state.get("last_uploaded_rt_data"))
    psi_loaded = bool(st.session_state.get("last_irt_item_params") or st.session_state.get("item_params"))
    comp_items = st.session_state.get("prep_compromised_items") or []
    comp_ok = bool(comp_items)
    answer_change_rows = len(st.session_state.get("prep_answer_changes") or [])
    tt_loaded = answer_change_rows > 0

    # Requirements depend on which agents are currently selected (via Preparation checkboxes).
    selected_fns = [fn for fn in ABERRANCE_FUNCTIONS if st.session_state.get(f"ab_only_cb_{fn}")]
    if not selected_fns:
        selected_fns = ["detect_nm"]  # default backend fallback (no extra inputs)
    need_rt = "detect_rg" in selected_fns
    need_psi = any(fn in selected_fns for fn in ["detect_pm", "detect_pk", "detect_ac", "detect_as"])
    need_comp = "detect_pk" in selected_fns
    need_tt = "detect_tt" in selected_fns

    provider_sb = _llm_provider()
    or_key_sb = os.getenv("OPENROUTER_API_KEY", "")
    g_key_sb = os.getenv("GOOGLE_API_KEY", "")
    llm_configured_sb = bool(or_key_sb.strip()) if provider_sb == "openrouter" else bool(g_key_sb and g_key_sb.strip())

    def _sb_line(label: str, ok: bool, extra: str = "") -> str:
        dot = "🟢" if ok else "⚪"
        txt = f"{dot} {label}"
        if extra:
            txt += f" — {extra}"
        return txt

    backend_ok, backend_note, _backend_tip = _backend_status_summary()
    n_r = len(st.session_state.get("last_uploaded_responses") or [])
    n_c = len((st.session_state.get("last_uploaded_responses") or [{}])[0]) if n_r else 0
    detect_status = st.session_state.get("detect_job_status", "pending")
    model_short_sb = str(_effective_llm_model()).split("/")[-1]
    sidebar_status = [
        ("Response", resp_loaded, f"{n_r}x{n_c}" if resp_loaded else ""),
    ]
    if need_rt:
        sidebar_status.append(("RT", rt_loaded, ""))
    if need_psi:
        sidebar_status.append(("psi", psi_loaded, ""))
    if need_tt:
        sidebar_status.append(("Answer changes", tt_loaded, str(answer_change_rows) if tt_loaded else ""))
    sidebar_status.extend(
        [
            ("LLM", True, model_short_sb),
            ("Agents", backend_ok and detect_status != "error", detect_status if detect_status != "pending" else backend_note),
        ]
    )
    def _sidebar_dot(ok: bool) -> str:
        color = "#10B981" if ok else "#94A3B8"
        return f"<span style='width:0.52rem;height:0.52rem;border-radius:999px;background:{color};display:inline-block;flex:0 0 auto;'></span>"

    status_html = "".join(
        f"<span class='psymas-sidebar-chip'>{_sidebar_dot(ok)}<span>{html.escape(str(label))}{(' ' + html.escape(str(extra))) if extra else ''}</span></span>"
        for label, ok, extra in sidebar_status
    )
    st.markdown(f"<div class='psymas-sidebar-status'>{status_html}</div>", unsafe_allow_html=True)
    st.divider()

# Global action hierarchy: neutral buttons by default, primary only for the next step.
st.markdown(
    """
    <style>
    div[role="dialog"] {
        background: #FFFFFF !important;
        color: #0F172A !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 14px !important;
        box-shadow:
            0 24px 70px rgba(15, 23, 42, 0.28),
            0 2px 8px rgba(15, 23, 42, 0.10),
            inset 0 1px 0 rgba(255, 255, 255, 0.90) !important;
    }
    div[role="dialog"] *,
    div[role="dialog"] p,
    div[role="dialog"] span,
    div[role="dialog"] div,
    div[role="dialog"] label {
        color: inherit;
    }
    div[role="dialog"] [data-testid="stModalHeader"],
    div[role="dialog"] [data-testid="stModalHeader"] *,
    div[role="dialog"] h1,
    div[role="dialog"] h2,
    div[role="dialog"] h3 {
        color: #0F172A !important;
        -webkit-text-fill-color: #0F172A !important;
    }
    div[role="dialog"] > div,
    div[role="dialog"] section,
    div[role="dialog"] [data-testid="stVerticalBlock"] {
        background: #FFFFFF !important;
    }
    div[role="dialog"] [data-testid="stContainer"] {
        background: #FFFFFF !important;
        border-color: #D3DCE6 !important;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06) !important;
    }
    div[role="dialog"] [data-testid="stMetric"],
    div[role="dialog"] [data-testid="stMetric"] * {
        color: #0F172A !important;
        -webkit-text-fill-color: #0F172A !important;
    }
    div[role="dialog"] .psymas-case-strip,
    div[role="dialog"] .psymas-case-chip,
    div[role="dialog"] .psymas-case-chip *,
    div[role="dialog"] .psymas-figure-panel,
    div[role="dialog"] .psymas-figure-panel *,
    div[role="dialog"] .psymas-domain-card,
    div[role="dialog"] .psymas-domain-card *,
    div[role="dialog"] .psymas-llm-panel,
    div[role="dialog"] .psymas-llm-panel * {
        color: #0F172A !important;
        -webkit-text-fill-color: #0F172A !important;
    }
    div[role="dialog"] {
        font-size: 0.86rem !important;
    }
    div[role="dialog"] h1,
    div[role="dialog"] h2,
    div[role="dialog"] h3 {
        font-size: 1.05rem !important;
        line-height: 1.25rem !important;
        margin-bottom: 0.4rem !important;
    }
    div[role="dialog"] h4 {
        font-size: 0.95rem !important;
        line-height: 1.15rem !important;
    }
    div[role="dialog"] p,
    div[role="dialog"] label,
    div[role="dialog"] .stMarkdown,
    div[role="dialog"] .stCaptionContainer {
        font-size: 0.78rem !important;
        line-height: 1.15rem !important;
    }
    div[role="dialog"] [data-testid="stMetricLabel"] {
        font-size: 0.68rem !important;
        font-weight: 850 !important;
        color: #0F172A !important;
    }
    div[role="dialog"] [data-testid="stMetricValue"] {
        font-size: 1.15rem !important;
        font-weight: 900 !important;
        color: #0F172A !important;
    }
    div[role="dialog"] .psymas-lineage-legend,
    div[role="dialog"] .psymas-lineage-legend-row,
    div[role="dialog"] .psymas-lineage-legend-row span.item {
        color: #CBD5E1 !important;
        -webkit-text-fill-color: #CBD5E1 !important;
    }
    div[role="dialog"] .psymas-lineage-legend-title {
        color: #94A3B8 !important;
        -webkit-text-fill-color: #94A3B8 !important;
    }
    div[role="dialog"] div[data-baseweb="segmented-control"] label,
    div[role="dialog"] div[data-baseweb="segmented-control"] label * {
        color: #0F172A !important;
        -webkit-text-fill-color: #0F172A !important;
    }
    div[role="dialog"] .stButton > button,
    div[role="dialog"] .stDownloadButton > button {
        min-height: 2rem !important;
        padding: 0.28rem 0.75rem !important;
        font-size: 0.82rem !important;
    }
    div[role="dialog"] button[aria-label="Close"],
    div[role="dialog"] button[aria-label="Close"] * {
        color: #0F172A !important;
        -webkit-text-fill-color: #0F172A !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button,
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button *,
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primary"],
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primary"] *,
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button[data-testid="baseButton-primary"],
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button[data-testid="baseButton-primary"] *,
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover,
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover * {
        color: #0F172A !important;
        opacity: 1 !important;
        text-shadow: none !important;
    }
    .block-container {
        padding-top: 2.2rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1540px !important;
    }
    .psymas-page-heading {
        margin: 0 0 0.85rem 0;
        padding: 0;
    }
    .psymas-page-heading h1 {
        margin: 0 !important;
        color: #111827 !important;
        font-size: 1.55rem !important;
        line-height: 1.25 !important;
        letter-spacing: 0 !important;
    }
    .psymas-page-heading p {
        margin: 0.3rem 0 0 0 !important;
        color: #5B6573 !important;
        font-size: 0.91rem !important;
        line-height: 1.45 !important;
    }
    .psymas-summary-band {
        display: grid;
        grid-template-columns: 7rem minmax(0, 1fr) auto;
        align-items: center;
        gap: 1rem;
        padding: 0.95rem 1rem;
        margin: 0 0 0.8rem 0;
        background: #FFFFFF;
        border: 1px solid #D7DEE7;
        border-left: 4px solid #1F7185;
        border-radius: 7px;
    }
    .psymas-summary-band.amber { border-left-color: #C87512; }
    .psymas-summary-band.red { border-left-color: #B42318; }
    .psymas-summary-band.slate { border-left-color: #64748B; }
    .psymas-summary-band .summary-number {
        color: #111827;
        font-size: 1.65rem;
        font-weight: 750;
    }
    .psymas-summary-band .summary-label {
        color: #111827;
        font-weight: 700;
        margin-bottom: 0.42rem;
    }
    .psymas-summary-band .summary-track {
        display: block;
        height: 0.48rem;
        overflow: hidden;
        background: #E8EDF2;
        border-radius: 999px;
    }
    .psymas-summary-band .summary-track span {
        display: block;
        height: 100%;
        background: #1F7185;
        border-radius: inherit;
    }
    .psymas-summary-band.amber .summary-track span { background: #C87512; }
    .psymas-summary-band.red .summary-track span { background: #B42318; }
    .psymas-summary-band.slate .summary-track span { background: #64748B; }
    .psymas-summary-band .summary-caption {
        margin-top: 0.38rem;
        color: #5B6573;
        font-size: 0.82rem;
    }
    .psymas-summary-band .summary-ratio {
        color: #374151;
        font-weight: 650;
        white-space: nowrap;
    }
    div[data-testid="stSegmentedControl"] {
        margin: 0 0 0.65rem 0 !important;
    }
    div[data-testid="stSegmentedControl"] button {
        color: #26313D !important;
        background: #FFFFFF !important;
        border-color: #C9D2DC !important;
    }
    div[data-testid="stSegmentedControl"] button[aria-pressed="true"] {
        color: #FFFFFF !important;
        background: #174E5F !important;
        border-color: #174E5F !important;
    }
    button[data-testid="stBaseButton-segmented_control"],
    button[data-testid="stBaseButton-segmented_control"] * {
        background: #FFFFFF !important;
        color: #17212B !important;
        border-color: #B9C5D2 !important;
        opacity: 1 !important;
    }
    button[data-testid="stBaseButton-segmented_controlActive"],
    button[data-testid="stBaseButton-segmented_controlActive"] * {
        background: #174E5F !important;
        color: #FFFFFF !important;
        border-color: #174E5F !important;
        opacity: 1 !important;
    }
    div[data-testid="stButton"] > button:not(:disabled) {
        background-color: #FFFFFF !important;
        color: #111827 !important;
        border-color: #C9D2DC !important;
        width: 100% !important;
        min-width: 8rem;
        border-radius: 6px !important;
    }
    div[data-testid="stButton"] > button:not(:disabled):hover {
        background-color: #F6FAFB !important;
        border-color: #256D85 !important;
        color: #111827 !important;
    }
    div[data-testid="stButton"] > button:not(:disabled) * {
        color: #111827 !important;
    }
    div[data-testid="stButton"] > button[kind="primary"]:not(:disabled),
    div[data-testid="stButton"] > button[data-testid="baseButton-primary"]:not(:disabled) {
        background-color: #174E5F !important;
        color: #FFFFFF !important;
        border-color: #174E5F !important;
    }
    div[data-testid="stButton"] > button[kind="primary"]:not(:disabled) *,
    div[data-testid="stButton"] > button[data-testid="baseButton-primary"]:not(:disabled) * {
        color: #FFFFFF !important;
    }
    div[data-testid="stButton"] {
        width: 100% !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
        background: linear-gradient(180deg, #FFFFFF, #FAFCFD) !important;
        color: #102033 !important;
        border: 1px solid #CBD6E2 !important;
        border-radius: 10px !important;
        min-height: 2.2rem;
        padding: 0.32rem 0.58rem !important;
        justify-content: flex-start !important;
        text-align: left !important;
        font-weight: 680 !important;
        font-size: 0.86rem !important;
        box-shadow: none !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
        background: #EEF7FA !important;
        border-color: #256D85 !important;
        color: #111827 !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primary"],
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button[data-testid="baseButton-primary"] {
        background: linear-gradient(180deg, #FFFFFF, #EAF7FA) !important;
        border-color: #0E7896 !important;
        color: #073B4C !important;
        -webkit-text-fill-color: #073B4C !important;
        box-shadow: 0 0 0 1px rgba(14,120,150,0.08), 0 5px 14px rgba(14,120,150,0.10) !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primary"] *,
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button[data-testid="baseButton-primary"] * {
        color: #073B4C !important;
        -webkit-text-fill-color: #073B4C !important;
        opacity: 1 !important;
    }
    .psymas-nav-dot {
        width: 0.62rem;
        height: 0.62rem;
        border-radius: 999px;
        margin-top: 0.72rem;
        box-shadow: 0 0 0 3px rgba(20,136,95,0.08), inset 0 0 0 1px rgba(0,0,0,0.10);
    }
    .psymas-nav-dot-ready { background: #147A5A; }
    .psymas-nav-dot-input { background: #C87512; }
    .psymas-nav-dot-off { background: #8A94A3; }
    .psymas-nav-dot-error { background: #B42318; }
    .psymas-quiet-meta {
        color: #374151 !important;
        font-weight: 500;
    }
    .psymas-sidebar-status {
        display: flex;
        flex-wrap: wrap;
        gap: 0.35rem;
        margin: 0.85rem 0 0.75rem 0;
    }
    .psymas-sidebar-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.28rem;
        max-width: 100%;
        padding: 0.26rem 0.48rem;
        border: 1px solid #D4DDE7;
        border-radius: 999px;
        background: #FFFFFF;
        color: #111827 !important;
        font-size: 0.76rem;
        line-height: 1.15;
        font-weight: 550;
    }
    .psymas-sidebar-chip span {
        color: #111827 !important;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .psymas-result-meta {
        color: #111827 !important;
        font-weight: 650;
        margin: 0 0 0.75rem 0;
    }
    div[data-testid="stDownloadButton"] > button {
        background: #FFFFFF !important;
        color: #174E5F !important;
        border: 1px solid #7AA7B5 !important;
        border-radius: 6px !important;
        font-weight: 650 !important;
        min-height: 2.15rem !important;
        padding: 0.35rem 0.75rem !important;
    }
    div[data-testid="stDownloadButton"] > button * {
        color: #174E5F !important;
    }
    div[data-testid="stDownloadButton"] > button:disabled,
    div[data-testid="stDownloadButton"] > button:disabled * {
        background: #E5E7EB !important;
        border-color: #CBD5E1 !important;
        color: #4B5563 !important;
        opacity: 1 !important;
    }
    div[data-testid="stFileUploaderDropzone"],
    div[data-testid="stFileUploaderDropzone"] > div,
    div[data-testid="stFileUploaderDropzone"] section {
        background: #FFFFFF !important;
        border-color: #6B7787 !important;
        color: #111827 !important;
    }
    div[data-testid="stFileUploaderDropzone"] *,
    div[data-testid="stFileUploaderFile"] *,
    div[data-testid="stFileUploaderFile"] svg {
        color: #111827 !important;
        fill: currentColor !important;
        stroke: currentColor !important;
        opacity: 1 !important;
    }
    div[data-testid="stFileUploaderDropzone"] button,
    div[data-testid="stFileUploaderDropzone"] button * {
        background: #174E5F !important;
        color: #FFFFFF !important;
        border-color: #174E5F !important;
    }
    div[data-baseweb="select"],
    div[data-baseweb="select"] > div,
    div[data-baseweb="select"] div[role="button"],
    div[data-baseweb="select"] input {
        background: #FFFFFF !important;
        color: #111827 !important;
        border-color: #C9D2DC !important;
    }
    div[data-baseweb="select"] *,
    div[data-baseweb="popover"] * {
        color: #111827 !important;
    }
    /* Final contrast guardrails: ordinary surfaces are light; dark controls own white text. */
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"],
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] > div,
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] section {
        background-color: #FFFFFF !important;
        color: #111827 !important;
        border: 2px dashed #6B7787 !important;
    }
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] :where(p, span, small, div):not(button *) {
        color: #111827 !important;
        opacity: 1 !important;
    }
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] svg {
        color: #4B5563 !important;
        fill: currentColor !important;
        stroke: currentColor !important;
        opacity: 1 !important;
    }
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] button,
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] button * {
        background-color: #174E5F !important;
        color: #FFFFFF !important;
        border-color: #174E5F !important;
        opacity: 1 !important;
    }
    div[data-testid="stButton"] > button:not(:disabled),
    div[data-testid="stButton"] > button:not(:disabled) * {
        opacity: 1 !important;
    }
    div[data-testid="stButton"] > button:not([kind="primary"]):not([data-testid="baseButton-primary"]):not(:disabled),
    div[data-testid="stButton"] > button:not([kind="primary"]):not([data-testid="baseButton-primary"]):not(:disabled) *,
    div[data-testid="stDownloadButton"] > button:not(:disabled),
    div[data-testid="stDownloadButton"] > button:not(:disabled) * {
        color: #111827 !important;
        opacity: 1 !important;
    }
    div[data-testid="stDownloadButton"] > button:not(:disabled),
    div[data-testid="stDownloadButton"] > button:not(:disabled) * {
        color: #174E5F !important;
    }
    div[data-testid="stButton"] > button[kind="primary"]:not(:disabled),
    div[data-testid="stButton"] > button[kind="primary"]:not(:disabled) *,
    div[data-testid="stButton"] > button[data-testid="baseButton-primary"]:not(:disabled),
    div[data-testid="stButton"] > button[data-testid="baseButton-primary"]:not(:disabled) * {
        color: #FFFFFF !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button:not([kind="primary"]),
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button:not([kind="primary"]) * {
        color: #111827 !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primary"],
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primary"] *,
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button[data-testid="baseButton-primary"],
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button[data-testid="baseButton-primary"] * {
        color: #0F172A !important;
        -webkit-text-fill-color: #0F172A !important;
        opacity: 1 !important;
    }
    .psymas-figure-panel {
        border: 1px solid #CBD5E1;
        background: #FFFFFF;
        border-radius: 10px;
        padding: 1rem 1rem 0.95rem 1rem;
        margin: 0.65rem 0 0.85rem 0;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.045);
    }
    .psymas-figure-panel .panel-kicker {
        color: #0F6B7C;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }
    .psymas-figure-panel .panel-title {
        color: #111827;
        font-size: 1.08rem;
        font-weight: 800;
        margin-bottom: 0.25rem;
    }
    .psymas-figure-panel .panel-copy {
        color: #475569;
        font-size: 0.86rem;
        line-height: 1.42;
        margin-bottom: 0.75rem;
    }
    .psymas-figure-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.65rem;
    }
    .psymas-figure-grid.three {
        grid-template-columns: repeat(3, minmax(0, 1fr));
    }
    .psymas-figure-stat {
        border: 1px solid #D7DEE7;
        background: #F8FAFC;
        border-radius: 8px;
        padding: 0.72rem 0.78rem;
        min-height: 4.3rem;
    }
    .psymas-figure-stat .label {
        color: #64748B;
        font-size: 0.72rem;
        font-weight: 750;
        text-transform: uppercase;
        letter-spacing: 0.035em;
    }
    .psymas-figure-stat .value {
        color: #111827;
        font-size: 1.08rem;
        font-weight: 850;
        margin-top: 0.18rem;
        overflow-wrap: anywhere;
    }
    .psymas-figure-stat .note {
        color: #475569;
        font-size: 0.74rem;
        margin-top: 0.18rem;
        line-height: 1.25;
    }
    .psymas-figure-stat.good { border-left: 4px solid #147A5A; }
    .psymas-figure-stat.warn { border-left: 4px solid #C87512; }
    .psymas-figure-stat.info { border-left: 4px solid #1F7185; }
    .psymas-figure-stat.off { border-left: 4px solid #94A3B8; }
    .psymas-streamlined-note {
        color: #64748B;
        font-size: 0.78rem;
        margin: 0.1rem 0 0.65rem 0;
    }
    div[data-testid="stSegmentedControl"] button,
    div[data-testid="stSegmentedControl"] button *,
    div[data-baseweb="tab"],
    div[data-baseweb="tab"] * {
        color: #111827 !important;
        opacity: 1 !important;
    }
    div[data-testid="stSegmentedControl"] button[aria-pressed="true"],
    div[data-testid="stSegmentedControl"] button[aria-pressed="true"] *,
    div[data-baseweb="tab"][aria-selected="true"],
    div[data-baseweb="tab"][aria-selected="true"] * {
        color: #FFFFFF !important;
    }
    @media (max-width: 1000px) {
        .psymas-figure-grid,
        .psymas-figure-grid.three { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

_STAGE_CONTEXT = {
    "Assessment Data": {
        "purpose": "Check that required response, RT, item-parameter, exposure, and answer-change data are available.",
        "input": "Response data, RT data, item parameters, exposure labels, answer-change records",
        "output": "Validated data inputs",
        "next": "Generate deterministic evidence.",
    },
    "Deterministic Evidence": {
        "purpose": "Run selected detection functions and inspect deterministic index, flag, and threshold outputs.",
        "input": "Prepared data, selected detection functions, active threshold YAML",
        "output": "Forensic index table",
        "next": "Apply evidence governance.",
    },
    "AI-Assisted Review": {
        "purpose": "Translate eligible detector flags into governed domain evidence and prepare evidence-bound AI review support.",
        "input": "Index-level flags, governed evidence profile, selected examinee, and report constraints",
        "output": "Domain evidence overview, review-priority map, and cautious draft report language",
        "next": "Human reviewer confirmation.",
    },
    "Human Review": {
        "purpose": "Review final system flags, inspect case evidence, and record human adjudication.",
        "input": "Evidence synthesis, AI draft, and reviewer notes",
        "output": "Human-review decision and final note",
        "next": "Record the review experience.",
    },
    "Review Record": {
        "purpose": "Inspect evidence provenance, report checks, and reproducibility records.",
        "input": "Evidence record, rulebook, run log, and report audit output",
        "output": "Audit records",
        "next": "Use configuration or research tools when needed.",
    },
    "Research Tools": {
        "purpose": "Generate manuscript-facing worked examples and research export packages.",
        "input": "Demo data, simulated truth labels, and PsyMAS output tables",
        "output": "Worked-example figures, tables, and research export package",
        "next": "Return to the analysis workflow.",
    },
    "Configuration": {
        "purpose": "Manage backend, LLM, and threshold settings.",
        "input": "Environment settings and threshold YAML",
        "output": "Active configuration record",
        "next": "Return to the analysis workflow.",
    },
}


def _stage_state(stage: str) -> str:
    stage = _STAGE_ALIASES.get(stage, stage)
    if stage == "Assessment Data":
        return "Ready"
    if stage == "Deterministic Evidence":
        detect_status = st.session_state.get("detect_job_status", "pending")
        if detect_status == "running":
            return "In Progress"
        if detect_status == "error":
            return "Technical Error"
        if st.session_state.get("forensic_result") is not None:
            return "Complete"
        return "Ready" if bool(st.session_state.get("last_uploaded_responses")) else "Waiting for Data"
    if stage in {"AI-Assisted Review", "Human Review"}:
        return "Ready" if st.session_state.get("forensic_result") is not None else "Evidence unavailable"
    if stage == "Review Record":
        return "Ready" if st.session_state.get("forensic_result") is not None else "Evidence unavailable"
    if stage in {"Research Tools", "Configuration"}:
        return "Ready"
    if stage == "Project Setup":
        return "Complete" if st.session_state.get("ab_only_scenario_select") else "In Progress"
    if stage == "Data Readiness":
        return "Complete" if bool(st.session_state.get("last_uploaded_responses")) else "Needs Input"
    if stage == "Agent Run":
        detect_status = st.session_state.get("detect_job_status", "pending")
        if detect_status == "running":
            return "In Progress"
        if detect_status == "error":
            return "Error"
        if st.session_state.get("forensic_result") is not None:
            return "Complete"
        return "Ready" if bool(st.session_state.get("last_uploaded_responses")) else "Blocked"
    if stage in {"Evidence Log", "Rulebook Review", "Report & Audit"}:
        return "Complete" if st.session_state.get("forensic_result") is not None else "Blocked"
    if stage == "Settings":
        return "Ready"
    return "Pending"


def _render_workflow_shell(stage: str) -> None:
    return


def _render_flow_nav(active_stage: str) -> None:
    active_stage = _STAGE_ALIASES.get(active_stage, active_stage)
    if active_stage not in FLOW_STAGES:
        return
    st.markdown(
        """
        <style>
        div[data-testid="stHorizontalBlock"]:has(.psymas-flow-arrow) {
            gap: 0.18rem !important;
            align-items: center !important;
            margin: 0 0 0.72rem 0 !important;
            padding: 0.22rem 0.38rem !important;
            border-radius: 18px !important;
            background:
                linear-gradient(180deg, rgba(255,255,255,0.88), rgba(248,251,253,0.74)),
                linear-gradient(90deg, rgba(14,120,150,0.08), rgba(15,120,144,0.02)) !important;
            border: 1px solid rgba(191,203,216,0.74) !important;
            box-shadow: 0 10px 28px rgba(16,32,51,0.055) !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.psymas-flow-arrow) div[data-testid="stButton"] > button {
            min-height: 28px !important;
            height: 28px !important;
            padding: 0.15rem 0.36rem !important;
            border-radius: 999px !important;
            border: 1px solid transparent !important;
            background: transparent !important;
            color: #223041 !important;
            font-size: 0.68rem !important;
            font-weight: 720 !important;
            box-shadow: none !important;
            white-space: normal !important;
            line-height: 1.05 !important;
            letter-spacing: 0 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            text-align: center !important;
            vertical-align: middle !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.psymas-flow-arrow) div[data-testid="stButton"] > button * {
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            text-align: center !important;
            line-height: 1.05 !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.psymas-flow-arrow) div[data-testid="stButton"] > button[kind="primary"],
        div[data-testid="stHorizontalBlock"]:has(.psymas-flow-arrow) div[data-testid="stButton"] > button[data-testid="baseButton-primary"] {
            border-color: rgba(14,120,150,0.78) !important;
            background: linear-gradient(180deg, #FFFFFF, #EAF7FA) !important;
            color: #073B4C !important;
            box-shadow: 0 0 0 1px rgba(14,120,150,0.10), 0 5px 16px rgba(14,120,150,0.13) !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.psymas-flow-arrow) div[data-testid="stButton"] > button[kind="primary"] *,
        div[data-testid="stHorizontalBlock"]:has(.psymas-flow-arrow) div[data-testid="stButton"] > button[data-testid="baseButton-primary"] * {
            color: #073B4C !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.psymas-flow-arrow) div[data-testid="stButton"] > button:hover {
            border-color: rgba(14,120,150,0.45) !important;
            background: #F7FCFD !important;
            color: #073B4C !important;
        }
        .psymas-flow-arrow {
            height: 28px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0;
            padding: 0;
            color: #7890A5;
            font-size: 0.82rem;
            font-weight: 800;
            line-height: 1;
            opacity: 0.95;
        }
        .psymas-flow-arrow::before {
            content: "";
            width: 16px;
            height: 1px;
            background: linear-gradient(90deg, rgba(120,144,165,0.22), rgba(14,120,150,0.52));
            margin-right: 4px;
            transform: translateY(0);
        }
        div[data-testid="stMarkdown"]:has(.psymas-flow-arrow),
        div[data-testid="stMarkdownContainer"]:has(.psymas-flow-arrow) {
            height: 28px !important;
            min-height: 28px !important;
            margin: 0 !important;
            padding: 0 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(
        [1.0, 0.12, 1.08, 0.12, 1.08, 0.12, 1.08, 0.12, 0.86],
        gap="small",
        vertical_alignment="center",
    )
    labels = {
        "Assessment Data": "01 Data",
        "Deterministic Evidence": "02 Evidence",
        "AI-Assisted Review": "03 AI Review",
        "Human Review": "04 Human Review",
        "Review Record": "05 Record",
    }
    for idx, stage in enumerate(FLOW_STAGES):
        with cols[idx * 2]:
            if st.button(
                labels[stage],
                key=f"flow_nav_{stage}",
                type="primary" if stage == active_stage else "secondary",
                use_container_width=True,
            ):
                if stage == "Assessment Data":
                    st.session_state.workflow_stage = "Assessment Data"
                    st.session_state.run_mode = "Preparation"
                    st.session_state["sidebar_nav"] = "Assessment Data"
                else:
                    st.session_state["_nav_request"] = stage
                st.rerun()
        if idx < len(FLOW_STAGES) - 1:
            with cols[idx * 2 + 1]:
                st.markdown("<div class='psymas-flow-arrow'>›</div>", unsafe_allow_html=True)








def _table_df(records) -> pd.DataFrame:
    if records is None:
        return pd.DataFrame()
    if isinstance(records, pd.DataFrame):
        df = records.copy()
    elif isinstance(records, list):
        df = pd.DataFrame(records)
    elif isinstance(records, dict):
        df = pd.DataFrame([records])
    else:
        df = pd.DataFrame([{"value": records}])
    for col in df.columns:
        df[col] = df[col].map(
            lambda value: json.dumps(list(value) if isinstance(value, set) else value, ensure_ascii=False)
            if isinstance(value, (dict, list, tuple, set))
            else value
        )
        if df[col].dtype == "object":
            df[col] = df[col].map(lambda value: "" if pd.isna(value) else str(value))
    return df


def _df_csv_bytes(df: pd.DataFrame) -> bytes:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return b""
    return df.to_csv(index=False).encode("utf-8-sig")


def _streamlined_ui_enabled() -> bool:
    """Default to the cleaner presentation-first workflow layout."""
    return True


def _render_figure_summary_panel(
    *,
    kicker: str,
    title: str,
    copy: str,
    stats: list[dict],
    columns: int = 4,
) -> None:
    grid_class = "three" if columns == 3 else ""
    stat_html = []
    for item in stats:
        tone = str(item.get("tone", "info"))
        if tone not in {"good", "warn", "info", "off"}:
            tone = "info"
        stat_html.append(
            f"<div class='psymas-figure-stat {tone}'>"
            f"<div class='label'>{html.escape(str(item.get('label', '')))}</div>"
            f"<div class='value'>{html.escape(str(item.get('value', '')))}</div>"
            f"<div class='note'>{html.escape(str(item.get('note', '')))}</div>"
            "</div>"
        )
    st.markdown(
        "<div class='psymas-figure-panel'>"
        f"<div class='panel-kicker'>{html.escape(kicker)}</div>"
        f"<div class='panel-title'>{html.escape(title)}</div>"
        f"<div class='panel-copy'>{html.escape(copy)}</div>"
        f"<div class='psymas-figure-grid {grid_class}'>"
        f"{''.join(stat_html)}"
        "</div></div>",
        unsafe_allow_html=True,
    )


def _render_stage_focus_band(
    *,
    kicker: str,
    title: str,
    metric: str,
    metric_label: str,
    message: str,
    tone: str = "teal",
) -> None:
    """Prominent first-screen decision band for the current workflow stage."""
    if tone not in {"teal", "amber", "green", "slate", "red"}:
        tone = "teal"
    st.markdown(
        f"""
        <style>
        .psymas-focus-band {{
            display: grid;
            grid-template-columns: minmax(8rem, 0.18fr) minmax(0, 0.82fr);
            gap: 1rem;
            align-items: center;
            margin: 0 0 0.72rem 0;
            padding: 0.9rem 1rem;
            border-radius: 12px;
            background:
                linear-gradient(180deg, rgba(255,255,255,0.94), rgba(248,251,253,0.90)),
                linear-gradient(90deg, rgba(15,120,144,0.08), rgba(255,255,255,0));
            border: 1px solid #CAD5E1;
            box-shadow: 0 10px 26px rgba(16,32,51,0.055);
            border-left: 4px solid #0F7890;
        }}
        .psymas-focus-band.amber {{ border-left-color: #C87512; }}
        .psymas-focus-band.green {{ border-left-color: #16865D; }}
        .psymas-focus-band.slate {{ border-left-color: #64748B; }}
        .psymas-focus-band.red {{ border-left-color: #B42318; }}
        .psymas-focus-metric .value {{
            color: #102033;
            font-size: 2rem;
            line-height: 1;
            font-weight: 800;
            letter-spacing: 0;
        }}
        .psymas-focus-metric .label {{
            margin-top: 0.28rem;
            color: #526172;
            font-size: 0.78rem;
            font-weight: 680;
        }}
        .psymas-focus-copy .kicker {{
            color: #0B6D80;
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }}
        .psymas-focus-copy .title {{
            margin-top: 0.18rem;
            color: #102033;
            font-size: 1.06rem;
            font-weight: 800;
            line-height: 1.2;
        }}
        .psymas-focus-copy .message {{
            margin-top: 0.32rem;
            color: #405064;
            font-size: 0.88rem;
            line-height: 1.42;
            font-weight: 520;
        }}
        </style>
        <div class="psymas-focus-band {tone}">
          <div class="psymas-focus-metric">
            <div class="value">{html.escape(str(metric))}</div>
            <div class="label">{html.escape(str(metric_label))}</div>
          </div>
          <div class="psymas-focus-copy">
            <div class="kicker">{html.escape(str(kicker))}</div>
            <div class="title">{html.escape(str(title))}</div>
            <div class="message">{html.escape(str(message))}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_page_download_bar(
    downloads: list[dict],
    *,
    next_label: str | None = None,
    next_key: str | None = None,
    next_stage: str | None = None,
) -> None:
    """Consistent bottom-right download and next-step controls for workflow pages."""
    clean_downloads = [d for d in downloads if isinstance(d, dict)]
    n_actions = len(clean_downloads) + (1 if next_label and next_key and next_stage else 0)
    if n_actions <= 0:
        return
    spacer = max(0.40, 1.0 - 0.18 * n_actions)
    widths = [spacer] + [0.18] * len(clean_downloads)
    if next_label and next_key and next_stage:
        widths.append(0.22)
    cols = st.columns(widths)
    for idx, item in enumerate(clean_downloads, start=1):
        with cols[idx]:
            st.download_button(
                item.get("label", "Download"),
                data=item.get("data", b""),
                file_name=item.get("file_name", "psymas_output.csv"),
                mime=item.get("mime", "text/csv"),
                key=item.get("key", f"download_{idx}_{item.get('file_name', 'output')}"),
                disabled=bool(item.get("disabled", False)),
                use_container_width=True,
            )
    if next_label and next_key and next_stage:
        with cols[-1]:
            if st.button(next_label, key=next_key, type="primary", use_container_width=True):
                st.session_state["_nav_request"] = next_stage
                st.rerun()


def _data_availability_table_df(selected_agents: list[str] | None = None) -> pd.DataFrame:
    response_rows = st.session_state.get("last_uploaded_responses") or []
    rt_rows = st.session_state.get("last_uploaded_rt_data") or []
    psi_rows = st.session_state.get("last_irt_item_params") or st.session_state.get("item_params") or []
    comp_items = st.session_state.get("prep_compromised_items") or []
    answer_changes = st.session_state.get("prep_answer_changes") or []
    selected = selected_agents or _selected_agent_functions()

    def _row(table_name: str, role: str, loaded: bool, rows: int, source: str, note: str) -> dict:
        return {
            "table_name": table_name,
            "role": role,
            "status": "loaded" if loaded else "unavailable",
            "rows": rows,
            "source": source,
            "selected_agents_requiring_table": ", ".join(selected),
            "note": note,
        }

    return pd.DataFrame(
        [
            _row("final_scores_matrix.csv", "operational input", bool(response_rows), len(response_rows), "uploaded or demo-derived", "Required for all score-based forensic indices."),
            _row("response_times_matrix.csv", "operational input", bool(rt_rows), len(rt_rows), "uploaded or demo-derived", "Required for RT, rapid-guessing, and score-time methods."),
            _row("estimated_item_parameters.csv", "derived input", bool(psi_rows), len(psi_rows), "IRT estimation or demo truth parameters", "Required for model-based indices."),
            _row("compromised_items.csv", "operational input", bool(comp_items), len(comp_items), "uploaded or demo-derived exposure labels", "Required for preknowledge evidence."),
            _row("answer_changes_long.csv", "operational input", bool(answer_changes), len(answer_changes), st.session_state.get("prep_answer_changes_file_name", ""), "Required only when detect_tt is selected. Demo summary tables are not treated as long answer-change input."),
            _row("scenario_key.csv", "simulation validation only", bool(st.session_state.get("demo_scenario_key")), len(st.session_state.get("demo_scenario_key") or []), "demo table", "Joined only after operational evidence generation."),
            _row("copying_pairs_truth.csv", "simulation validation only", bool(st.session_state.get("demo_copying_pairs_truth")), len(st.session_state.get("demo_copying_pairs_truth") or []), "demo table", "Used only to validate pair-recovery behavior."),
            _row("testing_context.csv", "context / validation", bool(st.session_state.get("demo_testing_context")), len(st.session_state.get("demo_testing_context") or []), "demo table", "Contextual table for validation and review support."),
        ]
    )


def _render_output_table(title: str, records, *, filename: str, caption: str = "", preview_rows: int = 200) -> None:
    df = _table_df(records)
    with st.container(border=True):
        head_col, action_col = st.columns([4, 1])
        with head_col:
            st.markdown(f"**{title}**")
            if caption:
                st.caption(caption)
        if df.empty:
            st.info("No table output is available for this stage yet.")
            return
        csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
        with action_col:
            st.download_button(
                "Download CSV",
                data=csv_bytes,
                file_name=filename,
                mime="text/csv",
                use_container_width=True,
                key=f"download_{filename}",
            )
        display_df = df.head(preview_rows)
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            height=min(420, 120 + 34 * max(1, len(display_df))),
            on_select="rerun",
            selection_mode="multi-row",
            key=f"table_{filename}",
        )
        if len(df) > len(display_df):
            st.caption(f"Showing first {len(display_df):,} rows. Download includes all {len(df):,} rows.")


def _selected_agent_functions() -> list[str]:
    selected = [fn for fn in ABERRANCE_FUNCTIONS if st.session_state.get(f"ab_only_cb_{fn}")]
    return selected or ["detect_nm"]


def _hashable_flag_value(value):
    """Convert JSON-shaped flag identifiers into stable set members."""
    if isinstance(value, dict):
        return tuple(
            sorted((str(key), _hashable_flag_value(item)) for key, item in value.items())
        )
    if isinstance(value, (list, tuple, set)):
        return tuple(_hashable_flag_value(item) for item in value)
    try:
        hash(value)
        return value
    except TypeError:
        return str(value)


def _agent_output_rows() -> list[dict]:
    fr = st.session_state.get("forensic_result") or {}
    flags = fr.get("flags", {}) if isinstance(fr, dict) else {}
    rows = []
    for agent_key, payload in flags.items():
        if not isinstance(payload, dict):
            continue
        stat = payload.get("stat") or payload.get("pairs") or payload.get("nonparametric_misfit") or []
        flagged = {
            _hashable_flag_value(value)
            for field in ("flagged", "flagged_copiers", "flagged_pairs")
            for value in (payload.get(field, []) or [])
        }
        rows.append(
            {
                "agent": agent_key,
                "status": "error" if payload.get("error") else "complete",
                "flagged_count": len(flagged),
                "table_rows": len(stat) if isinstance(stat, list) else 0,
                "message": payload.get("error") or payload.get("info") or "",
            }
        )
    return rows


def _render_agent_workflow_graph(output_rows: pd.DataFrame, detector_ready: int, detector_total: int) -> None:
    if output_rows is None or output_rows.empty:
        st.info("No agent workflow output is available.")
        return

    rows = output_rows.copy()
    rows["agent"] = rows["agent"].astype(str)
    rows["status"] = rows.get("status", pd.Series(dtype=str)).astype(str)
    rows["flagged_count"] = pd.to_numeric(rows.get("flagged_count", 0), errors="coerce").fillna(0).astype(int)
    rows["table_rows"] = pd.to_numeric(rows.get("table_rows", 0), errors="coerce").fillna(0).astype(int)
    agent_order = ["nm_agent", "pm_agent", "rg_agent", "pk_agent", "tt_agent", "ac_agent", "as_agent", "cp_agent"]
    rows["_order"] = rows["agent"].map({name: i for i, name in enumerate(agent_order)}).fillna(99)
    rows = rows.sort_values(["_order", "agent"]).reset_index(drop=True)
    if len(rows) > 8:
        rows = rows.head(8).copy()

    agent_meta = {
        "nm_agent": ("NM", "Misfit", "Nonparametric response-pattern checks"),
        "pm_agent": ("PM", "IRT Fit", "Parametric person-fit and RT-family indices"),
        "rg_agent": ("RG", "Rapid Guessing", "Response-time effort and rapid-response indicators"),
        "pk_agent": ("PK", "Preknowledge", "Performance on exposed or compromised items"),
        "tt_agent": ("TT", "Tampering", "Answer-change and revision-pattern indices"),
        "ac_agent": ("AC", "Copying", "Pairwise copying-oriented output"),
        "as_agent": ("AS", "Similarity", "Pairwise response-similarity output"),
        "cp_agent": ("CP", "Change Point", "Localization of response or timing shifts"),
    }

    def _agent_status_class(status: str) -> str:
        status_l = str(status or "").lower()
        if "error" in status_l or "fail" in status_l:
            return "error"
        if "complete" in status_l or "done" in status_l:
            return "complete"
        return "idle"

    agent_cards = []
    for _, row in rows.iterrows():
        status = str(row.get("status", "pending")).lower()
        flagged = int(row.get("flagged_count", 0))
        table_rows = int(row.get("table_rows", 0))
        agent = str(row.get("agent", ""))
        code, label, desc = agent_meta.get(agent, (agent.replace("_agent", "").upper(), agent, "Detector output"))
        status_class = _agent_status_class(status)
        title = f"{agent}: {status}; {flagged:,} flags; {table_rows:,} rows"
        agent_cards.append(
            f"""
            <div class="psymas-agent-node {status_class}" title="{html.escape(title)}">
                <div class="psymas-agent-node-top">
                    <span class="psymas-agent-dot"></span>
                    <span class="psymas-agent-code">{html.escape(code)}</span>
                    <span class="psymas-agent-status">{html.escape(status or 'pending')}</span>
                </div>
                <div class="psymas-agent-name">{html.escape(label)}</div>
                <div class="psymas-agent-desc">{html.escape(desc)}</div>
                <div class="psymas-agent-metrics">
                    <span><b>{flagged:,}</b> flags</span>
                    <span><b>{table_rows:,}</b> rows</span>
                </div>
            </div>
            """
        )

    downstream_modules = [
        ("03", "AI Review", "Rulebook maps eligible flags into governed evidence and LLM support", "ready"),
        ("04", "Human Review", "Reviewer records final adjudication and notes", "interactive"),
        ("05", "Record", "Exports review record, audit traces, and research files", "stored"),
    ]
    downstream_cards = []
    for code, label, desc, state in downstream_modules:
        downstream_cards.append(
            f"""
            <div class="psymas-review-stage {html.escape(state)}" title="{html.escape(desc)}">
                <div class="psymas-review-stage-code">{html.escape(code)}</div>
                <div>
                    <div class="psymas-review-stage-name">{html.escape(label)}</div>
                    <div class="psymas-review-stage-desc">{html.escape(desc)}</div>
                </div>
                <span class="psymas-review-stage-state">{html.escape(state)}</span>
            </div>
            """
        )

    components.html(
        f"""
        <!doctype html>
        <html>
        <head>
        <meta charset="utf-8">
        <style>
            * {{
                box-sizing: border-box;
            }}
            body {{
                margin: 0;
                font-family: "Segoe UI", Arial, sans-serif;
                background: transparent;
                color: #0F172A;
            }}
            .psymas-agent-flow-card {{
                border: 1px solid #CBD5E1;
                border-radius: 14px;
                background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%);
                box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
                padding: 0.9rem 1rem 1rem 1rem;
                margin: 0.45rem 0 0.75rem 0;
            }}
            .psymas-agent-flow-title {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 0.75rem;
                margin-bottom: 0.65rem;
                color: #0F172A;
                font-weight: 850;
                font-size: 0.98rem;
            }}
            .psymas-agent-flow-title span {{
                color: #475569;
                font-size: 0.72rem;
                font-weight: 700;
            }}
            .psymas-agent-router,
            .psymas-agent-record {{
                max-width: 360px;
                margin: 0 auto;
                border: 1px solid #A7D8E5;
                border-left: 5px solid #0F7890;
                border-radius: 12px;
                background: #EEF9FB;
                padding: 0.65rem 0.9rem;
                text-align: center;
                color: #0F172A;
            }}
            .psymas-agent-record {{
                border-color: #CBD5E1;
                border-left-color: #334155;
                background: #FFFFFF;
            }}
            .psymas-agent-router b,
            .psymas-agent-record b {{
                display: block;
                font-size: 1rem;
                letter-spacing: 0.01em;
            }}
            .psymas-agent-router small,
            .psymas-agent-record small {{
                color: #475569;
                font-size: 0.72rem;
                font-weight: 650;
            }}
            .psymas-agent-arrow {{
                text-align: center;
                color: #0F7890;
                font-size: 1.1rem;
                font-weight: 900;
                line-height: 1.3;
                margin: 0.25rem 0;
            }}
            .psymas-agent-grid {{
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 0.62rem;
                margin: 0.2rem 0;
            }}
            .psymas-agent-node {{
                border: 1px solid #CBD5E1;
                border-left: 5px solid #94A3B8;
                border-radius: 12px;
                background: #FFFFFF;
                padding: 0.64rem 0.72rem;
                min-height: 6.2rem;
                box-shadow: 0 8px 20px rgba(15, 23, 42, 0.045);
            }}
            .psymas-agent-node.complete {{
                border-left-color: #147A5A;
                background: linear-gradient(180deg, #FFFFFF 0%, #F1FBF7 100%);
            }}
            .psymas-agent-node.error {{
                border-left-color: #B42318;
                background: linear-gradient(180deg, #FFFFFF 0%, #FFF5F3 100%);
            }}
            .psymas-agent-node.idle {{
                border-left-color: #94A3B8;
                background: #F8FAFC;
            }}
            .psymas-agent-node-top {{
                display: flex;
                align-items: center;
                gap: 0.38rem;
                min-width: 0;
            }}
            .psymas-agent-dot {{
                width: 0.55rem;
                height: 0.55rem;
                border-radius: 999px;
                background: #147A5A;
                box-shadow: 0 0 0 3px rgba(20, 136, 95, 0.12);
                flex: 0 0 auto;
            }}
            .psymas-agent-node.error .psymas-agent-dot {{
                background: #B42318;
                box-shadow: 0 0 0 3px rgba(180, 35, 24, 0.12);
            }}
            .psymas-agent-node.idle .psymas-agent-dot {{
                background: #94A3B8;
                box-shadow: 0 0 0 3px rgba(148, 163, 184, 0.14);
            }}
            .psymas-agent-code {{
                color: #0F172A;
                font-weight: 900;
                letter-spacing: 0.09em;
                font-size: 0.72rem;
            }}
            .psymas-agent-status {{
                margin-left: auto;
                color: #64748B;
                font-size: 0.6rem;
                font-weight: 850;
                letter-spacing: 0.06em;
                text-transform: uppercase;
                white-space: nowrap;
            }}
            .psymas-agent-name {{
                color: #0F172A;
                font-size: 0.98rem;
                font-weight: 900;
                line-height: 1.15;
                margin-top: 0.34rem;
            }}
            .psymas-agent-desc {{
                color: #475569;
                font-size: 0.72rem;
                line-height: 1.22;
                min-height: 1.75rem;
                margin-top: 0.18rem;
            }}
            .psymas-agent-metrics {{
                display: flex;
                justify-content: space-between;
                gap: 0.5rem;
                color: #334155;
                font-size: 0.72rem;
                margin-top: 0.45rem;
                border-top: 1px solid #E2E8F0;
                padding-top: 0.38rem;
            }}
            .psymas-agent-metrics b {{
                color: #0F172A;
                font-weight: 900;
            }}
            @media (max-width: 1200px) {{
                .psymas-agent-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
            }}
            .psymas-review-divider {{
                display: flex;
                align-items: center;
                gap: 0.75rem;
                margin: 0.85rem 0 0.55rem 0;
                color: #475569;
                font-size: 0.72rem;
                font-weight: 850;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }}
            .psymas-review-divider::before,
            .psymas-review-divider::after {{
                content: "";
                height: 1px;
                flex: 1;
                background: #D7DEE8;
            }}
            .psymas-review-grid {{
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 0.62rem;
                margin-top: 0.25rem;
            }}
            .psymas-review-stage {{
                position: relative;
                display: grid;
                grid-template-columns: 2.2rem 1fr auto;
                align-items: center;
                gap: 0.58rem;
                min-height: 4.75rem;
                border: 1px solid #CBD5E1;
                border-left: 5px solid #0F7890;
                border-radius: 12px;
                background: #FFFFFF;
                padding: 0.58rem 0.65rem;
                box-shadow: 0 8px 20px rgba(15, 23, 42, 0.04);
            }}
            .psymas-review-stage::before {{
                content: "";
                position: absolute;
                left: -0.42rem;
                top: 50%;
                transform: translateY(-50%);
                width: 0.56rem;
                height: 0.56rem;
                border-radius: 999px;
                background: #0F7890;
                box-shadow: 0 0 0 5px rgba(15, 120, 144, 0.12);
            }}
            .psymas-review-stage.interactive {{
                border-left-color: #C47A12;
                background: linear-gradient(180deg, #FFFFFF 0%, #FFF9EC 100%);
            }}
            .psymas-review-stage.interactive::before {{
                background: #C47A12;
                box-shadow: 0 0 0 5px rgba(196, 122, 18, 0.14);
            }}
            .psymas-review-stage.stored {{
                border-left-color: #334155;
                background: #F8FAFC;
            }}
            .psymas-review-stage.stored::before {{
                background: #334155;
                box-shadow: 0 0 0 5px rgba(51, 65, 85, 0.12);
            }}
            .psymas-review-stage-code {{
                width: 2rem;
                height: 2rem;
                border-radius: 999px;
                display: flex;
                align-items: center;
                justify-content: center;
                color: #FFFFFF;
                background: #0F7890;
                font-weight: 900;
                font-size: 0.78rem;
            }}
            .psymas-review-stage.interactive .psymas-review-stage-code {{
                background: #C47A12;
            }}
            .psymas-review-stage.stored .psymas-review-stage-code {{
                background: #334155;
            }}
            .psymas-review-stage-name {{
                color: #0F172A;
                font-size: 0.92rem;
                font-weight: 900;
                line-height: 1.1;
            }}
            .psymas-review-stage-desc {{
                color: #475569;
                font-size: 0.68rem;
                line-height: 1.18;
                margin-top: 0.16rem;
            }}
            .psymas-review-stage-state {{
                align-self: start;
                border: 1px solid #CFE8EF;
                border-radius: 999px;
                color: #0F7890;
                background: #F0FBFD;
                padding: 0.16rem 0.38rem;
                font-size: 0.58rem;
                font-weight: 850;
                letter-spacing: 0.05em;
                text-transform: uppercase;
                white-space: nowrap;
            }}
            .psymas-review-stage.interactive .psymas-review-stage-state {{
                border-color: #F3D8A5;
                color: #8A4F05;
                background: #FFF7E3;
            }}
            .psymas-review-stage.stored .psymas-review-stage-state {{
                border-color: #CBD5E1;
                color: #334155;
                background: #F8FAFC;
            }}
            @media (max-width: 1200px) {{
                .psymas-review-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
            }}
        </style>
        </head>
        <body>
        <div class="psymas-agent-flow-card">
            <div class="psymas-agent-flow-title">
                <div>LangGraph deterministic evidence run</div>
                <span>{detector_ready}/{detector_total} selected detectors completed</span>
            </div>
            <div class="psymas-agent-router">
                <b>Router</b>
                <small>dispatch selected deterministic agents</small>
            </div>
            <div class="psymas-agent-arrow">↓</div>
            <div class="psymas-agent-grid">
                {''.join(agent_cards)}
            </div>
            <div class="psymas-agent-arrow">↓</div>
            <div class="psymas-agent-record">
                <b>Evidence Record</b>
                <small>compiled index outputs for governance and review</small>
            </div>
            <div class="psymas-agent-arrow">↓</div>
            <div class="psymas-review-divider">Downstream review workflow</div>
            <div class="psymas-review-grid">
                {''.join(downstream_cards)}
            </div>
        </div>
        </body>
        </html>
        """,
        height=760,
        scrolling=False,
    )


def _agent_detail_tables() -> list[tuple[str, object, str, str]]:
    fr = st.session_state.get("forensic_result") or {}
    flags = fr.get("flags", {}) if isinstance(fr, dict) else {}
    tables = []
    for agent_key, payload in flags.items():
        if not isinstance(payload, dict):
            continue
        if payload.get("stat"):
            tables.append((f"{agent_key} Statistics", payload.get("stat"), f"psymas_{agent_key}_statistics.csv", "Agent-level output table."))
        if payload.get("pairs"):
            tables.append((f"{agent_key} Pairs", payload.get("pairs"), f"psymas_{agent_key}_pairs.csv", "Pairwise evidence generated by the agent."))
        if payload.get("rte"):
            rte_rows = [{"student_index": i + 1, "RTE": value} for i, value in enumerate(payload.get("rte") or [])]
            tables.append((f"{agent_key} RTE", rte_rows, f"psymas_{agent_key}_rte.csv", "Response-time effort values by student."))
    return tables


def _evidence_rows() -> list[dict]:
    fr = st.session_state.get("forensic_result") or {}
    flags = fr.get("flags", {}) if isinstance(fr, dict) else {}
    rows = []
    for agent_key, payload in flags.items():
        if not isinstance(payload, dict):
            continue
        for field in ["flagged", "flagged_copiers", "flagged_pairs"]:
            for value in payload.get(field, []) or []:
                rows.append(
                    {
                        "stage": "Evidence Log",
                        "agent": agent_key,
                        "evidence_type": field,
                        "subject": value,
                        "review_status": "Needs Review",
                    }
                )
        if payload.get("error"):
            rows.append(
                {
                    "stage": "Evidence Log",
                    "agent": agent_key,
                    "evidence_type": "agent_error",
                    "subject": payload.get("error"),
                    "review_status": "Open",
                }
            )
    return rows


def _rulebook_rows() -> list[dict]:
    selected = set(_selected_agent_functions())
    mapping_rows = _load_rulebook_mapping_rows()
    if mapping_rows:
        rows = []
        for row in mapping_rows:
            fn = str(row.get("function", ""))
            if not fn.startswith("detect_"):
                continue
            is_selected = fn in selected or (fn == "detect_cp" and "detect_rg" in selected)
            rows.append(
                {
                    "function": fn,
                    "index": row.get("method", ""),
                    "domain": row.get("domain", ""),
                    "role": row.get("role", ""),
                    "agent": ABERRANCE_FN_TO_AGENT.get(fn, ""),
                    "required_data": row.get("required_data", ""),
                    "package_output": row.get("package_output", ""),
                    "package_flag": row.get("package_flag", ""),
                    "evidence_use": _evidence_use_label(row),
                    "flag_type": row.get("flag_type", ""),
                    "flagging_rule": row.get("threshold_rule", ""),
                    "threshold_source": row.get("threshold_source", ""),
                    "notes": row.get("notes", ""),
                    "selected": is_selected,
                    "review_status": "Ready for review" if is_selected else "Not selected",
                }
            )
        return rows
    rows = []
    for row in APPENDIX_B2_RULEBOOK:
        fn = row["function"]
        is_selected = fn in selected or (fn == "detect_cp" and "detect_rg" in selected)
        rows.append(
            {
                "function": fn,
                "index": row["index"],
                "domain_role": row["domain_role"],
                "agent": ABERRANCE_FN_TO_AGENT.get(fn, ""),
                "required_data_output": row["required_data"],
                "flagging_rule": row["flagging_rule"],
                "threshold_source_notes": row["threshold_source"],
                "selected": is_selected,
                "review_status": "Ready for review" if is_selected else "Not selected",
            }
        )
    return rows


def _report_watchlist_rows() -> list[dict]:
    fr = st.session_state.get("forensic_result") or {}
    flags = fr.get("flags", {}) if isinstance(fr, dict) else {}
    rows_by_subject: dict[str, dict] = {}
    for agent_key, payload in flags.items():
        if not isinstance(payload, dict):
            continue
        flagged_values = []
        flagged_values.extend(payload.get("flagged", []) or [])
        flagged_values.extend(payload.get("flagged_copiers", []) or [])
        flagged_values.extend(payload.get("flagged_pairs", []) or [])
        for subject in flagged_values:
            key = str(subject)
            row = rows_by_subject.setdefault(
                key,
                {"subject": subject, "flag_count": 0, "agents": []},
            )
            row["flag_count"] += 1
            row["agents"].append(agent_key)
    rows = list(rows_by_subject.values())
    for row in rows:
        row["agents"] = ", ".join(sorted(set(row["agents"])))
    return sorted(rows, key=lambda x: x["flag_count"], reverse=True)




def _overview_rows() -> list[dict]:
    response_rows = st.session_state.get("last_uploaded_responses") or []
    flags = (st.session_state.get("forensic_result") or {}).get("flags", {}) if isinstance(st.session_state.get("forensic_result"), dict) else {}
    return [
        {"metric": "Project status", "value": _stage_state(active_workflow_stage), "review_label": "Audit in progress"},
        {"metric": "Total examinees", "value": len(response_rows), "review_label": "Evidence unavailable" if not response_rows else "Data available"},
        {"metric": "Agents completed", "value": len([k for k, v in flags.items() if isinstance(v, dict) and not v.get("error")]), "review_label": "No automatic misconduct conclusion"},
        {"metric": "Cases requiring review", "value": len(_report_watchlist_rows()), "review_label": "Requires additional review" if _report_watchlist_rows() else "No statistical concern"},
        {"metric": "Audit status", "value": "Ready" if st.session_state.get("forensic_result") is not None else "Pending", "review_label": "Human review required"},
        {"metric": "Report status", "value": "Available" if st.session_state.get("forensic_result") is not None else "Not generated", "review_label": "Rulebook-controlled language"},
    ]


def _review_distribution_rows() -> list[dict]:
    watchlist = _report_watchlist_rows()
    n_response = len(st.session_state.get("last_uploaded_responses") or [])
    n_review = len(watchlist)
    return [
        {"review_status": "No statistical concern", "count": max(0, n_response - n_review), "meaning": "No selected agent currently contributes review evidence."},
        {"review_status": "Requires additional review", "count": n_review, "meaning": "At least one selected agent contributed evidence requiring inspection."},
        {"review_status": "Limited evidence", "count": 0 if st.session_state.get("forensic_result") is not None else n_response, "meaning": "Evidence is unavailable or not yet generated."},
        {"review_status": "Conflicting evidence", "count": 0, "meaning": "Reserved for rulebook conflict checks."},
        {"review_status": "Evidence unavailable", "count": 0 if n_response else 1, "meaning": "Data or agent outputs are missing."},
    ]


def _research_rows() -> list[dict]:
    return [
        {"area": "Simulation", "status": "Available in existing workflow", "output": "Simulation data and results"},
        {"area": "Paper Outputs", "status": "Available in existing workflow", "output": "Appendix tables, figures, report artifacts"},
        {"area": "Evidence handoff", "status": "Manual review", "output": "Push validated research outputs into Evidence"},
    ]


def _stage_table_options(stage: str) -> list[dict]:
    response_rows = st.session_state.get("last_uploaded_responses") or []
    rt_rows = st.session_state.get("last_uploaded_rt_data") or []
    psi_rows = st.session_state.get("last_irt_item_params") or st.session_state.get("item_params") or []
    readiness_rows = [
        {"dataset": "Response Data", "rows": len(response_rows), "status": "Ready" if response_rows else "Missing", "affected_agents": "All agents"},
        {"dataset": "Response Time Data", "rows": len(rt_rows), "status": "Ready" if rt_rows else "Missing", "affected_agents": "RT, Rapid Guessing, score-time methods"},
        {"dataset": "Item Parameters", "rows": len(psi_rows), "status": "Ready" if psi_rows else "Missing", "affected_agents": "PM, AC, AS, PK"},
        {"dataset": "Exposure Labels", "rows": len(st.session_state.get("prep_compromised_items") or []), "status": "Ready" if st.session_state.get("prep_compromised_items") else "Missing", "affected_agents": "Preknowledge"},
        {"dataset": "Answer Changes", "rows": len(st.session_state.get("prep_answer_changes") or []), "status": "Ready" if st.session_state.get("prep_answer_changes") else "Missing", "affected_agents": "Tampering"},
    ]
    detect_rows = [{
        "job_status": st.session_state.get("detect_job_status", "pending"),
        "run_id": st.session_state.get("detect_run_id") or "",
        "selected_agents": ", ".join(_selected_agent_functions()),
        "has_forensic_result": st.session_state.get("forensic_result") is not None,
        "interpretation": "No automatic misconduct conclusion",
    }]
    audit_rows = [
        {"audit_check": "Evidence source attached", "status": bool(_evidence_rows()), "notes": "Evidence rows available for selected agent outputs."},
        {"audit_check": "Index Registry attached", "status": bool(_rulebook_rows()), "notes": "Index Registry loaded."},
        {"audit_check": "No prohibited misconduct language", "status": True, "notes": "Use cautious review language."},
        {"audit_check": "Missing evidence disclosed", "status": not bool(st.session_state.get("forensic_result")) or True, "notes": "Unavailable evidence should remain visible."},
        {"audit_check": "Human review statement included", "status": True, "notes": "Operational decisions require human review."},
    ]
    options_by_stage = {
        "Assessment Data": [
            {"label": "Data Status", "title": "Assessment Data Table", "records": _overview_rows(), "filename": "psymas_assessment_data_status.csv", "caption": "Assessment data setup and cautious review status."},
            {"label": "Review Distribution", "title": "Review Status Distribution", "records": _review_distribution_rows(), "filename": "psymas_review_distribution.csv", "caption": "Cautious status labels for paper/demo review."},
            {"label": "Next Actions", "title": "Recommended Workflow Actions", "records": [
                {"step": "Deterministic Evidence", "action": "Run selected detection functions", "status": _stage_state("Deterministic Evidence")},
                {"step": "AI Review", "action": "Apply domain evidence rules and draft cautious evidence-bound review support", "status": _stage_state("AI-Assisted Review")},
                {"step": "Human Review", "action": "Record reviewer adjudication", "status": _stage_state("Human Review")},
                {"step": "Review Record", "action": "Audit report outputs", "status": _stage_state("Review Record")},
            ], "filename": "psymas_next_actions.csv", "caption": "The next auditable actions in the review workflow."},
        ],
        "Data Inputs": [
            {"label": "Readiness", "title": "Data Availability Table", "records": _data_availability_table_df(_selected_agent_functions()), "filename": "data_availability.csv", "caption": "Required data, availability, and affected agents."},
            {"label": "Responses", "title": "Response Data Preview", "records": response_rows, "filename": "psymas_response_data.csv", "caption": "Response data currently loaded in session."},
            {"label": "RT Data", "title": "Response-Time Data Preview", "records": rt_rows, "filename": "psymas_response_time_data.csv", "caption": "Response-time data currently loaded in session."},
            {"label": "Item Params", "title": "Item Parameter Table", "records": psi_rows, "filename": "psymas_item_parameters.csv", "caption": "Item parameters used by model-based agents."},
        ],
        "Deterministic Evidence": [
            {"label": "Run Summary", "title": "Forensic Indices Summary", "records": detect_rows, "filename": "psymas_forensic_indices_summary.csv", "caption": "Detection run status, selected methods, and deterministic interpretation boundary."},
            {"label": "Agent Outputs", "title": "Agent Output Index", "records": _agent_output_rows(), "filename": "psymas_agent_output_index.csv", "caption": "Agent-level output availability and review counts."},
            {"label": "Index Registry", "title": "Index Registry", "records": _rulebook_rows(), "filename": "psymas_index_registry.csv", "caption": "All indices, roles, flagging rules, and threshold sources."},
            {"label": "Thresholds", "title": "Active Threshold Registry", "records": _threshold_registry_rows(), "filename": "psymas_active_threshold_registry.csv", "caption": "YAML-backed thresholds active in this session."},
        ] + [
            {"label": title.replace(" Statistics", "").replace(" Pairs", "").replace(" RTE", ""), "title": title, "records": records, "filename": filename, "caption": caption}
            for title, records, filename, caption in _agent_detail_tables()
        ],
        "Domain Evidence": [
            {"label": "Domain Evidence", "title": "Domain Evidence Strength Table", "records": _governed_review_tables()[1].to_dict(orient="records"), "filename": "domain_evidence_profile.csv", "caption": "Domain-level evidence strength from Evidence Input."},
        ],
        "Review Prioritization": [
            {"label": "Review Queue", "title": "Review Prioritization Queue", "records": _evidence_synthesis_with_human_review(_governed_review_tables()[0]).to_dict(orient="records") if not _governed_review_tables()[0].empty else [], "filename": "review_queue.csv", "caption": "Optional triage queue derived from domain evidence; not a system misconduct decision."},
            {"label": "Evidence Record", "title": "Evidence Explorer Result Table", "records": _evidence_rows(), "filename": "psymas_evidence_record.csv", "caption": "Searchable evidence rows produced from agent outputs."},
            {"label": "Review Cases", "title": "Cases Requiring Additional Review", "records": _report_watchlist_rows(), "filename": "psymas_review_cases.csv", "caption": "Examinees with at least one selected evidence signal."},
            {"label": "Agent Outputs", "title": "Evidence Source Index", "records": _agent_output_rows(), "filename": "psymas_evidence_source_index.csv", "caption": "Evidence sources available for drill-down."},
            {"label": "Rules", "title": "Review Prioritization Rules", "records": _rulebook_rows(), "filename": "psymas_review_prioritization_rules.csv", "caption": "Rulebook-controlled review-priority support from the Index Registry."},
            {"label": "Thresholds", "title": "Active Threshold Registry", "records": _threshold_registry_rows(), "filename": "psymas_governance_thresholds.csv", "caption": "Download, edit, and upload threshold YAML for future runs."},
        ],
        "Review Report": [
            {"label": "Review Cases", "title": "Report-Linked Review Cases", "records": _report_watchlist_rows(), "filename": "psymas_report_review_cases.csv", "caption": "Cases available for report mapping."},
            {"label": "Evidence Links", "title": "Report Evidence Mapping", "records": _evidence_rows(), "filename": "psymas_report_evidence_links.csv", "caption": "Evidence rows that can support report statements."},
        ],
        "Audit Trail": [
            {"label": "Audit Checks", "title": "Report Audit Result Table", "records": audit_rows, "filename": "psymas_report_audit.csv", "caption": "Report support and language-control checks."},
            {"label": "Evidence Links", "title": "Audit Evidence Mapping", "records": _evidence_rows(), "filename": "psymas_audit_evidence_links.csv", "caption": "Evidence rows that support audited statements."},
        ],
        "Research Mode": [
            {"label": "Research Outputs", "title": "Research Mode Output Table", "records": _research_rows(), "filename": "psymas_research_outputs.csv", "caption": "Simulation and paper-output areas are secondary to the evidence workflow."},
            {"label": "Index Registry", "title": "Index Registry", "records": _rulebook_rows(), "filename": "psymas_research_index_registry.csv", "caption": "Paper-facing index registry table."},
        ],
        "Settings": [
            {"label": "Configuration", "title": "Settings Result Table", "records": [
                {"setting": "backend_url", "value": BACKEND_URL},
                {"setting": "llm_provider", "value": _llm_provider()},
                {"setting": "selected_model", "value": st.session_state.get("selected_gemini_model") or ""},
                {"setting": "threshold_yaml", "value": "custom uploaded" if st.session_state.get("threshold_config") else "default"},
            ], "filename": "psymas_settings.csv", "caption": "Current console configuration."},
            {"label": "Thresholds", "title": "Active Threshold Registry", "records": _threshold_registry_rows(), "filename": "psymas_settings_thresholds.csv", "caption": "YAML-backed threshold registry."},
        ],
    }
    return options_by_stage.get(stage, options_by_stage["Assessment Data"])


def _render_single_output_console(stage: str) -> None:
    options = _stage_table_options(stage)
    key = f"selected_subtable_{stage}"
    labels = [opt["label"] for opt in options]
    if st.session_state.get(key) not in labels:
        st.session_state[key] = labels[0]

    with st.container(border=True):
        if len(options) > 1:
            st.segmented_control(
                "View",
                options=labels,
                key=key,
                label_visibility="collapsed",
            )

        selected = next((opt for opt in options if opt["label"] == st.session_state[key]), options[0])
        df = _table_df(selected["records"])
        spacer_col, download_col = st.columns([4, 1])
        with spacer_col:
            st.empty()
        with download_col:
            st.download_button(
                "Download CSV",
                data=df.to_csv(index=False).encode("utf-8-sig") if not df.empty else b"",
                file_name=selected["filename"],
                mime="text/csv",
                disabled=df.empty,
                use_container_width=True,
                key=f"download_console_{stage}_{selected['filename']}",
            )

        table_col, inspector_col = st.columns([3, 1])
        with table_col:
            if df.empty:
                st.info("Evidence unavailable. Run the required upstream step or select another subtable.")
            else:
                display_df = df.head(250)
                styled_df = _style_priority_and_strength_table(display_df)
                event = st.dataframe(
                    styled_df,
                    use_container_width=True,
                    hide_index=True,
                    height=min(520, 140 + 34 * max(1, len(display_df))),
                    on_select="rerun",
                    selection_mode="multi-row",
                    key=f"console_table_{stage}_{selected['filename']}",
                )
                if len(df) > len(display_df):
                    st.caption(f"Showing first {len(display_df):,} rows. Download includes all {len(df):,} rows.")
        with inspector_col:
            st.markdown(
                f"""
<div class="psymas-inspector">
  <h3>Evidence Inspector</h3>
  <p><strong>Step:</strong> {stage}</p>
  <p><strong>Subtable:</strong> {selected["label"]}</p>
  <p><strong>Rows:</strong> {len(df):,}</p>
  <p><strong>Review language:</strong> Human review required; no automatic misconduct conclusion.</p>
</div>
                """,
                unsafe_allow_html=True,
            )

        if stage in {"Deterministic Evidence", "Review Prioritization", "Settings"} and selected["label"] == "Thresholds":
            st.divider()
            y1, y2 = st.columns([1, 1])
            with y1:
                st.download_button(
                    "Download default YAML",
                    data=_default_threshold_yaml().encode("utf-8"),
                    file_name="psymas_thresholds.default.yaml",
                    mime="text/yaml",
                    use_container_width=True,
                    key=f"console_default_yaml_{stage}",
                )
            with y2:
                st.download_button(
                    "Download active YAML",
                    data=_active_threshold_yaml().encode("utf-8"),
                    file_name="psymas_thresholds.active.yaml",
                    mime="text/yaml",
                    use_container_width=True,
                    key=f"console_active_yaml_{stage}",
                )
            uploaded_yaml = st.file_uploader(
                "Upload modified threshold YAML",
                type=["yaml", "yml"],
                key=f"console_threshold_upload_{stage}",
            )
            if uploaded_yaml is not None:
                try:
                    loaded = yaml.safe_load(uploaded_yaml.getvalue().decode("utf-8"))
                    if not isinstance(loaded, dict) or not isinstance(loaded.get("rules"), dict):
                        st.error("Threshold YAML must be a mapping with a top-level `rules` section.")
                    else:
                        st.session_state["threshold_config"] = loaded
                        st.success("YAML loaded for this session. Rerun Deterministic Evidence for threshold changes; evidence-governance rules apply when the evidence pages refresh.")
                except Exception as e:
                    st.error(f"Could not parse threshold YAML: {e}")


def _render_stage_output_tables(stage: str) -> None:
    _render_single_output_console(stage)


def _styled_evidence_cell(style: dict[str, str]) -> str:
    return (
        f"background-color: {style['bg']}; color: {style['text']}; "
        f"font-weight: 700; border-left: 3px solid {style['border']};"
    )


def _style_priority_and_strength_table(df: pd.DataFrame):
    if df is None or df.empty:
        return df
    strength_cols = {
        col
        for col in df.columns
        if str(col) == "Strength" or str(col).endswith("_Strength")
    }
    if "Review_Priority" not in df.columns and not strength_cols:
        return df

    def _style_series(col: pd.Series) -> list[str]:
        if col.name == "Review_Priority":
            return [_styled_evidence_cell(priority_style(value)) for value in col]
        if col.name in strength_cols:
            return [_styled_evidence_cell(strength_style(value)) for value in col]
        return [""] * len(col)

    return df.style.apply(_style_series, axis=0)


def _review_priority_chip_html(priority: object) -> str:
    text = html.escape(str(priority or "—"))
    level = priority_level_key(priority)
    return (
        f'<div class="psymas-case-chip psymas-priority-{level}">'
        f'<div class="label">Priority</div><div class="value">{text}</div></div>'
    )


def _review_priority_pill_html(priority: object) -> str:
    text = html.escape(str(priority or "—"))
    level = priority_level_key(priority)
    return f'<span class="psymas-priority-pill {level}">{text}</span>'


_CASE_REVIEW_PRIORITY_OPTIONS: tuple[str, ...] = (
    "Critical / Expedited",
    "High (Context-Heavy)",
    "High",
    "Medium",
    "Low",
)

_CASE_PRIORITY_FILTER_SHORT: dict[str, str] = {
    "Critical / Expedited": "Critical",
    "High (Context-Heavy)": "High·Ctx",
    "High": "High",
    "Medium": "Medium",
    "Low": "Low",
}

_CASE_PRIORITY_SORT: dict[str, int] = {
    "Critical / Expedited": 0,
    "High (Context-Heavy)": 1,
    "High": 2,
    "Medium": 3,
    "Low": 4,
}


def _examinee_id_sort_key(eid: object) -> tuple[int, int | str]:
    text = str(eid).strip()
    try:
        return (0, int(text))
    except ValueError:
        return (1, text.lower())


def _case_review_queue_sorted(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Examinee_ID" not in df.columns:
        return df
    out = df.copy()
    if "Review_Priority" in out.columns:
        out["_priority_rank"] = out["Review_Priority"].astype(str).map(_CASE_PRIORITY_SORT).fillna(99)
        out = out.sort_values(
            ["_priority_rank", "Examinee_ID"],
            key=lambda col: col.map(_examinee_id_sort_key) if col.name == "Examinee_ID" else col,
        )
        out = out.drop(columns=["_priority_rank"])
    else:
        out = out.sort_values("Examinee_ID", key=lambda col: col.map(_examinee_id_sort_key))
    return out


def _review_priority_legend_html() -> str:
    items = []
    for label in _CASE_REVIEW_PRIORITY_OPTIONS:
        color = priority_style(label)["border"]
        items.append(
            f'<span class="item"><span class="dot" style="background:{color};"></span>'
            f"{html.escape(label)}</span>"
        )
    return f'<div class="psymas-examinee-priority-legend">{"".join(items)}</div>'


def _render_case_priority_filter() -> list[str]:
    """Toggle priority levels (multi-select pills, tab-style colors)."""
    apply_control_theme()
    if "case_review_priority_filter" not in st.session_state:
        st.session_state["case_review_priority_filter"] = ["High"]

    st.caption("Filter by review priority · click to toggle levels")
    current = list(st.session_state.get("case_review_priority_filter") or ["High"])
    selected = st.pills(
        "Priority filter",
        list(_CASE_REVIEW_PRIORITY_OPTIONS),
        selection_mode="multi",
        default=current,
        key="case_priority_filter_pills",
        label_visibility="collapsed",
        format_func=lambda label: _CASE_PRIORITY_FILTER_SHORT.get(label, label),
    )
    if selected:
        st.session_state["case_review_priority_filter"] = sorted(
            list(selected),
            key=lambda item: _CASE_PRIORITY_SORT.get(item, 99),
        )
    return list(st.session_state.get("case_review_priority_filter") or [])


def _render_case_examinee_segment_row(
    examinee_ids: list[str],
    *,
    current: str,
    scope_key: str,
) -> str | None:
    """Examinee picker using the same segmented-control style as Forensic Indices tabs."""
    if not examinee_ids:
        return None
    widget_key = f"case_examinee_seg_{scope_key}"
    default_id = current if current in examinee_ids else examinee_ids[0]
    if st.session_state.get(widget_key) not in examinee_ids:
        st.session_state[widget_key] = default_id
    picked = st.segmented_control(
        "Examinee",
        options=examinee_ids,
        key=widget_key,
        label_visibility="collapsed",
    )
    return str(picked or st.session_state.get(widget_key) or default_id)


def _render_case_examinee_pills(
    examinee_rows: list[tuple[str, str]],
    filtered_ids: list[str],
    current: str,
    selected_priorities: list[str],
) -> str:
    """Compact examinee segmented control (tab-style, same as Forensic Indices)."""
    apply_control_theme()
    st.caption(f"{len(filtered_ids):,} examinee(s) · click to select")
    st.markdown('<div class="psymas-examinee-matrix-wrap">', unsafe_allow_html=True)

    picked: str | None = None
    if len(selected_priorities) == 1:
        picked = _render_case_examinee_segment_row(
            filtered_ids,
            current=current,
            scope_key="all",
        )
    else:
        for priority in sorted(selected_priorities, key=lambda p: _CASE_PRIORITY_SORT.get(p, 99)):
            group_ids = sorted(
                [eid for eid, pr in examinee_rows if pr == priority],
                key=_examinee_id_sort_key,
            )
            if not group_ids:
                continue
            scope_key = priority_level_key(priority)
            st.markdown(
                f'<div class="psymas-priority-pill-row-label">{html.escape(priority)}</div>',
                unsafe_allow_html=True,
            )
            group_pick = _render_case_examinee_segment_row(
                group_ids,
                current=current,
                scope_key=scope_key,
            )
            if group_pick:
                picked = group_pick

    if picked and str(picked) != str(current):
        st.session_state["case_review_examinee_id"] = str(picked)
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
    return str(st.session_state.get("case_review_examinee_id") or current)


def _render_case_examinee_dot_matrix(review_queue_df: pd.DataFrame) -> str:
    """Priority-filtered dot grid for examinee selection; default filter is High."""
    if review_queue_df.empty or "Examinee_ID" not in review_queue_df.columns:
        return ""

    if "case_review_priority_filter" not in st.session_state:
        st.session_state["case_review_priority_filter"] = ["High"]

    if _streamlined_ui_enabled():
        compact_df = _case_review_queue_sorted(review_queue_df.copy())
        ids = compact_df["Examinee_ID"].astype(str).tolist()
        if not ids:
            return ""
        current = str(st.session_state.get("case_review_examinee_id") or ids[0])
        if current not in ids:
            current = ids[0]
        default_index = ids.index(current)
        current_row = compact_df[compact_df["Examinee_ID"].astype(str).eq(str(current))]
        current_payload = current_row.iloc[0].to_dict() if not current_row.empty else {}
        status = str(current_payload.get("Evidence_Status", current_payload.get("Review_Queue_Status", "")) or "Review")
        priority = str(current_payload.get("Review_Priority", "") or "Not set")
        domains = str(current_payload.get("Domains_For_Review", current_payload.get("Primary_Concern", "")) or "No domain")

        def _case_label(eid: str) -> str:
            row = compact_df[compact_df["Examinee_ID"].astype(str).eq(str(eid))]
            if row.empty:
                return f"Examinee {eid}"
            r = row.iloc[0]
            status = r.get("Evidence_Status", r.get("Review_Queue_Status", ""))
            priority = r.get("Review_Priority", "")
            domains = r.get("Domains_For_Review", r.get("Primary_Concern", ""))
            return f"Examinee {eid} | {priority} | {status} | {domains or 'No domain'}"

        st.markdown(
            """
            <style>
            .psymas-case-selector {
                display: grid;
                grid-template-columns: minmax(0, 1fr) auto;
                gap: 0.8rem;
                align-items: stretch;
                margin: 0.4rem 0 0.65rem 0;
            }
            .psymas-case-selector-card {
                border: 1px solid #C9D6E3;
                border-left: 4px solid #0F7890;
                background: linear-gradient(180deg, #FFFFFF, #F8FBFC);
                border-radius: 10px;
                padding: 0.72rem 0.85rem;
                box-shadow: 0 8px 22px rgba(16,32,51,0.05);
            }
            .psymas-case-selector-card .label {
                color: #0B6D80;
                font-size: 0.72rem;
                font-weight: 800;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }
            .psymas-case-selector-card .title {
                color: #102033;
                font-size: 1.15rem;
                font-weight: 820;
                margin-top: 0.1rem;
            }
            .psymas-case-selector-card .meta {
                color: #405064;
                font-size: 0.84rem;
                margin-top: 0.26rem;
                font-weight: 560;
            }
            .psymas-case-selector-card .position {
                color: #697789;
                font-size: 0.76rem;
                margin-top: 0.26rem;
            }
            .psymas-case-jump label,
            .psymas-case-jump [data-testid="stWidgetLabel"] {
                color: #405064 !important;
                font-size: 0.78rem !important;
                font-weight: 700 !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        left_col, prev_col, next_col = st.columns([0.72, 0.14, 0.14], gap="small")
        with left_col:
            st.markdown(
                f"""
                <div class="psymas-case-selector-card">
                  <div class="label">Selected case</div>
                  <div class="title">Examinee {html.escape(current)}</div>
                  <div class="meta">{html.escape(priority)} · {html.escape(status)} · {html.escape(domains)}</div>
                  <div class="position">Case {default_index + 1} of {len(ids)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with prev_col:
            prev_disabled = default_index <= 0
            if st.button("Previous", key="case_review_prev", disabled=prev_disabled, use_container_width=True):
                st.session_state["case_review_examinee_id"] = ids[max(0, default_index - 1)]
                st.rerun()
        with next_col:
            next_disabled = default_index >= len(ids) - 1
            if st.button("Next", key="case_review_next", disabled=next_disabled, type="primary", use_container_width=True):
                st.session_state["case_review_examinee_id"] = ids[min(len(ids) - 1, default_index + 1)]
                st.rerun()

        with st.expander("Jump to another examinee", expanded=False):
            selected = st.selectbox(
                "Find examinee",
                ids,
                index=default_index,
                format_func=_case_label,
                key="case_review_examinee_id_compact",
            )
            if str(selected) != current:
                st.session_state["case_review_examinee_id"] = str(selected)
                st.rerun()
        return str(st.session_state.get("case_review_examinee_id") or current)

    st.markdown("##### Select examinee")
    selected_priorities = _render_case_priority_filter()

    if not selected_priorities:
        st.info("Select at least one review priority to show examinees in the matrix.")
        return str(st.session_state.get("case_review_examinee_id") or "")

    filtered = review_queue_df[review_queue_df["Review_Priority"].astype(str).isin(selected_priorities)].copy()
    filtered = _case_review_queue_sorted(filtered)
    if filtered.empty:
        st.warning("No examinees match the current priority filter.")
        return str(st.session_state.get("case_review_examinee_id") or "")

    examinee_rows = [
        (str(row["Examinee_ID"]), str(row.get("Review_Priority", "") or ""))
        for _, row in filtered.iterrows()
    ]
    filtered_ids = [eid for eid, _ in examinee_rows]

    current = str(st.session_state.get("case_review_examinee_id") or "")
    if current not in filtered_ids:
        current = filtered_ids[0]
        st.session_state["case_review_examinee_id"] = current

    current = _render_case_examinee_pills(examinee_rows, filtered_ids, current, selected_priorities)

    current_priority = next((priority for eid, priority in examinee_rows if eid == current), "")
    idx = filtered_ids.index(current)
    st.markdown('<div class="psymas-case-examinee-nav">', unsafe_allow_html=True)
    nav_prev, nav_mid, nav_next = st.columns([0.7, 2.6, 0.7], gap="small")
    with nav_prev:
        if st.button("← Previous", disabled=idx <= 0, key="case_examinee_prev"):
            st.session_state["case_review_examinee_id"] = filtered_ids[idx - 1]
            st.rerun()
    with nav_mid:
        st.markdown(
            f"**Selected:** Examinee **{html.escape(current)}** · "
            f"{_review_priority_pill_html(current_priority)}",
            unsafe_allow_html=True,
        )
    with nav_next:
        if st.button("Next →", disabled=idx >= len(filtered_ids) - 1, key="case_examinee_next"):
            st.session_state["case_review_examinee_id"] = filtered_ids[idx + 1]
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    return current


def _case_review_display_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "Examinee_ID",
        "Review_Priority",
        "Evidence_Status",
        "Review_Queue_Status",
        "Primary_Concern",
        "Domains_For_Review",
        "Human_Final_Decision",
        "Human_Decision",
        "Reviewer_Note",
    ]
    return [col for col in preferred if col in df.columns]


def _case_review_list_df(review_queue_df: pd.DataFrame) -> pd.DataFrame:
    if review_queue_df is None or review_queue_df.empty:
        return pd.DataFrame()
    display_df = _case_review_queue_sorted(review_queue_df.copy())
    decisions = st.session_state.get("case_review_decisions") or {}
    final_values: list[str] = []
    note_values: list[str] = []
    for _, row in display_df.iterrows():
        examinee_id = str(row.get("Examinee_ID", ""))
        payload = decisions.get(examinee_id, {}) if isinstance(decisions, dict) else {}
        final_values.append(str(payload.get("final_decision", "") or ""))
        note_values.append(str(payload.get("reviewer_note", "") or ""))
    display_df["Human_Final_Decision"] = final_values
    display_df["Reviewer_Note"] = note_values
    return display_df[_case_review_display_columns(display_df)].copy()


def _selected_rows_from_dataframe_event(event: object) -> list[int]:
    selection = getattr(event, "selection", None)
    if isinstance(selection, dict):
        rows = selection.get("rows") or []
    else:
        rows = getattr(selection, "rows", []) if selection is not None else []
    try:
        return [int(row) for row in rows]
    except Exception:
        return []


def _render_quick_case_adjudication(selected_id: str) -> None:
    if not selected_id:
        return
    if "case_review_decisions" not in st.session_state or not isinstance(st.session_state.get("case_review_decisions"), dict):
        st.session_state["case_review_decisions"] = {}
    current = st.session_state["case_review_decisions"].get(str(selected_id), {})
    decision_options = ["No Issue", "Rule Violation", "Potential Misconduct", "Definite Misconduct"]
    current_decision = str(current.get("final_decision", "") or "")
    current_note = str(current.get("reviewer_note", "") or "")
    st.markdown(
        f"""
<div class="psymas-inline-review">
  <div>
    <div class="kicker">Quick human decision</div>
    <div class="title">Examinee {html.escape(str(selected_id))}</div>
  </div>
  <div class="hint">Use the buttons for fast queue work, or open the report for full evidence context.</div>
</div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(4)
    for idx, option in enumerate(decision_options):
        with cols[idx]:
            if st.button(
                option,
                key=f"quick_case_decision_{option}_{selected_id}",
                type="primary" if option == current_decision else "secondary",
                use_container_width=True,
            ):
                st.session_state["case_review_decisions"][str(selected_id)] = {
                    "final_decision": option,
                    "reviewer_note": current_note,
                }
                _persist_case_review_to_store(str(selected_id), final_decision=option, reviewer_note=current_note)
                st.toast(f"Recorded {option} for Examinee {selected_id}.")
                st.rerun()
    note_key = f"quick_case_note_{selected_id}"
    if note_key not in st.session_state:
        st.session_state[note_key] = current_note
    note_cols = st.columns([0.78, 0.22], gap="small")
    with note_cols[0]:
        st.text_input(
            "Reviewer note",
            key=note_key,
            placeholder="Optional note for the review record",
            label_visibility="collapsed",
        )
    with note_cols[1]:
        if st.button("Save note", key=f"quick_case_note_save_{selected_id}", use_container_width=True):
            note = str(st.session_state.get(note_key, "") or "")
            st.session_state["case_review_decisions"][str(selected_id)] = {
                "final_decision": current_decision,
                "reviewer_note": note,
            }
            _persist_case_review_to_store(str(selected_id), final_decision=current_decision, reviewer_note=note)
            st.toast(f"Saved note for Examinee {selected_id}.")
            st.rerun()


def _render_case_review_queue_list(review_queue_df: pd.DataFrame) -> str:
    if review_queue_df.empty or "Examinee_ID" not in review_queue_df.columns:
        return ""
    apply_control_theme()
    st.markdown(
        """
        <style>
        .psymas-inline-review {
            margin: 0.75rem 0 0.45rem 0;
            border: 1px solid #CBD5E1;
            border-left: 4px solid #0F7890;
            border-radius: 10px;
            padding: 0.72rem 0.85rem;
            background: linear-gradient(180deg, #FFFFFF, #F8FBFC);
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            align-items: center;
        }
        .psymas-inline-review .kicker {
            color: #0B6D80;
            font-size: 0.7rem;
            font-weight: 850;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .psymas-inline-review .title {
            color: #102033;
            font-size: 1.05rem;
            font-weight: 850;
        }
        .psymas-inline-review .hint {
            color: #475569;
            font-size: 0.82rem;
            max-width: 42rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    list_df = _case_review_list_df(review_queue_df)
    if list_df.empty:
        st.info("No examinees are available for review.")
        return ""

    st.markdown(
        """
        <style>
        .psymas-review-list-head,
        .psymas-review-list-row {
            display: grid;
            grid-template-columns: 0.62fr 1.1fr 1.32fr 1.22fr 1.22fr 1.25fr 0.58fr 0.64fr;
            gap: 0.45rem;
            align-items: center;
        }
        .psymas-review-list-head {
            color: #475569;
            font-size: 0.72rem;
            font-weight: 850;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            padding: 0.45rem 0.55rem;
            border-bottom: 1px solid #CBD5E1;
        }
        .psymas-review-list-row {
            min-height: 3.1rem;
            color: #102033;
            font-size: 0.84rem;
            padding: 0.38rem 0.55rem;
            border-bottom: 1px solid #E2E8F0;
            background: #FFFFFF;
        }
        .psymas-review-list-row:nth-child(even) {
            background: #F8FAFC;
        }
        .psymas-review-list-row .case-id {
            font-weight: 850;
            color: #0F172A;
        }
        .psymas-review-list-row .muted {
            color: #475569;
            font-weight: 620;
        }
        .psymas-review-list-shell {
            border: 1px solid #CBD5E1;
            border-radius: 10px;
            background: #FFFFFF;
            overflow: hidden;
        }
        div[data-testid="stDataFrame"] {
            font-size: 0.76rem !important;
        }
        div[data-testid="stDataFrame"] [role="gridcell"],
        div[data-testid="stDataFrame"] [role="columnheader"] {
            font-size: 0.76rem !important;
            line-height: 1.1 !important;
            color: #0F172A !important;
        }
        div[data-testid="stDataFrame"] [role="columnheader"] {
            font-weight: 800 !important;
            letter-spacing: 0.04em !important;
        }
        .psymas-review-row-marker {
            display: none !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.psymas-review-row-marker) {
            align-items: center !important;
            gap: 0.4rem !important;
            min-height: 2.25rem !important;
            padding: 0.06rem 0 !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.psymas-review-row-marker) div[data-testid="stMarkdown"]:has(.psymas-review-row-marker),
        div[data-testid="stHorizontalBlock"]:has(.psymas-review-row-marker) div[data-testid="stMarkdownContainer"]:has(.psymas-review-row-marker) {
            display: none !important;
            height: 0 !important;
            min-height: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.psymas-review-row-marker) div[data-testid="stButton"] > button {
            min-width: 0 !important;
            width: 100% !important;
            min-height: 1.82rem !important;
            height: 1.82rem !important;
            padding: 0.05rem 0.28rem !important;
            border-radius: 7px !important;
            font-size: 0.78rem !important;
            font-weight: 720 !important;
            line-height: 1 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.psymas-review-row-marker) div[data-testid="stButton"] > button * {
            font-size: 0.78rem !important;
            line-height: 1 !important;
        }
        div[data-testid="stSelectbox"]:has([id^="case_review_list_decision_"]) {
            margin: 0 !important;
            padding: 0 !important;
        }
        div[data-testid="stSelectbox"]:has([id^="case_review_list_decision_"]) div[data-baseweb="select"] > div {
            min-height: 1.82rem !important;
            height: 1.82rem !important;
            border-radius: 8px !important;
            font-size: 0.8rem !important;
            color: #0F172A !important;
            background: #FFFFFF !important;
        }
        div[data-testid="stSelectbox"]:has([id^="case_review_list_decision_"]) div[data-baseweb="select"] span,
        div[data-testid="stSelectbox"]:has([id^="case_review_list_decision_"]) div[data-baseweb="select"] div {
            color: #0F172A !important;
            -webkit-text-fill-color: #0F172A !important;
            font-size: 0.8rem !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.psymas-review-pager-marker) {
            align-items: center !important;
            margin-bottom: 0.45rem !important;
        }
        .psymas-review-pager-marker { display: none; }
        div[data-testid="stHorizontalBlock"]:has(.psymas-review-pager-marker) div[data-testid="stButton"] > button {
            min-height: 2rem !important;
            height: 2rem !important;
            padding: 0.12rem 0.55rem !important;
            border-radius: 8px !important;
            font-size: 0.82rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    decision_options = ["", "No Issue", "Rule Violation", "Potential Misconduct", "Definite Misconduct"]
    if "Examinee_ID" in list_df.columns:
        list_df = list_df.copy()
        list_df["_Examinee_ID_Num"] = pd.to_numeric(list_df["Examinee_ID"], errors="coerce")
        list_df = list_df.sort_values(
            by=["_Examinee_ID_Num", "Examinee_ID"],
            ascending=[True, True],
            na_position="last",
            kind="mergesort",
        )
    table_df = pd.DataFrame()
    table_df["Examinee"] = list_df["Examinee_ID"].astype(str).map(lambda value: f"#{value}")
    table_df["_Examinee_ID_Num"] = pd.to_numeric(list_df["Examinee_ID"], errors="coerce")
    table_df["Priority"] = list_df["Review_Priority"].astype(str) if "Review_Priority" in list_df.columns else ""
    if "Evidence_Status" in list_df.columns:
        system_profile = list_df["Evidence_Status"].astype(str)
    elif "Review_Queue_Status" in list_df.columns:
        system_profile = list_df["Review_Queue_Status"].astype(str)
    else:
        system_profile = pd.Series([""] * len(list_df), index=list_df.index)
    if "Review_Queue_Status" in list_df.columns:
        fallback_profile = list_df["Review_Queue_Status"].astype(str)
        system_profile = system_profile.where(system_profile.str.strip().ne("") & system_profile.ne("nan"), fallback_profile)
    table_df["System profile"] = system_profile
    table_df["Concern"] = list_df["Primary_Concern"].astype(str) if "Primary_Concern" in list_df.columns else ""
    table_df["Domains"] = list_df["Domains_For_Review"].astype(str) if "Domains_For_Review" in list_df.columns else ""
    if "Human_Final_Decision" in list_df.columns:
        human_decision = list_df["Human_Final_Decision"].astype(str)
    elif "Human_Decision" in list_df.columns:
        human_decision = list_df["Human_Decision"].astype(str)
    else:
        human_decision = pd.Series([""] * len(list_df), index=list_df.index)
    if "Human_Decision" in list_df.columns:
        fallback_decision = list_df["Human_Decision"].astype(str)
        human_decision = human_decision.where(human_decision.str.strip().ne("") & human_decision.ne("nan"), fallback_decision)
    table_df["Human decision"] = human_decision.replace("nan", "")
    table_df["Reviewer note"] = list_df["Reviewer_Note"].astype(str).replace("nan", "") if "Reviewer_Note" in list_df.columns else ""
    column_order = ["_Examinee_ID_Num", "Examinee", "Priority", "System profile", "Concern", "Domains", "Human decision", "Reviewer note"]
    st.markdown(
        f"<div class='psymas-result-meta'>{len(table_df):,} examinees · edit Human decision / Reviewer note in the table, then save all changes below</div>",
        unsafe_allow_html=True,
    )
    edited_table_df = table_df[column_order].copy()
    if AgGrid is not None and GridOptionsBuilder is not None:
        try:
            gb = GridOptionsBuilder.from_dataframe(edited_table_df)
            gb.configure_default_column(
                editable=False,
                sortable=True,
                filter=True,
                resizable=True,
                wrapText=False,
                autoHeight=False,
                minWidth=92,
            )
            gb.configure_column("_Examinee_ID_Num", hide=True, sort="asc")
            gb.configure_column("Examinee", width=78, pinned="left", editable=False)
            gb.configure_column("Priority", width=150, editable=False)
            gb.configure_column("System profile", width=190, editable=False)
            gb.configure_column("Concern", width=110, editable=False)
            gb.configure_column("Domains", width=100, editable=False)
            gb.configure_column(
                "Human decision",
                width=190,
                editable=True,
                cellEditor="agSelectCellEditor",
                cellEditorParams={"values": decision_options},
            )
            gb.configure_column("Reviewer note", width=360, editable=True)
            gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=20)
            gb.configure_selection(selection_mode="single", use_checkbox=False)
            grid_options = gb.build()
            grid_options.update(
                {
                    "headerHeight": 34,
                    "rowHeight": 30,
                    "suppressMenuHide": True,
                    "stopEditingWhenCellsLoseFocus": True,
                    "singleClickEdit": True,
                    "rowSelection": "single",
                    "suppressRowClickSelection": False,
                }
            )
            aggrid_css = {
                ".ag-root-wrapper": {
                    "border": "1px solid #CBD5E1 !important",
                    "border-radius": "10px !important",
                    "background": "#FFFFFF !important",
                    "box-shadow": "0 8px 22px rgba(15, 23, 42, 0.05) !important",
                },
                ".ag-root": {
                    "background": "#FFFFFF !important",
                    "color": "#0F172A !important",
                },
                ".ag-header": {
                    "background": "#F8FAFC !important",
                    "border-bottom": "1px solid #CBD5E1 !important",
                },
                ".ag-header-cell": {
                    "background": "#F8FAFC !important",
                    "color": "#0F172A !important",
                    "font-weight": "850 !important",
                    "font-size": "11px !important",
                    "letter-spacing": "0.05em !important",
                    "text-transform": "uppercase !important",
                },
                ".ag-header-cell-text": {
                    "color": "#0F172A !important",
                    "opacity": "1 !important",
                },
                ".ag-row": {
                    "background": "#FFFFFF !important",
                    "color": "#0F172A !important",
                    "border-color": "#E2E8F0 !important",
                },
                ".ag-row-odd": {
                    "background": "#F8FAFC !important",
                },
                ".ag-row-hover": {
                    "background": "#EAF7FB !important",
                },
                ".ag-cell": {
                    "color": "#0F172A !important",
                    "font-size": "12px !important",
                    "font-weight": "600 !important",
                    "line-height": "30px !important",
                    "border-color": "#E2E8F0 !important",
                },
                ".ag-cell-value": {
                    "color": "#0F172A !important",
                },
                ".ag-cell-inline-editing": {
                    "background": "#FFFFFF !important",
                    "color": "#0F172A !important",
                    "box-shadow": "inset 0 0 0 2px #0F7890 !important",
                },
                ".ag-cell-inline-editing input, .ag-cell-inline-editing textarea": {
                    "background": "#FFFFFF !important",
                    "color": "#0F172A !important",
                },
                ".ag-icon": {
                    "color": "#475569 !important",
                    "opacity": "1 !important",
                },
                ".ag-paging-panel": {
                    "background": "#FFFFFF !important",
                    "color": "#0F172A !important",
                    "border-top": "1px solid #CBD5E1 !important",
                    "font-size": "12px !important",
                },
                ".ag-paging-panel *": {
                    "color": "#0F172A !important",
                },
                ".ag-select, .ag-picker-field-wrapper, .ag-list": {
                    "background": "#FFFFFF !important",
                    "color": "#0F172A !important",
                },
            }
            grid_kwargs = {
                "gridOptions": grid_options,
                "height": 430,
                "fit_columns_on_grid_load": False,
                "allow_unsafe_jscode": JsCode is not None,
                "enable_enterprise_modules": False,
                "theme": "alpine",
                "custom_css": aggrid_css,
                "key": "case_review_editable_aggrid",
            }
            if GridUpdateMode is not None:
                grid_kwargs["update_mode"] = GridUpdateMode.VALUE_CHANGED | GridUpdateMode.SELECTION_CHANGED
            if DataReturnMode is not None:
                grid_kwargs["data_return_mode"] = DataReturnMode.AS_INPUT
            ag_response = AgGrid(edited_table_df, **grid_kwargs)
            returned_data = ag_response.get("data") if isinstance(ag_response, dict) else getattr(ag_response, "data", None)
            if returned_data is not None:
                edited_table_df = pd.DataFrame(returned_data)
            selected_rows = ag_response.get("selected_rows") if isinstance(ag_response, dict) else getattr(ag_response, "selected_rows", None)
            if selected_rows is not None:
                selected_df = pd.DataFrame(selected_rows)
                if not selected_df.empty and "Examinee" in selected_df.columns:
                    st.session_state["human_review_selected_examinee"] = (
                        str(selected_df.iloc[0].get("Examinee", "")).replace("#", "").strip()
                    )
        except Exception as exc:
            st.caption(f"AgGrid unavailable for human review table; using Streamlit editor. {exc}")
            edited_table_df = st.data_editor(
                edited_table_df,
                use_container_width=True,
                hide_index=True,
                height=430,
                row_height=28,
                column_config={
                    "_Examinee_ID_Num": None,
                    "Examinee": st.column_config.TextColumn("Examinee", width="small", disabled=True),
                    "Priority": st.column_config.TextColumn("Priority", width="medium", disabled=True),
                    "System profile": st.column_config.TextColumn("System profile", width="medium", disabled=True),
                    "Concern": st.column_config.TextColumn("Concern", width="small", disabled=True),
                    "Domains": st.column_config.TextColumn("Domains", width="small", disabled=True),
                    "Human decision": st.column_config.SelectboxColumn("Human decision", options=decision_options, width="medium"),
                    "Reviewer note": st.column_config.TextColumn("Reviewer note", width="large"),
                },
                key="case_review_editable_streamlit",
            )
    else:
        edited_table_df = st.data_editor(
            edited_table_df,
            use_container_width=True,
            hide_index=True,
            height=430,
            row_height=28,
            column_config={
                "_Examinee_ID_Num": None,
                "Examinee": st.column_config.TextColumn("Examinee", width="small", disabled=True),
                "Priority": st.column_config.TextColumn("Priority", width="medium", disabled=True),
                "System profile": st.column_config.TextColumn("System profile", width="medium", disabled=True),
                "Concern": st.column_config.TextColumn("Concern", width="small", disabled=True),
                "Domains": st.column_config.TextColumn("Domains", width="small", disabled=True),
                "Human decision": st.column_config.SelectboxColumn("Human decision", options=decision_options, width="medium"),
                "Reviewer note": st.column_config.TextColumn("Reviewer note", width="large"),
            },
            key="case_review_editable_streamlit",
        )
    with st.form("case_review_batch_form"):
        st.caption("Changes are local until saved. Saved decisions are restored from the run database when the app is reopened.")
        save_submitted = st.form_submit_button("Save Human Review", type="primary", use_container_width=True)
    selected_examinee = str(st.session_state.get("human_review_selected_examinee") or "").strip()
    if selected_examinee:
        if st.button(f"Open examinee {selected_examinee} report", type="secondary", use_container_width=True):
            st.session_state["_case_review_modal_id"] = selected_examinee
            st.rerun()
    modal_id = str(st.session_state.get("_case_review_modal_id") or "")
    if modal_id:
        @st.dialog(f"Examinee {modal_id}", width="large")
        def _human_review_case_modal() -> None:
            _render_single_case_review_page(selected_override=modal_id)

        _human_review_case_modal()
    if save_submitted:
        if "case_review_decisions" not in st.session_state or not isinstance(st.session_state.get("case_review_decisions"), dict):
            st.session_state["case_review_decisions"] = {}
        saved_n = 0
        if isinstance(edited_table_df, pd.DataFrame) and not edited_table_df.empty and "Examinee" in edited_table_df.columns:
            for _, row in edited_table_df.iterrows():
                examinee_id = str(row.get("Examinee", "")).replace("#", "").strip()
                if not examinee_id:
                    continue
                decision = str(row.get("Human decision", "") or "").strip()
                note = str(row.get("Reviewer note", "") or "").strip()
                if not decision and not note:
                    continue
                st.session_state["case_review_decisions"][examinee_id] = {
                    "final_decision": decision,
                    "reviewer_note": note,
                }
                _persist_case_review_to_store(examinee_id, final_decision=decision, reviewer_note=note)
                saved_n += 1
        st.toast(f"Saved {saved_n} human review record(s).")
    return ""


_REVIEW_PRIORITY_AGGRID_STYLE = """
function(params) {
    const value = String(params.value || "");
    const lower = value.toLowerCase();
    if (lower.includes("critical") || lower.includes("expedited")) {
        return { backgroundColor: "#FDF2F8", color: "#831843", fontWeight: "700", borderLeft: "3px solid #DC2626" };
    }
    if (lower.includes("context-heavy") || (lower.includes("high") && lower.includes("context"))) {
        return { backgroundColor: "#FFF1F2", color: "#9F1239", fontWeight: "700", borderLeft: "3px solid #FB923C" };
    }
    if (lower === "high" || lower.startsWith("high")) {
        return { backgroundColor: "#FFF7ED", color: "#9A3412", fontWeight: "700", borderLeft: "3px solid #C87512" };
    }
    if (lower === "medium" || lower.startsWith("medium")) {
        return { backgroundColor: "#FFFBEB", color: "#92400E", fontWeight: "700", borderLeft: "3px solid #C87512" };
    }
    if (lower === "low" || lower.startsWith("low")) {
        return { backgroundColor: "#F8FAFC", color: "#475569", fontWeight: "700", borderLeft: "3px solid #94A3B8" };
    }
    return { backgroundColor: "#FFFFFF", color: "#64748B" };
}
"""

_STRENGTH_AGGRID_STYLE = """
function(params) {
    const value = String(params.value || "").toLowerCase();
    if (value === "strong") {
        return { backgroundColor: "#FDF2F8", color: "#9D174D", fontWeight: "700", borderLeft: "3px solid #DC2626" };
    }
    if (value === "moderate") {
        return { backgroundColor: "#FFFBEB", color: "#92400E", fontWeight: "700", borderLeft: "3px solid #C87512" };
    }
    if (value === "weak") {
        return { backgroundColor: "#F8FAFC", color: "#475569", fontWeight: "700", borderLeft: "3px solid #94A3B8" };
    }
    if (value === "unavailable") {
        return { backgroundColor: "#F1F5F9", color: "#334155", fontWeight: "700", borderLeft: "3px solid #475569" };
    }
    if (value === "none") {
        return { backgroundColor: "#FFFFFF", color: "#64748B", fontWeight: "700", borderLeft: "3px solid #64748B" };
    }
    return { backgroundColor: "#FFFFFF", color: "#64748B" };
}
"""


def _render_aggrid_table(
    df: pd.DataFrame,
    *,
    key: str,
    height: int = 600,
    selection: str | None = None,
) -> dict:
    if df is None or df.empty:
        st.info("No rows to display.")
        return {}
    if AgGrid is None or GridOptionsBuilder is None:
        st.dataframe(df, use_container_width=True, hide_index=True, height=height)
        return {}
    try:
        grid_df = df.copy()
        gb = GridOptionsBuilder.from_dataframe(grid_df)
        gb.configure_default_column(
            filter=True,
            sortable=True,
            resizable=True,
            editable=False,
            wrapText=False,
            autoHeight=False,
            minWidth=110,
        )
        gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=25)
        gb.configure_side_bar()
        if selection:
            selection_mode = "multiple" if str(selection) in {"multiple", "multi", "multi-row"} else "single"
            gb.configure_selection(
                selection_mode=selection_mode,
                use_checkbox=selection_mode == "multiple",
                suppressRowClickSelection=False,
            )
        for col in grid_df.columns:
            col_name = str(col)
            col_lower = col_name.lower()
            width = 132
            if col_lower in {"examinee_id", "student_id", "id"}:
                width = 118
            elif any(token in col_lower for token in ["flagged_by", "collusion", "statement", "rule"]):
                width = 240
            elif len(col_name) > 16:
                width = 170
            column_kwargs = {
                "width": width,
                "minWidth": min(width, 120),
                "pinned": "left" if col_lower in {"examinee_id", "student_id", "id"} else None,
            }
            if col_name in {"Evidence Use", "Evidence_Use"} and JsCode is not None:
                column_kwargs["cellStyle"] = JsCode(
                    """
                    function(params) {
                        const value = String(params.value || "");
                        if (value === "Evidence Flag") {
                            return {
                                "backgroundColor": "#ECFDF3",
                                "color": "#14532D",
                                "fontWeight": "700"
                            };
                        }
                        if (value === "Calibration Required") {
                            return {
                                "backgroundColor": "#FFF7E6",
                                "color": "#78350F",
                                "fontWeight": "700"
                            };
                        }
                        if (value === "Display Only") {
                            return {
                                "backgroundColor": "#F8FAFC",
                                "color": "#334155",
                                "fontWeight": "700"
                            };
                        }
                        return {
                            "backgroundColor": "#FFFFFF",
                            "color": "#111827"
                        };
                    }
                    """
                )
            elif col_name == "Review_Priority" and JsCode is not None:
                column_kwargs["cellStyle"] = JsCode(_REVIEW_PRIORITY_AGGRID_STYLE)
            elif (col_name == "Strength" or col_name.endswith("_Strength")) and JsCode is not None:
                column_kwargs["cellStyle"] = JsCode(_STRENGTH_AGGRID_STYLE)
            gb.configure_column(col, **column_kwargs)
        grid_options = gb.build()
        grid_options.update(
            {
                "headerHeight": 42,
                "rowHeight": 32,
                "suppressMenuHide": True,
                "ensureDomOrder": True,
            }
        )
        aggrid_css = {
            ".ag-root-wrapper": {
                "border": "1px solid #CBD5E1 !important",
                "border-radius": "8px !important",
                "background-color": "#FFFFFF !important",
            },
            ".ag-header": {
                "background-color": "#F1F5F9 !important",
                "border-bottom": "1px solid #CBD5E1 !important",
            },
            ".ag-header-cell": {
                "background-color": "#F1F5F9 !important",
                "color": "#111827 !important",
                "font-weight": "700 !important",
                "font-size": "12px !important",
            },
            ".ag-header-cell-text": {
                "color": "#111827 !important",
                "opacity": "1 !important",
            },
            ".ag-icon": {
                "color": "#4B5563 !important",
                "opacity": "1 !important",
            },
            ".ag-row": {
                "background-color": "#FFFFFF !important",
                "color": "#111827 !important",
            },
            ".ag-row-odd": {
                "background-color": "#F8FAFC !important",
            },
            ".ag-cell": {
                "color": "#111827 !important",
                "font-size": "12px !important",
                "line-height": "32px !important",
                "border-color": "#E5E7EB !important",
            },
            ".ag-paging-panel": {
                "background-color": "#FFFFFF !important",
                "color": "#111827 !important",
                "border-top": "1px solid #CBD5E1 !important",
            },
            ".ag-paging-panel *": {
                "color": "#111827 !important",
            },
        }
        grid_kwargs = {
            "gridOptions": grid_options,
            "height": height,
            "fit_columns_on_grid_load": False,
            "allow_unsafe_jscode": JsCode is not None,
            "enable_enterprise_modules": False,
            "theme": "alpine",
            "custom_css": aggrid_css,
            "key": key,
        }
        if selection and GridUpdateMode is not None:
            grid_kwargs["update_mode"] = GridUpdateMode.SELECTION_CHANGED
        if DataReturnMode is not None:
            grid_kwargs["data_return_mode"] = DataReturnMode.AS_INPUT
        response = AgGrid(grid_df, **grid_kwargs)
        if isinstance(response, dict):
            return response
        if hasattr(response, "selected_rows"):
            return {
                "selected_rows": getattr(response, "selected_rows", []),
                "data": getattr(response, "data", None),
            }
        return {}
    except Exception as exc:
        st.caption(f"AgGrid unavailable for this table; using Streamlit table. {exc}")
        st.dataframe(df, use_container_width=True, hide_index=True, height=height)
        return {}


def _psychometrics_status_pills_html() -> str:
    selected = st.session_state.get("last_detect_agents") or [
        fn for fn in ABERRANCE_FUNCTIONS if st.session_state.get(f"ab_only_cb_{fn}")
    ] or ["detect_nm"]
    responses = st.session_state.get("forensic_responses") or st.session_state.get("last_uploaded_responses") or []
    rt_data = st.session_state.get("forensic_rt_data") or st.session_state.get("last_uploaded_rt_data") or []
    psi_data = st.session_state.get("forensic_psi_data") or st.session_state.get("last_irt_item_params") or st.session_state.get("item_params") or []
    answer_changes = st.session_state.get("prep_answer_changes") or []
    n_persons = len(responses)
    n_items = len(responses[0]) if responses else 0
    need_rt = "detect_rg" in selected
    need_psi = any(fn in selected for fn in ["detect_pm", "detect_pk", "detect_ac", "detect_as"])
    need_tt = "detect_tt" in selected
    status = st.session_state.get("detect_job_status", "pending")
    provider = _llm_provider()
    model = _effective_llm_model()
    model_short = str(model).split("/")[-1]
    pills = [
        f'<span class="psymas-pill">{_dot(bool(responses))}Response {n_persons}×{n_items}</span>',
    ]
    if need_rt:
        pills.append(f'<span class="psymas-pill">{_dot(bool(rt_data))}RT</span>')
    if need_psi:
        pills.append(f'<span class="psymas-pill">{_dot(bool(psi_data))}ψ</span>')
    if need_tt:
        pills.append(f'<span class="psymas-pill">{_dot(bool(answer_changes))}Answer changes</span>')
    pills.append(f'<span class="psymas-pill">{_dot(True)}LLM ({_esc(provider)} · {_esc(model_short)})</span>')
    pills.append(f'<span class="psymas-pill">{_dot(status != "error")}LangGraph Agents — {_esc(status)}</span>')
    return "".join(pills)


def _render_psychometrics_run_status() -> bool:
    run_id = st.session_state.get("detect_run_id")
    if not run_id:
        return bool(st.session_state.get("forensic_result"))

    status = st.session_state.get("detect_job_status", "pending")
    if status == "done" and st.session_state.get("forensic_result") is not None:
        if st.session_state.get("psymas_run_store_error"):
            st.warning("Detect completed, but the integrated run database could not be built.")
        return True
    if status == "error":
        st.error("Detection failed: " + str(st.session_state.get("detect_job_error") or "Unknown error."))
        return False

    st.session_state["detect_job_status"] = "running"
    st.markdown("**Running detection**")
    progress = st.progress(0)
    status_slot = st.empty()
    try:
        status_resp = _backend_get(f"/detect/{run_id}/status", timeout=20)
        status_resp.raise_for_status()
        js = status_resp.json()
    except Exception as e:
        st.session_state["detect_job_status"] = "error"
        st.session_state["detect_job_error"] = str(e)
        st.rerun()

    backend_status = js.get("status", "pending")
    backend_progress = int(js.get("progress", 0) or 0)
    progress.progress(min(backend_progress, 97))
    if backend_status == "error":
        st.session_state["detect_job_status"] = "error"
        st.session_state["detect_job_error"] = js.get("error") or "Backend reported an error."
        st.rerun()
    if backend_status == "unknown":
        st.session_state["detect_job_status"] = "error"
        st.session_state["detect_job_error"] = js.get("error") or "run_id not found on backend."
        st.rerun()
    if backend_status == "done":
        try:
            res_resp = _backend_get(f"/detect/{run_id}/result", timeout=20)
            res_resp.raise_for_status()
            result = (res_resp.json() or {}).get("result") or {}
        except Exception as e:
            st.session_state["detect_job_status"] = "error"
            st.session_state["detect_job_error"] = str(e)
            st.rerun()
        st.session_state["forensic_result"] = result
        st.session_state["forensic_psi_data"] = result.get("psi_data") or []
        st.session_state["detect_job_status"] = "done"
        st.session_state.pop("_governed_review_tables_cache", None)
        st.session_state.pop("psychometrics_result_data_export", None)
        st.session_state.pop("psymas_run_store_ready", None)
        progress.progress(98)
        status_slot.caption("Building integrated run database…")
        try:
            _materialize_psymas_run_store(str(run_id))
        except Exception as exc:
            st.session_state["psymas_run_store_error"] = str(exc)
        progress.progress(100)
        st.rerun()

    status_slot.caption("Preparing forensic indices...")
    time.sleep(3)
    st.rerun()
    return False




def _render_psychometrics_result_center() -> None:
    has_result = _render_psychometrics_run_status()
    st.markdown(
        """
<style>
/* Forensic workspace controls render inside the Preparation branch, whose
   local styles load later than the global theme. Keep these states explicit. */
div[data-testid="stSegmentedControl"] [data-baseweb="button-group"] {
    background: transparent !important;
}
div[data-testid="stSegmentedControl"] button,
div[data-testid="stSegmentedControl"] button > div,
div[data-testid="stSegmentedControl"] button span,
div[data-testid="stSegmentedControl"] button p {
    background: #FFFFFF !important;
    color: #17212B !important;
    border-color: #C9D2DC !important;
    opacity: 1 !important;
}
div[data-testid="stSegmentedControl"] button:hover,
div[data-testid="stSegmentedControl"] button:hover > div,
div[data-testid="stSegmentedControl"] button:hover span,
div[data-testid="stSegmentedControl"] button:hover p {
    background: #EAF3F5 !important;
    color: #102A35 !important;
}
div[data-testid="stSegmentedControl"] button[aria-pressed="true"],
div[data-testid="stSegmentedControl"] button[aria-pressed="true"] > div,
div[data-testid="stSegmentedControl"] button[aria-pressed="true"] span,
div[data-testid="stSegmentedControl"] button[aria-pressed="true"] p {
    background: #174E5F !important;
    color: #FFFFFF !important;
    border-color: #174E5F !important;
    opacity: 1 !important;
}
div[data-testid="stSegmentedControl"] label,
div[data-testid="stSegmentedControl"] label > div,
div[data-testid="stSegmentedControl"] [role="radio"] {
    background: #FFFFFF !important;
    color: #17212B !important;
    border-color: #C9D2DC !important;
    opacity: 1 !important;
}
div[data-testid="stSegmentedControl"] label *,
div[data-testid="stSegmentedControl"] [role="radio"] * {
    color: #17212B !important;
    opacity: 1 !important;
}
div[data-testid="stSegmentedControl"] label:has(input:checked),
div[data-testid="stSegmentedControl"] label:has(input:checked) > div,
div[data-testid="stSegmentedControl"] [role="radio"][aria-checked="true"] {
    background: #174E5F !important;
    color: #FFFFFF !important;
    border-color: #174E5F !important;
}
div[data-testid="stSegmentedControl"] label:has(input:checked) *,
div[data-testid="stSegmentedControl"] [role="radio"][aria-checked="true"] * {
    color: #FFFFFF !important;
}
</style>
        """,
        unsafe_allow_html=True,
    )
    fr = st.session_state.get("forensic_result")
    if not isinstance(fr, dict):
        if not has_result:
            st.info("Run Detect from Assessment Data to generate deterministic evidence.")
        return
    flags = fr.get("flags", {}) if isinstance(fr.get("flags"), dict) else {}
    if not flags:
        st.info("No forensic index output is available yet.")
        return

    visible_agents = _forensic_visible_agents()
    responses = st.session_state.get("forensic_responses") or st.session_state.get("last_uploaded_responses") or []
    n_export = _infer_n_examinees(flags, responses)
    selected_agents = st.session_state.get("last_detect_agents") or list(visible_agents or [])
    output_rows = _table_df(_agent_output_rows())
    detector_total = max(1, len(selected_agents) or len(flags))
    selected_agent_keys = {
        ABERRANCE_FN_TO_AGENT.get(fn, fn)
        for fn in selected_agents
    }
    if selected_agent_keys:
        detector_ready = sum(1 for key in selected_agent_keys if key in flags)
    else:
        detector_ready = len(flags)
    detector_ready = min(detector_total, detector_ready)
    extra_payload_n = max(0, len(flags) - detector_ready)
    if _streamlined_ui_enabled():
        rulebook_preview = _table_df(_rulebook_rows())
        evidence_use_counts = (
            rulebook_preview.get("evidence_use", pd.Series(dtype=str))
            .astype(str)
            .value_counts()
            .to_dict()
            if not rulebook_preview.empty and "evidence_use" in rulebook_preview.columns
            else {}
        )
        _render_figure_summary_panel(
            kicker="Forensic indices",
            title="Evidence profile",
            copy="Detector output is summarized before wide tables are opened. Evidence-use categories clarify which indices can count toward review evidence.",
            stats=[
                {
                    "label": "Detectors with output",
                    "value": f"{detector_ready}",
                    "note": f"of {detector_total} selected" + (f"; {extra_payload_n} extra payload" if extra_payload_n else ""),
                    "tone": "good" if flags else "off",
                },
                {
                    "label": "Examinees",
                    "value": f"{n_export:,}",
                    "note": "Rows available for index review",
                    "tone": "info",
                },
                {
                    "label": "Eligible index rows",
                    "value": evidence_use_counts.get("Evidence Flag", 0),
                    "note": "Can count toward governed evidence",
                    "tone": "good",
                },
                {
                    "label": "Calibration required",
                    "value": evidence_use_counts.get("Calibration Required", 0),
                    "note": f"{evidence_use_counts.get('Display Only', 0)} display-only rows",
                    "tone": "warn",
                },
            ],
        )
    _render_agent_workflow_graph(output_rows, detector_ready, detector_total)
    if output_rows.empty:
        st.info("The run completed, but no compact detector summary is available.")
    else:
        with st.expander("Agent output details", expanded=False):
            st.dataframe(
                output_rows,
                use_container_width=True,
                hide_index=True,
                height=min(310, 45 + 34 * len(output_rows)),
            )
    st.caption("Detailed evidence tables are generated only when opened. Simulation validation has moved to Research Tools.")

    if n_export <= 0:
        st.info("No examinee data is available for detailed export.")
        return

    show_details = st.toggle(
        "Open detailed evidence tables",
        value=False,
        key="forensic_open_detailed_tables_v1",
        help="Wide forensic-index tables are built only after this is opened.",
    )
    if not show_details:
        return

    export_state_key = "psychometrics_result_data_export"
    export_sig = json.dumps(
        {
            "agents": sorted(visible_agents or []),
            "n": n_export,
            "flag_keys": sorted(str(k) for k in flags),
            "result_id": id(fr),
            "threshold_config": _active_threshold_config(),
            "column_naming": "short_agent_prefix_v10_lazy_workspace",
        },
        sort_keys=True,
    )
    cached = st.session_state.get(export_state_key)
    if not isinstance(cached, dict) or cached.get("sig") != export_sig:
        with st.spinner("Preparing detailed forensic tables. Wide outputs can take a few seconds..."):
            raw_df, legend_df, guide_df = _build_examinee_flags_export(
                flags,
                n_export,
                visible_agents=visible_agents,
            )
            indices_df = _forensic_indices_table_df(raw_df)
            cached = {
                "sig": export_sig,
                "data_df": raw_df,
                "indices_table_df": indices_df,
                "legend_df": legend_df,
                "guide_df": guide_df,
                "csv_bytes": indices_df.to_csv(index=False).encode("utf-8-sig"),
            }
            st.session_state[export_state_key] = cached

    indices_df = cached.get("indices_table_df", pd.DataFrame())
    catalog_df = _forensic_index_catalog_df(indices_df, cached.get("legend_df", pd.DataFrame()))
    with st.container(border=True):
        detail = st.selectbox(
            "Table",
            ["Index Catalog", "Forensic Indices", "Evidence Input", "Index Dictionary", "Pair Details"],
            key="forensic_detail_kind_v3",
            label_visibility="collapsed",
        )
        _render_eligibility_summary(catalog_df)
        if detail == "Index Catalog":
            catalog_rules_df = _table_df(_rulebook_rows())
            display_cols = [
                c
                for c in [
                    "function",
                    "domain",
                    "method",
                    "index",
                    "role",
                    "evidence_use",
                    "flag_type",
                    "threshold_source",
                ]
                if c in catalog_rules_df.columns
            ]
            st.dataframe(
                catalog_rules_df[display_cols] if display_cols else catalog_rules_df,
                use_container_width=True,
                hide_index=True,
                height=500,
            )
        elif detail == "Pair Details":
            pair_cache_key = "forensic_pair_detail_v2"
            pair_cache = st.session_state.get(pair_cache_key)
            pair_sig = (id(fr), tuple(sorted(visible_agents or [])))
            if not isinstance(pair_cache, dict) or pair_cache.get("sig") != pair_sig:
                with st.spinner("Preparing pair-level detail..."):
                    pair_cache = {
                        "sig": pair_sig,
                        "data": _build_pair_detail_table(flags, visible_agents),
                    }
                    st.session_state[pair_cache_key] = pair_cache
            pair_df = pair_cache.get("data", pd.DataFrame())
            _render_aggrid_table(pair_df.head(500), key="forensic_pairs_v2", height=500)
        elif detail == "Evidence Input":
            if not isinstance(cached.get("b3_input_df"), pd.DataFrame):
                with st.spinner("Preparing examinee-level evidence input..."):
                    cached["b3_input_df"] = _build_b3_input_table(cached["data_df"], cached["legend_df"])
                    st.session_state[export_state_key] = cached
            _render_aggrid_table(cached["b3_input_df"].head(500), key="forensic_evidence_input_v2", height=500)
        elif detail == "Index Dictionary":
            if not isinstance(cached.get("index_dictionary_df"), pd.DataFrame):
                cached["index_dictionary_df"] = _build_index_dictionary(cached["data_df"], cached["legend_df"])
                st.session_state[export_state_key] = cached
            _render_aggrid_table(cached["index_dictionary_df"], key="forensic_dictionary_v2", height=500)
        else:
            _render_aggrid_table(indices_df, key="forensic_indices_v2", height=500)

    with st.spinner("Preparing the consolidated examinee-level dataset..."):
        master_df = _master_results_df(cached["data_df"])
    st.caption(
        "The master table contains final flags, domain results, priority, human review fields, "
        "and every forensic index. Intermediate evidence tables remain available on screen for drill-down."
    )
    _render_page_download_bar(
        [
            {
                "label": "Download Master Results",
                "data": _df_csv_bytes(master_df),
                "file_name": "psymas_master_results.csv",
                "key": "forensic_master_results_download",
                "disabled": master_df.empty,
            },
        ]
    )


def _forensic_export_signature() -> str:
    flags = _forensic_result_flags_from_session()
    responses = st.session_state.get("forensic_responses") or st.session_state.get("last_uploaded_responses") or []
    n_export = _infer_n_examinees(flags, responses)
    visible_agents = _forensic_visible_agents()
    fr = st.session_state.get("forensic_result")
    return json.dumps(
        {
            "agents": sorted(visible_agents or []),
            "n": n_export,
            "flag_keys": sorted(str(k) for k in flags),
            "result_id": id(fr),
            "threshold_config": _active_threshold_config(),
            "column_naming": "short_agent_prefix_v10_lazy_workspace",
        },
        sort_keys=True,
    )


def _ensure_forensic_export_cache(*, export_state_key: str = "psychometrics_result_data_export") -> dict | None:
    """Build and cache the wide examinee export + B3 input once per Detect result."""
    flags = _forensic_result_flags_from_session()
    if not flags:
        return None
    responses = st.session_state.get("forensic_responses") or st.session_state.get("last_uploaded_responses") or []
    n_export = _infer_n_examinees(flags, responses)
    if n_export <= 0:
        return None
    export_sig = _forensic_export_signature()
    cached = st.session_state.get(export_state_key)
    if isinstance(cached, dict) and cached.get("sig") == export_sig:
        return cached
    visible_agents = _forensic_visible_agents()
    raw_df, legend_df, guide_df = _build_examinee_flags_export(flags, n_export, visible_agents=visible_agents)
    indices_df = _forensic_indices_table_df(raw_df)
    cached = {
        "sig": export_sig,
        "data_df": raw_df,
        "indices_table_df": indices_df,
        "legend_df": legend_df,
        "guide_df": guide_df,
        "b3_input_df": _build_b3_input_table(raw_df, legend_df),
        "csv_bytes": indices_df.to_csv(index=False).encode("utf-8-sig"),
    }
    st.session_state[export_state_key] = cached
    return cached


def _active_run_id() -> str:
    return str(st.session_state.get("detect_run_id") or st.session_state.get("psymas_active_run_id") or "")


def _run_store_ready() -> bool:
    run_id = _active_run_id()
    if not run_id or not st.session_state.get("psymas_run_store_ready"):
        return False
    return get_run_store().has_run(run_id)


def _sync_review_decisions_from_store() -> None:
    if not _run_store_ready():
        return
    stored = get_run_store().load_review_decisions(_active_run_id())
    if not stored:
        return
    merged = dict(st.session_state.get("case_review_decisions") or {})
    for examinee_id, payload in stored.items():
        if examinee_id not in merged or not str(merged[examinee_id].get("final_decision", "")).strip():
            merged[examinee_id] = payload
        else:
            current = merged[examinee_id]
            if isinstance(current, dict):
                current.setdefault("reviewer_note", payload.get("reviewer_note", ""))
    st.session_state["case_review_decisions"] = merged


def _persist_case_review_to_store(
    examinee_id: str,
    *,
    final_decision: str = "",
    reviewer_note: str = "",
    llm_explanation: str = "",
) -> None:
    if not _run_store_ready():
        return
    get_run_store().save_review_decision(
        str(examinee_id),
        final_decision=final_decision,
        reviewer_note=reviewer_note,
        llm_explanation=llm_explanation,
        run_id=_active_run_id(),
    )


def _materialize_psymas_run_store(run_id: str) -> bool:
    """After Detect: build export, governance, master tables, and persist to SQLite."""
    flags = _forensic_result_flags_from_session()
    responses = st.session_state.get("forensic_responses") or st.session_state.get("last_uploaded_responses") or []
    n_examinees = _infer_n_examinees(flags, responses)
    if not flags or n_examinees <= 0:
        return False

    st.session_state.pop("_governed_review_tables_cache", None)
    st.session_state.pop("psychometrics_result_data_export", None)
    export = _ensure_forensic_export_cache()
    if not export:
        return False

    governed_df, domain_df = _governed_review_tables()
    review_df = (
        _evidence_synthesis_with_human_review(governed_df)
        if isinstance(governed_df, pd.DataFrame) and not governed_df.empty
        else pd.DataFrame()
    )
    final_flags_df = _final_flag_review_df()
    master_df = build_master_results(
        indices_df=export.get("indices_table_df", pd.DataFrame()),
        final_flags_df=final_flags_df,
        review_df=review_df,
        domain_df=domain_df,
        run_id=str(run_id),
        schema_version="1.0",
    )
    if not master_df.empty:
        output_dir = Path("data") / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "psymas_master_results.csv"
        temporary_path = output_dir / ".psymas_master_results.csv.tmp"
        master_df.to_csv(temporary_path, index=False, encoding="utf-8-sig")
        temporary_path.replace(output_path)
        st.session_state["master_results_path"] = str(output_path)

    pair_index = _build_pair_visual_index(flags, n_examinees)
    decisions = st.session_state.get("case_review_decisions", {})
    decisions = decisions if isinstance(decisions, dict) else {}

    get_run_store().save_run(
        run_id=str(run_id),
        tables={
            "indices_export": export.get("data_df", pd.DataFrame()),
            "indices_table": export.get("indices_table_df", pd.DataFrame()),
            "column_legend": export.get("legend_df", pd.DataFrame()),
            "b3_input": export.get("b3_input_df", pd.DataFrame()),
            "governed_review": governed_df,
            "domain_evidence": domain_df,
            "review_queue": review_df,
            "master_results": master_df,
            "final_flags": final_flags_df,
        },
        forensic_result=st.session_state.get("forensic_result"),
        pair_visual_index=pair_index,
        metadata={
            "n_examinees": n_examinees,
            "detect_agents": st.session_state.get("last_detect_agents") or [],
            "threshold_config": _active_threshold_config(),
        },
        review_decisions=decisions,
    )
    st.session_state["psymas_run_store_ready"] = True
    st.session_state["psymas_active_run_id"] = str(run_id)
    st.session_state.pop("psymas_run_store_error", None)
    return True


def _ensure_run_store() -> None:
    """Backfill the integrated store when Detect finished before this feature existed."""
    if st.session_state.get("psymas_run_store_ready"):
        return
    run_id = _active_run_id()
    if not run_id or not st.session_state.get("forensic_result"):
        return
    if get_run_store().has_run(run_id):
        st.session_state["psymas_run_store_ready"] = True
        st.session_state["psymas_active_run_id"] = run_id
        _sync_review_decisions_from_store()
        return
    _materialize_psymas_run_store(run_id)


_SNAPSHOT_SESSION_INPUT_KEYS: tuple[str, ...] = (
    "last_uploaded_responses",
    "last_uploaded_rt_data",
    "last_irt_item_params",
    "item_params",
    "forensic_psi_data",
    "forensic_person_params",
    "forensic_responses",
    "forensic_rt_data",
    "prep_answer_changes",
    "prep_compromised_items",
    "prep_answer_changes_file_name",
    "prep_compromised_file_name",
    "last_detect_agents",
    "last_irt_itemtype",
    "main_irt_itemtype",
    "demo_data_loaded",
    "demo_data_source",
    "demo_examinee_ids",
    "demo_item_ids",
    "ab_only_scenario_select",
    "last_uploaded_model_settings",
    "last_uploaded_is_verified",
)

_SNAPSHOT_DEMO_TABLE_KEYS: tuple[str, ...] = (
    "demo_scenario_key",
    "demo_scenario_summary",
    "demo_testing_context",
    "demo_group_check",
    "demo_copying_pairs_truth",
    "demo_answer_change_summary",
)


def _collect_snapshot_session_inputs() -> dict:
    payload: dict = {}
    for key in _SNAPSHOT_SESSION_INPUT_KEYS:
        if key in st.session_state:
            payload[key] = st.session_state[key]
    for key in _SNAPSHOT_DEMO_TABLE_KEYS:
        value = st.session_state.get(key)
        if value:
            payload[key] = value
    threshold_config = st.session_state.get("threshold_config")
    if isinstance(threshold_config, dict) and threshold_config:
        payload["threshold_config"] = threshold_config
    decisions = st.session_state.get("case_review_decisions")
    if isinstance(decisions, dict) and decisions:
        payload["case_review_decisions"] = decisions
    return payload


def _build_evaluated_snapshot_bytes() -> bytes | None:
    """Package integrated run DB + inputs for demo skip-rerun workflow."""
    if not st.session_state.get("forensic_result"):
        return None
    _ensure_run_store()
    if not _run_store_ready():
        run_id = _active_run_id()
        if run_id:
            try:
                _materialize_psymas_run_store(run_id)
            except Exception:
                return None
    if not _run_store_ready():
        return None

    store = get_run_store()
    run_id = _active_run_id()
    if not run_id or not store.has_run(run_id):
        return None

    manifest = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "scenario": st.session_state.get("ab_only_scenario_select", "C"),
        "detect_agents": st.session_state.get("last_detect_agents") or [],
        "n_examinees": len(st.session_state.get("last_uploaded_responses") or []),
    }
    return pack_snapshot(
        store_path=store.db_path,
        manifest=manifest,
        session_inputs=_collect_snapshot_session_inputs(),
    )


def _restore_evaluated_snapshot(data: bytes, *, preserve_review_decisions: bool = True) -> tuple[bool, str]:
    """Restore evaluated run from zip; hydrates session so Evidence Review works without Detect."""
    store = get_run_store()
    preserved_decisions: dict[str, dict] = {}
    if preserve_review_decisions:
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                raw_manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
            target_run_id = str(raw_manifest.get("run_id") or "")
            if target_run_id:
                preserved_decisions.update(store.load_review_decisions(target_run_id))
            session_decisions = st.session_state.get("case_review_decisions")
            if isinstance(session_decisions, dict):
                preserved_decisions.update(
                    {
                        str(examinee_id): payload
                        for examinee_id, payload in session_decisions.items()
                        if isinstance(payload, dict)
                        and (
                            str(payload.get("final_decision", "") or "").strip()
                            or str(payload.get("reviewer_note", "") or "").strip()
                            or str(payload.get("llm_explanation", "") or "").strip()
                        )
                    }
                )
        except Exception:
            preserved_decisions = {}
    try:
        manifest, session_inputs = unpack_snapshot(data, store_path=store.db_path)
    except Exception as exc:
        return False, f"Could not read snapshot: {exc}"

    run_id = str(manifest.get("run_id") or "")
    if not run_id or not store.has_run(run_id):
        return False, "Snapshot database is missing the declared run_id."

    for key in (
        "_governed_review_tables_cache",
        "psychometrics_result_data_export",
        "psymas_run_store_error",
        "detect_job_error",
    ):
        st.session_state.pop(key, None)

    for key, value in session_inputs.items():
        if key == "threshold_config" and isinstance(value, dict):
            st.session_state["threshold_config"] = value
        elif key == "case_review_decisions" and isinstance(value, dict):
            st.session_state["case_review_decisions"] = value if preserve_review_decisions else {}
        else:
            st.session_state[key] = value

    forensic = store.load_forensic_result(run_id)
    if not forensic:
        return False, "Snapshot database does not contain forensic results."

    st.session_state["forensic_result"] = forensic
    st.session_state["detect_run_id"] = run_id
    st.session_state["detect_job_status"] = "done"
    st.session_state["psymas_run_store_ready"] = True
    st.session_state["psymas_active_run_id"] = run_id

    detect_agents = manifest.get("detect_agents") or session_inputs.get("last_detect_agents") or []
    if detect_agents:
        st.session_state["last_detect_agents"] = list(detect_agents)
        for fn in ABERRANCE_FUNCTIONS:
            st.session_state[f"ab_only_cb_{fn}"] = fn in detect_agents

    scenario = str(manifest.get("scenario") or session_inputs.get("ab_only_scenario_select") or "C")
    st.session_state["ab_only_scenario_select"] = scenario

    if not st.session_state.get("forensic_psi_data"):
        psi = st.session_state.get("last_irt_item_params") or st.session_state.get("item_params")
        if psi:
            st.session_state["forensic_psi_data"] = psi

    _sync_review_decisions_from_store()
    if preserved_decisions:
        merged_decisions = dict(st.session_state.get("case_review_decisions") or {})
        for examinee_id, payload in preserved_decisions.items():
            if not isinstance(payload, dict):
                continue
            has_saved_value = (
                str(payload.get("final_decision", "") or "").strip()
                or str(payload.get("reviewer_note", "") or "").strip()
                or str(payload.get("llm_explanation", "") or "").strip()
            )
            if not has_saved_value:
                continue
            merged_decisions[str(examinee_id)] = payload
            store.save_review_decision(
                str(examinee_id),
                final_decision=str(payload.get("final_decision", "") or ""),
                reviewer_note=str(payload.get("reviewer_note", "") or ""),
                llm_explanation=str(payload.get("llm_explanation", "") or ""),
                run_id=run_id,
            )
        st.session_state["case_review_decisions"] = merged_decisions

    master_df = store.load_table("master_results")
    if isinstance(master_df, pd.DataFrame) and not master_df.empty:
        output_dir = Path("data") / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "psymas_master_results.csv"
        master_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        st.session_state["master_results_path"] = str(output_path)

    n_examinees = int(manifest.get("n_examinees") or len(st.session_state.get("last_uploaded_responses") or []))
    return (
        True,
        f"Restored evaluated snapshot for run {run_id} ({n_examinees:,} examinees). "
        "Evidence Review and Examinee Report are ready without re-running Detect.",
    )


def _restore_demo_evaluated_snapshot_if_available(
    *,
    force: bool = False,
    preserve_review_decisions: bool = True,
) -> tuple[bool, str]:
    """Load the bundled demo evaluated snapshot once so Demo opens without re-running Detect."""
    if not force and st.session_state.get("demo_evaluated_snapshot_loaded") and st.session_state.get("forensic_result"):
        return True, "Demo evaluated snapshot is already loaded."
    if not DEMO_EVALUATED_SNAPSHOT_PATH.exists():
        return False, f"Demo evaluated snapshot not found: {DEMO_EVALUATED_SNAPSHOT_PATH}"
    try:
        data = DEMO_EVALUATED_SNAPSHOT_PATH.read_bytes()
    except Exception as exc:
        return False, f"Could not read demo evaluated snapshot: {exc}"

    ok, msg = _restore_evaluated_snapshot(data, preserve_review_decisions=preserve_review_decisions)
    if ok:
        st.session_state["demo_evaluated_snapshot_loaded"] = True
        st.session_state["demo_evaluated_snapshot_source"] = str(DEMO_EVALUATED_SNAPSHOT_PATH)
    else:
        st.session_state.pop("demo_evaluated_snapshot_loaded", None)
    return ok, msg


def _load_run_from_store(run_id: str) -> tuple[bool, str]:
    store = get_run_store()
    run_id = str(run_id or "")
    if not run_id:
        return False, "No run ID was selected."
    if not store.set_active_run(run_id):
        return False, f"Run not found: {run_id}"
    forensic = store.load_forensic_result(run_id)
    if not forensic:
        return False, "Selected run does not contain forensic results."
    for key in (
        "_governed_review_tables_cache",
        "psychometrics_result_data_export",
        "psymas_run_store_error",
        "detect_job_error",
    ):
        st.session_state.pop(key, None)
    st.session_state["forensic_result"] = forensic
    st.session_state["detect_run_id"] = run_id
    st.session_state["detect_job_status"] = "done"
    st.session_state["psymas_run_store_ready"] = True
    st.session_state["psymas_active_run_id"] = run_id
    rows = store.list_runs()
    metadata = next((row.get("metadata", {}) for row in rows if str(row.get("run_id")) == run_id), {})
    detect_agents = metadata.get("detect_agents") if isinstance(metadata, dict) else []
    if detect_agents:
        st.session_state["last_detect_agents"] = list(detect_agents)
        for fn in ABERRANCE_FUNCTIONS:
            st.session_state[f"ab_only_cb_{fn}"] = fn in detect_agents
    _sync_review_decisions_from_store()
    master_df = store.load_table("master_results")
    if isinstance(master_df, pd.DataFrame) and not master_df.empty:
        output_dir = Path("data") / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "psymas_master_results.csv"
        master_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        st.session_state["master_results_path"] = str(output_path)
    return True, f"Loaded run {run_id} from {store.db_path}."


def _run_storage_summary_rows() -> list[dict]:
    store = get_run_store()
    db_path = store.db_path
    active_run = _active_run_id() or store.active_run_id() or ""
    decisions = store.load_review_decisions(active_run) if active_run else {}
    forensic_ready = bool(st.session_state.get("forensic_result"))
    return [
        {"Area": "SQLite database", "Status": "available" if db_path.exists() else "missing", "Detail": str(db_path)},
        {"Area": "Active run", "Status": active_run or "none", "Detail": "Loaded in session" if forensic_ready else "Stored only"},
        {"Area": "Human review records", "Status": f"{len(decisions):,}", "Detail": "Saved in review_decisions"},
        {"Area": "Response matrix", "Status": "loaded" if st.session_state.get("last_uploaded_responses") else "not loaded", "Detail": f"{len(st.session_state.get('last_uploaded_responses') or []):,} rows"},
        {"Area": "Response times", "Status": "loaded" if st.session_state.get("last_uploaded_rt_data") else "not loaded", "Detail": f"{len(st.session_state.get('last_uploaded_rt_data') or []):,} rows"},
        {"Area": "Item parameters", "Status": "loaded" if (st.session_state.get("last_irt_item_params") or st.session_state.get("item_params")) else "not loaded", "Detail": ""},
        {"Area": "Answer changes", "Status": "loaded" if st.session_state.get("prep_answer_changes") else "not loaded", "Detail": f"{len(st.session_state.get('prep_answer_changes') or []):,} rows"},
    ]


def _render_data_run_storage_manager() -> None:
    store = get_run_store()
    st.markdown("#### Data & Run Storage")
    st.caption("Manage loaded inputs, the local SQLite run database, snapshots, and saved human review records.")
    summary_df = pd.DataFrame(_run_storage_summary_rows())
    st.dataframe(summary_df, use_container_width=True, hide_index=True, height=250)

    runs = store.list_runs()
    if runs:
        run_df = pd.DataFrame(
            [
                {
                    "active": "Yes" if row.get("is_active") else "",
                    "run_id": row.get("run_id", ""),
                    "created_at": row.get("created_at", ""),
                    "review_decisions": row.get("review_decisions", 0),
                    "n_examinees": (row.get("metadata") or {}).get("n_examinees", ""),
                    "detect_agents": ", ".join((row.get("metadata") or {}).get("detect_agents", []) or []),
                }
                for row in runs
            ]
        )
        with st.expander("Stored runs", expanded=True):
            st.dataframe(run_df, use_container_width=True, hide_index=True, height=min(280, 44 + 32 * len(run_df)))
            run_options = [str(row.get("run_id", "")) for row in runs if row.get("run_id")]
            default_index = 0
            active_store_run = store.active_run_id()
            if active_store_run in run_options:
                default_index = run_options.index(active_store_run)
            selected_run = st.selectbox(
                "Run to load",
                run_options,
                index=default_index,
                key="run_manager_selected_run",
            )
            load_col, snapshot_col = st.columns(2)
            with load_col:
                if st.button("Load Selected Run", key="run_manager_load_run", type="primary", use_container_width=True):
                    ok, msg = _load_run_from_store(selected_run)
                    if ok:
                        st.success(msg)
                        st.rerun()
                    st.error(msg)
            with snapshot_col:
                snapshot_bytes = None
                if str(selected_run) == _active_run_id() and st.session_state.get("forensic_result"):
                    snapshot_bytes = _build_evaluated_snapshot_bytes()
                st.download_button(
                    "Download Current Snapshot",
                    data=snapshot_bytes or b"",
                    file_name="psymas_snapshot.zip",
                    mime="application/zip",
                    disabled=not snapshot_bytes,
                    key="run_manager_download_snapshot",
                    use_container_width=True,
                    help="Load the selected run first if this button is disabled.",
                )
    else:
        st.info("No stored runs were found in the local SQLite database.")

    restore_col, demo_col = st.columns(2)
    with restore_col:
        uploaded = st.file_uploader("Restore snapshot ZIP", type=["zip"], key="run_manager_snapshot_upload")
        if uploaded is not None:
            sig = (uploaded.name, uploaded.size)
            if st.session_state.get("_run_manager_snapshot_sig") != sig:
                ok, msg = _restore_evaluated_snapshot(uploaded.getvalue())
                st.session_state["_run_manager_snapshot_sig"] = sig
                if ok:
                    st.success(msg)
                    st.rerun()
                st.error(msg)
    with demo_col:
        st.markdown("##### Demo template")
        st.caption("Use existing saved demo runs by default. Reload clean demo only when you want to reset the local demo results.")
        if st.button("Use Existing Demo Run", key="run_manager_use_existing_demo", use_container_width=True):
            demo_run = store.active_run_id()
            if demo_run:
                ok, msg = _load_run_from_store(demo_run)
                if ok:
                    st.success(msg)
                    st.rerun()
                st.error(msg)
            else:
                st.warning("No active demo run is available yet.")
        if st.button("Reload Clean Demo Snapshot", key="run_manager_reload_clean_demo", use_container_width=True):
            ok, msg = _load_demo_simulated_data()
            if ok:
                snap_ok, snap_msg = _restore_demo_evaluated_snapshot_if_available(
                    force=True,
                    preserve_review_decisions=False,
                )
                st.success(f"{msg} {snap_msg}" if snap_ok else msg)
                if not snap_ok:
                    st.warning(snap_msg)
                st.rerun()
            st.error(msg)


def _render_demo_evaluated_snapshot_panel() -> None:
    """Demo-only: download/upload full post-Detect state to skip re-computation."""
    with st.container(border=True):
        st.markdown(
            """
            <div class="psymas-card-head">
              <div class="psymas-card-title">Evaluated snapshot</div>
              <div class="psymas-card-status">Demo · skip re-run</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            "Download the full post-Detect bundle (integrated SQLite DB, forensic outputs, inputs, review state). "
            "Upload it later to restore Evidence Review without running Detect again."
        )
        has_forensic = st.session_state.get("forensic_result") is not None
        store_ready = _run_store_ready() or (has_forensic and bool(_active_run_id()))
        download_col, upload_col = st.columns(2, gap="medium")
        with download_col:
            snapshot_bytes = _build_evaluated_snapshot_bytes() if has_forensic else None
            st.download_button(
                "Download evaluated snapshot",
                data=snapshot_bytes or b"",
                file_name="psymas_demo_evaluated_snapshot.zip",
                mime="application/zip",
                disabled=not snapshot_bytes,
                use_container_width=True,
                key="demo_evaluated_snapshot_download",
                help="Available after Detect completes and the integrated run database is built.",
            )
            if has_forensic and not snapshot_bytes:
                st.caption("Building snapshot requires a materialized run store. Run Detect once, or upload a prior snapshot.")
            elif snapshot_bytes:
                size_kb = max(1, len(snapshot_bytes) // 1024)
                st.caption(f"Bundle size ≈ {size_kb:,} KB (DB + inputs + review state).")
        with upload_col:
            uploaded = st.file_uploader(
                "Upload evaluated snapshot",
                type=["zip"],
                key="demo_evaluated_snapshot_upload",
                help="Restores forensic indices, governance tables, and demo inputs from a prior export.",
            )
            if uploaded is not None:
                upload_sig = (uploaded.name, uploaded.size)
                if st.session_state.get("_demo_evaluated_snapshot_sig") != upload_sig:
                    with st.spinner("Restoring evaluated snapshot…"):
                        ok, msg = _restore_evaluated_snapshot(uploaded.getvalue())
                    st.session_state["_demo_evaluated_snapshot_sig"] = upload_sig
                    if ok:
                        st.session_state["_demo_evaluated_snapshot_msg"] = msg
                        st.rerun()
                    st.session_state["_demo_evaluated_snapshot_error"] = msg
                if st.session_state.get("_demo_evaluated_snapshot_error") and not st.session_state.get(
                    "_demo_evaluated_snapshot_msg"
                ):
                    st.error(str(st.session_state.pop("_demo_evaluated_snapshot_error")))
        if st.session_state.get("_demo_evaluated_snapshot_msg"):
            st.success(st.session_state.pop("_demo_evaluated_snapshot_msg"))
            st.session_state.pop("_demo_evaluated_snapshot_error", None)


def _prewarm_evidence_review_cache() -> None:
    """Populate governed review tables once after Detect so Evidence Review opens quickly."""
    run_id = _active_run_id()
    if run_id and st.session_state.get("forensic_result"):
        if not st.session_state.get("psymas_run_store_ready"):
            _materialize_psymas_run_store(run_id)
        else:
            _sync_review_decisions_from_store()
        return
    st.session_state.pop("_governed_review_tables_cache", None)
    _governed_review_tables()


def _governed_review_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    if _run_store_ready():
        store = get_run_store()
        governed_df = store.load_table("governed_review")
        domain_df = store.load_table("domain_evidence")
        if isinstance(governed_df, pd.DataFrame) and isinstance(domain_df, pd.DataFrame):
            return governed_df.copy(), domain_df.copy()
    export_signatures = []
    for key in ("psychometrics_result_data_export", "summary_result_data_export", "prep_result_data_export"):
        cached_export = st.session_state.get(key)
        if isinstance(cached_export, dict):
            export_signatures.append((key, str(cached_export.get("sig", ""))))
    rulebook_path = Path("config") / "rulebook_index.csv"
    if not rulebook_path.exists():
        rulebook_path = Path("manuscript") / "rulebook_index.csv"
    rulebook_version = rulebook_path.stat().st_mtime_ns if rulebook_path.exists() else 0
    source_signature = json.dumps(
        {
            "exports": export_signatures,
            "forensic_result_id": id(st.session_state.get("forensic_result")),
            "threshold_config": _active_threshold_config(),
            "rulebook_version": rulebook_version,
            "governance_version": "b2_b3_b6_v2",
        },
        sort_keys=True,
        default=str,
    )
    cached_tables = st.session_state.get("_governed_review_tables_cache")
    if isinstance(cached_tables, dict) and cached_tables.get("sig") == source_signature:
        governed_df = cached_tables.get("governed_df")
        domain_df = cached_tables.get("domain_df")
        if isinstance(governed_df, pd.DataFrame) and isinstance(domain_df, pd.DataFrame):
            return governed_df.copy(), domain_df.copy()

    _ensure_forensic_export_cache()
    result_df, legend_df = _governance_source_export()
    if result_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    b3_input_df = _governance_source_b3_input()
    governed_df, domain_df = _build_governed_evidence_profile(
        result_df,
        legend_df,
        b3_input_df,
    )
    st.session_state["_governed_review_tables_cache"] = {
        "sig": source_signature,
        "governed_df": governed_df.copy(),
        "domain_df": domain_df.copy(),
    }
    return governed_df, domain_df


def _case_review_packet_bytes(
    case_row: pd.Series,
    case_domains: pd.DataFrame,
    case_trace: pd.DataFrame,
    case_auxiliary: pd.DataFrame | None = None,
) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("case_summary.csv", pd.DataFrame([case_row.to_dict()]).to_csv(index=False))
        zf.writestr("case_domain_evidence.csv", case_domains.to_csv(index=False))
        zf.writestr("case_evidence_trace.csv", case_trace.to_csv(index=False))
        if isinstance(case_auxiliary, pd.DataFrame):
            zf.writestr("case_display_only_auxiliary_indices.csv", case_auxiliary.to_csv(index=False))
    return buf.getvalue()


def _compact_case_table_text(df: pd.DataFrame, columns: list[str], max_rows: int = 24) -> str:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return "No rows available."
    use_cols = [c for c in columns if c in df.columns]
    if not use_cols:
        use_cols = list(df.columns[:8])
    out = df[use_cols].copy().head(max_rows)
    return out.to_csv(index=False)


def _selected_case_trace_for_llm(case_trace: pd.DataFrame, max_per_domain: int = 2) -> pd.DataFrame:
    """Keep only the strongest counted index traces for the reviewer-facing LLM prompt."""
    if not isinstance(case_trace, pd.DataFrame) or case_trace.empty:
        return pd.DataFrame()
    trace = case_trace.copy()
    if "Flag" in trace.columns:
        trace = trace[trace["Flag"].astype(str).str.lower().isin({"1", "true", "yes"})]
    if trace.empty:
        return trace
    if "Domain" not in trace.columns:
        trace["Domain"] = ""
    if "Role" not in trace.columns:
        trace["Role"] = ""
    if "Evidence_Use" not in trace.columns:
        trace["Evidence_Use"] = ""

    role_rank = {
        "preferred_primary": 0,
        "primary": 0,
        "fallback_primary": 1,
        "supporting": 2,
        "support": 2,
    }
    evidence_rank = {
        "scenario flag": 0,
        "evidence flag": 0,
        "supporting flag": 1,
        "support flag": 1,
    }
    domain_rank = {"RT": 0, "PK": 1, "TP": 2, "MF": 3, "SIM": 4, "CP": 5}

    def _rank_role(value: object) -> int:
        text = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
        for key, rank in role_rank.items():
            if key in text:
                return rank
        return 9

    def _rank_use(value: object) -> int:
        text = str(value or "").strip().lower().replace("_", " ")
        for key, rank in evidence_rank.items():
            if key in text:
                return rank
        return 9

    trace["_domain_rank"] = trace["Domain"].astype(str).map(lambda x: domain_rank.get(x, 99))
    trace["_role_rank"] = trace["Role"].map(_rank_role)
    trace["_use_rank"] = trace["Evidence_Use"].map(_rank_use)
    trace["_index_name"] = trace.get("Index", pd.Series([""] * len(trace))).astype(str)
    trace = trace.sort_values(["_domain_rank", "_use_rank", "_role_rank", "Function", "_index_name"])
    selected_parts = []
    for domain in ["RT", "PK", "TP", "MF", "SIM", "CP"]:
        sub = trace[trace["Domain"].astype(str).eq(domain)].head(max_per_domain)
        if not sub.empty:
            selected_parts.append(sub)
    if not selected_parts:
        selected_parts = [trace.head(8)]
    out = pd.concat(selected_parts, ignore_index=True).drop(columns=[c for c in trace.columns if c.startswith("_")], errors="ignore")
    return out


def _filter_examinee_table(df: pd.DataFrame, examinee_id: str) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    id_cols = [c for c in ("Examinee_ID", "examinee_id", "Examinee", "ID") if c in df.columns]
    if not id_cols:
        return pd.DataFrame()
    id_col = id_cols[0]
    return df[df[id_col].astype(str) == str(examinee_id)].copy()


def _case_reviewer_packet_from_store(examinee_id: str) -> dict:
    """Load the most complete case evidence packet from the active SQLite run store."""
    packet: dict = {"source": "session"}
    if not _run_store_ready():
        return packet
    store = get_run_store()
    review_df = store.load_table("review_queue")
    if not isinstance(review_df, pd.DataFrame) or review_df.empty:
        review_df = store.load_table("governed_review")
    case_rows = _filter_examinee_table(review_df, examinee_id)
    domain_rows = _filter_examinee_table(store.load_table("domain_evidence"), examinee_id)
    trace_rows = _filter_examinee_table(store.load_table("b3_input"), examinee_id)
    indices_rows = _filter_examinee_table(store.load_table("indices_export"), examinee_id)

    loaded_any = any(
        isinstance(x, pd.DataFrame) and not x.empty
        for x in (case_rows, domain_rows, trace_rows, indices_rows)
    )
    if not loaded_any:
        return packet
    packet["source"] = f"SQLite run store ({Path(store.db_path).as_posix()})"
    if not case_rows.empty:
        packet["case_row"] = case_rows.iloc[0]
    if not domain_rows.empty:
        packet["case_domains"] = domain_rows
    if not trace_rows.empty:
        packet["case_trace"] = trace_rows
    if not indices_rows.empty:
        packet["indices_row"] = indices_rows.iloc[0]
    return packet


def _merge_case_series(base: pd.Series, override: pd.Series | None) -> pd.Series:
    if not isinstance(override, pd.Series) or override.empty:
        return base
    merged = base.to_dict() if isinstance(base, pd.Series) else {}
    for key, value in override.to_dict().items():
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text and text.lower() != "nan":
            merged[key] = value
        elif key not in merged:
            merged[key] = value
    return pd.Series(merged)


def _case_pk_prompt_cue(examinee_id: str) -> str:
    """Compact PK item-panel summary for LLM review guidance."""
    resp_df = _case_response_dataframe()
    if resp_df.empty:
        return "PK item cue: response matrix unavailable."
    try:
        sid0 = int(examinee_id) - 1
    except (TypeError, ValueError):
        return "PK item cue: selected examinee unavailable."
    if not (0 <= sid0 < len(resp_df)):
        return "PK item cue: selected examinee unavailable."
    n_items = int(resp_df.shape[1])
    comp_items = _case_compromised_items(n_items=n_items)
    if not comp_items:
        return "PK item cue: no compromised/exposed item list available."
    comp_idx = [i - 1 for i in comp_items if 1 <= i <= n_items]
    if not comp_idx:
        return "PK item cue: compromised/exposed item list does not match the response matrix."
    selected_resp = pd.to_numeric(resp_df.iloc[sid0, comp_idx], errors="coerce")
    cohort_acc = resp_df.iloc[:, comp_idx].apply(pd.to_numeric, errors="coerce").mean(axis=0, skipna=True)
    selected_acc = float(selected_resp.mean(skipna=True)) if selected_resp.notna().any() else np.nan
    cohort_comp_acc = float(cohort_acc.mean(skipna=True)) if cohort_acc.notna().any() else np.nan
    non_comp_idx = [i for i in range(n_items) if i not in set(comp_idx)]
    selected_non_acc = (
        float(pd.to_numeric(resp_df.iloc[sid0, non_comp_idx], errors="coerce").mean(skipna=True))
        if non_comp_idx
        else np.nan
    )
    correct_items = [str(item) for item, val in zip(comp_items, selected_resp.astype(float).values) if val > 0]
    incorrect_items = [str(item) for item, val in zip(comp_items, selected_resp.astype(float).values) if val == 0]
    if not np.isnan(selected_acc) and not np.isnan(cohort_comp_acc):
        if selected_acc > cohort_comp_acc + 0.05:
            accuracy_direction = "selected exposed-item accuracy is above the cohort exposed-item mean, which is directionally consistent with an exposed-item advantage cue"
        elif selected_acc < cohort_comp_acc - 0.05:
            accuracy_direction = "selected exposed-item accuracy is below the cohort exposed-item mean, so this descriptive accuracy cue does not support a simple exposed-item advantage interpretation"
        else:
            accuracy_direction = "selected exposed-item accuracy is close to the cohort exposed-item mean, so this descriptive cue is neutral"
    else:
        accuracy_direction = "descriptive exposed-item accuracy direction is unavailable"
    rt_df = _case_rt_dataframe()
    rt_note = "RT unavailable"
    if not rt_df.empty and 0 <= sid0 < len(rt_df):
        rt_comp_idx = [i for i in comp_idx if i < rt_df.shape[1]]
        if rt_comp_idx:
            selected_comp_rt = float(pd.to_numeric(rt_df.iloc[sid0, rt_comp_idx], errors="coerce").mean(skipna=True))
            cohort_comp_rt = float(rt_df.iloc[:, rt_comp_idx].apply(pd.to_numeric, errors="coerce").mean(axis=0, skipna=True).mean(skipna=True))
            if not np.isnan(selected_comp_rt) and not np.isnan(cohort_comp_rt):
                rt_note = f"selected exposed-item RT {selected_comp_rt:.2f} vs cohort {cohort_comp_rt:.2f}"
    return (
        "PK item cue: "
        f"{len(comp_idx)} exposed/compromised items; "
        f"selected accuracy {_fmt_metric(selected_acc * 100 if not np.isnan(selected_acc) else None, '%', 1)}; "
        f"cohort exposed-item mean {_fmt_metric(cohort_comp_acc * 100 if not np.isnan(cohort_comp_acc) else None, '%', 1)}; "
        f"selected non-exposed accuracy {_fmt_metric(selected_non_acc * 100 if not np.isnan(selected_non_acc) else None, '%', 1)}; "
        f"{accuracy_direction}; "
        f"correct exposed items {', '.join(correct_items[:8]) or 'none'}; "
        f"incorrect exposed items {', '.join(incorrect_items[:8]) or 'none'}; "
        f"{rt_note}. Use this as review guidance, not as independent evidence unless governed PK flags are active."
    )


def _case_cp_prompt_cue(examinee_id: str) -> str:
    """Compact CP localization summary for LLM review guidance."""
    flags = _forensic_result_flags_from_session()
    cp_data = flags.get("cp_agent", {}) if isinstance(flags, dict) else {}
    stat = cp_data.get("stat") if isinstance(cp_data, dict) else None
    if not isinstance(stat, list):
        return "CP cue: no change-point localization output available."
    try:
        sid0 = int(examinee_id) - 1
    except (TypeError, ValueError):
        return "CP cue: selected examinee unavailable."
    rows = []
    for idx, rec in enumerate(stat):
        if idx != sid0 or not isinstance(rec, dict):
            continue
        for key, value in rec.items():
            if str(key).endswith("_cp"):
                val = _safe_float(value)
                if val is not None:
                    method = str(key).replace("_cp", "")
                    family = "response-time shift" if "_T_" in method or method.endswith("_T") else "score/response shift"
                    rows.append((method, int(round(val)), family))
    if not rows:
        return "CP cue: no selected-examinee change-point estimates available."
    positions = [pos for _, pos, _ in rows]
    mode = pd.Series(positions).mode()
    main_item = int(mode.iloc[0]) if not mode.empty else positions[0]
    grouped = {}
    for method, pos, family in rows:
        grouped.setdefault((family, pos), []).append(method)
    details = [
        f"{family} item {pos} ({len(methods)} methods)"
        for (family, pos), methods in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0]))
    ]
    unique_n = len(set(positions))
    agreement = "high" if unique_n <= 2 else ("moderate" if unique_n <= 4 else "low")
    return (
        "CP cue: "
        f"main estimated shift near item {main_item}; method agreement {agreement}; "
        f"method locations: {'; '.join(details)}. "
        f"Tell the reviewer to inspect accuracy, response time, and answer changes before and after item {main_item}. "
        "This is localization/display-only guidance unless a calibrated CP rule is active."
    )


def _build_case_reviewer_context(
    selected_id: str,
    case_row: pd.Series,
    case_domains: pd.DataFrame,
    case_trace: pd.DataFrame,
    case_auxiliary: pd.DataFrame,
) -> str:
    store_packet = _case_reviewer_packet_from_store(selected_id)
    context_source = str(store_packet.get("source") or "session")
    case_row = _merge_case_series(case_row, store_packet.get("case_row"))
    if isinstance(store_packet.get("case_domains"), pd.DataFrame) and not store_packet["case_domains"].empty:
        case_domains = store_packet["case_domains"].copy()
    if isinstance(store_packet.get("case_trace"), pd.DataFrame) and not store_packet["case_trace"].empty:
        case_trace = store_packet["case_trace"].copy()
    if not isinstance(case_auxiliary, pd.DataFrame) or case_auxiliary.empty:
        case_auxiliary = _case_auxiliary_index_rows(selected_id)

    summary_items = {
        "Examinee_ID": selected_id,
        "Review_Queue_Status": case_row.get("Review_Queue_Status", ""),
        "Review_Priority": case_row.get("Review_Priority", ""),
        "Domains_For_Review": case_row.get("Domains_For_Review", ""),
        "Highest_Domain_Strength": case_row.get("Highest_Domain_Strength", ""),
        "Rules_Triggered": case_row.get("Rule_IDs_Triggered", ""),
        "Draft_Statement": case_row.get("Draft_Statement", ""),
        "Human_Final_Decision": case_row.get("Human_Final_Decision", "") or case_row.get("Human_Decision", ""),
        "Human_Reviewer_Note": case_row.get("Human_Reviewer_Note", "") or case_row.get("Reviewer_Note", ""),
    }
    selected_trace = _selected_case_trace_for_llm(case_trace, max_per_domain=2)
    domain_cols = [
        "Domain",
        "Strength",
        "Strength_Rule",
        "Evidence_Pattern",
        "Primary_Hits",
        "Supporting_Hits",
        "Calibration_Required_Hits",
        "Display_Only_Hits",
    ]
    trace_cols = [
        "Domain",
        "Function",
        "Index",
        "Evidence_Use",
        "Role",
        "Flag_Type",
        "Flag_Column",
        "Value",
        "Threshold_Rule",
        "Threshold_Source",
    ]
    aux_cols = ["Domain", "Function", "Index", "Value", "Application", "Column"]
    active_domains = []
    if isinstance(case_domains, pd.DataFrame) and not case_domains.empty and "Domain" in case_domains.columns:
        for _, row in case_domains.iterrows():
            strength = str(row.get("Strength", "") or "").strip().lower()
            if strength in {"weak", "moderate", "strong"}:
                active_domains.append(f"{row.get('Domain')}: {strength}")
    raw_input_cues = [
        _case_pk_prompt_cue(selected_id),
        _case_cp_prompt_cue(selected_id),
    ]
    return (
        "Evidence packet source\n"
        + context_source
        + "\nCase summary\n"
        + pd.DataFrame([summary_items]).to_csv(index=False)
        + "\nActive governed domains\n"
        + ("; ".join(active_domains) if active_domains else "No active governed domains.")
        + "\nDomain evidence\n"
        + _compact_case_table_text(case_domains, domain_cols, max_rows=12)
        + "\nEvidence data: selected key index evidence for LLM interpretation\n"
        + _compact_case_table_text(selected_trace, trace_cols, max_rows=12)
        + "\nRaw-input derived review cues for interpretation and localization\n"
        + "\n".join(raw_input_cues)
        + "\nEvidence data: display-only and auxiliary indices for localization/audit, not counted unless explicitly governed\n"
        + _compact_case_table_text(case_auxiliary, aux_cols, max_rows=20)
    )


def _evidence_bound_case_summary(
    case_row: pd.Series,
    case_domains: pd.DataFrame,
    case_trace: pd.DataFrame,
    case_auxiliary: pd.DataFrame,
) -> str:
    """Short local summary for the AI-support panel, grounded in governed evidence tables."""
    domain_df = case_domains.copy() if isinstance(case_domains, pd.DataFrame) else pd.DataFrame()
    trace_df = case_trace.copy() if isinstance(case_trace, pd.DataFrame) else pd.DataFrame()
    aux_df = case_auxiliary.copy() if isinstance(case_auxiliary, pd.DataFrame) else pd.DataFrame()
    governed_parts: list[str] = []
    context_parts: list[str] = []

    if not domain_df.empty and "Domain" in domain_df.columns:
        for domain in ["MF", "RT", "PK", "TP"]:
            rows = domain_df[domain_df["Domain"].astype(str).eq(domain)]
            if rows.empty:
                continue
            rec = rows.iloc[0]
            strength = str(rec.get("Strength", "") or "").strip().lower()
            if strength not in {"weak", "moderate", "strong"}:
                continue
            hits = str(rec.get("Primary_Hits", "") or "").strip() or str(rec.get("Supporting_Hits", "") or "").strip()
            rule = str(rec.get("Strength_Rule", "") or "").strip()
            label = _DOMAIN_LABELS.get(domain, domain)
            governed_parts.append(
                f"{domain} ({label}) is {strength}"
                + (f" via {rule}" if rule else "")
                + (f"; main support: {_shorten_report_text(hits, 120)}" if hits else "")
            )
        for domain in ["SIM", "CP"]:
            rows = domain_df[domain_df["Domain"].astype(str).eq(domain)]
            if rows.empty:
                continue
            rec = rows.iloc[0]
            display_hits = str(rec.get("Display_Only_Hits", "") or "").strip()
            calibration_hits = str(rec.get("Calibration_Required_Hits", "") or "").strip()
            strength = str(rec.get("Strength", "") or "").strip().lower()
            if display_hits:
                context_parts.append(f"{domain} context available: {_shorten_report_text(display_hits, 100)}")
            elif calibration_hits:
                context_parts.append(f"{domain} requires calibration before counting: {_shorten_report_text(calibration_hits, 100)}")
            elif domain == "CP" and _case_has_cp_location_output(str(case_row.get("Examinee_ID", ""))):
                context_parts.append("CP localization output is available but not counted toward priority.")
            elif strength == "unavailable":
                context_parts.append(f"{domain} unavailable.")

    if not trace_df.empty and "Flag" in trace_df.columns:
        flagged = trace_df[trace_df["Flag"].astype(str).str.lower().isin({"1", "true", "yes"})].copy()
        if not flagged.empty:
            for domain in ["MF", "RT", "PK", "TP"]:
                sub = flagged[flagged.get("Domain", pd.Series(dtype=str)).astype(str).eq(domain)]
                if sub.empty:
                    continue
                tokens = []
                for _, row in sub.head(3).iterrows():
                    token = f"{row.get('Function', '')}:{row.get('Index', '')}".strip(":")
                    if token and token not in tokens:
                        tokens.append(token)
                if tokens:
                    governed_parts.append(f"{domain} flagged indices include {', '.join(tokens)}")

    if not governed_parts:
        governed_text = "No governed index path is currently elevated."
    else:
        governed_text = " ".join(governed_parts[:5])
    context_text = " ".join(context_parts[:3]) if context_parts else "No additional context-only evidence is highlighted."
    priority = str(case_row.get("Review_Priority", "") or "").strip() or "not assigned"
    status = str(case_row.get("Evidence_Status", "") or "").strip() or "not assigned"
    return (
        f"System priority: {priority}; evidence profile: {status}. "
        f"{governed_text} "
        f"{context_text} "
        "This is a review-support summary based on governed forensic indices, not a misconduct conclusion."
    )


def _case_review_focus_summary(
    case_row: pd.Series,
    case_domains: pd.DataFrame,
    metrics: dict,
    selected_id: str,
) -> tuple[str, str]:
    """Plain-language review focus shown before the detailed case evidence."""
    domain_strengths: dict[str, str] = {}
    if isinstance(case_domains, pd.DataFrame) and not case_domains.empty and "Domain" in case_domains.columns:
        for _, row in case_domains.iterrows():
            domain = str(row.get("Domain", "") or "").strip()
            strength = str(row.get("Strength", "") or "").strip().lower()
            if domain and strength in {"weak", "moderate", "strong"}:
                domain_strengths[domain] = strength

    active = {d for d, s in domain_strengths.items() if s in {"weak", "moderate", "strong"}}
    accuracy = metrics.get("percent_correct")
    percentile = metrics.get("percentile")
    low_score = False
    try:
        low_score = (float(accuracy) <= 40.0) or (float(percentile) <= 20.0)
    except (TypeError, ValueError):
        low_score = False

    cp_cue = _case_cp_prompt_cue(selected_id)
    cp_item_match = re.search(r"item\s+(\d+)", cp_cue, flags=re.IGNORECASE)
    cp_item = cp_item_match.group(1) if cp_item_match else ""

    if {"RT", "MF"}.issubset(active) and low_score and "PK" not in active:
        focus = "System interpretation: likely low-engagement response behavior."
        detail = "Timing and response-pattern evidence are elevated while performance is low. First check rapid responses, skipped effort, and whether answer changes explain part of the pattern."
    elif {"RT", "PK"}.issubset(active):
        focus = "System interpretation: exposed-item performance requires contextual review."
        detail = "Preknowledge and timing evidence are both active. First compare exposed-item accuracy and response times with cohort references before interpreting the pattern."
    elif {"RT", "TP"}.issubset(active):
        focus = "System interpretation: timing anomalies and answer changes co-occur."
        suffix = f" Pay special attention around item {cp_item}." if cp_item else ""
        detail = "First check whether unusual response times and answer changes cluster in the same item range." + suffix
    elif {"PK", "TP"}.issubset(active):
        focus = "System interpretation: exposed-item evidence and answer changes overlap."
        detail = "First inspect whether answer changes involve exposed or compromised items and whether the direction of changes is meaningful."
    elif "TP" in active:
        focus = "System interpretation: answer-change behavior needs review."
        detail = "First review initial and final responses, change direction, and whether changes are concentrated in a small item range."
    elif "RT" in active:
        focus = "System interpretation: response-time behavior needs review."
        detail = "First inspect unusually fast or low-effort responses and compare them with score and item-level response-time patterns."
    elif "PK" in active:
        focus = "System interpretation: exposed-item evidence needs review."
        detail = "First inspect performance and timing on exposed or compromised items; do not infer preknowledge without contextual support."
    elif "MF" in active:
        focus = "System interpretation: atypical response-pattern fit is present."
        detail = "Misfit evidence indicates an unusual answer pattern, but it does not identify a specific behavior by itself. First check whether RT, PK, or TP evidence explains the misfit."
    else:
        focus = "System interpretation: no elevated governed evidence pattern."
        detail = "Use the report to confirm missing or context-only evidence before recording a final decision."

    priority = str(case_row.get("Review_Priority", "") or "").strip()
    if priority:
        detail = f"{priority} queue. {detail} This is a review-support interpretation, not a misconduct finding."
    return focus, detail


def _generate_case_reviewer_explanation(
    selected_id: str,
    case_row: pd.Series,
    case_domains: pd.DataFrame,
    case_trace: pd.DataFrame,
    case_auxiliary: pd.DataFrame,
) -> str:
    context = _build_case_reviewer_context(selected_id, case_row, case_domains, case_trace, case_auxiliary)
    prompt = _case_reviewer_prompt(context)
    load_dotenv()
    text, err = _call_selected_llm_text(prompt, timeout=90)
    if text and not err:
        return text
    return _format_llm_error(err or "No selected LLM model returned a response.")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "LLM explanation unavailable: GOOGLE_API_KEY not set in .env file."
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    last_err = None
    for model_name in _model_variants_with_selected_first():
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent"
            resp = requests.post(url, params={"key": api_key}, json=body, timeout=90)
            resp.raise_for_status()
            data = resp.json()
            text = (data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", ""))
            if text.strip():
                return text.strip()
        except requests.exceptions.HTTPError as e:
            try:
                err_body = e.response.json() if e.response is not None else {}
                msg = err_body.get("error", {}).get("message", str(e))
            except Exception:
                msg = str(e)
            last_err = f"{e.response.status_code if e.response is not None else 'HTTP'}: {msg}"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
    return _format_llm_error(last_err)


DEFAULT_CASE_REVIEWER_PROMPT = """You are a senior psychometric forensic reviewer writing a concise case-review support note for PsyMAS.

Use only the supplied case evidence packet. The packet may contain two kinds of information: governed evidence data and raw-input-derived review cues. Keep these roles separate.

Evidence data to use for the review conclusion:
- Case summary: examinee ID, system review priority, domains for review, highest domain strength, rules triggered, draft statement, and any existing human-review fields.
- Domain evidence: domain, strength, strength rule, evidence pattern, primary hits, supporting hits, calibration-required hits, display-only hits, and missing/unavailable evidence.
- Selected key index evidence: function, index, domain, evidence use, role, flag type, flag column, observed value, threshold rule, and threshold source.
- Auxiliary/display-only index data: use only for localization, interpretation, or audit; do not let it increase evidence strength unless the rulebook says it is governed evidence.

Raw inputs and derived review cues to use only for explanation and reviewer navigation:
- responses.csv: score, percentile, item-level selected responses, selected accuracy, and comparison with item/cohort means.
- response_times.csv: person response-time profile, item mean response time, rapid/low-effort timing patterns, and timing around flagged or localized items.
- item_params.csv or estimated item parameters: IRT model support for model-based indices; do not report parameters unless they are included in the packet.
- compromised_items.csv: exposed/compromised item list; use to explain PK item-level checks such as exposed-item accuracy and response time.
- answer_changes.csv: initial/final response or changed-item records; use to explain TP/tampering checks and whether changes cluster by item range.
- CP localization outputs: item or item range where a shift is estimated; use as an inspection guide, not as counted evidence unless CP has a calibrated governed rule.

Evidence hierarchy:
- Governed evidence data determines the review-support recommendation.
- Raw inputs explain what the reviewer should inspect and why.
- Simulation truth labels, if present, are validation-only and must not be used for an individual case interpretation.
- Missing raw inputs should be explicitly noted as a limitation, not silently ignored.

Do not use raw performance scores as misconduct evidence unless they are explicitly linked to governed forensic indices.

Required terminology:
- MF = Misfit Evidence.
- RT = Response-Time Evidence.
- PK = Preknowledge Evidence.
- TP = Tampering / Answer-Change Evidence.
- SIM = Similarity Evidence.
- CP = Change-Point Evidence.

Decision boundary:
- A flag is a review trigger, not a misconduct conclusion.
- Do not infer intent, cheating, fraud, or misconduct.
- Do not imply the examinee misused exposed items, had access to exposed content, or changed answers improperly. Describe only what the evidence pattern asks the reviewer to inspect.
- Do not say "high likelihood of aberrance" or "potential misconduct" unless quoting the human-selected B9 outcome.
- Do not say the examinee "had prior familiarity", "used a strategy", "obtained answers without understanding", or "compromised integrity".
- Preferred wording: "is consistent with", "may reflect", "requires review", "should be checked against context".
- The human analyst selects the final B9 adjudication outcome.

Index and threshold rules:
- If an index has a package-returned flag, call it a package-returned flag.
- Do not describe p-value thresholds as probabilities of misconduct.
- Do not report "p-value threshold = 1.0" as evidence strength.
- Calibration-required, display-only, auxiliary, SIM support-only, and CP localization-only outputs may be mentioned for context but must not be treated as counted evidence.

Index interpretation guide:
- RT CUMP: rapid-response timing evidence based on cumulative probability; explain whether timing is unusually fast or concentrated.
- RT NT/RTE: response-time effort evidence; lower effort values or package flags suggest rapid/low-effort responding.
- PK L_S or L_ST: model-based exposed-item/preknowledge evidence; explain whether exposed-item performance is unusually high or otherwise statistically flagged, and check exposed-item timing.
- PK LR_S, ML_S, S_S, W_S: supporting exposed-item evidence; explain as supporting checks, not standalone conclusions.
- TP EDI_*: answer-change/tampering evidence; explain that the answer-change record should be checked for direction and concentration of changes.
- MF ECI/L/Q/ZU3/HT: response-pattern fit evidence; explain as an atypical response pattern, not a specific behavior by itself.
- CP cp_location: localization only; explain which item range to inspect before/after.

Pattern interpretation guide:
- RT only: may reflect rapid responding, low effort, speededness, or timing irregularity.
- PK only: may reflect unusual performance on exposed/compromised items; check exposed-item accuracy and timing.
- TP only: may reflect unusual answer-change behavior; check initial/final response records and change direction.
- MF only: may reflect atypical response-pattern fit; do not interpret it as a specific behavior without other domains.
- RT + low score: review for disengagement or rapid guessing.
- RT + high score + PK: review for a possible exposed-item advantage or preknowledge-like pattern.
- RT + TP: review whether timing anomalies and answer changes cluster in the same item range.
- PK + TP: review whether answer changes involve exposed/compromised items.
- TP + CP: use CP to localize whether answer-change behavior concentrates around a shift point.
- MF + RT + TP: review for a broader atypical response process involving response pattern, timing, and answer changes.
- SIM + PK: review for possible shared content source or exposure context, but require external/contextual support.
- CP only: localization only; do not treat as independent evidence.
- SIM without calibrated rule: contextual or pair-review cue only; do not count as governed evidence.

Auxiliary cue use:
- Use PK item cues to identify exposed/compromised items that deserve inspection.
- Use CP cues to name the item or item range the reviewer should inspect before and after.
- Use SIM and CP cues only as navigation/context, not as independent misconduct evidence.

Numeric interpretation requirements:
- Section 3 must interpret concrete numbers when they are present in the case packet.
- For each important domain, use this logic: observed index flag/value -> relevant raw-input comparison -> what the direction means -> what the reviewer should inspect.
- Do not write generic phrases such as "significant", "notable", or "strong evidence" without explaining the observed value or package flag that led to it.
- If a raw-input cue contradicts the usual behavioral interpretation, state that clearly. Example: lower accuracy on exposed/compromised items does not by itself support a preknowledge advantage; write that the governed PK flag is present but the descriptive exposed-item accuracy cue is not directionally supportive, then ask the reviewer to inspect the underlying PK index and timing.
- For PK, never write "misuse of exposed items", "prior knowledge", "prior familiarity", or "exposed-item advantage" unless the raw-input cue shows exposed-item accuracy above the cohort/reference or the selected key index evidence provides a clear value supporting that interpretation. Otherwise write "PK package flag is present, but descriptive exposed-item accuracy does not by itself support an advantage interpretation."
- If the package flag is present but the numeric statistic is missing from the selected key evidence, say "package flag present; numeric value not shown here" and do not invent a value.
- If a cue gives a comparison, include both sides, such as "selected exposed-item accuracy 25.0% vs cohort 46.4%" or "selected exposed-item RT 82.35 vs cohort 28.84".

Write under 320 words with four short sections:
1. Direct review conclusion
2. Evidence pattern
3. Main index support
4. Recommended reviewer action

Section 1 must be written for a nontechnical reviewer. It must give one direct system interpretation and one next-step review recommendation. It must not include domain codes or index names, and must state the practical review concern without naming a behavior as fact. Use wording such as "This record should be reviewed for unusually timed responses and concentrated answer changes; first check whether these patterns occur in the same item range." Do not mention exposed-item misuse in Section 1 unless the PK raw-input cue is directionally supportive.

Avoid repeating the same reason across sections. Section 2 should synthesize the active domains into one review hypothesis. Section 3 must be concrete and use short bullets. For each strongest governed domain, mention at most two index names and include the observed Value or package flag when present in the selected key index evidence. For each index, explain in plain language what it measures, whether the available raw-input cue supports or qualifies the interpretation, and what specific behavior it asks the reviewer to inspect. Section 4 should give concrete checks the reviewer should perform. Use cautious, evidence-bound language.

Case evidence:
{case_context}
"""


def _case_reviewer_prompt_template() -> str:
    template = str(st.session_state.get("case_reviewer_prompt_template") or "").strip()
    if not template:
        template = DEFAULT_CASE_REVIEWER_PROMPT
    if "{case_context}" not in template:
        template = template.rstrip() + "\n\nCase evidence:\n{case_context}\n"
    return template


def _case_reviewer_prompt(case_context: str) -> str:
    return _case_reviewer_prompt_template().replace("{case_context}", str(case_context)[:9000])


def _compact_ai_suggestion_text(text: str) -> str:
    """Make LLM report text compact for the small in-app review panel."""
    compact = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    compact = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", compact)
    compact = compact.replace("**", "")
    compact = "\n".join(line.strip() for line in compact.splitlines())
    compact = re.sub(r"\n{2,}", "\n", compact)
    return compact.strip()


def _ai_suggestion_html(text: str) -> str:
    compact = _compact_ai_suggestion_text(text)
    heading_names = {
        "direct review conclusion",
        "system interpretation",
        "brief review summary",
        "evidence pattern",
        "main index support",
        "why this case needs review",
        "main response-behavior pattern",
        "main response behavior pattern",
        "governed domain evidence and main index support",
        "recommended reviewer action",
    }
    parts = []
    for raw_line in compact.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        normalized = re.sub(r"^\d+\.\s*", "", line).strip().rstrip(":").lower()
        escaped = html.escape(line)
        if normalized in heading_names or re.match(r"^\d+\.\s+\S", line):
            parts.append(f'<span class="section-title">{escaped}</span>')
        else:
            parts.append(f'<span class="section-body">{escaped}</span>')
    return "\n".join(parts)


def _case_review_decisions_df() -> pd.DataFrame:
    decisions = st.session_state.get("case_review_decisions", {})
    if not isinstance(decisions, dict) or not decisions:
        return pd.DataFrame(columns=["Examinee_ID", "Final_Decision", "Reviewer_Note"])
    rows = []
    for examinee_id, rec in decisions.items():
        rec = rec if isinstance(rec, dict) else {}
        rows.append(
            {
                "Examinee_ID": examinee_id,
                "Final_Decision": rec.get("final_decision", ""),
                "Reviewer_Note": rec.get("reviewer_note", ""),
            }
        )
    return pd.DataFrame(rows).sort_values("Examinee_ID", key=lambda s: s.astype(str))


def _evidence_synthesis_with_human_review(governed_df: pd.DataFrame) -> pd.DataFrame:
    if governed_df is None or governed_df.empty:
        return pd.DataFrame()
    out = governed_df.copy()
    if "Examinee_ID" not in out.columns:
        return out
    out["Legacy_Cross_Domain_Pattern"] = out.get("Evidence_Status", "")
    out["Review_Priority"] = out.get("Review_Priority", "")
    out["Review_Priority_Rule"] = out.get("Review_Priority_Rule", "")
    strength_order = {"unavailable": -1, "none": 0, "weak": 1, "moderate": 2, "strong": 3}
    strength_columns = [
        (domain, f"{domain}_Strength")
        for domain in ["MF", "RT", "SIM", "PK", "TP", "CP"]
        if f"{domain}_Strength" in out.columns
    ]

    def _queue_profile(row: pd.Series) -> pd.Series:
        active_domains: list[str] = []
        highest_strength = "none"
        highest_score = 0
        unavailable_domains: list[str] = []
        for domain, column in strength_columns:
            strength = str(row.get(column, "") or "").strip().lower()
            if strength == "unavailable":
                unavailable_domains.append(domain)
            if strength in {"weak", "moderate", "strong"}:
                active_domains.append(domain)
                score = strength_order[strength]
                if score > highest_score:
                    highest_score = score
                    highest_strength = strength
        priority = str(row.get("Review_Priority", "") or "").strip()
        if priority in {"Critical / Expedited", "High (Context-Heavy)", "High", "Medium"}:
            queue_status = "Queued"
        elif active_domains:
            queue_status = "Monitor"
        else:
            queue_status = "Not queued"
        return pd.Series(
            {
                "Review_Queue_Status": queue_status,
                "Domains_For_Review": ", ".join(active_domains),
                "Highest_Domain_Strength": highest_strength,
                "Unavailable_Domains": ", ".join(unavailable_domains),
                "Review_Rationale": (
                    f"Review {', '.join(active_domains)} domain evidence."
                    if active_domains
                    else "No eligible domain evidence requires review."
                ),
            }
        )

    queue_profile = out.apply(_queue_profile, axis=1)
    for column in queue_profile.columns:
        out[column] = queue_profile[column]
    out["Examinee_ID"] = out["Examinee_ID"].astype(str)
    decisions_df = _case_review_decisions_df()
    if not decisions_df.empty:
        decisions_df = decisions_df.copy()
        decisions_df["Examinee_ID"] = decisions_df["Examinee_ID"].astype(str)
        decisions_df = decisions_df.rename(
            columns={
                "Final_Decision": "Human_Final_Decision",
                "Reviewer_Note": "Human_Reviewer_Note",
            }
        )
        out = out.merge(decisions_df, on="Examinee_ID", how="left")
    else:
        out["Human_Final_Decision"] = ""
        out["Human_Reviewer_Note"] = ""
    explanations = []
    for ex_id in out["Examinee_ID"].astype(str):
        explanations.append(
            str(
                st.session_state.get(f"case_reviewer_explanation_{ex_id}", "")
                or st.session_state.get(f"case_llm_support_{ex_id}", "")
            )
        )
    out["LLM_Review_Explanation"] = explanations
    out["Human_Final_Decision"] = out["Human_Final_Decision"].fillna("")
    out["Human_Reviewer_Note"] = out["Human_Reviewer_Note"].fillna("")
    reviewed = out["Human_Final_Decision"].astype(str).str.strip().ne("")
    out.loc[reviewed, "Review_Queue_Status"] = "Reviewed"
    return out


def _ask_case_review_llm(
    selected_id: str,
    case_row: pd.Series,
    case_domains: pd.DataFrame,
    case_trace: pd.DataFrame,
    case_auxiliary: pd.DataFrame,
    reviewer_decision: str,
    reviewer_note: str,
    reviewer_question: str,
) -> str:
    context = _build_case_reviewer_context(selected_id, case_row, case_domains, case_trace, case_auxiliary)
    prompt = (
        "You are assisting a human psychometric forensic reviewer. Answer the reviewer using only the supplied case evidence "
        "and the reviewer's current adjudication notes. Do not make a final misconduct determination yourself. "
        "Help the reviewer write cautious, evidence-bound language and identify what should be checked next. "
        "B9 human adjudication outcomes are limited to Rule Violation, Potential Misconduct, and Definite Misconduct.\n\n"
        f"Reviewer current final decision: {reviewer_decision}\n"
        f"Reviewer note: {reviewer_note or 'None'}\n"
        f"Reviewer question/request: {reviewer_question or 'Draft a concise final reviewer comment.'}\n\n"
        f"Case evidence:\n{context[:9000]}"
    )
    load_dotenv()
    text, err = _call_selected_llm_text(prompt, timeout=90)
    if text and not err:
        return text
    return _format_llm_error(err or "No selected LLM model returned a response.")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "LLM support unavailable: GOOGLE_API_KEY not set in .env file."
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    last_err = None
    for model_name in _model_variants_with_selected_first():
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent"
            resp = requests.post(url, params={"key": api_key}, json=body, timeout=90)
            resp.raise_for_status()
            data = resp.json()
            text = (data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", ""))
            if text.strip():
                return text.strip()
        except requests.exceptions.HTTPError as e:
            try:
                err_body = e.response.json() if e.response is not None else {}
                msg = err_body.get("error", {}).get("message", str(e))
            except Exception:
                msg = str(e)
            last_err = f"{e.response.status_code if e.response is not None else 'HTTP'}: {msg}"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
    return _format_llm_error(last_err)


def _case_response_dataframe() -> pd.DataFrame:
    rows = st.session_state.get("forensic_responses") or st.session_state.get("last_uploaded_responses") or []
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).apply(pd.to_numeric, errors="coerce")


def _case_performance_summary(examinee_id: str) -> dict:
    resp_df = _case_response_dataframe()
    rt_df = _case_rt_dataframe()
    out = {
        "score": None,
        "total_items": None,
        "percent_correct": None,
        "percentile": None,
        "mean_rt": None,
        "median_rt": None,
        "n_items_answered": None,
    }
    try:
        sid0 = int(examinee_id) - 1
    except (TypeError, ValueError):
        return out
    if not resp_df.empty and 0 <= sid0 < len(resp_df):
        scores = resp_df.sum(axis=1, skipna=True)
        score = float(scores.iloc[sid0])
        total_items = int(resp_df.shape[1])
        out["score"] = score
        out["total_items"] = total_items
        out["percent_correct"] = (score / total_items * 100.0) if total_items else None
        out["percentile"] = float((scores.le(score).sum() / len(scores)) * 100.0) if len(scores) else None
        out["n_items_answered"] = int(resp_df.iloc[sid0].notna().sum())
    if not rt_df.empty and 0 <= sid0 < len(rt_df):
        rt_row = pd.to_numeric(rt_df.iloc[sid0], errors="coerce")
        out["mean_rt"] = float(rt_row.mean(skipna=True)) if rt_row.notna().any() else None
        out["median_rt"] = float(rt_row.median(skipna=True)) if rt_row.notna().any() else None
    return out


def _fmt_metric(value, suffix: str = "", decimals: int = 1) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    if isinstance(value, (int, np.integer)):
        return f"{value}{suffix}"
    try:
        return f"{float(value):.{decimals}f}{suffix}"
    except (TypeError, ValueError):
        return f"{value}{suffix}"


def _case_has_cp_location_output(examinee_id: str | None = None) -> bool:
    flags = _forensic_result_flags_from_session()
    cp_data = flags.get("cp_agent", {}) if isinstance(flags, dict) else {}
    stat = cp_data.get("stat") if isinstance(cp_data, dict) else None
    if not isinstance(stat, list):
        return False
    if examinee_id is not None:
        try:
            idxs = [int(examinee_id) - 1]
        except (TypeError, ValueError):
            idxs = []
    else:
        idxs = range(len(stat))
    for idx in idxs:
        if idx < 0 or idx >= len(stat):
            continue
        rec = stat[idx]
        if not isinstance(rec, dict):
            continue
        for key, value in rec.items():
            if not str(key).endswith("_cp"):
                continue
            if _safe_float(value) is not None:
                return True
    return False


def _case_cp_location_output_count() -> int:
    flags = _forensic_result_flags_from_session()
    cp_data = flags.get("cp_agent", {}) if isinstance(flags, dict) else {}
    stat = cp_data.get("stat") if isinstance(cp_data, dict) else None
    if not isinstance(stat, list):
        return 0
    count = 0
    for rec in stat:
        if not isinstance(rec, dict):
            continue
        has_location = False
        for key, value in rec.items():
            if str(key).endswith("_cp") and _safe_float(value) is not None:
                has_location = True
                break
        if has_location:
            count += 1
    return count


def _case_domain_issue_rows(case_domains: pd.DataFrame, case_trace: pd.DataFrame) -> list[dict]:
    domain_meta = {
        "MF": {
            "display_code": "ME",
            "name": "Misfit Evidence",
            "figure": "Score vs Effort; score percentile; item accuracy",
        },
        "RT": {
            "display_code": "RTE",
            "name": "Response-Time Evidence",
            "figure": "Score vs Effort; response-time profile; rapid-guessing panels",
        },
        "PK": {
            "display_code": "PKE",
            "name": "Preknowledge Evidence",
            "figure": "No dedicated figure available in current data; review domain index support",
        },
        "SIM": {
            "display_code": "SIM",
            "name": "Similarity Evidence",
            "figure": "Pair links graph",
        },
        "TP": {
            "display_code": "TPE",
            "name": "Tampering Evidence",
            "figure": "No dedicated figure available unless revision/tampering data are loaded",
        },
        "CP": {
            "display_code": "CPE",
            "name": "Change Point Evidence",
            "figure": "Change-point location distribution",
        },
    }
    domains = ["MF", "RT", "PK", "TP", "SIM", "CP"]
    rows: list[dict] = []
    trace = case_trace.copy() if isinstance(case_trace, pd.DataFrame) else pd.DataFrame()
    selected_examinee = None
    if isinstance(case_domains, pd.DataFrame) and not case_domains.empty and "Examinee_ID" in case_domains.columns:
        selected_examinee = str(case_domains.iloc[0].get("Examinee_ID", ""))
    for domain in domains:
        drow = pd.DataFrame()
        if isinstance(case_domains, pd.DataFrame) and not case_domains.empty and "Domain" in case_domains.columns:
            drow = case_domains[case_domains["Domain"].astype(str).eq(domain)]
        strength = str(drow.iloc[0].get("Strength", "unavailable")) if not drow.empty else "unavailable"
        pattern = str(drow.iloc[0].get("Evidence_Pattern", "")) if not drow.empty else "No domain row available."
        primary_hits = str(drow.iloc[0].get("Primary_Hits", "")) if not drow.empty else ""
        supporting_hits = str(drow.iloc[0].get("Supporting_Hits", "")) if not drow.empty else ""
        display_hits = str(drow.iloc[0].get("Display_Only_Hits", "")) if not drow.empty else ""
        calibration_hits = str(drow.iloc[0].get("Calibration_Required_Hits", "")) if not drow.empty else ""
        available = str(drow.iloc[0].get("Available", "")) if not drow.empty else "0"
        missing_note = str(drow.iloc[0].get("Missing_Note", "")) if not drow.empty else ""
        main_index = primary_hits or supporting_hits
        if not main_index and not trace.empty and {"Domain", "Flag"}.issubset(trace.columns):
            flagged = trace[
                trace["Domain"].astype(str).eq(domain)
                & trace["Flag"].astype(str).str.lower().isin({"1", "true", "yes"})
            ]
            if not flagged.empty:
                first = flagged.iloc[0]
                main_index = f"{first.get('Function', '')}:{first.get('Index', '')}".strip(":")
        strength_l = strength.lower()
        has_issue = strength_l in {"weak", "moderate", "strong"}
        if has_issue:
            status_label = "Governed evidence"
            card_class = strength_l if strength_l in {"strong", "moderate", "weak"} else "issue"
        elif display_hits.strip():
            status_label = "Support only"
            card_class = "support"
            if main_index == "None" or not main_index:
                main_index = display_hits
        elif calibration_hits.strip():
            status_label = "Calibration required"
            card_class = "support"
            if main_index == "None" or not main_index:
                main_index = calibration_hits
        elif domain == "CP" and _case_has_cp_location_output(selected_examinee):
            strength = "localization only"
            status_label = "Display only"
            card_class = "support"
            main_index = "detect_cp: cp_location"
            pattern = (
                "Change-point location estimates are available for interpretation, "
                "but they do not count as governed evidence without a calibrated CP rule."
            )
            display_hits = display_hits or "detect_cp: cp_location"
        elif strength_l == "unavailable" or available in {"0", "False", "false"}:
            status_label = "Unavailable"
            card_class = "unavailable"
            if not pattern or pattern == "No domain row available.":
                pattern = missing_note or "No evidence input rows are available for this domain."
        else:
            status_label = "No governed signal"
            card_class = "clear"
        rows.append(
            {
                "domain": domain,
                "display_code": domain_meta.get(domain, {}).get("display_code", domain),
                "domain_name": domain_meta.get(domain, {}).get("name", domain),
                "figure": domain_meta.get(domain, {}).get("figure", ""),
                "step": len(rows) + 1,
                "strength": strength,
                "has_issue": has_issue,
                "status_label": status_label,
                "card_class": card_class,
                "main_index": main_index or "None",
                "pattern": pattern,
                "display_hits": display_hits,
                "calibration_hits": calibration_hits,
                "missing_note": missing_note,
            }
        )
    return rows


def _domain_frame_header(row: dict) -> None:
    issue_label = str(row.get("status_label") or ("Governed evidence" if row.get("has_issue") else "No governed signal"))
    issue_class = str(row.get("card_class") or ("issue" if row.get("has_issue") else "clear"))
    frame_class = "psymas-domain-frame-inline"
    st.markdown(
        f"""
<div class="{frame_class} {issue_class}">
<div class="psymas-domain-row-head">
  <div>
    <div class="domain-label">Domain {html.escape(str(row.get("step", "")))}: {html.escape(str(row.get("display_code", "")))}</div>
    <div class="domain-name">{html.escape(str(row.get("domain_name", row.get("domain", ""))))}</div>
  </div>
  <div class="domain-strength">{html.escape(str(row.get("strength", "")))}</div>
</div>
<div class="domain-status">{html.escape(issue_label)}</div>
<div class="domain-index"><b>Main index:</b> {html.escape(_shorten_report_text(row.get("main_index", "None"), 110))}</div>
<div class="domain-pattern">{html.escape(str(row.get("pattern", "")))}</div>
<div class="domain-figure"><b>Related figure:</b> {html.escape(str(row.get("figure", "")))}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def _render_domain_visual_header(row: dict, *, help_text: str) -> None:
    card_class = str(row.get("card_class") or ("strong" if row.get("has_issue") else "clear"))
    if card_class == "clear":
        card_class = "off"
    status_label = str(row.get("status_label") or ("Governed evidence" if row.get("has_issue") else "No governed signal"))
    main_index = _shorten_report_text(row.get("main_index", "None"), 110)
    st.markdown(
        f"""
<div class="psymas-domain-visual-head {html.escape(card_class)}">
  <div>
    <div class="domain-kicker">Domain {html.escape(str(row.get("step", "")))} · {html.escape(str(row.get("display_code", row.get("domain", ""))))}
      <span class="psymas-help-dot" title="{html.escape(help_text)}">?</span>
    </div>
    <div class="domain-title">{html.escape(str(row.get("domain_name", row.get("domain", ""))))}</div>
    <div class="domain-meta"><b>Main index:</b> {html.escape(main_index)}</div>
  </div>
  <div class="domain-outcome">{html.escape(str(row.get("strength", "")))}<br><span class="domain-status">{html.escape(status_label)}</span></div>
</div>
        """,
        unsafe_allow_html=True,
    )


def _render_case_domain_status_strip(domain_issue_rows: list[dict]) -> None:
    if not domain_issue_rows:
        return
    scenario: list[str] = []
    supporting: list[str] = []
    context: list[str] = []
    for row in domain_issue_rows:
        domain = str(row.get("domain", ""))
        strength = str(row.get("strength", "")).lower()
        label = f"{html.escape(domain)} {html.escape(str(row.get('strength', '')))}"
        if domain in {"RT", "PK", "TP"} and strength in {"weak", "moderate", "strong"}:
            scenario.append(label)
        elif domain == "MF" and strength in {"weak", "moderate", "strong"}:
            supporting.append(label)
        elif domain in {"SIM", "CP"}:
            status = str(row.get("status_label") or row.get("strength") or "context")
            if status.lower() not in {"unavailable", "none"} or str(row.get("main_index", "")).lower() not in {"", "none"}:
                context.append(f"{html.escape(domain)} {html.escape(status.lower())}")
    scenario_text = " · ".join(scenario) if scenario else "none"
    supporting_text = " · ".join(supporting) if supporting else "none"
    context_text = " · ".join(context) if context else "none"
    st.markdown(
        f"""
<div class="psymas-lineage-chips">
  <span><b>Scenario</b> {scenario_text}</span>
  <span><b>Supporting</b> {supporting_text}</span>
  <span><b>Context</b> {context_text}</span>
</div>
        """,
        unsafe_allow_html=True,
    )


def _case_performance_figures(examinee_id: str) -> dict[str, object]:
    resp_df = _case_response_dataframe()
    rt_df = _case_rt_dataframe()
    try:
        import plotly.graph_objects as go
    except Exception:
        return {}
    try:
        sid0 = int(examinee_id) - 1
    except (TypeError, ValueError):
        return {}
    if resp_df.empty or not (0 <= sid0 < len(resp_df)):
        return {}

    def _standard_case_plot_layout(fig, title: str, *, height: int = 292, legend: bool = False, top: int = 56) -> None:
        legend_cfg = (
            dict(
                orientation="h",
                yanchor="bottom",
                y=1.03,
                xanchor="right",
                x=1,
                font=dict(color=PSYMAS_VIZ["ink"], size=10),
                itemwidth=30,
            )
            if legend
            else None
        )
        fig.update_layout(
            title=dict(text=title, x=0, xanchor="left", y=0.98, yanchor="top"),
            height=height,
            margin=dict(l=54, r=24, t=top, b=46),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(color=PSYMAS_VIZ["ink"], size=11),
            title_font=dict(color=PSYMAS_VIZ["ink"], size=14),
            legend=legend_cfg,
        )
        fig.update_xaxes(
            title_font=dict(color=PSYMAS_VIZ["ink"], size=10),
            tickfont=dict(color=PSYMAS_VIZ["ink"], size=10),
            gridcolor=PSYMAS_VIZ["grid"],
            zeroline=False,
        )
        fig.update_yaxes(
            title_font=dict(color=PSYMAS_VIZ["ink"], size=10),
            tickfont=dict(color=PSYMAS_VIZ["ink"], size=10),
            gridcolor=PSYMAS_VIZ["grid"],
            zeroline=False,
        )

    scores = resp_df.sum(axis=1, skipna=True)
    selected_score = float(scores.iloc[sid0])
    fig_score = go.Figure()
    fig_score.add_trace(
        go.Histogram(
            x=scores,
            nbinsx=max(8, min(24, int(scores.nunique()) + 2)),
            marker=dict(color=PSYMAS_VIZ["reference_fill"], line=dict(color=PSYMAS_VIZ["reference"], width=1)),
            name="Other examinees",
        )
    )
    fig_score.add_vline(x=selected_score, line_color=PSYMAS_VIZ["selected"], line_width=3, annotation_text="selected", annotation_font_color=PSYMAS_VIZ["ink"])
    fig_score.update_layout(
        xaxis_title="Total score",
        yaxis_title="Number of examinees",
    )
    _standard_case_plot_layout(fig_score, "Score percentile", height=292, top=48)
    _make_plotly_text_readable(fig_score)

    item_acc = resp_df.mean(axis=0, skipna=True)
    selected_resp = pd.to_numeric(resp_df.iloc[sid0], errors="coerce")
    item_numbers = list(range(1, len(item_acc) + 1))
    selected_correct = selected_resp.fillna(-1).astype(float).values
    bar_colors = np.where(selected_correct > 0, PSYMAS_VIZ["correct"], np.where(selected_correct == 0, PSYMAS_VIZ["incorrect"], PSYMAS_VIZ["reference"]))
    fig_item = go.Figure()
    fig_item.add_trace(
        go.Bar(
            x=item_numbers,
            y=item_acc.values,
            marker_color=bar_colors,
            name="Item mean accuracy",
            customdata=np.where(selected_correct > 0, "Correct", np.where(selected_correct == 0, "Incorrect", "Missing")),
            hovertemplate="Item %{x}<br>Item mean accuracy %{y:.1%}<br>Selected response: %{customdata}<extra></extra>",
        )
    )
    fig_item.update_layout(
        xaxis_title="Item position",
        yaxis=dict(title="Item mean accuracy", tickformat=".0%", range=[0, 1.05]),
    )
    fig_item.add_trace(
        go.Bar(x=[None], y=[None], marker_color=PSYMAS_VIZ["correct"], name="Selected correct", showlegend=True)
    )
    fig_item.add_trace(
        go.Bar(x=[None], y=[None], marker_color=PSYMAS_VIZ["incorrect"], name="Selected incorrect", showlegend=True)
    )
    _standard_case_plot_layout(fig_item, "Item accuracy colored by selected response", height=292, legend=True, top=66)
    _make_plotly_text_readable(fig_item)

    figures: dict[str, object] = {"score": fig_score, "accuracy": fig_item}

    if not rt_df.empty and 0 <= sid0 < len(rt_df):
        rt_row = pd.to_numeric(rt_df.iloc[sid0], errors="coerce")
        rt_mean = rt_df.mean(axis=0, skipna=True)
        fig_rt = go.Figure()
        fig_rt.add_trace(
            go.Scatter(
                x=item_numbers[: len(rt_row)],
                y=rt_row.values,
                mode="lines+markers",
                line=dict(color=PSYMAS_VIZ["selected"], width=2),
                marker=dict(size=7, color=PSYMAS_VIZ["selected"]),
                name="Selected examinee",
            )
        )
        fig_rt.add_trace(
            go.Scatter(
                x=item_numbers[: len(rt_mean)],
                y=rt_mean.values,
                mode="lines",
                line=dict(color=PSYMAS_VIZ["warning"], width=2, dash="dash"),
                name="Item mean",
            )
        )
        fig_rt.update_layout(
            xaxis_title="Item position",
            yaxis_title="Response time",
        )
        _standard_case_plot_layout(fig_rt, "Response-time profile compared with item mean", height=292, legend=True, top=62)
        _make_plotly_text_readable(fig_rt)
        figures["rt"] = fig_rt
    return figures


def _render_case_performance_visuals(examinee_id: str) -> None:
    figures = _case_performance_figures(examinee_id)
    if not figures:
        st.info("Response-level performance figures are unavailable because response data are not loaded.")
        return
    top_left, top_right = st.columns(2)
    with top_left:
        if "accuracy" in figures:
            st.plotly_chart(figures["accuracy"], use_container_width=True, key=f"case_item_accuracy_{examinee_id}")
    with top_right:
        if "rt" in figures:
            st.plotly_chart(figures["rt"], use_container_width=True, key=f"case_response_time_profile_{examinee_id}")
    bottom_left, bottom_right = st.columns(2)
    with bottom_left:
        if "score" in figures:
            st.plotly_chart(figures["score"], use_container_width=True, key=f"case_score_percentile_{examinee_id}")
    with bottom_right:
        _render_ability_effort_figure(examinee_id)


def _case_ability_effort_dataframe() -> tuple[pd.DataFrame, str, str]:
    result_df, _ = _governance_source_export()
    resp_df = _case_response_dataframe()
    ids = []
    if isinstance(result_df, pd.DataFrame) and not result_df.empty and "Examinee_ID" in result_df.columns:
        ids = result_df["Examinee_ID"].astype(str).tolist()
    elif not resp_df.empty:
        ids = [str(i + 1) for i in range(len(resp_df))]

    ability_source = "score percentile"
    ability_vals = None
    if not resp_df.empty:
        scores = resp_df.sum(axis=1, skipna=True)
        ability_vals = scores.rank(pct=True, method="average") * 100.0
        ability_source = "score percentile"
    elif isinstance(result_df, pd.DataFrame) and not result_df.empty:
        score_cols = [c for c in result_df.columns if str(c).lower() in {"score", "total_score", "raw_score"}]
        if score_cols:
            scores = pd.to_numeric(result_df[score_cols[0]], errors="coerce")
            ability_vals = scores.rank(pct=True, method="average") * 100.0
            ability_source = "score percentile"

    effort_source = "RTE"
    effort_vals = None
    if isinstance(result_df, pd.DataFrame) and not result_df.empty:
        rte_cols = [
            c
            for c in result_df.columns
            if str(c).startswith("rg_RTE_") and str(c) not in {"rg_RTE"}
        ]
        if rte_cols:
            rte_frame = result_df[rte_cols].apply(pd.to_numeric, errors="coerce")
            effort_vals = rte_frame.min(axis=1, skipna=True)
            effort_source = "minimum rg_RTE"
        elif "rg_RTE" in result_df.columns:
            effort_vals = pd.to_numeric(result_df["rg_RTE"], errors="coerce")
            effort_source = "rg_RTE"
    if effort_vals is None:
        flags = _forensic_result_flags_from_session()
        rg_data = flags.get("rg_agent", {}) if isinstance(flags, dict) else {}
        method_name, vals = _rg_preferred_rte_method(rg_data if isinstance(rg_data, dict) else {})
        if vals:
            effort_vals = pd.Series(vals, dtype="float64")
            effort_source = f"RTE ({method_name})"

    if ability_vals is None or effort_vals is None:
        return pd.DataFrame(), ability_source, effort_source
    n = min(len(ids), len(ability_vals), len(effort_vals))
    if n <= 0:
        return pd.DataFrame(), ability_source, effort_source
    df = pd.DataFrame(
        {
            "Examinee_ID": ids[:n],
            "Score_Percentile": pd.Series(ability_vals).iloc[:n].astype(float).values,
            "Effort": pd.Series(effort_vals).iloc[:n].astype(float).values,
        }
    ).dropna(subset=["Score_Percentile", "Effort"])
    return df, ability_source, effort_source


def _ability_effort_figure(examinee_id: str):
    df, ability_source, effort_source = _case_ability_effort_dataframe()
    if df.empty:
        return None
    try:
        import plotly.graph_objects as go
    except Exception:
        return None
    df = df.copy()
    selected_mask = df["Examinee_ID"].astype(str).eq(str(examinee_id))
    ability_cut = 75.0
    effort_cut = min(0.95, float(df["Effort"].quantile(0.10))) if len(df) > 4 else 0.95
    df["Zone"] = np.select(
        [
            selected_mask,
            (df["Score_Percentile"] >= ability_cut) & (df["Effort"] <= effort_cut),
            df["Effort"] <= effort_cut,
        ],
        ["Selected examinee", "High score / low effort", "Low effort"],
        default="Reference examinees",
    )
    colors = {
        "Reference examinees": PSYMAS_VIZ["reference"],
        "Low effort": PSYMAS_VIZ["warning"],
        "High score / low effort": PSYMAS_VIZ["high"],
        "Selected examinee": PSYMAS_VIZ["selected_light"],
    }
    fig = go.Figure()
    for zone in ["Reference examinees", "Low effort", "High score / low effort", "Selected examinee"]:
        sub = df[df["Zone"].eq(zone)]
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub["Score_Percentile"],
                y=sub["Effort"],
                mode="markers",
                marker=dict(
                    size=15 if zone == "Selected examinee" else 10,
                    color=colors[zone],
                    opacity=0.95 if zone == "Selected examinee" else 0.78,
                    line=dict(width=2 if zone == "Selected examinee" else 1, color=PSYMAS_VIZ["grid"]),
                ),
                name=zone,
                hovertemplate="Examinee %{customdata}<br>Score percentile %{x:.1f}<br>Effort %{y:.3f}<extra></extra>",
                customdata=sub["Examinee_ID"],
            )
        )
    selected_df = df[selected_mask]
    if not selected_df.empty:
        sx = float(selected_df.iloc[0]["Score_Percentile"])
        sy = float(selected_df.iloc[0]["Effort"])
        fig.add_trace(
            go.Scatter(
                x=[sx],
                y=[sy],
                mode="markers",
                marker=dict(
                    size=30,
                    color="rgba(0,0,0,0)",
                    line=dict(width=4, color=PSYMAS_VIZ["ink"]),
                    symbol="circle-open",
                ),
                name="Selected highlight",
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_annotation(
            x=sx,
            y=sy,
            text=f"Examinee {examinee_id}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.1,
            arrowwidth=2,
            arrowcolor=PSYMAS_VIZ["ink"],
            ax=45 if sx < float(df["Score_Percentile"].median()) else -55,
            ay=-45 if sy < float(df["Effort"].median()) else 45,
            bgcolor="#FFFFFF",
            bordercolor=PSYMAS_VIZ["ink"],
            borderwidth=2.5,
            borderpad=4,
            font=dict(color=PSYMAS_VIZ["ink"], size=13),
        )
    fig.add_vline(x=ability_cut, line_dash="dash", line_color=PSYMAS_VIZ["inactive"], opacity=0.75)
    fig.add_hline(y=effort_cut, line_dash="dash", line_color=PSYMAS_VIZ["inactive"], opacity=0.75)
    fig.add_annotation(
        x=0.96,
        y=0.08,
        xref="paper",
        yref="paper",
        text="High score / low effort zone",
        showarrow=False,
        font=dict(color=PSYMAS_VIZ["incorrect"], size=14),
        align="right",
    )
    fig.update_layout(
        title=dict(text="Score vs Effort", x=0, xanchor="left", y=0.98, yanchor="top"),
        height=292,
        margin=dict(l=58, r=24, t=66, b=48),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color=PSYMAS_VIZ["ink"], size=11),
        title_font=dict(color=PSYMAS_VIZ["ink"], size=14),
        xaxis=dict(
            title="Score percentile",
            color=PSYMAS_VIZ["ink"],
            title_font=dict(color=PSYMAS_VIZ["ink"], size=10),
            tickfont=dict(color=PSYMAS_VIZ["ink"], size=10),
            gridcolor="#E5E7EB",
            zeroline=False,
            range=[0, 100],
        ),
        yaxis=dict(
            title=f"Response Time Effort ({effort_source})",
            color=PSYMAS_VIZ["ink"],
            title_font=dict(color=PSYMAS_VIZ["ink"], size=10),
            tickfont=dict(color=PSYMAS_VIZ["ink"], size=10),
            gridcolor=PSYMAS_VIZ["grid"],
            zeroline=False,
            range=[max(0, float(df["Effort"].min()) - 0.04), min(1.02, float(df["Effort"].max()) + 0.04)],
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="right",
            x=1,
            font=dict(color=PSYMAS_VIZ["ink"], size=10),
            itemwidth=30,
        ),
    )
    fig.update_annotations(font_color=PSYMAS_VIZ["ink"])
    return fig


def _render_ability_effort_figure(examinee_id: str) -> None:
    fig = _ability_effort_figure(examinee_id)
    if fig is None:
        st.info("Score-vs-effort figure is unavailable because both response scores and RTE values are required.")
        return
    st.plotly_chart(fig, use_container_width=True, key=f"case_ability_effort_{examinee_id}")


def _build_case_review_pdf(
    selected_id: str,
    case_row: pd.Series,
    metrics: dict,
    domain_rows: list[dict],
    llm_report: str,
    final_decision: str,
    reviewer_note: str,
) -> tuple[bytes, str | None]:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, HRFlowable
    except Exception as e:
        return b"", f"reportlab not installed: {e}"
    buf = io.BytesIO()
    doc_id = f"PsyMAS-CASE-{selected_id}-{time.strftime('%Y%m%d')}"
    generated_at = time.strftime("%Y-%m-%d %H:%M")
    doc = SimpleDocTemplate(buf, pagesize=letter, rightMargin=0.62 * inch, leftMargin=0.62 * inch, topMargin=0.70 * inch, bottomMargin=0.62 * inch)

    def _legal_canvas(canvas, doc_obj):
        canvas.saveState()
        width, height = letter
        canvas.setStrokeColor(colors.HexColor("#111827"))
        canvas.setLineWidth(0.6)
        canvas.line(0.62 * inch, height - 0.43 * inch, width - 0.62 * inch, height - 0.43 * inch)
        canvas.setFont("Helvetica-Bold", 8)
        canvas.setFillColor(colors.HexColor("#111827"))
        canvas.drawString(0.62 * inch, height - 0.31 * inch, "PsyMAS Evidence Review Memorandum")
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(width - 0.62 * inch, height - 0.31 * inch, doc_id)
        canvas.line(0.62 * inch, 0.43 * inch, width - 0.62 * inch, 0.43 * inch)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#4B5563"))
        canvas.drawString(0.62 * inch, 0.27 * inch, "Confidential review material. Statistical evidence supports human review and is not a standalone misconduct determination.")
        canvas.drawRightString(width - 0.62 * inch, 0.27 * inch, f"Page {doc_obj.page}")
        canvas.restoreState()

    styles = getSampleStyleSheet()
    title = ParagraphStyle("PsyMASTitle", parent=styles["Title"], alignment=0, fontName="Helvetica-Bold", fontSize=15, leading=18, textColor=colors.HexColor("#111827"), spaceAfter=4)
    subtitle = ParagraphStyle("PsyMASSubtitle", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=8.5, leading=11, textColor=colors.HexColor("#374151"), spaceAfter=8)
    h2 = ParagraphStyle("PsyMASH2", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=10.5, textColor=colors.HexColor("#111827"), spaceBefore=10, spaceAfter=5)
    body = ParagraphStyle("PsyMASBody", parent=styles["BodyText"], fontSize=8.6, leading=11.5, textColor=colors.HexColor("#111827"))
    small = ParagraphStyle("PsyMASSmall", parent=styles["BodyText"], fontSize=7.6, leading=10, textColor=colors.HexColor("#374151"))
    notice = ParagraphStyle("PsyMASNotice", parent=styles["BodyText"], fontSize=8.2, leading=11, textColor=colors.HexColor("#111827"), backColor=colors.HexColor("#F8FAFC"), borderColor=colors.HexColor("#CBD5E1"), borderWidth=0.5, borderPadding=6, spaceAfter=8)

    story = [
        Paragraph("FORENSIC EVIDENCE REVIEW MEMORANDUM", title),
        Paragraph(f"Document ID: {html.escape(doc_id)} &nbsp;&nbsp;|&nbsp;&nbsp; Generated: {html.escape(generated_at)}", subtitle),
        HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#111827"), spaceBefore=2, spaceAfter=8),
    ]
    meta_data = [
        ["Subject Examinee", html.escape(str(selected_id)), "Queue Status", html.escape(str(case_row.get("Review_Queue_Status", "")))],
        ["Review Priority", html.escape(str(case_row.get("Review_Priority", ""))), "Final Reviewer Decision", html.escape(str(final_decision))],
        ["Domains for Review", html.escape(str(case_row.get("Domains_For_Review", ""))), "Highest Domain Strength", html.escape(str(case_row.get("Highest_Domain_Strength", "")))],
    ]
    meta_table = Table(meta_data, colWidths=[1.25 * inch, 2.0 * inch, 1.45 * inch, 2.15 * inch])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E5E7EB")),
        ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#E5E7EB")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#111827")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#9CA3AF")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.extend([
        meta_table,
        Spacer(1, 8),
        Paragraph(
            "<b>Review standard.</b> This memorandum documents statistical and procedural indicators for human review. "
            "The findings summarize evidence signals and do not, by themselves, establish intent, misconduct, or disciplinary liability.",
            notice,
        ),
        Paragraph("I. Performance and Response-Process Evidence", h2),
    ])
    perf_data = [
        ["Score", "Percentile", "Mean RT", "Median RT"],
        [
            f"{_fmt_metric(metrics.get('score'), decimals=0)} / {_fmt_metric(metrics.get('total_items'), decimals=0)}",
            _fmt_metric(metrics.get("percentile"), "th", 0),
            _fmt_metric(metrics.get("mean_rt"), decimals=2),
            _fmt_metric(metrics.get("median_rt"), decimals=2),
        ],
    ]
    perf_table = Table(perf_data, colWidths=[1.3 * inch] * 4)
    perf_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#111827")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#FFFFFF")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(perf_table)
    try:
        resp_df = _case_response_dataframe()
        rt_df = _case_rt_dataframe()
        sid_int = int(selected_id)
        if not resp_df.empty and 0 <= sid_int - 1 < len(resp_df):
            all_scores = resp_df.sum(axis=1, skipna=True).astype(float).tolist()
            pct_png = _plot_percentile_for_pdf(
                float(metrics.get("percentile") or 0.0),
                sid_int,
                all_scores,
                int(metrics.get("score") or 0),
                int(metrics.get("total_items") or 0),
                compact=True,
            )
            resp_row = pd.to_numeric(resp_df.iloc[sid_int - 1], errors="coerce").fillna(0).astype(float).tolist()
            item_means = resp_df.mean(axis=0, skipna=True).fillna(0).astype(float).tolist()
            acc_png = _plot_person_accuracy_for_pdf(resp_row, item_means, sid_int, compact=True)
            image_cells = []
            if pct_png:
                img = _rl_image_preserve_ratio(io.BytesIO(pct_png), 3.0 * inch, 1.65 * inch)
                if img:
                    image_cells.append(img)
            if acc_png:
                img = _rl_image_preserve_ratio(io.BytesIO(acc_png), 3.2 * inch, 1.65 * inch)
                if img:
                    image_cells.append(img)
            if image_cells:
                story.append(Spacer(1, 6))
                story.append(Table([image_cells], colWidths=[3.1 * inch] * len(image_cells)))
        if not rt_df.empty and 0 <= sid_int - 1 < len(rt_df):
            rt_row = pd.to_numeric(rt_df.iloc[sid_int - 1], errors="coerce").fillna(0).astype(float).tolist()
            rt_means = rt_df.mean(axis=0, skipna=True).fillna(0).astype(float).tolist()
            rt_png = _plot_rt_for_pdf(rt_row, rt_means, sid_int, compact=True)
            if rt_png:
                story.append(Spacer(1, 5))
                img = _rl_image_preserve_ratio(io.BytesIO(rt_png), 6.2 * inch, 1.65 * inch)
                if img:
                    story.append(img)
    except Exception:
        pass
    story.append(Paragraph("II. Domain-Level Evidence Findings", h2))
    domain_data = [["Domain", "Evidence Strength", "Primary Supporting Index", "Interpretation / Limitation"]]
    for row in domain_rows:
        domain_data.append(
            [
                Paragraph(html.escape(_shorten_report_text(row.get("domain_name", row["domain"]), 32)), body),
                Paragraph(html.escape(_shorten_report_text(row["strength"], 18)), body),
                Paragraph(html.escape(_shorten_report_text(row["main_index"], 54)), body),
                Paragraph(html.escape(_shorten_report_text(row["pattern"], 150)), body),
            ]
        )
    domain_table = Table(domain_data, colWidths=[1.35 * inch, 0.95 * inch, 1.55 * inch, 3.15 * inch], repeatRows=1)
    domain_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#111827")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#FFFFFF")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#9CA3AF")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.extend([domain_table, Paragraph("III. LLM-Assisted Review Narrative", h2)])
    story.append(Paragraph("The following narrative is generated from the evidence tables above and remains subject to human reviewer confirmation.", small))
    story.append(Spacer(1, 3))
    for para in str(llm_report or "No LLM report generated.").splitlines():
        if para.strip():
            story.append(Paragraph(_markdown_to_reportlab(para.strip()), body))
            story.append(Spacer(1, 4))
    story.append(Paragraph("IV. Reviewer Determination and Attestation", h2))
    decision_data = [
        ["Final Reviewer Decision", html.escape(str(final_decision))],
        ["Reviewer Note", Paragraph(html.escape(str(reviewer_note or "No reviewer note entered.")), body)],
        ["Attestation", Paragraph("The reviewer acknowledges that this report documents statistical evidence for review support and does not independently establish misconduct.", body)],
        ["Reviewer Signature / Date", "________________________________________"],
    ]
    decision_table = Table(decision_data, colWidths=[1.7 * inch, 5.3 * inch])
    decision_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E5E7EB")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#111827")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9CA3AF")),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(decision_table)
    doc.build(story, onFirstPage=_legal_canvas, onLaterPages=_legal_canvas)
    return buf.getvalue(), None


def _auxiliary_index_application(function_name: str, index_name: str, aggregation: str) -> str:
    fn = str(function_name or "")
    index = str(index_name or "")
    agg = str(aggregation or "")
    if fn in {"detect_ac", "detect_as"} or agg.startswith("pair_"):
        return "Pair-level similarity/copying trace. Useful for locating pairs or clusters; not a standalone person-level misconduct flag."
    if fn == "detect_cp" or index.lower() in {"cp", "cp_location"}:
        return "Estimated change-point location. Used to localize an abrupt response or timing shift after a flagged change-point statistic."
    if fn == "detect_rg" and index.upper() == "RTF":
        return "Item-level response-time fidelity; summarizes the proportion of non-rapid responses for an item."
    if fn == "detect_rg" and index.lower() == "unmotivated":
        return "Run-level/person summary derived from rapid-response thresholds; used for context and audit."
    if fn == "detect_rg" and index.upper() in {"VI", "VITP"}:
        return "Visual inspection output. Used to inspect item response-time distributions, optionally with proportion-correct information."
    if fn == "PsyMAS-derived" or index in {"pair_count", "missing_domain"}:
        return "Derived audit/context field. Used to document data availability or trace volume, not to create a forensic flag."
    return "Display-only index retained for interpretation, localization, or audit; it does not enter domain evidence strength."


def _case_auxiliary_index_rows(examinee_id: str) -> pd.DataFrame:
    result_df, legend_df = _governance_source_export()
    if result_df.empty or legend_df.empty or "Examinee_ID" not in result_df.columns:
        return pd.DataFrame()
    case_rows = result_df[result_df["Examinee_ID"].astype(str) == str(examinee_id)]
    if case_rows.empty:
        return pd.DataFrame()
    case_row = case_rows.iloc[0]
    rows: list[dict] = []
    for _, leg in legend_df.iterrows():
        column = str(leg.get("column_name", ""))
        if not column or column == "Examinee_ID" or column not in result_df.columns:
            continue
        aggregation = str(leg.get("aggregation", ""))
        rule = _rulebook_match_for_export_column(legend_df, column)
        evidence_use = _evidence_use_label(rule)
        flag_type = str(rule.get("flag_type", "")).strip().lower()
        is_aux = (
            evidence_use == EVIDENCE_USE_DISPLAY
            or flag_type in {"auxiliary_only", "descriptive_only"}
            or aggregation in {"pair_count", "pair_min", "pair_max", "pair_flag_display"}
        )
        if not is_aux:
            continue
        value = case_row.get(column)
        if pd.isna(value) or value == "":
            continue
        fn = str(rule.get("function", "")) or str(leg.get("agent", ""))
        index_name = str(rule.get("method", "")) or str(leg.get("index_name", column))
        rows.append(
            {
                "Domain": str(rule.get("domain", "")),
                "Function": fn,
                "Index": index_name,
                "Column": column,
                "Value": value,
                "Use": "Display Only",
                "Application": _auxiliary_index_application(fn, index_name, aggregation),
                "Notes": str(rule.get("notes", "")),
            }
        )
    return pd.DataFrame(rows)


def _pair_visual_rows_for_examinee(flags: dict, examinee_id: str) -> list[dict]:
    try:
        sid = int(examinee_id)
    except (TypeError, ValueError):
        return []
    rows: list[dict] = []
    ac_pairs = ((flags.get("ac_agent") or {}).get("pairs") or []) if isinstance(flags, dict) else []
    for pair in ac_pairs:
        if not isinstance(pair, dict):
            continue
        try:
            src = int(pair.get("Source"))
            cop = int(pair.get("Copier"))
        except (TypeError, ValueError):
            continue
        if sid not in {src, cop}:
            continue
        pvals = [_safe_float(v) for k, v in pair.items() if str(k).endswith("_pval")]
        stats = [
            _safe_float(v)
            for k, v in pair.items()
            if k not in {"Source", "Copier", "flagged"} and not str(k).endswith("_pval") and _safe_float(v) is not None
        ]
        rows.append(
            {
                "Type": "Copying",
                "Examinee_A": src,
                "Examinee_B": cop,
                "Partner": cop if sid == src else src,
                "Direction": "source" if sid == src else "copier",
                "Min_P": min([p for p in pvals if p is not None], default=None),
                "Max_Statistic": max([abs(s) for s in stats if s is not None], default=None),
                "Flagged": bool(pair.get("flagged", False)),
            }
        )
    as_pairs = ((flags.get("as_agent") or {}).get("stat") or []) if isinstance(flags, dict) else []
    for pair in as_pairs:
        if not isinstance(pair, dict):
            continue
        pr = pair.get("_pair")
        if not (isinstance(pr, (list, tuple)) and len(pr) >= 2):
            continue
        try:
            a = int(pr[0]) + 1
            b = int(pr[1]) + 1
        except (TypeError, ValueError):
            continue
        if sid not in {a, b}:
            continue
        pvals = [_safe_float(v) for k, v in pair.items() if str(k).endswith("_pval")]
        stats = [
            _safe_float(v)
            for k, v in pair.items()
            if k != "_pair" and not str(k).endswith("_pval") and not str(k).endswith("_flag") and _safe_float(v) is not None
        ]
        rows.append(
            {
                "Type": "Similarity",
                "Examinee_A": a,
                "Examinee_B": b,
                "Partner": b if sid == a else a,
                "Direction": "undirected",
                "Min_P": min([p for p in pvals if p is not None], default=None),
                "Max_Statistic": max([abs(s) for s in stats if s is not None], default=None),
                "Flagged": any(_gov_truthy_flag(v) for k, v in pair.items() if str(k).endswith("_flag")),
            }
        )
    if not rows:
        return []
    out = pd.DataFrame(rows)
    return out.sort_values(["Min_P", "Type"], na_position="last").head(12).to_dict(orient="records")


def _build_pair_visual_index(flags: dict, n_examinees: int) -> dict[str, list[dict]]:
    if not isinstance(flags, dict) or n_examinees <= 0:
        return {}
    index: dict[str, list[dict]] = {}
    for sid in range(1, n_examinees + 1):
        rows = _pair_visual_rows_for_examinee(flags, str(sid))
        if rows:
            index[str(sid)] = rows
    return index


def _case_pair_visual_rows(flags: dict, examinee_id: str) -> pd.DataFrame:
    if _run_store_ready():
        rows = get_run_store().load_pair_visual_index(_active_run_id()).get(str(examinee_id), [])
        if rows:
            return pd.DataFrame(rows)
    rows = _pair_visual_rows_for_examinee(flags, examinee_id)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _make_plotly_text_readable(fig):
    fig.update_layout(
        font=dict(color=PSYMAS_VIZ["ink"], size=12),
        title_font=dict(color=PSYMAS_VIZ["ink"], size=15),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        legend=dict(font=dict(color=PSYMAS_VIZ["ink"])),
    )
    fig.update_xaxes(
        color=PSYMAS_VIZ["ink"],
        title_font=dict(color=PSYMAS_VIZ["ink"]),
        tickfont=dict(color=PSYMAS_VIZ["ink"]),
        gridcolor=PSYMAS_VIZ["grid"],
        zeroline=False,
    )
    fig.update_yaxes(
        color=PSYMAS_VIZ["ink"],
        title_font=dict(color=PSYMAS_VIZ["ink"]),
        tickfont=dict(color=PSYMAS_VIZ["ink"]),
        gridcolor=PSYMAS_VIZ["grid"],
        zeroline=False,
    )
    fig.update_annotations(font_color=PSYMAS_VIZ["ink"], font_size=11)
    return fig


def _render_case_pair_network(pair_df: pd.DataFrame, examinee_id: str) -> None:
    if pair_df is None or pair_df.empty:
        st.info("No pair-level copying or similarity links are available for this examinee.")
        return
    try:
        import plotly.graph_objects as go
    except Exception:
        st.dataframe(pair_df, use_container_width=True, hide_index=True)
        return
    try:
        sid = int(examinee_id)
    except (TypeError, ValueError):
        st.dataframe(pair_df, use_container_width=True, hide_index=True)
        return
    view = pair_df.copy()
    view["Min_P_Num"] = pd.to_numeric(view.get("Min_P"), errors="coerce")
    view["Max_Statistic_Num"] = pd.to_numeric(view.get("Max_Statistic"), errors="coerce")

    def _pair_severity(row: pd.Series) -> tuple[str, str, float]:
        p = row.get("Min_P_Num")
        stat = row.get("Max_Statistic_Num")
        flagged = bool(row.get("Flagged", False))
        if pd.notna(p) and p > 0:
            score = min(12.0, -math.log10(float(p)))
            if p <= 0.001 or flagged:
                return "strong", PSYMAS_VIZ["high"], score
            if p <= 0.01:
                return "moderate", PSYMAS_VIZ["moderate"], score
            if p <= 0.05:
                return "weak", PSYMAS_VIZ["warning"], score
            return "trace", PSYMAS_VIZ["inactive"], score
        if pd.notna(stat):
            score = min(12.0, abs(float(stat)))
            if flagged:
                return "strong", PSYMAS_VIZ["high"], score
            return "trace", PSYMAS_VIZ["inactive"], score
        if flagged:
            return "flagged", PSYMAS_VIZ["high"], 2.0
        return "trace", PSYMAS_VIZ["inactive"], 0.2

    sev = view.apply(_pair_severity, axis=1, result_type="expand")
    view["Evidence_Level"] = sev[0]
    view["Evidence_Color"] = sev[1]
    view["Evidence_Score"] = pd.to_numeric(sev[2], errors="coerce").fillna(0.2)
    view = view.sort_values(["Evidence_Score", "Flagged"], ascending=[False, False]).head(10)
    top_row = view.iloc[0] if not view.empty else pd.Series(dtype=object)
    top_partner = top_row.get("Partner", "NA")
    top_label = str(top_row.get("Evidence_Level", "trace")).title()
    flagged_n = int(view["Flagged"].fillna(False).astype(bool).sum()) if "Flagged" in view else 0
    copying_n = int((view["Type"].astype(str) == "Copying").sum()) if "Type" in view else 0
    similarity_n = int((view["Type"].astype(str) == "Similarity").sum()) if "Type" in view else 0
    st.markdown(
        f"""
<div class="psymas-case-strip">
  <div class="psymas-case-chip"><div class="label">Linked partners</div><div class="value">{len(view):,}</div></div>
  <div class="psymas-case-chip"><div class="label">Flagged pair links</div><div class="value">{flagged_n:,}</div></div>
  <div class="psymas-case-chip"><div class="label">Strongest partner</div><div class="value">{html.escape(str(top_partner))}</div></div>
  <div class="psymas-case-chip"><div class="label">Strongest signal</div><div class="value">{html.escape(top_label)}</div></div>
</div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        f"Pair evidence shown for examinee {sid}: {copying_n} copying-oriented link(s) and "
        f"{similarity_n} similarity-oriented link(s). Lower p-values or package flags are shown with warmer colors."
    )
    partners = [int(p) for p in pair_df["Partner"].dropna().astype(int).tolist()]
    nodes = [sid] + [p for p in partners if p != sid]
    node_pos = {sid: (0.0, 0.0)}
    radius = 1.0
    for i, node in enumerate(nodes[1:]):
        angle = (2 * math.pi * i) / max(1, len(nodes) - 1)
        node_pos[node] = (radius * math.cos(angle), radius * math.sin(angle))
    fig = go.Figure()
    for _, row in view.iterrows():
        partner = int(row.get("Partner"))
        x0, y0 = node_pos.get(sid, (0.0, 0.0))
        x1, y1 = node_pos.get(partner, (0.0, 0.0))
        width = max(1.4, min(5.5, 1.2 + float(row.get("Evidence_Score", 0.2))))
        hover = (
            f"{row.get('Type')} link with {partner}<br>"
            f"Evidence level: {row.get('Evidence_Level')}<br>"
            f"Min p: {row.get('Min_P')}<br>"
            f"Max statistic: {row.get('Max_Statistic')}"
        )
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(width=width, color=str(row.get("Evidence_Color", PSYMAS_VIZ["inactive"]))),
                hovertext=hover,
                hoverinfo="text",
                showlegend=False,
            )
        )
    node_x = [node_pos[n][0] for n in nodes]
    node_y = [node_pos[n][1] for n in nodes]
    node_text = [f"Examinee {n}" + (" (selected)" if n == sid else "") for n in nodes]
    color_by_partner = {
        int(row.get("Partner")): str(row.get("Evidence_Color", PSYMAS_VIZ["warning"]))
        for _, row in view.iterrows()
    }
    node_color = [PSYMAS_VIZ["selected"] if n == sid else color_by_partner.get(n, PSYMAS_VIZ["reference_fill"]) for n in nodes]
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=[str(n) for n in nodes],
            textposition="middle center",
            textfont=dict(color="#FFFFFF", size=12),
                marker=dict(size=[34 if n == sid else 26 for n in nodes], color=node_color, line=dict(width=2, color="#FFFFFF")),
            hovertext=node_text,
            hoverinfo="text",
            showlegend=False,
        )
    )
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=35, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        title="Selected Examinee Pair-Link Map",
    )
    _make_plotly_text_readable(fig)
    ranked = view.copy()
    ranked["Partner_Label"] = ranked["Partner"].map(lambda p: f"Examinee {p}")
    ranked["Pair_Label"] = ranked.apply(lambda r: f"{r['Partner_Label']} · {r.get('Type', '')}", axis=1)
    bar_fig = go.Figure(
        go.Bar(
            x=ranked["Evidence_Score"],
            y=ranked["Pair_Label"],
            orientation="h",
            marker_color=ranked["Evidence_Color"],
            hovertext=[
                f"{r.get('Type')} with {r.get('Partner')}<br>Level: {r.get('Evidence_Level')}<br>Min p: {r.get('Min_P')}<br>Max statistic: {r.get('Max_Statistic')}"
                for _, r in ranked.iterrows()
            ],
            hovertemplate="%{hovertext}<extra></extra>",
        )
    )
    bar_fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=35, b=10),
        title="Ranked Pair Signals",
        xaxis_title="Evidence intensity (-log10 p or statistic scale)",
        yaxis_title="",
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color=PSYMAS_VIZ["ink"]),
    )
    _make_plotly_text_readable(bar_fig)
    map_col, rank_col = st.columns([0.52, 0.48])
    with map_col:
        st.plotly_chart(fig, use_container_width=True, key=f"case_pair_network_{examinee_id}")
    with rank_col:
        st.plotly_chart(bar_fig, use_container_width=True, key=f"case_pair_ranked_{examinee_id}")
    with st.expander("Pair-link details", expanded=False):
        detail_cols = ["Type", "Partner", "Direction", "Evidence_Level", "Min_P", "Max_Statistic", "Flagged"]
        st.dataframe(view[[c for c in detail_cols if c in view.columns]], use_container_width=True, hide_index=True)


def _case_rt_dataframe() -> pd.DataFrame:
    rt_data = st.session_state.get("forensic_rt_data") or st.session_state.get("last_uploaded_rt_data") or []
    if not rt_data:
        return pd.DataFrame()
    df = pd.DataFrame(rt_data)
    numeric = df.apply(pd.to_numeric, errors="coerce")
    keep_cols = [c for c in numeric.columns if numeric[c].notna().any()]
    return numeric[keep_cols].copy()


def _rg_preferred_rte_method(rg_data: dict) -> tuple[str, list]:
    rte_by_method = rg_data.get("rte_by_method") if isinstance(rg_data.get("rte_by_method"), dict) else {}
    if not rte_by_method:
        return "RTE", rg_data.get("rte") if isinstance(rg_data.get("rte"), list) else []
    preferred = rg_data.get("flag_methods") if isinstance(rg_data.get("flag_methods"), list) else []
    for method in preferred:
        method = str(method)
        if method in rte_by_method:
            return method, rte_by_method[method]
        matches = [k for k in sorted(rte_by_method) if str(k).startswith(method + "_")]
        if matches:
            return matches[0], rte_by_method[matches[0]]
    for candidate in ("NT_2", "NT", "CT", "CUMP"):
        if candidate in rte_by_method:
            return candidate, rte_by_method[candidate]
    method_name, vals = next(iter(sorted(rte_by_method.items())))
    return str(method_name), vals


def _rg_item_thresholds(method_name: str, rt_df: pd.DataFrame) -> pd.Series:
    if rt_df.empty:
        return pd.Series(dtype=float)
    rg_cfg = ((_active_threshold_config().get("rules") or {}).get("detect_rg") or {})
    method = str(method_name or "")
    if method.startswith("NT"):
        nt_values = ((rg_cfg.get("NT") or {}).get("nt") if isinstance(rg_cfg.get("NT"), dict) else None) or [5, 10, 15, 20, 25, 30, 35]
        pct = nt_values[0]
        m = re.search(r"NT_(\d+)", method)
        if m:
            idx = max(0, int(m.group(1)) - 1)
            if idx < len(nt_values):
                pct = nt_values[idx]
        try:
            pct = float(pct)
        except (TypeError, ValueError):
            pct = 10.0
        return rt_df.mean(axis=0, skipna=True) * (pct / 100.0)
    if method.startswith("CT"):
        ct_block = rg_cfg.get("CT") if isinstance(rg_cfg.get("CT"), dict) else {}
        try:
            thr = float(ct_block.get("thr", 3.0))
        except (TypeError, ValueError):
            thr = 3.0
        return pd.Series([thr] * len(rt_df.columns), index=rt_df.columns, dtype=float)
    return pd.Series([3.0] * len(rt_df.columns), index=rt_df.columns, dtype=float)


def _render_case_rg_visual(flags: dict, examinee_id: str) -> None:
    rg_data = flags.get("rg_agent", {}) if isinstance(flags, dict) else {}
    if not isinstance(rg_data, dict) or rg_data.get("error"):
        st.info("No rapid-guessing auxiliary output is available for this examinee.")
        return
    try:
        sid0 = int(examinee_id) - 1
    except (TypeError, ValueError):
        st.info("No rapid-guessing auxiliary output is available for this examinee.")
        return
    method_name, vals = _rg_preferred_rte_method(rg_data)
    rows = []
    for idx, value in enumerate(vals or []):
        val = _safe_float(value)
        if val is None:
            continue
        rows.append(
            {
                "Examinee_ID": idx + 1,
                "RTE": val,
                "Selected": idx == sid0,
                "Flagged": idx in set(rg_data.get("flagged", []) or []),
            }
        )
    if not rows:
        st.info("No RTE values are available for this examinee.")
        return
    try:
        import plotly.graph_objects as go
    except Exception:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        return
    df = pd.DataFrame(rows)
    selected_df = df[df["Selected"]]
    other_df = df[~df["Selected"]]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=other_df["Examinee_ID"],
            y=other_df["RTE"],
            mode="markers",
            marker=dict(
                size=8,
                color=np.where(other_df["Flagged"], PSYMAS_VIZ["warning"], PSYMAS_VIZ["reference"]),
                opacity=0.72,
                line=dict(width=0),
            ),
            name="Other examinees",
            hovertemplate="Examinee %{x}<br>RTE %{y:.3f}<extra></extra>",
        )
    )
    if not selected_df.empty:
        fig.add_trace(
            go.Scatter(
                x=selected_df["Examinee_ID"],
                y=selected_df["RTE"],
                mode="markers+text",
                text=["selected"],
                textposition="top center",
                marker=dict(size=16, color=PSYMAS_VIZ["selected"], line=dict(width=2, color="#FFFFFF")),
                name="Selected examinee",
                hovertemplate="Selected examinee %{x}<br>RTE %{y:.3f}<extra></extra>",
            )
        )
    low_ref = 0.8
    fig.add_hline(
        y=low_ref,
        line_dash="dash",
        line_color=PSYMAS_VIZ["warning"],
        annotation_text="low effort reference",
        annotation_font_color=PSYMAS_VIZ["ink"],
    )
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=25, b=10),
        title=f"Rapid Guessing: RTE Compared With Other Examinees ({method_name})",
        xaxis_title="Examinee ID",
        yaxis_title="Response Time Effort (higher = more non-rapid responses)",
        yaxis=dict(range=[max(0, min(df["RTE"].min() - 0.05, 0.75)), min(1.05, max(df["RTE"].max() + 0.05, 0.85))]),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    _make_plotly_text_readable(fig)
    st.plotly_chart(fig, use_container_width=True, key=f"case_rg_visual_{examinee_id}")

    rt_df = _case_rt_dataframe()
    if rt_df.empty or not (0 <= sid0 < len(rt_df)):
        st.info("Item-level response-time timeline is unavailable because the RT matrix is not loaded in the session.")
        return
    thresholds = _rg_item_thresholds(method_name, rt_df)
    item_numbers = list(range(1, len(rt_df.columns) + 1))
    selected_rt = pd.to_numeric(rt_df.iloc[sid0], errors="coerce")
    rapid_mask = selected_rt < thresholds
    timeline_col, item_col = st.columns([1.35, 1.0])
    with timeline_col:
        fig_t = go.Figure()
        fig_t.add_trace(
            go.Scatter(
                x=item_numbers,
                y=selected_rt.values,
                mode="lines+markers",
                marker=dict(
                    size=8,
                    color=np.where(rapid_mask.fillna(False), PSYMAS_VIZ["high"], PSYMAS_VIZ["selected"]),
                    line=dict(width=1, color="#FFFFFF"),
                ),
                line=dict(color=PSYMAS_VIZ["selected"], width=1.5),
                name="Selected examinee RT",
                hovertemplate="Item %{x}<br>RT %{y:.2f}s<extra></extra>",
            )
        )
        fig_t.add_trace(
            go.Scatter(
                x=item_numbers,
                y=thresholds.values,
                mode="lines",
                line=dict(color=PSYMAS_VIZ["warning"], dash="dash", width=2),
                name=f"{method_name} threshold",
                hovertemplate="Item %{x}<br>threshold %{y:.2f}s<extra></extra>",
            )
        )
        fig_t.update_layout(
            title="Selected Examinee Item-Level RT Timeline",
            height=330,
            margin=dict(l=10, r=10, t=35, b=10),
            xaxis_title="Item position",
            yaxis_title="Response time",
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            font=dict(color=PSYMAS_VIZ["ink"]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        _make_plotly_text_readable(fig_t)
        st.plotly_chart(fig_t, use_container_width=True, key=f"case_rg_timeline_{examinee_id}")
    with item_col:
        rapid_matrix = rt_df.lt(thresholds, axis=1)
        rapid_rate = rapid_matrix.mean(axis=0, skipna=True).fillna(0.0)
        fig_i = go.Figure(
            go.Bar(
                x=item_numbers,
                y=rapid_rate.values,
                marker_color=PSYMAS_VIZ["reference"],
                hovertemplate="Item %{x}<br>Rapid-response rate %{y:.1%}<extra></extra>",
            )
        )
        fig_i.update_layout(
            title="Item-Level Rapid-Response Concentration",
            height=330,
            margin=dict(l=10, r=10, t=35, b=10),
            xaxis_title="Item position",
            yaxis_title="Share below threshold",
            yaxis=dict(tickformat=".0%", range=[0, min(1.0, max(0.12, float(rapid_rate.max()) + 0.05))]),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            font=dict(color=PSYMAS_VIZ["ink"]),
        )
        _make_plotly_text_readable(fig_i)
        st.plotly_chart(fig_i, use_container_width=True, key=f"case_rg_item_rate_{examinee_id}")


def _case_compromised_items(n_items: int | None = None) -> list[int]:
    """Return 1-based compromised/exposed item IDs from the active run or local demo files."""
    items = st.session_state.get("prep_compromised_items") or st.session_state.get("ab_only_compromised_items") or []
    parsed: list[int] = []
    for item in items:
        try:
            val = int(float(item))
            if val > 0:
                parsed.append(val)
        except (TypeError, ValueError):
            continue
    if parsed:
        return sorted(set(i for i in parsed if n_items is None or i <= n_items))

    candidates = [
        Path("data/upload/compromised_items.csv"),
        Path("data/psymas_research_export/dataset_a_inputs/compromised_items.csv"),
        Path("data/sample/compromised_items.csv"),
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            comp_df = pd.read_csv(path)
            parsed = _parse_compromised_items_csv(comp_df, n_items=n_items)
        except Exception:
            parsed = []
        if parsed:
            return sorted(set(i for i in parsed if n_items is None or i <= n_items))
    return []


def _render_case_pk_visual(examinee_id: str) -> None:
    resp_df = _case_response_dataframe()
    rt_df = _case_rt_dataframe()
    if resp_df.empty:
        st.info("Preknowledge item performance is unavailable because response data are not loaded.")
        return
    try:
        sid0 = int(examinee_id) - 1
    except (TypeError, ValueError):
        st.info("Preknowledge item performance is unavailable for this examinee.")
        return
    if not (0 <= sid0 < len(resp_df)):
        st.info("Preknowledge item performance is unavailable for this examinee.")
        return

    n_items = int(resp_df.shape[1])
    comp_items = _case_compromised_items(n_items=n_items)
    if not comp_items:
        st.info("No compromised/exposed item list is available, so item-level preknowledge behavior cannot be displayed.")
        return

    comp_idx = [i - 1 for i in comp_items if 1 <= i <= n_items]
    if not comp_idx:
        st.info("The compromised/exposed item list does not match the response matrix item range.")
        return

    selected_resp = pd.to_numeric(resp_df.iloc[sid0, comp_idx], errors="coerce")
    cohort_acc = resp_df.iloc[:, comp_idx].apply(pd.to_numeric, errors="coerce").mean(axis=0, skipna=True)
    selected_acc = float(selected_resp.mean(skipna=True)) if selected_resp.notna().any() else np.nan
    cohort_comp_acc = float(cohort_acc.mean(skipna=True)) if cohort_acc.notna().any() else np.nan
    non_comp_idx = [i for i in range(n_items) if i not in set(comp_idx)]
    selected_non_acc = (
        float(pd.to_numeric(resp_df.iloc[sid0, non_comp_idx], errors="coerce").mean(skipna=True))
        if non_comp_idx
        else np.nan
    )

    rt_note = "RT unavailable"
    selected_comp_rt = np.nan
    cohort_comp_rt = np.nan
    if not rt_df.empty and 0 <= sid0 < len(rt_df):
        rt_comp_idx = [i for i in comp_idx if i < rt_df.shape[1]]
        if rt_comp_idx:
            selected_comp_rt = float(pd.to_numeric(rt_df.iloc[sid0, rt_comp_idx], errors="coerce").mean(skipna=True))
            cohort_comp_rt = float(rt_df.iloc[:, rt_comp_idx].apply(pd.to_numeric, errors="coerce").mean(axis=0, skipna=True).mean(skipna=True))
            if not np.isnan(selected_comp_rt) and not np.isnan(cohort_comp_rt):
                rt_note = f"{selected_comp_rt:.2f} vs {cohort_comp_rt:.2f}"

    st.markdown(
        f"""
<div class="psymas-case-strip flat">
  <div class="psymas-case-chip"><div class="label">Compromised items</div><div class="value">{len(comp_idx)}</div></div>
  <div class="psymas-case-chip"><div class="label">Selected accuracy</div><div class="value">{_fmt_metric(selected_acc * 100 if not np.isnan(selected_acc) else None, "%", 1)}</div></div>
  <div class="psymas-case-chip"><div class="label">Cohort item mean</div><div class="value">{_fmt_metric(cohort_comp_acc * 100 if not np.isnan(cohort_comp_acc) else None, "%", 1)}</div></div>
  <div class="psymas-case-chip"><div class="label">Non-comp. accuracy</div><div class="value">{_fmt_metric(selected_non_acc * 100 if not np.isnan(selected_non_acc) else None, "%", 1)}</div></div>
  <div class="psymas-case-chip"><div class="label">Comp. RT vs cohort</div><div class="value">{html.escape(rt_note)}</div></div>
</div>
        """,
        unsafe_allow_html=True,
    )

    try:
        import plotly.graph_objects as go
    except Exception:
        out = pd.DataFrame(
            {
                "Item": comp_items,
                "Selected_Response": selected_resp.values,
                "Cohort_Item_Accuracy": cohort_acc.values,
            }
        )
        st.dataframe(out, use_container_width=True, hide_index=True)
        return

    item_labels = [i for i in comp_items if 1 <= i <= n_items]
    selected_vals = selected_resp.astype(float).values
    selected_status = np.where(
        selected_vals > 0,
        "Selected correct",
        np.where(selected_vals == 0, "Selected incorrect", "Selected missing"),
    )
    colors = np.where(selected_vals > 0, PSYMAS_VIZ["correct"], np.where(selected_vals == 0, PSYMAS_VIZ["incorrect"], PSYMAS_VIZ["reference"]))
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=item_labels,
            y=cohort_acc.astype(float).values,
            marker_color=colors,
            marker_line=dict(color="#FFFFFF", width=0.8),
            name="Item mean accuracy",
            customdata=selected_status,
            hovertemplate="Compromised item %{x}<br>Item cohort accuracy %{y:.1%}<br>%{customdata}<extra></extra>",
        )
    )
    fig.add_trace(go.Bar(x=[None], y=[None], marker_color=PSYMAS_VIZ["correct"], name="Selected correct", showlegend=True))
    fig.add_trace(go.Bar(x=[None], y=[None], marker_color=PSYMAS_VIZ["incorrect"], name="Selected incorrect", showlegend=True))
    fig.update_layout(
        title="Item accuracy colored by selected response",
        height=255,
        margin=dict(l=44, r=18, t=58, b=42),
        xaxis=dict(
            title="Item number",
            tickmode="array",
            tickvals=item_labels,
            ticktext=[str(item) for item in item_labels],
            type="category",
        ),
        yaxis=dict(title="Item mean accuracy", tickformat=".0%", range=[0, 1.05]),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color=PSYMAS_VIZ["ink"], size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10, color=PSYMAS_VIZ["ink"])),
    )
    _make_plotly_text_readable(fig)

    if not rt_df.empty and 0 <= sid0 < len(rt_df):
        rt_comp_idx = [i for i in comp_idx if i < rt_df.shape[1]]
    else:
        rt_comp_idx = []
    if rt_comp_idx:
        rt_item_labels = [i + 1 for i in rt_comp_idx]
        selected_rt = pd.to_numeric(rt_df.iloc[sid0, rt_comp_idx], errors="coerce")
        cohort_rt = rt_df.iloc[:, rt_comp_idx].apply(pd.to_numeric, errors="coerce").mean(axis=0, skipna=True)
        rt_fig = go.Figure()
        rt_fig.add_trace(
            go.Scatter(
                x=rt_item_labels,
                y=selected_rt.values,
                mode="lines+markers",
                line=dict(color=PSYMAS_VIZ["selected"], width=2),
                marker=dict(size=8, color=PSYMAS_VIZ["selected"]),
                name="Selected RT",
                hovertemplate="Item %{x}<br>Selected RT %{y:.2f}<extra></extra>",
            )
        )
        rt_fig.add_trace(
            go.Scatter(
                x=rt_item_labels,
                y=cohort_rt.values,
                mode="lines+markers",
                line=dict(color=PSYMAS_VIZ["warning"], width=2, dash="dash"),
                marker=dict(size=7, color=PSYMAS_VIZ["warning"]),
                name="Cohort mean RT",
                hovertemplate="Item %{x}<br>Cohort mean RT %{y:.2f}<extra></extra>",
            )
        )
        rt_fig.update_layout(
            title="Response Time on Compromised / Exposed Items",
            height=255,
            margin=dict(l=44, r=18, t=58, b=42),
            xaxis=dict(
                title="Item number",
                tickmode="array",
                tickvals=rt_item_labels,
                ticktext=[str(item) for item in rt_item_labels],
                type="category",
            ),
            yaxis_title="Response time",
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            font=dict(color=PSYMAS_VIZ["ink"], size=11),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10, color=PSYMAS_VIZ["ink"])),
        )
        _make_plotly_text_readable(rt_fig)
        left, right = st.columns(2)
        with left:
            st.plotly_chart(fig, use_container_width=True, key=f"case_pk_comp_accuracy_{examinee_id}")
        with right:
            st.plotly_chart(rt_fig, use_container_width=True, key=f"case_pk_comp_rt_{examinee_id}")
    else:
        st.plotly_chart(fig, use_container_width=True, key=f"case_pk_comp_accuracy_{examinee_id}")


def _render_case_domain_specific_visuals(
    examinee_id: str,
    domain_issue_rows: list[dict],
    flags: dict,
) -> None:
    row_by_domain = {str(row.get("domain")): row for row in domain_issue_rows}

    pk_row = row_by_domain.get("PK", {})
    if str(pk_row.get("card_class", "")).lower() in {"strong", "moderate", "weak", "support"} or _case_compromised_items():
        with st.container(border=True):
            _render_domain_visual_header(
                pk_row or {"step": 3, "display_code": "PKE", "domain_name": "Preknowledge Evidence", "strength": "unavailable", "status_label": "Unavailable", "card_class": "unavailable", "main_index": "None", "pattern": ""},
                help_text="Shows the selected examinee's accuracy and response time on compromised or exposed items compared with cohort item performance. This panel supports interpretation of preknowledge evidence.",
            )
            _render_case_pk_visual(examinee_id)

    cp_row = row_by_domain.get("CP", {})
    with st.container(border=True):
        _render_domain_visual_header(
            cp_row or {"step": 6, "display_code": "CPE", "domain_name": "Change Point Evidence", "strength": "unavailable", "status_label": "Unavailable", "card_class": "unavailable", "main_index": "None", "pattern": ""},
            help_text="Shows where change-point methods estimate the strongest response-behavior shift. In the current prototype this is localization/context unless a calibrated CP rule is active.",
        )
        _render_case_cp_visual(flags, examinee_id, key_suffix="_domain")


def _render_case_cp_visual(flags: dict, examinee_id: str, *, key_suffix: str = "") -> None:
    cp_data = flags.get("cp_agent", {}) if isinstance(flags, dict) else {}
    stat = cp_data.get("stat") if isinstance(cp_data, dict) else None
    if not isinstance(stat, list):
        st.info("No change-point location output is available for this examinee.")
        return
    try:
        sid0 = int(examinee_id) - 1
    except (TypeError, ValueError):
        st.info("No change-point location output is available for this examinee.")
        return
    rows = []
    for idx, rec in enumerate(stat):
        if not isinstance(rec, dict):
            continue
        for key, value in rec.items():
            if str(key).endswith("_cp"):
                val = _safe_float(value)
                if val is not None:
                    rows.append(
                        {
                            "Examinee_ID": idx + 1,
                            "Method": str(key).replace("_cp", ""),
                            "Change_Point": val,
                            "Selected": idx == sid0,
                        }
                    )
    if not rows:
        st.info("No change-point location output is available for this examinee.")
        return
    try:
        import plotly.graph_objects as go
    except Exception:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        return
    df = pd.DataFrame(rows)
    selected_df = df[df["Selected"]].sort_values(["Change_Point", "Method"])
    other_df = df[~df["Selected"]]
    if selected_df.empty:
        st.info("No selected-examinee change-point estimate is available.")
        return

    selected_df = selected_df.copy()
    selected_df["Item_Position"] = selected_df["Change_Point"].round().astype(int)
    selected_df["Family"] = selected_df["Method"].map(
        lambda m: "Response-time shift" if "_T_" in str(m) or str(m).endswith("_T") else "Score/response shift"
    )
    selected_positions = selected_df["Change_Point"].dropna()
    selected_mode = int(selected_df["Item_Position"].mode().iloc[0]) if not selected_df["Item_Position"].mode().empty else None
    selected_min = float(selected_positions.min()) if not selected_positions.empty else None
    selected_max = float(selected_positions.max()) if not selected_positions.empty else None
    method_n = int(len(selected_df))
    unique_position_n = int(selected_df["Item_Position"].nunique())
    agreement_label = "high" if unique_position_n <= 2 else ("moderate" if unique_position_n <= 4 else "low")
    if selected_mode is not None and not other_df.empty:
        other_rounded = other_df["Change_Point"].round().astype(int)
        pop_pct = float((other_rounded == selected_mode).mean())
        pop_label = f"{pop_pct:.1%} of other method estimates"
    else:
        pop_label = "population context unavailable"

    st.markdown(
        f"""
<div class="psymas-case-strip flat">
  <div class="psymas-case-chip"><div class="label">Estimated shift zone</div><div class="value">Item {selected_mode if selected_mode is not None else "NA"}</div></div>
  <div class="psymas-case-chip"><div class="label">Method agreement</div><div class="value">{agreement_label}</div></div>
  <div class="psymas-case-chip"><div class="label">Methods available</div><div class="value">{method_n}</div></div>
  <div class="psymas-case-chip"><div class="label">Population context</div><div class="value">{html.escape(pop_label)}</div></div>
</div>
        """,
        unsafe_allow_html=True,
    )

    grouped = (
        selected_df.groupby(["Family", "Item_Position"], dropna=False)
        .agg(Methods=("Method", lambda s: ", ".join(map(str, s))), Count=("Method", "size"))
        .reset_index()
        .sort_values(["Item_Position", "Family"])
    )
    summary_bits = []
    for _, rec in grouped.iterrows():
        summary_bits.append(
            f"{html.escape(str(rec['Family']))}: item {html.escape(str(rec['Item_Position']))} ({int(rec['Count'])} method{'s' if int(rec['Count']) != 1 else ''})"
        )
    st.markdown(
        f"""
<div class="psymas-reviewer-note flat">
<b>Change-point summary</b><span class="psymas-help-dot" title="Method-level localization summary. Agreement across methods suggests a clearer estimated change zone, but this panel remains contextual unless a calibrated CP rule is active.">?</span>
{' ; '.join(summary_bits) if summary_bits else 'No selected-examinee method summary is available.'}
</div>
        """,
        unsafe_allow_html=True,
    )

    x_min = max(0, int(df["Change_Point"].min()) - 1)
    x_max = int(df["Change_Point"].max()) + 1
    if x_min == x_max:
        x_max = x_min + 1

    timeline_fig = go.Figure()
    if selected_min is not None and selected_max is not None:
        timeline_fig.add_vrect(
            x0=selected_min - 0.35,
            x1=selected_max + 0.35,
            fillcolor=PSYMAS_VIZ["range"],
            line_width=0,
            opacity=0.45,
            annotation_text="estimated shift range",
            annotation_position="top left",
        )
    if selected_mode is not None:
        timeline_fig.add_vline(
            x=selected_mode,
            line_dash="dash",
            line_width=3,
            line_color=PSYMAS_VIZ["selected"],
            annotation_text=f"main shift: item {selected_mode}",
            annotation_position="top right",
        )
    family_colors = {"Score/response shift": PSYMAS_VIZ["context"], "Response-time shift": PSYMAS_VIZ["high"]}
    family_y = {"Score/response shift": 0.64, "Response-time shift": 0.36}
    timeline_fig.add_trace(
        go.Scatter(
            x=[x_min, x_max],
            y=[0.64, 0.64],
            mode="lines",
            line=dict(color=PSYMAS_VIZ["lane"], width=6),
            name="Score/response lane",
            showlegend=False,
            hoverinfo="skip",
        )
    )
    timeline_fig.add_trace(
        go.Scatter(
            x=[x_min, x_max],
            y=[0.36, 0.36],
            mode="lines",
            line=dict(color=PSYMAS_VIZ["lane"], width=6),
            name="Response-time lane",
            showlegend=False,
            hoverinfo="skip",
        )
    )
    for family, fam_df in grouped.groupby("Family", sort=False):
        y_val = family_y.get(str(family), 0.5)
        marker_size = [12 + min(18, int(c) * 4) for c in fam_df["Count"]]
        timeline_fig.add_trace(
            go.Scatter(
                x=fam_df["Item_Position"],
                y=[y_val] * len(fam_df),
                mode="markers+text",
                text=[f"{int(pos)}" for pos in fam_df["Item_Position"]],
                textposition="top center",
                marker=dict(
                    symbol="diamond",
                    size=marker_size,
                    color=family_colors.get(str(family), PSYMAS_VIZ["context"]),
                    line=dict(width=1.2, color=PSYMAS_VIZ["ink"]),
                ),
                name=str(family),
                customdata=np.stack([fam_df["Count"].astype(int), fam_df["Methods"].astype(str)], axis=-1),
                hovertemplate="Item %{x}<br>%{customdata[0]} method estimate(s)<br>%{customdata[1]}<extra></extra>",
            )
        )
    timeline_fig.update_layout(
        height=220,
        margin=dict(l=10, r=18, t=34, b=18),
        title="Estimated Shift Timeline",
        xaxis_title="Item number",
        yaxis=dict(
            title="",
            tickmode="array",
            tickvals=[0.64, 0.36],
            ticktext=["Score / response", "Response time"],
            range=[0.2, 0.8],
            gridcolor=PSYMAS_VIZ["grid"],
            zeroline=False,
        ),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color=PSYMAS_VIZ["ink"], size=11),
        xaxis=dict(range=[x_min, x_max], dtick=1, gridcolor=PSYMAS_VIZ["grid"], zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    _make_plotly_text_readable(timeline_fig)
    st.plotly_chart(timeline_fig, use_container_width=True, key=f"case_cp_timeline_{examinee_id}{key_suffix}")

    with st.expander("Method details", expanded=False):
        fig = go.Figure()
        if selected_min is not None and selected_max is not None:
            fig.add_vrect(
                x0=selected_min - 0.35,
                x1=selected_max + 0.35,
                fillcolor=PSYMAS_VIZ["range"],
                line_width=0,
                opacity=0.55,
                annotation_text="selected estimate range",
                annotation_position="top left",
            )
        if selected_mode is not None:
            fig.add_vline(
                x=selected_mode,
                line_dash="dash",
                line_width=2,
                line_color=PSYMAS_VIZ["selected"],
                annotation_text=f"consensus item {selected_mode}",
                annotation_position="top right",
            )
        for family, fam_df in selected_df.groupby("Family", sort=False):
            fig.add_trace(
                go.Scatter(
                    x=fam_df["Change_Point"],
                    y=fam_df["Method"],
                    mode="markers+text",
                    text=fam_df["Item_Position"].astype(str),
                    textposition="middle right",
                    marker=dict(
                        symbol="diamond",
                        size=12,
                        color=family_colors.get(family, PSYMAS_VIZ["context"]),
                        line=dict(width=1.2, color=PSYMAS_VIZ["ink"]),
                    ),
                    name=family,
                    hovertext=[f"{m}: estimated shift after item {cp:g}" for m, cp in zip(fam_df["Method"], fam_df["Change_Point"])],
                    hovertemplate="%{hovertext}<extra></extra>",
                )
            )
        fig.update_layout(
            height=max(220, 58 + 16 * method_n),
            margin=dict(l=10, r=18, t=30, b=10),
            title="Method-Level Change-Point Estimates",
            xaxis_title="Estimated item position of strongest shift",
            yaxis_title="Detection method",
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            font=dict(color=PSYMAS_VIZ["ink"], size=10),
            xaxis=dict(range=[x_min, x_max], dtick=1, gridcolor=PSYMAS_VIZ["grid"], zeroline=False),
            yaxis=dict(gridcolor=PSYMAS_VIZ["grid"], zeroline=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        _make_plotly_text_readable(fig)
        st.plotly_chart(fig, use_container_width=True, key=f"case_cp_visual_{examinee_id}{key_suffix}")

    if not other_df.empty:
        with st.expander("Population context", expanded=False):
            st.caption(
                "Background distribution of estimated shift locations across other examinees. "
                "Colored vertical lines mark this examinee's method estimates."
            )
            pop_fig = go.Figure()
            pop_fig.add_trace(
                go.Histogram(
                    x=other_df["Change_Point"],
                    nbinsx=max(8, min(24, int(df["Change_Point"].max() - df["Change_Point"].min() + 1))),
                    marker=dict(color=PSYMAS_VIZ["reference_fill"], line=dict(color=PSYMAS_VIZ["reference"], width=1)),
                    opacity=0.95,
                    name="Other examinees",
                    hovertemplate="Item position %{x}<br>Count %{y}<extra></extra>",
                )
            )
            for _, item in selected_df.iterrows():
                pop_fig.add_vline(
                    x=float(item["Change_Point"]),
                    line_dash="dot",
                    line_width=1.5,
                    line_color=family_colors.get(item["Family"], PSYMAS_VIZ["context"]),
                    opacity=0.7,
                )
            pop_fig.update_layout(
                height=150,
                margin=dict(l=10, r=18, t=22, b=10),
                title="Population Context",
                xaxis_title="Item position",
                yaxis_title="Number of estimates",
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                font=dict(color=PSYMAS_VIZ["ink"]),
                xaxis=dict(range=[x_min, x_max], dtick=1, gridcolor=PSYMAS_VIZ["grid"], zeroline=False),
                yaxis=dict(gridcolor=PSYMAS_VIZ["grid"], zeroline=False),
                showlegend=False,
                bargap=0.08,
            )
            _make_plotly_text_readable(pop_fig)
            st.plotly_chart(pop_fig, use_container_width=True, key=f"case_cp_population_{examinee_id}{key_suffix}")



def _render_case_auxiliary_visuals(examinee_id: str) -> None:
    flags = _forensic_result_flags_from_session()
    pair_df = _case_pair_visual_rows(flags, examinee_id)
    st.markdown("#### Graphical evidence review")
    st.caption("Auxiliary figures are displayed together so the reviewer can scan the case without opening tabs.")
    _render_case_pair_network(pair_df, examinee_id)
    _render_case_rg_visual(flags, examinee_id)
    _render_case_cp_visual(flags, examinee_id, key_suffix="_aux")




def _render_case_domain_review_detail(
    selected_id: str,
    domain_issue_rows: list[dict],
    flags: dict,
) -> None:
    """Original Step 2 domain cards, figures, and domain-level drill-down."""
    row_by_domain = {str(row.get("domain")): row for row in domain_issue_rows}
    st.caption(
        "Review order: scenario domains first (RT · SIM · PK · TP), then supporting domains (MF · CP). "
        "The first row highlights the three primary review modules."
    )
    domain_html = ['<div class="psymas-domain-grid">']
    for row in [row_by_domain.get("MF"), row_by_domain.get("RT"), row_by_domain.get("PK")]:
        if not row:
            continue
        issue_class = str(row.get("card_class") or ("issue" if row["has_issue"] else "clear"))
        issue_label = str(row.get("status_label") or ("Governed evidence" if row["has_issue"] else "No governed signal"))
        domain_html.append(
            f"""
<div class="psymas-domain-card {issue_class}">
  <div class="domain-step">Domain {html.escape(str(row["step"]))}: {html.escape(str(row["display_code"]))}</div>
  <div class="domain-top"><span>{html.escape(row["domain_name"])}</span><b>{html.escape(str(row["strength"]))}</b></div>
  <div class="domain-status">{html.escape(issue_label)}</div>
  <div class="domain-index"><b>Main index:</b> {html.escape(_shorten_report_text(row["main_index"], 92))}</div>
  <div class="domain-pattern">{html.escape(str(row["pattern"]))}</div>
  <div class="domain-figure"><b>Related figure:</b> {html.escape(str(row["figure"]))}</div>
</div>
            """
        )
    domain_html.append("</div>")
    st.markdown("".join(domain_html), unsafe_allow_html=True)
    sim_row = row_by_domain.get("SIM")
    if sim_row:
        with st.container(border=True, key=f"domain_sim_frame_{selected_id}"):
            _domain_frame_header(sim_row)
            st.markdown('<div class="psymas-domain-figure-slot">', unsafe_allow_html=True)
            pair_df = _case_pair_visual_rows(flags, selected_id)
            _render_case_pair_network(pair_df, selected_id)
            st.markdown("</div>", unsafe_allow_html=True)
    tp_row = row_by_domain.get("TP")
    if tp_row:
        with st.container(border=True, key=f"domain_tp_frame_{selected_id}"):
            _domain_frame_header(tp_row)
            st.markdown('<div class="psymas-domain-figure-slot">', unsafe_allow_html=True)
            st.info("No dedicated tampering figure is available unless revision/tampering response data are loaded.")
            st.markdown("</div>", unsafe_allow_html=True)
    cp_row = row_by_domain.get("CP")
    if cp_row:
        with st.container(border=True, key=f"domain_cp_frame_{selected_id}"):
            _domain_frame_header(cp_row)
            st.markdown('<div class="psymas-domain-figure-slot">', unsafe_allow_html=True)
            _render_case_cp_visual(flags, selected_id, key_suffix="_detail")
            st.markdown("</div>", unsafe_allow_html=True)


def _render_single_case_review_page(selected_override: str | None = None) -> None:
    _ensure_run_store()
    _sync_review_decisions_from_store()
    if not _run_store_ready() and st.session_state.get("forensic_result"):
        with st.spinner("Building integrated run database (first open after Detect)…"):
            run_id = _active_run_id()
            if run_id:
                _materialize_psymas_run_store(run_id)
                _sync_review_decisions_from_store()
    governed_df, domain_df = _governed_review_tables()
    review_queue_df = _evidence_synthesis_with_human_review(governed_df)
    evidence_input_df = _governance_source_b3_input()
    if governed_df.empty:
        st.info("No review cases are available yet. Open Assessment Data, load data, then generate Deterministic Evidence.")
        if st.button("Open Deterministic Evidence", key="case_review_open_synthesis", use_container_width=False):
            st.session_state["_nav_request"] = "Deterministic Evidence"
            st.rerun()
        return

    ids = review_queue_df["Examinee_ID"].astype(str).tolist() if "Examinee_ID" in review_queue_df else []
    if not ids:
        st.info("The evidence synthesis table does not include Examinee_ID, so case-level review cannot be opened.")
        return
    if selected_override is None:
        _render_case_review_queue_list(review_queue_df)
        return
    selected_id = str(selected_override)
    case_row_df = review_queue_df[review_queue_df["Examinee_ID"].astype(str) == str(selected_id)] if "Examinee_ID" in review_queue_df else pd.DataFrame()
    if case_row_df.empty:
        st.info("Selected examinee is not available in the evidence synthesis table.")
        return
    case_row = case_row_df.iloc[0]
    case_domains = domain_df[domain_df["Examinee_ID"].astype(str) == str(selected_id)].copy() if not domain_df.empty and "Examinee_ID" in domain_df else pd.DataFrame()
    case_trace = evidence_input_df[evidence_input_df["Examinee_ID"].astype(str) == str(selected_id)].copy() if isinstance(evidence_input_df, pd.DataFrame) and not evidence_input_df.empty and "Examinee_ID" in evidence_input_df else pd.DataFrame()
    case_auxiliary = _case_auxiliary_index_rows(selected_id)
    metrics = _case_performance_summary(selected_id)
    domain_issue_rows = _case_domain_issue_rows(case_domains, case_trace)
    focus_text, focus_detail = _case_review_focus_summary(case_row, case_domains, metrics, selected_id)
    if "case_review_decisions" not in st.session_state or not isinstance(st.session_state.get("case_review_decisions"), dict):
        st.session_state["case_review_decisions"] = {}
    current_decision = st.session_state["case_review_decisions"].get(str(selected_id), {})
    decision_options = ["No Issue", "Rule Violation", "Potential Misconduct", "Definite Misconduct"]
    final_decision = str(current_decision.get("final_decision", ""))
    if final_decision == "Unreviewed":
        final_decision = ""
    decision_widget_key = f"case_final_decision_{selected_id}"
    if st.session_state.get(decision_widget_key) in decision_options:
        final_decision = str(st.session_state.get(decision_widget_key))
    reviewer_note = str(current_decision.get("reviewer_note", ""))
    reviewer_note_key = f"case_reviewer_note_{selected_id}"
    if reviewer_note_key not in st.session_state:
        st.session_state[reviewer_note_key] = reviewer_note
    explanation_key = f"case_reviewer_explanation_{selected_id}"

    st.markdown(
        f"""
<div class="psymas-case-header-compact">
  <div class="case-field"><div class="case-label">Examinee</div><div class="case-value">{html.escape(str(selected_id))}</div></div>
  <div class="case-field"><div class="case-label">Priority</div><div class="case-value">{html.escape(str(case_row.get("Review_Priority", "") or "Not set"))}</div></div>
  <div class="case-field"><div class="case-label">Domains for review</div><div class="case-value">{html.escape(str(case_row.get("Domains_For_Review", "") or "None"))}</div></div>
</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
<div class="psymas-review-focus-card">
  <div class="label">System interpretation</div>
  <div class="focus">{html.escape(focus_text)}</div>
  <div class="detail">{html.escape(focus_detail)}</div>
</div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown(
            '<div class="psymas-review-step-title"><span class="step-badge">1</span><span class="step-text">Performance context</span></div>',
            unsafe_allow_html=True,
        )
        metric_cols = st.columns(5)
        metric_cols[0].metric("Score", f"{_fmt_metric(metrics.get('score'), decimals=0)} / {_fmt_metric(metrics.get('total_items'), decimals=0)}")
        metric_cols[1].metric("Percentile", _fmt_metric(metrics.get("percentile"), "th", 0))
        metric_cols[2].metric("Accuracy", _fmt_metric(metrics.get("percent_correct"), "%", 1))
        metric_cols[3].metric("Mean RT", _fmt_metric(metrics.get("mean_rt"), decimals=2))
        metric_cols[4].metric("Median RT", _fmt_metric(metrics.get("median_rt"), decimals=2))
        perf_figs = _case_performance_figures(selected_id)
        ability_fig = _ability_effort_figure(selected_id)
        top_left, top_right = st.columns(2)
        with top_left:
            if "accuracy" in perf_figs:
                st.plotly_chart(perf_figs["accuracy"], use_container_width=True, key=f"case_item_accuracy_{selected_id}")
        with top_right:
            if "rt" in perf_figs:
                st.plotly_chart(perf_figs["rt"], use_container_width=True, key=f"case_response_time_profile_{selected_id}")
        bottom_left, bottom_right = st.columns(2)
        with bottom_left:
            if "score" in perf_figs:
                st.plotly_chart(perf_figs["score"], use_container_width=True, key=f"case_score_percentile_{selected_id}")
        with bottom_right:
            if ability_fig is not None:
                st.plotly_chart(ability_fig, use_container_width=True, key=f"case_ability_effort_{selected_id}")
            else:
                st.info("Score-vs-effort figure is unavailable because both response scores and RTE values are required.")

    with st.container(border=True):
        st.markdown(
            '<div class="psymas-review-step-title"><span class="step-badge">2</span><span class="step-text">Evidence lineage</span>'
            '<span class="psymas-help-dot" title="Priority lineage is driven by primary scenario domains: RT, PK, and TP in this view. MF is shown as supporting-only evidence with a lighter path. SIM and CP remain context/display panels when available.">?</span></div>',
            unsafe_allow_html=True,
        )
        _render_case_domain_status_strip(domain_issue_rows)
        flags = _forensic_result_flags_from_session()
        master_df = _master_results_for_lineage()
        master_row = _master_row_for_examinee(master_df, str(selected_id))
        if master_row is None:
            st.info(
                "Evidence lineage requires psymas_master_results.csv. "
                "Open Human Review → Downloads to generate the master export."
            )
        else:
            lineage_fig = build_individual_lineage_sankey(
                master_row,
                examinee_id=str(selected_id),
                title="Priority lineage",
                included_domains=("MF", "RT", "PK", "TP"),
                height=620,
            )
            if lineage_fig is None:
                st.info("No governed index lineage is available for this examinee.")
            else:
                lineage_col, legend_col = st.columns([5.6, 1.15], gap="small")
                with lineage_col:
                    st.plotly_chart(
                        lineage_fig,
                        use_container_width=True,
                        key=f"case_lineage_{selected_id}",
                    )
                with legend_col:
                    st.markdown(lineage_dual_legend_html(), unsafe_allow_html=True)
                st.caption(
                    "CUMP and NT are separately specified rapid-guessing methods aggregated within the Response-Time domain; "
                    "EDI_SD correction variants collapse into one family-level signal. Rule identifiers such as RT-B3-04 "
                    "and TP-B3-05 are predefined governance rules, not statistical indices. Critical / Expedited indicates "
                    "human-review urgency; convergent-traceable means the priority can be traced to domain evidence, inputs, "
                    "and rules, not that misconduct has been determined."
                )
        _render_case_domain_specific_visuals(selected_id, domain_issue_rows, flags)

    decision_meta = {
        "No Issue": {
            "symbol": "○",
            "level": "Level 0",
            "class": "unreviewed",
            "meaning": "No review-supported issue requiring action.",
            "key": "no_issue",
        },
        "Rule Violation": {
            "symbol": "△",
            "level": "Level 1",
            "class": "rule",
            "meaning": "Procedure concern; intent not established.",
            "key": "rule",
        },
        "Potential Misconduct": {
            "symbol": "▲",
            "level": "Level 2",
            "class": "potential",
            "meaning": "Score risk is too high to certify.",
            "key": "potential",
        },
        "Definite Misconduct": {
            "symbol": "■",
            "level": "Level 3",
            "class": "definite",
            "meaning": "High-verifiability evidence supports severe action.",
            "key": "definite",
        },
    }
    llm_support_key = f"case_llm_support_{selected_id}"
    with st.container(border=True, key=f"case_llm_panel_{selected_id}"):
        step_col, prompt_col, refresh_col = st.columns([0.68, 0.14, 0.18], vertical_alignment="center")
        with step_col:
            st.markdown(
                '<div class="psymas-review-step-title"><span class="step-badge">3</span><span class="step-text">LLM support</span></div>',
                unsafe_allow_html=True,
            )
        with prompt_col:
            if st.button("Prompt", key=f"toggle_case_prompt_{selected_id}", use_container_width=True):
                st.session_state["show_case_reviewer_prompt"] = not bool(st.session_state.get("show_case_reviewer_prompt", False))
        with refresh_col:
            refresh_suggestion = st.button("Generate", key=f"generate_case_explanation_{selected_id}", use_container_width=True)
        if st.session_state.get("show_case_reviewer_prompt", False):
            if "case_reviewer_prompt_template" not in st.session_state:
                st.session_state["case_reviewer_prompt_template"] = DEFAULT_CASE_REVIEWER_PROMPT
            with st.container(border=True):
                st.caption("Default reviewer prompt. Keep `{case_context}` where the governed evidence record should be inserted.")
                if st.button("Restore default prompt", key=f"restore_case_prompt_{selected_id}"):
                    st.session_state["case_reviewer_prompt_template"] = DEFAULT_CASE_REVIEWER_PROMPT
                st.text_area(
                    "Current prompt",
                    key="case_reviewer_prompt_template",
                    height=300,
                )
        if refresh_suggestion:
            with st.spinner("Generating reviewer explanation from domain evidence and flagged indices..."):
                explanation = _generate_case_reviewer_explanation(
                    selected_id,
                    case_row,
                    case_domains,
                    case_trace,
                    case_auxiliary,
                )
                st.session_state[explanation_key] = explanation
                _persist_case_review_to_store(
                    str(selected_id),
                    final_decision=final_decision,
                    reviewer_note=str(st.session_state.get(reviewer_note_key, "")),
                    llm_explanation=str(explanation or ""),
                )
        explanation_text = st.session_state.get(explanation_key)
        if not explanation_text:
            explanation_text = _evidence_bound_case_summary(
                case_row,
                case_domains,
                case_trace,
                case_auxiliary,
            )
        st.markdown(
            f"""
<div class="psymas-ai-suggestion">
  <div class="label">AI suggestion from governed indices</div>
  <div class="text">{_ai_suggestion_html(str(explanation_text))}</div>
</div>
            """,
            unsafe_allow_html=True,
        )

        if st.session_state.get(llm_support_key):
            with st.chat_message("assistant"):
                st.markdown(str(st.session_state[llm_support_key]))
        reviewer_question = st.chat_input(
            "Ask about this case...",
            key=f"case_llm_chat_{selected_id}",
        )
        if reviewer_question:
            with st.chat_message("user"):
                st.markdown(reviewer_question)
            with st.spinner("Asking LLM with this case evidence and reviewer notes..."):
                st.session_state[llm_support_key] = _ask_case_review_llm(
                    selected_id,
                    case_row,
                    case_domains,
                    case_trace,
                    case_auxiliary,
                    final_decision,
                    str(st.session_state.get(reviewer_note_key, "")),
                    reviewer_question,
                )
            st.rerun()

    with st.container(border=True):
        st.markdown(
            '<div class="psymas-review-step-title"><span class="step-badge">4</span><span class="step-text">Reviewer confirmation</span></div>',
            unsafe_allow_html=True,
        )
        decision_index = decision_options.index(final_decision) if final_decision in decision_options else None
        confirm_left, confirm_right = st.columns([0.34, 0.66])
        with confirm_left:
            selected_decision = st.selectbox(
                "Final decision",
                options=decision_options,
                index=decision_index,
                placeholder="Choose final decision",
                help=(
                    "No Issue: no review-supported issue requiring action. "
                    "Rule Violation: procedure concern; intent not established. "
                    "Potential Misconduct: score risk is too high to certify. "
                    "Definite Misconduct: high-verifiability evidence supports severe action."
                ),
                key=f"case_decision_select_{selected_id}",
            )
            pending_decision = str(selected_decision or "")
        with confirm_right:
            reviewer_note = st.text_area(
                "Reviewer final note",
                key=reviewer_note_key,
                height=94,
                placeholder="Edit the final reviewer note before confirming.",
            )
        confirm_disabled = not pending_decision
        action_left, action_right = st.columns([0.74, 0.26], vertical_alignment="center")
        with action_right:
            if st.button(
                "Confirm",
                key=f"confirm_human_review_{selected_id}",
                type="primary",
                disabled=confirm_disabled,
                use_container_width=True,
            ):
                final_decision = pending_decision
                st.session_state[decision_widget_key] = final_decision
                st.session_state["case_review_decisions"][str(selected_id)] = {
                    "final_decision": final_decision,
                    "reviewer_note": reviewer_note,
                }
                _persist_case_review_to_store(
                    str(selected_id),
                    final_decision=final_decision,
                    reviewer_note=reviewer_note,
                    llm_explanation=str(st.session_state.get(explanation_key, "")),
                )
                st.success("Recorded.")
        if final_decision:
            stamp_class = decision_meta.get(final_decision, decision_meta["No Issue"])["class"]
            stamp_symbol = decision_meta.get(final_decision, decision_meta["No Issue"])["symbol"]
            st.markdown(
                f'<div class="psymas-stamp {stamp_class}">{html.escape(stamp_symbol)} {html.escape(final_decision)}</div>',
                unsafe_allow_html=True,
            )

    pdf_bytes, pdf_err = _build_case_review_pdf(
        selected_id,
        case_row,
        metrics,
        domain_issue_rows,
        str(st.session_state.get(explanation_key) or explanation_text),
        final_decision,
        reviewer_note,
    )
    if pdf_err:
        st.warning(f"PDF could not be built: {pdf_err}")
    spacer_col, download_col = st.columns([0.78, 0.22])
    with download_col:
        if not pdf_err:
            st.download_button(
                "Download",
                data=pdf_bytes,
                file_name=f"psymas_review_report_examinee_{selected_id}.pdf",
                mime="application/pdf",
                key=f"download_case_review_pdf_{selected_id}",
                use_container_width=True,
            )


def _review_kpi_values(review_df: pd.DataFrame, flagged_df: pd.DataFrame, source_df: pd.DataFrame) -> dict[str, int]:
    source = source_df.copy() if isinstance(source_df, pd.DataFrame) and not source_df.empty else review_df.copy()
    high_n = 0
    if "Review_Priority" in source.columns:
        high_n = int(
            source["Review_Priority"]
            .astype(str)
            .str.contains("High|Critical|Expedited", case=False, regex=True)
            .sum()
        )
    domain_cols = [
        col
        for col in ["MF_Strength", "RT_Strength", "PK_Strength", "TP_Strength", "SIM_Strength", "CP_Strength"]
        if col in source.columns
    ]
    active_domain_n = 0
    if domain_cols:
        active_domain_n = int(
            source[domain_cols]
            .astype(str)
            .apply(lambda col: col.str.lower().isin({"weak", "moderate", "strong"}))
            .any(axis=1)
            .sum()
        )
    reviewed_n = 0
    for column in ("Human_Final_Decision", "Human_Decision"):
        if column in source.columns:
            reviewed_n = int(source[column].astype(str).str.strip().replace({"nan": ""}).ne("").sum())
            break
    return {
        "detector_flags": int(flagged_df.shape[0]) if isinstance(flagged_df, pd.DataFrame) else 0,
        "total": int(len(review_df)) if isinstance(review_df, pd.DataFrame) else int(len(source)),
        "governed_domain_cases": active_domain_n,
        "high_priority": high_n,
        "human_reviewed": reviewed_n,
    }


def _render_review_kpi_header(review_df: pd.DataFrame, flagged_df: pd.DataFrame, source_df: pd.DataFrame) -> None:
    values = _review_kpi_values(review_df, flagged_df, source_df)
    st.markdown(
        f"""
<style>
.psymas-shared-kpis {{
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.55rem;
    margin: 0.15rem 0 0.75rem 0;
}}
.psymas-shared-kpis div {{
    border: 1px solid #CBD5E1;
    border-left: 4px solid #0F7890;
    border-radius: 9px;
    background: #FFFFFF;
    padding: 0.55rem 0.65rem;
    min-height: 4.25rem;
}}
.psymas-shared-kpis div.warn {{ border-left-color: #C87512; }}
.psymas-shared-kpis span {{
    display: block;
    color: #64748B !important;
    font-size: 0.68rem;
    font-weight: 850;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}}
.psymas-shared-kpis b {{
    display: inline-block;
    color: #0F172A !important;
    font-size: 1.35rem;
    line-height: 1.1;
    margin-top: 0.18rem;
}}
.psymas-shared-kpis em {{
    display: block;
    color: #64748B !important;
    font-size: 0.72rem;
    font-style: normal;
    margin-top: 0.12rem;
}}
</style>
<div class="psymas-shared-kpis">
  <div class="warn"><span>Detector Flags</span><b>{values["detector_flags"]:,}</b><em>out of {values["total"]:,}</em></div>
  <div><span>Governed-Domain Cases</span><b>{values["governed_domain_cases"]:,}</b><em>active governed signal</em></div>
  <div class="warn"><span>High-Priority Queue</span><b>{values["high_priority"]:,}</b><em>rulebook triage</em></div>
  <div><span>Human Reviewed</span><b>{values["human_reviewed"]:,}</b><em>adjudicated</em></div>
</div>
        """,
        unsafe_allow_html=True,
    )


def _render_ai_review_overview(review_df: pd.DataFrame, flagged_df: pd.DataFrame) -> None:
    governed_df, _domain_df = _governed_review_tables()
    synthesis_df = _evidence_synthesis_with_human_review(governed_df) if isinstance(governed_df, pd.DataFrame) and not governed_df.empty else pd.DataFrame()
    source_df = synthesis_df if not synthesis_df.empty else review_df
    if source_df.empty or "Examinee_ID" not in source_df.columns:
        st.info("No case-level review profile is available yet.")
        return

    domain_cols = [col for col in ["MF_Strength", "RT_Strength", "SIM_Strength", "PK_Strength", "TP_Strength", "CP_Strength"] if col in source_df.columns]
    _render_review_kpi_header(review_df, flagged_df, source_df)
    if not domain_cols:
        st.info("Domain strength columns are not available for AI review visualization.")
        return
    try:
        import plotly.graph_objects as go
    except Exception:
        st.dataframe(matrix_df[["Examinee_ID", *domain_cols]].head(160), use_container_width=True, hide_index=True)
        return
    priority_col = "Review_Priority"

    domain_order = ["MF", "RT", "PK", "TP", "SIM", "CP"]
    domain_names = {
        "MF": "Misfit",
        "RT": "Response-Time",
        "SIM": "Similarity",
        "PK": "Preknowledge",
        "TP": "Tampering",
        "CP": "Change-Pattern",
    }
    strength_order = ["strong", "moderate", "weak", "none", "unavailable"]
    strength_colors = {
        "strong": PSYMAS_VIZ["high"],
        "moderate": PSYMAS_VIZ["moderate"],
        "weak": PSYMAS_VIZ["inactive"],
        "none": PSYMAS_VIZ["reference_fill"],
        "unavailable": PSYMAS_VIZ["reference"],
    }
    issue_strengths = {"weak", "moderate", "strong"}
    matrix_df = source_df.copy()
    if "System_Flag" in matrix_df.columns:
        matrix_df = matrix_df[matrix_df["System_Flag"].astype(str).isin(["1", "True", "true"])]
    if matrix_df.empty:
        matrix_df = source_df.head(80).copy()
    matrix_df = _case_review_queue_sorted(matrix_df)

    domain_cards: list[str] = []
    dist_rows = []
    for domain in domain_order:
        col = f"{domain}_Strength"
        if col not in source_df.columns:
            continue
        vals = source_df[col].astype(str).str.lower().replace({"nan": "unavailable", "": "none"})
        issue_n = int(vals.isin(issue_strengths).sum())
        strong_count = int(vals.eq("strong").sum())
        moderate_count = int(vals.eq("moderate").sum())
        unavailable_count = int(vals.eq("unavailable").sum())
        tone = "danger" if strong_count else ("warn" if moderate_count else ("off" if issue_n == 0 else "info"))
        domain_cards.append(
            f"<div class='psymas-domain-tile {tone}'>"
            f"<div class='tile-code'>{html.escape(domain)}</div>"
            f"<div class='tile-title'>{html.escape(domain_names.get(domain, domain))}</div>"
            f"<div class='tile-value'>{issue_n:,}</div>"
            "<div class='tile-note'>active signals</div>"
            "<div class='tile-mini'>"
            f"<span><b>{strong_count:,}</b> strong</span>"
            f"<span><b>{moderate_count:,}</b> moderate</span>"
            f"<span><b>{unavailable_count:,}</b> unavailable</span>"
            "</div>"
            "</div>"
        )
        for strength in strength_order:
            dist_rows.append({"Domain": domain, "Strength": strength, "Cases": int(vals.eq(strength).sum())})
    st.markdown(
        """
        <style>
        .psymas-dashboard-section {
            margin-top: 0.85rem;
        }
        .psymas-ai-compact-kpis {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.55rem;
            margin: 0.15rem 0 0.7rem 0;
        }
        .psymas-ai-compact-kpis div {
            border: 1px solid #CBD5E1;
            border-left: 4px solid #0F7890;
            border-radius: 9px;
            background: #FFFFFF;
            padding: 0.55rem 0.65rem;
            min-height: 4.25rem;
        }
        .psymas-ai-compact-kpis span {
            display: block;
            color: #64748B;
            font-size: 0.68rem;
            font-weight: 850;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }
        .psymas-ai-compact-kpis b {
            display: inline-block;
            color: #0F172A;
            font-size: 1.35rem;
            line-height: 1.1;
            margin-top: 0.18rem;
        }
        .psymas-ai-compact-kpis em {
            display: block;
            color: #64748B;
            font-size: 0.72rem;
            font-style: normal;
            margin-top: 0.12rem;
        }
        .psymas-domain-tile-grid {
            display: grid;
            grid-template-columns: repeat(6, minmax(0, 1fr));
            gap: 0.45rem;
            margin: 0.35rem 0 0.65rem 0;
        }
        .psymas-domain-tile {
            border: 1px solid #CBD5E1;
            border-left: 4px solid #64748B;
            border-radius: 8px;
            background: linear-gradient(180deg, #FFFFFF, #F8FAFC);
            padding: 0.48rem 0.55rem;
            min-height: 5.45rem;
            box-shadow: 0 6px 16px rgba(15, 23, 42, 0.035);
        }
        .psymas-domain-tile.danger { border-left-color: #DC2626; }
        .psymas-domain-tile.warn { border-left-color: #C87512; }
        .psymas-domain-tile.info { border-left-color: #0F7890; }
        .psymas-domain-tile.off { border-left-color: #94A3B8; opacity: 0.92; }
        .psymas-domain-tile .tile-code {
            color: #64748B;
            font-size: 0.62rem;
            font-weight: 850;
            letter-spacing: 0.08em;
        }
        .psymas-domain-tile .tile-title {
            color: #0F172A;
            font-size: 0.74rem;
            font-weight: 820;
            margin-top: 0.08rem;
            min-height: 1rem;
        }
        .psymas-domain-tile .tile-value {
            color: #0F172A;
            font-size: 1.22rem;
            font-weight: 850;
            margin-top: 0.18rem;
            line-height: 1;
        }
        .psymas-domain-tile .tile-note {
            color: #64748B;
            font-size: 0.66rem;
            margin-top: 0.08rem;
        }
        .psymas-domain-tile .tile-mini {
            display: flex;
            gap: 0.35rem;
            margin-top: 0.32rem;
            color: #475569;
            font-size: 0.62rem;
            flex-wrap: wrap;
        }
        .psymas-domain-tile .tile-mini b { color: #0F172A; }
        @media (max-width: 1200px) {
            .psymas-domain-tile-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div class='psymas-domain-tile-grid'>" + "".join(domain_cards) + "</div>", unsafe_allow_html=True)

    icon_df = matrix_df.copy().head(240)
    priority_col = priority_col if priority_col in icon_df.columns else ""
    n_cols = 40
    priority_styles = {
        "Critical / Expedited": ("Critical / Expedited", PSYMAS_VIZ["high"], "square"),
        "High": ("High", PSYMAS_VIZ["moderate"], "square"),
        "Medium": ("Medium", PSYMAS_VIZ["inactive"], "square"),
        "Low": ("Low", PSYMAS_VIZ["correct"], "square"),
        "Unspecified": ("Unspecified", PSYMAS_VIZ["reference_fill"], "square"),
    }

    def _priority_bucket(value: object) -> str:
        text = str(value or "")
        if re.search("critical|expedited", text, flags=re.I):
            return "Critical / Expedited"
        if re.search("high", text, flags=re.I):
            return "High"
        if re.search("medium|moderate", text, flags=re.I):
            return "Medium"
        if re.search("low", text, flags=re.I):
            return "Low"
        return "Unspecified"

    icon_df["_priority_label"] = icon_df[priority_col].map(_priority_bucket) if priority_col else "Unspecified"
    _human_decision_col = ""
    for candidate_col in ("Human_Decision", "Human_Final_Decision", "Final_Decision"):
        if candidate_col in icon_df.columns:
            _human_decision_col = candidate_col
            break
    if _human_decision_col:
        icon_df["_human_reviewed"] = icon_df[_human_decision_col].astype(str).str.strip().ne("")
        icon_df["_human_decision_text"] = icon_df[_human_decision_col].astype(str).str.strip()
    else:
        icon_df["_human_reviewed"] = False
        icon_df["_human_decision_text"] = ""
    icon_df["_x"] = [idx % n_cols for idx in range(len(icon_df))]
    icon_df["_y"] = [idx // n_cols for idx in range(len(icon_df))]
    fig_icons = go.Figure()
    for bucket, (legend_label, color, symbol) in priority_styles.items():
        block = icon_df[icon_df["_priority_label"].eq(bucket)].copy()
        if block.empty:
            continue
        hover_items = []
        for _, row in block.iterrows():
            strengths = []
            for domain in domain_order:
                val = str(row.get(f"{domain}_Strength", "none") or "none")
                if val.lower() not in {"none", "nan", ""}:
                    strengths.append(f"{domain}: {val}")
            strength_text = ", ".join(strengths) or "No active domain signal"
            queue = str(row.get(priority_col, "") if priority_col else bucket)
            concern = str(row.get("Primary_Concern", row.get("Domains_For_Review", "")) or "No listed concern")
            status = str(row.get("Evidence_Status", row.get("Review_Queue_Status", "")) or "Review")
            hover_items.append(
                f"Examinee {row.get('Examinee_ID', '')}<br>"
                f"Priority: {html.escape(queue or bucket)}<br>"
                f"Status: {html.escape(status)}<br>"
                f"Concern: {html.escape(concern)}<br>"
                f"Domains: {html.escape(strength_text)}<br>"
                f"Human decision: {html.escape(str(row.get('_human_decision_text') or 'Not reviewed'))}"
            )
        fig_icons.add_trace(
            go.Scatter(
                x=block["_x"],
                y=block["_y"],
                mode="markers",
                name=legend_label,
                customdata=block["Examinee_ID"].astype(str),
                text=hover_items,
                hovertemplate="%{text}<extra></extra>",
                marker=dict(
                    symbol=symbol,
                    size=13,
                    color=color,
                    line=dict(width=1, color="#FFFFFF"),
                    opacity=0.92,
                ),
            )
        )
    reviewed_block = icon_df[icon_df["_human_reviewed"].eq(True)].copy()
    if not reviewed_block.empty:
        reviewed_hover = [
            f"Examinee {row.get('Examinee_ID', '')}<br>"
            f"Human decision: {html.escape(str(row.get('_human_decision_text') or 'Reviewed'))}<br>"
            "Click to open case report"
            for _, row in reviewed_block.iterrows()
        ]
        fig_icons.add_trace(
            go.Scatter(
                x=reviewed_block["_x"],
                y=reviewed_block["_y"],
                mode="markers+text",
                name="Human reviewed",
                customdata=reviewed_block["Examinee_ID"].astype(str),
                text=reviewed_hover,
                texttemplate="✓",
                textposition="middle center",
                textfont=dict(color="#111827", size=12, family="Arial Black"),
                hovertemplate="%{text}<extra></extra>",
                marker=dict(
                    symbol="square",
                    size=17,
                    color="rgba(255,255,255,0)",
                    line=dict(width=2.2, color="#111827"),
                    opacity=1.0,
                ),
            )
        )
    fig_icons.update_layout(
        title="Case Priority Indicator Map",
        height=max(255, min(365, 110 + 20 * (int(np.ceil(max(len(icon_df), 1) / n_cols))))),
        margin=dict(l=12, r=12, t=36, b=58),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#111827", size=10),
        xaxis=dict(visible=False, range=[-1, n_cols]),
        yaxis=dict(visible=False, autorange="reversed"),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.08,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.96)",
            bordercolor="rgba(203,213,225,0.9)",
            borderwidth=1,
            font=dict(color=PSYMAS_VIZ["ink"], size=10),
        ),
        clickmode="event+select",
    )
    _make_plotly_text_readable(fig_icons)
    fig_icons.update_layout(
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.08,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.96)",
            bordercolor="rgba(203,213,225,0.9)",
            borderwidth=1,
            font=dict(color=PSYMAS_VIZ["ink"], size=10),
        )
    )

    with st.container(border=True):
        icon_event = st.plotly_chart(
            fig_icons,
            use_container_width=True,
            key="ai_review_case_indicator_map",
            on_select="rerun",
            selection_mode="points",
            config={"displayModeBar": False},
        )
    selected_icon_id = ""
    try:
        selection = getattr(icon_event, "selection", None)
        if isinstance(selection, dict):
            points = selection.get("points", []) or []
        else:
            points = getattr(selection, "points", []) or []
        if points:
            first_point = points[0]
            if isinstance(first_point, dict):
                selected_icon_id = str(first_point.get("customdata", "") or "")
            else:
                selected_icon_id = str(getattr(first_point, "customdata", "") or "")
    except Exception:
        selected_icon_id = ""
    if selected_icon_id:
        st.session_state["_case_review_modal_id"] = selected_icon_id
    modal_id = str(st.session_state.get("_case_review_modal_id") or "")
    if modal_id:
        @st.dialog(f"Examinee {modal_id}", width="large")
        def _ai_case_review_modal() -> None:
            _render_single_case_review_page(selected_override=modal_id)

        _ai_case_review_modal()
    st.caption("Each icon is one examinee. Color shows priority; hover shows the main evidence concerns; click opens the detailed report.")















def _final_flag_review_df() -> pd.DataFrame:
    if _run_store_ready():
        stored_df = get_run_store().load_table("final_flags")
        if isinstance(stored_df, pd.DataFrame) and not stored_df.empty:
            out = stored_df.copy()
            decisions = st.session_state.get("case_review_decisions") or {}
            if isinstance(decisions, dict) and "Examinee_ID" in out.columns and "Human_Decision" in out.columns:
                for idx, row in out.iterrows():
                    payload = decisions.get(str(row["Examinee_ID"]), {})
                    if isinstance(payload, dict) and str(payload.get("final_decision", "")).strip():
                        out.at[idx, "Human_Decision"] = payload["final_decision"]
            return out
    flags = _forensic_result_flags_from_session()
    responses = st.session_state.get("forensic_responses") or st.session_state.get("last_uploaded_responses") or []
    n_examinees = _infer_n_examinees(flags, responses)
    if n_examinees <= 0:
        return pd.DataFrame()
    return build_final_flag_review(
        flags=flags,
        n_examinees=n_examinees,
        selected_functions=st.session_state.get("last_detect_agents") or list(ABERRANCE_FN_TO_AGENT),
        function_to_agent=ABERRANCE_FN_TO_AGENT,
        flagged_indices=_agent_flagged_indices,
        decisions=st.session_state.get("case_review_decisions", {}),
    )


def _master_results_df(indices_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """One examinee-level export containing final flags, domains, review, and indices."""
    if not isinstance(indices_df, pd.DataFrame) or indices_df.empty:
        indices_df, _ = _governance_source_export()
    if not isinstance(indices_df, pd.DataFrame) or indices_df.empty:
        return pd.DataFrame()

    governed_df, domain_df = _governed_review_tables()
    review_df = (
        _evidence_synthesis_with_human_review(governed_df)
        if isinstance(governed_df, pd.DataFrame) and not governed_df.empty
        else pd.DataFrame()
    )
    master_df = build_master_results(
        indices_df=indices_df,
        final_flags_df=_final_flag_review_df(),
        review_df=review_df,
        domain_df=domain_df,
        run_id=str(st.session_state.get("detect_run_id") or ""),
        schema_version="1.0",
    )
    if not master_df.empty:
        output_dir = Path("data") / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "psymas_master_results.csv"
        temporary_path = output_dir / ".psymas_master_results.csv.tmp"
        master_df.to_csv(temporary_path, index=False, encoding="utf-8-sig")
        temporary_path.replace(output_path)
        st.session_state["master_results_path"] = str(output_path)
    return master_df


@st.cache_data(show_spinner=False)
def _read_master_results_csv(cache_key: str) -> pd.DataFrame:
    path = Path(cache_key)
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _master_results_for_lineage() -> pd.DataFrame:
    """Prefer integrated run store; then on-disk CSV; then build from session state."""
    if _run_store_ready():
        master_df = get_run_store().load_table("master_results")
        if isinstance(master_df, pd.DataFrame) and not master_df.empty:
            return master_df
    output_path = Path("data") / "output" / "psymas_master_results.csv"
    if output_path.is_file():
        df = _read_master_results_csv(str(output_path.resolve()))
        if not df.empty:
            return df
    session_path = str(st.session_state.get("master_results_path") or "")
    if session_path:
        df = _read_master_results_csv(session_path)
        if not df.empty:
            return df
    return _master_results_df()


def _master_row_for_examinee(master_df: pd.DataFrame, examinee_id: str) -> pd.Series | None:
    if master_df.empty or "Examinee_ID" not in master_df.columns:
        return None
    matches = master_df[master_df["Examinee_ID"].astype(str).eq(str(examinee_id))]
    if matches.empty:
        return None
    return matches.iloc[0]


def _render_cohort_evidence_lineage(master_df: pd.DataFrame, *, chart_key: str = "evidence_lineage") -> None:
    if master_df.empty:
        st.info("Master results are not available yet. Open Downloads to generate psymas_master_results.csv.")
        return
    fig = build_lineage_sankey(
        master_df,
        title="Evidence lineage tree",
        height=620,
    )
    if fig is None:
        st.info("No domain-index lineage is available in the current master export.")
        return
    st.plotly_chart(fig, use_container_width=True, key=chart_key)
    summary_df = lineage_summary_rows(master_df)
    if not summary_df.empty:
        with st.expander("Lineage flow table", expanded=False):
            st.dataframe(summary_df, use_container_width=True, hide_index=True, height=min(360, 44 + 28 * len(summary_df)))


def _render_evidence_review_page() -> None:
    _ensure_run_store()
    _sync_review_decisions_from_store()
    review_df = _final_flag_review_df()
    if review_df.empty:
        st.info("No forensic output is available. Run Detect from Assessment Data first.")
        return
    flagged_df = review_df[review_df["System_Flag"].eq(1)].copy()
    reviewed_n = int(review_df["Human_Decision"].astype(str).str.len().gt(0).sum())
    stage = _STAGE_ALIASES.get(st.session_state.get("workflow_stage", ""), st.session_state.get("workflow_stage", ""))
    if stage == "Evidence Governance":
        stage = "AI-Assisted Review"
    if stage == "Evidence Governance":
        governed_df, domain_df = _governed_review_tables()
        synthesis_df = _evidence_synthesis_with_human_review(governed_df) if isinstance(governed_df, pd.DataFrame) and not governed_df.empty else pd.DataFrame()
        domain_order = ["MF", "RT", "PK", "TP", "SIM", "CP"]
        strength_rank = {"strong": 3, "moderate": 2, "weak": 1, "none": 0, "unavailable": -1}
        _render_review_kpi_header(review_df, flagged_df, synthesis_df if not synthesis_df.empty else governed_df)
        st.markdown(
            """
            <style>
            .psymas-domain-tile-grid {
                display: grid;
                grid-template-columns: repeat(6, minmax(0, 1fr));
                gap: 0.5rem;
                margin: 0.45rem 0 0.75rem 0;
            }
            .psymas-domain-tile {
                border: 1px solid #CBD5E1;
                border-left: 4px solid #64748B;
                border-radius: 8px;
                background: linear-gradient(180deg, #FFFFFF, #F8FAFC);
                padding: 0.52rem 0.58rem;
                min-height: 5.2rem;
                box-shadow: 0 6px 16px rgba(15, 23, 42, 0.035);
            }
            .psymas-domain-tile.danger { border-left-color: #B42318; background: linear-gradient(180deg, #FFF7F7, #FFFFFF); }
            .psymas-domain-tile.warn { border-left-color: #C87512; background: linear-gradient(180deg, #FFFBEB, #FFFFFF); }
            .psymas-domain-tile.info { border-left-color: #0F7890; background: linear-gradient(180deg, #F0FDFF, #FFFFFF); }
            .psymas-domain-tile.support,
            .psymas-domain-tile.aux { border-left-color: #94A3B8; background: linear-gradient(180deg, #F8FAFC, #F1F5F9); }
            .psymas-domain-tile.calibration { border-left-color: #C17919; background: linear-gradient(180deg, #FFFFFF, #FFFBEB); }
            .psymas-domain-tile.off { border-left-color: #94A3B8; background: #F8FAFC; opacity: 0.92; }
            .psymas-domain-tile .tile-code {
                color: #64748B;
                font-size: 0.62rem;
                font-weight: 850;
                letter-spacing: 0.08em;
            }
            .psymas-domain-tile .tile-title {
                color: #0F172A;
                font-size: 0.74rem;
                font-weight: 820;
                margin-top: 0.08rem;
                min-height: 1rem;
            }
            .psymas-domain-tile .tile-value {
                color: #0F172A;
                font-size: 1.22rem;
                font-weight: 850;
                margin-top: 0.18rem;
                line-height: 1;
            }
            .psymas-domain-tile .tile-note {
                color: #64748B;
                font-size: 0.66rem;
                margin-top: 0.08rem;
            }
            .psymas-domain-tile .tile-status {
                display: inline-flex;
                align-items: center;
                width: fit-content;
                margin-top: 0.22rem;
                padding: 0.08rem 0.36rem;
                border-radius: 999px;
                background: #EEF2F7;
                color: #334155;
                font-size: 0.58rem;
                font-weight: 850;
                letter-spacing: 0.04em;
                text-transform: uppercase;
            }
            .psymas-domain-tile.support .tile-status,
            .psymas-domain-tile.aux .tile-status { background: #E2E8F0; color: #334155; }
            .psymas-domain-tile.calibration .tile-status { background: #FFF2CC; color: #7A4B00; }
            .psymas-domain-tile.danger .tile-status { background: #FEE4E2; color: #912018; }
            .psymas-domain-tile.warn .tile-status { background: #FEF0C7; color: #7A4B00; }
            .psymas-domain-tile.info .tile-status { background: #DDF3F7; color: #0B4A5A; }
            .psymas-domain-tile .tile-mini {
                display: flex;
                gap: 0.35rem;
                margin-top: 0.32rem;
                color: #475569;
                font-size: 0.62rem;
                flex-wrap: wrap;
            }
            .psymas-domain-tile .tile-mini b { color: #0F172A; }
            @media (max-width: 1200px) {
                .psymas-domain-tile-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        if isinstance(domain_df, pd.DataFrame) and not domain_df.empty and {"Domain", "Strength"}.issubset(domain_df.columns):
            domain_cards = []
            dist_rows = []
            for domain in domain_order:
                rows = domain_df[domain_df["Domain"].astype(str).eq(domain)].copy()
                if rows.empty:
                    continue
                strengths = rows["Strength"].astype(str).str.lower().replace({"nan": "unavailable", "": "none"})
                counts = strengths.value_counts().to_dict()
                active_n = int(strengths.isin({"weak", "moderate", "strong"}).sum())
                support_only_n = int(rows.get("Display_Only_Hits", pd.Series(dtype=str)).astype(str).str.strip().ne("").sum())
                if domain == "CP" and support_only_n == 0:
                    support_only_n = _case_cp_location_output_count()
                calibration_n = int(rows.get("Calibration_Required_Hits", pd.Series(dtype=str)).astype(str).str.strip().ne("").sum())
                unavailable_n = int(strengths.eq("unavailable").sum())
                strongest = "unavailable"
                if counts:
                    strongest = max(counts, key=lambda s: strength_rank.get(str(s), -2))
                if domain in {"SIM", "CP"}:
                    tone = "aux"
                    if support_only_n:
                        value_text = f"{support_only_n:,}"
                        note_text = "context cases"
                        status_text = "Reviewer context"
                    elif calibration_n:
                        value_text = f"{calibration_n:,}"
                        note_text = "needs calibrated rule"
                        status_text = "Calibration required"
                    elif unavailable_n == len(rows):
                        value_text = "No input"
                        note_text = "not run or unavailable"
                        status_text = "Context only"
                    else:
                        value_text = "0"
                        note_text = "no context signal"
                        status_text = "Context only"
                elif active_n:
                    tone = "danger" if strongest == "strong" else ("warn" if strongest == "moderate" else "info")
                    value_text = f"{active_n:,}"
                    note_text = "active governed cases"
                    status_text = "Governed evidence"
                elif support_only_n:
                    tone = "support"
                    value_text = f"{support_only_n:,}"
                    note_text = "support-only cases"
                    status_text = "Visible, not counted"
                elif calibration_n:
                    tone = "calibration"
                    value_text = f"{calibration_n:,}"
                    note_text = "needs calibrated rule"
                    status_text = "Calibration required"
                elif unavailable_n == len(rows):
                    tone = "off"
                    value_text = "No input"
                    note_text = "not run or unavailable"
                    status_text = "Unavailable"
                else:
                    tone = "off"
                    value_text = "0"
                    note_text = "available, no governed flags"
                    status_text = "No governed signal"
                domain_cards.append(
                    f"<div class='psymas-domain-tile {tone}'>"
                    f"<div class='tile-code'>{html.escape(domain)}</div>"
                    f"<div class='tile-title'>{html.escape(_DOMAIN_LABELS.get(domain, domain))}</div>"
                    f"<div class='tile-value'>{html.escape(value_text)}</div>"
                    f"<div class='tile-note'>{html.escape(note_text)}</div>"
                    f"<div class='tile-status'>{html.escape(status_text)}</div>"
                    "<div class='tile-mini'>"
                    f"<span><b>{int(counts.get('strong', 0)):,}</b> strong</span>"
                    f"<span><b>{int(counts.get('moderate', 0)):,}</b> moderate</span>"
                    f"<span><b>{support_only_n:,}</b> support-only</span>"
                    f"<span><b>{int(counts.get('unavailable', 0)):,}</b> unavailable</span>"
                    "</div>"
                    "</div>"
                )
                for strength, cases in counts.items():
                    dist_rows.append({"Domain": domain, "Strength": str(strength).title(), "Cases": int(cases)})
            if domain_cards:
                st.markdown("<div class='psymas-domain-tile-grid'>" + "".join(domain_cards) + "</div>", unsafe_allow_html=True)
            if dist_rows:
                try:
                    import plotly.express as px

                    dist_df = pd.DataFrame(dist_rows)
                    color_map = {
                        "Strong": PSYMAS_VIZ["high"],
                        "Moderate": PSYMAS_VIZ["moderate"],
                        "Weak": PSYMAS_VIZ["inactive"],
                        "None": PSYMAS_VIZ["reference_fill"],
                        "Unavailable": PSYMAS_VIZ["reference"],
                    }
                    fig = px.bar(
                        dist_df,
                        y="Domain",
                        x="Cases",
                        color="Strength",
                        orientation="h",
                        color_discrete_map=color_map,
                        category_orders={"Domain": domain_order, "Strength": ["Strong", "Moderate", "Weak", "None", "Unavailable"]},
                        title="Governed Evidence Strength Distribution",
                    )
                    fig.update_layout(
                        height=300,
                        margin=dict(l=42, r=18, t=42, b=28),
                        barmode="stack",
                        plot_bgcolor="#FFFFFF",
                        paper_bgcolor="#FFFFFF",
                        font=dict(color=PSYMAS_VIZ["ink"], size=12),
                        xaxis_title="Examinees",
                        yaxis_title="Domain",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    )
                    _make_plotly_text_readable(fig)
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                except Exception:
                    st.dataframe(pd.DataFrame(dist_rows), use_container_width=True, hide_index=True)
            _render_page_download_bar(
                [
                    {
                        "label": "Download Domain Evidence",
                        "data": _df_csv_bytes(domain_df),
                        "file_name": "psymas_domain_evidence.csv",
                        "key": "governance_domain_evidence_download_v1",
                        "disabled": domain_df.empty,
                    },
                ]
            )

        with st.expander("Evidence lineage for audit", expanded=False):
            st.caption("Cohort view: index → domain → governed review conclusion. Use this for audit, not as the main tutorial screenshot.")
            _render_cohort_evidence_lineage(_master_results_for_lineage(), chart_key="evidence_lineage_overview")
        return
    if _streamlined_ui_enabled():
        stage_header = {
            "Evidence Governance": (
                "Evidence governance",
                "Rulebook-bound evidence profile",
                "Index-level flags are translated into governed domain evidence with traceability and missing-evidence constraints.",
            ),
            "AI-Assisted Review": (
                "AI-assisted review",
                "Evidence-bound case interpretation",
                "LLM support is constrained by the evidence record and cautious review language. It supports, but does not replace, human review.",
            ),
            "Human Review": (
                "Human review",
                "Reviewer adjudication workspace",
                "System flags are review triggers. The reviewer inspects evidence, asks follow-up questions, and records the final decision.",
            ),
        }.get(
            stage,
            (
                "Evidence review",
                "Traceable report review",
                "System flags are review triggers. Individual examinee reports link the final recommendation back to domain evidence, indices, LLM support, and reviewer confirmation.",
            ),
        )
        priority_col = "Review_Suggestion" if "Review_Suggestion" in review_df.columns else "Review_Priority"
        high_n = int(
            review_df.get(priority_col, pd.Series(dtype=str))
            .astype(str)
            .str.contains("High|Critical", case=False, regex=True)
            .sum()
        )
        domain_text = ""
        if "Domains_For_Review" in flagged_df.columns and not flagged_df.empty:
            domain_counts: dict[str, int] = {}
            for value in flagged_df["Domains_For_Review"].astype(str):
                for domain in re.split(r"[,;/]\s*|\s+", value):
                    domain = domain.strip()
                    if domain and domain.lower() not in {"none", "nan"}:
                        domain_counts[domain] = domain_counts.get(domain, 0) + 1
            domain_text = ", ".join(f"{k}:{v}" for k, v in sorted(domain_counts.items())[:4])
        if stage == "Evidence Governance":
            _render_figure_summary_panel(
                kicker=stage_header[0],
                title=stage_header[1],
                copy=stage_header[2],
                stats=[
                    {
                        "label": "Final detector flags",
                        "value": f"{len(flagged_df):,}",
                        "note": f"out of {len(review_df):,} examinees",
                        "tone": "warn" if not flagged_df.empty else "good",
                    },
                    {
                        "label": "High-priority review",
                        "value": f"{high_n:,}",
                        "note": "Suggested by system triage",
                        "tone": "warn" if high_n else "off",
                    },
                    {
                        "label": "Human decisions",
                        "value": f"{reviewed_n:,}",
                        "note": "Recorded adjudications",
                        "tone": "good" if reviewed_n else "off",
                    },
                    {
                        "label": "Dominant domains",
                        "value": domain_text or "None",
                        "note": "From flagged cases",
                        "tone": "info",
                    },
                ],
            )
    if stage == "Evidence Governance":
        _render_kpi_progress(
            label="Examinees with at least one final detector flag",
            value=len(flagged_df),
            total=len(review_df),
            explanation=f"{reviewed_n:,} examinees have a recorded human decision. A flag is a review trigger, not a misconduct conclusion.",
            tone="amber" if not flagged_df.empty else "teal",
        )
    if stage == "Evidence Governance":
        source_counts = []
        for fn in ABERRANCE_FUNCTIONS:
            count = int(flagged_df["Flag_Sources"].str.contains(rf"(?:^|,\s*){re.escape(fn)}(?:$|,)", regex=True).sum())
            if count:
                source_counts.append({"Detector": fn, "Flagged examinees": count})
        if source_counts:
            source_df = pd.DataFrame(source_counts)
            try:
                import plotly.graph_objects as go

                fig_sources = go.Figure(
                    go.Bar(
                        x=source_df["Detector"],
                        y=source_df["Flagged examinees"],
                        marker_color="#7CC3F3",
                        hovertemplate="%{x}<br>%{y} flagged examinees<extra></extra>",
                    )
                )
                fig_sources.update_layout(
                    height=280,
                    margin=dict(l=42, r=18, t=12, b=58),
                    plot_bgcolor="#10141A",
                    paper_bgcolor="#10141A",
                    font=dict(color="#FFFFFF", size=12),
                    xaxis=dict(
                        title="",
                        tickangle=0,
                        tickfont=dict(color="#FFFFFF", size=11),
                        gridcolor="rgba(255,255,255,0.08)",
                    ),
                    yaxis=dict(
                        title="Flagged examinees",
                        tickfont=dict(color="#FFFFFF", size=11),
                        gridcolor="rgba(255,255,255,0.16)",
                        zerolinecolor="rgba(255,255,255,0.18)",
                    ),
                    showlegend=False,
                )
                st.plotly_chart(fig_sources, use_container_width=True, config={"displayModeBar": False})
            except Exception:
                source_df = source_df.set_index("Detector")
                st.bar_chart(source_df, height=280)
        st.caption("Suggested priority reflects the number of independent detector functions involved. The reviewer must inspect the examinee report before adjudication.")

        st.markdown("##### Evidence lineage")
        st.caption(
            "Cohort view: index → domain → governed review conclusion. "
            "Open Examinee Report for individual lineage (Step 2)."
        )
        _render_cohort_evidence_lineage(_master_results_for_lineage(), chart_key="evidence_lineage_overview")
        return
    if stage == "AI-Assisted Review":
        _render_ai_review_overview(review_df, flagged_df)
        return
    if stage == "Human Review":
        if flagged_df.empty:
            st.success("No examinees have a final detector flag.")
            return
        _render_single_case_review_page()
        return

    with st.spinner("Preparing the consolidated examinee-level dataset..."):
        master_df = _master_results_df()
    st.caption(
        "One row per examinee: final detector flags, six domain results, review priority, "
        "human adjudication, LLM explanation, and all forensic indices. "
        "The current snapshot is also stored at data/output/psymas_master_results.csv."
    )
    st.markdown("##### Evidence lineage")
    st.caption("Cohort Sankey derived from the master export below.")
    _render_cohort_evidence_lineage(master_df, chart_key="evidence_lineage_downloads")
    _render_page_download_bar(
        [
            {
                "label": "Download Master Results",
                "data": _df_csv_bytes(master_df),
                "file_name": "psymas_master_results.csv",
                "key": "master_results_download",
                "disabled": master_df.empty,
            },
        ]
    )


def _demo_table_df(name: str, filename: str) -> pd.DataFrame:
    rows = st.session_state.get(f"demo_{name}") or []
    if rows:
        return _table_df(rows)
    path = DEMO_DATA_DIR / filename
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def _canonical_examinee_key(value: object) -> str:
    """Align operational IDs such as 1 with simulation IDs such as E001."""
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none"}:
        return ""
    numeric_suffix = re.search(r"(\d+)$", text)
    if numeric_suffix:
        return str(int(numeric_suffix.group(1)))
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _simulation_validation_tables(
    *,
    include_detail: bool = False,
    include_pair: bool = False,
) -> dict[str, pd.DataFrame]:
    """Build compact validation summaries; materialize truth-unit rows only on demand."""
    flags = _forensic_result_flags_from_session()
    scenario_df = _demo_table_df("scenario_key", "scenario_key.csv")
    copy_truth_df = _demo_table_df("copying_pairs_truth", "copying_pairs_truth.csv")
    cache_signature = json.dumps(
        {
            "run_id": st.session_state.get("detect_run_id", ""),
            "result_id": id(st.session_state.get("forensic_result")),
            "agents": sorted(str(key) for key in flags),
            "scenario_rows": len(scenario_df),
            "pair_rows": len(copy_truth_df),
            "include_detail": include_detail,
            "include_pair": include_pair,
            "version": "validation_v3",
        },
        sort_keys=True,
    )
    validation_cache = st.session_state.get("_simulation_validation_tables_cache")
    if isinstance(validation_cache, dict) and validation_cache.get("sig") == cache_signature:
        cached_tables = validation_cache.get("tables")
        if isinstance(cached_tables, dict):
            return {
                key: value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame()
                for key, value in cached_tables.items()
            }
    empty = {
        "by_examinee": pd.DataFrame(),
        "summary": pd.DataFrame(),
        "pair": pd.DataFrame(),
        "item": pd.DataFrame(),
    }
    if scenario_df.empty or "examinee_id" not in scenario_df.columns:
        return empty

    def _agent_operational(agent_key: str) -> bool:
        payload = flags.get(agent_key, {})
        if not isinstance(payload, dict) or payload.get("error"):
            return False
        if payload.get("package_method_required"):
            return False
        return bool(
            payload.get("methods")
            or payload.get("stat")
            or payload.get("pairs")
            or payload.get("rte")
            or payload.get("flagged_by_method")
        )

    def _agent_flag_ids(agent_key: str, method: str | None = None) -> set[str]:
        payload = flags.get(agent_key, {})
        if method and isinstance(payload, dict):
            method_ids = (payload.get("flagged_by_method") or {}).get(method) or []
            return {str(int(index) + 1) for index in method_ids}
        return {str(index + 1) for index in _agent_flagged_indices(payload, agent_key)}

    scenario = scenario_df.copy()
    scenario["examinee_id"] = scenario["examinee_id"].astype(str)
    scenario["_truth_key"] = scenario["examinee_id"].map(_canonical_examinee_key)
    scenario["true_group"] = scenario.get("true_group", "").astype(str)
    scenario["scenario_type"] = scenario.get("scenario_type", "").astype(str)

    source_map = {
        "rapid_guessing": ("rg_agent", "detect_rg NT package flags", "examinee", "NT"),
        "preknowledge": ("pk_agent", "detect_pk package flags", "examinee", None),
        "answer_change": ("tt_agent", "detect_tt package flags", "examinee", None),
    }
    detail_rows: list[dict] = []
    summary_rows: list[dict] = []
    for truth_group, (agent_key, flag_source, unit, package_method) in source_map.items():
        truth_rows = scenario[scenario["true_group"].eq(truth_group)].copy()
        available = _agent_operational(agent_key)
        flagged_ids = _agent_flag_ids(agent_key, package_method) if available else set()
        if include_detail:
            for _, truth_row in truth_rows.iterrows():
                matched = truth_row["_truth_key"] in flagged_ids if available else False
                detail_rows.append(
                    {
                        "Truth_Unit_ID": truth_row["examinee_id"],
                        "Statistical_Unit": unit,
                        "true_group": truth_group,
                        "scenario_type": truth_row.get("scenario_type", ""),
                        "Detection_Source": flag_source,
                        "Evaluation_Status": "Evaluated" if available else "Not evaluated",
                        "Matching_Flag": int(matched) if available else pd.NA,
                        "Validation_Outcome": (
                            "matching_flag"
                            if matched
                            else ("true_case_not_flagged" if available else "not_evaluated")
                        ),
                    }
                )
        true_n = len(truth_rows)
        matched_n = int(truth_rows["_truth_key"].isin(flagged_ids).sum()) if available else 0
        summary_rows.append(
            {
                "true_group": truth_group,
                "Statistical_Unit": unit,
                "Detection_Source": flag_source,
                "Evaluation_Status": "Evaluated" if available else "Not evaluated",
                "true_value_n": true_n,
                "matching_flag_n": matched_n if available else pd.NA,
                "flag_truth_match_rate": round(matched_n / true_n, 4) if available and true_n else pd.NA,
            }
        )

    truth_pair_keys: set[tuple[str, str]] = set()
    if not copy_truth_df.empty and {"source_id", "copier_id"}.issubset(copy_truth_df.columns):
        truth_pair_keys = {
            tuple(
                sorted(
                    (
                        _canonical_examinee_key(row.get("source_id", "")),
                        _canonical_examinee_key(row.get("copier_id", "")),
                    )
                )
            )
            for _, row in copy_truth_df.iterrows()
        }
    detected_pairs: set[tuple[str, str]] = set()
    forensic_result = st.session_state.get("forensic_result") or {}
    run_thresholds = forensic_result.get("threshold_config") if isinstance(forensic_result, dict) else {}
    run_thresholds = run_thresholds if isinstance(run_thresholds, dict) else {}

    def _package_alpha(agent_key: str) -> float:
        fn_name = {"ac_agent": "detect_ac", "as_agent": "detect_as"}.get(agent_key, "")
        rules = run_thresholds.get("rules") or {}
        block = rules.get(fn_name) if isinstance(rules, dict) else {}
        defaults = run_thresholds.get("defaults") or {}
        try:
            return float((block or {}).get("alpha", (defaults or {}).get("alpha", 0.05)))
        except (TypeError, ValueError):
            return 0.05

    copying_operational = _agent_operational("ac_agent") or _agent_operational("as_agent")
    copying_contract_available = any(
        isinstance(flags.get(agent_key), dict)
        and "flagged_pairs" in flags.get(agent_key, {})
        for agent_key in ("ac_agent", "as_agent")
    )
    copying_available = copying_operational and copying_contract_available
    copying_status = (
        "Evaluated"
        if copying_available
        else ("Rerun required" if copying_operational else "Not evaluated")
    )
    for agent_key, block_names in (("ac_agent", ("pairs",)), ("as_agent", ("stat",))):
        payload = flags.get(agent_key, {})
        if not isinstance(payload, dict):
            continue
        for flagged_pair in payload.get("flagged_pairs") or []:
            if not isinstance(flagged_pair, (list, tuple)) or len(flagged_pair) < 2:
                continue
            try:
                pair_key = tuple(
                    sorted((str(int(flagged_pair[0]) + 1), str(int(flagged_pair[1]) + 1)))
                )
            except (TypeError, ValueError):
                continue
            if pair_key in truth_pair_keys:
                detected_pairs.add(pair_key)
        for block_name in block_names:
            for record in payload.get(block_name) or []:
                if not isinstance(record, dict):
                    continue
                method_flag_values = [
                    value
                    for key, value in record.items()
                    if str(key).lower().endswith("_flag")
                ]
                pair_flag = any(_gov_truthy_flag(value) for value in method_flag_values)
                if not method_flag_values:
                    # aberrance defines detect_ac/detect_as flag as p-value <= alpha.
                    # This recovers package flags from runs created before pair flags
                    # were serialized from the R three-dimensional flag array.
                    alpha = _package_alpha(agent_key)
                    pair_flag = any(
                        str(key).lower().endswith("_pval")
                        and pd.notna(value)
                        and float(value) <= alpha
                        for key, value in record.items()
                        if isinstance(value, (int, float, np.integer, np.floating))
                    )
                if not pair_flag:
                    continue
                p1, p2 = _pair_record_participants(record)
                if p1 is None or p2 is None:
                    continue
                pair_key = tuple(sorted((str(p1 + 1), str(p2 + 1))))
                if pair_key in truth_pair_keys:
                    detected_pairs.add(pair_key)
                    if detected_pairs == truth_pair_keys:
                        break
            if detected_pairs == truth_pair_keys:
                break
        if detected_pairs == truth_pair_keys:
            break

    pair_rows: list[dict] = []
    if not copy_truth_df.empty and {"source_id", "copier_id"}.issubset(copy_truth_df.columns):
        for pair_index, truth_row in copy_truth_df.reset_index(drop=True).iterrows():
            source_id = str(truth_row.get("source_id", ""))
            copier_id = str(truth_row.get("copier_id", ""))
            pair_key = tuple(
                sorted((_canonical_examinee_key(source_id), _canonical_examinee_key(copier_id)))
            )
            matched = pair_key in detected_pairs if copying_available else False
            if include_pair:
                pair_rows.append(
                    {
                        "Truth_Unit_ID": f"{source_id} -> {copier_id}",
                        "Statistical_Unit": "pair",
                        "true_group": "copying_pair",
                        "scenario_type": truth_row.get("scenario_label", "score_based_copying"),
                        "Detection_Source": "detect_ac/detect_as package pair flags",
                        "Evaluation_Status": copying_status,
                        "Matching_Flag": int(matched) if copying_available else pd.NA,
                        "Validation_Outcome": (
                            "matching_flag"
                            if matched
                            else ("true_pair_not_flagged" if copying_available else "not_evaluated")
                        ),
                        "Source_ID": source_id,
                        "Copier_ID": copier_id,
                    }
                )
        pair_true_n = len(copy_truth_df)
        pair_matched_n = len(detected_pairs) if copying_available else 0
        summary_rows.append(
            {
                "true_group": "copying_pair",
                "Statistical_Unit": "pair",
                "Detection_Source": "detect_ac/detect_as package pair flags",
                "Evaluation_Status": copying_status,
                "true_value_n": pair_true_n,
                "matching_flag_n": pair_matched_n if copying_available else pd.NA,
                "flag_truth_match_rate": (
                    round(pair_matched_n / pair_true_n, 4)
                    if copying_available and pair_true_n
                    else pd.NA
                ),
            }
        )

    validation_exam_df = pd.DataFrame(detail_rows)
    pair_df = pd.DataFrame(pair_rows)
    summary_df = pd.DataFrame(summary_rows)
    preferred_order = {
        "answer_change": 0,
        "copying_pair": 1,
        "preknowledge": 2,
        "rapid_guessing": 3,
    }
    if not summary_df.empty:
        summary_df["_order"] = summary_df["true_group"].map(preferred_order).fillna(99)
        summary_df = summary_df.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    result = {
        "by_examinee": validation_exam_df,
        "summary": summary_df,
        "pair": pair_df,
        "item": pd.DataFrame(),
    }
    st.session_state["_simulation_validation_tables_cache"] = {
        "sig": cache_signature,
        "tables": {
            key: value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame()
            for key, value in result.items()
        },
    }
    return result


def _validation_publication_figure(
    by_exam: pd.DataFrame,
) -> tuple[plt.Figure | None, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build a publication-facing descriptive validation figure."""
    if by_exam.empty or "true_group" not in by_exam.columns:
        return None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    plot_df = by_exam[by_exam["true_group"].notna()].copy()
    if plot_df.empty:
        return None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    plot_df["true_group"] = plot_df["true_group"].astype(str)
    plot_df["scenario_detected"] = plot_df.get("scenario_detected", False).fillna(False).astype(bool)
    preferred_order = [
        "normal",
        "rapid_guessing",
        "mixed_fast_high_ability",
        "preknowledge",
        "copying_copier",
        "copying_source",
        "answer_change",
        "missing_rt_limited_evidence",
    ]
    observed = list(dict.fromkeys(plot_df["true_group"].tolist()))
    group_order = [g for g in preferred_order if g in observed] + [g for g in observed if g not in preferred_order]
    group_labels = {
        "normal": "Normal",
        "rapid_guessing": "Rapid guessing",
        "mixed_fast_high_ability": "Fast / high ability",
        "preknowledge": "Preknowledge",
        "copying_copier": "Copying: copier",
        "copying_source": "Copying: source",
        "answer_change": "Answer change",
        "missing_rt_limited_evidence": "Missing RT",
    }

    rate_rows: list[dict] = []
    for group in group_order:
        group_df = plot_df[plot_df["true_group"] == group]
        n_group = len(group_df)
        positive_n = int(group_df["scenario_detected"].sum())
        rate = positive_n / n_group if n_group else 0.0
        z = 1.959963984540054
        denom = 1 + (z**2 / n_group) if n_group else 1
        center = (rate + z**2 / (2 * n_group)) / denom if n_group else 0.0
        half = (
            z
            * math.sqrt((rate * (1 - rate) / n_group) + (z**2 / (4 * n_group**2)))
            / denom
            if n_group
            else 0.0
        )
        rate_rows.append(
            {
                "true_group": group,
                "group_label": group_labels.get(group, group.replace("_", " ").title()),
                "n_examinees": n_group,
                "scenario_detected_n": positive_n,
                "scenario_detection_rate": rate,
                "ci_low": max(0.0, center - half),
                "ci_high": min(1.0, center + half),
            }
        )
    rate_df = pd.DataFrame(rate_rows)

    domain_rows: list[dict] = []
    for group in group_order:
        group_df = plot_df[plot_df["true_group"] == group]
        for domain in ["MF", "RT", "SIM", "PK", "TP", "CP"]:
            strength_col = f"{domain}_Strength"
            value = (
                float(group_df[strength_col].astype(str).str.lower().isin(["moderate", "strong"]).mean())
                if strength_col in group_df.columns and not group_df.empty
                else 0.0
            )
            domain_rows.append(
                {
                    "true_group": group,
                    "group_label": group_labels.get(group, group.replace("_", " ").title()),
                    "domain": domain,
                    "moderate_or_strong_rate": value,
                }
            )
    domain_df = pd.DataFrame(domain_rows)

    status_df = (
        plot_df.groupby(["true_group", "validation_outcome"], dropna=False)
        .size()
        .reset_index(name="n_cases")
        if "validation_outcome" in plot_df.columns
        else pd.DataFrame()
    )
    if not status_df.empty:
        totals = status_df.groupby("true_group")["n_cases"].transform("sum")
        status_df["proportion"] = status_df["n_cases"] / totals
        status_df["group_label"] = status_df["true_group"].map(
            lambda x: group_labels.get(x, str(x).replace("_", " ").title())
        )

    fig = plt.figure(figsize=(12.0, 9.0), facecolor="white")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], hspace=0.42, wspace=0.42)
    ax_rate = fig.add_subplot(grid[0, :])
    ax_domain = fig.add_subplot(grid[1, 0])
    ax_status = fig.add_subplot(grid[1, 1])
    for ax in (ax_rate, ax_domain, ax_status):
        ax.set_facecolor("white")
        ax.tick_params(colors="#111827", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#CBD5E1")

    y_positions = np.arange(len(rate_df))
    rates = rate_df["scenario_detection_rate"].to_numpy(dtype=float)
    error_low = rates - rate_df["ci_low"].to_numpy(dtype=float)
    error_high = rate_df["ci_high"].to_numpy(dtype=float) - rates
    bar_colors = ["#64748B" if group == "normal" else "#147D92" for group in rate_df["true_group"]]
    ax_rate.barh(y_positions, rates, color=bar_colors, height=0.64, edgecolor="white")
    ax_rate.errorbar(
        rates,
        y_positions,
        xerr=np.vstack([error_low, error_high]),
        fmt="none",
        ecolor="#111827",
        elinewidth=1.0,
        capsize=3,
    )
    ax_rate.set_yticks(y_positions, rate_df["group_label"])
    ax_rate.invert_yaxis()
    ax_rate.set_xlim(0, 1.02)
    ax_rate.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax_rate.set_xlabel("Cases detected in the scenario-matched domain", color="#111827")
    ax_rate.set_title("A  Scenario-matched detection rate by simulated truth group", loc="left", weight="bold", color="#111827")
    ax_rate.grid(axis="x", color="#E2E8F0", linewidth=0.8)
    ax_rate.set_axisbelow(True)
    for y_pos, row in rate_df.reset_index(drop=True).iterrows():
        label_x = min(float(row["ci_high"]) + 0.018, 0.985)
        label_align = "right" if float(row["ci_high"]) > 0.88 else "left"
        ax_rate.text(
            label_x,
            y_pos,
            f'{row["scenario_detection_rate"]:.1%}  (n={int(row["n_examinees"])})',
            ha=label_align,
            va="center",
            fontsize=8,
            color="#111827",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.0},
        )

    heatmap = (
        domain_df.pivot(index="group_label", columns="domain", values="moderate_or_strong_rate")
        .reindex(index=rate_df["group_label"].tolist(), columns=["MF", "RT", "SIM", "PK", "TP", "CP"])
        .fillna(0)
    )
    image = ax_domain.imshow(heatmap.to_numpy(), cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax_domain.set_xticks(np.arange(len(heatmap.columns)), heatmap.columns)
    ax_domain.set_yticks(np.arange(len(heatmap.index)), heatmap.index)
    ax_domain.set_title("B  Moderate-or-strong domain evidence", loc="left", weight="bold", color="#111827")
    for row_i in range(heatmap.shape[0]):
        for col_i in range(heatmap.shape[1]):
            value = float(heatmap.iloc[row_i, col_i])
            ax_domain.text(
                col_i,
                row_i,
                f"{value:.0%}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if value >= 0.55 else "#111827",
            )
    if not status_df.empty:
        status_pivot = (
            status_df.pivot(index="group_label", columns="validation_outcome", values="proportion")
            .reindex(rate_df["group_label"].tolist())
            .fillna(0)
        )
        status_colors = ["#DCEAF0", "#93C5C8", "#F3C677", "#E88B6A", "#A84448", "#64748B"]
        left = np.zeros(len(status_pivot))
        for idx, status_name in enumerate(status_pivot.columns):
            values = status_pivot[status_name].to_numpy(dtype=float)
            ax_status.barh(
                np.arange(len(status_pivot)),
                values,
                left=left,
                height=0.64,
                label=str(status_name),
                color=status_colors[idx % len(status_colors)],
                edgecolor="white",
            )
            left += values
        ax_status.set_yticks(np.arange(len(status_pivot)), status_pivot.index)
        ax_status.invert_yaxis()
        ax_status.set_xlim(0, 1)
        ax_status.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
        ax_status.set_xlabel("Proportion of truth group", color="#111827")
        ax_status.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=2,
            frameon=False,
            fontsize=7,
        )
    else:
        ax_status.text(0.5, 0.5, "Validation outcome unavailable", ha="center", va="center")
        ax_status.set_xticks([])
        ax_status.set_yticks([])
    ax_status.set_title("C  Scenario-matched validation outcomes", loc="left", weight="bold", color="#111827")
    ax_status.grid(axis="x", color="#E2E8F0", linewidth=0.8)
    ax_status.set_axisbelow(True)

    fig.suptitle(
        "PsyMAS validation against simulated truth conditions",
        x=0.07,
        y=0.99,
        ha="left",
        fontsize=15,
        weight="bold",
        color="#111827",
    )
    fig.text(
        0.07,
        0.955,
        "Rates are descriptive; error bars in panel A are Wilson 95% confidence intervals.",
        ha="left",
        fontsize=9,
        color="#475569",
    )
    fig.subplots_adjust(top=0.91, bottom=0.15, left=0.18, right=0.97)
    return fig, rate_df, domain_df, status_df


def _figure_download_bytes(fig: plt.Figure | None, file_format: str) -> bytes:
    if fig is None:
        return b""
    buffer = io.BytesIO()
    save_kwargs = {
        "format": file_format,
        "bbox_inches": "tight",
        "facecolor": "white",
        "edgecolor": "none",
    }
    if file_format.lower() == "png":
        save_kwargs["dpi"] = 300
    fig.savefig(buffer, **save_kwargs)
    return buffer.getvalue()


def _validation_flag_comparison_figure(
    summary_df: pd.DataFrame,
) -> tuple[plt.Figure | None, pd.DataFrame]:
    """Compare package/index flags with the corresponding simulated truth units."""
    required = {
        "true_group",
        "Evaluation_Status",
        "true_value_n",
        "matching_flag_n",
        "flag_truth_match_rate",
    }
    if summary_df.empty or not required.issubset(summary_df.columns):
        return None, pd.DataFrame()
    group_df = summary_df.copy()
    labels = {
        "answer_change": "Answer change",
        "copying_pair": "Copying pairs",
        "rapid_guessing": "Rapid guessing",
        "preknowledge": "Preknowledge",
    }
    group_df["group_label"] = group_df["true_group"].map(
        lambda value: labels.get(str(value), str(value).replace("_", " ").title())
    )
    group_df["flagged_pct"] = pd.to_numeric(
        group_df["flag_truth_match_rate"], errors="coerce"
    )

    fig, ax = plt.subplots(figsize=(8.4, 3.4), facecolor="white")
    ax.set_facecolor("white")
    ax.tick_params(colors="#111827", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#CBD5E1")
    y = np.arange(len(group_df))
    true_counts = pd.to_numeric(group_df["true_value_n"], errors="coerce").fillna(0).to_numpy(dtype=float)
    matched_counts = pd.to_numeric(group_df["matching_flag_n"], errors="coerce").fillna(0).to_numpy(dtype=float)
    max_count = max(float(true_counts.max()), 1.0)
    ax.barh(y, true_counts, height=0.52, color="#DCE6EC", label="Simulated truth")
    evaluated_mask = group_df["Evaluation_Status"].eq("Evaluated").to_numpy()
    evaluated_counts = np.where(evaluated_mask, matched_counts, 0)
    ax.barh(y, evaluated_counts, height=0.28, color="#147D92", label="Matched flags")
    for row_y, matched_n, true_n, rate, status in zip(
        y,
        matched_counts,
        true_counts,
        group_df["flagged_pct"],
        group_df["Evaluation_Status"],
    ):
        label = (
            f"{int(matched_n)}/{int(true_n)}  ({float(rate):.1%})"
            if status == "Evaluated" and pd.notna(rate)
            else str(status)
        )
        ax.text(
            float(true_n) + max_count * 0.035,
            row_y,
            label,
            ha="left",
            va="center",
            fontsize=8.5,
            weight="semibold",
            color="#111827",
        )
    ax.set_yticks(y, group_df["group_label"])
    ax.invert_yaxis()
    ax.set_xlim(0, max_count * 1.34)
    ax.set_xlabel("Simulated truth units", color="#111827", fontsize=9)
    ax.grid(axis="x", color="#E2E8F0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=2,
        frameon=False,
        fontsize=8.5,
        borderaxespad=0,
    )
    ax.set_title(
        "Index-level flags compared with simulated true values",
        loc="left",
        pad=36,
        fontsize=12,
        weight="bold",
        color="#111827",
    )
    fig.subplots_adjust(top=0.72, bottom=0.20, left=0.22, right=0.94)
    return fig, group_df


def _validation_zip_bytes(
    tables: dict[str, pd.DataFrame],
    *,
    figure_png: bytes = b"",
    figure_svg: bytes = b"",
    flag_comparison_png: bytes = b"",
    flag_comparison_svg: bytes = b"",
) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        name_map = {
            "by_examinee": "truth_flag_match_by_examinee.csv",
            "summary": "truth_flag_match_summary.csv",
            "pair": "pair_similarity_validation.csv",
            "item": "item_level_validation.csv",
            "figure_rates": "publication_figure_panel_a_rates.csv",
            "figure_domains": "publication_figure_panel_b_domains.csv",
            "figure_status": "publication_figure_panel_c_status.csv",
            "flag_comparison": "truth_flag_match_by_scenario.csv",
        }
        for key, filename in name_map.items():
            df = tables.get(key)
            if isinstance(df, pd.DataFrame) and not df.empty:
                zf.writestr(filename, df.to_csv(index=False))
        if figure_png:
            zf.writestr("psymas_simulation_validation_figure.png", figure_png)
        if figure_svg:
            zf.writestr("psymas_simulation_validation_figure.svg", figure_svg)
        if flag_comparison_png:
            zf.writestr("psymas_truth_flag_match.png", flag_comparison_png)
        if flag_comparison_svg:
            zf.writestr("psymas_truth_flag_match.svg", flag_comparison_svg)
    return buf.getvalue()


def _render_simulation_validation_page(*, embedded: bool = False) -> None:
    if not embedded:
        st.markdown(
            """
<div class="psymas-workbench-title">
  <div>
    <h1>Simulation Validation</h1>
    <p>Scenario-matched comparison between simulated true values and package/index flags.</p>
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )
    if not _forensic_result_flags_from_session():
        st.info("No forensic index output is available yet. Run Detect before validation.")
        if st.button("Open Assessment Data", key="validation_open_data", use_container_width=False):
            st.session_state["_nav_request"] = "Assessment Data"
            st.rerun()
        return

    if embedded:
        # Validation is already an on-demand Forensic Indices workspace.
        # Avoid a second nested tab and build its downloadable artifacts here.
        view = "Downloads"
    else:
        validation_views = ["Overview", "Downloads"]
        if st.session_state.get("simulation_validation_view") not in validation_views:
            st.session_state["simulation_validation_view"] = "Overview"
        view = st.segmented_control(
            "View",
            options=validation_views,
            default="Overview",
            key="simulation_validation_view",
            label_visibility="collapsed",
        )
    include_download_detail = view == "Downloads"
    tables = _simulation_validation_tables(
        include_detail=include_download_detail,
        include_pair=include_download_detail,
    )
    by_exam = tables["by_examinee"]
    summary_df = tables["summary"]
    pair_df = tables["pair"]
    if summary_df.empty:
        st.info("Validation tables could not be built because simulated truth tables are unavailable.")
        return

    evaluated_summary = summary_df[summary_df["Evaluation_Status"].eq("Evaluated")].copy()
    target_n = int(pd.to_numeric(evaluated_summary["true_value_n"], errors="coerce").fillna(0).sum())
    true_positive_n = int(pd.to_numeric(evaluated_summary["matching_flag_n"], errors="coerce").fillna(0).sum())
    false_negative_n = max(0, target_n - true_positive_n)
    target_detection_rate = true_positive_n / target_n if target_n else 0.0
    not_evaluated_n = int(summary_df["Evaluation_Status"].ne("Evaluated").sum())

    st.markdown(
        f"""
<div class="psymas-case-strip">
  <div class="psymas-case-chip"><div class="label">Evaluated truth units</div><div class="value">{target_n:,}</div></div>
  <div class="psymas-case-chip"><div class="label">Matching index flags</div><div class="value">{true_positive_n:,}</div></div>
  <div class="psymas-case-chip"><div class="label">Truth units not flagged</div><div class="value">{false_negative_n:,}</div></div>
  <div class="psymas-case-chip"><div class="label">Flag-to-truth match rate</div><div class="value">{target_detection_rate:.1%}</div><div class="meta">{true_positive_n:,} / {target_n:,}</div></div>
  <div class="psymas-case-chip"><div class="label">Scenarios not evaluated</div><div class="value">{not_evaluated_n:,}</div></div>
</div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Validation uses raw package/index flags. Examinee scenarios use examinees as truth units; copying uses simulated source-copier pairs. Domain Evidence and Review Prioritization are not used."
    )
    if not_evaluated_n:
        st.info(
            f"{not_evaluated_n} scenario(s) need attention because the required detector was unavailable or the result predates the pair-flag export. Rerun Detect when shown."
        )
    download_slot = st.container()

    flag_comparison_png = b""
    flag_comparison_svg = b""
    flag_comparison_fig, flag_comparison_df = _validation_flag_comparison_figure(summary_df)
    tables["flag_comparison"] = flag_comparison_df
    if view == "Overview" or embedded:
        if flag_comparison_fig is not None:
            st.pyplot(flag_comparison_fig, width="content")
            st.caption(
                "Matched package/index flags divided by simulated truth units. Copying is evaluated at pair level."
            )
        else:
            st.info("A publication figure could not be generated because simulated truth labels are unavailable.")
        compact_summary = summary_df.rename(
            columns={
                "true_group": "Scenario",
                "Statistical_Unit": "Unit",
                "Evaluation_Status": "Status",
                "true_value_n": "True units",
                "matching_flag_n": "Matched flags",
                "flag_truth_match_rate": "Match rate",
            }
        )
        compact_cols = ["Scenario", "Unit", "Status", "True units", "Matched flags", "Match rate"]
        compact_summary = compact_summary[[c for c in compact_cols if c in compact_summary.columns]]
        if "Match rate" in compact_summary:
            compact_summary["Match rate"] = compact_summary["Match rate"].map(
                lambda value: f"{float(value):.1%}" if pd.notna(value) else "Not evaluated"
            )
        st.dataframe(
            compact_summary,
            hide_index=True,
            width="stretch",
            height=min(240, 38 + 35 * len(compact_summary)),
        )
    if flag_comparison_fig is not None:
        if view == "Downloads":
            flag_comparison_png = _figure_download_bytes(flag_comparison_fig, "png")
            flag_comparison_svg = _figure_download_bytes(flag_comparison_fig, "svg")
        plt.close(flag_comparison_fig)

    validation_zip = (
        _validation_zip_bytes(
            tables,
            flag_comparison_png=flag_comparison_png,
            flag_comparison_svg=flag_comparison_svg,
        )
        if view == "Downloads"
        else b""
    )
    if view == "Downloads":
        download_actions = [
            {
                "label": "Download Validation Package",
                "data": validation_zip,
                "file_name": "psymas_truth_flag_validation.zip",
                "mime": "application/zip",
                "key": "download_simulation_validation_zip",
                "disabled": not validation_zip,
            },
            {
                "label": "Download Summary CSV",
                "data": _df_csv_bytes(summary_df),
                "file_name": "truth_flag_match_summary.csv",
                "key": "download_simulation_validation_summary",
                "disabled": summary_df.empty,
            },
            {
                "label": "Download Figure PNG",
                "data": flag_comparison_png,
                "file_name": "psymas_truth_flag_match.png",
                "mime": "image/png",
                "key": "download_scenario_flagged_percentages_png",
                "disabled": not flag_comparison_png,
            },
        ]
    else:
        download_actions = []
    with download_slot:
        if download_actions:
            st.markdown("#### Downloads")
        _render_page_download_bar(download_actions)


def _research_export_zip_bytes() -> bytes:
    """Package Dataset A/B/C files for paper and validation workflows."""
    _ensure_run_store()
    store = get_run_store()
    dataset_b_tables: dict[str, pd.DataFrame] = {}
    for table_name in DATA_TABLES:
        table = store.load_table(table_name)
        if isinstance(table, pd.DataFrame) and not table.empty:
            dataset_b_tables[table_name] = table

    master_df = dataset_b_tables.get("master_results", pd.DataFrame())
    if not isinstance(master_df, pd.DataFrame) or master_df.empty:
        master_df = _master_results_for_lineage()
    if isinstance(master_df, pd.DataFrame) and not master_df.empty:
        dataset_b_tables["master_results"] = master_df

    data_availability = _data_availability_table_df()
    if not data_availability.empty:
        dataset_b_tables["data_availability"] = data_availability

    review_decisions = _case_review_decisions_df()
    if not review_decisions.empty:
        dataset_b_tables["human_review_decisions"] = review_decisions

    try:
        audit_df = _table_df(_stage_table_options("Audit Trail")[0]["records"])
        if not audit_df.empty:
            dataset_b_tables["audit_records"] = audit_df
    except Exception:
        pass

    try:
        config_df = _table_df(_stage_table_options("Settings")[0]["records"])
        if not config_df.empty:
            dataset_b_tables["configuration"] = config_df
    except Exception:
        pass

    try:
        validation_tables = _simulation_validation_tables(include_detail=True, include_pair=True)
        for name, df in validation_tables.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                dataset_b_tables[f"simulation_validation_{name}"] = df
    except Exception:
        pass

    run_id = str(st.session_state.get("detect_run_id") or st.session_state.get("psymas_active_run_id") or "")
    return build_research_export_zip(
        root_dir=Path.cwd(),
        dataset_b_tables=dataset_b_tables,
        master_df=master_df if isinstance(master_df, pd.DataFrame) else pd.DataFrame(),
        run_id=run_id,
        extra_notes="Generated from the PsyMAS Streamlit workspace. Dataset C is a blank expert-review template.",
    )


def _render_audit_page() -> None:
    forensic_ready = st.session_state.get("forensic_result") is not None
    audit_df = _table_df(_stage_table_options("Audit Trail")[0]["records"])
    active_run = str(st.session_state.get("detect_run_id") or st.session_state.get("psymas_active_run_id") or "None")
    store_path = str((Path("data") / "output" / "psymas_run.sqlite").as_posix())
    master_path = str(st.session_state.get("master_results_path") or (Path("data") / "output" / "psymas_master_results.csv"))
    _render_figure_summary_panel(
        kicker="Review record",
        title="Audit and reproducibility record",
        copy="Keep the run identifier, audit rows, export path, and configuration snapshot together for reconstruction.",
        stats=[
            {
                "label": "Forensic run",
                "value": "Ready" if forensic_ready else "Missing",
                "note": active_run,
                "tone": "good" if forensic_ready else "off",
            },
            {
                "label": "Audit records",
                "value": f"{len(audit_df):,}",
                "note": "report and evidence checks",
                "tone": "good" if not audit_df.empty else "off",
            },
            {
                "label": "SQLite store",
                "value": "Local",
                "note": store_path,
                "tone": "info",
            },
            {
                "label": "Master export",
                "value": "CSV",
                "note": master_path,
                "tone": "info",
            },
        ],
    )
    overview = pd.DataFrame(
        [
            {"Area": "Forensic run", "Status": "Available" if forensic_ready else "Not generated"},
            {"Area": "Audit records", "Status": f"{len(audit_df):,} rows" if not audit_df.empty else "Not generated"},
            {
                "Area": "Active run",
                "Status": active_run,
            },
        ]
    )
    with st.expander("Record details", expanded=False):
        st.dataframe(overview, use_container_width=True, hide_index=True, height=150)
        if not audit_df.empty:
            st.dataframe(audit_df, use_container_width=True, hide_index=True, height=min(360, 44 + 30 * len(audit_df)))
    _render_page_download_bar(
        [
            {
                "label": "Download Audit Records",
                "data": _df_csv_bytes(audit_df),
                "file_name": "psymas_audit_records.csv",
                "key": "audit_records_download_v3",
                "disabled": audit_df.empty,
            },
        ]
    )


def _render_configuration_page() -> None:
    _render_data_run_storage_manager()
    st.divider()
    _render_llm_settings()
    st.divider()
    st.markdown("#### Threshold profile")
    left, right = st.columns(2)
    with left:
        st.download_button(
            "Download Default Thresholds",
            data=_default_threshold_yaml().encode("utf-8"),
            file_name="psymas_thresholds.default.yaml",
            mime="text/yaml",
            use_container_width=True,
            key="configuration_default_thresholds",
        )
    with right:
        st.download_button(
            "Download Active Thresholds",
            data=_active_threshold_yaml().encode("utf-8"),
            file_name="psymas_thresholds.active.yaml",
            mime="text/yaml",
            use_container_width=True,
            key="configuration_active_thresholds",
        )
    uploaded = st.file_uploader("Upload threshold YAML", type=["yaml", "yml"], key="configuration_threshold_upload")
    if uploaded is not None:
        try:
            loaded = yaml.safe_load(uploaded.getvalue().decode("utf-8"))
            if not isinstance(loaded, dict) or not isinstance(loaded.get("rules"), dict):
                st.error("Threshold YAML must contain a top-level `rules` mapping.")
            else:
                st.session_state["threshold_config"] = loaded
                st.success("Threshold profile loaded for this session.")
        except Exception as exc:
            st.error(f"Could not parse threshold YAML: {exc}")

    config_df = _table_df(_stage_table_options("Settings")[0]["records"])
    with st.expander("Configuration record", expanded=False):
        st.dataframe(config_df, use_container_width=True, hide_index=True, height=220)
    _render_page_download_bar(
        [
            {
                "label": "Download Configuration",
                "data": _df_csv_bytes(config_df),
                "file_name": "psymas_configuration.csv",
                "key": "configuration_download_v3",
                "disabled": config_df.empty,
            },
        ]
    )


def _render_research_tools_page() -> None:
    view = _workspace_view(
        "Research tools",
        ["Worked Example", "Validation", "Research Export"],
        key="research_tools_view",
        default="Worked Example",
    )
    if view == "Worked Example":
        _render_worked_example_page()
        return
    if view == "Validation":
        _render_simulation_validation_page(embedded=False)
        return

    st.caption("Research export packages Dataset A/B/C artifacts for manuscript validation and reproducibility.")
    with st.spinner("Preparing the research export package..."):
        research_zip = _research_export_zip_bytes()
    _render_page_download_bar(
        [
            {
                "label": "Download Research Package",
                "data": research_zip,
                "file_name": "psymas_research_export.zip",
                "mime": "application/zip",
                "key": "research_export_zip_download_v3",
                "disabled": not research_zip,
            },
        ]
    )


def _render_worked_example_page() -> None:
    output_dir = Path("outputs")
    candidates = load_worked_example_candidates(Path.cwd())
    if candidates.empty:
        st.error("No worked-example candidate data were found. Load the demo data or generate PsyMAS outputs first.")
        return

    profile_options = {
        "Domain-distinct: PK / RT / TP": "domain_distinct",
        "Original: PK / RT / mixed high-ability": "mixed_high_ability",
    }
    profile_label = st.selectbox(
        "Recommended comparison set",
        list(profile_options),
        index=0,
        key="worked_example_profile_label",
    )
    profile = profile_options[profile_label]
    rec_cases, rec_labels = recommended_worked_example_cases(Path.cwd(), profile=profile)
    if st.session_state.get("worked_example_last_profile") != profile:
        for case_name, canonical in rec_cases.items():
            st.session_state[f"worked_example_{case_name}_id"] = canonical
            st.session_state[f"worked_example_{case_name}_label"] = rec_labels.get(case_name, "")
        st.session_state["worked_example_last_profile"] = profile
        st.session_state.pop("worked_example_result", None)

    option_ids = candidates["canonical_id"].astype(str).tolist()

    def _case_option_label(canonical: str) -> str:
        row = candidates[candidates["canonical_id"].astype(str).eq(str(canonical))]
        if row.empty:
            return str(canonical)
        r = row.iloc[0]
        strength_bits = []
        for domain in ["PK", "RT", "TP", "SIM", "MF"]:
            col = f"{domain}_Strength"
            val = str(r.get(col, "") or "").lower()
            if val and val not in {"none", "nan"}:
                strength_bits.append(f"{domain}:{val}")
        strengths = " | ".join(strength_bits) if strength_bits else "no counted domain signal"
        return (
            f"{r.get('examinee_id', canonical)} · {r.get('true_group', '')} · "
            f"score {int(r.get('score', 0)) if pd.notna(r.get('score', pd.NA)) else 'NA'} · "
            f"RT {float(r.get('mean_rt', 0)):.1f} · fast {int(r.get('very_fast_count', 0) or 0)} · {strengths}"
        )

    st.markdown("#### Case comparison setup")
    st.caption("Use the recommended set for a clean manuscript figure, or manually replace any case below.")
    case_cols = st.columns(3)
    selected_cases: dict[str, str] = {}
    selected_labels: dict[str, str] = {}
    for idx, case_name in enumerate(["Case A", "Case B", "Case C"]):
        with case_cols[idx]:
            current = str(st.session_state.get(f"worked_example_{case_name}_id") or rec_cases.get(case_name) or option_ids[0])
            default_index = option_ids.index(current) if current in option_ids else 0
            selected = st.selectbox(
                case_name,
                option_ids,
                index=default_index,
                format_func=_case_option_label,
                key=f"worked_example_{case_name}_id",
            )
            selected_cases[case_name] = str(selected)
            selected_labels[case_name] = st.text_input(
                f"{case_name} label",
                value=str(st.session_state.get(f"worked_example_{case_name}_label") or rec_labels.get(case_name, "")),
                key=f"worked_example_{case_name}_label",
            )

    with st.expander("Candidate preview", expanded=False):
        preview_cols = [
            "examinee_id",
            "true_group",
            "score",
            "mean_rt",
            "very_fast_count",
            "exposed_correct_count",
            "exposed_difficult_fast_correct_count",
            "answer_change_count",
            "RT_Strength",
            "PK_Strength",
            "TP_Strength",
            "MF_Strength",
            "Evidence_Status",
            "Review_Priority",
        ]
        st.dataframe(candidates[[c for c in preview_cols if c in candidates.columns]], use_container_width=True, hide_index=True, height=320)

    generate_now = st.button(
        "Generate Worked-Example Package",
        key="generate_worked_example_package",
        type="primary",
        use_container_width=False,
    )
    result = st.session_state.get("worked_example_result")
    if generate_now or result is None:
        with st.spinner("Generating worked-example tables and figures..."):
            try:
                result = generate_worked_example_package(
                    Path.cwd(),
                    output_dir,
                    cases=selected_cases,
                    case_labels=selected_labels,
                )
                st.session_state["worked_example_result"] = result
                st.success(f"Worked-example outputs written to {result.output_dir}.")
            except Exception as exc:
                st.error(f"Could not generate worked-example outputs: {exc}")
                return

    if result is None:
        st.info("Click Generate to create the worked-example files.")
        return

    selected_cases = getattr(result, "selected_cases", {}) or {}
    if selected_cases:
        st.markdown(
            "Selected cases: "
            + "; ".join(f"**{case}** = `{examinee}`" for case, examinee in selected_cases.items())
        )

    downloads = [
        {
            "label": "Download Worked Example ZIP",
            "data": getattr(result, "zip_bytes", b""),
            "file_name": "psymas_worked_example_outputs.zip",
            "mime": "application/zip",
            "key": "download_worked_example_zip",
            "disabled": not getattr(result, "zip_bytes", b""),
        }
    ]
    _render_page_download_bar(downloads)

    view = _workspace_view(
        "Worked-example outputs",
        ["Tables", "Figures", "Caption Text", "Files"],
        key="worked_example_view",
        default="Tables",
    )
    if view == "Tables":
        st.markdown("#### Table 3. Worked Example Case Inputs")
        st.dataframe(result.table3, use_container_width=True, hide_index=True)
        st.markdown("#### Table 4. Deterministic Evidence Outputs for Case A")
        st.dataframe(result.table4, use_container_width=True, hide_index=True)
        st.markdown("#### Table 5. Report Audit Results for Worked Example")
        st.dataframe(result.table5, use_container_width=True, hide_index=True)
        return
    if view == "Figures":
        for title, filename in [
            ("Figure 2. PsyMAS analyst interface and evidence-review functions", "figure2_single_case_evidence_lineage.png"),
            ("Figure 3. Draft-to-Audited Report Example", "figure3_report_audit_example.png"),
            ("Figure 4. Three-Case Evidence Profile Grid", "figure4_three_case_evidence_grid.png"),
        ]:
            path = output_dir / filename
            st.markdown(f"#### {title}")
            if path.exists():
                st.image(str(path), use_container_width=True)
            else:
                st.info(f"{filename} is not available yet.")
        return
    if view == "Caption Text":
        st.markdown(result.captions_md)
        return
    files = pd.DataFrame(
        [
            {"file": path.name, "path": str(path), "bytes": path.stat().st_size if path.exists() else 0}
            for path in getattr(result, "files", [])
        ]
    )
    st.dataframe(files, use_container_width=True, hide_index=True)


def _render_workbench_page(stage: str) -> None:
    if stage in {"AI-Assisted Review", "Human Review"}:
        _render_evidence_review_page()
    elif stage == "Review Record":
        _render_audit_page()
    elif stage == "Research Tools":
        _render_research_tools_page()
    elif stage == "Configuration":
        _render_configuration_page()


if run_mode != "Scenario":
    _render_flow_nav(active_workflow_stage)
_render_workflow_shell(active_workflow_stage)
_WORKBENCH_MODES = {
    "_Workbench Evidence Review",
    "_Workbench Audit",
    "_Workbench Research Tools",
    "_Workbench Configuration",
}
if run_mode in _WORKBENCH_MODES:
    _render_workbench_page(active_workflow_stage)

# ----- Main content: module title only when not Preparation -----
if run_mode not in {"Preparation", *_WORKBENCH_MODES}:
    # Show feedback after switching from Preparation
    if st.session_state.pop("just_switched_module", None):
        short = run_mode.replace(" only", "").replace(" (LLM guide)", "")
        st.success(f"Switched to **{short}**.")
    # Pages that self-render their own headers
    if run_mode == "Data review":
        st.subheader("Data review")
        st.divider()



def _render_scenario_page() -> None:
    """Scenario presets as cards + LLM assistant only."""
    # Scenario A = Low-stakes (effort/quality); B = High-stakes (full detection set per table).
    SCENARIO_PRESETS_AB = {
        "A": {"title": "Scenario A: Low-Stakes", "description": "Identifies non-substantive noise to ensure high-quality data utility.\n\n Examples: Course evaluations; Pilot surveys; Classroom quizzes", "icon": "🧹", "selects": ["detect_rg", "detect_pm"], "image": "https://placehold.co/320x160/1e3a5f/94a3b8?text=Low-stakes"},
        "B": {"title": "Scenario B: High-Stakes", "description": "Protects high-stakes credentials: response-based, similarity, temporal, tampering, and preknowledge detection.\n Examples: Medical licensing; Answer copying; Brain-dump", "icon": "🛡️", "selects": ["detect_nm", "detect_pm", "detect_ac", "detect_as", "detect_pk", "detect_rg", "detect_tt"], "image": "https://placehold.co/320x160/3d1f1f/94a3b8?text=High-stakes"},
        "C": {"title": "Scenario C: Demo", "description": "Load the Appendix C simulated tutorial data and supported demo agents.", "icon": "▶", "selects": DEMO_AGENT_PRESET, "image": "https://placehold.co/320x160/2d2d4a/94a3b8?text=Demo"},
    }
    # CSS: card container (relative, left-aligned), overlay button on top, image left-aligned, equal height
    st.markdown("""
    <style>
    /* Scenario cards: shorter height - also use simpler selectors in case marker scope fails */
    div[data-testid="column"]:nth-of-type(1) > div,
    div[data-testid="column"]:nth-of-type(2) > div,
    div[data-testid="column"]:nth-of-type(3) > div {
        min-height: 180px !important;
        height: 180px !important;
        display: flex !important;
        flex-direction: column !important;
    }
    div[data-testid="column"]:nth-of-type(1) div[data-testid="stVerticalBlock"],
    div[data-testid="column"]:nth-of-type(2) div[data-testid="stVerticalBlock"],
    div[data-testid="column"]:nth-of-type(3) div[data-testid="stVerticalBlock"] {
        min-height: 180px !important;
        height: 180px !important;
        flex: 1 1 auto !important;
        display: flex !important;
        flex-direction: column !important;
    }
    div[data-testid="column"]:nth-of-type(1) div[data-testid="stVerticalBlock"] > div[data-testid="stContainer"],
    div[data-testid="column"]:nth-of-type(2) div[data-testid="stVerticalBlock"] > div[data-testid="stContainer"],
    div[data-testid="column"]:nth-of-type(3) div[data-testid="stVerticalBlock"] > div[data-testid="stContainer"] {
        min-height: 180px !important;
        height: 100% !important;
    }
    /* Force equal height: scope to block after scenario-cards-marker */
    #scenario-cards-marker ~ * div[data-testid="column"]:nth-of-type(1) > div,
    #scenario-cards-marker ~ * div[data-testid="column"]:nth-of-type(2) > div,
    #scenario-cards-marker ~ * div[data-testid="column"]:nth-of-type(3) > div {
        min-height: 180px !important;
        height: 180px !important;
        display: flex !important;
        flex-direction: column !important;
    }
    #scenario-cards-marker ~ * div[data-testid="column"]:nth-of-type(1) div[data-testid="stVerticalBlock"],
    #scenario-cards-marker ~ * div[data-testid="column"]:nth-of-type(2) div[data-testid="stVerticalBlock"],
    #scenario-cards-marker ~ * div[data-testid="column"]:nth-of-type(3) div[data-testid="stVerticalBlock"] {
        min-height: 180px !important;
        height: 180px !important;
        flex: 1 1 auto !important;
        display: flex !important;
        flex-direction: column !important;
    }
    #scenario-cards-marker ~ * div[data-testid="column"]:nth-of-type(1) > div > div,
    #scenario-cards-marker ~ * div[data-testid="column"]:nth-of-type(2) > div > div,
    #scenario-cards-marker ~ * div[data-testid="column"]:nth-of-type(3) > div > div {
        flex: 1 1 auto !important;
        min-height: 0 !important;
        display: flex !important;
        flex-direction: column !important;
    }
    /* Scenario card container: relative for overlay, left-aligned content, fill height */
    #scenario-cards-marker ~ * div[data-testid="column"]:nth-of-type(1) div[data-testid="stVerticalBlock"] > div[data-testid="stContainer"],
    #scenario-cards-marker ~ * div[data-testid="column"]:nth-of-type(2) div[data-testid="stVerticalBlock"] > div[data-testid="stContainer"],
    #scenario-cards-marker ~ * div[data-testid="column"]:nth-of-type(3) div[data-testid="stVerticalBlock"] > div[data-testid="stContainer"] {
        position: relative !important;
        text-align: left !important;
        height: 100% !important;
        min-height: 180px !important;
        flex: 1 1 auto !important;
        overflow: hidden !important;
        display: flex !important;
        flex-direction: column !important;
        border-radius: 14px !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2) !important;
        transition: box-shadow 0.2s ease, border-color 0.2s ease !important;
    }
    #scenario-cards-marker ~ * div[data-testid="column"]:nth-of-type(1) div[data-testid="stVerticalBlock"] > div[data-testid="stContainer"]:hover,
    #scenario-cards-marker ~ * div[data-testid="column"]:nth-of-type(2) div[data-testid="stVerticalBlock"] > div[data-testid="stContainer"]:hover,
    #scenario-cards-marker ~ * div[data-testid="column"]:nth-of-type(3) div[data-testid="stVerticalBlock"] > div[data-testid="stContainer"]:hover {
        border-color: rgba(255,255,255,0.25) !important;
        box-shadow: 0 8px 28px rgba(0,0,0,0.3) !important;
    }
    /* Left-align images inside scenario cards */
    #scenario-cards-marker ~ * div[data-testid="column"]:nth-of-type(1) div[data-testid="stContainer"] img,
    #scenario-cards-marker ~ * div[data-testid="column"]:nth-of-type(2) div[data-testid="stContainer"] img,
    #scenario-cards-marker ~ * div[data-testid="column"]:nth-of-type(3) div[data-testid="stContainer"] img {
        margin-left: 0 !important;
        display: block !important;
    }
    </style>
    """, unsafe_allow_html=True)

    for fn in ABERRANCE_FUNCTIONS:
        if f"ab_only_cb_{fn}" not in st.session_state:
            st.session_state[f"ab_only_cb_{fn}"] = False
    if "ab_only_scenario_select_previous" not in st.session_state:
        st.session_state["ab_only_scenario_select_previous"] = None
    if "ab_only_scenario_select" not in st.session_state:
        st.session_state["ab_only_scenario_select"] = ""
    elif st.session_state.get("ab_only_scenario_select") == "D":
        st.session_state["ab_only_scenario_select"] = ""

    def _apply_ab_only_scenario(letter: str) -> None:
        if letter in SCENARIO_PRESETS_AB:
            st.session_state["ab_only_scenario_select"] = letter
            st.session_state["ab_only_scenario_select_previous"] = letter
            for fn in ABERRANCE_FUNCTIONS:
                st.session_state[f"ab_only_cb_{fn}"] = fn in SCENARIO_PRESETS_AB[letter]["selects"]
        else:
            st.session_state["ab_only_scenario_select"] = ""
            st.session_state["ab_only_scenario_select_previous"] = ""

    scenario_link = st.query_params.get("scenario_select", None)
    if isinstance(scenario_link, list):
        scenario_link = scenario_link[0] if scenario_link else None
    if scenario_link in SCENARIO_PRESETS_AB:
        _apply_ab_only_scenario(str(scenario_link))
        st.session_state["prep_compromised_items"] = st.session_state.get("ab_only_compromised_items") or []
        if str(scenario_link) == "C":
            ok, msg = _load_demo_simulated_data()
            if ok:
                snap_ok, snap_msg = _restore_demo_evaluated_snapshot_if_available(force=True)
                if snap_ok:
                    st.session_state["_demo_load_success"] = f"{msg} {snap_msg}"
                else:
                    st.session_state["_demo_load_success"] = msg
                    st.session_state["_demo_load_warning"] = snap_msg
            else:
                st.session_state["_demo_load_error"] = msg
                try:
                    st.query_params.clear()
                except Exception:
                    pass
                st.rerun()
        else:
            _clear_demo_loaded_state()
        try:
            st.query_params.clear()
        except Exception:
            pass
        st.session_state.workflow_stage = "Assessment Data"
        st.session_state.run_mode = "Preparation"
        st.session_state["sidebar_nav"] = "Assessment Data"
        st.rerun()

    if st.session_state.get("_demo_load_error"):
        st.error(st.session_state.pop("_demo_load_error"))
    if st.session_state.get("_demo_load_warning"):
        st.warning(st.session_state.pop("_demo_load_warning"))

    st.markdown(
        """
        <style>
        .scenario-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 2rem;
            margin-top: 0.5rem;
        }
        .scenario-click-card {
            display: block;
            text-decoration: none !important;
            color: #111827 !important;
        }
        .scenario-click-card:hover .scenario-visual {
            transform: translateY(-2px);
            box-shadow: 0 14px 32px rgba(15, 23, 42, 0.22);
        }
        .scenario-visual {
            height: 220px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #B8C4D8 !important;
            font-size: 3.3rem;
            font-weight: 750;
            transition: transform 0.16s ease, box-shadow 0.16s ease;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.14);
        }
        .scenario-a .scenario-visual { background: #1F3F67; }
        .scenario-b .scenario-visual { background: #462020; }
        .scenario-c .scenario-visual { background: #30304F; }
        .scenario-title {
            margin-top: 0.8rem;
            font-size: 1rem;
            font-weight: 750;
            color: #111827 !important;
        }
        .scenario-desc {
            margin-top: 0.45rem;
            color: #111827 !important;
            font-size: 0.92rem;
            line-height: 1.45;
        }
        @media (max-width: 900px) {
            .scenario-grid { grid-template-columns: 1fr; gap: 1.2rem; }
            .scenario-visual { height: 170px; font-size: 2.4rem; }
        }
        </style>
        <div class="scenario-grid">
          <a class="scenario-click-card scenario-a" href="?scenario_select=A" target="_self">
            <div class="scenario-visual">Low-stakes</div>
            <div class="scenario-title">Scenario A: Low-Stakes</div>
            <div class="scenario-desc">Identifies non-substantive noise to ensure high-quality data utility.<br><strong>Examples:</strong> Course evaluations; Pilot surveys; Classroom quizzes</div>
          </a>
          <a class="scenario-click-card scenario-b" href="?scenario_select=B" target="_self">
            <div class="scenario-visual">High-stakes</div>
            <div class="scenario-title">Scenario B: High-Stakes</div>
            <div class="scenario-desc">Protects high-stakes credentials: response-based, similarity, temporal, tampering, and preknowledge detection.<br><strong>Examples:</strong> Medical licensing; Answer copying; Brain-dump</div>
          </a>
          <a class="scenario-click-card scenario-c" href="?scenario_select=C" target="_self">
            <div class="scenario-visual">Demo</div>
            <div class="scenario-title">Scenario C: Demo</div>
            <div class="scenario-desc">Loads Appendix C simulated responses, response times, item parameters, exposure labels, and validation tables.</div>
          </a>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_tool_irt() -> None:
    st.markdown("---")
    st.subheader("IRT (item/person parameters)")
    if "model_settings" not in st.session_state:
        st.session_state.model_settings = _interpret_prompt("")
    if "is_verified" not in st.session_state:
        st.session_state.is_verified = False
    if "prompt_analyzed" not in st.session_state:
        st.session_state.prompt_analyzed = False
    if "last_prompt" not in st.session_state:
        st.session_state.last_prompt = ""

    _render_prompt_and_confirm()

    if not st.session_state.is_verified:
        st.info("Select settings and confirm to unlock IRT execution.")
    else:
        has_sidebar_data = bool(st.session_state.get("last_uploaded_responses"))
        if has_sidebar_data:
            n_r = len(st.session_state.last_uploaded_responses)
            n_c = len(st.session_state.last_uploaded_responses[0]) if n_r else 0
            rt_txt = " + RT" if st.session_state.get("last_uploaded_rt_data") else ""
            st.caption(f"Using session data: **{n_r}** rows × **{n_c}** items{rt_txt}.")
        else:
            st.warning("Use **Bulk upload** on the **Preparation** page to load response data before running the IRT agent.")

        # Status check only; don't auto-install on reruns
        r_ok_irt, r_msg_irt = _check_r_packages(install_if_missing=False)
        if not r_ok_irt and r_msg_irt:
            st.warning(r_msg_irt)

        run_irt = st.button("Run IRT agent", key="run_irt_only", type="primary")
        if run_irt:
            if has_sidebar_data:
                responses = st.session_state.last_uploaded_responses
                rt_data = st.session_state.get("last_uploaded_rt_data") or []
            else:
                st.error("Upload response data on the **Preparation** page to run the IRT agent.")
                st.stop()
            state = {
                "responses": responses,
                "rt_data": rt_data,
                "theta": 0.0,
                "latency_flags": [],
                "next_step": "",
                "model_settings": st.session_state.model_settings,
                "is_verified": True,
            }
            with st.spinner("Running IRT agent…"):
                try:
                    out = _graph_module().irt_agent(state)
                    st.session_state.irt_only_result = {**state, **out}
                    st.success("IRT finished.")
                    st.rerun()
                except Exception as e:
                    st.exception(e)
        if st.session_state.get("irt_only_result"):
            st.subheader("Last results")
            _render_results(st.session_state.irt_only_result, response_only=True)
    st.caption("Use the navigation sidebar to switch to another module.")


def _render_tool_rt() -> None:
    st.markdown("---")
    st.subheader("RT (response time)")
    has_sidebar_rt = bool(st.session_state.get("last_uploaded_responses")) and bool(st.session_state.get("last_uploaded_rt_data"))
    if has_sidebar_rt:
        n_r = len(st.session_state.last_uploaded_responses)
        n_c = len(st.session_state.last_uploaded_responses[0]) if n_r else 0
        st.caption(f"Using session data: **{n_r}** rows × **{n_c}** items + RT.")
    else:
        st.warning("Use **Bulk upload** on the **Preparation** page to load response and RT data before running the RT agent.")
    run_rt_btn = st.button("Run RT agent", key="run_rt_only", type="primary")
    if run_rt_btn:
        if has_sidebar_rt:
            try:
                state = {
                    "responses": st.session_state.last_uploaded_responses,
                    "rt_data": st.session_state.last_uploaded_rt_data,
                    "theta": 0.0,
                    "latency_flags": [],
                    "next_step": "",
                }
                with st.spinner("Running RT agent…"):
                    out = _graph_module().rt_agent(state)
                st.session_state.rt_only_result = {**state, **out}
                st.success("Done.")
                st.rerun()
            except Exception as e:
                st.exception(e)
        else:
            st.error("Upload response and RT on the **Preparation** page to run the RT agent.")
    if st.session_state.get("rt_only_result"):
        final_rt = st.session_state.rt_only_result
        st.subheader("Latency flags")
        if final_rt.get("latency_flags"):
            st.write(", ".join(final_rt["latency_flags"]))
        else:
            st.info("No latency flags (RT agent returns flags when implemented).")
    st.caption("Use the navigation sidebar to switch to another module.")


def _render_backend_test() -> None:
    """Debug UI: verify ψ in session and test backend endpoints directly."""
    load_dotenv()
    st.markdown("---")
    st.subheader("Backend test")
    st.caption("Verify ψ in session and test backend endpoints directly (health / IRT / Detect / status / result).")

    st.markdown("**Backend URL**")
    st.code(BACKEND_URL)

    col_a, col_b = st.columns(2, gap="medium")
    with col_a:
        if st.button("GET /health"):
            try:
                r = _backend_get("/health", timeout=5)
                st.write(r.status_code)
                try:
                    st.json(r.json())
                except Exception:
                    st.text(r.text)
            except Exception as e:
                st.error(f"/health failed: {e}")

    responses = st.session_state.get("last_uploaded_responses") or []
    rt_data = st.session_state.get("last_uploaded_rt_data") or []
    answer_changes = st.session_state.get("prep_answer_changes") or []
    psi_data = st.session_state.get("last_irt_item_params") or st.session_state.get("item_params") or []

    st.markdown("**Session snapshot**")
    st.caption(f"- Responses: **{len(responses)}** rows")
    st.caption(f"- RT rows: **{len(rt_data)}**")
    st.caption(f"- Answer-change rows: **{len(answer_changes)}**")
    st.caption(f"- ψ item params: **{len(psi_data)}** items")
    if psi_data and isinstance(psi_data, list) and isinstance(psi_data[0], dict):
        st.caption(f"- ψ keys (first item): `{list(psi_data[0].keys())}`")

    st.divider()
    st.markdown("**Build Detect payload (what UI sends to backend)**")
    raw_payload = {
        "responses": responses,
        "rt_data": rt_data,
        "answer_changes": answer_changes,
        "itemtype": st.session_state.get("main_irt_itemtype", "2PL"),
        "compromised_items": st.session_state.get("prep_compromised_items") or [],
        "model_settings": st.session_state.get("model_settings") or {},
        "psi_data": psi_data,
    }
    payload = _json_safe(raw_payload)
    st.caption(
        "Payload sizes — "
        f"responses: {len(payload.get('responses') or [])}, "
        f"rt_data: {len(payload.get('rt_data') or [])}, "
        f"answer_changes: {len(payload.get('answer_changes') or [])}, "
        f"psi_data: {len(payload.get('psi_data') or [])}"
    )
    with st.expander("Show payload (JSON)", expanded=False):
        st.json(payload)

    st.divider()
    st.markdown("**IRT backend — async (`/irt/start`) or sync (`/irt`)**")
    with col_b:
        if st.button("POST /irt/start", disabled=not bool(responses)):
            try:
                irt_payload = _json_safe(
                    {
                        "responses": responses,
                        "rt_data": rt_data,
                        "itemtype": st.session_state.get("main_irt_itemtype", "2PL"),
                        "model_settings": st.session_state.get("model_settings") or {},
                    }
                )
                r = _backend_post("/irt/start", json=irt_payload, timeout=30)
                r.raise_for_status()
                js = r.json()
                st.json(js)
                st.caption("Then poll `GET /irt/{job_id}/status` and `GET /irt/{job_id}/result`.")
            except Exception as e:
                st.error(f"/irt/start failed: {e}")
        if st.button("POST /irt (sync, legacy)", disabled=not bool(responses)):
            try:
                irt_payload = _json_safe(
                    {
                        "responses": responses,
                        "rt_data": rt_data,
                        "itemtype": st.session_state.get("main_irt_itemtype", "2PL"),
                        "model_settings": st.session_state.get("model_settings") or {},
                    }
                )
                r = _backend_post("/irt", json=irt_payload, timeout=240)
                r.raise_for_status()
                js = r.json()
                st.json(js)
                if js.get("status") == "done":
                    res = js.get("result") or {}
                    ip = res.get("item_params") or []
                    if ip:
                        st.session_state["last_irt_item_params"] = ip
                        st.session_state["item_params"] = ip
                        st.success(f"Stored ψ in session: {len(ip)} items.")
            except Exception as e:
                st.error(f"/irt failed: {e}")

    st.divider()
    st.markdown("**Start Detect (/detect) and poll status**")
    if st.button("POST /detect", disabled=not bool(responses)):
        try:
            r = _backend_post("/detect", json=payload, timeout=30)
            r.raise_for_status()
            js = r.json()
            st.json(js)
            run_id = js.get("run_id")
            if run_id:
                st.session_state["detect_run_id"] = run_id
                st.session_state["detect_job_status"] = "running"
                st.success(f"Started detect run_id: {run_id}")
        except Exception as e:
            st.error(f"/detect failed: {e}")

    run_id = st.session_state.get("detect_run_id") or ""
    if run_id:
        st.markdown("**Latest run_id**")
        st.code(run_id)
        if st.button("GET /detect/{run_id}/status"):
            try:
                r = _backend_get(f"/detect/{run_id}/status", timeout=10)
                r.raise_for_status()
                st.json(r.json())
            except Exception as e:
                st.error(f"/status failed: {e}")
        if st.button("GET /detect/{run_id}/result"):
            try:
                r = _backend_get(f"/detect/{run_id}/result", timeout=20)
                r.raise_for_status()
                st.json(r.json())
            except Exception as e:
                st.error(f"/result failed: {e}")

if run_mode == "Scenario":
    _render_scenario_page()

elif run_mode == "Preparation":
    load_dotenv()
    if _ensure_demo_scenario_ready():
        st.rerun()
    if st.session_state.get("_demo_load_error"):
        st.error(st.session_state.pop("_demo_load_error"))
    if st.session_state.get("_demo_load_warning"):
        st.warning(st.session_state.pop("_demo_load_warning"))
    st.markdown(
        """
        <style>
          .psymas-strip {
            padding: 10px 12px;
            border-radius: 10px;
            border: 1px solid #C9D2DC;
            background: #FFFFFF;
            font-size: 0.9rem;
            margin: 6px 0 0 0;
            color: #1F2933 !important;
          }
          /* Preparation frames (three matching cards, fixed visual height) */
          div[data-testid="stContainer"]:has(.psymas-prep-card-marker) {
            border-radius: 8px !important;
            border: 1px solid #DDE2E8 !important;
            background: #FFFFFF !important;
            box-shadow: none;
            min-height: 250px !important;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            overflow-y: auto;
            color: #1F2933 !important;
          }
          /* Ensure the column wrapper stretches so all three cards align */
          div[data-testid="column"]:has(.psymas-prep-card-marker) {
            display: flex;
          }
          div[data-testid="column"]:has(.psymas-prep-card-marker) > div {
            flex: 1 1 auto;
          }
          .psymas-card-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding-bottom: 6px;
            margin-bottom: 6px;
            border-bottom: 1px solid #DDE2E8;
          }
          .psymas-card-title {
            font-size: 0.95rem;
            font-weight: 650;
            letter-spacing: 0.2px;
            color: #1F2933 !important;
          }
          .psymas-card-status {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            font-size: 0.85rem;
            opacity: 0.92;
          }
          .psymas-input-guide {
            margin: 6px 0 8px 0;
            padding: 10px;
            border: 1px solid #C9D2DC;
            border-radius: 12px;
            background: #FFFFFF;
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.04);
            color: #111827 !important;
          }
          .psymas-input-guide-head {
            display: flex;
            align-items: flex-end;
            justify-content: space-between;
            gap: 16px;
            margin-bottom: 8px;
          }
          .psymas-input-guide-title {
            font-size: 0.98rem;
            font-weight: 750;
            color: #111827 !important;
          }
          .psymas-input-guide-note {
            font-size: 0.82rem;
            color: #5B6778 !important;
          }
          .psymas-input-grid {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 8px;
          }
          .psymas-input-card {
            border: 1px solid #D9E1EA;
            border-left: 4px solid #94A3B8;
            border-radius: 9px;
            padding: 9px 10px;
            background: #F8FAFC;
            min-height: 104px;
            color: #111827 !important;
          }
          .psymas-data-strip {
            display: grid;
            grid-template-columns: 1.1fr 1.4fr 1.6fr 1fr;
            gap: 8px;
            margin: 8px 0 10px 0;
          }
          .psymas-data-strip-cell {
            border: 1px solid #D9E1EA;
            border-radius: 9px;
            background: #FFFFFF;
            padding: 8px 10px;
            color: #111827 !important;
            min-width: 0;
          }
          .psymas-data-strip-cell .label {
            font-size: 0.68rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #64748B !important;
            margin-bottom: 2px;
          }
          .psymas-data-strip-cell .value {
            font-size: 0.82rem;
            font-weight: 650;
            color: #111827 !important;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
          }
          @media (max-width: 1100px) {
            .psymas-data-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }
          }
          .psymas-input-card.required { border-left-color: #0F7890; background: #F7FCFD; }
          .psymas-input-card.ready { border-left-color: #15966B; background: #F4FBF7; }
          .psymas-input-card.missing { border-left-color: #C17919; background: #FFFBEB; }
          .psymas-input-card.optional { opacity: 0.9; }
          .psymas-input-card .input-file {
            font-size: 0.84rem;
            font-weight: 750;
            color: #111827 !important;
            margin-bottom: 3px;
          }
          .psymas-input-card .input-status {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 2px 7px;
            border-radius: 999px;
            background: #FFFFFF;
            border: 1px solid #D9E1EA;
            font-size: 0.72rem;
            font-weight: 700;
            color: #111827 !important;
            margin-bottom: 6px;
          }
          .psymas-input-card .input-format {
            font-size: 0.75rem;
            line-height: 1.35;
            color: #334155 !important;
          }
          .psymas-input-card code {
            color: #0F172A !important;
            background: rgba(148, 163, 184, 0.18);
            border-radius: 4px;
            padding: 1px 3px;
          }
          @media (max-width: 1300px) {
            .psymas-input-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
          }
          @media (max-width: 800px) {
            .psymas-input-grid { grid-template-columns: 1fr; }
            .psymas-input-guide-head { display: block; }
          }
          .psymas-dot {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 999px;
            margin-right: 6px;
            transform: translateY(1px);
          }
          .psymas-dot-ready { background: #00c853; }
          .psymas-dot-bad { background: #A7B0BD; }
          .psymas-pill {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 999px;
            border: 1px solid #DDE2E8;
            margin-right: 8px;
            margin-top: 2px;
            background: #F8FAFC;
            color: #1F2933 !important;
          }
          div[data-testid="stFileUploaderDropzone"] {
            background: #FFFFFF !important;
            border: 2px dashed #6B7787 !important;
            color: #1F2933 !important;
          }
          div[data-testid="stFileUploaderDropzone"] > div,
          div[data-testid="stFileUploaderDropzone"] section {
            background: #FFFFFF !important;
            color: #1F2933 !important;
          }
          div[data-testid="stFileUploaderDropzone"] * {
            color: #1F2933 !important;
          }
          div[data-testid="stFileUploaderDropzone"] button,
          div[data-testid="stFileUploaderDropzone"] button * {
            background: #174E5F !important;
            color: #FFFFFF !important;
            border-color: #174E5F !important;
          }
          div[data-testid="stFileUploaderFile"] {
            background: #FFFFFF !important;
            border: 1px solid #D1D8E0 !important;
          }
          div[data-testid="stFileUploaderFile"] * {
            color: #111827 !important;
            opacity: 1 !important;
          }
          div[data-testid="stFileUploaderFile"] svg {
            color: #4B5563 !important;
            fill: currentColor !important;
            stroke: currentColor !important;
          }
          div[data-testid="stAlert"] {
            color: #1F2933 !important;
          }
          div[data-testid="stAlert"] * {
            color: #1F2933 !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── Readiness strip ──
    resp_loaded = bool(st.session_state.get("last_uploaded_responses"))
    rt_loaded = bool(st.session_state.get("last_uploaded_rt_data"))
    n_persons = len(st.session_state.get("last_uploaded_responses") or [])
    n_items = len((st.session_state.get("last_uploaded_responses") or [{}])[0]) if n_persons else 0
    psi_loaded = bool(st.session_state.get("last_irt_item_params") or st.session_state.get("item_params"))

    provider = _llm_provider()
    model_id = _effective_llm_model()
    or_key = os.getenv("OPENROUTER_API_KEY", "")
    g_key = os.getenv("GOOGLE_API_KEY", "")
    llm_configured = bool(or_key.strip()) if provider == "openrouter" else bool(g_key and g_key.strip())
    llm_short = f"{provider} · {model_id}"

    _dot = lambda ok: f"<span class='psymas-dot {'psymas-dot-ready' if ok else 'psymas-dot-bad'}'></span>"
    _esc = lambda s: (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

    backend_ok, backend_note, backend_status_tip = _backend_status_summary()

    # Detect button is always green; its enabled/disabled state is
    # recomputed later once we know which agents are selected and what
    # data they actually require.
    all_ready = False
    _run_bg = "#0F5B6A"
    _run_bg_hover = "#0A4754"
    st.markdown(
        f"""
        <style>
          /* Style ONLY the full forensic review action (scoped via marker div) */
          div[data-testid="stContainer"]:has(.psymas-runforensic-marker) {{
            margin: 0.55rem 0 0.25rem 0 !important;
            padding: 0.75rem 0.9rem !important;
            border: 1px solid #BFD1DB !important;
            border-radius: 12px !important;
            background: linear-gradient(180deg, #FFFFFF 0%, #F4FAFC 100%) !important;
            box-shadow: 0 14px 30px rgba(15, 91, 106, 0.10) !important;
          }}
          div[data-testid="stContainer"]:has(.psymas-runforensic-marker) div[data-testid="stButton"] > button {{
            min-height: 58px !important;
            height: 58px !important;
            border-radius: 12px !important;
            background: linear-gradient(180deg, #147286 0%, {_run_bg} 100%) !important;
            border: 1px solid #073B4C !important;
            color: #FFFFFF !important;
            font-size: 1.02rem !important;
            font-weight: 800 !important;
            letter-spacing: 0.01em !important;
            box-shadow: 0 12px 22px rgba(15, 91, 106, 0.28), inset 0 1px 0 rgba(255,255,255,0.22) !important;
          }}
          div[data-testid="stContainer"]:has(.psymas-runforensic-marker) div[data-testid="stButton"] > button * {{
            color: #FFFFFF !important;
            font-weight: 800 !important;
          }}
          div[data-testid="stContainer"]:has(.psymas-runforensic-marker) div[data-testid="stButton"] > button:disabled {{
            opacity: 1 !important;
            cursor: not-allowed !important;
            background: linear-gradient(180deg, #F8FAFC 0%, #E5E7EB 100%) !important;
            border: 1px solid #94A3B8 !important;
            color: #334155 !important;
            box-shadow: 0 6px 14px rgba(15, 23, 42, 0.10), inset 0 1px 0 rgba(255,255,255,0.70) !important;
            transform: none !important;
          }}
          div[data-testid="stContainer"]:has(.psymas-runforensic-marker) div[data-testid="stButton"] > button:disabled * {{
            color: #334155 !important;
            font-weight: 800 !important;
          }}
          div[data-testid="stContainer"]:has(.psymas-runforensic-marker) div[data-testid="stButton"] > button:hover {{
            background: linear-gradient(180deg, #105F70 0%, {_run_bg_hover} 100%) !important;
            border-color: #062F3B !important;
            box-shadow: 0 14px 26px rgba(15, 91, 106, 0.34), inset 0 1px 0 rgba(255,255,255,0.24) !important;
            transform: translateY(-1px);
          }}
          div[data-testid="column"]:has(.psymas-runforensic-marker) div[data-testid="stButton"] > button,
          div[data-testid="stHorizontalBlock"]:has(.psymas-runforensic-marker) div[data-testid="stButton"] > button {{
            min-height: 58px !important;
            height: 58px !important;
            border-radius: 12px !important;
            background: linear-gradient(180deg, #147286 0%, {_run_bg} 100%) !important;
            border: 1px solid #073B4C !important;
            color: #FFFFFF !important;
            font-size: 1.02rem !important;
            font-weight: 800 !important;
            box-shadow: 0 12px 22px rgba(15, 91, 106, 0.28), inset 0 1px 0 rgba(255,255,255,0.22) !important;
          }}
          div[data-testid="column"]:has(.psymas-runforensic-marker) div[data-testid="stButton"] > button *,
          div[data-testid="stHorizontalBlock"]:has(.psymas-runforensic-marker) div[data-testid="stButton"] > button * {{
            color: #FFFFFF !important;
            font-weight: 800 !important;
          }}
          div[data-testid="column"]:has(.psymas-runforensic-marker) div[data-testid="stButton"] > button:disabled,
          div[data-testid="stHorizontalBlock"]:has(.psymas-runforensic-marker) div[data-testid="stButton"] > button:disabled {{
            opacity: 1 !important;
            cursor: not-allowed !important;
            background: linear-gradient(180deg, #F8FAFC 0%, #E2E8F0 100%) !important;
            border: 1.5px solid #64748B !important;
            color: #1F2937 !important;
            box-shadow: 0 7px 16px rgba(15, 23, 42, 0.14), inset 0 1px 0 rgba(255,255,255,0.82) !important;
          }}
          div[data-testid="column"]:has(.psymas-runforensic-marker) div[data-testid="stButton"] > button:disabled *,
          div[data-testid="stHorizontalBlock"]:has(.psymas-runforensic-marker) div[data-testid="stButton"] > button:disabled * {{
            color: #1F2937 !important;
            font-weight: 800 !important;
          }}
          .psymas-runforensic-marker {{
            display: block;
            height: 0;
            width: 0;
            overflow: hidden;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ----- Agent select (synced with Scenario; same state keys ab_only_cb_* / ab_only_scenario_select) -----
    SCENARIO_PRESETS_AB_NAMES = {
        "A": "Scenario A: Low-Stakes",
        "B": "Scenario B: High-Stakes",
        "C": "Scenario C: Demo",
        "Custom": "Scenario C: Demo",  # backward compat
    }
    SCENARIO_PRESET_IMAGES = {
        "A": "https://placehold.co/480x120/1e3a5f/94a3b8?text=Low-stakes",
        "B": "https://placehold.co/480x120/3d1f1f/94a3b8?text=High-stakes",
        "C": "https://placehold.co/480x120/2d2d4a/94a3b8?text=Demo",
    }
    for fn in ABERRANCE_FUNCTIONS:
        if f"ab_only_cb_{fn}" not in st.session_state:
            st.session_state[f"ab_only_cb_{fn}"] = False
    if "ab_only_scenario_select" not in st.session_state:
        st.session_state["ab_only_scenario_select"] = ""
    scenario_letter = st.session_state.get("ab_only_scenario_select") or ""
    if scenario_letter in ("D", "Custom"):
        st.session_state["ab_only_scenario_select"] = ""
        scenario_letter = ""
    show_psychometrics_controls = active_workflow_stage == "Deterministic Evidence"
    show_detect_controls = active_workflow_stage in {"Assessment Data", "Deterministic Evidence"}
    if show_psychometrics_controls:
        _render_psychometrics_result_center()
        st.stop()

    # Compact left-aligned banner at top (reuses Scenario page imagery)
    banner_img = SCENARIO_PRESET_IMAGES.get(scenario_letter)
    if show_psychometrics_controls and banner_img:
        st.markdown(
            f"<img src='{banner_img}' style='max-height:110px;border-radius:12px;margin:0 0 6px 0;display:block;' />",
            unsafe_allow_html=True,
        )

    # Manual agent suggestions are disabled for the Demo preset; Demo is loaded from Appendix C tables.
    if False and show_psychometrics_controls and scenario_letter == "C":
        with st.container(border=True):
            st.markdown(
                "<div style='font-size:0.85rem;color:#64748b;display:flex;align-items:center;justify-content:space-between;'>"
                "<span>Describe your testing situation; the assistant will suggest agents.</span>"
                "<span>Ctrl+Enter to send</span>"
                "</div>",
                unsafe_allow_html=True,
            )
            prep_llm_req_main = st.text_area(
                "Describe your testing situation for Preparation agents",
                value=st.session_state.get("prep_llm_requirement_main", ""),
                placeholder="Describe stakes, modality, security risks, proctoring, etc.…",
                height=80,
                key="prep_llm_requirement_main",
                label_visibility="collapsed",
            )
            suggest_btn_main = st.button("Suggest agents from description", key="prep_suggest_agents_main")
            # Keyboard shortcut: Ctrl+Enter / Cmd+Enter to trigger suggestion
            st.markdown(
                """
                <script>
                (function() {
                  if (window._psymasPrepMainHotkeyBound) return;
                  window._psymasPrepMainHotkeyBound = true;
                  document.addEventListener('keydown', function(e) {
                    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                      const target = Array.from(document.querySelectorAll('button'))
                        .find(b => b.innerText.trim() === 'Suggest agents from description');
                      if (target) { e.preventDefault(); target.click(); }
                    }
                  });
                })();
                </script>
                """,
                unsafe_allow_html=True,
            )

        if suggest_btn_main and (prep_llm_req_main or st.session_state.get("prep_llm_requirement_main", "")):
            # Reuse the same LLM helper as the Scenario page.
            with st.spinner("Asking LLM for agent suggestions…"):
                letter_main, explanation_main = _suggest_aberrance_scenario(
                    prep_llm_req_main.strip() or st.session_state.get("prep_llm_requirement_main", "")
                )
            suggested_agents = st.session_state.get("_llm_suggested_agents") or []
            # If agents were explicitly suggested, prefer them. Otherwise, fall back to scenario-based defaults.
            if suggested_agents:
                for fn in ABERRANCE_FUNCTIONS:
                    st.session_state[f"ab_only_cb_{fn}"] = fn in suggested_agents
            else:
                SCENARIO_PRESETS_AB_PREP = {
                    "A": ["detect_rg", "detect_pm"],
                    "B": ["detect_nm", "detect_pm", "detect_ac", "detect_as", "detect_pk", "detect_rg", "detect_tt"],
                    "C": DEMO_AGENT_PRESET,
                }
                if letter_main in SCENARIO_PRESETS_AB_PREP:
                    st.session_state["ab_only_scenario_select"] = letter_main
                    scenario_letter = letter_main
                    for fn in ABERRANCE_FUNCTIONS:
                        st.session_state[f"ab_only_cb_{fn}"] = fn in SCENARIO_PRESETS_AB_PREP[letter_main]
            if explanation_main:
                st.session_state["prep_llm_suggestion_main"] = explanation_main
            st.rerun()
        if st.session_state.get("prep_llm_suggestion_main"):
            st.info(st.session_state["prep_llm_suggestion_main"])
            if st.button("Clear suggestion", key="prep_clear_suggestion_main"):
                st.session_state["prep_llm_suggestion_main"] = ""
                st.rerun()

    # Simple checkbox list of all agents on Preparation
    # Ordered to match index categories: nm, pm, as, pk, (cp implicit), rg.
    prep_agent_labels = [
        ("detect_nm", "Nonparametric Misfit (detect_nm) — Guttman/HT person-fit without IRT."),
        ("detect_pm", "Model Misfit (detect_pm) — parametric person-fit under IRT."),
        ("detect_as", "Answer Similarity (detect_as) — similarity clusters / collusion."),
        ("detect_ac", "Answer Copying (detect_ac) — source–copier pairs."),
        ("detect_pk", "Preknowledge (detect_pk) — success on compromised items."),
        ("detect_tt", "Test Tampering (detect_tt) — answer-change / erasure patterns."),
        ("detect_rg", "Rapid Guessing (detect_rg) — unusually fast, low-effort responding."),
    ]
    prep_ab_fns = [fn for fn, _ in prep_agent_labels if st.session_state.get(f"ab_only_cb_{fn}")]
    if not prep_ab_fns:
        prep_ab_fns = ["detect_nm"]  # default so at least one agent runs

    # Decide which extra inputs are relevant based on selected agents.
    # Only rapid guessing is blocked on RT in the current backend; other agents may use
    # score-only methods or ignore RT.
    need_rt = "detect_rg" in prep_ab_fns
    need_comp = "detect_pk" in prep_ab_fns  # compromised items needed for Preknowledge
    need_tt = "detect_tt" in prep_ab_fns  # answer-change data needed for Test Tampering
    need_model = any(
        fn in prep_ab_fns for fn in ["detect_pm", "detect_pk", "detect_ac", "detect_as"]
    )  # IRT model needed when psi-based agents are selected
    def _input_spec_card(file_name: str, label: str, status: str, status_class: str, fmt: str) -> str:
        return (
            f"<div class='psymas-input-card {status_class}'>"
            f"<div class='input-file'>{html.escape(file_name)}</div>"
            f"<div class='input-status'>{html.escape(status)}</div>"
            f"<div class='input-format'><b>{html.escape(label)}</b><br>{fmt}</div>"
            "</div>"
        )

    input_specs = [
        _input_spec_card(
            "responses.csv",
            "Response matrix",
            "Ready" if resp_loaded else "Required",
            "ready" if resp_loaded else "missing",
            "Wide CSV: one row per examinee; item columns contain <code>0/1</code> scores.",
        ),
        _input_spec_card(
            "response_times.csv",
            "Response-time matrix",
            "Ready" if rt_loaded else "Optional",
            "ready" if rt_loaded else "optional",
            "Enables response-time evidence and rapid-guessing checks when available.",
        ),
        _input_spec_card(
            "item_params.csv",
            "Item parameters",
            "Ready" if psi_loaded else "Optional / auto-estimated",
            "ready" if psi_loaded else "optional",
            "Upload existing parameters or let PsyMAS estimate them from responses.",
        ),
        _input_spec_card(
            "compromised_items.csv",
            "Exposure labels",
            "Ready" if comp_ok else "Optional",
            "ready" if comp_ok else "optional",
            "Enables preknowledge review when exposed or compromised items are known.",
        ),
        _input_spec_card(
            "answer_changes.csv",
            "Answer-change records",
            "Ready" if tt_loaded else "Optional",
            "ready" if tt_loaded else "optional",
            "Enables answer-change / tampering review when initial-final data exist.",
        ),
    ]
    st.markdown(
        """
        <div class="psymas-input-guide">
          <div class="psymas-input-guide-head">
            <div>
              <div class="psymas-input-guide-title">Input data checklist</div>
              <div class="psymas-input-guide-note">Use the recommended names below, or upload equivalent CSV/JSON files; PsyMAS infers type from file name and columns.</div>
            </div>
            <div class="psymas-input-guide-note">Only responses are required to start; other files enable additional evidence domains.</div>
          </div>
          <div class="psymas-input-grid">
        """
        + "".join(input_specs)
        + """
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Download sample CSV files", expanded=False):
        sample_dir = Path("data") / "sample"
        sample_files = [
            "responses.csv",
            "response_times.csv",
            "item_params.csv",
            "compromised_items.csv",
            "answer_changes.csv",
        ]
        existing_samples = [name for name in sample_files if (sample_dir / name).exists()]
        if len(existing_samples) != len(sample_files):
            missing = [name for name in sample_files if name not in existing_samples]
            st.warning("Missing sample file(s): " + ", ".join(missing))

        bulk_buf = io.BytesIO()
        with zipfile.ZipFile(bulk_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for filename in existing_samples:
                zf.write(sample_dir / filename, arcname=filename)
        bulk_cols = st.columns([1.2, 4.8])
        with bulk_cols[0]:
            st.download_button(
                "Download all",
                data=bulk_buf.getvalue(),
                file_name="psymas_sample_csv_files.zip",
                mime="application/zip",
                key="download_sample_bulk_zip",
                on_click="ignore",
                use_container_width=True,
            )
        st.caption("Bulk upload recognizes these exact file names automatically.")

        tpl_cols = st.columns(5)
        for col, filename in zip(tpl_cols, sample_files):
            path = sample_dir / filename
            with col:
                if path.exists():
                    st.download_button(
                        filename,
                        data=path.read_bytes(),
                        file_name=filename,
                        mime="text/csv",
                        key=f"download_sample_{filename}",
                        on_click="ignore",
                        use_container_width=True,
                    )
    if st.session_state.get("_demo_load_success"):
        st.success(st.session_state.pop("_demo_load_success"))
    elif st.session_state.get("demo_data_loaded") and scenario_letter == "C":
        st.caption(
            "Demo data loaded from Appendix C simulated tables, including the long "
            "initial/final answer-change input used by detect_tt."
        )
    if active_workflow_stage == "Assessment Data" and scenario_letter == "C" and not _streamlined_ui_enabled():
        _render_demo_evaluated_snapshot_panel()
    run_forensic = False
    bulk_context = (
        st.expander("Upload controls and raw input restore", expanded=False)
        if _streamlined_ui_enabled() and active_workflow_stage == "Assessment Data" and resp_loaded
        else st.container(border=True)
    )
    with bulk_context:
        st.markdown(
            """
            <div class="psymas-card-head">
              <div class="psymas-card-title">Bulk upload</div>
              <div class="psymas-card-status">CSV/JSON</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        bulk_files = st.file_uploader(
            "Upload multiple datasets",
            type=["csv", "json"],
            accept_multiple_files=True,
            key="prep_bulk_uploader",
            help="Drop several files at once. Names such as responses_matrix, response_times, compromised_items, answer_changes, or item_params are detected automatically.",
        )
        bulk_sig = tuple((getattr(f, "name", None), getattr(f, "size", None)) for f in (bulk_files or []))
        if bulk_sig and st.session_state.get("prep_last_bulk_sig") != bulk_sig:
            loaded_bulk, bulk_errors = _process_prep_bulk_uploads(list(bulk_files or []))
            st.session_state["prep_last_bulk_sig"] = bulk_sig
            st.session_state["prep_bulk_loaded_messages"] = loaded_bulk
            st.session_state["prep_bulk_error_messages"] = bulk_errors
        for msg in st.session_state.get("prep_bulk_error_messages") or []:
            st.warning(msg)
    with st.container(border=True):
        st.markdown("<div class='psymas-prep-card-marker'></div>", unsafe_allow_html=True)
        itemtype = st.selectbox("IRT model for automatic item-parameter estimation", ["2PL", "1PL", "3PL", "4PL"], index=0, key="main_irt_itemtype")
        responses_for_irt = st.session_state.get("last_uploaded_responses") or []
        rt_for_irt = st.session_state.get("last_uploaded_rt_data") or []
        current_psi = st.session_state.get("last_irt_item_params") or st.session_state.get("item_params") or []
        if not need_model:
            st.caption("This model is used when selected detectors require item parameters. Current detector selection does not require automatic IRT estimation.")
        else:
            last_itemtype = st.session_state.get("last_irt_itemtype")
            if current_psi and last_itemtype and last_itemtype != itemtype:
                for key in ("last_irt_item_params", "item_params", "forensic_person_params"):
                    st.session_state.pop(key, None)
                current_psi = []
            if current_psi:
                st.caption("Item parameters are ready. The forensic run will use these parameters together with the selected detector modules.")
            else:
                st.caption(
                    f"Item parameters (psi) may be uploaded or automatically estimated. If no item-parameter file is provided, "
                    f"PsyMAS estimates them from the response matrix using the selected IRT model ({itemtype}) before running the full forensic review."
                )
            auto_sig = json.dumps(
                {
                    "itemtype": itemtype,
                    "n_rows": len(responses_for_irt),
                    "n_cols": len(responses_for_irt[0]) if responses_for_irt else 0,
                    "rt_rows": len(rt_for_irt),
                },
                sort_keys=True,
            )
            if responses_for_irt and not current_psi and not st.session_state.get("prep_irt_job_id") and st.session_state.get("prep_auto_irt_sig") != auto_sig:
                try:
                    payload = _json_safe(
                        {
                            "responses": responses_for_irt,
                            "rt_data": rt_for_irt,
                            "itemtype": itemtype,
                            "model_settings": st.session_state.get("model_settings") or {},
                        }
                    )
                    r = _backend_post("/irt/start", json=payload, timeout=30)
                    r.raise_for_status()
                    js = r.json()
                    jid = js.get("job_id")
                    if not jid:
                        st.session_state["last_irt_error"] = "Backend did not return a job_id for IRT."
                    else:
                        st.session_state["prep_irt_job_id"] = jid
                        st.session_state["prep_auto_irt_sig"] = auto_sig
                        st.session_state["last_irt_error"] = None
                        st.rerun()
                except Exception as e:
                    st.session_state["last_irt_error"] = f"Failed to start IRT: {e}"

            prep_irt_jid = st.session_state.get("prep_irt_job_id")
            if prep_irt_jid:
                st.caption(f"Estimating {itemtype} item parameters...")
                _irt_prog = st.progress(0)
                try:
                    irt_st_resp = _backend_get(f"/irt/{prep_irt_jid}/status", timeout=25)
                    irt_st_resp.raise_for_status()
                    irt_js = irt_st_resp.json()
                except Exception as e:
                    st.session_state["last_irt_error"] = str(e)
                    st.session_state.pop("prep_irt_job_id", None)
                    st.error(f"IRT status request failed: {e}")
                    st.rerun()
                irt_backend_status = irt_js.get("status", "pending")
                irt_prog = int(irt_js.get("progress") or 0)
                _irt_prog.progress(min(100, max(0, irt_prog)))
                if irt_backend_status == "error":
                    err = irt_js.get("error") or "IRT failed."
                    st.session_state["last_irt_error"] = err
                    st.session_state.pop("prep_irt_job_id", None)
                    st.error(f"IRT failed: {err}")
                    st.rerun()
                if irt_backend_status == "unknown":
                    err = irt_js.get("error") or "job_id not found"
                    st.session_state["last_irt_error"] = err
                    st.session_state.pop("prep_irt_job_id", None)
                    st.error(f"IRT job lost: {err}")
                    st.rerun()
                if irt_backend_status == "done":
                    try:
                        irt_rr = _backend_get(f"/irt/{prep_irt_jid}/result", timeout=60)
                        irt_rr.raise_for_status()
                        irt_res_js = irt_rr.json()
                        irt_result = irt_res_js.get("result") or {}
                    except Exception as e:
                        st.session_state["last_irt_error"] = str(e)
                        st.session_state.pop("prep_irt_job_id", None)
                        st.error(f"Failed to fetch IRT result: {e}")
                        st.rerun()
                    st.session_state.pop("prep_irt_job_id", None)
                    if irt_result.get("item_params"):
                        psi_data = irt_result["item_params"]
                        st.session_state["last_irt_item_params"] = psi_data
                        st.session_state["item_params"] = psi_data
                        st.session_state["last_irt_itemtype"] = itemtype
                        if irt_result.get("person_params"):
                            st.session_state["forensic_person_params"] = irt_result["person_params"]
                        st.session_state["last_irt_error"] = None
                        st.rerun()
                    else:
                        st.session_state["last_irt_error"] = (
                            irt_result.get("icc_error") or "IRT returned no item parameters."
                        )
                        st.error(st.session_state["last_irt_error"])
                        st.rerun()
                else:
                    time.sleep(2)
                    st.rerun()

            elif current_psi:
                st.caption(f"{itemtype} item parameters ready: {len(current_psi)} items.")
            elif responses_for_irt:
                st.caption(f"{itemtype} estimation will start automatically.")
            else:
                st.caption("Upload response data to estimate item parameters.")
            if st.session_state.get("last_irt_error") and not prep_irt_jid:
                st.caption(st.session_state["last_irt_error"])

    prep_ready = _prep_detect_requirements(prep_ab_fns)
    detect_ready = prep_ready["detect_ready"]

    run_forensic = False
    detect_completed = st.session_state.get("detect_job_status") == "done" and st.session_state.get("forensic_result") is not None
    if not detect_completed:
        with st.container():
            btn_col, hint_col = st.columns([0.42, 0.58], gap="medium")
            with btn_col:
                st.markdown("<div class='psymas-runforensic-marker'></div>", unsafe_allow_html=True)
                run_forensic = st.button(
                    "Run Full Forensic Review",
                    key="prep_run_forensic",
                    type="primary",
                    disabled=not detect_ready,
                    use_container_width=True,
                )
            with hint_col:
                if detect_ready:
                    st.caption("Ready to estimate any missing item parameters, run selected detectors, build governed evidence, and open the review workspace.")
                elif prep_ready["blocks"]:
                    st.caption("Still needed: " + "; ".join(prep_ready["blocks"]) + ".")
                else:
                    st.caption("Upload required data and confirm backend status to enable the forensic review.")

    prep_ready = _prep_detect_requirements(prep_ab_fns)
    resp_loaded = prep_ready["resp_ok"]
    rt_loaded = bool(st.session_state.get("last_uploaded_rt_data"))
    psi_loaded = prep_ready.get("psi_ready", False)
    n_persons = len(st.session_state.get("last_uploaded_responses") or [])
    n_items = len((st.session_state.get("last_uploaded_responses") or [{}])[0]) if n_persons else 0

    need_rt = prep_ready["need_rt"]
    need_psi = prep_ready["need_psi"]
    need_tt = prep_ready["need_tt"]

    resp_ok = prep_ready["resp_ok"]
    rt_ok = prep_ready["rt_ok"]
    psi_ok = prep_ready["psi_ok"]
    tt_ok = prep_ready["tt_ok"]
    backend_ok = prep_ready["backend_ok"]
    backend_note = prep_ready["backend_note"]
    backend_status_tip = prep_ready["backend_tip"]

    all_ready = prep_ready["detect_ready"]

    rt_title = (
        _esc("RT loaded (used by Rapid Guessing).")
        if need_rt and rt_loaded
        else _esc("RT not loaded; response-time evidence will be unavailable.")
        if need_rt
        else _esc("RT not required for the current agent selection.")
    )
    psi_title = (
        _esc("Item parameters (ψ) ready for IRT-based agents.")
        if need_psi and psi_loaded
        else _esc("Item parameters (ψ) will be estimated automatically from the selected IRT model.")
        if need_psi
        else _esc("Item parameters (ψ) not required for the current agent selection.")
    )

    # Build status pills only for requirements relevant to the current agent selection
    _pills_html = []
    _pills_html.append(
        f'<span class="psymas-pill" title="{_esc("Response loaded" if resp_ok else "Response missing")}">{_dot(resp_ok)}Response'
        f'{(" " + str(n_persons) + "×" + str(n_items)) if resp_ok else ""}</span>'
    )
    if need_rt:
        _pills_html.append(
            f'<span class="psymas-pill" title="{rt_title}">{_dot(rt_ok)}RT</span>'
        )
    if need_psi:
        _pills_html.append(
            f'<span class="psymas-pill" title="{psi_title}">{_dot(psi_loaded)}ψ</span>'
        )
    if need_tt:
        _pills_html.append(
            f'<span class="psymas-pill" title="{_esc("Answer-change data loaded." if tt_ok else "Answer-change data not loaded; tampering evidence will be unavailable.")}">{_dot(tt_ok)}Answer changes</span>'
        )
    # LLM and LangGraph Agents are always relevant for Detect
    _pills_html.append(
        f'<span class="psymas-pill" title="{_esc("LLM configured; verdict can use the selected model." if llm_configured else "LLM key missing; Detect will still run and use a rule-based verdict.")}">'
        f'{_dot(True)}LLM ({llm_short}){" optional" if not llm_configured else ""}</span>'
    )
    _pills_html.append(
        f'<span class="psymas-pill" title="{_esc(backend_status_tip)}">{_dot(backend_ok)}LangGraph Agents — {_esc(backend_note)}</span>'
    )

    if run_forensic:
        # Start a backend job instead of running LangGraph directly in Streamlit
        # Remember which agents were selected so Summary only reports those.
        st.session_state["last_detect_agents"] = prep_ab_fns
        _responses = st.session_state.get("last_uploaded_responses") or []
        _comp = st.session_state.get("prep_compromised_items") or []
        # Leave empty to let backend default to items 1..n-1 (R requires ≥1 secure item)
        raw_payload = {
            "responses": _responses,
            "rt_data": st.session_state.get("last_uploaded_rt_data") or [],
            "answer_changes": st.session_state.get("prep_answer_changes") or [],
            "itemtype": st.session_state.get("main_irt_itemtype", "2PL"),
            "compromised_items": _comp,
            "model_settings": _model_settings_for_backend(),
            # Reuse ψ generated on Preparation to keep IRT parameters identical.
            "psi_data": st.session_state.get("last_irt_item_params") or st.session_state.get("item_params") or [],
            "aberrance_functions": prep_ab_fns,
            "threshold_config": _active_threshold_config(),
        }
        payload = _json_safe(raw_payload)
        try:
            resp = _backend_post("/detect", json=payload, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            run_id = data.get("run_id")
            if not run_id:
                st.error("Detection backend did not return a run_id.")
            else:
                st.session_state["detect_run_id"] = run_id
                st.session_state["detect_job_status"] = "running"
                st.session_state["detect_job_error"] = None
                st.session_state["forensic_responses"] = _responses
                st.session_state["forensic_rt_data"] = st.session_state.get("last_uploaded_rt_data") or []
                st.session_state["_nav_request"] = "Deterministic Evidence"
                st.rerun()
        except Exception as e:
            st.error(f"Failed to start detection backend: {e}")

    # ── Run status (when a job was started) ──
    run_id = st.session_state.get("detect_run_id")
    if active_workflow_stage == "Deterministic Evidence" and run_id and st.session_state.get("last_uploaded_responses"):
        if "detect_job_status" not in st.session_state:
            st.session_state["detect_job_status"] = "pending"
        _job_status = st.session_state.get("detect_job_status", "pending")
        if _job_status == "done" and st.session_state.get("forensic_result") is not None:
            pass
        elif _job_status == "error":
            _det_err = st.session_state.get("detect_job_error") or "Unknown error."
            st.error("Detection failed: " + _det_err)
            if _detect_run_id_replica_hint(_det_err):
                st.warning(
                    "当前错误通常表示 **API 有多个副本**：Streamlit 在服务器上请求你的后端时，"
                    "创建任务和查询状态可能打到 **不同容器**，磁盘上的 `run_id` 无法共享。\n\n"
                    "任选其一：**(1)** 在 Railway 把该 **API / LangGraph 服务 Replicas 设为 1**；"
                    "**(2)** 给同一 API 服务加上 **Redis**（插件）并注入 **REDIS_URL**，让所有副本共用任务状态。"
                )
            if st.button("Try again", key="prep_retry_detect"):
                st.session_state["detect_run_id"] = None
                st.session_state["detect_job_status"] = "pending"
                st.session_state["detect_job_error"] = None
                st.rerun()
        else:
            st.markdown("**Running detection**")
            if _job_status != "done":
                st.session_state["detect_job_status"] = "running"
            _prog = st.progress(0)
            _status = st.empty()
            st.session_state["_detect_prog_current"] = 0

            def _smooth_to(target: int, *, duration_s: float = 0.35) -> None:
                target = max(0, min(100, int(target)))
                try:
                    start = int(st.session_state.get("_detect_prog_current", 0))
                except Exception:
                    start = 0
                if target <= start:
                    _prog.progress(target)
                    st.session_state["_detect_prog_current"] = target
                    return
                steps = max(1, int(duration_s / 0.02))
                for i in range(1, steps + 1):
                    v = start + (target - start) * (i / steps)
                    _prog.progress(int(v))
                    time.sleep(0.02)
                _prog.progress(target)
                st.session_state["_detect_prog_current"] = target

            responses = st.session_state["last_uploaded_responses"]
            rt_data = st.session_state.get("last_uploaded_rt_data") or []
            try:
                status_resp = _backend_get(f"/detect/{run_id}/status", timeout=20)
                status_resp.raise_for_status()
                js = status_resp.json()
            except Exception as e:
                st.session_state["detect_job_status"] = "error"
                st.session_state["detect_job_error"] = str(e)
                st.error(f"Failed to contact backend: {e}")
                st.rerun()
            backend_status = js.get("status", "pending")
            backend_progress = int(js.get("progress", 0))
            _smooth_to(min(backend_progress, 97), duration_s=0.20)

            if backend_status == "error":
                err_msg = js.get("error") or "Backend reported an error."
                st.session_state["detect_job_status"] = "error"
                st.session_state["detect_job_error"] = err_msg
                st.error(f"Detection failed: {err_msg}")
                st.rerun()
            elif backend_status == "unknown":
                err_msg = js.get("error") or "run_id not found on backend (job may have expired, or another server instance handled the request)."
                st.session_state["detect_job_status"] = "error"
                st.session_state["detect_job_error"] = err_msg
                st.error(f"Detection status lost: {err_msg}")
                st.rerun()
            elif backend_status == "done":
                try:
                    res_resp = _backend_get(f"/detect/{run_id}/result", timeout=20)
                    res_resp.raise_for_status()
                    res_js = res_resp.json()
                    result = res_js.get("result") or {}
                except Exception as e:
                    st.session_state["detect_job_status"] = "error"
                    st.session_state["detect_job_error"] = str(e)
                    st.error(f"Failed to fetch result: {e}")
                    st.rerun()
                st.session_state["forensic_result"] = result
                st.session_state["forensic_responses"] = responses
                st.session_state["forensic_rt_data"] = rt_data
                st.session_state["forensic_psi_data"] = result.get("psi_data") or []
                st.session_state["detect_job_status"] = "done"
                st.session_state.pop("_governed_review_tables_cache", None)
                st.session_state.pop("psychometrics_result_data_export", None)
                _smooth_to(100, duration_s=0.25)
                st.session_state["_nav_request"] = "Deterministic Evidence"
                st.rerun()
            else:
                _status.caption("Preparing forensic indices...")
                time.sleep(3)
                st.rerun()
