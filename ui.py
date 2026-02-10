"""
Streamlit UI for loading response and response-time datasets, then running the
psych_workflow graph.

Data loading lives here—not in the Orchestrator. The Orchestrator only routes;
the UI reads the two files and passes them as initial state into the graph.
"""
import warnings

# Suppress rpy2 "R is not initialized by the main thread" warning (harmless in cloud/Streamlit)
warnings.filterwarnings("ignore", message=".*main thread.*")

from pathlib import Path
import io
import json
import math  # stdlib (not 'maths')
import os
import re
import base64
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import time
import uuid
import requests
from dotenv import load_dotenv

from graph import analyze_prompt, psych_workflow, forensic_workflow, irt_agent, rt_agent, aberrance_agent

# Initialize R early so it is ready when needed
try:
    import rpy2.robjects as _ro
    _ro.r("1+1")
except Exception:
    pass

# Model engine options for Psych-MAS Summary (display label, API model name)
GEMINI_MODEL_OPTIONS = [
    ("Gemini 1.5 Flash (latest)", "models/gemini-1.5-flash-latest"),
    ("Gemini 1.5 Flash 002", "models/gemini-1.5-flash-002"),
    ("Gemini 1.5 Flash 001", "models/gemini-1.5-flash-001"),
    ("Gemini 2.0 Flash", "models/gemini-2.0-flash"),
    ("Gemini 2.5 Flash", "models/gemini-2.5-flash"),
]
DEFAULT_GEMINI_MODEL_IDS = [api_id for _, api_id in GEMINI_MODEL_OPTIONS]

# OpenRouter: model list lives in openrouter_models.py
from openrouter_models import OPENROUTER_FREE_MODEL_IDS, OPENROUTER_FREE_MODELS

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Aberrance tab: function list and payload key (used when building workflow payload)
ABERRANCE_FUNCTIONS = ["detect_rg", "detect_pm", "detect_ac", "detect_pk", "detect_as", "detect_nm", "detect_tt"]


def _call_openrouter(api_key: str, model_id: str, messages: list[dict], timeout: int = 90) -> tuple[str | None, str | None]:
    """Call OpenRouter chat completions. Return (text, None) on success or (None, error_message)."""
    if not model_id or not messages:
        return None, "No model or messages."
    headers = {"Content-Type": "application/json"}
    if api_key and api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"
    try:
        body = {"model": model_id, "messages": messages}
        resp = requests.post(OPENROUTER_API_URL, headers=headers, json=body, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message", {})
        text = msg.get("content") or ""
        return (text.strip() or "No response from model.", None)
    except requests.exceptions.HTTPError as e:
        try:
            err_body = e.response.json() if e.response is not None else {}
            msg = err_body.get("error", {}).get("message", str(e))
        except Exception:
            msg = str(e)
        code = e.response.status_code if e.response is not None else 0
        if code == 402:
            msg += " Add credits at openrouter.ai/credits (free models require a non-negative balance)."
        return None, f"{code}: {msg}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _test_openrouter_model(api_key: str, model_id: str, timeout: int = 20) -> tuple[bool, str | None, float]:
    """Test one OpenRouter model with a minimal message. Return (ok, error_message, response_time_sec)."""
    t0 = time.perf_counter()
    text, err = _call_openrouter(api_key, model_id, [{"role": "user", "content": "Hi"}], timeout=timeout)
    elapsed = time.perf_counter() - t0
    return (bool(text and not err), err, elapsed)


def _test_openrouter_api_key(api_key: str, timeout: int = 15) -> tuple[bool, str]:
    """Verify OPENROUTER_API_KEY with a minimal request. Return (True, message) or (False, error_message)."""
    if not api_key or not api_key.strip():
        return False, "OPENROUTER_API_KEY not set in .env. Add it for higher limits (get key at openrouter.ai)."
    # Don't hardcode the first model: model availability can change and a 404 does not imply a bad key.
    # Try the currently selected model first, then fall back through the free list.
    selected = st.session_state.get("selected_gemini_model")
    candidates = []
    if selected and isinstance(selected, str):
        candidates.append(selected)
    candidates.extend([m for m in OPENROUTER_FREE_MODEL_IDS if m not in candidates])

    last_err = None
    for mid in candidates[: min(10, len(candidates))]:
        ok, err, _ = _test_openrouter_model(api_key.strip(), mid, timeout)
        if ok:
            return True, f"OpenRouter API key is valid (model ok: {mid})."
        last_err = err

    if last_err and "401" in str(last_err):
        return False, "Invalid or unauthorized OpenRouter key (401). Check OPENROUTER_API_KEY in .env."
    if last_err and "402" in str(last_err):
        return False, "OpenRouter: insufficient credits (402). Add credits at openrouter.ai/credits — free models need a non-negative balance."
    if last_err and "404" in str(last_err):
        return False, (
            "OpenRouter key may be valid, but the tested models returned 404 (no endpoints found). "
            "Pick a different OpenRouter model in **Model engine**, or refresh the OpenRouter model list."
        )
    if last_err:
        return False, f"OpenRouter: {last_err}"
    return False, "OpenRouter request failed (no response)."


def _llm_provider() -> str:
    """Current LLM provider: 'openrouter' or 'google' (OpenRouter is default)."""
    return st.session_state.get("llm_provider", "openrouter")


def _current_model_ids() -> list[str]:
    """Return list of model IDs to use for current provider (discovered for Google, or OpenRouter free list)."""
    if _llm_provider() == "openrouter":
        return OPENROUTER_FREE_MODEL_IDS
    return st.session_state.get("discovered_model_ids") or DEFAULT_GEMINI_MODEL_IDS


def _current_model_options() -> list[tuple[str, str]]:
    """Return [(display_name, model_id), ...] for current provider."""
    if _llm_provider() == "openrouter":
        return OPENROUTER_FREE_MODELS
    return st.session_state.get("discovered_model_options") or GEMINI_MODEL_OPTIONS


def _effective_llm_model() -> str:
    """Return the model id to use for all LLM calls: pinned model if set for current provider, else current selection."""
    provider = _llm_provider()
    pinned_provider = st.session_state.get("pinned_llm_provider")
    pinned_model = st.session_state.get("pinned_llm_model")
    if pinned_provider == provider and pinned_model:
        model_ids = _current_model_ids()
        if pinned_model in model_ids:
            return pinned_model
    model_ids = _current_model_ids()
    selected = st.session_state.get("selected_gemini_model", model_ids[0] if model_ids else (DEFAULT_GEMINI_MODEL_IDS[0] if provider == "google" else OPENROUTER_FREE_MODEL_IDS[0]))
    if selected not in model_ids:
        selected = model_ids[0] if model_ids else (DEFAULT_GEMINI_MODEL_IDS[0] if provider == "google" else OPENROUTER_FREE_MODEL_IDS[0])
    return selected


def _model_variants_with_selected_first() -> list[str]:
    """Return model list with effective (pinned or selected) model first (for Psych-MAS Summary)."""
    model_ids = _current_model_ids()
    effective = _effective_llm_model()
    if effective not in model_ids:
        effective = model_ids[0] if model_ids else (DEFAULT_GEMINI_MODEL_IDS[0] if _llm_provider() == "google" else OPENROUTER_FREE_MODEL_IDS[0])
    return [effective] + [m for m in model_ids if m != effective]


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


def _drop_index_column(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    first_col = df.columns[0]
    if str(first_col).strip() == "" or str(first_col).startswith("Unnamed"):
        return df.drop(columns=[first_col])

    maybe_index = pd.to_numeric(df[first_col], errors="coerce")
    if maybe_index.notna().all():
        vals = maybe_index.astype(int)
        if vals.is_unique:
            sorted_vals = sorted(vals.tolist())
            if sorted_vals == list(range(1, len(vals) + 1)) or sorted_vals == list(range(0, len(vals))):
                return df.drop(columns=[first_col])

    return df


def _align_rt_columns_to_response(rt_df: pd.DataFrame, resp_df: pd.DataFrame) -> pd.DataFrame:
    """Set RT DataFrame column names to match response columns by position so keys align (same layout)."""
    if rt_df.shape[1] != resp_df.shape[1]:
        return rt_df
    rt_df = rt_df.copy()
    rt_df.columns = list(resp_df.columns)[: rt_df.shape[1]]
    return rt_df


def _coerce_numeric(df: pd.DataFrame, label: str) -> pd.DataFrame:
    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    non_numeric_cols = []
    for col in df.columns:
        if (df[col].notna() & numeric_df[col].isna()).any():
            non_numeric_cols.append(str(col))
    if non_numeric_cols:
        raise ValueError(
            f"{label} has non-numeric values in columns: {', '.join(non_numeric_cols)}."
        )
    return numeric_df


def _validate_binary_responses(resp_df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = _coerce_numeric(resp_df, "Response data")
    invalid_cols = []
    for col in numeric_df.columns:
        values = numeric_df[col].dropna().unique().tolist()
        if values and not set(values).issubset({0, 1}):
            invalid_cols.append(str(col))
    if invalid_cols:
        raise ValueError(
            "Response data must be binary (0/1). Invalid columns: "
            + ", ".join(invalid_cols)
        )
    return numeric_df


def _interpret_prompt(
    prompt: str,
    *,
    llm_provider: str | None = None,
    llm_model_id: str | None = None,
    openrouter_api_key: str | None = None,
    google_api_key: str | None = None,
) -> dict:
    """Run prompt interpretation using the given provider and model (from Model engine)."""
    provider = llm_provider if llm_provider is not None else st.session_state.get("llm_provider", "openrouter")
    model_id = llm_model_id if llm_model_id is not None else _effective_llm_model()
    if openrouter_api_key is None and provider == "openrouter":
        load_dotenv()
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")
    if google_api_key is None and provider == "google":
        load_dotenv()
        google_api_key = os.getenv("GOOGLE_API_KEY", "")
    return analyze_prompt(
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
    return psych_workflow.invoke(initial_state)


def _run_workflow(payload_json: str) -> dict:
    """Run workflow; use version so cache invalidates when graph changes."""
    return _run_workflow_cached(payload_json, _WORKFLOW_VERSION)


def _psych_describe_responses(resp_df: pd.DataFrame) -> tuple[pd.DataFrame | None, str | None]:
    """Run psych::describe in R on response data. Returns (describe_df, None) or (None, error_msg)."""
    if resp_df.empty or resp_df.shape[1] == 0:
        return None, "No response data."
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
    except Exception as exc:
        return None, f"rpy2/R not available: {exc}"
    try:
        with (ro.default_converter + pandas2ri.converter).context():
            ro.globalenv["resp_df"] = resp_df
            desc = ro.r("psych::describe(resp_df)")
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
    """Create a Wright Map using R's WrightMap package."""
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
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
        with (ro.default_converter + pandas2ri.converter).context():
            # Convert DataFrames to R objects
            ro.globalenv["item_df"] = item_df
            ro.globalenv["person_df"] = person_df
            ro.globalenv["wright_map_path"] = str(wright_map_path)
            ro.globalenv["b_col"] = b_col
            ro.globalenv["ability_col"] = ability_col
            
            ro.r(
                """
                library(WrightMap)
                library(grDevices)
                
                # Extract item difficulties as a matrix/vector
                item_difficulty <- as.numeric(item_df[[b_col]])
                
                # Extract person abilities as a vector
                person_ability <- as.numeric(person_df[[ability_col]])
                
                # Create Wright Map
                # item.prop = 0.8 means 80% space for items, 20% for persons (histogram)
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


def _suggest_aberrance_scenario(user_description: str) -> tuple[str, str]:
    """Use LLM (same provider/model as Psych-MAS Assistant) to suggest scenario A, B, or C. Returns (letter 'A'|'B'|'C'|'', explanation)."""
    if not (user_description or "").strip():
        return "", "Describe your testing situation (e.g. classroom quiz, certification exam, at-home test)."
    load_dotenv()
    prompt = (
        "You are helping choose an aberrant-behavior detection scenario for a psychometric test.\n\n"
        "Scenarios:\n"
        "A = Data Cleaning (Low-Stakes): surveys, classroom quizzes, pilot tests; focus on lazy/random responders. Uses: detect_rg, detect_pm.\n"
        "B = Exam Security (High-Stakes/Proctored): certification or standardized exams in controlled setting; focus on copying and pre-knowledge. Uses: detect_ac, detect_pk, detect_pm.\n"
        "C = Remote Monitoring (Online/Unproctored): at-home testing; focus on collusion, rapid guessing, inconsistent performance. Uses: detect_as, detect_rg, detect_nm.\n\n"
        f"User's situation: {user_description.strip()[:500]}\n\n"
        "Reply with exactly two lines. Line 1: only one letter, A, B, or C. Line 2: one short sentence explaining why that scenario fits."
    )
    # Use same provider and model as Psych-MAS Assistant (Model engine tab)
    if _llm_provider() == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        model_ids = _model_variants_with_selected_first()
        for model_id in model_ids:
            text, err = _call_openrouter(api_key, model_id, [{"role": "user", "content": prompt}], timeout=60)
            if err or not text:
                continue
            letter, explanation = _parse_scenario_suggestion(text)
            return letter, (explanation[:400] if explanation else "Suggested based on your description.")
        return "", "OpenRouter: no model returned a response. Try another model in Model engine or set OPENROUTER_API_KEY in .env."
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
            text = (resp.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "") or "").strip()
            letter, explanation = _parse_scenario_suggestion(text)
            return letter, (explanation[:400] if explanation else "Suggested based on your description.")
        except requests.exceptions.HTTPError as e:
            last_err = f"{e.response.status_code if e.response is not None else 'HTTP'}: {e}"
            if e.response is not None and e.response.status_code == 404:
                continue
            break
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            break
    return "", (last_err or "No model responded. Check Model engine and GOOGLE_API_KEY.")


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
    if _llm_provider() == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        model_ids = _model_variants_with_selected_first()
        for model_id in model_ids:
            text, err = _call_openrouter(api_key, model_id, [{"role": "user", "content": prompt}], timeout=60)
            if err or not text:
                continue
            module, reason = _parse_module_suggestion(text)
            return module, (reason[:400] if reason else "Suggested based on your description.")
        return "", "OpenRouter: no model returned a response. Set Model engine or OPENROUTER_API_KEY."
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
    if _llm_provider() == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        model_ids = _model_variants_with_selected_first()
        for model_id in model_ids:
            text, err = _call_openrouter(api_key, model_id, [{"role": "user", "content": prompt}], timeout=90)
            if err:
                continue
            if text:
                return text
        return _format_llm_error("OpenRouter: no model returned a response. Try another model or set OPENROUTER_API_KEY in .env.")
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
            st.link_button("Open LangGraph API", url=LANGGRAPH_API_URL, type="primary", use_container_width=True)
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
            if "llm_provider" not in st.session_state:
                st.session_state.llm_provider = "openrouter"
            st.radio(
                "LLM provider",
                options=["openrouter", "google"],
                format_func=lambda x: "OpenRouter (free)" if x == "openrouter" else "Google (Gemini)",
                key="llm_provider",
                horizontal=True,
                help="OpenRouter offers free models; optional OPENROUTER_API_KEY in .env (get one at openrouter.ai).",
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
            else:
                if "openrouter_api_key_test_result" not in st.session_state:
                    st.session_state.openrouter_api_key_test_result = None
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
                st.caption("OpenRouter free models. Set OPENROUTER_API_KEY in .env for higher limits (get key at openrouter.ai).")
            _model_options = _current_model_options()
            _model_ids = _current_model_ids()
            if "pinned_llm_model" not in st.session_state:
                st.session_state.pinned_llm_model = None
            if "pinned_llm_provider" not in st.session_state:
                st.session_state.pinned_llm_provider = None
            _pinned = st.session_state.pinned_llm_model if st.session_state.pinned_llm_provider == st.session_state.llm_provider else None
            if "selected_gemini_model" not in st.session_state:
                st.session_state.selected_gemini_model = (_pinned if _pinned and _pinned in _model_ids else None) or (_model_ids[0] if _model_ids else (DEFAULT_GEMINI_MODEL_IDS[0] if st.session_state.llm_provider == "google" else OPENROUTER_FREE_MODEL_IDS[0]))
            if st.session_state.selected_gemini_model not in _model_ids:
                st.session_state.selected_gemini_model = (_pinned if _pinned and _pinned in _model_ids else None) or (_model_ids[0] if _model_ids else (DEFAULT_GEMINI_MODEL_IDS[0] if st.session_state.llm_provider == "google" else OPENROUTER_FREE_MODEL_IDS[0]))
            st.selectbox(
                "LLM model for Psych-MAS Summary",
                options=_model_ids,
                format_func=lambda x: next((label for label, api_id in _model_options if api_id == x), _model_id_to_display_name(x) if "/" not in str(x) else str(x).split("/")[-1].replace("-", " ").title()),
                key="selected_gemini_model",
                help="Google: click 'Check model availability' to refresh from API. OpenRouter: free models, no key required for basic use.",
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
                    with st.spinner("Testing OpenRouter free models…"):
                        avail = {}
                        errs = {}
                        times = {}
                        for model_id in OPENROUTER_FREE_MODEL_IDS:
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
                    selected_id = st.session_state.get("selected_gemini_model", OPENROUTER_FREE_MODEL_IDS[0])
                    st.caption("Model status (🟢 available, 🔴 unavailable; **Selected** = current choice; response time in seconds)")
                    errs = st.session_state.get("model_availability_errors", {})
                    times = st.session_state.get("model_availability_times", {})
                    for label, api_id in OPENROUTER_FREE_MODELS:
                        status = "🟢" if st.session_state.model_availability.get(api_id, False) else "🔴"
                        current = " **Selected**" if api_id == selected_id else ""
                        err_msg = errs.get(api_id)
                        err_suffix = f" — *{err_msg}*" if err_msg else ""
                        t = times.get(api_id)
                        time_s = f" *({t:.1f}s)*" if t is not None else ""
                        st.markdown(f"{status} {label}{time_s}{current}{err_suffix}")
            elif _api_key:
                if st.button("Check model availability", key="check_model_availability"):
                    with st.spinner("Discovering models (ListModels)…"):
                        discovered = _discover_gemini_models(_api_key)
                    if discovered:
                        st.session_state.discovered_model_options = discovered
                        st.session_state.discovered_model_ids = [mid for _, mid in discovered]
                        if st.session_state.selected_gemini_model not in st.session_state.discovered_model_ids:
                            _pinned = st.session_state.pinned_llm_model if st.session_state.pinned_llm_provider == st.session_state.llm_provider else None
                            st.session_state.selected_gemini_model = (_pinned if _pinned and _pinned in st.session_state.discovered_model_ids else None) or st.session_state.discovered_model_ids[0]
                    else:
                        st.session_state.discovered_model_options = None
                        st.session_state.discovered_model_ids = None
                    with st.spinner("Testing models…"):
                        ids_to_test = st.session_state.discovered_model_ids or DEFAULT_GEMINI_MODEL_IDS
                        avail, errs, times = _check_models_availability(_api_key, ids_to_test)
                        st.session_state.model_availability = avail
                        st.session_state.model_availability_errors = errs
                        st.session_state.model_availability_times = times
                    st.rerun()
                if st.session_state.model_availability is not None:
                    opts = st.session_state.discovered_model_options or GEMINI_MODEL_OPTIONS
                    selected_id = st.session_state.get("selected_gemini_model", _model_ids[0] if _model_ids else "")
                    st.caption("Model status (🟢 available, 🔴 unavailable; **Selected** = current choice; response time in seconds). List from API when you clicked Check.")
                    errs = st.session_state.get("model_availability_errors", {})
                    times = st.session_state.get("model_availability_times", {})
                    for label, api_id in opts:
                        status = "🟢" if st.session_state.model_availability.get(api_id, False) else "🔴"
                        current = " **Selected**" if api_id == selected_id else ""
                        err_msg = errs.get(api_id)
                        err_suffix = f" — *{err_msg}*" if err_msg else ""
                        t = times.get(api_id)
                        time_s = f" *({t:.1f}s)*" if t is not None else ""
                        st.markdown(f"{status} {label}{time_s}{current}{err_suffix}")
            else:
                st.caption("Set GOOGLE_API_KEY in .env to check Google model availability.")

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
            submit_question = st.form_submit_button("Ask LLM", use_container_width=True)
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
                "title": "Scenario A: Data Cleaning (Low-Stakes)",
                "icon": "🧹",
                "description": "Best for surveys, classroom quizzes, or pilot tests. Focuses on removing lazy or random responders.",
                "selects": ["detect_rg", "detect_pm"],
            },
            "B": {
                "title": "Scenario B: Exam Security (High-Stakes/Proctored)",
                "icon": "🛡️",
                "description": "Best for certification exams or standardized tests in a controlled environment. Focuses on copying and pre-knowledge.",
                "selects": ["detect_ac", "detect_pk", "detect_pm"],
            },
            "C": {
                "title": "Scenario C: Remote Monitoring (Online/Unproctored)",
                "icon": "📶",
                "description": "Best for at-home testing. Focuses on collusion, rapid guessing, and inconsistent performance.",
                "selects": ["detect_as", "detect_rg", "detect_nm"],
            },
        }
        for fn in ABERRANCE_FUNCTIONS:
            if f"aberrance_cb_{fn}" not in st.session_state:
                st.session_state[f"aberrance_cb_{fn}"] = False
        if "aberrance_scenario_select_previous" not in st.session_state:
            st.session_state["aberrance_scenario_select_previous"] = None

        def _apply_aberrance_scenario(letter: str, *, update_select: bool = True) -> None:
            """Set scenario dropdown (if update_select) and function-list checkboxes to match the given scenario (A, B, or C). Do not set the select key after the selectbox is instantiated (Streamlit disallows it)."""
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
            suggest_btn = st.button("Suggest scenario from description", key="aberrance_suggest_scenario", use_container_width=True)
        if suggest_btn and (llm_req or st.session_state.get("aberrance_llm_requirement", "")):
            with st.spinner("Asking LLM..."):
                letter, explanation = _suggest_aberrance_scenario(llm_req.strip() or st.session_state.get("aberrance_llm_requirement", ""))
            if letter and letter in SCENARIO_PRESETS:
                _apply_aberrance_scenario(letter)
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
            ("C", f"📶 {SCENARIO_PRESETS['C']['title']}"),
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
        if scenario_choice in ("A", "B", "C") and scenario_choice != prev:
            _apply_aberrance_scenario(scenario_choice, update_select=False)
            st.rerun()
        if scenario_choice in ("", "Custom"):
            st.session_state["aberrance_scenario_select_previous"] = scenario_choice
        if scenario_choice == "A":
            st.caption(SCENARIO_PRESETS["A"]["description"])
        elif scenario_choice == "B":
            st.caption(SCENARIO_PRESETS["B"]["description"])
        elif scenario_choice == "C":
            st.caption(SCENARIO_PRESETS["C"]["description"])

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

        gen_btn = st.button("Generate results", key="aberrance_generate_results", type="primary", use_container_width=True)
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
**Detection functions (aberrance v0.3.0, CRAN):**

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
    page_title="PsyMAS-Aberrance",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dashboard-style CSS: cards, spacing, status strip
st.markdown("""
<style>
    /* Card-like sections */
    div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stMarkdown"]) .stMarkdown h3 {
        margin-top: 0.5rem;
        margin-bottom: 0.25rem;
        font-weight: 600;
    }
    /* Status strip (works in light/dark) */
    .status-strip {
        padding: 0.5rem 0.75rem;
        border-radius: 0.375rem;
        border: 1px solid rgba(49, 51, 63, 0.2);
        font-size: 0.875rem;
        margin-bottom: 1rem;
    }
    [data-theme="dark"] .status-strip { border-color: rgba(250, 250, 250, 0.2); }
    /* Sidebar nav emphasis */
    section[data-testid="stSidebar"] .stRadio label { font-weight: 500; }
    section[data-testid="stSidebar"] h1 { font-size: 1.25rem; margin-bottom: 0.25rem; }
</style>
""", unsafe_allow_html=True)

# ----- Sidebar: navigation -----
# Use a separate key for the radio so we can set run_mode from Main-page buttons (Confirm / Go to).
NAV_OPTIONS = [
    "Preparation",
    "Detect Progress",
    "Data review",
    "Aberrance Summary",
    "Student Profile",
    "Collusion Network",
    "Individual Aberrance",
    "Temporal Forensics",
    "Tools",
]
if "run_mode" not in st.session_state:
    st.session_state.run_mode = "Preparation"
run_mode = st.session_state.get("run_mode", "Preparation")

# Allow safe programmatic navigation without fighting the sidebar radio widget.
# Any code can set st.session_state["_nav_request"] = "<page>" and st.rerun().
_nav_req = st.session_state.pop("_nav_request", None)
if _nav_req in NAV_OPTIONS:
    st.session_state.run_mode = _nav_req
    # Sync the sidebar selection BEFORE the widget is created (avoids StreamlitAPIException).
    st.session_state["sidebar_nav"] = _nav_req
    run_mode = _nav_req

# Back-compat: older sessions may still point to removed pages
_LEGACY_TOOL_PAGES = {"Aberrance only": "Aberrance", "IRT only": "IRT", "RT only": "RT", "Full workflow": "Aberrance"}
if run_mode in _LEGACY_TOOL_PAGES:
    st.session_state["tools_mode"] = _LEGACY_TOOL_PAGES[run_mode]
    st.session_state.run_mode = "Tools"
    st.rerun()
if run_mode == "Command Center":
    st.session_state.run_mode = "Aberrance Summary"
    st.rerun()

with st.sidebar:
    st.title("PsyMAS-Aberrance")
    st.caption("Ver. 0.2.0— Psychometric MAS for detecting aberrant test behavior.")
    st.divider()
    _has_resp = bool(st.session_state.get("last_uploaded_responses"))
    _has_rt = bool(st.session_state.get("last_uploaded_rt_data"))
    _has_psi = bool(st.session_state.get("last_irt_item_params") or st.session_state.get("item_params"))
    _has_forensic = bool(st.session_state.get("forensic_result"))
    _nav_ready = {
        "Preparation": True,
        "Detect Progress": False,
        "Data review": _has_resp,
        "Aberrance Summary": _has_forensic,
        "Student Profile": _has_forensic,
        "Collusion Network": _has_forensic,
        "Individual Aberrance": _has_forensic,
        "Temporal Forensics": _has_forensic,
        "Tools": _has_resp,
    }
    def _nav_label(x: str) -> str:
        base = {
            "Preparation": "🧪 Preparation",
            "Detect Progress": "⏳ Detect Progress",
            "Data review": "📋 Data review",
            "Tools": "🧰 Tools",
            "Aberrance Summary": "🧾 Aberrance Summary",
            "Student Profile": "👤 Student Profile",
            "Collusion Network": "🕸️ Collusion Network",
            "Individual Aberrance": "📊 Individual Aberrance",
            "Temporal Forensics": "⏳ Temporal Forensics",
        }.get(x, x)
        # Show a dot for every option (clearer readiness affordance)
        return ("🟢 " if _nav_ready.get(x) else "⚪ ") + base
    sidebar_choice = st.radio(
        "Navigate",
        options=NAV_OPTIONS,
        format_func=_nav_label,
        key="sidebar_nav",
                label_visibility="visible",
            )
    if sidebar_choice != run_mode:
        st.session_state.run_mode = sidebar_choice
        st.rerun()
    st.caption("Start on **Preparation** to upload data / generate ψ / test LLM.")
    st.divider()

    st.markdown("**Data upload**")
    sidebar_resp = st.file_uploader(
        "Response (CSV)",
        type=["csv"],
        key="sidebar_resp",
        help="Item responses 0/1, rows=persons, cols=items.",
    )
    sidebar_rt = st.file_uploader(
        "RT (CSV, optional)",
        type=["csv"],
        key="sidebar_rt",
        help="Same layout as response. Required for rapid guessing.",
    )
    if sidebar_resp is not None:
        try:
            sidebar_resp.seek(0)
            _r = pd.read_csv(sidebar_resp)
            _r = _drop_index_column(_r)
            _r = _validate_binary_responses(_r)
            st.session_state.last_uploaded_responses = _r.to_dict(orient="records")
            _rt_list = []
            if sidebar_rt is not None:
                try:
                    sidebar_rt.seek(0)
                    _rt = pd.read_csv(sidebar_rt)
                    _rt = _drop_index_column(_rt)
                    _rt = _coerce_numeric(_rt, "RT")
                    if _r.shape[1] == _rt.shape[1]:
                        _rt = _align_rt_columns_to_response(_rt, _r)
                        _rt_list = _rt.to_dict(orient="records")
                except Exception:
                    pass
            st.session_state.last_uploaded_rt_data = _rt_list
        except Exception as e:
            st.warning(f"Response file: {e}")
    # Allow RT upload even when Response wasn't re-uploaded in this rerun
    elif sidebar_rt is not None and st.session_state.get("last_uploaded_responses"):
        try:
            sidebar_rt.seek(0)
            _rt = pd.read_csv(sidebar_rt)
            _rt = _drop_index_column(_rt)
            _rt = _coerce_numeric(_rt, "RT")
            _r_prev = pd.DataFrame(st.session_state.last_uploaded_responses)
            if _r_prev.shape[1] == _rt.shape[1]:
                _rt = _align_rt_columns_to_response(_rt, _r_prev)
            # Store regardless of column match; downstream uses first N columns by position
            st.session_state.last_uploaded_rt_data = _rt.to_dict(orient="records")
        except Exception:
            # Keep prior RT if parsing fails
            pass
    if st.session_state.get("last_uploaded_responses"):
        n_r = len(st.session_state.last_uploaded_responses)
        n_c = len(st.session_state.last_uploaded_responses[0]) if n_r else 0
        st.caption(f"✓ Response: {n_r} rows × {n_c} items")
        if st.session_state.get("last_uploaded_rt_data"):
            st.caption("✓ RT: loaded")
    st.divider()

# ----- Main content: module title only when not Preparation -----
if run_mode != "Preparation":
    # Show feedback after switching from Preparation
    if st.session_state.pop("just_switched_module", None):
        short = run_mode.replace(" only", "").replace(" (LLM guide)", "")
        st.success(f"Switched to **{short}**.")
    # Pages that self-render their own headers
    if run_mode == "Data review":
        st.subheader("Data review")
        st.divider()

def _render_tool_aberrance() -> None:
    SCENARIO_PRESETS_AB = {
        "A": {"title": "Scenario A: Data Cleaning (Low-Stakes)", "description": "Surveys, classroom quizzes; focus on lazy/random responders.", "selects": ["detect_rg", "detect_pm"]},
        "B": {"title": "Scenario B: Exam Security (High-Stakes/Proctored)", "description": "Certification exams; focus on copying and pre-knowledge.", "selects": ["detect_ac", "detect_pk", "detect_pm"]},
        "C": {"title": "Scenario C: Remote Monitoring (Online/Unproctored)", "description": "At-home testing; focus on collusion, rapid guessing.", "selects": ["detect_as", "detect_rg", "detect_nm"]},
    }
    for fn in ABERRANCE_FUNCTIONS:
        if f"ab_only_cb_{fn}" not in st.session_state:
            st.session_state[f"ab_only_cb_{fn}"] = False
    if "ab_only_scenario_select_previous" not in st.session_state:
        st.session_state["ab_only_scenario_select_previous"] = None
    if "ab_only_scenario_select" not in st.session_state:
        st.session_state["ab_only_scenario_select"] = ""

    def _apply_ab_only_scenario(letter: str, *, update_select: bool = True) -> None:
        if letter not in SCENARIO_PRESETS_AB:
            return
        if update_select:
            st.session_state["ab_only_scenario_select"] = letter
        st.session_state["ab_only_scenario_select_previous"] = letter
        for fn in ABERRANCE_FUNCTIONS:
            st.session_state[f"ab_only_cb_{fn}"] = fn in SCENARIO_PRESETS_AB[letter]["selects"]

    # ----- 1. Scenario -----
    with st.container():
        st.markdown("**1. Scenario**")
        st.caption("Choose a scenario; it will auto-select the function list below. Or use the LLM to suggest one.")
    st.caption("Choose a scenario; it will auto-select the function list below. Or use the LLM dialogue to suggest one.")
    SCENARIO_OPTIONS_AB = [
        ("", "— Select a scenario (optional) —"),
        ("A", f"🧹 {SCENARIO_PRESETS_AB['A']['title']}"),
        ("B", f"🛡️ {SCENARIO_PRESETS_AB['B']['title']}"),
        ("C", f"📶 {SCENARIO_PRESETS_AB['C']['title']}"),
        ("Custom", "Custom (manual selection only)"),
    ]
    option_values_ab = [x[0] for x in SCENARIO_OPTIONS_AB]
    option_label_map_ab = dict(SCENARIO_OPTIONS_AB)
    scenario_choice_ab = st.selectbox(
        "Scenario",
        options=option_values_ab,
        format_func=lambda x: option_label_map_ab.get(x, x),
        key="ab_only_scenario_select",
        label_visibility="collapsed",
    )
    prev_ab = st.session_state.get("ab_only_scenario_select_previous")
    if scenario_choice_ab in ("A", "B", "C") and scenario_choice_ab != prev_ab:
        _apply_ab_only_scenario(scenario_choice_ab, update_select=False)
        st.rerun()
    if scenario_choice_ab in ("", "Custom"):
        st.session_state["ab_only_scenario_select_previous"] = scenario_choice_ab
    if scenario_choice_ab in ("A", "B", "C"):
        st.caption(SCENARIO_PRESETS_AB[scenario_choice_ab]["description"])

    # ----- 2. Options (LLM + function list) -----
    with st.container():
        st.markdown("**2. Options**")
        st.caption("Describe your context for an LLM suggestion, then pick detection functions.")
    llm_req_ab = st.text_area(
        "Describe your testing situation",
        value=st.session_state.get("ab_only_llm_requirement", ""),
        placeholder="e.g. Low-stakes quizzes in class. / High-stakes licensure exam. / Online unproctored assessment.",
        height=100,
        key="ab_only_llm_requirement",
        label_visibility="collapsed",
    )
    suggest_btn_ab = st.button("Suggest scenario from description", key="ab_only_suggest_scenario", use_container_width=True)
    if suggest_btn_ab and (llm_req_ab or st.session_state.get("ab_only_llm_requirement", "")):
        with st.spinner("Asking LLM…"):
            letter_ab, explanation_ab = _suggest_aberrance_scenario(llm_req_ab.strip() or st.session_state.get("ab_only_llm_requirement", ""))
        if letter_ab and letter_ab in SCENARIO_PRESETS_AB:
            _apply_ab_only_scenario(letter_ab)
            st.session_state["ab_only_llm_suggestion"] = explanation_ab
            st.rerun()
        elif explanation_ab:
            st.session_state["ab_only_llm_suggestion"] = explanation_ab
            st.rerun()
    if st.session_state.get("ab_only_llm_suggestion"):
        st.info(st.session_state["ab_only_llm_suggestion"])
        if st.button("Clear suggestion", key="ab_only_clear_suggestion"):
            st.session_state["ab_only_llm_suggestion"] = ""
            st.rerun()

    st.caption("Select which detection functions to run. Scenario and LLM above update these; changing reflects as Custom.")
    ab_fns = []
    for fn, label_help in [
        ("detect_rg", ("Rapid Guessing (detect_rg)", "Detects participants answering too quickly.")),
        ("detect_pm", ("Model Misfit (detect_pm)", "Detects general \"odd\" behavior that doesn't fit standard models.")),
        ("detect_ac", ("Answer Copying (detect_ac)", "Detects if a student copied from another.")),
        ("detect_as", ("Answer Similarity (detect_as)", "Detects suspicious groups with nearly identical answers.")),
        ("detect_pk", ("Preknowledge (detect_pk)", "Detects suspiciously well on leaked items.")),
        ("detect_nm", ("Guttman Errors (detect_nm)", "Detects hard-right but easy-wrong patterns.")),
        ("detect_tt", ("Test Tampering (detect_tt)", "Requires erasure data.")),
    ]:
        if st.checkbox(label_help[0], value=st.session_state.get(f"ab_only_cb_{fn}", False), key=f"ab_only_cb_{fn}", help=label_help[1]):
            ab_fns.append(fn)
    st.divider()

    # ----- 3. Data & run -----
    with st.container():
        st.markdown("**3. Data & run**")
    # Compromised items for detect_pk (Scenario B / preknowledge)
    ab_only_compromised_key = "ab_only_compromised_items"
    if ab_only_compromised_key not in st.session_state:
        st.session_state[ab_only_compromised_key] = []
    if "detect_pk" in ab_fns or scenario_choice_ab == "B":
        comp_default = ", ".join(map(str, st.session_state.get(ab_only_compromised_key) or []))
        comp_input = st.text_input(
            "Compromised item numbers (for Preknowledge)",
            value=comp_default,
            key="ab_only_compromised_input",
            placeholder="e.g. 1, 5, 10, 15",
            help="Comma-separated 1-based item indices that are known to be compromised/leaked. Required for detect_pk (Scenario B).",
        )
        try:
            comp_list = [int(x.strip()) for x in comp_input.split(",") if x.strip() and x.strip().isdigit()]
            if comp_list:
                st.session_state[ab_only_compromised_key] = comp_list
            elif comp_input.strip():
                st.session_state[ab_only_compromised_key] = []
        except Exception:
            pass
        if "detect_pk" in ab_fns and not (st.session_state.get(ab_only_compromised_key)):
            st.caption("Enter at least one compromised item number to run Preknowledge detection.")

    # Model misfit, answer copying, and preknowledge need item parameters (ψ); we get them from the IRT node, not CSV
    needs_item_params = bool(
        scenario_choice_ab == "B"
        or "detect_pm" in ab_fns
        or "detect_ac" in ab_fns
        or "detect_pk" in ab_fns
    )
    if needs_item_params:
        st.info(
            "**Model misfit (detect_pm), answer copying (detect_ac), and preknowledge (detect_pk)** require IRT item parameters (ψ). "
            "When you click **Run**, the IRT agent will fit the model first to produce ψ, then the Aberrance agent will use them."
        )

    has_sidebar_data = bool(st.session_state.get("last_uploaded_responses"))
    ab_resp = None  # no page-specific uploaders; sidebar only
    ab_rt = None
    if has_sidebar_data:
        n_r = len(st.session_state.last_uploaded_responses)
        n_c = len(st.session_state.last_uploaded_responses[0]) if n_r else 0
        rt_txt = " + RT" if st.session_state.get("last_uploaded_rt_data") else ""
        st.caption(f"✓ Using data from **sidebar**: **{n_r}** rows × **{n_c}** items{rt_txt}.")
    else:
        st.warning("Upload **Response (CSV)** in the **sidebar** to run aberrance analysis.")
    if ab_fns and "detect_rg" in ab_fns and not st.session_state.get("last_uploaded_rt_data"):
        st.warning("**Rapid Guessing (detect_rg)** benefits from RT data. Upload RT in the **sidebar**.")
    # Status strip: what will run
    has_data = ab_resp is not None or has_sidebar_data
    status_parts = [f"Data: {'✓' if has_data else '✗'}"]
    if needs_item_params:
        status_parts.append("(IRT will run first for ψ)")
    st.markdown(f'<div class="status-strip">{ " | ".join(status_parts) }</div>', unsafe_allow_html=True)
    run_ab = st.button("Run", key="run_ab_only", type="primary", use_container_width=True)
    if run_ab and has_data:
        try:
            if ab_resp is not None:
                ab_resp.seek(0)
                ab_df = pd.read_csv(ab_resp)
                ab_df = _drop_index_column(ab_df)
                ab_df = _validate_binary_responses(ab_df)
                responses = ab_df.to_dict(orient="records")
                rt_data = []
                if ab_rt is not None:
                    try:
                        ab_rt.seek(0)
                        rt_df = pd.read_csv(ab_rt)
                        rt_df = _drop_index_column(rt_df)
                        rt_df = _coerce_numeric(rt_df, "RT")
                        if ab_df.shape[1] == rt_df.shape[1]:
                            rt_df = _align_rt_columns_to_response(rt_df, ab_df)
                            rt_data = rt_df.to_dict(orient="records")
                        else:
                            st.warning(f"RT file has {rt_df.shape[1]} columns but response has {ab_df.shape[1]}; RT must match. Rapid guessing will not run.")
                    except Exception as e:
                        st.warning(f"Could not load RT file (check numeric values and layout): {e}. Rapid guessing will not run.")
                model_settings = st.session_state.get("model_settings", {})
            else:
                responses = st.session_state.last_uploaded_responses
                rt_data = st.session_state.get("last_uploaded_rt_data") or []
                model_settings = st.session_state.get("last_uploaded_model_settings") or st.session_state.get("model_settings", {})
            # Item params: use from session if already from IRT, else IRT will run first when needed
            item_params_for_run = st.session_state.get("item_params") or []
            state = {
                "responses": responses,
                "rt_data": rt_data,
                "theta": 0.0,
                "latency_flags": [],
                "next_step": "",
                "model_settings": model_settings,
                "is_verified": True,
                "aberrance_functions": ab_fns if ab_fns else ["detect_nm"],
                "item_params": item_params_for_run,
                "compromised_items": st.session_state.get(ab_only_compromised_key) or [],
            }
            # When pm/ac/pk need item params and we don't have them, run IRT first to fit model and get ψ
            if needs_item_params and not item_params_for_run:
                with st.spinner("Running IRT agent to get item parameters (ψ)…"):
                    irt_out = irt_agent(state)
                state = {**state, **irt_out}
                if state.get("item_params"):
                    st.session_state["item_params"] = state["item_params"]
                    st.caption(f"✓ IRT produced {len(state['item_params'])} item parameters.")
                else:
                    irt_err = state.get("icc_error", "IRT did not return item parameters.")
                    st.warning(f"IRT could not produce item parameters: {irt_err}. Model misfit / answer copying / preknowledge will not run.")
            with st.spinner("Running Aberrance agent…"):
                out = aberrance_agent(state)
            st.session_state["aberrance_only_result"] = {**state, **out}
            st.success("Done.")
            st.rerun()
        except Exception as e:
            st.exception(e)
    elif run_ab and not has_data:
        st.error("Upload response data (CSV) in the **sidebar** to run the Aberrance tool.")

    # ----- 4. Results -----
    st.divider()
    st.markdown("**4. Results**")
    _ab_result = st.session_state.get("aberrance_only_result")
    if _ab_result:
        final_ab = _ab_result
        aberrance = final_ab.get("aberrance_results") or {}
        if aberrance.get("error"):
            st.warning(aberrance["error"])
        if aberrance.get("info"):
            st.info(aberrance["info"])
        # Diagnostic expander: shows exactly what ran and what was returned
        with st.expander("Run diagnostics", expanded=False):
            diag = []
            diag.append(f"**aberrance_functions sent:** {final_ab.get('aberrance_functions', '(not set)')}")
            ip = final_ab.get("item_params")
            diag.append(f"**item_params in state:** {'Yes, ' + str(len(ip)) + ' items' if ip else 'None / empty'}")
            if ip:
                ip0 = ip[0] if ip else {}
                diag.append(f"**item_params columns:** {list(ip0.keys())}")
            diag.append(f"**aberrance_results keys:** {list(aberrance.keys())}")
            diag.append(f"**parametric_misfit:** {'Yes, ' + str(len(aberrance.get('parametric_misfit', []))) + ' records' if aberrance.get('parametric_misfit') else 'None / empty'}")
            diag.append(f"**rapid_guessing:** {aberrance.get('rapid_guessing', False)}")
            diag.append(f"**rapid_guessing_error:** {aberrance.get('rapid_guessing_error', '(none)')}")
            diag.append(f"**methods:** {aberrance.get('methods', [])}")
            diag.append(f"**info:** {aberrance.get('info', '(none)')}")
            diag.append(f"**error:** {aberrance.get('error', '(none)')}")
            st.markdown("\n\n".join(diag))
        if "nonparametric_misfit" in aberrance or "parametric_misfit" in aberrance or "preknowledge" in aberrance or "answer_copying_pairs" in aberrance or aberrance.get("rapid_guessing"):
            flagged_persons = set(aberrance.get("flagged_persons") or [])
            flagged_copiers = set(aberrance.get("flagged_copiers") or [])
            flagged_rg_ab = set(aberrance.get("flagged_persons_rg") or [])
            n_persons = aberrance.get("n_persons") or 0
            n_flagged = aberrance.get("n_flagged", 0)
            methods_display = aberrance.get("methods") or []
            # Build one table from all per-person results: nm, pm, preknowledge (same row count)
            df_all = None
            for key in ("nonparametric_misfit", "parametric_misfit", "preknowledge"):
                recs = aberrance.get(key) or []
                if not recs or (df_all is not None and len(recs) != len(df_all)):
                    continue
                part = pd.DataFrame(recs)
                prefix = {"preknowledge": "PK_"}.get(key, "")
                if prefix and len(part.columns) > 0:
                    part = part.add_prefix(prefix)
                if df_all is None:
                    df_all = part.copy()
                else:
                    for c in part.columns:
                        if c not in df_all.columns:
                            df_all[c] = part[c]
                if n_persons == 0:
                    n_persons = len(part)
            if df_all is not None and len(df_all) > 0:
                n_persons = n_persons or len(df_all)
                df_all.insert(0, "Person", range(1, len(df_all) + 1))
                if "answer_copying_pairs" in aberrance:
                    df_all["Flagged_copying"] = [1 if i in flagged_copiers else 0 for i in range(len(df_all))]
                if aberrance.get("rapid_guessing"):
                    df_all["Flagged_rg"] = [1 if i in flagged_rg_ab else 0 for i in range(len(df_all))]
                df_all["Flagged"] = [1 if i in flagged_persons else 0 for i in range(len(df_all))]
                cap = f"**{n_flagged}** of **{n_persons}** persons flagged. "
                st.caption(cap)
                st.dataframe(df_all, height=min(450, 120 + 32 * len(df_all)), use_container_width=True)
            elif n_persons and n_persons > 0:
                df_fallback = pd.DataFrame({
                    "Person": range(1, int(n_persons) + 1),
                    "Flagged_copying": [1 if i in flagged_copiers else 0 for i in range(int(n_persons))],
                    "Flagged": [1 if i in flagged_persons else 0 for i in range(int(n_persons))],
                })
                st.caption(f"Person-fit statistics were not returned (n_persons={n_persons}, n_flagged={n_flagged}). Methods: {methods_display}. Ensure **detect_nm** (Guttman) and/or **detect_pm** / **detect_pk** are selected; for model misfit and preknowledge the IRT agent will fit the model first when you click **Run**. For **preknowledge** enter **compromised item numbers**.")
                st.dataframe(df_fallback, height=min(450, 120 + 32 * int(n_persons)), use_container_width=True)
            else:
                st.caption(f"No person-level records (n_persons={n_persons}, n_flagged={n_flagged}). Methods: {methods_display}")
            # Answer copying: show source–copier pairs in expander
            ac_pairs = aberrance.get("answer_copying_pairs") or []
            if ac_pairs:
                with st.expander("Answer copying: source–copier pairs", expanded=False):
                    df_ac = pd.DataFrame(ac_pairs)
                    st.dataframe(df_ac, height=min(300, 100 + 32 * len(df_ac)), use_container_width=True)
                    st.caption("Source = suspected source; Copier = suspected copier. Other columns are copying statistics (e.g. OMG_S, GBT_S).")
        elif not aberrance.get("error") and not aberrance.get("info"):
            st.caption("Result returned but no person-fit table (keys: " + ", ".join(aberrance.keys()) + ").")
    else:
        st.caption("No results yet. Upload response data above and click **Run**.")
    st.caption("Use the **sidebar** to switch to Main or another module.")


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
            st.caption(f"✓ Using data from **sidebar**: **{n_r}** rows × **{n_c}** items{rt_txt}.")
        else:
            st.warning("Upload **Response (CSV)** in the **sidebar** to run the IRT agent.")

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
                st.error("Upload response data in the **sidebar** to run the IRT agent.")
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
                    out = irt_agent(state)
                    st.session_state.irt_only_result = {**state, **out}
                    st.success("IRT finished.")
                    st.rerun()
                except Exception as e:
                    st.exception(e)
        if st.session_state.get("irt_only_result"):
            st.subheader("Last results")
            _render_results(st.session_state.irt_only_result, response_only=True)
    st.caption("Use the **sidebar** to switch to Main or another module.")


def _render_tool_rt() -> None:
    st.markdown("---")
    st.subheader("RT (response time)")
    has_sidebar_rt = bool(st.session_state.get("last_uploaded_responses")) and bool(st.session_state.get("last_uploaded_rt_data"))
    if has_sidebar_rt:
        n_r = len(st.session_state.last_uploaded_responses)
        n_c = len(st.session_state.last_uploaded_responses[0]) if n_r else 0
        st.caption(f"✓ Using data from **sidebar**: **{n_r}** rows × **{n_c}** items + RT.")
    else:
        st.warning("Upload **Response (CSV)** and **RT (CSV)** in the **sidebar** to run the RT agent.")
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
                    out = rt_agent(state)
                st.session_state.rt_only_result = {**state, **out}
                st.success("Done.")
                st.rerun()
            except Exception as e:
                st.exception(e)
        else:
            st.error("Upload response and RT in the **sidebar** to run the RT agent.")
    if st.session_state.get("rt_only_result"):
        final_rt = st.session_state.rt_only_result
        st.subheader("Latency flags")
        if final_rt.get("latency_flags"):
            st.write(", ".join(final_rt["latency_flags"]))
        else:
            st.info("No latency flags (RT agent returns flags when implemented).")
    st.caption("Use the **sidebar** to switch to Main or another module.")


if run_mode == "Preparation":
    load_dotenv()

    st.markdown(
        """
        <style>
          .psymas-strip {
            padding: 10px 12px;
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.10);
            background: rgba(255,255,255,0.02);
            font-size: 0.9rem;
            margin-bottom: 10px;
          }
          /* Preparation frames (three matching cards, fixed visual height) */
          div[data-testid="stContainer"]:has(.psymas-prep-card-marker) {
            border-radius: 14px !important;
            border: 1px solid rgba(255,255,255,0.10) !important;
            background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.015)) !important;
            box-shadow: 0 10px 28px rgba(0,0,0,0.28);
            height: 320px !important;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            overflow-y: auto;
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
            padding-bottom: 8px;
            margin-bottom: 10px;
            border-bottom: 1px solid rgba(255,255,255,0.08);
          }
          .psymas-card-title {
            font-size: 0.95rem;
            font-weight: 650;
            letter-spacing: 0.2px;
          }
          .psymas-card-status {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            font-size: 0.85rem;
            opacity: 0.92;
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
          .psymas-dot-bad { background: #d50000; }
          .psymas-pill {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.12);
            margin-right: 8px;
            margin-top: 2px;
            background: rgba(255,255,255,0.01);
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

    provider = st.session_state.get("llm_provider", "openrouter")
    model_id = _effective_llm_model()
    or_key = os.getenv("OPENROUTER_API_KEY", "")
    g_key = os.getenv("GOOGLE_API_KEY", "")
    llm_configured = bool(or_key.strip()) if provider == "openrouter" else bool(g_key and g_key.strip())
    llm_short = f"{provider} · {model_id}"

    _dot = lambda ok: f"<span class='psymas-dot {'psymas-dot-ready' if ok else 'psymas-dot-bad'}'></span>"
    _esc = lambda s: (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

    # Detect button is always green; it is disabled until everything is ready.
    all_ready = bool(resp_loaded and rt_loaded and psi_loaded and llm_configured)
    _run_bg = "#00c853"
    _run_bg_hover = "#00b248"
    st.markdown(
        f"""
        <style>
          /* Style ONLY the Detect button (scoped via marker div) */
          div[data-testid="stContainer"]:has(.psymas-runforensic-marker) div[data-testid="stButton"] > button {{
            background: {_run_bg} !important;
            border: 1px solid rgba(255,255,255,0.18) !important;
            color: white !important;
            font-weight: 650 !important;
          }}
          div[data-testid="stContainer"]:has(.psymas-runforensic-marker) div[data-testid="stButton"] > button:disabled {{
            opacity: 1 !important;
            cursor: not-allowed !important;
          }}
          div[data-testid="stContainer"]:has(.psymas-runforensic-marker) div[data-testid="stButton"] > button:hover {{
            background: {_run_bg_hover} !important;
            border-color: rgba(255,255,255,0.28) !important;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="psymas-strip">
          <span class="psymas-pill" title="{_esc('Response loaded' if resp_loaded else 'Response missing')}">{_dot(resp_loaded)}Response{(' ' + str(n_persons) + '×' + str(n_items)) if resp_loaded else ''}</span>
          <span class="psymas-pill" title="{_esc('RT loaded' if rt_loaded else 'RT missing')}">{_dot(rt_loaded)}RT</span>
          <span class="psymas-pill" title="{_esc('Item parameters (ψ) ready' if psi_loaded else 'Item parameters (ψ) missing')}">{_dot(psi_loaded)}ψ</span>
          <span class="psymas-pill" title="{_esc('LLM configured' if llm_configured else 'LLM key missing for selected provider')}">{_dot(llm_configured)}LLM ({llm_short})</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_upload, col_irt, col_llm = st.columns(3, gap="large")

    with col_upload:
        with st.container(border=True):
            st.markdown("<div class='psymas-prep-card-marker'></div>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="psymas-card-head">
                  <div class="psymas-card-title">Upload</div>
                  <div class="psymas-card-status">
                    <span title="{_esc('Response loaded' if resp_loaded else 'Response missing')}">{_dot(resp_loaded)}Resp</span>
                    <span title="{_esc('RT loaded' if rt_loaded else 'RT missing')}">{_dot(rt_loaded)}RT</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            up_resp = st.file_uploader("Response (CSV)", type=["csv"], key="main_resp_uploader")
            up_rt = st.file_uploader("RT (CSV, optional)", type=["csv"], key="main_rt_uploader")
            # Avoid infinite rerun loops: only process when the uploaded file changes.
            resp_sig = (getattr(up_resp, "name", None), getattr(up_resp, "size", None)) if up_resp is not None else None
            rt_sig = (getattr(up_rt, "name", None), getattr(up_rt, "size", None)) if up_rt is not None else None

            if resp_sig and st.session_state.get("prep_last_resp_sig") != resp_sig:
                try:
                    up_resp.seek(0)
                    _r = pd.read_csv(up_resp)
                    _r = _drop_index_column(_r)
                    _r = _validate_binary_responses(_r)
                    st.session_state.last_uploaded_responses = _r.to_dict(orient="records")
                    st.session_state["prep_last_resp_sig"] = resp_sig
                    # If RT is already selected, process it now too.
                    if rt_sig:
                        try:
                            up_rt.seek(0)
                            _rt = pd.read_csv(up_rt)
                            _rt = _drop_index_column(_rt)
                            _rt = _coerce_numeric(_rt, "RT")
                            if _r.shape[1] == _rt.shape[1]:
                                _rt = _align_rt_columns_to_response(_rt, _r)
                            st.session_state.last_uploaded_rt_data = _rt.to_dict(orient="records")
                            st.session_state["prep_last_rt_sig"] = rt_sig
                        except Exception:
                            st.session_state.last_uploaded_rt_data = []
                    st.rerun()
                except Exception as e:
                    st.error(f"{e}")

            # RT upload can happen without re-uploading Response
            if rt_sig and st.session_state.get("prep_last_rt_sig") != rt_sig and st.session_state.get("last_uploaded_responses"):
                try:
                    up_rt.seek(0)
                    _rt = pd.read_csv(up_rt)
                    _rt = _drop_index_column(_rt)
                    _rt = _coerce_numeric(_rt, "RT")
                    _r_prev = pd.DataFrame(st.session_state.last_uploaded_responses)
                    if _r_prev.shape[1] == _rt.shape[1]:
                        _rt = _align_rt_columns_to_response(_rt, _r_prev)
                    st.session_state.last_uploaded_rt_data = _rt.to_dict(orient="records")
                    st.session_state["prep_last_rt_sig"] = rt_sig
                    st.rerun()
                except Exception as e:
                    st.error(f"{e}")

    with col_irt:
        with st.container(border=True):
            st.markdown("<div class='psymas-prep-card-marker'></div>", unsafe_allow_html=True)
            ip = st.session_state.get("last_irt_item_params") or []
            ip_ready = bool(ip)
            st.markdown(
                f"""
                <div class="psymas-card-head">
                  <div class="psymas-card-title">Model</div>
                  <div class="psymas-card-status">{_dot(ip_ready)}{'Ready' if ip_ready else 'Not ready'}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            itemtype = st.selectbox("Model", ["2PL", "1PL", "3PL", "4PL"], index=0, key="main_irt_itemtype")
            comp_input = st.text_input("Compromised (optional)", key="prep_compromised_input", label_visibility="collapsed", placeholder="Compromised item IDs, e.g. 3,7,12")
            try:
                st.session_state["prep_compromised_items"] = [int(x.strip()) for x in (comp_input or "").split(",") if x.strip().isdigit()]
            except Exception:
                st.session_state["prep_compromised_items"] = []
            run_irt = st.button("Generate item parameters", key="main_generate_item_params", use_container_width=True)
            if run_irt:
                responses = st.session_state.get("last_uploaded_responses") or []
                rt_data = st.session_state.get("last_uploaded_rt_data") or []
                if not responses:
                    st.session_state["last_irt_error"] = "Upload Response first."
                    st.error(st.session_state["last_irt_error"])
                else:
                    # Use the same IRT state construction as Command Center
                    st.session_state["last_irt_error"] = None
                    with st.spinner(f"Running IRT ({itemtype}) to estimate item parameters…"):
                        try:
                            _cc_model_settings = {
                                **(st.session_state.get("model_settings") or {}),
                                "itemtype": itemtype,
                            }
                            irt_state = {
                                "responses": responses,
                                "rt_data": rt_data,
                                "theta": 0.0,
                                "latency_flags": [],
                                "next_step": "start",
                                "model_settings": _cc_model_settings,
                                "is_verified": True,
                            }
                            irt_result = irt_agent(irt_state)
                        except Exception as e:
                            irt_result = {}
                            st.session_state["last_irt_error"] = f"IRT estimation failed: {e}"
                    if irt_result.get("item_params"):
                        psi_data = irt_result["item_params"]
                        st.session_state["last_irt_item_params"] = psi_data
                        st.session_state["item_params"] = psi_data
                        if irt_result.get("person_params"):
                            st.session_state["forensic_person_params"] = irt_result["person_params"]
                        st.rerun()
                    else:
                        st.session_state["last_irt_error"] = (
                            st.session_state.get("last_irt_error")
                            or irt_result.get("icc_error")
                            or "IRT returned no item parameters."
                        )
                        st.error(st.session_state["last_irt_error"])
            if st.session_state.get("last_irt_error"):
                st.caption(st.session_state["last_irt_error"])

    with col_llm:
        with st.container(border=True):
            st.markdown("<div class='psymas-prep-card-marker'></div>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="psymas-card-head">
                  <div class="psymas-card-title">LLM</div>
                  <div class="psymas-card-status">{_dot(llm_configured)}{'Ready' if llm_configured else 'Not ready'}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if "llm_provider" not in st.session_state:
                st.session_state.llm_provider = "openrouter"
            st.radio(
                "Provider",
                options=["openrouter", "google"],
                key="llm_provider",
                horizontal=True,
                label_visibility="collapsed",
            )
            _model_ids = _current_model_ids()
            if "selected_gemini_model" not in st.session_state:
                st.session_state.selected_gemini_model = _model_ids[0] if _model_ids else ""
            if _model_ids and st.session_state.selected_gemini_model not in _model_ids:
                st.session_state.selected_gemini_model = _model_ids[0]
            st.selectbox(
                "Model",
                options=_model_ids,
                key="selected_gemini_model",
                label_visibility="collapsed",
            )
            test = st.button("Test LLM", key="main_test_llm", use_container_width=True)
            if test:
                load_dotenv()
                provider_now = st.session_state.get("llm_provider", "openrouter")
                if provider_now == "openrouter":
                    key = os.getenv("OPENROUTER_API_KEY", "")
                    ok, msg = _test_openrouter_api_key(key)
                    st.session_state["main_llm_test"] = (ok, msg)
                else:
                    key = os.getenv("GOOGLE_API_KEY", "")
                    ok, msg = _test_api_key(key)
                    st.session_state["main_llm_test"] = (ok, msg)
                st.rerun()
            if st.session_state.get("main_llm_test") is not None:
                ok, msg = st.session_state["main_llm_test"]
                (st.success if ok else st.error)(msg)

    st.divider()
    with st.container():
        st.markdown("<div class='psymas-runforensic-marker'></div>", unsafe_allow_html=True)
        run_forensic = st.button(
            "Detect",
            key="prep_run_forensic",
            use_container_width=True,
            disabled=not all_ready,
        )

    if run_forensic:
        # Start a dedicated progress page run (avoids doing heavy work on Preparation)
        _payload = {
            "itemtype": st.session_state.get("main_irt_itemtype", "2PL"),
            "compromised_items": st.session_state.get("prep_compromised_items") or [],
        }
        st.session_state["detect_payload"] = _payload
        st.session_state["detect_job_id"] = str(uuid.uuid4())
        st.session_state["detect_job_status"] = "pending"
        st.session_state["detect_job_started_at"] = time.time()
        st.session_state["detect_job_error"] = None
        st.session_state["_nav_request"] = "Detect Progress"
        st.rerun()

elif run_mode == "Detect Progress":
    load_dotenv()
    st.subheader("⏳ Detect Progress")
    st.caption("Running IRT (ψ) and 8 aberrance agents. Keep this page open until it finishes.")

    # Green button styling (same color as Detect)
    _run_bg = "#00c853"
    _run_bg_hover = "#00b248"
    st.markdown(
        f"""
        <style>
          div[data-testid="stContainer"]:has(.psymas-goresults-marker) div[data-testid="stButton"] > button {{
            background: {_run_bg} !important;
            border: 1px solid rgba(255,255,255,0.18) !important;
            color: white !important;
            font-weight: 650 !important;
          }}
          div[data-testid="stContainer"]:has(.psymas-goresults-marker) div[data-testid="stButton"] > button:hover {{
            background: {_run_bg_hover} !important;
            border-color: rgba(255,255,255,0.28) !important;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.get("last_uploaded_responses"):
        st.error("No Response data in session. Go back to **Preparation** and upload Response (and RT).")
        if st.button("Back to Preparation", use_container_width=True):
            st.session_state.run_mode = "Preparation"
            st.rerun()
    else:
        # Ensure there's a job record even if user navigates here directly.
        if "detect_job_id" not in st.session_state:
            st.session_state["detect_job_id"] = str(uuid.uuid4())
        if "detect_job_status" not in st.session_state:
            st.session_state["detect_job_status"] = "pending"
        if "detect_job_started_at" not in st.session_state:
            st.session_state["detect_job_started_at"] = time.time()

        _job_status = st.session_state.get("detect_job_status", "pending")
        if _job_status == "done" and st.session_state.get("forensic_result"):
            st.success("Detection completed.")
            with st.container():
                st.markdown("<div class='psymas-goresults-marker'></div>", unsafe_allow_html=True)
                if st.button("Go to Aberrance Summary", use_container_width=True, type="primary"):
                    st.session_state["_nav_request"] = "Aberrance Summary"
                    st.session_state.just_switched_module = "Aberrance Summary"
                    st.rerun()
        else:
            # Run now (synchronous) with live UI updates.
            st.session_state["detect_job_status"] = "running"
            st.session_state["detect_job_error"] = None

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
            _payload = st.session_state.get("detect_payload") or {}
            ci = _payload.get("compromised_items") or []
            _prep_itemtype = _payload.get("itemtype") or st.session_state.get("main_irt_itemtype", "2PL")

            st.markdown("**Working status**")
            # Placeholder for a LangGraph-style tree visualization (Graphviz), centered in the layout
            _left_spacer, tree_col, _right_spacer = st.columns([2, 7, 2])
            with tree_col:
                graph_placeholder = st.empty()

            # Define agents for LangGraph-style visualization (LangGraph-style tree).
            _agent_defs = [
                ("router", "Router"),
                ("pm_agent", "Model Misfit (detect_pm)"),
                ("nm_agent", "Guttman / Nonparametric (detect_nm)"),
                ("ac_agent", "Answer Copying (detect_ac)"),
                ("as_agent", "Answer Similarity (detect_as)"),
                ("rg_agent", "Rapid Guessing (detect_rg)"),
                ("cp_agent", "Change Point (detect_cp)"),
                ("tt_agent", "Tampering (detect_tt)"),
                ("pk_agent", "Preknowledge (detect_pk)"),
                ("synthesizer", "Manager synthesis (LLM)"),
            ]

            # Track node states for the tree. Values: "pending", "running", "done", "error".
            node_states: dict[str, str] = {
                "router": "pending",
                "pm_agent": "pending",
                "nm_agent": "pending",
                "ac_agent": "pending",
                "as_agent": "pending",
                "rg_agent": "pending",
                "cp_agent": "pending",
                "tt_agent": "pending",
                "pk_agent": "pending",
                "synthesizer": "pending",
            }

            def _render_agent_graph() -> None:
                """Render LangGraph-style tree using Graphviz, styled to be compact and modern."""
                color_map = {
                    "pending": "#4b5563",   # grey
                    "running": "#fbbf24",   # amber
                    "done": "#16a34a",      # green
                    "error": "#f87171",     # red
                }

                def _node_line(node_id: str, label: str) -> str:
                    state = node_states.get(node_id, "pending")
                    color = color_map.get(state, "#4b5563")
                    safe_label = label.replace('"', '\\"')
                    return (
                        f'"{node_id}" [label="{safe_label}", '
                        f'style="filled", fillcolor="{color}", fontcolor="white"];'
                    )

                lines = [
                    "digraph G {",
                    '  graph [bgcolor="#0e1117", ranksep=0.35, nodesep=0.22, margin=0.10];',
                    '  node [shape=box, width=0.9, height=0.28, fontsize=6, penwidth=0.6, fontname="Segoe UI", fixedsize=true];',
                    '  edge [color="#4b5563", penwidth=0.5, arrowsize=0.30];',
                    "  rankdir=TB;",
                    f"  {_node_line('router', 'Router')}",
                    "  {rank=same; "
                    + " ".join(f'\"{k}\"' for k in ["pm_agent", "nm_agent", "ac_agent", "as_agent"])
                    + "};",
                    "  {rank=same; "
                    + " ".join(f'\"{k}\"' for k in ["rg_agent", "cp_agent", "tt_agent", "pk_agent"])
                    + "};",
                    f"  {_node_line('pm_agent', 'Parametric')}",
                    f"  {_node_line('nm_agent', 'Nonparametric')}",
                    f"  {_node_line('ac_agent', 'Copying')}",
                    f"  {_node_line('as_agent', 'Similarity')}",
                    f"  {_node_line('rg_agent', 'Rapid guessing')}",
                    f"  {_node_line('cp_agent', 'Change point')}",
                    f"  {_node_line('tt_agent', 'Tampering')}",
                    f"  {_node_line('pk_agent', 'Preknowledge')}",
                    f"  {_node_line('synthesizer', 'Manager (LLM)')}",
                    # Router → specialists
                    '  "router" -> "pm_agent";',
                    '  "router" -> "nm_agent";',
                    '  "router" -> "ac_agent";',
                    '  "router" -> "as_agent";',
                    '  "router" -> "rg_agent";',
                    '  "router" -> "cp_agent";',
                    '  "router" -> "tt_agent";',
                    '  "router" -> "pk_agent";',
                    # Specialists → synthesizer
                    '  "pm_agent" -> "synthesizer";',
                    '  "nm_agent" -> "synthesizer";',
                    '  "ac_agent" -> "synthesizer";',
                    '  "as_agent" -> "synthesizer";',
                    '  "rg_agent" -> "synthesizer";',
                    '  "cp_agent" -> "synthesizer";',
                    '  "tt_agent" -> "synthesizer";',
                    '  "pk_agent" -> "synthesizer";',
                    "}",
                ]
                # Render using the full width of the center column for better readability
                graph_placeholder.graphviz_chart("\n".join(lines), use_container_width=True)

            # Initial render so the tree appears immediately on entering Detect Progress.
            _render_agent_graph()

            # IRT status (ψ generation) shown separately from the agent tree
            left, right = st.columns(2, gap="small")
            with left:
                irt_box = st.status("IRT (ψ generation)", state="running", expanded=False)
            with right:
                rt_box = st.status("Response Time data (RT)", state="complete" if rt_data else "error", expanded=False)
            _smooth_to(5, duration_s=0.15)

            psi_data = []
            try:
                _status.markdown(f"**Step 1/2** — IRT (`{_prep_itemtype}`)")
                _smooth_to(12, duration_s=0.20)
                _cc_model_settings = {
                    **(st.session_state.get("model_settings") or {}),
                    "itemtype": _prep_itemtype,
                }
                irt_state = {
                    "responses": responses,
                    "rt_data": rt_data,
                    "theta": 0.0,
                    "latency_flags": [],
                    "next_step": "start",
                    "model_settings": _cc_model_settings,
                    "is_verified": True,
                }
                irt_result = irt_agent(irt_state)
                if irt_result.get("item_params"):
                    psi_data = irt_result["item_params"]
                    st.session_state["last_irt_item_params"] = psi_data
                    st.session_state["item_params"] = psi_data
                    if irt_result.get("person_params"):
                        st.session_state["forensic_person_params"] = irt_result["person_params"]
                    irt_box.update(state="complete", expanded=False)
                else:
                    irt_box.update(state="error", expanded=False)
                    st.warning("IRT returned no item parameters. Agents requiring ψ (pm, ac, pk) will be limited.")
                _smooth_to(35, duration_s=0.30)
            except Exception as e:
                irt_box.update(state="error", expanded=False)
                st.warning(f"IRT estimation failed: {e}. Agents requiring ψ will be limited.")
                _smooth_to(35, duration_s=0.20)

            # Step 2: Forensic workflow (stream to update per-agent status)
            _status.markdown("**Step 2/2** — Aberrance detection (8 agents)")
            # Mark router as running once the aberrance phase starts and refresh the tree.
            node_states["router"] = "running"
            _render_agent_graph()
            _smooth_to(45, duration_s=0.20)

            _llm_provider_now = st.session_state.get("llm_provider", "openrouter")
            _effective_model_now = _effective_llm_model()
            _or_key = os.getenv("OPENROUTER_API_KEY", "")
            _google_key = os.getenv("GOOGLE_API_KEY", "")
            _chosen_provider = _llm_provider_now
            _chosen_model = _effective_model_now
            if _llm_provider_now == "openrouter" and _or_key:
                _candidates = _model_variants_with_selected_first()
                _picked = None
                _picked_err = None
                for _mid in _candidates:
                    ok, err, _elapsed = _test_openrouter_model(_or_key, _mid, timeout=20)
                    if ok:
                        _picked = _mid
                        break
                    _picked_err = err
                if _picked:
                    _chosen_model = _picked
                else:
                    st.warning(f"OpenRouter test failed for selected models. Last error: {_picked_err}")
            elif _llm_provider_now == "openrouter" and not _or_key and _google_key:
                _chosen_provider = "google"

            _forensic_model_settings = {
                **(st.session_state.get("model_settings") or {}),
                "llm_provider": _chosen_provider,
                "openrouter_api_key": _or_key,
                "google_api_key": _google_key,
                "llm_model_id": _chosen_model,
            }
            forensic_state = {
                "responses": responses,
                "rt_data": rt_data,
                "theta": 0.0,
                "latency_flags": [],
                "next_step": "start",
                "model_settings": _forensic_model_settings,
                "is_verified": True,
                "psi_data": psi_data,
                "compromised_items": ci,
                "flags": {},
                "final_report": "",
            }

            result = dict(forensic_state)
            result.setdefault("flags", {})

            def _merge_state_fragment(fragment: dict) -> None:
                if not isinstance(fragment, dict):
                    return
                for kk, vv in fragment.items():
                    if kk == "flags" and isinstance(vv, dict):
                        result["flags"] = {**(result.get("flags") or {}), **vv}
                    else:
                        result[kk] = vv

            done: set[str] = set()
            expected = len(_agent_defs)

            def _mark_done(node_name: str) -> None:
                if node_name in done:
                    return
                done.add(node_name)
                maybe_err = None
                try:
                    if node_name not in ("router", "synthesizer"):
                        maybe_err = (result.get("flags") or {}).get(node_name, {}).get("error")
                except Exception:
                    maybe_err = None
                # Update node state for graph
                if node_name in node_states:
                    node_states[node_name] = "error" if maybe_err else "done"
                _render_agent_graph()
                # Progress advances with completions (after IRT we are ~35%).
                p = 35 + int(60 * (len(done) / max(1, expected)))
                _smooth_to(min(97, p), duration_s=0.10)

            try:
                stream_ok = False
                try:
                    for chunk in forensic_workflow.stream(forensic_state, stream_mode="updates"):
                        stream_ok = True
                        if not isinstance(chunk, dict):
                            continue
                        if any(k in chunk for k in ("flags", "final_report")):
                            _merge_state_fragment(chunk)
                            continue
                        for node_name, fragment in chunk.items():
                            if isinstance(fragment, dict):
                                _merge_state_fragment(fragment)
                            _mark_done(node_name)
                except TypeError:
                    for chunk in forensic_workflow.stream(forensic_state):
                        stream_ok = True
                        if not isinstance(chunk, dict):
                            continue
                        if any(k in chunk for k in ("flags", "final_report")):
                            _merge_state_fragment(chunk)
                            continue
                        for node_name, fragment in chunk.items():
                            if isinstance(fragment, dict):
                                _merge_state_fragment(fragment)
                            _mark_done(node_name)

                if not stream_ok:
                    result = forensic_workflow.invoke(forensic_state)
                    for node_name, _label in _agent_defs:
                        if node_name in node_states:
                            node_states[node_name] = "done"
                    _render_agent_graph()

                # Ensure any not-emitted nodes are marked complete and mark synthesizer done
                for node_name, _label in _agent_defs:
                    if node_name not in done and node_name in node_states:
                        node_states[node_name] = "done"
                node_states["synthesizer"] = "done"
                _render_agent_graph()
                _smooth_to(100, duration_s=0.25)

                st.session_state["forensic_result"] = result
                st.session_state["forensic_responses"] = responses
                st.session_state["forensic_rt_data"] = rt_data
                st.session_state["forensic_psi_data"] = psi_data
                st.session_state["detect_job_status"] = "done"
                st.success("Detection completed.")
                with st.container():
                    st.markdown("<div class='psymas-goresults-marker'></div>", unsafe_allow_html=True)
                    if st.button("Go to Aberrance Summary", use_container_width=True, type="primary"):
                        st.session_state.run_mode = "Aberrance Summary"
                        st.session_state.just_switched_module = "Aberrance Summary"
                        st.rerun()
            except Exception as e:
                st.session_state["detect_job_status"] = "error"
                st.session_state["detect_job_error"] = str(e)
                st.exception(e)
                if st.button("Back to Preparation", use_container_width=True):
                    st.session_state.run_mode = "Preparation"
                    st.rerun()

elif run_mode == "Data review":
    resp_data = st.session_state.get("last_uploaded_responses") or []
    rt_data = st.session_state.get("last_uploaded_rt_data") or []
    if not resp_data:
        st.info("Upload **Response (CSV)** in the **sidebar** to see the data table and distribution here.")
    else:
        resp_df = pd.DataFrame(resp_data)
        n_rows, n_cols = resp_df.shape
        st.caption(f"Session data: **{n_rows}** persons × **{n_cols}** items." + (" + RT" if rt_data else ""))

        # ----- Response data table -----
        tab_resp, tab_rt, tab_stats = st.tabs(["Response data", "RT data", "Item statistics"])
        with tab_resp:
            st.dataframe(resp_df, height=min(400, 120 + 32 * n_rows), use_container_width=True)
        with tab_rt:
            if rt_data:
                rt_df = pd.DataFrame(rt_data)
                st.dataframe(rt_df, height=min(400, 120 + 32 * len(rt_df)), use_container_width=True)
            else:
                st.info("No RT data in session. Upload **RT (CSV)** in the sidebar.")
        with tab_stats:
            # Per-item statistics
            numeric_resp = resp_df.apply(pd.to_numeric, errors="coerce")
            stats = pd.DataFrame({
                "Item": numeric_resp.columns,
                "N": numeric_resp.count().values,
                "p (mean)": numeric_resp.mean().round(3).values,
                "SD": numeric_resp.std().round(3).values,
            })
            if rt_data:
                rt_df = pd.DataFrame(rt_data)
                rt_num = rt_df.iloc[:, :n_cols].apply(pd.to_numeric, errors="coerce")
                stats["RT mean"] = rt_num.mean().round(2).values
                stats["RT SD"] = rt_num.std().round(2).values
            st.dataframe(stats, height=min(400, 120 + 32 * len(stats)), use_container_width=True)

        # ----- Distribution figures -----
        st.markdown("**Distributions**")
        fig, axes = plt.subplots(1, 2 if rt_data else 1, figsize=(10, 4) if rt_data else (6, 4))
        if not rt_data:
            axes = [axes]
        # Left: distribution of total score (sum across items per person)
        score = resp_df.apply(pd.to_numeric, errors="coerce").sum(axis=1)
        axes[0].hist(score, bins=min(30, max(1, int(score.nunique()))), edgecolor="black", alpha=0.7)
        axes[0].set_xlabel("Total score")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title("Score distribution")
        axes[0].grid(True, alpha=0.3)
        if rt_data:
            rt_df = pd.DataFrame(rt_data)
            rt_block = rt_df.iloc[:, : min(rt_df.shape[1], n_cols)].apply(pd.to_numeric, errors="coerce").fillna(0)
            mean_rt = rt_block.mean(axis=1)
            axes[1].hist(mean_rt, bins=min(30, max(1, int(mean_rt.nunique()))), edgecolor="black", alpha=0.7, color="orange")
            axes[1].set_xlabel("Mean RT (per person)")
            axes[1].set_ylabel("Frequency")
            axes[1].set_title("Response-time distribution")
            axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    st.caption("Use the **sidebar** to switch to another module.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Tools (Aberrance / IRT / RT)
# ══════════════════════════════════════════════════════════════════════════════
elif run_mode == "Tools":
    st.subheader("🧰 Tools")
    st.caption("Run one module at a time: aberrant behavior detection (person-fit), IRT modeling, or response-time (RT) analysis.")
    if "tools_mode" not in st.session_state:
        st.session_state["tools_mode"] = "Aberrance"
    _tools_mode = st.radio(
        "Choose a tool",
        options=["Aberrance", "IRT", "RT"],
        key="tools_mode",
        horizontal=True,
        label_visibility="collapsed",
    )
    st.divider()
    if _tools_mode == "Aberrance":
        _render_tool_aberrance()
    elif _tools_mode == "IRT":
        _render_tool_irt()
    else:
        _render_tool_rt()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Aberrance Summary (dashboard)
# ══════════════════════════════════════════════════════════════════════════════
elif run_mode == "Aberrance Summary":
    st.markdown("""
    <style>
    .command-header {background:linear-gradient(135deg,#1a1a2e,#16213e);color:#0ff;
    padding:20px;border-radius:10px;margin-bottom:20px;border-left:4px solid #0ff;}
    .command-header h2{color:#0ff;margin:0;}
    .command-header p{color:#a0a0b0;margin:5px 0 0 0;}
    .threat-critical {background:#2d0000;border-left:4px solid #ff0000;padding:15px;border-radius:5px;margin:10px 0;}
    .threat-warning {background:#2d2d00;border-left:4px solid #ffaa00;padding:15px;border-radius:5px;margin:10px 0;}
    .threat-clear {background:#002d00;border-left:4px solid #00ff00;padding:15px;border-radius:5px;margin:10px 0;}
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="command-header"><h2>🧾 Aberrance Summary</h2><p>Forensic verdict and flags</p></div>', unsafe_allow_html=True)

    st.caption("Run forensic analysis from **Preparation** (Detect), then review results here.")
    st.divider()

    # ── Dashboard (results) ──
    if not st.session_state.get("forensic_result"):
        st.info("Click **RUN FORENSIC ANALYSIS** above to generate the dashboard.")
    else:
        fr = st.session_state["forensic_result"]
        flags = fr.get("flags", {})
        final_report = fr.get("final_report", "")

        # ── Top Section: Generative Report ──
        st.subheader("Forensic Verdict")
        if final_report:
            report_lower = final_report.lower()
            if "critical" in report_lower:
                st.markdown(f'<div class="threat-critical">{final_report}</div>', unsafe_allow_html=True)
            elif "warning" in report_lower:
                st.markdown(f'<div class="threat-warning">{final_report}</div>', unsafe_allow_html=True)
            else:
                st.markdown(final_report)
        else:
            st.info("No report generated (LLM may not be configured).")
        st.divider()

        # ── Middle Section: Visuals ──
        vis_col1, vis_col2 = st.columns(2)

        # Collusion Graph
        with vis_col1:
            st.subheader("Collusion Graph")
            ac_data = flags.get("ac_agent", {})
            if ac_data.get("error"):
                st.caption(f"ac_agent error: {ac_data['error']}")
            elif ac_data.get("pairs"):
                try:
                    import networkx as nx
                    G = nx.Graph()
                    n_resp = len(st.session_state.get("forensic_responses", []))
                    for i in range(n_resp):
                        G.add_node(i + 1)
                    flagged_copiers = set(ac_data.get("flagged_copiers", []))
                    sig_pairs = []
                    for pair in ac_data["pairs"]:
                        src, cop = pair.get("Source", 0), pair.get("Copier", 0)
                        stats = {k: v for k, v in pair.items() if k not in ("Source", "Copier") and isinstance(v, (int, float))}
                        if any(abs(v) > 2 for v in stats.values()):
                            sig_pairs.append((src, cop))
                            G.add_edge(src, cop)
                    if sig_pairs:
                        fig_g, ax_g = plt.subplots(figsize=(6, 5))
                        pos = nx.spring_layout(G, seed=42)
                        node_colors = ["#ff4444" if (n - 1) in flagged_copiers else "#4488ff" for n in G.nodes()]
                        nx.draw(G, pos, ax=ax_g, with_labels=True, node_color=node_colors,
                                node_size=300, font_size=8, edge_color="#666666", width=1.5,
                                font_color="white")
                        ax_g.set_facecolor("#1a1a2e")
                        fig_g.patch.set_facecolor("#1a1a2e")
                        ax_g.set_title(f"Copying Network ({len(sig_pairs)} significant pairs)", color="white")
                        st.pyplot(fig_g)
                        plt.close(fig_g)
                    else:
                        st.caption("No significant copying pairs detected.")
                except ImportError:
                    st.caption("Install `networkx` for collusion graph visualization.")
                except Exception as e:
                    st.caption(f"Could not render collusion graph: {e}")
            else:
                st.caption("No answer-copying data available.")

        # Effort Matrix (Ability vs RTE)
        with vis_col2:
            st.subheader("Effort Matrix")
            rg_data = flags.get("rg_agent", {})
            nm_data = flags.get("nm_agent", {})
            if rg_data.get("error"):
                st.caption(f"rg_agent error: {rg_data['error']}")
            elif rg_data.get("rte"):
                rte_vals = rg_data["rte"]
                ability = []
                if nm_data.get("stat"):
                    for rec in nm_data["stat"]:
                        ability.append(float(rec.get("ZU3_S", 0)))
                else:
                    ability = list(range(len(rte_vals)))
                n = min(len(rte_vals), len(ability))
                if n > 0:
                    scatter_df = pd.DataFrame({
                        "Ability (ZU3_S)": ability[:n],
                        "RTE": rte_vals[:n],
                        "Student": [f"S{i+1}" for i in range(n)],
                    })
                    fig_e, ax_e = plt.subplots(figsize=(6, 5))
                    ax_e.set_facecolor("#1a1a2e")
                    fig_e.patch.set_facecolor("#1a1a2e")
                    med_ability = scatter_df["Ability (ZU3_S)"].median()
                    med_rte = scatter_df["RTE"].median()
                    colors = []
                    for _, row in scatter_df.iterrows():
                        if row["Ability (ZU3_S)"] > med_ability and row["RTE"] < med_rte:
                            colors.append("#ff4444")
                        elif row["RTE"] < med_rte:
                            colors.append("#ffaa00")
                        else:
                            colors.append("#44ff44")
                    ax_e.scatter(scatter_df["Ability (ZU3_S)"], scatter_df["RTE"],
                                 c=colors, s=40, alpha=0.7, edgecolors="white", linewidth=0.5)
                    ax_e.axhline(y=med_rte, color="#555555", linestyle="--", alpha=0.5)
                    ax_e.axvline(x=med_ability, color="#555555", linestyle="--", alpha=0.5)
                    ax_e.set_xlabel("Ability (ZU3_S)", color="white")
                    ax_e.set_ylabel("Response Time Effort (RTE)", color="white")
                    ax_e.set_title("Ability vs Effort", color="white")
                    ax_e.tick_params(colors="white")
                    ax_e.text(0.95, 0.05, "Smart Cheater\nZone", transform=ax_e.transAxes,
                              ha="right", va="bottom", color="#ff4444", fontsize=9, fontstyle="italic")
                    st.pyplot(fig_e)
                    plt.close(fig_e)
                else:
                    st.caption("Insufficient data for effort matrix.")
            else:
                st.caption("No rapid-guessing RTE data (upload RT data and rerun).")

        st.divider()

        # ── Bottom Section: The Watchlist ──
        st.subheader("The Watchlist — High-Risk Students")
        all_flagged = {}
        for agent_name, agent_data in flags.items():
            if not isinstance(agent_data, dict):
                continue
            flagged_list = agent_data.get("flagged", []) + agent_data.get("flagged_copiers", [])
            for p in flagged_list:
                if p not in all_flagged:
                    all_flagged[p] = {"Student_ID": p + 1, "Flagged_By": [], "Flags_Count": 0,
                                       "Misfit_Score": None, "RTE_Score": None, "Collusion_Partners": ""}
                all_flagged[p]["Flagged_By"].append(agent_name)
                all_flagged[p]["Flags_Count"] += 1

        nm_stat = flags.get("nm_agent", {}).get("stat", [])
        rg_rte = flags.get("rg_agent", {}).get("rte", [])
        ac_pairs = flags.get("ac_agent", {}).get("pairs", [])
        for p, info in all_flagged.items():
            if p < len(nm_stat) and nm_stat:
                info["Misfit_Score"] = round(float(nm_stat[p].get("ZU3_S", 0)), 3)
            if p < len(rg_rte) and rg_rte:
                info["RTE_Score"] = round(float(rg_rte[p]), 3)
            partners = set()
            for pair in ac_pairs:
                src, cop = pair.get("Source", 0), pair.get("Copier", 0)
                if src == p + 1:
                    partners.add(cop)
                elif cop == p + 1:
                    partners.add(src)
            if partners:
                info["Collusion_Partners"] = ", ".join(str(x) for x in sorted(partners))

        if all_flagged:
            watchlist_df = pd.DataFrame(sorted(all_flagged.values(), key=lambda x: -x["Flags_Count"]))
            watchlist_df["Flagged_By"] = watchlist_df["Flagged_By"].apply(lambda x: ", ".join(x))
            display_cols = ["Student_ID", "Flags_Count", "Misfit_Score", "RTE_Score", "Collusion_Partners", "Flagged_By"]
            st.dataframe(watchlist_df[display_cols], use_container_width=True, hide_index=True)
            st.caption("Navigate to **Student Profile** for drill-down analysis.")
        else:
            st.info("No students were flagged across any agents.")

        # Agent Status Overview
        with st.expander("Agent Status Summary"):
            for agent_name in ["nm_agent", "pm_agent", "ac_agent", "as_agent", "rg_agent", "cp_agent", "tt_agent", "pk_agent"]:
                data = flags.get(agent_name, {})
                if data.get("error"):
                    st.markdown(f"❌ **{agent_name}**: {data['error']}")
                elif data.get("info"):
                    st.markdown(f"ℹ️ **{agent_name}**: {data['info']}")
                else:
                    n_flagged = len(data.get("flagged", [])) + len(data.get("flagged_copiers", []))
                    st.markdown(f"✅ **{agent_name}**: {n_flagged} flagged")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Student Profile (Drill-Down)
# ══════════════════════════════════════════════════════════════════════════════
elif run_mode == "Student Profile":
    st.markdown("""
    <style>
    .profile-header {background:linear-gradient(135deg,#1a1a2e,#16213e);color:#e2b714;
    padding:20px;border-radius:10px;margin-bottom:20px;border-left:4px solid #e2b714;}
    .profile-header h2{color:#e2b714;margin:0;}
    .profile-header p{color:#a0a0b0;margin:5px 0 0 0;}
    .rap-pass {color:#00ff00;font-weight:bold;}
    .rap-fail {color:#ff4444;font-weight:bold;}
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="profile-header"><h2>👤 Student Profile</h2><p>Forensic drill-down for individual examinees</p></div>', unsafe_allow_html=True)

    if not st.session_state.get("forensic_result"):
        st.warning("No forensic results yet. Go to **Preparation** to run the analysis first.")
    else:
        fr = st.session_state["forensic_result"]
        flags = fr.get("flags", {})
        responses = st.session_state.get("forensic_responses", [])
        rt_data = st.session_state.get("forensic_rt_data", [])
        n_students = len(responses)

        if n_students == 0:
            st.warning("No student data available.")
        else:
            # ── Student Risk Bubble Grid ──
            import plotly.graph_objects as go
            import math

            _bub_agents = ["nm_agent", "pm_agent", "ac_agent", "as_agent", "rg_agent", "cp_agent", "tt_agent", "pk_agent"]
            _bub_agent_labels = ["NM", "PM", "AC", "AS", "RG", "CP", "TT", "PK"]
            # Count total flags per student
            _bub_totals = []
            _bub_details = []  # per-student detail strings for hover
            for sid_i in range(n_students):
                count = 0
                parts = []
                for ai, ag in enumerate(_bub_agents):
                    ag_data = flags.get(ag, {})
                    flagged_list = ag_data.get("flagged", []) + ag_data.get("flagged_copiers", []) + ag_data.get("flagged_pairs", [])
                    if sid_i in flagged_list:
                        count += 1
                        parts.append(_bub_agent_labels[ai])
                _bub_totals.append(count)
                _bub_details.append(", ".join(parts) if parts else "Clean")

            # Arrange students in a grid (columns determined by width)
            _bub_cols = min(20, n_students)
            _bub_rows = math.ceil(n_students / _bub_cols)
            _bub_x, _bub_y, _bub_ids, _bub_hover = [], [], [], []
            _bub_colors, _bub_sizes = [], []
            _bub_max_flags = max(max(_bub_totals), 1)
            for sid_i in range(n_students):
                col = sid_i % _bub_cols
                row = sid_i // _bub_cols
                _bub_x.append(col)
                _bub_y.append(row)
                _bub_ids.append(sid_i + 1)
                _bub_colors.append(_bub_totals[sid_i])
                # Size: minimum for 0 flags, scales up with flag count
                _bub_sizes.append(12 + _bub_totals[sid_i] * 8)
                _bub_hover.append(
                    f"Student {sid_i+1}<br>Flags: {_bub_totals[sid_i]} / {len(_bub_agents)}<br>{_bub_details[sid_i]}"
                )

            # Student ID labels (show inside bubble)
            _bub_text = [str(i + 1) for i in range(n_students)]

            _bub_fig = go.Figure(data=go.Scatter(
                x=_bub_x, y=_bub_y,
                mode="markers+text",
                marker=dict(
                    size=_bub_sizes,
                    color=_bub_colors,
                    colorscale=[[0, "#2d6a4f"], [0.15, "#40916c"], [0.35, "#fca311"], [0.6, "#e85d04"], [1, "#d00000"]],
                    cmin=0, cmax=_bub_max_flags,
                    line=dict(width=1, color="rgba(255,255,255,0.2)"),
                    colorbar=dict(
                        title=dict(text="Flags", font=dict(color="white")),
                        tickfont=dict(color="white"),
                        thickness=12, len=0.6,
                    ),
                ),
                text=_bub_text,
                textfont=dict(color="white", size=8),
                textposition="middle center",
                hovertext=_bub_hover,
                hovertemplate="%{hovertext}<extra></extra>",
                customdata=_bub_ids,
            ))
            _bub_height = max(300, _bub_rows * 50 + 100)
            _bub_fig.update_layout(
                title="Student Risk Overview — click a bubble to inspect",
                title_font_color="white",
                template="plotly_dark",
                paper_bgcolor="#1a1a2e",
                plot_bgcolor="#1a1a2e",
                height=_bub_height,
                margin=dict(t=45, b=20, l=20, r=40),
                xaxis=dict(visible=False, range=[-1, _bub_cols]),
                yaxis=dict(visible=False, range=[_bub_rows, -1]),  # reversed so row 0 is top
                showlegend=False,
            )

            _bub_event = st.plotly_chart(_bub_fig, use_container_width=True, on_select="rerun", key="profile_heatmap")

            # Handle click to select student
            if _bub_event and _bub_event.selection and _bub_event.selection.points:
                _clicked = _bub_event.selection.points[0]
                _clicked_cd = _clicked.get("customdata")
                if _clicked_cd is not None:
                    try:
                        _clicked_sid = int(_clicked_cd)
                        if 1 <= _clicked_sid <= n_students and _clicked_sid != st.session_state.get("profile_student_id", 1):
                            st.session_state["profile_student_id"] = _clicked_sid
                            st.rerun()
                    except (ValueError, TypeError):
                        pass

            st.divider()

            # Build flagged student list with details
            _all_agents = ["nm_agent", "pm_agent", "ac_agent", "as_agent", "rg_agent", "cp_agent", "tt_agent", "pk_agent"]
            _agent_short = {"nm_agent": "NM", "pm_agent": "PM", "ac_agent": "AC", "as_agent": "AS",
                            "rg_agent": "RG", "cp_agent": "CP", "tt_agent": "TT", "pk_agent": "PK"}
            _flagged_options = []  # list of (student_id_1based, label_str)
            for _si in range(n_students):
                _flag_names = []
                for _ag in _all_agents:
                    _ag_d = flags.get(_ag, {})
                    _fl = _ag_d.get("flagged", []) + _ag_d.get("flagged_copiers", []) + _ag_d.get("flagged_pairs", [])
                    if _si in _fl:
                        _flag_names.append(_agent_short[_ag])
                if _flag_names:
                    _flagged_options.append((_si + 1, f"Student {_si + 1}  —  {', '.join(_flag_names)} ({len(_flag_names)} flags)"))

            # Student selector — main content area
            # Build per-student flag map for filtering
            _student_flags_map = {}  # sid_1based -> set of short agent names
            for _si in range(n_students):
                _sflags = set()
                for _ag in _all_agents:
                    _ag_d = flags.get(_ag, {})
                    _fl = _ag_d.get("flagged", []) + _ag_d.get("flagged_copiers", []) + _ag_d.get("flagged_pairs", [])
                    if _si in _fl:
                        _sflags.add(_agent_short[_ag])
                if _sflags:
                    _student_flags_map[_si + 1] = _sflags

            # Determine which agents actually flagged anyone (for the filter)
            _active_agents = sorted({a for s in _student_flags_map.values() for a in s})

            _fcol1, _fcol2 = st.columns([1, 1])
            with _fcol1:
                _show_all = st.toggle("Show all students (not just flagged)", value=False, key="profile_show_all")
            with _fcol2:
                _agent_filter = st.multiselect(
                    "Filter by issue",
                    options=_active_agents if _active_agents else ["(none)"],
                    default=[],
                    key="profile_agent_filter",
                    help="Select one or more agents to show only students flagged by those specific checks.",
                    disabled=not _active_agents,
                )

            # Build filtered options
            if _show_all:
                _sel_options = []
                for _si in range(n_students):
                    _sflags = _student_flags_map.get(_si + 1, set())
                    # Apply agent filter if any selected
                    if _agent_filter and not _sflags.intersection(_agent_filter):
                        continue
                    if _sflags:
                        _sel_options.append((_si + 1, f"Student {_si + 1}  ⚠️ {', '.join(sorted(_sflags))} ({len(_sflags)} flags)"))
                    else:
                        _sel_options.append((_si + 1, f"Student {_si + 1}  ✅ Clean"))
            else:
                _sel_options = []
                for sid_1, sflags in sorted(_student_flags_map.items()):
                    if _agent_filter and not sflags.intersection(_agent_filter):
                        continue
                    _sel_options.append((sid_1, f"Student {sid_1}  —  {', '.join(sorted(sflags))} ({len(sflags)} flags)"))

            if not _sel_options:
                if _agent_filter:
                    st.info(f"No students flagged by **{', '.join(_agent_filter)}**. Clear the filter or toggle 'Show all'.")
                else:
                    st.success("No students were flagged by any agent — all clean.")
                _sel_options = [(_si + 1, f"Student {_si + 1}") for _si in range(n_students)]

            _sel_ids = [opt[0] for opt in _sel_options]
            _sel_labels = {opt[0]: opt[1] for opt in _sel_options}
            _prof_default = st.session_state.get("profile_student_id", _sel_ids[0] if _sel_ids else 1)
            if _prof_default not in _sel_ids:
                _prof_default = _sel_ids[0]
            selected_student = st.selectbox(
                "Select Student",
                options=_sel_ids,
                index=_sel_ids.index(_prof_default),
                format_func=lambda x: _sel_labels.get(x, f"Student {x}"),
                key="profile_student_id",
            )
            sid = selected_student - 1  # 0-based index

            st.subheader(f"Student {selected_student} of {n_students}")
            st.divider()

            # ── Response Table with Total Score ──
            if sid < len(responses):
                st.markdown("### Response Record")
                resp_row = responses[sid]
                resp_vals = list(resp_row.values())
                resp_keys = list(resp_row.keys())
                correct = sum(1 for v in resp_vals if v == 1)
                total_items = len(resp_vals)
                pct = 100 * correct / total_items if total_items > 0 else 0

                # Build a single-row DataFrame with item columns + Total
                import pandas as _pd_prof
                _resp_display = {str(k): v for k, v in resp_row.items()}
                _resp_display["TOTAL"] = f"{correct}/{total_items}"
                _resp_df = _pd_prof.DataFrame([_resp_display])
                _resp_df.index = [f"Student {selected_student}"]

                # Style: highlight incorrect answers (0) in red, correct (1) in green
                def _style_resp(val):
                    if val == 0:
                        return "background-color: #ff444444; color: #ff6b6b; font-weight: bold;"
                    elif val == 1:
                        return "background-color: #00ff0022; color: #40916c;"
                    else:
                        return "color: #e2b714; font-weight: bold;"
                st.dataframe(
                    _resp_df.style.map(_style_resp),
                    use_container_width=True, height=80,
                )
                st.caption(f"**Total Score: {correct} / {total_items} ({pct:.1f}%)**")
                st.divider()

            # ── Forensic Timeline ──
            st.markdown("### Forensic Timeline")
            if rt_data and sid < len(rt_data):
                student_rt = rt_data[sid]
                rt_values = [float(v) for v in student_rt.values() if v is not None]
                if rt_values:
                    import plotly.graph_objects as _go_tl
                    _tl_items = list(range(1, len(rt_values) + 1))
                    _tl_items_str = [str(i) for i in _tl_items]  # discrete labels

                    _tl_fig = _go_tl.Figure()
                    # Fill area
                    _tl_fig.add_trace(_go_tl.Scatter(
                        x=_tl_items_str, y=rt_values,
                        fill="tozeroy", fillcolor="rgba(0,255,255,0.08)",
                        line=dict(color="#0ff", width=2),
                        mode="lines+markers",
                        marker=dict(size=5, color="#0ff"),
                        name="Response Time",
                        hovertemplate="Item %{x}<br>RT: %{y:.2f}<extra></extra>",
                    ))
                    # Overlay change point from cp_agent
                    cp_data = flags.get("cp_agent", {})
                    if cp_data.get("stat") and sid < len(cp_data["stat"]):
                        cp_rec = cp_data["stat"][sid]
                        cp_val = None
                        for k, v in cp_rec.items():
                            if isinstance(v, (int, float)) and v > 0:
                                cp_val = v
                                break
                        if cp_val and 1 <= cp_val <= len(rt_values):
                            _tl_fig.add_vline(
                                x=str(int(cp_val)), line_dash="dash", line_color="#ff4444", line_width=2,
                                annotation_text=f"Change Point (item {int(cp_val)})",
                                annotation_font_color="#ff4444",
                                annotation_font_size=13,
                            )
                    _tl_fig.update_layout(
                        title=f"Response Time Across Test — Student {selected_student}",
                        xaxis_title="Item Number",
                        yaxis_title="Response Time",
                        template="plotly_dark",
                        paper_bgcolor="#1a1a2e",
                        plot_bgcolor="#0f3460",
                        font=dict(family="sans-serif", size=14, color="white"),
                        height=400,
                        margin=dict(t=50, b=50, l=60, r=30),
                        xaxis=dict(
                            type="category",
                            tickangle=-45 if len(_tl_items) > 30 else 0,
                            dtick=max(1, len(_tl_items) // 25),  # avoid overcrowding
                        ),
                        showlegend=False,
                    )
                    st.plotly_chart(_tl_fig, use_container_width=True, key=f"tl_{selected_student}")
                else:
                    st.caption("No valid RT values for this student.")
            else:
                st.caption("No response-time data available for forensic timeline.")

            st.divider()

            # ── The Rap Sheet ──
            st.markdown("### The Rap Sheet")
            # Short help texts matching the Aberrance page style
            agent_display = {
                "nm_agent": ("Guttman Errors (detect_nm)", "ZU3_S, HT_S", "Detects hard-right but easy-wrong patterns."),
                "pm_agent": ("Model Misfit (detect_pm)", "L_S_TS, L_T, Q_ST_TS, L_ST_TS", "Detects general odd behavior that doesn't fit standard models."),
                "ac_agent": ("Answer Copying (detect_ac)", "OMG_S, GBT_S", "Detects if a student copied from another."),
                "as_agent": ("Answer Similarity (detect_as)", "M4_S", "Detects suspicious groups with nearly identical answers."),
                "rg_agent": ("Rapid Guessing (detect_rg)", "RTE (NT method)", "Detects participants answering too quickly."),
                "cp_agent": ("Change Point (detect_cp)", "MCP", "Detects speed or performance shift mid-test."),
                "tt_agent": ("Test Tampering (detect_tt)", "EDI_SD", "Requires erasure data."),
                "pk_agent": ("Preknowledge (detect_pk)", "L_S, S_S, W_S", "Detects suspiciously well on leaked items."),
            }
            rap_cols = st.columns(2)
            for i, (agent_name, (display_name, methods, help_text)) in enumerate(agent_display.items()):
                with rap_cols[i % 2]:
                    agent_data = flags.get(agent_name, {})
                    if agent_data.get("error"):
                        st.checkbox(f"⚠️ {display_name}: {agent_data['error'][:60]}", value=False, disabled=True, help=help_text, key=f"rap_{agent_name}_{selected_student}")
                    elif agent_data.get("info"):
                        st.checkbox(f"ℹ️ {display_name}: {agent_data['info'][:60]}", value=False, disabled=True, help=help_text, key=f"rap_{agent_name}_{selected_student}")
                    else:
                        flagged = agent_data.get("flagged", [])
                        flagged_copiers = agent_data.get("flagged_copiers", [])
                        is_flagged = sid in flagged or sid in flagged_copiers
                        # Get p-value if available
                        pval_text = ""
                        if agent_data.get("stat") and sid < len(agent_data["stat"]):
                            rec = agent_data["stat"][sid]
                            pvals = {k: v for k, v in rec.items() if "_pval" in k.lower() and isinstance(v, (int, float))}
                            if pvals:
                                min_p = min(pvals.values())
                                pval_text = f" — p = {min_p:.4f}"
                        if is_flagged:
                            st.checkbox(f"❌ {display_name} (Flagged{pval_text})", value=True, disabled=True, help=help_text, key=f"rap_{agent_name}_{selected_student}")
                        else:
                            st.checkbox(f"✅ {display_name} (Pass{pval_text})", value=False, disabled=True, help=help_text, key=f"rap_{agent_name}_{selected_student}")

            st.divider()

            # ── Manager Insight ──
            st.markdown("### Manager Insight")
            # Check if student is in the final report
            final_report = fr.get("final_report", "")
            student_mentioned = f"Student {selected_student}" in final_report or f"student {selected_student}" in final_report.lower()
            if student_mentioned:
                # Extract relevant lines
                lines = final_report.split("\n")
                relevant = [l for l in lines if str(selected_student) in l]
                if relevant:
                    st.markdown("\n".join(relevant))
                else:
                    st.markdown(final_report)
            else:
                # Generate a brief summary from flags
                flagged_agents = []
                for agent_name, agent_data in flags.items():
                    if not isinstance(agent_data, dict):
                        continue
                    if sid in agent_data.get("flagged", []) or sid in agent_data.get("flagged_copiers", []):
                        flagged_agents.append(agent_name)
                if flagged_agents:
                    st.markdown(f"**Student {selected_student}** was flagged by: {', '.join(flagged_agents)}.")
                    # Specific insights per agent
                    if "ac_agent" in flagged_agents:
                        ac_pairs = flags.get("ac_agent", {}).get("pairs", [])
                        partners = set()
                        for p in ac_pairs:
                            if p.get("Source") == selected_student:
                                partners.add(p.get("Copier"))
                            elif p.get("Copier") == selected_student:
                                partners.add(p.get("Source"))
                        if partners:
                            st.markdown(f"- Potential copying relationship with student(s): {sorted(partners)}")
                    if "rg_agent" in flagged_agents:
                        rte_vals = flags.get("rg_agent", {}).get("rte", [])
                        if sid < len(rte_vals):
                            st.markdown(f"- Response Time Effort (RTE): {rte_vals[sid]:.3f}")
                    if "cp_agent" in flagged_agents:
                        cp_stat = flags.get("cp_agent", {}).get("stat", [])
                        if sid < len(cp_stat):
                            st.markdown(f"- Change point detected: {cp_stat[sid]}")
                else:
                    st.success(f"Student {selected_student} was not flagged by any agent.")

            # (Response table now shown at the top of the profile)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Collusion Network (Interactive Module A)
# ══════════════════════════════════════════════════════════════════════════════
elif run_mode == "Collusion Network":
    st.markdown("""
    <style>
    .colnet-header {background:linear-gradient(135deg,#1a1a2e,#16213e);color:#ff6b6b;
    padding:20px;border-radius:10px;margin-bottom:20px;border-left:4px solid #ff6b6b;}
    .colnet-header h2{color:#ff6b6b;margin:0;}
    .colnet-header p{color:#a0a0b0;margin:5px 0 0 0;}
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="colnet-header"><h2>Collusion Network</h2><p>Social graph of answer copying and answer similarity</p></div>', unsafe_allow_html=True)

    if not st.session_state.get("forensic_result"):
        st.warning("No forensic results yet. Go to **Preparation** to run the analysis first.")
    else:
        fr = st.session_state["forensic_result"]
        flags = fr.get("flags", {})
        ac_data = flags.get("ac_agent", {})
        as_data = flags.get("as_agent", {})
        n_students = len(st.session_state.get("forensic_responses", []))

        # Show individual agent errors/info
        if ac_data.get("error"):
            st.warning(f"**ac_agent (Answer Copying):** {ac_data['error']}")
        if as_data.get("error"):
            st.warning(f"**as_agent (Answer Similarity):** {as_data['error']}")
        if ac_data.get("error") and as_data.get("error"):
            pass  # both errored, warnings already shown above
        else:
            # Method filter
            available_methods = []
            if ac_data.get("pairs"):
                available_methods.extend(["OMG_S", "GBT_S"])
            if as_data.get("stat"):
                available_methods.append("M4_S")
            if not available_methods:
                st.info("No significant copying or similarity pairs found. Re-run the forensic analysis from **Preparation** to refresh results.")
            else:
                selected_methods = st.multiselect(
                    "Filter edges by method",
                    options=available_methods,
                    default=available_methods,
                    key="colnet_methods",
                )
                # Filtering mode: R-corrected flags (default, reliable) or raw p-value (exploratory)
                st.markdown("**Filtering mode**")
                _filter_mode = st.radio(
                    "Edge filter",
                    ["R-corrected flags (recommended)", "Raw p-value (exploratory)"],
                    index=0, horizontal=True, key="colnet_filter_mode",
                    help="R-corrected uses the flag output from detect_ac/detect_as which accounts for multiple comparisons. Raw p-value uses uncorrected p-values — expect many false positives with large samples.",
                )
                _use_raw_p = _filter_mode.startswith("Raw")
                sig_threshold = 0.05
                if _use_raw_p:
                    sig_threshold = st.slider("p-value threshold (uncorrected)", 0.0001, 0.05, 0.001, 0.0001, format="%.4f", key="colnet_sig")
                    st.caption(f"⚠️ With ~{n_students * (n_students - 1) // 2:,} pair comparisons, raw p < {sig_threshold:.4f} may still show false positives.")

                try:
                    from pyvis.network import Network
                    import streamlit.components.v1 as components
                    import tempfile

                    net = Network(
                        directed=True, height="600px", width="100%",
                        bgcolor="#1a1a2e", font_color="white",
                    )
                    net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=150)

                    # Collect nodes involved in edges (only add relevant nodes, not all N students)
                    _involved_nodes = set()
                    edge_count = 0
                    _edge_list = []  # (src, dst, width, color, title)

                    # ac_agent edges (directed: Source -> Copier)
                    if ac_data.get("pairs"):
                        for pair in ac_data["pairs"]:
                            src = pair.get("Source", 0)
                            cop = pair.get("Copier", 0)
                            is_rflagged = pair.get("flagged", False)
                            omg_stat = abs(float(pair.get("OMG_S", 0) or 0))
                            gbt_stat = abs(float(pair.get("GBT_S", 0) or 0))
                            omg_p = float(pair.get("OMG_S_pval", 1.0) or 1.0)
                            gbt_p = float(pair.get("GBT_S_pval", 1.0) or 1.0)
                            if "OMG_S" in selected_methods:
                                show = is_rflagged if not _use_raw_p else (omg_p < sig_threshold)
                                if show:
                                    width = max(1, min(omg_stat, 10))
                                    _edge_list.append((src, cop, width, "#ff6b6b", f"OMG_S={omg_stat:.2f} (p={omg_p:.4f})", "to"))
                                    _involved_nodes.update([src, cop])
                            if "GBT_S" in selected_methods:
                                show = is_rflagged if not _use_raw_p else (gbt_p < sig_threshold)
                                if show:
                                    width = max(1, min(gbt_stat, 10))
                                    _edge_list.append((src, cop, width, "#ffa502", f"GBT_S={gbt_stat:.2f} (p={gbt_p:.4f})", "to"))
                                    _involved_nodes.update([src, cop])

                    # as_agent edges (undirected: similarity clusters)
                    if "M4_S" in selected_methods and as_data.get("stat"):
                        for rec in as_data["stat"]:
                            m4_stat = abs(float(rec.get("M4_S", 0) or 0))
                            m4_p = float(rec.get("M4_S_pval", 1.0) or 1.0)
                            pair_tuple = rec.get("_pair")
                            if pair_tuple and len(pair_tuple) == 2:
                                # as_agent doesn't store per-pair flag, use p-value
                                show = m4_p < (sig_threshold if _use_raw_p else 0.001)
                                if show:
                                    s1, s2 = pair_tuple
                                    width = max(1, min(m4_stat, 10))
                                    _edge_list.append((s1 + 1, s2 + 1, width, "#2ed573", f"M4_S={m4_stat:.2f} (p={m4_p:.4f})", ""))
                                    _involved_nodes.update([s1 + 1, s2 + 1])

                    edge_count = len(_edge_list)

                    # Only add nodes that are involved in edges (cleaner graph)
                    flagged_copiers = set(ac_data.get("flagged_copiers", []))
                    as_flagged = set(as_data.get("flagged_pairs", []))
                    for nid in sorted(_involved_nodes):
                        idx0 = nid - 1
                        color = "#ff4444" if idx0 in flagged_copiers or idx0 in as_flagged else "#4488ff"
                        net.add_node(nid, label=f"S{nid}", color=color, size=18,
                                     title=f"Student {nid}")

                    for src, dst, width, color, title, arrows in _edge_list:
                        net.add_edge(src, dst, value=width, color=color, title=title, arrows=arrows)

                    _mode_label = "R-corrected flags" if not _use_raw_p else f"raw p < {sig_threshold:.4f}"
                    st.caption(f"Showing **{edge_count}** edges ({_mode_label}) for methods: {', '.join(selected_methods)}")

                    if edge_count > 0:
                        # Render pyvis HTML
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8") as f:
                            net.save_graph(f.name)
                            f.seek(0)
                        with open(f.name, "r", encoding="utf-8") as hf:
                            html_content = hf.read()
                        components.html(html_content, height=620, scrolling=False)
                    else:
                        st.success("No significant copying or similarity pairs detected — this is a **clean result**.")
                        st.caption("If you suspect copying, try switching to 'Raw p-value (exploratory)' mode with a small threshold.")

                    # Legend
                    with st.expander("Legend"):
                        st.markdown("""
- **Red edges**: OMG_S (answer copying — omega statistic)
- **Orange edges**: GBT_S (answer copying — GBT statistic)
- **Green edges**: M4_S (answer similarity — undirected)
- **Red nodes**: Flagged students (copiers or similar-pair members)
- **Blue nodes**: Non-flagged students
- **Arrow direction**: Source → Copier (from detect_ac pairs)
- **Edge thickness**: Proportional to statistic magnitude
- **R-corrected mode**: Uses R's `flag` output which corrects for multiple comparisons (~{0} pair tests)
- **Raw p-value mode**: Uses uncorrected p-values — many false positives expected
- **Hover** over edges to see the statistic and p-value
                        """.format(n_students * (n_students - 1) // 2))
                except ImportError:
                    st.error("The `pyvis` package is required for the Collusion Network. Install it with: `pip install pyvis`")
                except Exception as e:
                    st.error(f"Could not render collusion network: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Individual Aberrance — Quad-Chart (Interactive Module B)
# ══════════════════════════════════════════════════════════════════════════════
elif run_mode == "Individual Aberrance":
    st.markdown("""
    <style>
    .aberr-header {background:linear-gradient(135deg,#1a1a2e,#16213e);color:#ffa502;
    padding:20px;border-radius:10px;margin-bottom:20px;border-left:4px solid #ffa502;}
    .aberr-header h2{color:#ffa502;margin:0;}
    .aberr-header p{color:#a0a0b0;margin:5px 0 0 0;}
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="aberr-header"><h2>Individual Aberrance — Quad-Chart</h2><p>Ability vs Misfit colored by Response Time Effort</p></div>', unsafe_allow_html=True)

    if not st.session_state.get("forensic_result"):
        st.warning("No forensic results yet. Go to **Preparation** to run the analysis first.")
    else:
        import plotly.graph_objects as go

        fr = st.session_state["forensic_result"]
        flags = fr.get("flags", {})
        responses = st.session_state.get("forensic_responses", [])
        n_students = len(responses)

        # X-axis: theta
        person_params = st.session_state.get("forensic_person_params", [])
        theta_vals = []
        theta_source = "total score"
        if person_params:
            pp_df = pd.DataFrame(person_params)
            for col in ["F1", "theta", "ability", "F"]:
                if col in pp_df.columns:
                    theta_vals = pd.to_numeric(pp_df[col], errors="coerce").fillna(0).tolist()
                    theta_source = f"IRT ({col})"
                    break
        if not theta_vals and responses:
            # Fallback: total score as ability proxy
            theta_vals = [sum(1 for v in r.values() if v == 1) for r in responses]
            theta_source = "total score (IRT not available)"

        # Y-axis: L_ST_TS from pm_agent
        pm_data = flags.get("pm_agent", {})
        pm_stat = pm_data.get("stat", [])
        lst_ts_vals = []
        y_label = "L_ST_TS"
        if pm_stat:
            lst_ts_vals = [float(rec.get("L_ST_TS", 0) or 0) for rec in pm_stat]
        if not lst_ts_vals:
            # Fallback to ZU3_S from nm_agent
            nm_stat = flags.get("nm_agent", {}).get("stat", [])
            if nm_stat:
                lst_ts_vals = [float(rec.get("ZU3_S", 0) or 0) for rec in nm_stat]
                y_label = "ZU3_S (nm fallback)"

        # Color: RTE from rg_agent
        rg_data = flags.get("rg_agent", {})
        rte_vals = rg_data.get("rte", [])

        n = min(len(theta_vals), len(lst_ts_vals)) if lst_ts_vals else 0
        if n == 0:
            st.warning("Insufficient data for the quad-chart. Ensure pm_agent (or nm_agent) ran successfully and item parameters were available.")
            if pm_data.get("error"):
                st.caption(f"pm_agent error: {pm_data['error']}")
        else:
            # Build scatter data
            student_ids = [f"S{i+1}" for i in range(n)]
            theta_plot = theta_vals[:n]
            misfit_plot = lst_ts_vals[:n]

            # RTE for color (pad if shorter)
            if rte_vals and len(rte_vals) >= n:
                color_vals = rte_vals[:n]
                color_label = "RTE"
                colorscale = [[0, "#ff4444"], [0.5, "#ffa502"], [1.0, "#4488ff"]]
            else:
                color_vals = [0] * n
                color_label = "RTE (not available)"
                colorscale = [[0, "#4488ff"], [1, "#4488ff"]]

            fig = go.Figure()

            # Danger zone rectangle: theta > 2.0, L_ST_TS < -1.96
            fig.add_shape(
                type="rect",
                x0=2.0, x1=max(max(theta_plot) + 0.5, 3.5),
                y0=min(min(misfit_plot) - 0.5, -4), y1=-1.96,
                fillcolor="rgba(255,0,0,0.1)", line=dict(color="rgba(255,0,0,0.3)", dash="dash"),
                layer="below",
            )
            fig.add_annotation(
                x=max(max(theta_plot), 2.5), y=min(min(misfit_plot), -2.5),
                text="Danger Zone<br>(High Ability + High Misfit)",
                showarrow=False, font=dict(color="#ff4444", size=11),
            )

            # Reference lines
            fig.add_hline(y=-1.96, line_dash="dash", line_color="rgba(255,255,255,0.3)",
                          annotation_text="p = .05", annotation_position="top right")
            fig.add_vline(x=2.0, line_dash="dash", line_color="rgba(255,255,255,0.3)")

            # Scatter points
            fig.add_trace(go.Scatter(
                x=theta_plot, y=misfit_plot,
                mode="markers",
                marker=dict(
                    size=10, color=color_vals,
                    colorscale=colorscale,
                    colorbar=dict(title=dict(text=color_label, font=dict(color="white")), tickfont=dict(color="white")),
                    line=dict(width=1, color="white"),
                ),
                text=student_ids,
                customdata=list(range(1, n + 1)),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    f"{theta_source}: %{{x:.2f}}<br>"
                    f"{y_label}: %{{y:.3f}}<br>"
                    f"{color_label}: %{{marker.color:.3f}}<br>"
                    "<extra></extra>"
                ),
            ))

            fig.update_layout(
                xaxis_title=f"Ability ({theta_source})",
                yaxis_title=y_label,
                template="plotly_dark",
                paper_bgcolor="#1a1a2e",
                plot_bgcolor="#0f3460",
                height=550,
                margin=dict(t=40, b=60),
            )

            event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="quad_chart")

            # Handle click to navigate to Student Profile
            if event and event.selection and event.selection.points:
                clicked_point = event.selection.points[0]
                clicked_idx = clicked_point.get("point_index", 0)
                clicked_student = clicked_idx + 1
                st.info(f"Selected: **Student {clicked_student}**")
                if st.button(f"Go to Student Profile for Student {clicked_student}", key="goto_profile_from_quad"):
                    st.session_state["profile_student_id"] = clicked_student
                    st.session_state.run_mode = "Student Profile"
                    st.rerun()

            # Summary stats
            with st.expander("Interpretation Guide"):
                st.markdown(f"""
**Axes:**
- **X-axis**: Estimated ability ({theta_source}). Higher = more able.
- **Y-axis**: {y_label} — standardized joint likelihood statistic. More negative = worse fit.

**Color**: Response Time Effort (RTE) from rapid-guessing detection.
- **Red** = low effort (rapid guessing suspected)
- **Blue** = high effort (normal engagement)

**Danger Zone** (shaded red): Students with high ability (> 2.0) AND high misfit (< -1.96).
These are "Smart Cheaters" — they score well but their response patterns don't match the expected model, possibly due to preknowledge or answer copying.

**Interaction**: Click any point to navigate to that student's detailed profile.
                """)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Temporal Forensics (Interactive Module C)
# ══════════════════════════════════════════════════════════════════════════════
elif run_mode == "Temporal Forensics":
    st.markdown("""
    <style>
    .temp-header {background:linear-gradient(135deg,#1a1a2e,#16213e);color:#7bed9f;
    padding:20px;border-radius:10px;margin-bottom:20px;border-left:4px solid #7bed9f;}
    .temp-header h2{color:#7bed9f;margin:0;}
    .temp-header p{color:#a0a0b0;margin:5px 0 0 0;}
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="temp-header"><h2>Temporal & Tampering Forensics</h2><p>Change points, erasure analysis, and preknowledge detection</p></div>', unsafe_allow_html=True)

    if not st.session_state.get("forensic_result"):
        st.warning("No forensic results yet. Go to **Preparation** to run the analysis first.")
    else:
        import plotly.graph_objects as go
        import math as _math_tf

        fr = st.session_state["forensic_result"]
        flags = fr.get("flags", {})
        responses = st.session_state.get("forensic_responses", [])
        rt_data = st.session_state.get("forensic_rt_data", [])
        n_students = len(responses)

        # ── Change-Point Bubble Grid ──
        if n_students > 0:
            _tf_agents = ["cp_agent", "pk_agent", "tt_agent"]
            _tf_labels = ["CP", "PK", "TT"]
            _tf_totals = []
            _tf_details = []
            for sid_i in range(n_students):
                count = 0
                parts = []
                for ai, ag in enumerate(_tf_agents):
                    ag_data = flags.get(ag, {})
                    flagged_list = ag_data.get("flagged", []) + ag_data.get("flagged_copiers", []) + ag_data.get("flagged_pairs", [])
                    if sid_i in flagged_list:
                        count += 1
                        parts.append(_tf_labels[ai])
                _tf_totals.append(count)
                _tf_details.append(", ".join(parts) if parts else "Clean")

            _tf_cols = min(20, n_students)
            _tf_rows = _math_tf.ceil(n_students / _tf_cols)
            _tf_x, _tf_y, _tf_ids, _tf_hover = [], [], [], []
            _tf_colors, _tf_sizes, _tf_text = [], [], []
            _tf_max = max(max(_tf_totals), 1)
            for sid_i in range(n_students):
                _tf_x.append(sid_i % _tf_cols)
                _tf_y.append(sid_i // _tf_cols)
                _tf_ids.append(sid_i + 1)
                _tf_colors.append(_tf_totals[sid_i])
                _tf_sizes.append(12 + _tf_totals[sid_i] * 8)
                _tf_hover.append(f"Student {sid_i+1}<br>Flags: {_tf_totals[sid_i]} (CP/PK/TT)<br>{_tf_details[sid_i]}")
                _tf_text.append(str(sid_i + 1))

            _tf_fig = go.Figure(data=go.Scatter(
                x=_tf_x, y=_tf_y,
                mode="markers+text",
                marker=dict(
                    size=_tf_sizes,
                    color=_tf_colors,
                    colorscale=[[0, "#2d6a4f"], [0.35, "#fca311"], [0.7, "#e85d04"], [1, "#d00000"]],
                    cmin=0, cmax=_tf_max,
                    line=dict(width=1, color="rgba(255,255,255,0.2)"),
                    colorbar=dict(
                        title=dict(text="Flags", font=dict(color="white")),
                        tickfont=dict(color="white"),
                        thickness=12, len=0.6,
                    ),
                ),
                text=_tf_text,
                textfont=dict(color="white", size=8),
                textposition="middle center",
                hovertext=_tf_hover,
                hovertemplate="%{hovertext}<extra></extra>",
                customdata=_tf_ids,
            ))
            _tf_height = max(280, _tf_rows * 50 + 100)
            _tf_fig.update_layout(
                title="Temporal & Tampering Risk — click a bubble to inspect",
                title_font_color="white",
                template="plotly_dark",
                paper_bgcolor="#1a1a2e",
                plot_bgcolor="#1a1a2e",
                height=_tf_height,
                margin=dict(t=45, b=20, l=20, r=40),
                xaxis=dict(visible=False, range=[-1, _tf_cols]),
                yaxis=dict(visible=False, range=[_tf_rows, -1]),
                showlegend=False,
            )

            _tf_event = st.plotly_chart(_tf_fig, use_container_width=True, on_select="rerun", key="tf_bubble_grid")

            # Handle click to select student for the Speed Shift tab
            if _tf_event and _tf_event.selection and _tf_event.selection.points:
                _tf_clicked = _tf_event.selection.points[0]
                _tf_cd = _tf_clicked.get("customdata")
                if _tf_cd is not None:
                    try:
                        _tf_sid = int(_tf_cd)
                        if 1 <= _tf_sid <= n_students and _tf_sid != st.session_state.get("temp_cp_student", 1):
                            st.session_state["temp_cp_student"] = _tf_sid
                            st.rerun()
                    except (ValueError, TypeError):
                        pass

            st.divider()

        tab_speed, tab_erasure, tab_pk = st.tabs(["Speed Shift Tracker", "Erasure Heatmap", "Compromised List"])

        # ── Tab 1: Speed Shift Tracker (detect_cp) ──
        with tab_speed:
            st.markdown("#### Speed Shift Tracker")
            st.caption("Response time across the test with change-point overlay from detect_cp.")

            if not rt_data:
                st.info("No response-time data available. Upload RT data in the Evidence Room and rerun.")
            elif n_students == 0:
                st.info("No student data available.")
            else:
                cp_data = flags.get("cp_agent", {})
                if cp_data.get("error"):
                    st.warning(f"cp_agent error: {cp_data['error']}")

                _tf_default = st.session_state.get("temp_cp_student", 1)
                if _tf_default < 1 or _tf_default > n_students:
                    _tf_default = 1
                sel_student = st.selectbox(
                    "Select student",
                    options=list(range(1, n_students + 1)),
                    index=_tf_default - 1,
                    format_func=lambda x: f"Student {x}",
                    key="temp_cp_student",
                )
                sid = sel_student - 1

                if sid < len(rt_data):
                    student_rt = rt_data[sid]
                    rt_values = []
                    for v in student_rt.values():
                        try:
                            rt_values.append(float(v))
                        except (TypeError, ValueError):
                            rt_values.append(None)

                    valid_rt = [v for v in rt_values if v is not None]
                    if valid_rt:
                        items = list(range(1, len(rt_values) + 1))
                        fig_cp = go.Figure()
                        fig_cp.add_trace(go.Scatter(
                            x=items, y=rt_values,
                            mode="lines+markers",
                            name="Response Time",
                            line=dict(color="#0ff", width=2),
                            marker=dict(size=5, color="#0ff"),
                            hovertemplate="Item %{x}<br>RT: %{y:.2f}s<extra></extra>",
                        ))

                        # Change point overlay
                        cp_val = None
                        if cp_data.get("stat") and sid < len(cp_data["stat"]):
                            cp_rec = cp_data["stat"][sid]
                            for k, v in cp_rec.items():
                                if isinstance(v, (int, float)) and v > 0:
                                    cp_val = v
                                    break
                        if cp_val and 1 <= cp_val <= len(rt_values):
                            fig_cp.add_vline(
                                x=cp_val, line_dash="dash", line_color="#ff4444", line_width=2,
                                annotation_text=f"Change Point (item {int(cp_val)})",
                                annotation_font_color="#ff4444",
                            )

                        fig_cp.update_layout(
                            xaxis_title="Item Position",
                            yaxis_title="Response Time (s)",
                            title=f"Student {sel_student} — Response Time Across Test",
                            template="plotly_dark",
                            paper_bgcolor="#1a1a2e",
                            plot_bgcolor="#0f3460",
                            height=400,
                        )
                        st.plotly_chart(fig_cp, use_container_width=True)

                        # Flagged status
                        if cp_data.get("flagged") and sid in cp_data["flagged"]:
                            st.error(f"Student {sel_student} is **flagged** by change-point detection.")
                        elif cp_val:
                            st.caption(f"Change point detected at item {int(cp_val)} but student was not flagged at the significance threshold.")
                        else:
                            st.caption("No change point detected for this student.")
                    else:
                        st.caption("No valid RT values for this student.")
                else:
                    st.caption("Student index out of range for RT data.")

        # ── Tab 2: Erasure Heatmap (detect_tt — stub) ──
        with tab_erasure:
            st.markdown("#### Erasure Heatmap")
            tt_data = flags.get("tt_agent", {})
            st.markdown("""
            <div style="background:#2d2d00;border-left:4px solid #ffaa00;padding:20px;border-radius:5px;margin:10px 0;">
            <h4 style="color:#ffaa00;margin-top:0;">Erasure Data Required</h4>
            <p style="color:#d0d0d0;">
            Test Tampering detection (<code>detect_tt</code>) requires <strong>erasure data</strong> — a record of
            initial and final answer selections per student per item. This data is typically collected by
            computer-based testing platforms that log answer changes.</p>
            <p style="color:#d0d0d0;">
            <strong>What this module would show:</strong><br>
            A heatmap (students x items) colored by the Erasure Detection Index (<code>EDI_SD</code>).
            Items with statistically significant wrong-to-right answer changes are highlighted,
            indicating potential unauthorized answer correction or external assistance.</p>
            <p style="color:#a0a0a0;font-size:0.85em;">
            To enable this module, provide erasure data (initial responses, final responses, distractor matrices)
            when the <code>detect_tt</code> agent is fully implemented.</p>
            </div>
            """, unsafe_allow_html=True)
            if tt_data.get("info"):
                st.caption(tt_data["info"])

        # ── Tab 3: Compromised List (detect_pk) ──
        with tab_pk:
            st.markdown("#### Compromised Item Analysis")
            st.caption("Students flagged by preknowledge detection on known compromised items.")
            pk_data = flags.get("pk_agent", {})

            if pk_data.get("error"):
                st.warning(f"pk_agent error: {pk_data['error']}")
            elif pk_data.get("info"):
                st.info(pk_data["info"])
            elif pk_data.get("stat"):
                pk_stat = pk_data["stat"]
                pk_flagged = set(pk_data.get("flagged", []))

                # Build ranked table
                pk_rows = []
                for i, rec in enumerate(pk_stat):
                    pk_rows.append({
                        "Student_ID": i + 1,
                        "L_S": round(float(rec.get("L_S", 0) or 0), 4),
                        "S_S": round(float(rec.get("S_S", 0) or 0), 4),
                        "W_S": round(float(rec.get("W_S", 0) or 0), 4),
                        "Flagged": i in pk_flagged,
                    })
                pk_df = pd.DataFrame(pk_rows).sort_values("L_S", ascending=True)

                # Highlight flagged rows
                def _pk_highlight(row):
                    if row["Flagged"]:
                        return ["background-color: #3d0000; color: #ff6b6b"] * len(row)
                    return [""] * len(row)

                st.dataframe(
                    pk_df.style.apply(_pk_highlight, axis=1),
                    use_container_width=True, hide_index=True,
                )
                n_pk_flagged = sum(1 for r in pk_rows if r["Flagged"])
                st.caption(f"{n_pk_flagged} of {len(pk_rows)} students flagged for preknowledge.")

                with st.expander("Interpretation"):
                    st.markdown("""
- **L_S**: Likelihood-based statistic on compromised items. More negative values indicate unexpected correct responses on compromised items.
- **S_S**: Score-based statistic for preknowledge.
- **W_S**: Wald-type statistic for preknowledge.
- **Flagged**: Student met the significance threshold (alpha = 0.05) on at least one method.
                    """)
            else:
                st.info("No preknowledge data. Specify compromised item IDs in **Preparation** to enable this analysis.")

