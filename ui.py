"""
Streamlit UI for loading response and response-time datasets, then running the
psych_workflow graph.

Data loading lives here—not in the Orchestrator. The Orchestrator only routes;
the UI reads the two files and passes them as initial state into the graph.
"""

from pathlib import Path
import io
import json
import math
import os
import re
import base64
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import time
import requests
from dotenv import load_dotenv

from graph import analyze_prompt, psych_workflow

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
    ok, err, _ = _test_openrouter_model(api_key.strip(), OPENROUTER_FREE_MODEL_IDS[0], timeout)
    if ok:
        return True, "OpenRouter API key is valid."
    if err and "401" in str(err):
        return False, "Invalid or unauthorized OpenRouter key (401). Check OPENROUTER_API_KEY in .env."
    if err and "402" in str(err):
        return False, "OpenRouter: insufficient credits (402). Add credits at openrouter.ai/credits — free models need a non-negative balance."
    if err:
        return False, f"OpenRouter: {err}"
    return False, "OpenRouter request failed."


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


@st.cache_data(show_spinner=False)
def _run_workflow_cached(payload_json: str) -> dict:
    payload = json.loads(payload_json)
    initial_state = {
        "responses": payload["responses"],
        "rt_data": payload["rt_data"],
        "theta": 0.0,
        "latency_flags": [],
        "next_step": "",
        "model_settings": payload["model_settings"],
        "is_verified": payload["is_verified"],
    }
    return psych_workflow.invoke(initial_state)


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


def _check_r_packages() -> tuple[bool, str | None]:
    """Check if R and required packages (mirt, WrightMap, psych) are available. Returns (ok, error_message)."""
    try:
        import rpy2.robjects as ro
    except Exception as exc:
        return False, f"R/rpy2 not available ({exc}). Install R and Python rpy2 for IRT analysis."
    try:
        for pkg in ("mirt", "WrightMap", "psych"):
            r_ok = ro.r(f'require("{pkg}", quietly=TRUE)')
            # R returns logical vector of length 1 (or None in some rpy2 envs); get scalar safely
            if r_ok is None:
                ok = False
            elif len(r_ok):
                ok = bool(r_ok[0])
            else:
                ok = False
            if not ok:
                return False, (
                    f"R package '{pkg}' not installed. "
                    "Run in terminal: Rscript install_r_packages.R  "
                    "Or in R: install.packages(c('mirt','WrightMap','psych'), repos='https://cloud.r-project.org')"
                )
        return True, None
    except Exception as exc:
        return False, f"Could not check R packages: {exc}"


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
        "You are a psychometrics expert assistant helping users interpret IRT analysis results. "
        "You have access to item parameters, person parameters, and item fit statistics. "
        "Provide clear, helpful explanations and insights based on the analysis context provided. "
        "If the question is about specific values or patterns, refer to the available data. "
        "Be conversational and educational."
    )
    
    user_prompt = (
        f"Analysis Context:\n{context_text}\n\n"
        f"User Question: {question}\n\n"
        "Please provide a helpful analysis or answer to the user's question."
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

        doc.build(story)
        return buffer.getvalue(), None
    except Exception as e:
        return b"", f"{type(e).__name__}: {e}"


def _render_results(final: dict) -> None:
    tab1, tab2 = st.tabs(["📊 Response Results", "⏱️ RT Analysis"])

    with tab1:
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
            st.caption("Ask questions about your psychometric analysis results.")
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

    with tab2:
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

st.set_page_config(page_title="Psych MAS — IRT & RT", layout="centered")
st.title("Psych MAS — IRT & RT")

st.markdown(
    "Upload **response** and **response-time** CSVs."
    #"workflow as initial state; the Orchestrator then dispatches to IRT, RT, and Analyze."
)

st.subheader("Input your psychometric analysis task")
if "model_settings" not in st.session_state:
    st.session_state.model_settings = _interpret_prompt("")
if "is_verified" not in st.session_state:
    st.session_state.is_verified = False
if "prompt_analyzed" not in st.session_state:
    st.session_state.prompt_analyzed = False
if "last_prompt" not in st.session_state:
    st.session_state.last_prompt = ""

col_input, col_confirm = st.columns(2)

with col_input:
    st.subheader("Psych-MAS Assistant")
    tab_prompt, tab_model_engine = st.tabs(["Prompt", "Model engine"])
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

if st.session_state.is_verified:
    col1, col2 = st.columns(2)

    with col1:
        resp_file = st.file_uploader(
            "Response data (CSV)",
            type=["csv"],
            help="Item responses (e.g. rows = persons, columns = items, 0/1).",
        )

    with col2:
        rt_file = st.file_uploader(
            "Response-time data (CSV, optional)",
            type=["csv"],
            help="Response times in the same row/column layout as responses. Optional.",
        )

    if resp_file is not None:
        with st.expander("Response data preview"):
            resp_file.seek(0)
            resp_df = pd.read_csv(resp_file)
            st.dataframe(resp_df.head(10), width="stretch")

    if rt_file is not None:
        with st.expander("Response-time data preview"):
            rt_file.seek(0)
            rt_df = pd.read_csv(rt_file)
            st.dataframe(rt_df.head(10), width="stretch")

    r_ok, r_msg = _check_r_packages()
    if not r_ok and r_msg:
        st.warning(r_msg)

    run = st.button("Run workflow")
else:
    resp_file = None
    rt_file = None
    run = False

if run:
    if resp_file is None:
        st.error("Please upload response data (CSV) to run the workflow.")
        st.stop()
    if not st.session_state.is_verified:
        st.error("Please confirm model settings before running the workflow.")
        st.stop()

    # Reset file position in case the preview already read the stream
    resp_file.seek(0)
    try:
        resp_df = pd.read_csv(resp_file)
    except pd.errors.EmptyDataError:
        st.error("Response file is empty or has no parseable columns. Re-upload a valid CSV.")
        st.stop()

    resp_df = _drop_index_column(resp_df)
    try:
        resp_df = _validate_binary_responses(resp_df)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    if rt_file is not None:
        rt_file.seek(0)
        try:
            rt_df = pd.read_csv(rt_file)
        except pd.errors.EmptyDataError:
            st.error("Response-time file is empty or has no parseable columns. Re-upload or omit.")
            st.stop()
        rt_df = _drop_index_column(rt_df)
        try:
            rt_df = _coerce_numeric(rt_df, "Response-time data")
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
        if resp_df.shape[1] != rt_df.shape[1]:
            st.error(
                "Response and response-time files must have the same number of columns "
                f"after cleaning. Responses: {resp_df.shape[1]}, RT: {rt_df.shape[1]}."
            )
            st.stop()
        rt_data = rt_df.to_dict(orient="records")
    else:
        rt_data = []

    # Match graph expectations: list of dicts (DataFrame-friendly in irt_agent)
    responses = resp_df.to_dict(orient="records")

    payload_json = json.dumps(
        {
        "responses": responses,
        "rt_data": rt_data,
            "model_settings": st.session_state.model_settings,
            "is_verified": st.session_state.is_verified,
        },
        sort_keys=True,
    )

    with st.spinner("Running workflow…"):
        try:
            final = _run_workflow_cached(payload_json)
            st.session_state.last_result = final
            st.session_state.last_payload = payload_json
            st.success("Workflow finished.")
            _render_results(final)
        except Exception as e:
            st.exception(e)

if "last_result" in st.session_state and not run:
    st.subheader("Last results")
    _render_results(st.session_state.last_result)

