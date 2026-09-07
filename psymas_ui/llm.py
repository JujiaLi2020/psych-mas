"""LLM provider configuration and text-generation helpers for PsyMAS."""

from __future__ import annotations

import os
import time

import requests
import streamlit as st
from dotenv import load_dotenv

from mmls import (
    DEFAULT_GEMINI_MODEL_IDS,
    GEMINI_MODEL_OPTIONS,
    LOCAL_OLLAMA_MODEL_IDS,
    LOCAL_OLLAMA_MODELS,
    OPENROUTER_FREE_MODEL_IDS,
    OPENROUTER_FREE_MODELS,
)


OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OLLAMA_CHAT_URL = os.getenv("OLLAMA_CHAT_URL", "http://localhost:11434/api/chat")


def call_openrouter(api_key: str, model_id: str, messages: list[dict], timeout: int = 90) -> tuple[str | None, str | None]:
    """Call OpenRouter chat completions. Return (text, None) or (None, error_message)."""
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


def call_ollama(model_id: str, messages: list[dict], timeout: int = 120) -> tuple[str | None, str | None]:
    """Call local Ollama chat API. Return (text, None) or (None, error_message)."""
    if not model_id or not messages:
        return None, "No local model or messages."
    try:
        body = {"model": model_id, "messages": messages, "stream": False}
        resp = requests.post(OLLAMA_CHAT_URL, json=body, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        msg = data.get("message", {}) if isinstance(data, dict) else {}
        text = msg.get("content") or data.get("response") if isinstance(data, dict) else ""
        return (str(text).strip() or "No response from local model.", None)
    except requests.exceptions.ConnectionError:
        return None, f"Cannot reach local Ollama at {OLLAMA_CHAT_URL}. Start Ollama and pull the selected model."
    except requests.exceptions.HTTPError as e:
        try:
            msg = e.response.text if e.response is not None else str(e)
        except Exception:
            msg = str(e)
        code = e.response.status_code if e.response is not None else 0
        return None, f"Ollama HTTP {code}: {msg}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def test_ollama_model(model_id: str, timeout: int = 30) -> tuple[bool, str | None, float]:
    """Test one local Ollama model with a minimal message."""
    t0 = time.perf_counter()
    text, err = call_ollama(model_id, [{"role": "user", "content": "Hi"}], timeout=timeout)
    elapsed = time.perf_counter() - t0
    return (bool(text and not err), err, elapsed)


def test_openrouter_model(api_key: str, model_id: str, timeout: int = 20) -> tuple[bool, str | None, float]:
    """Test one OpenRouter model with a minimal message."""
    t0 = time.perf_counter()
    text, err = call_openrouter(api_key, model_id, [{"role": "user", "content": "Hi"}], timeout=timeout)
    elapsed = time.perf_counter() - t0
    return (bool(text and not err), err, elapsed)


def test_openrouter_api_key(api_key: str, timeout: int = 15) -> tuple[bool, str]:
    """Verify OPENROUTER_API_KEY with a minimal request."""
    if not api_key or not api_key.strip():
        return False, "OPENROUTER_API_KEY not set in .env. Add it for higher limits (get key at openrouter.ai)."
    selected = st.session_state.get("selected_gemini_model")
    candidates = []
    if selected and isinstance(selected, str):
        candidates.append(selected)
    candidates.extend([m for m in OPENROUTER_FREE_MODEL_IDS if m not in candidates])

    last_err = None
    for mid in candidates[: min(10, len(candidates))]:
        ok, err, _ = test_openrouter_model(api_key.strip(), mid, timeout)
        if ok:
            return True, f"OpenRouter API key is valid (model ok: {mid})."
        last_err = err

    if last_err and "401" in str(last_err):
        return False, "Invalid or unauthorized OpenRouter key (401). Check OPENROUTER_API_KEY in .env."
    if last_err and "402" in str(last_err):
        return False, "OpenRouter: insufficient credits (402). Add credits at openrouter.ai/credits - free models need a non-negative balance."
    if last_err and "404" in str(last_err):
        return False, (
            "OpenRouter key may be valid, but the tested models returned 404 (no endpoints found). "
            "Pick a different OpenRouter model in Model engine, or refresh the OpenRouter model list."
        )
    if last_err:
        return False, f"OpenRouter: {last_err}"
    return False, "OpenRouter request failed (no response)."


def _openrouter_price_is_free(model: dict) -> bool:
    pricing = model.get("pricing") if isinstance(model, dict) else {}
    if not isinstance(pricing, dict):
        return False
    values = [pricing.get("prompt"), pricing.get("completion")]
    if not values or any(value is None for value in values):
        return False
    try:
        return all(float(str(value)) == 0 for value in values)
    except Exception:
        return False


def _openrouter_model_label(model: dict) -> str:
    model_id = str(model.get("id") or "").strip()
    name = str(model.get("name") or model_id).strip()
    suffix = " · free" if _openrouter_price_is_free(model) else ""
    return f"{name}{suffix}" if name else model_id


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_openrouter_model_options_cached(api_key_present: bool = False) -> tuple[list[tuple[str, str]], str | None]:
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip() if api_key_present else ""
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        resp = requests.get(OPENROUTER_MODELS_URL, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        rows = data.get("data") if isinstance(data, dict) else []
        if not isinstance(rows, list):
            return [], "OpenRouter model list response did not contain a data list."
        models = []
        seen = set()
        for row in rows:
            if not isinstance(row, dict):
                continue
            model_id = str(row.get("id") or "").strip()
            if not model_id or model_id in seen:
                continue
            seen.add(model_id)
            models.append(
                {
                    "id": model_id,
                    "label": _openrouter_model_label(row),
                    "free": _openrouter_price_is_free(row),
                }
            )
        models.sort(key=lambda item: (not item["free"], item["label"].lower(), item["id"].lower()))
        options = [(item["label"], item["id"]) for item in models]
        if not options:
            return [], "OpenRouter returned an empty model list."
        return options, None
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"


def load_openrouter_model_options(force: bool = False) -> tuple[list[tuple[str, str]], str | None, bool]:
    """Load the curated OpenRouter model list used by the prototype UI."""
    st.session_state.openrouter_model_options = OPENROUTER_FREE_MODELS
    st.session_state.openrouter_model_ids = OPENROUTER_FREE_MODEL_IDS
    st.session_state.openrouter_model_list_error = None
    return OPENROUTER_FREE_MODELS, None, True


def preferred_llm_provider() -> str:
    """Choose the configured LLM provider."""
    return "openrouter"


def llm_provider() -> str:
    """Current LLM provider. Defaults to OpenRouter but allows local Ollama."""
    provider = st.session_state.get("llm_provider") or preferred_llm_provider()
    if provider not in {"openrouter", "local_ollama"}:
        provider = "openrouter"
    return provider


def current_model_ids() -> list[str]:
    """Return model IDs for the current provider."""
    if llm_provider() == "openrouter":
        return st.session_state.get("openrouter_model_ids") or OPENROUTER_FREE_MODEL_IDS
    if llm_provider() == "local_ollama":
        return LOCAL_OLLAMA_MODEL_IDS
    return st.session_state.get("discovered_model_ids") or DEFAULT_GEMINI_MODEL_IDS


def current_model_options() -> list[tuple[str, str]]:
    """Return [(display_name, model_id), ...] for the current provider."""
    if llm_provider() == "openrouter":
        return st.session_state.get("openrouter_model_options") or OPENROUTER_FREE_MODELS
    if llm_provider() == "local_ollama":
        return LOCAL_OLLAMA_MODELS
    return st.session_state.get("discovered_model_options") or GEMINI_MODEL_OPTIONS


def effective_llm_model() -> str:
    """Return the pinned model when valid, otherwise the current selected model."""
    provider = llm_provider()
    pinned_provider = st.session_state.get("pinned_llm_provider")
    pinned_model = st.session_state.get("pinned_llm_model")
    if pinned_provider == provider and pinned_model:
        model_ids = current_model_ids()
        if pinned_model in model_ids:
            return pinned_model
    model_ids = current_model_ids()
    default_model = model_ids[0] if model_ids else (
        LOCAL_OLLAMA_MODEL_IDS[0] if provider == "local_ollama" else OPENROUTER_FREE_MODEL_IDS[0]
    )
    selected = st.session_state.get("selected_gemini_model", default_model)
    if selected not in model_ids:
        selected = default_model
    return selected


def model_settings_for_backend() -> dict:
    """Merge session model settings with LLM keys from env for backend report generation."""
    load_dotenv()
    base = dict(st.session_state.get("model_settings") or {})
    base["llm_provider"] = llm_provider()
    base["llm_model_id"] = effective_llm_model()
    base["openrouter_api_key"] = os.getenv("OPENROUTER_API_KEY", "")
    base["google_api_key"] = os.getenv("GOOGLE_API_KEY", "")
    base["ollama_chat_url"] = OLLAMA_CHAT_URL
    return base


def model_variants_with_selected_first() -> list[str]:
    """Return model list with the effective model first."""
    model_ids = current_model_ids()
    effective = effective_llm_model()
    if effective not in model_ids:
        provider = llm_provider()
        effective = model_ids[0] if model_ids else (
            LOCAL_OLLAMA_MODEL_IDS[0] if provider == "local_ollama" else OPENROUTER_FREE_MODEL_IDS[0]
        )
    return [effective] + [m for m in model_ids if m != effective]


def call_selected_llm_text(prompt: str, *, timeout: int = 90) -> tuple[str | None, str | None]:
    """Call the selected OpenRouter or local Ollama model for text-only prompts."""
    provider = llm_provider()
    if provider == "local_ollama":
        for model_id in model_variants_with_selected_first():
            text, err = call_ollama(model_id, [{"role": "user", "content": prompt}], timeout=timeout)
            if text and not err:
                return text, None
        return None, f"Local Ollama: no response. Start Ollama at {OLLAMA_CHAT_URL} and pull the selected model."
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    for model_id in model_variants_with_selected_first():
        text, err = call_openrouter(api_key, model_id, [{"role": "user", "content": prompt}], timeout=timeout)
        if text and not err:
            return text, None
    return None, "OpenRouter: no model returned a response. Try another model in Model engine or set OPENROUTER_API_KEY in .env."
