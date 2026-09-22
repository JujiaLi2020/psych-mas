"""Deployment policies shared by the UI and LLM configuration layer."""

from __future__ import annotations

import os


def deployment_profile() -> str:
    return os.getenv("PSYMAS_DEPLOYMENT_PROFILE", "desktop").strip().lower() or "desktop"


def locked_llm_configuration() -> bool:
    """Return whether the deployment administrator owns the LLM configuration."""
    value = os.getenv("PSYMAS_LOCK_LLM", "").strip().lower()
    return deployment_profile() in {"railway", "public"} or value in {"1", "true", "yes", "on"}


def managed_llm_provider() -> str:
    """Return the provider allowed by a locked deployment."""
    provider = os.getenv("PSYMAS_LLM_PROVIDER", "openrouter").strip().lower()
    return provider if provider in {"openrouter", "local_ollama"} else "openrouter"


def managed_llm_model() -> str:
    return os.getenv("PSYMAS_OPENROUTER_MODEL_ID", "").strip()


def managed_llm_message() -> str:
    provider = managed_llm_provider()
    model = managed_llm_model() or "the configured model"
    if provider == "openrouter":
        return f"This hosted deployment uses the administrator-configured API and model ({model}). Provider, model, and API credentials are managed by the deployment administrator."
    return f"This deployment uses the administrator-configured local model ({model}). Provider, model, and endpoint are managed by the deployment administrator."
