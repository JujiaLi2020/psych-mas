"""Shared application constants for the PsyMAS Streamlit workbench."""

from __future__ import annotations

from pathlib import Path


APP_VERSION = "0.7.5"

PSYMAS_VIZ = {
    "ink": "#0F172A",
    "muted": "#64748B",
    "grid": "#E5E7EB",
    "reference": "#94A3B8",
    "reference_fill": "#D8E2EC",
    "selected": "#0F6B7C",
    "selected_light": "#2F9BB3",
    "correct": "#147A5A",
    "incorrect": "#B42318",
    "warning": "#C87512",
    "high": "#DC2626",
    "moderate": "#C87512",
    "context": "#6D5BA6",
    "inactive": "#6B7C90",
    "lane": "#D7DEE8",
    "range": "#F5E9C8",
}

ABERRANCE_FUNCTIONS = ["detect_rg", "detect_pm", "detect_ac", "detect_pk", "detect_as", "detect_nm", "detect_tt"]
ABERRANCE_FN_TO_AGENT = {
    "detect_nm": "nm_agent",
    "detect_pm": "pm_agent",
    "detect_ac": "ac_agent",
    "detect_as": "as_agent",
    "detect_rg": "rg_agent",
    "detect_cp": "cp_agent",
    "detect_tt": "tt_agent",
    "detect_pk": "pk_agent",
}

DEMO_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEMO_EVALUATED_SNAPSHOT_PATH = DEMO_DATA_DIR / "psymas_demo_evaluated_snapshot.zip"
DEMO_AGENT_PRESET = ["detect_nm", "detect_pm", "detect_ac", "detect_as", "detect_pk", "detect_rg", "detect_tt"]

DOMAIN_ORDER = ["MF", "RT", "PK", "TP", "SIM", "CP", "CTX"]
DOMAIN_LABELS = {
    "MF": "Misfit",
    "RT": "Response-Time",
    "PK": "Preknowledge",
    "TP": "Tampering",
    "SIM": "Similarity",
    "CP": "Change Point",
    "CTX": "Context",
}
