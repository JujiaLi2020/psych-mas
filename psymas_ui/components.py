from __future__ import annotations

import html

import streamlit as st


def apply_control_theme() -> None:
    """Use Streamlit's native control styling."""
    return
def workspace_view(label: str, options: list[str], *, key: str, default: str) -> str:
    apply_control_theme()
    if st.session_state.get(key) not in options:
        st.session_state[key] = default
    return (
        st.segmented_control(
            label,
            options=options,
            key=key,
            label_visibility="collapsed",
        )
        or default
    )


def workspace_header(title: str, description: str) -> None:
    st.markdown(
        f"""
<div class="psymas-page-heading">
  <h1>{html.escape(title)}</h1>
  <p>{html.escape(description)}</p>
</div>
        """,
        unsafe_allow_html=True,
    )


def kpi_progress(
    *,
    label: str,
    value: int | float,
    total: int | float,
    explanation: str,
    tone: str = "teal",
) -> None:
    total_value = float(total or 0)
    value_number = float(value or 0)
    percent = max(0.0, min(100.0, (value_number / total_value * 100.0) if total_value else 0.0))
    tone_class = tone if tone in {"teal", "amber", "red", "slate"} else "teal"
    st.markdown(
        f"""
<div class="psymas-summary-band {tone_class}">
  <div class="summary-number">{percent:.1f}%</div>
  <div class="summary-body">
    <div class="summary-label">{html.escape(label)}</div>
    <div class="summary-track"><span style="width:{percent:.2f}%"></span></div>
    <div class="summary-caption">{html.escape(explanation)}</div>
  </div>
  <div class="summary-ratio">{value_number:,.0f} / {total_value:,.0f}</div>
</div>
        """,
        unsafe_allow_html=True,
    )
