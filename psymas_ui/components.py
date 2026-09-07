from __future__ import annotations

import html

import streamlit as st


def apply_control_theme() -> None:
    """Apply high-contrast control colors after page-local Streamlit styles."""
    st.markdown(
        """
<style>
div[data-testid="stSegmentedControl"] button,
div[data-testid="stSegmentedControl"] button > div,
div[data-testid="stSegmentedControl"] button span,
div[data-testid="stSegmentedControl"] button p,
div[data-testid="stSegmentedControl"] label,
div[data-testid="stSegmentedControl"] label > div,
div[data-testid="stSegmentedControl"] label *,
div[data-testid="stSegmentedControl"] [role="radio"],
div[data-testid="stSegmentedControl"] [role="radio"] * {
    background: #FFFFFF !important;
    color: #17212B !important;
    border-color: #B9C5D2 !important;
    opacity: 1 !important;
}
div[data-testid="stSegmentedControl"] button:hover,
div[data-testid="stSegmentedControl"] button:hover *,
div[data-testid="stSegmentedControl"] label:hover,
div[data-testid="stSegmentedControl"] label:hover * {
    background: #EAF3F5 !important;
    color: #102A35 !important;
}
div[data-testid="stSegmentedControl"] button[aria-pressed="true"],
div[data-testid="stSegmentedControl"] button[aria-pressed="true"] *,
div[data-testid="stSegmentedControl"] label:has(input:checked),
div[data-testid="stSegmentedControl"] label:has(input:checked) *,
div[data-testid="stSegmentedControl"] [role="radio"][aria-checked="true"],
div[data-testid="stSegmentedControl"] [role="radio"][aria-checked="true"] * {
    background: #174E5F !important;
    color: #FFFFFF !important;
    border-color: #174E5F !important;
    opacity: 1 !important;
}
button[data-testid="stBaseButton-segmented_control"],
button[data-testid="stBaseButton-segmented_control"] *,
button[kind="segmented_control"],
button[kind="segmented_control"] * {
    background: #FFFFFF !important;
    color: #17212B !important;
    border-color: #B9C5D2 !important;
    opacity: 1 !important;
}
button[data-testid="stBaseButton-segmented_controlActive"],
button[data-testid="stBaseButton-segmented_controlActive"] *,
button[kind="segmented_controlActive"],
button[kind="segmented_controlActive"] * {
    background: #174E5F !important;
    color: #FFFFFF !important;
    border-color: #174E5F !important;
    opacity: 1 !important;
}
button[role="tab"],
button[role="tab"] *,
div[data-baseweb="tab"],
div[data-baseweb="tab"] * {
    background: #FFFFFF !important;
    color: #17212B !important;
    border-color: #B9C5D2 !important;
    opacity: 1 !important;
}
button[role="tab"][aria-selected="true"],
button[role="tab"][aria-selected="true"] *,
div[data-baseweb="tab"][aria-selected="true"],
div[data-baseweb="tab"][aria-selected="true"] * {
    background: #174E5F !important;
    color: #FFFFFF !important;
    border-color: #174E5F !important;
}
div[data-testid="stPills"] {
    flex-wrap: wrap !important;
    gap: 0.28rem 0.32rem !important;
}
div[data-testid="stPills"] button,
div[data-testid="stPills"] button > div,
div[data-testid="stPills"] button span,
div[data-testid="stPills"] button p,
div[data-testid="stPills"] button * {
    background: #FFFFFF !important;
    background-color: #FFFFFF !important;
    color: #17212B !important;
    -webkit-text-fill-color: #17212B !important;
    border-color: #B9C5D2 !important;
    opacity: 1 !important;
}
div[data-testid="stPills"] button[aria-checked="true"],
div[data-testid="stPills"] button[aria-checked="true"] *,
div[data-testid="stPills"] button[aria-pressed="true"],
div[data-testid="stPills"] button[aria-pressed="true"] * {
    background: #174E5F !important;
    background-color: #174E5F !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    border-color: #174E5F !important;
    opacity: 1 !important;
}
div[data-testid="stSegmentedControl"] {
    flex-wrap: wrap !important;
}
</style>
        """,
        unsafe_allow_html=True,
    )


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
