"""Shared state contract for PsyMAS LangGraph workflows."""

from typing import Annotated, NotRequired, TypedDict


def merge_flags(old: dict, new: dict) -> dict:
    """Merge specialist results written by parallel graph nodes."""
    merged = {**old} if old else {}
    if new:
        merged.update(new)
    return merged


class State(TypedDict):
    """Serializable state shared by psychometric and forensic workflows."""

    responses: list
    rt_data: list
    theta: float
    latency_flags: list[str]
    next_step: str
    model_settings: NotRequired[dict]
    is_verified: NotRequired[bool]
    rt_plot_path: NotRequired[str]
    icc_plot_path: NotRequired[str]
    icc_error: NotRequired[str]
    item_params: NotRequired[list[dict]]
    person_params: NotRequired[list[dict]]
    item_fit: NotRequired[list[dict]]
    model_fit: NotRequired[dict]
    aberrance_results: NotRequired[dict]
    aberrance_functions: NotRequired[list]
    compromised_items: NotRequired[list]
    answer_changes: NotRequired[list[dict]]
    threshold_config: NotRequired[dict]
    psi_data: NotRequired[list[dict]]
    flags: Annotated[dict, merge_flags]
    final_report: NotRequired[str]
    reporter_brief: NotRequired[str]
