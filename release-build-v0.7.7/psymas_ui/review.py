from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

import pandas as pd


def build_final_flag_review(
    *,
    flags: Mapping[str, object],
    n_examinees: int,
    selected_functions: Iterable[str],
    function_to_agent: Mapping[str, str],
    flagged_indices: Callable[[object, str], set[int]],
    decisions: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Build the lightweight review queue without domain synthesis."""
    if n_examinees <= 0:
        return pd.DataFrame()
    flagged_by: list[list[str]] = [[] for _ in range(n_examinees)]
    for function_name in selected_functions:
        agent_key = function_to_agent.get(function_name, "")
        if not agent_key or agent_key not in flags:
            continue
        for index in flagged_indices(flags.get(agent_key, {}), agent_key):
            if 0 <= index < n_examinees:
                flagged_by[index].append(function_name)

    decision_map = decisions if isinstance(decisions, Mapping) else {}
    rows = []
    for index, sources in enumerate(flagged_by, start=1):
        unique_sources = sorted(set(sources))
        flag_count = len(unique_sources)
        suggestion = (
            "High"
            if flag_count >= 3
            else "Moderate"
            if flag_count == 2
            else "Low"
            if flag_count == 1
            else "None"
        )
        decision = decision_map.get(str(index), {})
        decision = decision if isinstance(decision, Mapping) else {}
        rows.append(
            {
                "Examinee_ID": str(index),
                "System_Flag": 1 if flag_count else 0,
                "Flag_Count": flag_count,
                "Flag_Sources": ", ".join(unique_sources),
                "Review_Suggestion": suggestion,
                "Human_Decision": str(decision.get("final_decision", "")),
                "Reviewer_Note": str(decision.get("reviewer_note", "")),
            }
        )
    return pd.DataFrame(rows)
