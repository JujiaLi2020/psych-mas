"""Input-table normalization and validation for the Streamlit application."""

import re

import pandas as pd


def drop_index_column(df: pd.DataFrame) -> pd.DataFrame:
    """Remove a CSV row-number column when it is clearly an exported index."""
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


def align_rt_columns_to_response(rt_df: pd.DataFrame, resp_df: pd.DataFrame) -> pd.DataFrame:
    """Align response-time columns with response columns by position."""
    if rt_df.shape[1] != resp_df.shape[1]:
        return rt_df
    aligned = rt_df.copy()
    aligned.columns = list(resp_df.columns)[: aligned.shape[1]]
    return aligned


def coerce_numeric(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Convert a table to numeric values and report the offending columns."""
    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    non_numeric_cols = [
        str(col) for col in df.columns if (df[col].notna() & numeric_df[col].isna()).any()
    ]
    if non_numeric_cols:
        raise ValueError(f"{label} has non-numeric values in columns: {', '.join(non_numeric_cols)}.")
    return numeric_df


def validate_binary_responses(resp_df: pd.DataFrame) -> pd.DataFrame:
    """Validate a dichotomous 0/1 response matrix."""
    numeric_df = coerce_numeric(resp_df, "Response data")
    invalid_cols = []
    for col in numeric_df.columns:
        values = numeric_df[col].dropna().unique().tolist()
        if values and not set(values).issubset({0, 1}):
            invalid_cols.append(str(col))
    if invalid_cols:
        raise ValueError("Response data must be binary (0/1). Invalid columns: " + ", ".join(invalid_cols))
    return numeric_df


def parse_compromised_items_csv(comp_df: pd.DataFrame, *, n_items: int | None = None) -> list[int]:
    """Parse 1-based compromised item identifiers from supported CSV layouts."""
    if comp_df.empty:
        return []
    df = drop_index_column(comp_df).copy()
    if df.empty:
        return []

    normalized = {str(c).strip().lower().replace(" ", "_"): c for c in df.columns}
    item_col = None
    for name in (
        "item_id", "item", "item_number", "item_no", "question_id", "question",
        "compromised_item", "compromised_items", "id",
    ):
        if name in normalized:
            item_col = normalized[name]
            break
    if item_col is None:
        for col in df.columns:
            if pd.to_numeric(df[col], errors="coerce").notna().any():
                item_col = col
                break
    if item_col is None:
        raise ValueError("Compromised-items CSV must include at least one numeric item-id column.")

    mask = pd.Series([True] * len(df), index=df.index)
    for flag_name in ("compromised", "is_compromised", "flag", "leaked"):
        if flag_name in normalized:
            raw = df[normalized[flag_name]]
            if raw.dtype == bool:
                mask = raw.fillna(False)
            else:
                mask = raw.astype(str).str.strip().str.lower().isin(
                    {"1", "true", "t", "yes", "y", "exposed", "leaked", "compromised"}
                )
            break

    raw_vals = df.loc[mask, item_col]
    vals = pd.to_numeric(raw_vals, errors="coerce")
    if vals.notna().any():
        items = sorted({int(v) for v in vals.dropna().tolist() if int(v) > 0})
    else:
        found: list[int] = []
        for cell in raw_vals.astype(str).tolist():
            found.extend(int(x) for x in re.findall(r"\d+", cell))
        items = sorted({item for item in found if item > 0})
    if n_items and n_items > 0:
        items = [item for item in items if item <= n_items]
    return items


def validate_answer_changes_csv(changes_df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize long answer-change data for detect_tt."""
    if changes_df.empty:
        raise ValueError("Answer-change CSV is empty.")
    df = drop_index_column(changes_df).copy()
    lowered = {str(c).strip().lower(): c for c in df.columns}
    if any(name not in lowered for name in ("examinee_id", "item_id")):
        raise ValueError("Answer-change CSV must include columns: examinee_id, item_id.")

    has_changed = "changed" in lowered
    initial_col = lowered.get("initial_response") or lowered.get("initial_score")
    final_col = lowered.get("final_response") or lowered.get("final_score")
    has_initial_final = initial_col is not None and final_col is not None
    if not has_changed and not has_initial_final:
        raise ValueError(
            "Answer-change CSV must include changed, or initial/final response columns "
            "(initial_response/final_response or initial_score/final_score)."
        )
    if has_initial_final:
        initial = pd.to_numeric(df[initial_col], errors="coerce")
        final = pd.to_numeric(df[final_col], errors="coerce")
        if initial.isna().any() or final.isna().any():
            raise ValueError("Initial and final answer-change values must be numeric.")
        if not set(initial.astype(int).unique()).issubset({0, 1}) or not set(final.astype(int).unique()).issubset({0, 1}):
            raise ValueError("Initial and final answer-change values must use 0/1 scores.")
        df["initial_response"] = initial.astype(int)
        df["final_response"] = final.astype(int)
        if not has_changed:
            df["changed"] = (df["initial_response"] != df["final_response"]).astype(int)
            lowered["changed"] = "changed"
            has_changed = True
    if has_changed:
        changed = pd.to_numeric(df[lowered["changed"]], errors="coerce")
        if changed.notna().any():
            bad = sorted(set(changed.dropna().astype(int).tolist()) - {0, 1})
            if bad:
                raise ValueError("Column changed must use 0/1 values.")
            df["changed"] = changed.fillna(0).astype(int)
    return df
