from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


DOMAIN_ORDER = ("MF", "RT", "SIM", "PK", "TP", "CP")


def _with_string_id(df: pd.DataFrame | None) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty or "Examinee_ID" not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out["Examinee_ID"] = out["Examinee_ID"].astype(str)
    return out


def _domain_summary(domain_df: pd.DataFrame | None) -> pd.DataFrame:
    source = _with_string_id(domain_df)
    if source.empty or "Domain" not in source.columns:
        return pd.DataFrame()

    source["Domain"] = source["Domain"].astype(str)
    rows: list[dict[str, object]] = []
    for examinee_id, examinee_rows in source.groupby("Examinee_ID", sort=False):
        record: dict[str, object] = {"Examinee_ID": examinee_id}
        for domain in DOMAIN_ORDER:
            matches = examinee_rows[examinee_rows["Domain"].eq(domain)]
            if matches.empty:
                record[f"{domain}_Strength"] = ""
                record[f"{domain}_Rule"] = ""
                record[f"{domain}_Flagged_Indices"] = ""
                continue
            row = matches.iloc[0]
            primary = str(row.get("Primary_Hits", "") or "").strip()
            supporting = str(row.get("Supporting_Hits", "") or "").strip()
            flags = str(row.get("Evidence_Flags", "") or "").strip()
            combined = ", ".join(dict.fromkeys(
                item.strip()
                for text in (flags, primary, supporting)
                for item in text.split(",")
                if item.strip()
            ))
            record[f"{domain}_Strength"] = row.get("Strength", "")
            record[f"{domain}_Rule"] = row.get("Strength_Rule", "")
            record[f"{domain}_Flagged_Indices"] = combined
        rows.append(record)
    return pd.DataFrame(rows)


def build_master_results(
    *,
    indices_df: pd.DataFrame,
    final_flags_df: pd.DataFrame | None = None,
    review_df: pd.DataFrame | None = None,
    domain_df: pd.DataFrame | None = None,
    run_id: str = "",
    schema_version: str = "1.0",
) -> pd.DataFrame:
    """Create one examinee-level table containing final review and all index outputs."""
    indices = _with_string_id(indices_df)
    if indices.empty:
        return pd.DataFrame()

    master = indices.copy()
    final_flags = _with_string_id(final_flags_df)
    review = _with_string_id(review_df)
    domains = _domain_summary(domain_df)

    for supplemental in (final_flags, review, domains):
        if supplemental.empty:
            continue
        duplicate_columns = [
            column
            for column in supplemental.columns
            if column != "Examinee_ID" and column in master.columns
        ]
        if duplicate_columns:
            master = master.drop(columns=duplicate_columns)
        master = master.merge(supplemental, on="Examinee_ID", how="left")

    for metadata_column in ("Run_ID", "Schema_Version"):
        if metadata_column in master.columns:
            master = master.drop(columns=[metadata_column])
    master.insert(1, "Run_ID", str(run_id or ""))
    master.insert(2, "Schema_Version", schema_version)

    preferred = [
        "Examinee_ID",
        "Run_ID",
        "Schema_Version",
        "System_Flag",
        "Flag_Count",
        "Flag_Sources",
        "Review_Suggestion",
        "Evidence_Status",
        "Review_Priority",
        "Review_Priority_Rule",
        "Review_Queue_Status",
        "Primary_Concern",
        "Rule_IDs_Triggered",
        "Missing_Evidence",
        "Draft_Statement",
        "Audit_Status",
        "Human_Final_Decision",
        "Human_Decision",
        "Human_Reviewer_Note",
        "Reviewer_Note",
        "LLM_Review_Explanation",
    ]
    for domain in DOMAIN_ORDER:
        preferred.extend(
            [
                f"{domain}_Strength",
                f"{domain}_Rule",
                f"{domain}_Flagged_Indices",
            ]
        )

    ordered = [column for column in preferred if column in master.columns]
    ordered.extend(column for column in master.columns if column not in ordered)
    master = master[ordered]

    duplicate_aliases = {
        "Human_Decision": "Human_Final_Decision",
        "Reviewer_Note": "Human_Reviewer_Note",
    }
    for alias, canonical in duplicate_aliases.items():
        if alias in master.columns and canonical in master.columns:
            canonical_values = master[canonical].fillna("").astype(str)
            alias_values = master[alias].fillna("").astype(str)
            master[canonical] = canonical_values.where(canonical_values.str.strip().ne(""), alias_values)
            master = master.drop(columns=[alias])

    object_columns: Iterable[str] = master.select_dtypes(include=["object"]).columns
    for column in object_columns:
        master[column] = master[column].fillna("")
    return master
