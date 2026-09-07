from __future__ import annotations

import io
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DATASET_A_FILES = (
    ("data/upload/final_scores_matrix.csv", "responses"),
    ("data/upload/response_times_matrix.csv", "response_times"),
    ("data/upload/initial_scores_matrix.csv", "initial_responses"),
    ("data/upload/answer_changes_long.csv", "answer_changes"),
    ("data/upload/compromised_items.csv", "exposure_labels"),
    ("data/upload/item_metadata.csv", "item_metadata"),
    ("data/upload/examinee_metadata.csv", "examinee_metadata"),
    ("data/response_long.csv", "long_response_table"),
    ("data/scenario_key.csv", "truth_labels"),
    ("data/scenario_summary.csv", "truth_summary"),
    ("data/copying_pairs_truth.csv", "pair_truth"),
    ("data/answer_change_summary.csv", "answer_change_truth_summary"),
    ("data/testing_context.csv", "testing_context"),
    ("data/group_check.csv", "group_check"),
)

DATASET_C_COLUMNS = (
    "reviewer_id",
    "case_id",
    "examinee_id",
    "report_version",
    "clarity_score",
    "traceability_score",
    "caution_score",
    "usefulness_score",
    "overclaiming_concern",
    "review_confidence",
    "mixed_evidence_reasonable",
    "open_comment",
    "timestamp",
)


def expert_review_template() -> pd.DataFrame:
    return pd.DataFrame(columns=DATASET_C_COLUMNS)


def expert_review_codebook() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"field": "reviewer_id", "description": "Anonymous expert reviewer identifier.", "coding": "Text"},
            {"field": "case_id", "description": "Stable reviewed case identifier.", "coding": "Text"},
            {"field": "examinee_id", "description": "PsyMAS examinee identifier shown in the case report.", "coding": "Text"},
            {"field": "report_version", "description": "Report or run version reviewed by the expert.", "coding": "Text"},
            {"field": "clarity_score", "description": "The report is clear and understandable.", "coding": "1=strongly disagree to 5=strongly agree"},
            {"field": "traceability_score", "description": "The report can be traced back to evidence and indices.", "coding": "1=strongly disagree to 5=strongly agree"},
            {"field": "caution_score", "description": "The report avoids equating flags with misconduct.", "coding": "1=strongly disagree to 5=strongly agree"},
            {"field": "usefulness_score", "description": "The report is useful for human review.", "coding": "1=strongly disagree to 5=strongly agree"},
            {"field": "overclaiming_concern", "description": "The report overstates what the evidence can support.", "coding": "1=no concern to 5=major concern"},
            {"field": "review_confidence", "description": "The reviewer feels confident making a next-step decision.", "coding": "1=low to 5=high"},
            {"field": "mixed_evidence_reasonable", "description": "Mixed or limited evidence is explained reasonably.", "coding": "1=strongly disagree to 5=strongly agree"},
            {"field": "open_comment", "description": "Optional qualitative comment.", "coding": "Text"},
            {"field": "timestamp", "description": "Date/time of expert review.", "coding": "ISO 8601 preferred"},
        ]
    )


def research_manifest(*, run_id: str = "", notes: str = "") -> dict[str, Any]:
    return {
        "package": "PsyMAS research export",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": str(run_id or ""),
        "schema_version": "research-export-1.0",
        "purpose": {
            "Dataset A": "Semi-simulated inputs and truth labels for statistical and workflow validation.",
            "Dataset B": "PsyMAS outputs, evidence traces, audit records, and review artifacts.",
            "Dataset C": "Expert review rating template for human-review readiness evaluation.",
        },
        "notes": notes,
    }


def research_readme() -> str:
    return """# PsyMAS Research Export

This package separates the research data into three folders.

## Dataset A: Semi-Simulated Forensic Dataset

Inputs and truth labels used for the worked example and proof-of-concept validation. These files are not used to create operational flags after detection; truth labels are joined only on the validation page.

## Dataset B: PsyMAS Output Logs

System outputs generated from the detection and review workflow. This folder contains forensic index tables, evidence input tables, domain evidence profiles, review queues, master results, and lightweight audit/traceability summaries when available.

## Dataset C: Expert Review Ratings

A blank expert-review template and codebook. This dataset is intended for psychometrics, testing, or assessment-security experts who review PsyMAS-generated evidence profiles and reports. Experts rate clarity, traceability, caution, usefulness, overclaiming concern, and review confidence; they do not recompute the forensic statistics.

## Recommended Reporting Use

- Dataset A supports detection coverage and false-positive analyses against simulated truth labels.
- Dataset B supports traceability, audit completeness, and workflow reproducibility analyses.
- Dataset C supports human-review readiness evaluation.
"""


def research_data_dictionary_markdown() -> str:
    return """# PsyMAS Research Export Data Dictionary

## Package Structure

### Dataset A: Semi-Simulated Forensic Dataset

These files define the controlled worked-example dataset. They include operational inputs such as response matrices and response-time matrices, plus simulated truth labels used only for validation. Truth labels should not be used to generate operational flags.

### Dataset B: PsyMAS Output Logs

These files are generated by PsyMAS after detection and review-support processing. They document the path from raw detector outputs to evidence inputs, domain profiles, review suggestions, master results, and audit/traceability records.

### Dataset C: Expert Review Ratings

These files support a separate human-review readiness study. Experts review PsyMAS reports and rate clarity, traceability, caution, usefulness, overclaiming concern, and confidence. Experts do not recompute forensic statistics.

## Key Files

| Folder | File | Purpose |
|---|---|---|
| root | `manifest.json` | Export timestamp, run id, schema version, and package purpose. |
| root | `README.md` | High-level description of Dataset A, Dataset B, and Dataset C. |
| root | `DATA_DICTIONARY.md` | Human-readable explanation of package contents. |
| root | `data_dictionary.csv` | Machine-readable file-level dictionary. |
| Dataset A | `final_scores_matrix.csv` | Dichotomous final response matrix used by score-based detectors. |
| Dataset A | `response_times_matrix.csv` | Response-time matrix used by RT and rapid-guessing detectors. |
| Dataset A | `initial_scores_matrix.csv` | Initial response matrix used with final responses for answer-change records. |
| Dataset A | `answer_changes_long.csv` | Long-format answer-change records for tampering/revision analysis. |
| Dataset A | `compromised_items.csv` | Exposure/compromised item labels for preknowledge analysis. |
| Dataset A | `item_metadata.csv` | Item-level simulation metadata and item parameters when available. |
| Dataset A | `examinee_metadata.csv` | Examinee-level simulation metadata when available. |
| Dataset A | `scenario_key.csv` | Examinee-level simulated truth labels for validation only. |
| Dataset A | `copying_pairs_truth.csv` | Pair-level simulated truth labels for copying validation only. |
| Dataset B | `master_results.csv` | Main examinee-level PsyMAS output: flags, domain strengths, review fields, indices. |
| Dataset B | `indices_export.csv` | Wide exported forensic index values when available. |
| Dataset B | `indices_table.csv` | Display-oriented forensic index table when available. |
| Dataset B | `b3_input.csv` | Evidence-input flags used to derive domain evidence profiles. |
| Dataset B | `domain_evidence.csv` | Domain-level evidence strength profiles. |
| Dataset B | `review_queue.csv` | Final detector flag summary and suggested review priority. |
| Dataset B | `claim_to_evidence_trace.csv` | Claim-level trace showing which evidence columns support each summary statement. |
| Dataset B | `audit_completeness_summary.csv` | Lightweight audit metrics for traceability and cautious-language checks. |
| Dataset B | `simulation_validation_summary.csv` | Scenario-level match rates between simulated truth units and index flags. |
| Dataset C | `expert_review_ratings_template.csv` | Blank template for expert ratings. |
| Dataset C | `expert_review_codebook.csv` | Field definitions and coding rules for the expert rating template. |

## Interpretation Rules

1. Dataset A is the validation input and truth-label source.
2. Dataset B is the system output and audit trail.
3. Dataset C is collected separately from human experts.
4. Simulated truth labels are joined only for validation analyses; they are not used by PsyMAS to create flags.
5. A system flag is a review trigger, not a misconduct conclusion.
"""


def research_data_dictionary() -> pd.DataFrame:
    rows = [
        ("root", "manifest.json", "Package metadata", "Export timestamp, run id, schema version, and package purpose."),
        ("root", "README.md", "Package guide", "High-level explanation of the three research datasets."),
        ("root", "DATA_DICTIONARY.md", "Documentation", "Human-readable data dictionary for the export package."),
        ("root", "data_dictionary.csv", "Documentation", "Machine-readable file-level data dictionary."),
        ("Dataset A", "final_scores_matrix.csv", "Operational input", "Dichotomous final response matrix."),
        ("Dataset A", "response_times_matrix.csv", "Operational input", "Response-time matrix for timing and rapid-guessing analyses."),
        ("Dataset A", "initial_scores_matrix.csv", "Operational input", "Initial response matrix for answer-change analysis."),
        ("Dataset A", "answer_changes_long.csv", "Operational input", "Long answer-change records."),
        ("Dataset A", "compromised_items.csv", "Operational input", "Exposure or compromised item labels."),
        ("Dataset A", "item_metadata.csv", "Simulation metadata", "Item parameters and item-level metadata when available."),
        ("Dataset A", "examinee_metadata.csv", "Simulation metadata", "Examinee-level metadata when available."),
        ("Dataset A", "response_long.csv", "Simulation source", "Long-form source table used to construct demo inputs."),
        ("Dataset A", "scenario_key.csv", "Truth labels", "Examinee-level simulated truth labels for validation only."),
        ("Dataset A", "scenario_summary.csv", "Truth labels", "Scenario-level truth-label summary."),
        ("Dataset A", "copying_pairs_truth.csv", "Truth labels", "Pair-level simulated copying truth labels."),
        ("Dataset A", "answer_change_summary.csv", "Truth labels", "Answer-change scenario summary."),
        ("Dataset A", "testing_context.csv", "Context", "Contextual simulation metadata."),
        ("Dataset A", "group_check.csv", "Context", "Group-level simulation check table."),
        ("Dataset B", "master_results.csv", "Primary output", "One row per examinee with flags, domain strengths, review fields, and indices."),
        ("Dataset B", "indices_export.csv", "Detector output", "Wide forensic index export when available."),
        ("Dataset B", "indices_table.csv", "Detector output", "Display-oriented index table when available."),
        ("Dataset B", "column_legend.csv", "Metadata", "Index column legend when available."),
        ("Dataset B", "b3_input.csv", "Evidence input", "Eligible evidence flags used for domain evidence profiles."),
        ("Dataset B", "domain_evidence.csv", "Evidence profile", "Domain-level evidence strengths."),
        ("Dataset B", "review_queue.csv", "Review support", "Final detector flag summary and suggested priority."),
        ("Dataset B", "final_flags.csv", "Review support", "Lightweight final detector flag table."),
        ("Dataset B", "claim_to_evidence_trace.csv", "Audit output", "Claim-to-evidence trace generated from master results."),
        ("Dataset B", "audit_completeness_summary.csv", "Audit output", "Audit and cautious-language summary metrics."),
        ("Dataset B", "simulation_validation_*.csv", "Validation output", "Validation summaries and details when simulated truth labels are available."),
        ("Dataset C", "expert_review_ratings_template.csv", "Expert review", "Blank human-review readiness rating template."),
        ("Dataset C", "expert_review_codebook.csv", "Expert review", "Codebook for expert rating fields."),
    ]
    return pd.DataFrame(rows, columns=["dataset", "file", "type", "description"])


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame()
    return df.to_csv(index=False).encode("utf-8-sig")


def build_claim_trace(master_df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(master_df, pd.DataFrame) or master_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    domain_cols = [c for c in master_df.columns if c.endswith("_Strength")]
    for _, row in master_df.iterrows():
        examinee_id = str(row.get("Examinee_ID", ""))
        system_flag = str(row.get("System_Flag", ""))
        flag_sources = str(row.get("Flag_Sources", ""))
        evidence_status = str(row.get("Evidence_Status", ""))
        priority = str(row.get("Review_Priority", row.get("Review_Suggestion", "")))
        rows.append(
            {
                "Examinee_ID": examinee_id,
                "claim_type": "system_review_trigger",
                "claim": f"System flag={system_flag}; priority={priority}",
                "evidence_source": "final detector flags and review-priority fields",
                "evidence_columns": "System_Flag, Flag_Count, Flag_Sources, Review_Priority",
                "evidence_value": flag_sources,
                "audit_note": "Flag is a human-review trigger, not a misconduct conclusion.",
            }
        )
        for col in domain_cols:
            domain = col.replace("_Strength", "")
            strength = str(row.get(col, ""))
            index_col = f"{domain}_Flagged_Indices"
            rows.append(
                {
                    "Examinee_ID": examinee_id,
                    "claim_type": "domain_strength",
                    "claim": f"{domain} evidence strength={strength}",
                    "evidence_source": "domain evidence profile",
                    "evidence_columns": f"{col}, {domain}_Rule, {index_col}",
                    "evidence_value": str(row.get(index_col, "")),
                    "audit_note": "Domain strength is derived from eligible evidence-input flags.",
                }
            )
        if evidence_status:
            rows.append(
                {
                    "Examinee_ID": examinee_id,
                    "claim_type": "case_profile",
                    "claim": evidence_status,
                    "evidence_source": "case synthesis rules",
                    "evidence_columns": "Evidence_Status, Rule_IDs_Triggered, Missing_Evidence",
                    "evidence_value": str(row.get("Rule_IDs_Triggered", "")),
                    "audit_note": "Case profile summarizes evidence for review; it is not a final human adjudication.",
                }
            )
    return pd.DataFrame(rows)


def build_audit_completeness(master_df: pd.DataFrame, claim_trace_df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(master_df, pd.DataFrame) or master_df.empty:
        return pd.DataFrame()
    n = len(master_df)
    text_cols = [c for c in ("Draft_Statement", "LLM_Review_Explanation", "Human_Reviewer_Note") if c in master_df.columns]
    restricted_re = r"\b(?:cheat|cheating|misconduct proven|guilty|fraud)\b"
    restricted_hits = 0
    if text_cols:
        text_blob = master_df[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
        restricted_hits = int(text_blob.str.contains(restricted_re, case=False, regex=True).sum())
    linked_claims = 0 if claim_trace_df.empty else int(claim_trace_df["evidence_value"].fillna("").astype(str).str.len().gt(0).sum())
    total_claims = 0 if claim_trace_df.empty else len(claim_trace_df)
    missing_disclosed = 0
    if "Missing_Evidence" in master_df.columns:
        missing_disclosed = int(master_df["Missing_Evidence"].fillna("").astype(str).str.len().gt(0).sum())
    return pd.DataFrame(
        [
            {"metric": "examinees_in_master_results", "value": n, "interpretation": "One row per examinee in the consolidated PsyMAS output."},
            {"metric": "claim_trace_rows", "value": total_claims, "interpretation": "Number of claim-to-evidence trace rows generated for Dataset B."},
            {"metric": "claims_with_nonempty_evidence_value", "value": linked_claims, "interpretation": "Trace rows with a nonempty evidence value."},
            {"metric": "cases_with_missing_evidence_disclosed", "value": missing_disclosed, "interpretation": "Cases where unavailable or missing evidence is visible in the master table."},
            {"metric": "restricted_language_hits", "value": restricted_hits, "interpretation": "Potential overclaiming terms found in report/review text fields; should be checked manually."},
        ]
    )


def _write_df(zf: zipfile.ZipFile, name: str, df: pd.DataFrame) -> None:
    zf.writestr(name, dataframe_to_csv_bytes(df))


def build_research_export_zip(
    *,
    root_dir: Path,
    dataset_b_tables: dict[str, pd.DataFrame],
    master_df: pd.DataFrame,
    run_id: str = "",
    extra_notes: str = "",
) -> bytes:
    root_dir = Path(root_dir)
    manifest = research_manifest(run_id=run_id, notes=extra_notes)
    claim_trace_df = build_claim_trace(master_df)
    audit_completeness_df = build_audit_completeness(master_df, claim_trace_df)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        zf.writestr("README.md", research_readme())
        zf.writestr("DATA_DICTIONARY.md", research_data_dictionary_markdown())
        _write_df(zf, "data_dictionary.csv", research_data_dictionary())

        dataset_a_manifest: list[dict[str, Any]] = []
        for rel_path, role in DATASET_A_FILES:
            path = root_dir / rel_path
            row = {
                "file": Path(rel_path).name,
                "source_path": rel_path.replace("\\", "/"),
                "role": role,
                "included": path.is_file(),
                "bytes": path.stat().st_size if path.is_file() else 0,
            }
            dataset_a_manifest.append(row)
            if path.is_file():
                zf.write(path, f"dataset_a_inputs/{Path(rel_path).name}")
        _write_df(zf, "dataset_a_inputs/dataset_a_manifest.csv", pd.DataFrame(dataset_a_manifest))

        for table_name, df in sorted(dataset_b_tables.items()):
            safe_name = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in table_name).strip("_")
            _write_df(zf, f"dataset_b_outputs/{safe_name}.csv", df if isinstance(df, pd.DataFrame) else pd.DataFrame())
        _write_df(zf, "dataset_b_outputs/claim_to_evidence_trace.csv", claim_trace_df)
        _write_df(zf, "dataset_b_outputs/audit_completeness_summary.csv", audit_completeness_df)

        _write_df(zf, "dataset_c_expert_review/expert_review_ratings_template.csv", expert_review_template())
        _write_df(zf, "dataset_c_expert_review/expert_review_codebook.csv", expert_review_codebook())
    return buf.getvalue()
