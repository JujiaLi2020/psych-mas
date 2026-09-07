from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyBboxPatch


DOMAIN_COLUMNS = ["RT", "SIM", "PK", "TP", "MF", "CP"]
DOMAIN_FULL = {
    "RT": "Response-Time",
    "SIM": "Similarity",
    "PK": "Preknowledge",
    "TP": "Tampering",
    "MF": "Misfit",
    "CP": "Change-Pattern",
}
STRENGTH_SCORE = {"unavailable": -1, "none": 0, "weak": 1, "moderate": 2, "strong": 3}
FIG_FLOW_SIZE = (15.5, 4.6)
FIG_INTERFACE_SIZE = (15.5, 5.4)
FIG_TITLE_STYLE = {"fontsize": 13, "weight": "bold", "color": "#111827"}
FLOW_BOX = dict(boxstyle="round,pad=0.50", facecolor="#f8fafc", edgecolor="#64748b", linewidth=1.15)
FLOW_ARROW = dict(arrowstyle="->", color="#475569", lw=1.5)
DEFAULT_CASE_LABELS = {
    "Case A": "Possible item preknowledge",
    "Case B": "Rapid guessing",
    "Case C": "Suspicious answer changes",
}
MIXED_CASE_LABELS = {
    "Case A": "Possible item preknowledge",
    "Case B": "Rapid guessing",
    "Case C": "Mixed fast high-ability performance",
}


@dataclass
class WorkedExampleResult:
    output_dir: Path
    table3: pd.DataFrame
    table4: pd.DataFrame
    table5: pd.DataFrame
    captions_md: str
    selected_cases: dict[str, str]
    files: list[Path]
    zip_bytes: bytes


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _read_matrix(path: Path) -> pd.DataFrame:
    df = _read_csv(path)
    if df.empty:
        return df
    first = df.columns[0]
    if str(first).startswith("Unnamed") or first == "":
        df = df.rename(columns={first: "examinee_id"})
    if "examinee_id" not in df.columns:
        df = df.rename(columns={df.columns[0]: "examinee_id"})
    df["examinee_id"] = df["examinee_id"].astype(str)
    return df.set_index("examinee_id")


def _canonical_id(value: Any) -> str:
    text = str(value or "").strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    return str(int(digits)) if digits else text


def _examinee_label(canonical: str) -> str:
    try:
        return f"E{int(canonical):03d}"
    except Exception:
        return str(canonical)


def _matrix_row(matrix: pd.DataFrame, canonical_id: str) -> pd.Series:
    if matrix.empty:
        return pd.Series(dtype=float)
    keys = [_examinee_label(canonical_id), str(canonical_id)]
    for key in keys:
        if key in matrix.index:
            return pd.to_numeric(matrix.loc[key], errors="coerce")
    return pd.Series(dtype=float)


def _master_row(master: pd.DataFrame, canonical_id: str) -> pd.Series:
    if master.empty or "Examinee_ID" not in master.columns:
        return pd.Series(dtype=object)
    tmp = master.copy()
    tmp["_key"] = tmp["Examinee_ID"].map(_canonical_id)
    match = tmp[tmp["_key"].eq(str(canonical_id))]
    return match.iloc[0] if not match.empty else pd.Series(dtype=object)


def _scenario_table(root_dir: Path) -> pd.DataFrame:
    scenario = _read_csv(root_dir / "data" / "scenario_key.csv")
    if scenario.empty:
        return scenario
    scenario["examinee_id"] = scenario["examinee_id"].astype(str)
    scenario["_key"] = scenario["examinee_id"].map(_canonical_id)
    return scenario


def _case_metrics(root_dir: Path) -> pd.DataFrame:
    responses = _read_matrix(root_dir / "data" / "upload" / "final_scores_matrix.csv")
    rt = _read_matrix(root_dir / "data" / "upload" / "response_times_matrix.csv")
    item_meta = _read_csv(root_dir / "data" / "upload" / "item_metadata.csv")
    comp = _read_csv(root_dir / "data" / "upload" / "compromised_items.csv")
    changes = _read_csv(root_dir / "data" / "upload" / "answer_changes_long.csv")
    scenario = _scenario_table(root_dir)

    item_ids = [str(c) for c in responses.columns]
    exposed_items = set(comp.get("item_id", pd.Series(dtype=str)).astype(str).tolist())
    if not item_meta.empty and "item_id" in item_meta.columns:
        item_meta = item_meta.copy()
        item_meta["item_id"] = item_meta["item_id"].astype(str)
    exposed_b = (
        pd.to_numeric(comp.get("true_b", pd.Series(dtype=float)), errors="coerce")
        if not comp.empty
        else pd.Series(dtype=float)
    )
    difficult_cut = float(exposed_b.median()) if len(exposed_b.dropna()) else 0.0
    difficult_exposed = set(
        comp.loc[pd.to_numeric(comp.get("true_b", 0), errors="coerce").ge(difficult_cut), "item_id"].astype(str).tolist()
    ) if not comp.empty and "item_id" in comp.columns else set()

    change_counts = pd.Series(dtype=float)
    if not changes.empty and "examinee_id" in changes.columns and "changed" in changes.columns:
        changes = changes.copy()
        changes["_key"] = changes["examinee_id"].map(_canonical_id)
        change_counts = pd.to_numeric(changes["changed"], errors="coerce").fillna(0).groupby(changes["_key"]).sum()

    rows = []
    for raw_id in responses.index:
        canonical = _canonical_id(raw_id)
        r = _matrix_row(responses, canonical)
        t = _matrix_row(rt, canonical)
        score = float(r.sum(skipna=True)) if not r.empty else np.nan
        mean_rt = float(t.mean(skipna=True)) if not t.empty else np.nan
        very_fast = int(t.lt(10).sum()) if not t.empty else 0
        exposed_correct = int(r[[c for c in item_ids if c in exposed_items]].eq(1).sum()) if exposed_items else 0
        fast_difficult = 0
        for item in [c for c in item_ids if c in difficult_exposed]:
            if item in r.index and item in t.index and r[item] == 1 and pd.notna(t[item]) and t[item] < max(10, t.median(skipna=True)):
                fast_difficult += 1
        missing = []
        if r.isna().any():
            missing.append("response")
        if t.empty or t.isna().any():
            missing.append("response-time")
        srow = scenario[scenario["_key"].eq(canonical)].iloc[0] if not scenario.empty and scenario["_key"].eq(canonical).any() else {}
        rows.append(
            {
                "canonical_id": canonical,
                "examinee_id": _examinee_label(canonical),
                "true_group": srow.get("true_group", "") if isinstance(srow, pd.Series) else "",
                "scenario_type": srow.get("scenario_type", "") if isinstance(srow, pd.Series) else "",
                "score": score,
                "mean_rt": mean_rt,
                "very_fast_count": very_fast,
                "exposed_correct_count": exposed_correct,
                "exposed_difficult_fast_correct_count": fast_difficult,
                "answer_change_count": int(change_counts.get(canonical, 0)),
                "missing_data_status": ", ".join(missing) if missing else "None",
            }
        )
    return pd.DataFrame(rows)


def _choose_case(metrics: pd.DataFrame, truth_group: str, sort_cols: list[str], ascending: list[bool]) -> str:
    if metrics.empty:
        return ""
    subset = metrics[metrics["true_group"].astype(str).eq(truth_group)].copy()
    if subset.empty:
        subset = metrics.copy()
    subset = subset.sort_values(sort_cols, ascending=ascending)
    return str(subset.iloc[0]["canonical_id"])


def _strength_rank_series(values: pd.Series) -> pd.Series:
    return values.fillna("").astype(str).str.lower().map(STRENGTH_SCORE).fillna(-1)


def _choose_answer_change_case(metrics: pd.DataFrame) -> str:
    subset = metrics[metrics["true_group"].astype(str).eq("answer_change")].copy()
    if subset.empty:
        return _choose_case(metrics, "answer_change", ["answer_change_count"], [False])
    subset["_tp_rank"] = _strength_rank_series(subset.get("TP_Strength", pd.Series(index=subset.index, dtype=str)))
    subset["_rt_rank"] = _strength_rank_series(subset.get("RT_Strength", pd.Series(index=subset.index, dtype=str)))
    subset = subset.sort_values(
        ["_tp_rank", "_rt_rank", "answer_change_count", "score"],
        ascending=[False, True, False, False],
    )
    return str(subset.iloc[0]["canonical_id"])


def _select_cases(metrics: pd.DataFrame, *, profile: str = "domain_distinct") -> dict[str, str]:
    if profile == "mixed_high_ability":
        return {
            "Case A": _choose_case(
                metrics,
                "preknowledge",
                ["exposed_difficult_fast_correct_count", "exposed_correct_count", "score"],
                [False, False, False],
            ),
            "Case B": _choose_case(
                metrics,
                "rapid_guessing",
                ["very_fast_count", "score", "mean_rt"],
                [False, True, True],
            ),
            "Case C": _choose_case(
                metrics,
                "mixed_fast_high_ability",
                ["score", "mean_rt", "very_fast_count"],
                [False, True, False],
            ),
        }
    return {
        "Case A": _choose_case(
            metrics,
            "preknowledge",
            ["exposed_difficult_fast_correct_count", "exposed_correct_count", "score"],
            [False, False, False],
        ),
        "Case B": _choose_case(
            metrics,
            "rapid_guessing",
            ["very_fast_count", "score", "mean_rt"],
            [False, True, True],
        ),
        "Case C": _choose_answer_change_case(metrics),
    }


def _observed_pattern(case_name: str, row: pd.Series) -> str:
    if case_name == "Case A":
        return "Fast and correct responses on exposed difficult items"
    if case_name == "Case B":
        return "Very short response times with lower accuracy"
    if str(row.get("true_group", "")) == "answer_change" or int(row.get("answer_change_count", 0) or 0) >= 8:
        return "Concentrated answer changes requiring review"
    return "Fast responses with high score and limited supporting evidence"


def _available_sources(row: pd.Series) -> str:
    sources = ["responses"]
    if str(row.get("missing_data_status", "")).lower().find("response-time") < 0:
        sources.append("response times")
    if int(row.get("exposed_correct_count", 0) or 0) > 0:
        sources.append("exposure labels")
    if int(row.get("answer_change_count", 0) or 0) > 0:
        sources.append("answer changes")
    return ", ".join(sources)


def _build_table3(metrics: pd.DataFrame, cases: dict[str, str], case_labels: dict[str, str] | None = None) -> pd.DataFrame:
    scenario_labels = case_labels or DEFAULT_CASE_LABELS
    rows = []
    for case_name, canonical in cases.items():
        row = metrics[metrics["canonical_id"].eq(canonical)].iloc[0]
        rows.append(
            {
                "Case": case_name,
                "Examinee ID": row["examinee_id"],
                "Illustrative scenario": scenario_labels.get(case_name, row.get("true_group", "")),
                "Score": int(row["score"]) if pd.notna(row["score"]) else "",
                "Mean RT": f"{float(row['mean_rt']):.2f}" if pd.notna(row["mean_rt"]) else "Unavailable",
                "Key observed pattern": _observed_pattern(case_name, row),
                "Available evidence sources": _available_sources(row),
                "Missing evidence, if any": row["missing_data_status"],
            }
        )
    return pd.DataFrame(rows)


def _strength(row: pd.Series, domain: str) -> str:
    value = str(row.get(f"{domain}_Strength", "") or "").strip().lower()
    return value or "unavailable"


def _flagged_indices(row: pd.Series, domain: str) -> str:
    text = str(row.get(f"{domain}_Flagged_Indices", "") or "").strip()
    if text.lower() in {"nan", "none"}:
        text = ""
    if text:
        return text
    prefix_map = {
        "PK": ("pk_",),
        "RT": ("rg_", "rt_", "pm_"),
        "MF": ("nm_", "pm_"),
        "SIM": ("ac_", "as_"),
        "TP": ("tt_",),
        "CP": ("cp_",),
    }
    hits = []
    for col, val in row.items():
        if any(str(col).startswith(prefix) for prefix in prefix_map.get(domain, ())):
            if str(col).endswith("_flag") and pd.to_numeric(pd.Series([val]), errors="coerce").fillna(0).iloc[0] == 1:
                hits.append(str(col))
    return ", ".join(hits[:4])


def _build_table4(master: pd.DataFrame, case_a_id: str) -> pd.DataFrame:
    row = _master_row(master, case_a_id)
    evidence_use = {
        "PK": "Scenario Flag",
        "RT": "Scenario Flag",
        "MF": "Supporting Flag",
        "SIM": "Calibration Required",
        "TP": "Scenario Flag",
        "CP": "Display Only",
    }
    source = {
        "PK": "Preknowledge index",
        "RT": "Response-time index",
        "MF": "Person-fit or misfit index",
        "SIM": "Similarity or copying index",
        "TP": "Tampering or answer-change index",
        "CP": "Change-pattern index",
    }
    rows = []
    for domain in ["PK", "RT", "MF", "SIM", "TP", "CP"]:
        strength = _strength(row, domain)
        flagged = _flagged_indices(row, domain)
        flag = "Unavailable" if strength == "unavailable" else ("Yes" if STRENGTH_SCORE.get(strength, 0) > 0 else "No")
        rows.append(
            {
                "Evidence source": source[domain],
                "Function or method": flagged or "No counted indicator",
                "Domain": f"{domain} ({DOMAIN_FULL[domain]})",
                "Statistic or value": flagged or strength,
                "Flag": flag,
                "Evidence use": evidence_use[domain],
                "Domain strength contribution": strength.title(),
                "Interpretation for review": _interpretation(domain, strength),
            }
        )
    return pd.DataFrame(rows)


def _interpretation(domain: str, strength: str) -> str:
    if strength == "unavailable":
        return "Evidence source unavailable for this case."
    if domain == "SIM" and strength in {"none", "weak"}:
        return "Similarity outputs remain visible but do not contribute because calibrated pairwise rules were not active for this case."
    if domain == "CP" and strength in {"none", "weak"}:
        return "Change-pattern outputs are retained for localization or audit and are not counted as independent evidence here."
    if strength in {"moderate", "strong"}:
        if domain == "PK":
            return "Pattern is consistent with possible preknowledge and requires review."
        if domain == "RT":
            return "Timing pattern contributes to review priority."
        return "Domain contributes supporting evidence for review."
    if strength == "weak":
        return "Weak signal retained as supporting context."
    return "No counted concern in this domain."


def _build_table5() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Draft claim": "The examinee had item preknowledge.",
                "Audit issue": "Overclaiming and intent language",
                "Rule or control": "Restricted misconduct language",
                "Revised claim": "The response pattern is consistent with possible item preknowledge.",
                "Human-review implication": "Requires human review and contextual evidence.",
            },
            {
                "Draft claim": "No timing concern was found.",
                "Audit issue": "Response-time data unavailable",
                "Rule or control": "Missing evidence must not be treated as normal evidence",
                "Revised claim": "Response-time evidence was unavailable and could not be evaluated.",
                "Human-review implication": "Interpretation is limited.",
            },
            {
                "Draft claim": "The examinee should be sanctioned.",
                "Audit issue": "Premature final decision",
                "Rule or control": "Human analysts retain final judgment",
                "Revised claim": "The case is prioritized for human review.",
                "Human-review implication": "No operational action is determined by PsyMAS.",
            },
        ]
    )


def _save_table(df: pd.DataFrame, csv_path: Path) -> list[Path]:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    files = [csv_path]
    xlsx_path = csv_path.with_suffix(".xlsx")
    try:
        with pd.ExcelWriter(xlsx_path) as writer:
            df.to_excel(writer, index=False, sheet_name="table")
        files.append(xlsx_path)
    except Exception:
        pass
    return files


def _draw_lineage(table4: pd.DataFrame, case_a_label: str, output_dir: Path) -> list[Path]:
    """Draw Figure 2 from real PsyMAS UI screenshots.

    The figure should represent the software, not a hand-drawn conceptual mockup.
    Expected screenshot files are stored under ``output_dir / "screenshots"``:
    panel_a_data_inputs.png, panel_b_evidence_profile.png, and
    panel_c_traceable_report_review.png. If any are missing, generate an explicit
    placeholder that tells the author which real screenshots still need to be
    captured.
    """
    screenshot_dir = output_dir / "screenshots"
    expected = [
        ("Panel A", "Data and module status", screenshot_dir / "panel_a_data_inputs.png"),
        ("Panel B", "Evidence profile", screenshot_dir / "panel_b_evidence_profile.png"),
        ("Panel C", "Traceable report review", screenshot_dir / "panel_c_traceable_report_review.png"),
    ]
    existing = [(label, title, path) for label, title, path in expected if path.exists()]
    if len(existing) == len(expected):
        return _draw_interface_screenshots(existing, output_dir)
    return _draw_missing_screenshot_notice(expected, existing, output_dir)


def _draw_interface_screenshots(panels: list[tuple[str, str, Path]], output_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=FIG_INTERFACE_SIZE, gridspec_kw={"wspace": 0.08})
    fig.subplots_adjust(left=0.025, right=0.985, top=0.86, bottom=0.10)
    fig.suptitle(
        "Figure 2. PsyMAS Analyst Interface and Evidence-Review Functions",
        x=0.025,
        y=0.965,
        ha="left",
        fontsize=14,
        weight="bold",
        color="#111827",
    )
    for ax, (label, title, path) in zip(axes, panels):
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.add_patch(
            FancyBboxPatch(
                (0.00, 0.00),
                1.00,
                0.91,
                boxstyle="round,pad=0.010,rounding_size=0.018",
                facecolor="#ffffff",
                edgecolor="#cbd5e1",
                linewidth=1.0,
            )
        )
        ax.text(0.025, 0.970, label, fontsize=8.6, weight="bold", color="#0f766e", ha="left", va="top")
        ax.text(0.025, 0.925, title, fontsize=10.2, weight="bold", color="#111827", ha="left", va="top")
        image = plt.imread(path)
        # Crop the Streamlit chrome lightly: keep the real sidebar and content,
        # but remove browser/figure whitespace if present.
        h, w = image.shape[:2]
        crop = image[int(h * 0.02): int(h * 0.98), int(w * 0.00): int(w * 1.00)]
        zoom = min(0.22, 0.70 / max(crop.shape[0] / 900, 1))
        ab = AnnotationBbox(OffsetImage(crop, zoom=zoom), (0.50, 0.365), frameon=False, box_alignment=(0.5, 0.5))
        ax.add_artist(ab)
    fig.text(
        0.025,
        0.040,
        "Note. Figure panels are real PsyMAS interface screenshots captured from the running Streamlit application.",
        ha="left",
        va="bottom",
        fontsize=8.0,
        color="#475569",
    )
    return _save_figure(fig, output_dir / "figure2_single_case_evidence_lineage.png")


def _draw_missing_screenshot_notice(
    expected: list[tuple[str, str, Path]],
    existing: list[tuple[str, str, Path]],
    output_dir: Path,
) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=FIG_INTERFACE_SIZE, gridspec_kw={"wspace": 0.10})
    fig.subplots_adjust(left=0.025, right=0.985, top=0.86, bottom=0.12)
    fig.suptitle(
        "Figure 2. PsyMAS Analyst Interface and Evidence-Review Functions",
        x=0.025,
        y=0.965,
        ha="left",
        fontsize=14,
        weight="bold",
        color="#111827",
    )
    for ax in axes:
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    existing_paths = {path for _, _, path in existing}
    for ax, (label, title, path) in zip(axes, expected):
        ax.add_patch(
            FancyBboxPatch(
                (0.00, 0.00),
                1.00,
                0.94,
                boxstyle="round,pad=0.012,rounding_size=0.025",
                facecolor="#ffffff",
                edgecolor="#cbd5e1",
                linewidth=1.0,
            )
        )
        ax.text(0.04, 0.885, label, fontsize=8.5, weight="bold", color="#0f766e", ha="left", va="center")
        ax.text(0.04, 0.835, title, fontsize=11.0, weight="bold", color="#111827", ha="left", va="center")
        if path in existing_paths:
            image = plt.imread(path)
            h, w = image.shape[:2]
            crop = image[int(h * 0.02): int(h * 0.98), int(w * 0.00): int(w * 1.00)]
            zoom = min(0.22, 0.70 / max(crop.shape[0] / 900, 1))
            ab = AnnotationBbox(OffsetImage(crop, zoom=zoom), (0.50, 0.365), frameon=False, box_alignment=(0.5, 0.5))
            ax.add_artist(ab)
        else:
            ax.add_patch(FancyBboxPatch((0.055, 0.30), 0.890, 0.26, boxstyle="round,pad=0.014,rounding_size=0.018", facecolor="#fff7ed", edgecolor="#fdba74", linewidth=1.0))
            ax.text(0.075, 0.485, "Real screenshot needed", fontsize=10.0, weight="bold", color="#9a3412", ha="left", va="center")
            ax.text(0.075, 0.425, fill(f"Save a real PsyMAS screenshot here: {path}", width=44), fontsize=7.8, color="#7c2d12", ha="left", va="center")
            ax.text(0.075, 0.350, "This placeholder prevents conceptual mockups from being used as software screenshots.", fontsize=7.2, color="#7c2d12", ha="left", va="center")
    fig.text(
        0.025,
        0.045,
        "Note. Figure 2 is generated only from real PsyMAS screenshots. Missing panels are shown explicitly.",
        ha="left",
        va="bottom",
        fontsize=8.0,
        color="#475569",
    )
    return _save_figure(fig, output_dir / "figure2_single_case_evidence_lineage.png")


def _draw_audit_example(output_dir: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=FIG_FLOW_SIZE)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    items = [
        (0.16, "Draft Reporter Output", '"The examinee had item\npreknowledge."', "#fff7ed", 9.4),
        (0.50, "Audit Issue", "Restricted language\nUnsupported intent claim\nUnsupported evidence-to-claim\nboundary", "#f8fafc", 9.0),
        (0.84, "Audited Human-Review Language", '"The response pattern is\nconsistent with possible item\npreknowledge and requires\nadditional human review."', "#ecfdf5", 9.0),
    ]
    for x, title, body, color, fsize in items:
        ax.text(
            x,
            0.56,
            f"{title}\n\n{body}",
            ha="center",
            va="center",
            fontsize=fsize,
            color="#111827",
            bbox={**FLOW_BOX, "facecolor": color},
        )
    ax.annotate("", xy=(0.36, 0.56), xytext=(0.27, 0.56), arrowprops=FLOW_ARROW)
    ax.annotate("", xy=(0.72, 0.56), xytext=(0.63, 0.56), arrowprops=FLOW_ARROW)
    ax.set_title("Figure 3. Draft-to-Audited Report Example", loc="left", **FIG_TITLE_STYLE)
    ax.text(0.5, 0.12, "PsyMAS reports statistical evidence for review. It does not determine misconduct.", ha="center", fontsize=9, color="#475569")
    return _save_figure(fig, output_dir / "figure3_report_audit_example.png")


def _draw_grid(master: pd.DataFrame, cases: dict[str, str], output_dir: Path, case_labels: dict[str, str] | None = None) -> list[Path]:
    labels = {}
    text_rows = []
    for case_name, canonical in cases.items():
        row = _master_row(master, canonical)
        label = (case_labels or {}).get(case_name, "")
        labels[case_name] = f"{case_name}: {label or _examinee_label(canonical)}"
        strengths = [_strength(row, d) for d in DOMAIN_COLUMNS]
        strengths = [_strength(row, d) for d in DOMAIN_COLUMNS]
        profile = _scenario_aware_profile(case_name, case_labels or {}, row, strengths)
        priority = _display_priority(str(row.get("Review_Priority", row.get("Review_Suggestion", "Review")) or "Review"))
        text_rows.append(strengths + [profile, priority])

    fig, ax = plt.subplots(figsize=(15.5, 4.3))
    ax.axis("off")
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 4.2)
    headers = [f"{d}\n{DOMAIN_FULL[d]}" for d in DOMAIN_COLUMNS] + ["Case evidence\nprofile", "Review\npriority"]
    widths = [1.05] * 6 + [2.25, 1.55]
    x0 = 2.5
    xs = [x0]
    for width in widths[:-1]:
        xs.append(xs[-1] + width)
    y_header = 3.35
    row_h = 0.82
    colors = {
        "unavailable": ("#eef2f7", "#111827"),
        "none": ("#fffde7", "#111827"),
        "weak": ("#dbeafe", "#111827"),
        "moderate": ("#8ecae6", "#111827"),
        "strong": ("#1d4e89", "white"),
    }
    ax.text(0, 4.05, "Figure 4. Three-Case Evidence Profile Grid", fontsize=13, weight="bold", color="#111827", ha="left", va="top")
    for j, header in enumerate(headers):
        ax.text(xs[j] + widths[j] / 2, y_header, header, ha="center", va="center", fontsize=8.3, weight="bold", color="#111827")
    for i, (case_name, vals) in enumerate(zip(cases.keys(), text_rows)):
        y = 2.72 - i * row_h
        ax.text(x0 - 0.12, y, labels[case_name], ha="right", va="center", fontsize=8.7, color="#111827")
        for j, value in enumerate(vals):
            if j < len(DOMAIN_COLUMNS):
                key = str(value).lower()
                face, text_color = colors.get(key, colors["none"])
                label = str(value).title()
                fsize = 8.4
            else:
                face, text_color = "#f8fafc", "#111827"
                label = fill(str(value), width=24 if j == 6 else 15)
                fsize = 7.5
            rect = plt.Rectangle((xs[j], y - row_h / 2), widths[j], row_h * 0.88, facecolor=face, edgecolor="#cbd5e1", linewidth=0.8)
            ax.add_patch(rect)
            ax.text(xs[j] + widths[j] / 2, y, label, ha="center", va="center", fontsize=fsize, color=text_color)
    return _save_figure(fig, output_dir / "figure4_three_case_evidence_grid.png")


def _scenario_aware_profile(case_name: str, case_labels: dict[str, str], row: pd.Series, strengths: list[str]) -> str:
    label = str(case_labels.get(case_name, "")).lower()
    strength_map = dict(zip(DOMAIN_COLUMNS, strengths))
    if "preknowledge" in label:
        if strength_map.get("PK") in {"moderate", "strong"} and strength_map.get("RT") in {"moderate", "strong"}:
            return "Cross-scenario pattern: PK + RT"
        if strength_map.get("PK") in {"moderate", "strong"}:
            return "Scenario-supported preknowledge pattern"
        return "Preknowledge scenario; limited counted support"
    if "rapid" in label:
        if strength_map.get("RT") in {"moderate", "strong"}:
            return "Single-scenario rapid-response signal"
        return "Rapid-response scenario; limited counted support"
    if "answer" in label or "tamper" in label:
        tp_indicator_count = _flagged_indicator_count(row, "TP")
        if strength_map.get("TP") == "strong" and tp_indicator_count >= 2:
            return "Scenario-supported answer-change pattern"
        if strength_map.get("TP") in {"moderate", "strong", "weak"}:
            return "Single-scenario answer-change signal"
        return "Answer-change scenario; limited counted support"
    if "mixed" in label or "high-ability" in label:
        return "Mixed fast high-ability pattern; interpret with caution"
    status = str(row.get("Evidence_Status", "") or "").strip()
    if status:
        return "Scenario-aware review profile"
    return "Review profile unavailable"


def _flagged_indicator_count(row: pd.Series, domain: str) -> int:
    text = str(row.get(f"{domain}_Flagged_Indices", "") or "").strip()
    if text and text.lower() not in {"nan", "none"}:
        return len([part for part in text.split(",") if part.strip()])
    prefixes = {
        "RT": ("rg_",),
        "TP": ("tt_",),
        "PK": ("pk_",),
        "SIM": ("ac_", "as_"),
        "MF": ("nm_", "pm_"),
        "CP": ("cp_",),
    }.get(domain, ())
    count = 0
    for col, val in row.items():
        if any(str(col).startswith(prefix) for prefix in prefixes) and str(col).endswith("_flag"):
            if pd.to_numeric(pd.Series([val]), errors="coerce").fillna(0).iloc[0] == 1:
                count += 1
    return count


def _display_priority(value: str) -> str:
    text = str(value or "").strip()
    if text.lower() == "high (context-heavy)":
        return "High, contextual review needed"
    if text.lower() == "critical / expedited":
        return "Critical, expedited review"
    return text


def _save_figure(fig: plt.Figure, png_path: Path) -> list[Path]:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = png_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [png_path, pdf_path]


def _captions(table3: pd.DataFrame, table4: pd.DataFrame, table5: pd.DataFrame, cases: dict[str, str]) -> str:
    case_a = _examinee_label(cases.get("Case A", ""))
    return f"""# Worked Example Caption Text

## Table 3 Summary

Table 3 summarizes three representative semi-simulated cases used to illustrate the PsyMAS evidence workflow. The default worked example contrasts possible item preknowledge, rapid guessing, and suspicious answer-change evidence. Scenario labels are used only for worked-example selection and validation, not as misconduct conclusions.

## Table 4 Summary

Table 4 reports deterministic evidence outputs for Case A ({case_a}). The table separates the observed statistical signals from their evidence use, domain contribution, and review interpretation. Flags are presented as review triggers. Calibration-required and display-only outputs, such as inactive similarity or change-pattern outputs, remain visible for transparency but do not contribute to evidence strength unless the relevant calibrated rule is active.

## Figure 2 Summary

Figure 2 shows the PsyMAS analyst interface as three coordinated evidence-review functions. Panel A summarizes available data, missing data, and deterministic-module eligibility. Panel B summarizes domain-level flags, statistics, threshold status, evidence strength, and missing evidence. Panel C shows traceable report review, including drafted language, evidence links, audit warnings, and human-review controls.

In the default worked example, Case A is labeled as a cross-scenario PK/RT pattern because both preknowledge evidence and response-time evidence are active. This avoids treating a second active primary domain as merely incidental unless the rulebook explicitly defines it as supporting timing evidence for the preknowledge scenario.

## Figure 3 Summary

Figure 3 illustrates the audit layer by contrasting an overclaiming draft statement with cautious human-review language. The revised statement preserves the evidentiary signal while removing intent language and final-decision claims.

## Figure 4 Summary

Figure 4 compares the three representative cases across six evidence domains. The grid uses neutral shading to show unavailable, none, weak, moderate, and strong evidence levels while preserving the distinction between evidence strength and review priority.

The case-profile column follows the scenario-aware case-profile rule: a case with multiple active primary scenario domains is shown as cross-scenario. A case with one active primary domain is shown as a single-scenario signal unless multiple eligible indicators support the same scenario strongly enough to label it scenario-supported.

## Table 5 Summary

Table 5 provides examples of report-audit controls applied to worked-example statements. The revisions show how PsyMAS keeps evidence, interpretation, and human adjudication separate.
"""


def _zip_files(files: list[Path]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            if path.exists():
                zf.write(path, path.name)
    return buf.getvalue()


def _metrics_with_master(root_dir: Path | str = ".") -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(root_dir)
    metrics = _case_metrics(root)
    master = _read_csv(root / "data" / "output" / "psymas_master_results.csv")
    if master.empty:
        master = _read_csv(root / "data" / "psymas_research_export" / "dataset_b_outputs" / "master_results.csv")
    if not master.empty and "Examinee_ID" in master.columns and not metrics.empty:
        master = master.copy()
        master["_key"] = master["Examinee_ID"].map(_canonical_id)
        keep = ["_key", "Evidence_Status", "Review_Priority"] + [f"{d}_Strength" for d in DOMAIN_COLUMNS]
        metrics = metrics.merge(master[[c for c in keep if c in master.columns]], left_on="canonical_id", right_on="_key", how="left")
        metrics = metrics.drop(columns=[c for c in ["_key"] if c in metrics.columns])
    return metrics, master


def load_worked_example_candidates(root_dir: Path | str = ".") -> pd.DataFrame:
    metrics, _ = _metrics_with_master(root_dir)
    if metrics.empty:
        return metrics
    cols = [
        "canonical_id",
        "examinee_id",
        "true_group",
        "score",
        "mean_rt",
        "very_fast_count",
        "exposed_correct_count",
        "exposed_difficult_fast_correct_count",
        "answer_change_count",
        "RT_Strength",
        "SIM_Strength",
        "PK_Strength",
        "TP_Strength",
        "MF_Strength",
        "CP_Strength",
        "Evidence_Status",
        "Review_Priority",
    ]
    return metrics[[c for c in cols if c in metrics.columns]].copy()


def recommended_worked_example_cases(root_dir: Path | str = ".", *, profile: str = "domain_distinct") -> tuple[dict[str, str], dict[str, str]]:
    metrics, _ = _metrics_with_master(root_dir)
    cases = _select_cases(metrics, profile=profile)
    labels = MIXED_CASE_LABELS if profile == "mixed_high_ability" else DEFAULT_CASE_LABELS
    return cases, labels


def generate_worked_example_package(
    root_dir: Path | str = ".",
    output_dir: Path | str = "outputs",
    *,
    cases: dict[str, str] | None = None,
    case_labels: dict[str, str] | None = None,
) -> WorkedExampleResult:
    root = Path(root_dir)
    out = root / output_dir
    out.mkdir(parents=True, exist_ok=True)
    metrics, master = _metrics_with_master(root)
    cases = cases or _select_cases(metrics)
    case_labels = case_labels or DEFAULT_CASE_LABELS
    table3 = _build_table3(metrics, cases, case_labels)
    table4 = _build_table4(master, cases["Case A"])
    table5 = _build_table5()
    captions = _captions(table3, table4, table5, cases)

    files: list[Path] = []
    files.extend(_save_table(table3, out / "table3_worked_example_case_inputs.csv"))
    files.extend(_save_table(table4, out / "table4_caseA_deterministic_outputs.csv"))
    files.extend(_save_table(table5, out / "table5_report_audit_results.csv"))
    files.extend(_draw_lineage(table4, table3.iloc[0]["Examinee ID"], out))
    files.extend(_draw_audit_example(out))
    files.extend(_draw_grid(master, cases, out, case_labels))
    captions_path = out / "worked_example_caption_text.md"
    captions_path.write_text(captions, encoding="utf-8")
    files.append(captions_path)
    zip_bytes = _zip_files(files)
    return WorkedExampleResult(
        output_dir=out,
        table3=table3,
        table4=table4,
        table5=table5,
        captions_md=captions,
        selected_cases={k: _examinee_label(v) for k, v in cases.items()},
        files=files,
        zip_bytes=zip_bytes,
    )
