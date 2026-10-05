"""Evidence lineage tree: forensic indices → domains → review conclusions."""

from __future__ import annotations

from collections import Counter

import pandas as pd

try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover
    go = None  # type: ignore

SCENARIO_DOMAIN_ORDER = ("RT", "SIM", "PK", "TP")
SUPPORTING_DOMAIN_ORDER = ("MF", "CP")
DOMAIN_ORDER = SCENARIO_DOMAIN_ORDER + SUPPORTING_DOMAIN_ORDER

DOMAIN_LABELS = {
    "MF": "Misfit",
    "RT": "Response-Time",
    "SIM": "Similarity",
    "PK": "Preknowledge",
    "CP": "Change Point",
    "TP": "Tampering",
}

DOMAIN_TIER = {
    "RT": "scenario",
    "SIM": "scenario",
    "PK": "scenario",
    "TP": "scenario",
    "MF": "supporting",
    "CP": "supporting",
}

STRENGTH_LEGEND = (
    ("Strong", "#315A91"),
    ("Moderate", "#91AED3"),
    ("Weak", "#94A3B8"),
    ("None", "#64748B"),
    ("Unavailable", "#475569"),
)

PRIORITY_LEGEND = (
    ("Critical / Expedited", "#B42318"),
    ("High · Context review", "#E58A2B"),
    ("High", "#E58A2B"),
    ("Medium", "#E8C547"),
    ("Low", "#8295AB"),
)

# Evidence colors are distinct from review urgency.
_LAYER_COLORS = {
    "index": "#4F7EB6",  # flagged detector input
    "domain": "#315A91",  # evidence blue
    "conclusion": "#A78BFA",  # violet
    "other": "#64748B",
}

# Four columns: Plotly needs every domain to link forward to a rule node (layout anchor).
_LINEAGE_COLUMN_X = {
    "index": 0.01,
    "domain": 0.33,
    "rule": 0.66,
    "final": 0.96,
}

# Typography for Sankey node labels, titles, and column guides.
_LINEAGE_FONT_FAMILY = "Segoe UI, Roboto, Helvetica, Arial, sans-serif"
_LINEAGE_FONT_BASE = 15
_LINEAGE_FONT_TITLE = 16
_LINEAGE_FONT_SUBTITLE = 11
_LINEAGE_FONT_COLUMN = 13
_LINEAGE_FONT_TIER = 11
_LINEAGE_NODE_THICKNESS = 18
_LINEAGE_NODE_PAD = 14

_OTHER_INDEX_KEY = "__other_indices__"


def _clean_text(value: object) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def normalize_index_token(raw: str) -> str:
    """Merge flag / pval_flag variants of the same forensic index."""
    token = _clean_text(raw)
    if not token or token == _OTHER_INDEX_KEY:
        return token
    if ":" not in token:
        return token
    fn_part, _, idx_part = token.partition(":")
    idx_norm = idx_part.strip()
    idx_norm = (
        idx_norm.replace("_pval_flag", "")
        .replace("_pval", "")
        .replace("_flag", "")
        .strip()
    )
    fn_l = fn_part.strip().lower()
    idx_l = idx_norm.lower()
    if fn_l == "detect_tt":
        if idx_l.startswith("edi_sd_") or idx_l in {"edi_sd_no / co / ts", "tt_edi_sd"}:
            idx_norm = "tt_EDI_SD family"
        elif idx_l.startswith("edi_r_") or idx_l in {"edi_r_no / co / ts", "tt_edi_r"}:
            idx_norm = "tt_EDI_R family"
        elif idx_l.startswith("gbt_sd"):
            idx_norm = "tt_GBT_SD"
        elif idx_l.startswith("gbt_r"):
            idx_norm = "tt_GBT_R"
        elif idx_l.startswith("l_sd"):
            idx_norm = "tt_L_SD"
        elif idx_l.startswith("l_r"):
            idx_norm = "tt_L_R"
    elif fn_l == "detect_pm" and idx_l.startswith("pm_"):
        idx_norm = idx_norm
    return f"{fn_part}:{idx_norm}" if idx_norm else token


def _final_review_label(row: pd.Series) -> str:
    status = short_conclusion_label(row.get("Review_Priority"), row.get("Evidence_Status"))
    concern = _clean_text(row.get("Primary_Concern"))
    if concern:
        return f"{status} | {concern}"
    return status


def extract_individual_lineage(row: pd.Series) -> dict:
    """Structured lineage for one examinee: all domains + indices + final review."""
    domains: list[dict[str, object]] = []
    for code in DOMAIN_ORDER:
        strength = _clean_text(row.get(f"{code}_Strength")) or "none"
        rule = _clean_text(row.get(f"{code}_Rule")) or "—"
        indices: list[str] = []
        for token in parse_domain_indices(row.get(f"{code}_Flagged_Indices")):
            norm = normalize_index_token(token)
            if norm and norm not in indices:
                indices.append(norm)
        indices.sort(key=lambda item: short_index_label(item, truncate=False))
        domains.append(
            {
                "code": code,
                "label": DOMAIN_LABELS.get(code, code),
                "tier": DOMAIN_TIER.get(code, ""),
                "strength": strength,
                "rule": rule,
                "outcome": f"{strength} · {rule}",
                "indices": indices,
            }
        )
    review = {
        "priority": _clean_text(row.get("Review_Priority")) or "—",
        "status": _clean_text(row.get("Evidence_Status")) or "—",
        "concern": _clean_text(row.get("Primary_Concern")) or "—",
        "suggestion": _clean_text(row.get("Review_Suggestion")) or "—",
        "sources": _clean_text(row.get("Flag_Sources")) or "—",
        "label": _final_review_label(row),
    }
    return {"domains": domains, "review": review}


def short_index_label(raw: str, *, truncate: bool = True) -> str:
    """Compact index label for small Sankey nodes."""
    token = _clean_text(raw)
    if not token:
        return ""
    if token == _OTHER_INDEX_KEY:
        return "Other indices"
    if ":" in token:
        fn_part, _, idx_part = token.partition(":")
        fn_short = fn_part.replace("detect_", "")
        idx_short = idx_part.strip().replace("_flag", "").replace("_pval", "")
        if fn_short == "tt" and idx_short.startswith("tt_"):
            idx_short = idx_short[3:]
        if fn_short == "pm" and idx_short.startswith("pm_"):
            idx_short = idx_short[3:]
        if truncate and len(idx_short) > 22:
            idx_short = idx_short[:20] + "…"
        return f"{fn_short} · {idx_short}"
    return token.replace("detect_", "")


def short_conclusion_label(priority: object, status: object) -> str:
    """Compact final-review label (rightmost tree column)."""
    priority_text = _clean_text(priority) or "—"
    status_text = _clean_text(status) or "—"
    if "no substantive" in status_text.lower():
        return "No evidence pattern"
    if "weak" in status_text.lower() and priority_text.lower() == "low":
        return "Low · weak signal"
    if "single-scenario" in status_text.lower():
        return f"{priority_text} · single-scenario"
    if "cross-scenario" in status_text.lower():
        return f"{priority_text} · cross-scenario"
    return f"{priority_text} · {status_text.split('(')[0].strip()}"


def parse_domain_indices(text: object) -> list[str]:
    raw = _clean_text(text)
    if not raw:
        return []
    out: list[str] = []
    for chunk in raw.replace("|", ",").replace(";", ",").split(","):
        token = _clean_text(chunk)
        if token:
            out.append(token)
    return out


def _iter_examinee_flows(
    df: pd.DataFrame,
    *,
    examinee_id: str | None = None,
) -> list[tuple[str, str, str]]:
    """Yield (index_token, domain_code, conclusion_label) triples."""
    if df.empty:
        return []
    view = df
    if examinee_id is not None:
        view = view[view["Examinee_ID"].astype(str).eq(str(examinee_id))]
    flows: list[tuple[str, str, str]] = []
    for _, row in view.iterrows():
        conclusion = short_conclusion_label(
            row.get("Review_Priority"),
            row.get("Evidence_Status"),
        )
        for domain in DOMAIN_ORDER:
            col = f"{domain}_Flagged_Indices"
            if col not in view.columns:
                continue
            for index_token in parse_domain_indices(row.get(col)):
                norm = normalize_index_token(index_token)
                if norm:
                    flows.append((norm, domain, conclusion))
    return flows


def aggregate_flow_counts(
    df: pd.DataFrame,
    *,
    examinee_id: str | None = None,
    max_indices: int = 26,
) -> tuple[Counter[tuple[str, str]], Counter[tuple[str, str]], list[str]]:
    """
    Return (index→domain counts, domain→conclusion counts, kept index keys).
    Low-frequency indices are rolled into _OTHER_INDEX_KEY for readability.
    """
    raw_flows = _iter_examinee_flows(df, examinee_id=examinee_id)
    index_totals: Counter[str] = Counter()
    for index_token, _domain, _conclusion in raw_flows:
        index_totals[index_token] += 1

    if examinee_id is None and max_indices > 0 and len(index_totals) > max_indices:
        keep = {key for key, _ in index_totals.most_common(max_indices - 1)}
        kept_indices = sorted(keep)
    else:
        keep = set(index_totals)
        kept_indices = sorted(keep)

    id_to_domain: Counter[tuple[str, str]] = Counter()
    domain_to_conclusion: Counter[tuple[str, str]] = Counter()
    for index_token, domain, conclusion in raw_flows:
        index_key = index_token if index_token in keep else _OTHER_INDEX_KEY
        id_to_domain[(index_key, domain)] += 1
        domain_to_conclusion[(domain, conclusion)] += 1

    if _OTHER_INDEX_KEY in {k for k, _ in id_to_domain} and _OTHER_INDEX_KEY not in kept_indices:
        kept_indices.append(_OTHER_INDEX_KEY)

    return id_to_domain, domain_to_conclusion, kept_indices


def _node_id(layer: str, key: str) -> str:
    return f"{layer}|{key}"


def build_lineage_sankey(
    df: pd.DataFrame,
    *,
    examinee_id: str | None = None,
    max_indices: int = 26,
    title: str = "Evidence lineage",
    height: int = 520,
) -> go.Figure | None:
    """Left-to-right Sankey: forensic index → evidence domain → review conclusion."""
    if go is None or df.empty:
        return None

    id_to_domain, domain_to_conclusion, _kept = aggregate_flow_counts(
        df,
        examinee_id=examinee_id,
        max_indices=max_indices,
    )
    if not id_to_domain and not domain_to_conclusion:
        return None

    index_keys = sorted({src for src, _ in id_to_domain})
    if _OTHER_INDEX_KEY in index_keys:
        index_keys = [k for k in index_keys if k != _OTHER_INDEX_KEY] + [_OTHER_INDEX_KEY]
    domain_keys = [d for d in DOMAIN_ORDER if any(k == d for k, _ in domain_to_conclusion)]
    conclusion_keys = sorted({dst for _, dst in domain_to_conclusion})

    node_keys: list[str] = []
    node_labels: list[str] = []
    node_colors: list[str] = []
    node_layers: list[str] = []
    index_pos: dict[str, int] = {}
    domain_pos: dict[str, int] = {}
    conclusion_pos: dict[str, int] = {}

    def _add_node(layer: str, key: str, label: str, color: str) -> int:
        node_id = _node_id(layer, key)
        lookup = index_pos if layer == "index" else domain_pos if layer == "domain" else conclusion_pos
        if key in lookup:
            return lookup[key]
        idx = len(node_keys)
        node_keys.append(node_id)
        node_labels.append(label)
        node_colors.append(color)
        node_layers.append(layer)
        lookup[key] = idx
        return idx

    for key in index_keys:
        color = _LAYER_COLORS["other"] if key == _OTHER_INDEX_KEY else _LAYER_COLORS["index"]
        _add_node("index", key, short_index_label(key), color)
    for key in domain_keys:
        _add_node("domain", key, f"{key} · {DOMAIN_LABELS.get(key, key)}", _LAYER_COLORS["domain"])
    for key in conclusion_keys:
        _add_node("conclusion", key, key, conclusion_priority_color(key))

    n_index = max(len(index_keys), 1)
    n_domain = max(len(domain_keys), 1)
    n_conclusion = max(len(conclusion_keys), 1)
    node_x: list[float] = []
    node_y: list[float] = []
    for layer, key in zip(node_layers, [n.split("|", 1)[1] for n in node_keys]):
        if layer == "index":
            rank = index_keys.index(key)
            node_x.append(0.01)
            node_y.append(1.0 - (rank + 0.5) / n_index)
        elif layer == "domain":
            rank = domain_keys.index(key)
            node_x.append(0.5)
            node_y.append(1.0 - (rank + 0.5) / n_domain)
        else:
            rank = conclusion_keys.index(key)
            node_x.append(0.99)
            node_y.append(1.0 - (rank + 0.5) / n_conclusion)

    sources: list[int] = []
    targets: list[int] = []
    values: list[float] = []
    link_colors: list[str] = []
    for (index_key, domain_key), count in sorted(id_to_domain.items(), key=lambda item: -item[1]):
        if count <= 0:
            continue
        sources.append(index_pos[index_key])
        targets.append(domain_pos[domain_key])
        values.append(int(count))
        link_colors.append("rgba(79, 126, 182, 0.30)")

    for (domain_key, conclusion_key), count in sorted(domain_to_conclusion.items(), key=lambda item: -item[1]):
        if count <= 0:
            continue
        sources.append(domain_pos[domain_key])
        targets.append(conclusion_pos[conclusion_key])
        values.append(int(count))
        link_colors.append("rgba(49, 90, 145, 0.30)")

    if not values:
        return None

    subtitle = (
        f"Examinee {examinee_id}"
        if examinee_id
        else f"{len(df):,} examinees · aggregated index→domain→conclusion flows"
    )

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                textfont=dict(family=_LINEAGE_FONT_FAMILY, size=_LINEAGE_FONT_BASE,
                              color="#17324D", shadow="none", weight=600),
                node=dict(
                    pad=_LINEAGE_NODE_PAD,
                    thickness=_LINEAGE_NODE_THICKNESS,
                    line=dict(color="rgba(148, 163, 184, 0.45)", width=0.6),
                    label=node_labels,
                    color=node_colors,
                    x=node_x,
                    y=node_y,
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color=link_colors,
                    hovertemplate="%{source.label} → %{target.label}<br>flow=%{value}<extra></extra>",
                ),
            )
        ]
    )
    fig.update_layout(
        title=dict(text=f"{title}<br><sup style='color:#52677D'>{subtitle}</sup>", x=0.01, font=dict(size=_LINEAGE_FONT_TITLE)),
        font=dict(family=_LINEAGE_FONT_FAMILY, size=_LINEAGE_FONT_BASE, color="#17324D"),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=16, r=16, t=64, b=16),
        height=height,
    )
    return fig


def _strength_color(strength: str) -> str:
    text = strength.lower()
    if text in {"strong", "critical"}:
        return "#315A91"
    if text in {"moderate", "medium"}:
        return "#91AED3"
    if text in {"weak", "low"}:
        return "#94A3B8"
    if text in {"unavailable", "missing"}:
        return "#475569"
    if text in {"none", ""}:
        return "#64748B"
    return "#64748B"


def priority_level_key(priority: object) -> str:
    text = _clean_text(priority).lower()
    if "critical" in text or "expedited" in text:
        return "critical"
    if text == "high" or text.startswith("high"):
        return "high"
    if text == "medium" or text.startswith("medium"):
        return "medium"
    if text == "low" or text.startswith("low"):
        return "low"
    return "unknown"


def priority_color(priority: object) -> str:
    return {
        "critical": "#B42318",
        "high": "#E58A2B",
        "medium": "#E8C547",
        "low": "#8295AB",
        "unknown": "#64748B",
    }.get(priority_level_key(priority), "#64748B")


def priority_style(priority: object) -> dict[str, str]:
    """Review urgency colors, distinct from evidence strength."""
    return {
        "critical": {"bg": "#FEF2F2", "text": "#991B1B", "border": "#B42318"},
        "high": {"bg": "#FFF7ED", "text": "#9A3412", "border": "#E58A2B"},
        "medium": {"bg": "#FFFBEB", "text": "#715C12", "border": "#E8C547"},
        "low": {"bg": "#F1F5F9", "text": "#475569", "border": "#8295AB"},
        "unknown": {"bg": "#FFFFFF", "text": "#64748B", "border": "#CBD5E1"},
    }.get(priority_level_key(priority), {"bg": "#FFFFFF", "text": "#64748B", "border": "#CBD5E1"})


def strength_style(strength: object) -> dict[str, str]:
    text = _clean_text(strength).lower()
    if text == "strong":
        return {"bg": "#EDF2FA", "text": "#315A91", "border": "#315A91"}
    if text == "moderate":
        return {"bg": "#F3F6FC", "text": "#315A91", "border": "#91AED3"}
    if text == "weak":
        return {"bg": "#F8FAFC", "text": "#475569", "border": "#94A3B8"}
    if text == "unavailable":
        return {"bg": "#F1F5F9", "text": "#334155", "border": "#475569"}
    if text == "none":
        return {"bg": "#FFFFFF", "text": "#64748B", "border": "#64748B"}
    return {"bg": "#FFFFFF", "text": "#64748B", "border": "#CBD5E1"}


def conclusion_priority_color(conclusion: object) -> str:
    text = _clean_text(conclusion)
    if not text:
        return _LAYER_COLORS["conclusion"]
    for label, color in PRIORITY_LEGEND:
        if text.lower().startswith(label.lower()):
            return color
    head = text.split("·", 1)[0].strip()
    if head:
        return priority_color(head)
    return _LAYER_COLORS["conclusion"]


def _domain_y_position(code: str) -> float:
    """Scenario domains in upper band; MF/CP supporting domains in lower band (Plotly y=0 is top)."""
    if code in SCENARIO_DOMAIN_ORDER:
        rank = SCENARIO_DOMAIN_ORDER.index(code)
        n = len(SCENARIO_DOMAIN_ORDER)
        band_lo, band_hi = 0.07, 0.43
    elif code in SUPPORTING_DOMAIN_ORDER:
        rank = SUPPORTING_DOMAIN_ORDER.index(code)
        n = len(SUPPORTING_DOMAIN_ORDER)
        band_lo, band_hi = 0.57, 0.93
    else:
        return 0.5
    return band_lo + (rank + 0.5) / n * (band_hi - band_lo)


def _domain_compact_label(code: str) -> str:
    label = DOMAIN_LABELS.get(code, code)
    if DOMAIN_TIER.get(code) == "supporting":
        return f"{code}·{label} · supporting only"
    return f"{code}·{label}"


def _domain_drives_priority(domain: dict[str, object]) -> bool:
    """Only primary scenario domains draw a full-strength path into review priority."""
    return DOMAIN_TIER.get(str(domain.get("code", ""))) == "scenario"


def _supporting_link_value(index_count: int) -> float:
    """Keep supporting-only domains visible without making them look priority-driving."""
    return max(0.7, min(1.6, 0.22 * max(index_count, 1)))


def _outcome_compact_label(code: str, strength: str, rule: str) -> str:
    r = _clean_text(rule) or "—"
    if len(r) > 9:
        r = r[:8] + "…"
    return f"{code}·{strength}·{r}"


def _rule_compact_label(code: str, rule: str) -> str:
    r = _clean_text(rule) or "—"
    if len(r) > 12:
        r = r[:11] + "…"
    return f"{code}·{r}"


def _domain_display_label(code: str) -> str:
    label = DOMAIN_LABELS.get(code, code)
    tier = DOMAIN_TIER.get(code, "")
    if tier == "supporting":
        return f"{code} · {label} (supporting)"
    return f"{code} · {label}"


def _legend_layout(*, title: str, y: float, x: float = 0.0, xanchor: str = "left") -> dict:
    return dict(
        orientation="h",
        yanchor="top",
        y=y,
        xanchor=xanchor,
        x=x,
        font=dict(size=_LINEAGE_FONT_BASE, color="#17324D"),
        bgcolor="#F8FAFC",
        bordercolor="rgba(148, 163, 184, 0.35)",
        borderwidth=1,
        title=dict(text=title, font=dict(size=_LINEAGE_FONT_SUBTITLE, color="#52677D")),
    )


def _legend_row_html(title: str, items: tuple[tuple[str, str], ...]) -> str:
    parts = [
        f'<span class="psymas-lineage-legend-title">{title}</span>',
    ]
    for label, color in items:
        parts.append(
            f'<span class="item"><span class="dot" style="background:{color};"></span>'
            f"{label}</span>"
        )
    return f'<div class="psymas-lineage-legend-row">{"".join(parts)}</div>'


def lineage_dual_legend_html() -> str:
    """Evidence roles and a separate, explicit four-level review-priority key."""
    priority_items = ''.join(
        '<div style="display:flex;align-items:flex-start;gap:0.45rem;margin-top:0.4rem;">'
        f'<span class="dot" style="background:{priority_color(label)};"></span>'
        f'<span style="font-size:0.76rem;font-weight:700;color:#17324D;">{label}</span></div>'
        for label in ('Critical / Expedited','High','Medium','Low')
    )
    return (
        '<div class="psymas-lineage-legend">'
        '<div class="psymas-lineage-legend-heading">Color key</div>'
        '<div class="psymas-lineage-legend-row"><span class="dot" style="background:#315A91;"></span>'
        '<span><b>Strong evidence</b><em>domain evidence strength</em></span></div>'
        '<div class="psymas-lineage-legend-row"><span class="dot" style="background:#91AED3;"></span>'
        '<span><b>Moderate evidence</b><em>domain evidence strength</em></span></div>'
        '<div class="psymas-lineage-legend-row"><span class="dot" style="background:#4F7EB6;"></span>'
        '<span><b>Flagged input</b><em>detector flag is active</em></span></div>'
        '<div class="psymas-lineage-legend-row"><span class="dot" style="background:#8295AB;"></span>'
        '<span><b>Supporting only</b><em>context for interpretation</em></span></div>'
        '<div class="psymas-lineage-legend-row"><span class="dot" style="background:#D8E2EC;"></span>'
        '<span><b>Inactive/context</b><em>shown but not counted toward priority</em></span></div>'
        + '<div class="psymas-lineage-legend-row"><span style="width:100%;">'
        '<b>Review priority</b><em>human-review urgency</em>'
        + priority_items + '</span></div>'
        + "</div>"
    )


def _add_strength_legend(fig: go.Figure, *, include_priority: bool = False) -> None:
    """Plotly-native legend (cohort charts). Individual lineage uses lineage_dual_legend_html() instead."""
    if include_priority:
        fig.update_layout(showlegend=False, margin=dict(b=48))
        return
    for label, color in STRENGTH_LEGEND:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=10, color=color, symbol="square", line=dict(width=0)),
                name=label,
                showlegend=True,
                hoverinfo="skip",
            )
        )
    fig.update_layout(
        legend=_legend_layout(title="Domain strength", y=-0.03),
        margin=dict(b=96),
    )


def _add_tier_guides(fig: go.Figure) -> None:
    column_guides: tuple[tuple[float, str, int], ...] = (
        (_LINEAGE_COLUMN_X["index"], "Flagged index", _LINEAGE_NODE_THICKNESS // 2 + 3),
        (_LINEAGE_COLUMN_X["domain"], "Domain", _LINEAGE_NODE_THICKNESS // 2 + 3),
        (_LINEAGE_COLUMN_X["rule"], "Rule", _LINEAGE_NODE_THICKNESS // 2 + 3),
        (_LINEAGE_COLUMN_X["final"], "Review priority", 20),
    )
    for x_pos, label, xshift in column_guides:
        fig.add_annotation(
            x=x_pos,
            y=1,
            yshift=30,
            yanchor="bottom",
            xref="paper",
            yref="paper",
            text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(size=_LINEAGE_FONT_COLUMN, color="#17324D"),
            xanchor="left",
            xshift=xshift,
        )


def individual_lineage_index_rows(row: pd.Series) -> pd.DataFrame:
    """One row per forensic index with full downstream review fields (for comparison)."""
    lineage = extract_individual_lineage(row)
    review = lineage["review"]
    rows: list[dict[str, object]] = []
    for domain in lineage["domains"]:
        code = str(domain["code"])
        for index_token in domain["indices"]:
            rows.append(
                {
                    "Index": short_index_label(str(index_token), truncate=False),
                    "Domain": f"{code} · {domain['label']}",
                    "Domain_Strength": domain["strength"],
                    "Domain_Rule": domain["rule"],
                    "Review_Priority": review["priority"],
                    "Evidence_Status": review["status"],
                    "Primary_Concern": review["concern"],
                    "Review_Suggestion": review["suggestion"],
                    "Flag_Sources": review["sources"],
                }
            )
    return pd.DataFrame(rows)


def individual_lineage_domain_rows(row: pd.Series) -> pd.DataFrame:
    """Fixed six-domain summary — same shape for every examinee."""
    lineage = extract_individual_lineage(row)
    review = lineage["review"]
    rows: list[dict[str, object]] = []
    for domain in lineage["domains"]:
        indices = domain["indices"]
        rows.append(
            {
                "Tier": "Scenario" if domain.get("tier") == "scenario" else "Supporting",
                "Domain": _domain_display_label(str(domain["code"])),
                "Strength": domain["strength"],
                "Rule": domain["rule"],
                "Flagged_Indices": ", ".join(short_index_label(i, truncate=False) for i in indices) or "—",
                "Review_Priority": review["priority"],
                "Evidence_Status": review["status"],
                "Primary_Concern": review["concern"],
            }
        )
    return pd.DataFrame(rows)


_INACTIVE_DOMAIN_COLOR = "#D8E2EC"
_INACTIVE_RULE_COLOR = "#D8E2EC"
_SUPPORTING_NODE_COLOR = "#8295AB"
_INACTIVE_LINK_COLOR = "rgba(100, 116, 139, 0.08)"
_SUPPORTING_LINK_COLOR = "rgba(100, 116, 139, 0.16)"
_INACTIVE_LINK_VALUE = 0.12


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    color = str(hex_color or "").lstrip("#")
    if len(color) != 6:
        return f"rgba(100, 116, 139, {alpha:.2f})"
    try:
        r = int(color[0:2], 16)
        g = int(color[2:4], 16)
        b = int(color[4:6], 16)
    except ValueError:
        return f"rgba(100, 116, 139, {alpha:.2f})"
    return f"rgba({r}, {g}, {b}, {alpha:.2f})"


def _domain_has_index_evidence(domain: dict[str, object]) -> bool:
    indices = domain.get("indices")
    return bool(indices) if isinstance(indices, list) else bool(indices)


def build_individual_lineage_sankey(
    row: pd.Series,
    *,
    examinee_id: str | None = None,
    title: str = "Evidence lineage",
    included_domains: tuple[str, ...] | list[str] | None = None,
    domain_status: dict[str, dict] | None = None,
    height: int = 600,
) -> go.Figure | None:
    """
    Four-layer Sankey (Plotly layout requires a forward link per domain row):
    index → domain (color = strength) → rule → review priority.
    All six domain rows are always shown; only active paths reach priority.
    """
    if go is None:
        return None

    lineage = extract_individual_lineage(row)
    review = lineage["review"]
    domains: list[dict[str, object]] = lineage["domains"]
    if included_domains:
        keep_domains = {str(domain) for domain in included_domains}
        domains = [domain for domain in domains if str(domain.get("code")) in keep_domains]

    if not domains:
        return None

    index_entries: list[tuple[str, str]] = []
    active_domains: list[dict[str, object]] = []
    for domain in domains:
        code = str(domain["code"])
        if not _domain_has_index_evidence(domain):
            continue
        active_domains.append(domain)
        for index_token in domain["indices"]:
            index_entries.append((str(index_token), code))

    indices_by_domain: dict[str, list[str]] = {str(d["code"]): [] for d in active_domains}
    for index_token, domain_code in index_entries:
        indices_by_domain.setdefault(domain_code, []).append(index_token)

    node_labels: list[str] = []
    node_colors: list[str] = []
    node_x: list[float] = []
    node_y: list[float] = []
    index_pos: dict[str, int] = {}
    domain_pos: dict[str, int] = {}
    rule_pos: dict[str, int] = {}

    for domain_code, tokens in indices_by_domain.items():
        n = max(len(tokens), 1)
        for rank, index_token in enumerate(tokens):
            index_pos[index_token] = len(node_labels)
            node_labels.append(short_index_label(index_token, truncate=True))
            node_colors.append(_LAYER_COLORS["index"] if DOMAIN_TIER.get(domain_code) == "scenario" else _SUPPORTING_NODE_COLOR)
            node_x.append(_LINEAGE_COLUMN_X["index"])
            # Distribute the full input column inside the plot, rather than
            # clustering around domain centers and pushing later labels outside.
            input_rank = len(index_pos) - 1
            node_y.append(0.06 + 0.84 * input_rank / max(len(index_entries) - 1, 1))

    for domain in domains:
        code = str(domain["code"])
        has_indices = _domain_has_index_evidence(domain)
        domain_pos[code] = len(node_labels)
        label = _domain_compact_label(code)
        if has_indices:
            label += f" · {domain['strength']}"
        if not has_indices:
            status=(domain_status or {}).get(code,{})
            missing=str(status.get('Available','')).lower() in {'0','0.0','false'} or str(domain['strength']).lower() in {'unavailable','missing'}
            label = f"{label} · unavailable" if missing else f"{label} · no index"
        node_labels.append(label)
        node_colors.append(
            (_strength_color(str(domain["strength"])) if _domain_drives_priority(domain) else _SUPPORTING_NODE_COLOR) if has_indices else _INACTIVE_DOMAIN_COLOR
        )
        node_x.append(_LINEAGE_COLUMN_X["domain"])
        node_y.append(_domain_y_position(code))

    for domain in domains:
        code = str(domain["code"])
        has_indices = _domain_has_index_evidence(domain)
        rule = str(domain["rule"])
        rule_pos[code] = len(node_labels)
        node_labels.append(_rule_compact_label(code, rule))
        node_colors.append((_strength_color(str(domain["strength"])) if _domain_drives_priority(domain) else _SUPPORTING_NODE_COLOR) if has_indices else _INACTIVE_DOMAIN_COLOR)
        node_x.append(_LINEAGE_COLUMN_X["rule"])
        node_y.append(_domain_y_position(code))

    final_pos = len(node_labels)
    final_label = short_conclusion_label(review.get("priority"), review.get("status"))
    node_labels.append(final_label)
    node_colors.append(priority_color(review["priority"]))
    node_x.append(_LINEAGE_COLUMN_X["final"])
    node_y.append(0.5)

    sources: list[int] = []
    targets: list[int] = []
    values: list[int] = []
    link_colors: list[str] = []

    for index_token, domain_code in index_entries:
        domain = next((d for d in active_domains if str(d.get("code")) == domain_code), {})
        drives_priority = _domain_drives_priority(domain)
        sources.append(index_pos[index_token])
        targets.append(domain_pos[domain_code])
        values.append(1 if drives_priority else 0.35)
        link_colors.append("rgba(79, 126, 182, 0.32)" if drives_priority else "rgba(100, 116, 139, 0.24)")

    for domain in domains:
        code = str(domain["code"])
        has_indices = _domain_has_index_evidence(domain)
        n_idx = len(domain["indices"]) if has_indices else 0
        drives_priority = _domain_drives_priority(domain)
        sources.append(domain_pos[code])
        targets.append(rule_pos[code])
        if not has_indices:
            values.append(_INACTIVE_LINK_VALUE)
            link_colors.append(_INACTIVE_LINK_COLOR)
        elif drives_priority:
            values.append(max(1, n_idx))
            link_colors.append("rgba(49, 90, 145, 0.30)")
        else:
            values.append(_supporting_link_value(n_idx))
            link_colors.append(_SUPPORTING_LINK_COLOR)

    for domain in domains:
        if not _domain_has_index_evidence(domain):
            continue
        code = str(domain["code"])
        n_idx = len(domain["indices"])
        sources.append(rule_pos[code])
        targets.append(final_pos)
        if _domain_drives_priority(domain):
            values.append(max(1, n_idx))
            link_colors.append(_hex_to_rgba(_strength_color(str(domain["strength"])), 0.38))
        else:
            values.append(_supporting_link_value(n_idx))
            link_colors.append(_SUPPORTING_LINK_COLOR)

    if not values:
        return None

    n_indices = len(index_entries)
    n_flagged_indices = sum(len(d["indices"]) for d in active_domains)
    n_active_domains = len(active_domains)
    n_priority_domains = sum(1 for d in active_domains if _domain_drives_priority(d))
    n_supporting_domains = max(0, n_active_domains - n_priority_domains)
    n_inactive_domains = len(domains) - n_active_domains
    eid = examinee_id or _clean_text(row.get("Examinee_ID")) or "—"
    final_short = short_conclusion_label(review.get("priority"), review.get("status"))
    dynamic_height = max(int(height), 240 + n_indices * 34 + len(domains) * 26)

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                textfont=dict(family=_LINEAGE_FONT_FAMILY, size=16,
                              color="#17324D", shadow="none", weight=600),
                node=dict(
                    pad=_LINEAGE_NODE_PAD,
                    thickness=_LINEAGE_NODE_THICKNESS,
                    line=dict(color="rgba(148, 163, 184, 0.45)", width=0.6),
                    label=node_labels,
                    color=node_colors,
                    x=node_x,
                    y=node_y,
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color=link_colors,
                    hovertemplate="%{source.label} → %{target.label}<extra></extra>",
                ),
            )
        ]
    )
    fig.update_layout(
        title=dict(
            text=(
                f"{title}<br><sup style='color:#52677D'>Examinee {eid} · "
                f"{n_indices} displayed index families · {n_flagged_indices} flagged · {n_priority_domains} priority domains · "
                f"final: {final_short}</sup>"
            ),
            x=0.01,
            y=1,
            yref="container",
            yanchor="top",
            pad=dict(t=12),
            font=dict(size=_LINEAGE_FONT_TITLE),
        ),
        font=dict(family=_LINEAGE_FONT_FAMILY, size=_LINEAGE_FONT_BASE, color="#17324D"),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=24, r=210, t=148, b=72),
        height=dynamic_height,
        showlegend=False,
    )
    # Sankey places terminal labels to the left. Use an explicit right-side
    # annotation so priority text cannot be confused with upstream rule labels.
    if index_entries:
        hover_labels = list(node_labels)
        fig.data[0].node.customdata = hover_labels
        fig.data[0].node.hovertemplate = '%{customdata}<extra></extra>'
        # Strength remains in hover and color; keep displayed domain labels concise.
        for domain in domains:
            if _domain_has_index_evidence(domain):
                code = str(domain['code'])
                node_labels[domain_pos[code]] = _domain_compact_label(code)
        node_labels[final_pos] = ''
        fig.data[0].node.label = node_labels
        fig.add_annotation(name='review_priority_label',x=_LINEAGE_COLUMN_X['final'],y=0.5,
            xref='paper',yref='paper',xshift=20,text='<b>' + final_label.replace(' · ','<br>') + '</b>',
            showarrow=False,xanchor='left',align='left',font=dict(size=16,color='#17324D'))
    if not index_entries:
        fig.data = ()
        for row_index, domain in enumerate(domains):
            code = str(domain['code'])
            y = 1 - (row_index + 1) / (len(domains) + 1)
            status = (domain_status or {}).get(code, {})
            missing = str(status.get('Available','')).lower() in {'0','0.0','false'} or str(domain['strength']).lower() in {'unavailable','missing'}
            label = _domain_compact_label(code) + (' · unavailable' if missing else ' · no index')
            display_label = label.replace(' · supporting only · ', '<br>supporting only · ')
            fig.add_shape(type='rect',xref='paper',yref='paper',x0=_LINEAGE_COLUMN_X['domain'],
                x1=_LINEAGE_COLUMN_X['rule'],y0=y-0.006,y1=y+0.006,
                fillcolor=_INACTIVE_LINK_COLOR,line=dict(width=0))
            for layer,text in [('domain',display_label),('rule',_rule_compact_label(code,str(domain['rule'])))]:
                x = _LINEAGE_COLUMN_X[layer]
                fig.add_shape(type='rect',xref='paper',yref='paper',x0=x-0.01,x1=x+0.01,
                    y0=y-0.006,y1=y+0.006,fillcolor=_INACTIVE_DOMAIN_COLOR,line=dict(width=0))
                fig.add_annotation(x=x+0.013,y=y,xref='paper',yref='paper',text=text,
                    showarrow=False,xanchor='left',yanchor='middle',font=dict(size=16,color='#17324D'))
        fig.add_shape(type='rect',xref='paper',yref='paper',x0=_LINEAGE_COLUMN_X['final']-0.01,
            x1=_LINEAGE_COLUMN_X['final']+0.01,y0=0.49,y1=0.51,
            fillcolor=priority_color(review['priority']),line=dict(width=0))
        fig.add_annotation(name='review_priority_label',x=_LINEAGE_COLUMN_X['final'],y=0.5,xref='paper',yref='paper',xshift=20,
            text='<b>' + final_label.replace(' · ','<br>') + '</b>',showarrow=False,xanchor='left',align='left',font=dict(size=16,color='#17324D'))
        fig.update_xaxes(visible=False,range=[0,1]);fig.update_yaxes(visible=False,range=[0,1])
    _add_tier_guides(fig)
    return fig


def lineage_summary_rows(df: pd.DataFrame, *, examinee_id: str | None = None) -> pd.DataFrame:
    """Tabular summary backing the lineage chart."""
    id_to_domain, domain_to_conclusion, _ = aggregate_flow_counts(df, examinee_id=examinee_id)
    rows: list[dict[str, object]] = []
    for (index_key, domain_key), count in sorted(id_to_domain.items(), key=lambda item: (-item[1], item[0][0])):
        rows.append(
            {
                "Stage": "Index → Domain",
                "Source": short_index_label(index_key),
                "Target": f"{domain_key} · {DOMAIN_LABELS.get(domain_key, domain_key)}",
                "Flow": int(count),
            }
        )
    for (domain_key, conclusion_key), count in sorted(
        domain_to_conclusion.items(), key=lambda item: (-item[1], item[0][0])
    ):
        rows.append(
            {
                "Stage": "Domain → Conclusion",
                "Source": f"{domain_key} · {DOMAIN_LABELS.get(domain_key, domain_key)}",
                "Target": conclusion_key,
                "Flow": int(count),
            }
        )
    return pd.DataFrame(rows)
