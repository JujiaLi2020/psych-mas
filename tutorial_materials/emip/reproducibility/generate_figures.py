"""Create compact, data-derived teaching figures for the EM:IP tutorial."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "tutorial_materials" / "emip" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
DB_PACKET = ROOT / "tutorial_materials" / "emip" / "case_332"


def load_session():
    with zipfile.ZipFile(ROOT / "data" / "psymas_demo_evaluated_snapshot.zip") as z:
        return json.loads(z.read("session_inputs.json"))


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#d9e1e8", linewidth=0.7)
    ax.set_axisbelow(True)


def main():
    session = load_session()
    responses = pd.DataFrame(session["forensic_responses"])
    rts = pd.DataFrame(session["forensic_rt_data"])
    response = responses.iloc[331].astype(float).to_numpy()
    rt = rts.iloc[331].astype(float).to_numpy()
    item_names = list(responses.columns)
    positions = np.arange(1, len(item_names) + 1)
    accuracy = responses.mean(axis=0).to_numpy()
    rt_mean = rts.mean(axis=0).to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    axes[0].bar(positions, accuracy, color="#8ea3b8", edgecolor="#49657d", linewidth=0.4)
    axes[0].bar(positions, np.where(response == 1, accuracy, 0), color="#1f7a68", width=0.8, label="selected correct")
    axes[0].bar(positions, np.where(response == 0, accuracy, 0), color="#c73b3b", width=0.8, label="selected incorrect")
    axes[0].set(title="Selected responses against item accuracy", xlabel="Item number", ylabel="Cohort item accuracy")
    axes[0].set_ylim(0, 1.05); axes[0].legend(frameon=False, fontsize=8); style(axes[0])
    axes[1].plot(positions, rt, color="#17647a", marker="o", markersize=3, label="Examinee 332")
    axes[1].plot(positions, rt_mean, color="#c47a0d", linestyle="--", label="Cohort item mean")
    axes[1].set(title="Response time relative to item means", xlabel="Item number", ylabel="Response time")
    axes[1].legend(frameon=False, fontsize=8); style(axes[1])
    fig.savefig(OUT / "case_332_performance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    domain = pd.read_csv(DB_PACKET / "domain_evidence.csv")
    order = ["MF", "RT", "SIM", "PK", "TP", "CP"]
    colors = {"strong": "#c73b3b", "moderate": "#c47a0d", "weak": "#6f879d", "none": "#dce4eb", "unavailable": "#9aaabd"}
    values = [domain.set_index("Domain").loc[d, "Strength"] for d in order]
    fig, ax = plt.subplots(figsize=(7.5, 3.2), constrained_layout=True)
    ax.barh(order[::-1], [1] * len(order), color=[colors[v] for v in values[::-1]], edgecolor="white")
    for y, d, v in zip(range(len(order)), order[::-1], values[::-1]):
        ax.text(0.02, y, f"{d}  {v}", va="center", ha="left", color="white" if v in {"strong", "moderate"} else "#172a3a", weight="bold")
    ax.set_xlim(0, 1); ax.set_xticks([]); ax.set_title("Examinee 332: governed domain profile", loc="left", weight="bold")
    ax.spines[:].set_visible(False); ax.grid(False)
    fig.savefig(OUT / "case_332_domain_profile.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    b3 = pd.read_csv(DB_PACKET / "b3_input.csv")
    flagged = b3[(b3["Flag"] == 1) & (b3["Evidence_Eligible"] == 1)].copy()
    family = flagged.groupby(["Aggregation_Family", "Domain", "Role"], as_index=False).size()
    fig, ax = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
    ax.set_xlim(0, 3); ax.set_ylim(-0.5, max(2, len(family) - 0.5)); ax.axis("off")
    active_domains = [d for d in ("RT", "TP") if d in set(family["Domain"])]
    domain_y = {d: y for d, y in zip(active_domains, (0.8, 1.8))}
    for i, row in family.iterrows():
        y = len(family) - 1 - i
        x1, x2 = 0.25, 1.55
        ax.annotate("", xy=(x2 - 0.08, domain_y.get(row.Domain, 2.9)), xytext=(x1 + 0.08, y), arrowprops={"arrowstyle": "-", "color": "#6f879d", "lw": 1.5})
        ax.text(x1, y, str(row.Aggregation_Family), ha="center", va="center", fontsize=9, bbox={"boxstyle": "round,pad=.35", "fc": "#e8f2f5", "ec": "#2b9bb6"})
    for d, y in domain_y.items():
        ax.text(x2, y, d, ha="center", va="center", fontsize=10, weight="bold", bbox={"boxstyle": "round,pad=.4", "fc": "#fff4dc" if d == "RT" else "#fde8e8", "ec": "#c47a0d" if d == "RT" else "#c73b3b"})
    ax.text(2.6, 1.35, "Critical / Expedited", ha="center", va="center", fontsize=11, weight="bold", bbox={"boxstyle": "round,pad=.5", "fc": "#fbe5e5", "ec": "#c73b3b"})
    for d, y in domain_y.items():
        ax.annotate("", xy=(2.4, 1.35), xytext=(1.75, y), arrowprops={"arrowstyle": "-", "color": "#c47a0d" if d == "RT" else "#c73b3b", "lw": 2})
    ax.set_title("Evidence families to governed domains and review priority", loc="left", weight="bold")
    fig.savefig(OUT / "case_332_evidence_lineage.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
