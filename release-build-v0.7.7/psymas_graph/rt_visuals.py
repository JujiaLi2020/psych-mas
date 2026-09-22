"""Response-time visualizations used by graph nodes."""

from pathlib import Path
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent


def response_time_histograms(
    rt_df: pd.DataFrame,
    resp_df: pd.DataFrame,
    color: str = "lightgray",
) -> str:
    """Create item-level RT histograms and return the saved image path."""
    n_responses = len(resp_df)
    n_rt_items = rt_df.shape[1]
    n_response_items = resp_df.shape[1]
    if n_rt_items != n_response_items:
        raise ValueError(
            f"RT and Resp must have the same number of columns. RT: {n_rt_items}, Resp: {n_response_items}"
        )
    if n_responses == 0:
        raise ValueError("Resp has no rows")
    if n_rt_items == 0:
        raise ValueError("RT has no columns")

    correct_proportions = (resp_df.sum(axis=0) / n_responses).round(2)
    figure, axes = plt.subplots(3, 4, figsize=(12, 9))
    axes = axes.flatten()
    shown_items = min(n_rt_items, 12)
    for item_index in range(shown_items):
        axis = axes[item_index]
        axis.hist(rt_df.iloc[:, item_index].dropna(), bins=15, color=color, edgecolor="white")
        proportion = correct_proportions.iloc[item_index]
        proportion_text = f"{proportion:.2f}" if pd.notna(proportion) else "—"
        axis.set_title(
            f"RT Distr. for Item {item_index + 1}\nCorrect Proportion: {proportion_text}",
            fontsize=9,
        )
    for axis_index in range(shown_items, len(axes)):
        axes[axis_index].set_visible(False)
    plt.tight_layout()

    candidates = (
        PROJECT_DIR / "data" / "rt_hist.png",
        Path(tempfile.gettempdir()) / "psych_mas_rt_hist.png",
    )
    output_path = None
    for candidate in candidates:
        try:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(candidate, dpi=100, bbox_inches="tight")
            output_path = candidate
            break
        except OSError:
            continue
    plt.close(figure)
    if output_path is None:
        raise RuntimeError("Could not save rt_hist.png to project data/ or temp dir")
    return str(output_path)
