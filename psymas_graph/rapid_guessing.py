"""Person-level calibration helpers for rapid-guessing output."""

from __future__ import annotations

import math
from collections.abc import Sequence


def explicit_package_flags(values):
    """Count only package TRUE/1; NA, NaN, None and R NA sentinels are not hits."""
    import numpy as np
    import pandas as pd

    array = np.asarray(values)
    missing = pd.isna(array)
    # R logical NA can arrive as the signed 32-bit minimum, rather than NaN.
    if array.dtype.kind in "iuf":
        missing = missing | (array == -2147483648)
    positive = np.zeros(array.shape, dtype=bool)
    positive[~missing] = array[~missing] == 1
    return positive, missing


def summarize_package_flags(values):
    """Return person hits and a separate missing/unknown audit state.

    Axis zero indexes persons; reduce all item/threshold axes. A person with
    no explicit hit and some missing flags is unknown, not a proven negative.
    """
    import numpy as np

    positive, missing = explicit_package_flags(values)
    if positive.ndim < 1:
        raise ValueError("Package flags must have a person axis")
    axes = tuple(range(1, positive.ndim))
    hits = np.any(positive, axis=axes) if axes else positive
    missing_counts = np.sum(missing, axis=axes) if axes else missing.astype(int)
    valid_counts = np.sum(~missing, axis=axes) if axes else (~missing).astype(int)
    states = ["positive" if hit else "unknown" if absent or valid == 0 else "negative"
              for hit, absent, valid in zip(hits, missing_counts, valid_counts)]
    return {"flagged": np.flatnonzero(hits).tolist(),
            "missing_counts": missing_counts.astype(int).tolist(),
            "valid_counts": valid_counts.astype(int).tolist(),
            "person_state": states,
            "policy": "only_explicit_TRUE_or_1_counts; missing_is_unknown"}


def calibrated_rte_flags(
    rte_values: Sequence[object],
    valid_proportions: Sequence[object],
    *,
    rte_max: float = 0.90,
    min_valid_proportion: float = 0.90,
) -> list[int]:
    """Return zero-based persons meeting the documented RTE screening rule."""
    flagged: list[int] = []
    for index, (raw_rte, raw_valid) in enumerate(zip(rte_values, valid_proportions)):
        try:
            rte = float(raw_rte)
            valid = float(raw_valid)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(rte) or not math.isfinite(valid):
            continue
        if valid >= min_valid_proportion and rte <= rte_max:
            flagged.append(index)
    return flagged
