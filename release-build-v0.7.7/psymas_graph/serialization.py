"""Serialization helpers for graph results and API payloads."""

import numpy as np
import pandas as pd


def records(obj) -> list[dict]:
    """Convert pandas, NumPy, mapping, and sequence outputs to records."""
    if obj is None:
        return []
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict(orient="records")
        except TypeError:
            pass
    try:
        if isinstance(obj, (np.recarray, np.ndarray)) and getattr(obj, "dtype", None) is not None:
            if obj.dtype.names:
                return pd.DataFrame(obj).to_dict(orient="records")
    except Exception:
        pass
    if isinstance(obj, list):
        if all(isinstance(value, dict) for value in obj):
            return obj
        try:
            return pd.DataFrame(obj).to_dict(orient="records")
        except Exception:
            return [{"value": value} for value in obj]
    if isinstance(obj, dict):
        return [obj]
    return [{"value": obj}]
