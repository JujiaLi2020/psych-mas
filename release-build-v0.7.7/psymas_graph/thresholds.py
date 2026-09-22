"""Threshold access and normalization shared by forensic detector nodes."""

import numpy as np

from .state import State


def threshold_rules(state: State) -> dict:
    config = state.get("threshold_config") or {}
    rules = config.get("rules") if isinstance(config, dict) else {}
    return rules if isinstance(rules, dict) else {}


def threshold_alpha(state: State, function_name: str, default: float = 0.05) -> float:
    config = state.get("threshold_config") or {}
    rules = threshold_rules(state)
    function_rules = rules.get(function_name, {})
    function_rules = function_rules if isinstance(function_rules, dict) else {}
    defaults = config.get("defaults") or {} if isinstance(config, dict) else {}
    value = function_rules.get("alpha", defaults.get("alpha", default))
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = default
    return min(max(value, 0.000001), 0.999999)


def pairwise_alpha(state: State, function_name: str, n_pairs: int, default: float = 0.05) -> float:
    """Dampen pairwise alpha when the number of comparisons is large."""
    base = threshold_alpha(state, function_name, default)
    if n_pairs <= 2000:
        return base
    adjusted = min(base, 0.05 / (n_pairs ** 0.5))
    return min(max(adjusted, 1e-6), 0.999999)


def orient_pair_flag_matrix(flag_array, n_pairs: int, n_methods: int):
    """Return a pairs-by-methods boolean matrix from common R orientations."""
    if flag_array is None:
        return None
    array = np.asarray(flag_array, dtype=bool)
    if array.ndim < 2 or n_pairs <= 0 or n_methods <= 0:
        return array
    if array.shape[:2] == (n_pairs, n_methods):
        return array
    if array.shape[:2] == (n_methods, n_pairs):
        return array.T
    if array.shape[1] == n_pairs and array.shape[0] != n_pairs:
        return array.T
    return array


def threshold_block(state: State, function_name: str, index_name: str) -> dict:
    block = (threshold_rules(state).get(function_name) or {}).get(index_name) or {}
    return block if isinstance(block, dict) else {}


def threshold_float(
    state: State,
    function_name: str,
    index_name: str,
    key: str,
    default: float,
) -> float:
    value = threshold_block(state, function_name, index_name).get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def threshold_enabled(
    state: State,
    function_name: str,
    index_name: str,
    default: bool = True,
) -> bool:
    return bool(threshold_block(state, function_name, index_name).get("enabled", default))


def r_number(value: float) -> str:
    """Format a Python number for safe interpolation into generated R code."""
    return f"{float(value):.12g}"
