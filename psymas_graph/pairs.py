"""Pair-order helpers for directional copying analyses."""

from __future__ import annotations

import math
from collections.abc import Iterable


def directional_pair_orders(n_persons: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Return both source-copier directions represented by forward/reversed matrices."""
    forward = [(source, copier) for source in range(n_persons) for copier in range(source + 1, n_persons)]
    reverse = [
        (n_persons - 1 - source, n_persons - 1 - copier)
        for source in range(n_persons)
        for copier in range(source + 1, n_persons)
    ]
    return forward, reverse


def canonical_pair(pair: tuple[int, int]) -> tuple[int, int]:
    """Return a stable undirected pair key without assigning behavioral roles."""
    return tuple(sorted((int(pair[0]), int(pair[1]))))


def benjamini_hochberg(p_values: Iterable[float | None]) -> list[float | None]:
    """Return Benjamini-Hochberg adjusted p-values in the original order."""
    values = list(p_values)
    valid = []
    for index, value in enumerate(values):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            valid.append((index, numeric))
    adjusted: list[float | None] = [None] * len(values)
    if not valid:
        return adjusted
    ordered = sorted(valid, key=lambda item: item[1])
    running = 1.0
    total = len(ordered)
    for rank_index in range(total - 1, -1, -1):
        original_index, p_value = ordered[rank_index]
        rank = rank_index + 1
        running = min(running, p_value * total / rank)
        adjusted[original_index] = min(max(running, 0.0), 1.0)
    return adjusted


def pair_agreement(responses, pair: tuple[int, int]) -> float | None:
    """Calculate valid-item exact agreement for one zero-based examinee pair."""
    first, second = canonical_pair(pair)
    try:
        first_row = responses[first]
        second_row = responses[second]
    except (IndexError, TypeError):
        return None
    agreements = []
    for first_value, second_value in zip(first_row, second_row):
        try:
            if math.isnan(float(first_value)) or math.isnan(float(second_value)):
                continue
        except (TypeError, ValueError):
            continue
        agreements.append(first_value == second_value)
    return sum(agreements) / len(agreements) if agreements else None


def simes_p_value(p_values: Iterable[float | None]) -> float | None:
    """Combine dependent method p-values for one pair using Simes' rule."""
    valid = []
    for value in p_values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            valid.append(min(max(numeric, 0.0), 1.0))
    if not valid:
        return None
    ordered = sorted(valid)
    total = len(ordered)
    return min(1.0, min(total * value / rank for rank, value in enumerate(ordered, start=1)))


def calibrated_pair_candidates(
    records: list[dict],
    pairs: list[tuple[int, int]],
    responses,
    *,
    q_alpha: float = 0.05,
    minimum_method_families: int = 2,
    minimum_agreement_rate: float = 0.90,
) -> tuple[set[tuple[int, int]], dict[tuple[int, int], dict]]:
    """Apply per-method BH correction and effect-size gates to pair records."""
    p_value_columns = sorted({
        str(key)
        for record in records
        for key in record
        if str(key).endswith("_pval")
    })
    for column in p_value_columns:
        adjusted = benjamini_hochberg([record.get(column) for record in records])
        q_column = f"{column[:-5]}_qval"
        for record, q_value in zip(records, adjusted):
            if q_value is not None:
                record[q_column] = q_value

    combined_p_values = [
        simes_p_value(record.get(column) for column in p_value_columns)
        for record in records
    ]
    combined_q_values = benjamini_hochberg(combined_p_values)
    for record, p_value, q_value in zip(records, combined_p_values, combined_q_values):
        if p_value is not None:
            record["pair_simes_pval"] = p_value
        if q_value is not None:
            record["pair_BH_qval"] = q_value

    evidence: dict[tuple[int, int], dict] = {}
    for record, pair in zip(records, pairs):
        canonical = canonical_pair(pair)
        entry = evidence.setdefault(
            canonical,
            {"method_families": set(), "q_values": {}, "pair_q_value": 1.0},
        )
        pair_q_value = record.get("pair_BH_qval")
        try:
            entry["pair_q_value"] = min(entry["pair_q_value"], float(pair_q_value))
        except (TypeError, ValueError):
            pass
        for key, value in record.items():
            key_text = str(key)
            if not key_text.endswith("_pval") or key_text == "pair_simes_pval":
                continue
            try:
                p_value = float(value)
            except (TypeError, ValueError):
                continue
            method = key_text[:-5]
            q_value = record.get(f"{method}_qval")
            if q_value is not None:
                entry["q_values"][method] = min(float(q_value), entry["q_values"].get(method, 1.0))
            if p_value <= 0.05:
                entry["method_families"].add(method.split("_", 1)[0])

    candidates: set[tuple[int, int]] = set()
    for pair, entry in evidence.items():
        agreement = pair_agreement(responses, pair)
        entry["agreement_rate"] = agreement
        entry["method_families"] = sorted(entry["method_families"])
        entry["candidate"] = bool(
            entry["pair_q_value"] <= q_alpha
            and len(entry["method_families"]) >= minimum_method_families
            and agreement is not None
            and agreement >= minimum_agreement_rate
        )
        if entry["candidate"]:
            candidates.add(pair)
    return candidates, evidence


def confirmed_similarity_pairs(ac_pairs, as_pairs) -> set[tuple[int, int]]:
    """Return canonical pairs independently supported by AC and AS."""
    ac = {canonical_pair(tuple(pair)) for pair in ac_pairs or []}
    answer_similarity = {canonical_pair(tuple(pair)) for pair in as_pairs or []}
    return ac & answer_similarity
