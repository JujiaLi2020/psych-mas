"""Pure helpers for governed evidence aggregation.

These helpers intentionally avoid Streamlit so the evidence-governance rules can
be unit-tested without importing the full UI application.
"""

from __future__ import annotations

from collections.abc import Iterable


INELIGIBLE_VARIANT_TREATMENTS = {
    "sensitivity_only",
    "display_only",
    "audit_only",
    "no_until_calibrated",
}


def variant_allowed_for_b3(variant_treatment: object) -> bool:
    """Return whether a registry variant treatment may feed B3."""
    treatment = str(variant_treatment or "").strip().lower()
    return treatment not in INELIGIBLE_VARIANT_TREATMENTS


def b3_family_signal_key(
    *,
    function: object,
    aggregation_family: object,
    variant_treatment: object,
    index_column: object = "",
    hit_label: object = "",
) -> tuple[str, str]:
    """Return the de-duplication key and display label for one B3 evidence row.

    Raw correction variants are retained in detail tables, but when a row is a
    configured family-level signal, B3 counts the aggregation family once.
    """
    fn = str(function or "").strip()
    family = str(aggregation_family or "").strip()
    treatment = str(variant_treatment or "").strip()
    fallback_label = str(hit_label or index_column or "").strip()
    if family and treatment == "family_level_signal":
        return f"{fn}|{family}", f"{fn}:{family}"
    key = str(index_column or fallback_label).strip()
    return key, fallback_label


def unique_b3_family_signals(rows: Iterable[dict]) -> list[dict]:
    """Collapse eligible evidence rows to distinct B3 family-level signals."""
    signals: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        if not variant_allowed_for_b3(row.get("variant_treatment")):
            continue
        key, label = b3_family_signal_key(
            function=row.get("function"),
            aggregation_family=row.get("aggregation_family"),
            variant_treatment=row.get("variant_treatment"),
            index_column=row.get("index_column", ""),
            hit_label=row.get("hit_label", ""),
        )
        if not key or key in seen:
            continue
        seen.add(key)
        out = dict(row)
        out["signal_key"] = key
        out["signal_label"] = label
        signals.append(out)
    return signals
