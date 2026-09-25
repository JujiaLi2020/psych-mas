"""Pure helpers for governed evidence aggregation.

These helpers intentionally avoid Streamlit so the evidence-governance rules can
be unit-tested without importing the full UI application.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
import json

import yaml


INELIGIBLE_VARIANT_TREATMENTS = {
    "sensitivity_only",
    "display_only",
    "audit_only",
    "no_until_calibrated",
}


@lru_cache(maxsize=4)
def load_b3_rulebook(path: str = "config/b3_index_mapping.yaml") -> dict:
    """Load the canonical B3 rulebook used to resolve a case evidence packet."""
    rulebook_path = Path(path)
    if not rulebook_path.exists():
        fallback = Path("manuscript") / "b3_index_mapping.yaml"
        if fallback.exists():
            rulebook_path = fallback
    if not rulebook_path.exists():
        return {}
    try:
        with rulebook_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        return loaded if isinstance(loaded, dict) else {}
    except (OSError, yaml.YAMLError):
        return {}


def build_case_rulebook_resolution(
    rulebook: dict,
    *,
    active_domains: Iterable[str],
    domain_rows: Iterable[dict],
    trace_rows: Iterable[dict],
) -> str:
    """Return a compact, case-specific application of the YAML rulebook.

    The full YAML remains configuration. The LLM receives only the rules that
    apply to this examinee, together with the already-resolved B3 and priority
    outputs. This prevents the model from treating general policy text as a
    second decision engine.
    """
    active = {str(domain or "").strip().upper() for domain in active_domains if str(domain or "").strip()}
    domain_policy = rulebook.get("domain_policy", {}) if isinstance(rulebook, dict) else {}
    domain_rows = [dict(row) for row in domain_rows if isinstance(row, dict)]
    trace_rows = [dict(row) for row in trace_rows if isinstance(row, dict)]

    resolved_domains = []
    rule_ids = set()
    for row in domain_rows:
        domain = str(row.get("Domain", "") or "").strip().upper()
        if domain not in active:
            continue
        rule_id = str(row.get("Strength_Rule", "") or "").strip()
        if rule_id:
            rule_ids.add(rule_id)
        policy = domain_policy.get(domain, {}) if isinstance(domain_policy, dict) else {}
        resolved_domains.append(
            {
                "domain": domain,
                "label": policy.get("label", row.get("Domain_Label", domain)),
                "priority_role": policy.get("priority_role", "unspecified"),
                "strength": row.get("Strength", ""),
                "b3_rule": rule_id,
                "evidence_pattern": row.get("Evidence_Pattern", ""),
            }
        )

    functions = {
        str(row.get("Function", "") or "").strip()
        for row in trace_rows
        if str(row.get("Function", "") or "").strip()
    }
    family_policy = rulebook.get("family_policy", {}) if isinstance(rulebook, dict) else {}
    applicable_family_policy = {
        function: family_policy[function]
        for function in sorted(functions)
        if isinstance(family_policy, dict) and function in family_policy
    }

    families = []
    seen_families = set()
    for row in trace_rows:
        family = str(
            row.get("Aggregation_Family")
            or row.get("aggregation_family")
            or row.get("Index")
            or ""
        ).strip()
        if not family or family in seen_families:
            continue
        seen_families.add(family)
        families.append(
            {
                "family": family,
                "function": row.get("Function") or row.get("function", ""),
                "domain": row.get("Domain") or row.get("domain", ""),
                "role": row.get("Role") or row.get("role", ""),
                "evidence_use": row.get("Evidence_Use") or row.get("evidence_use", ""),
            }
        )

    resolution = {
        "source": "config/b3_index_mapping.yaml",
        "rulebook_version": rulebook.get("version", "") if isinstance(rulebook, dict) else "",
        "active_domains": resolved_domains,
        "eligible_family_signals": families,
        "b3_rules_applied": {
            rule_id: (rulebook.get("b3_domain_strength_rules", {}).get(rule_id, {}) if isinstance(rulebook, dict) else {})
            for rule_id in sorted(rule_ids)
        },
        "family_policies_applied": applicable_family_policy,
        "priority_policy": rulebook.get("priority_policy", {}) if isinstance(rulebook, dict) else {},
        "reporting_boundary": (rulebook.get("reporting_policy", {}) if isinstance(rulebook, dict) else {}),
    }
    return json.dumps(resolution, ensure_ascii=False, indent=2, sort_keys=True, default=str)


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
