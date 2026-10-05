"""Evidence-link validation for reviewer-facing AI report text."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
from psymas_ui.report_display import clean_report_display


ACTIVE_STRENGTHS = {"weak", "moderate", "strong"}
DOMAIN_PATTERNS = {
    "MF": r"\b(?:MF|misfit|response[- ]pattern fit|person[- ]fit)\b",
    "RT": r"\b(?:RT|response[- ]time|rapid respond|rapid answer|low effort|speededness)\b",
    "PK": r"\b(?:PK|preknowledge|prior knowledge|exposed[- ]item|compromised[- ]item)\b",
    "TP": r"\b(?:TP|tampering|answer[- ]change|changed responses?)\b",
    "SIM": r"\b(?:SIM|similarity|copying|shared response pattern)\b",
    "CP": r"\b(?:CP|change[- ]?point|change[- ]?pattern|localization cue)\b",
}
EVIDENCE_ASSERTION = re.compile(
    r"\b(?:evidence|signal|flag(?:ged)?|indicator|indicates?|shows?|supports?|"
    r"consistent with|unusual|atypical|anomal|irregular|weak|moderate|strong)\b",
    flags=re.IGNORECASE,
)
EVIDENCE_CITATION = re.compile(r"\[E:([A-Z]{2,3})/([^\]\r\n]+)\]", flags=re.IGNORECASE)
PRIORITY_CITATION = re.compile(r"\[P:([^\]\r\n]+)\]", flags=re.IGNORECASE)
TRAILING_SENTENCE_CITATIONS = re.compile(
    r"([.!?])([ \t]+)((?:\[(?:E:[A-Z]{2,3}/[^\]\r\n]+|P:[^\]\r\n]+)\][ \t]*)+)",
    flags=re.IGNORECASE,
)
PK_TIMING_LIMITATION = re.compile(
    r"\b(?:the\s+)?(?:response[- ]time|timing)\s+comparison\s+"
    r"(?:alone|by itself)\s+(?:does not|cannot|can not)\s+"
    r"(?:establish|demonstrate|prove|confirm|support)\s+"
    r"(?:rapid responding|rapid responses?|rapid[- ]guessing|low effort|speededness)\b",
    re.IGNORECASE,
)
NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    try:
        return float(value) != 0
    except (TypeError, ValueError):
        return _clean(value).lower() in {"true", "yes", "flagged"}


def _normal_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", _clean(value).lower()).strip("_")


def _active_domains(case_domains: pd.DataFrame) -> dict[str, str]:
    if not isinstance(case_domains, pd.DataFrame) or case_domains.empty or "Domain" not in case_domains.columns:
        return {}
    return {
        _clean(row.get("Domain")).upper(): _clean(row.get("Strength")).lower()
        for _, row in case_domains.iterrows()
    }


def _index_registry(
    case_trace: pd.DataFrame,
    active_domains: dict[str, str],
) -> tuple[dict[str, tuple[str, str]], dict[str, set[str]]]:
    known: dict[str, tuple[str, str]] = {}
    allowed: dict[str, set[str]] = {}
    if not isinstance(case_trace, pd.DataFrame) or case_trace.empty:
        return known, allowed
    for _, row in case_trace.iterrows():
        domain = _clean(row.get("Domain")).upper()
        family = _clean(row.get("Aggregation_Family")) or _clean(row.get("Index"))
        family_key = _normal_token(family)
        eligible = _truthy(row.get("Evidence_Eligible", True))
        flagged = _truthy(row.get("Flag", row.get("Value", False)))
        if domain in active_domains and active_domains.get(domain) in ACTIVE_STRENGTHS and eligible and flagged:
            allowed.setdefault(domain, set()).add(family_key)
        for column in ("Aggregation_Family", "Index", "Index_Column", "Flag_Column"):
            token = _clean(row.get(column))
            if len(token) < 2 or not re.search(r"[A-Za-z]", token):
                continue
            known[_normal_token(token)] = (domain, family_key)
    return known, allowed


def validate_evidence_links(
    text: str,
    case_row: pd.Series,
    case_domains: pd.DataFrame,
    case_trace: pd.DataFrame,
    context_facts: dict[str, Any] | None = None,
) -> tuple[str, list[str]]:
    """Validate machine-readable citations and case-specific evidence claims.

    Accepted citations are ``[E:DOMAIN/FAMILY]`` and ``[P:PRIORITY]``. They are
    removed only after every cited claim has passed the case-packet allowlist.
    """
    report = _clean(text)
    strengths = _active_domains(case_domains)
    active = {domain for domain, strength in strengths.items() if strength in ACTIVE_STRENGTHS}
    known_indices, allowed_families = _index_registry(case_trace, strengths)
    context_facts = context_facts or {}
    violations: list[str] = []

    evidence_citations = [
        (match.group(0), match.group(1).upper(), _normal_token(match.group(2)))
        for match in EVIDENCE_CITATION.finditer(report)
    ]
    for _, domain, family in evidence_citations:
        if domain not in active:
            violations.append(f"citation points to inactive {domain} evidence")
        elif family != "domain" and family not in allowed_families.get(domain, set()):
            violations.append(f"citation points to an unlinked {domain} family: {family}")

    expected_priority = _clean(case_row.get("Review_Priority"))
    priority_citations = [match.group(1).strip() for match in PRIORITY_CITATION.finditer(report)]
    for priority in priority_citations:
        if priority.casefold() != expected_priority.casefold():
            violations.append(f"priority citation does not match {expected_priority or 'unassigned'}")

    # Models commonly put a citation after the sentence's punctuation. Attach
    # same-line tags to that sentence before splitting; do not move citations
    # across line breaks or infer absent citations.
    sentence_report = TRAILING_SENTENCE_CITATIONS.sub(
        lambda match: " " + match.group(3).strip() + match.group(1) + " ", report
    )
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", sentence_report) if part.strip()]
    for sentence in sentences:
        sentence_without_tags = EVIDENCE_CITATION.sub("", PRIORITY_CITATION.sub("", sentence))
        sentence_citations = {
            (match.group(1).upper(), _normal_token(match.group(2)))
            for match in EVIDENCE_CITATION.finditer(sentence)
        }
        sentence_priority_citations = [
            match.group(1).strip() for match in PRIORITY_CITATION.finditer(sentence)
        ]
        for domain, pattern in DOMAIN_PATTERNS.items():
            domain_text = sentence_without_tags
            # Remove only this narrow, explicitly negated PK comparison clause.
            # Any other RT claim in the same sentence remains subject to audit.
            if domain == "RT" and ("PK", "domain") in sentence_citations and re.search(
                r"\bexposed[- ]item\b", sentence_without_tags, re.I
            ):
                domain_text = PK_TIMING_LIMITATION.sub("", domain_text)
            if not re.search(pattern, domain_text, flags=re.IGNORECASE):
                continue
            # A numeric timing comparison explicitly identified as descriptive
            # PK context is not an RT detector claim. Keep actual RT assertions
            # (including a claim appended to the same sentence) blocked.
            descriptive_pk_timing = (
                domain == "RT"
                and ("PK", "domain") in sentence_citations
                and re.search(r"exposed[- ]item response time is\s+\d", sentence_without_tags, re.I)
                and re.search(r"\bversus\b", sentence_without_tags, re.I)
                and re.search(r"descriptive checks?, not independent evidence", sentence_without_tags, re.I)
                and not re.search(
                    r"\b(?:RT|response[- ]time|rapid responding|low effort)\s+"
                    r"(?:evidence|signal|flag|is (?:strong|moderate|weak|unusual|atypical))",
                    sentence_without_tags, re.I,
                )
            )
            if domain not in active and EVIDENCE_ASSERTION.search(sentence_without_tags) and not descriptive_pk_timing:
                violations.append(f"inactive {domain} described as evidence")
            if domain in active and EVIDENCE_ASSERTION.search(domain_text) and not descriptive_pk_timing:
                if not any(cited_domain == domain for cited_domain, _ in sentence_citations):
                    violations.append(f"uncited {domain} evidence claim")
            strength_claims = re.findall(
                rf"\b(weak|moderate|strong)\s+(?:governed\s+)?{pattern}"
                rf"|{pattern}(?:\s+(?:evidence|signal|strength))?"
                rf"\s*(?:(?:is|are|was|remains)\s+|[:=]\s*)?\b(weak|moderate|strong)\b",
                sentence_without_tags,
                flags=re.IGNORECASE,
            )
            stated_strengths = {
                value.lower()
                for claim in strength_claims
                for value in claim
                if value
            }
            if domain in active and any(value != strengths.get(domain) for value in stated_strengths):
                violations.append(f"{domain} strength does not match governed evidence")

        for token, (domain, family) in known_indices.items():
            # Match whole names in the original prose: bare L_ST must not also
            # match the distinct prefixed names pk_L_ST or pm_L_ST.
            token_pattern = re.escape(token).replace("_", "[-_ ]")
            if not token or not re.search(rf"(?<![a-z0-9_]){token_pattern}(?![a-z0-9_])", sentence_without_tags, re.I):
                continue
            if domain not in active or family not in allowed_families.get(domain, set()):
                violations.append(f"unlinked index or family named: {token}")
            elif (domain, family) not in sentence_citations:
                violations.append(f"uncited index family: {token}")

        priority_claim = re.search(
            r"\b(?:critical|expedited|urgent|high|medium|low)(?:[- ]priority|\s+queue|\s+review|\s+case)\b",
            sentence_without_tags,
            flags=re.IGNORECASE,
        )
        if priority_claim and not sentence_priority_citations:
            violations.append("uncited priority claim")
        if priority_claim:
            term = priority_claim.group(0).lower()
            expected_lower = expected_priority.lower()
            if any(word in term for word in ("critical", "expedited", "urgent")) and expected_priority != "Critical / Expedited":
                violations.append("priority overstatement")
            elif "high" in term and not expected_lower.startswith("high"):
                violations.append("priority mismatch")
            elif "medium" in term and expected_priority != "Medium":
                violations.append("priority mismatch")
            elif "low" in term and expected_priority != "Low":
                violations.append("priority mismatch")

        count_match = re.search(
            r"\b(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b"
            r"\s+(?:exposed|compromised|exposed/compromised)[- ]items?\b",
            sentence_without_tags,
            flags=re.IGNORECASE,
        )
        expected_count = context_facts.get("exposed_item_count")
        if count_match and expected_count is not None:
            token = count_match.group(1).lower()
            reported_count = int(token) if token.isdigit() else NUMBER_WORDS[token]
            if reported_count != int(expected_count):
                violations.append("incorrect exposed-item count")

    clean_report = EVIDENCE_CITATION.sub("", PRIORITY_CITATION.sub("", report))
    return clean_report_display(clean_report), sorted(set(violations))


def case_citation_contract(case_row: pd.Series, case_domains: pd.DataFrame, case_trace: pd.DataFrame) -> str:
    """Give the model exact case-specific tags accepted by this validator."""
    strengths = _active_domains(case_domains)
    _, allowed = _index_registry(case_trace, strengths)
    tags = []
    for domain, strength in sorted(strengths.items()):
        if strength not in ACTIVE_STRENGTHS:
            continue
        tags.append(f"[E:{domain}/domain]")
        tags.extend(f"[E:{domain}/{family}]" for family in sorted(allowed.get(domain, set())))
    priority = _clean(case_row.get("Review_Priority"))
    return (
        f"Exact review priority: {priority or 'unassigned'}.\n"
        f"Only allowed priority citation: [P:{priority}].\n"
        "Allowed evidence citations (copy exactly): " + (", ".join(tags) or "none") + ".\n"
        "Attach each citation to its own claim before the sentence-ending punctuation. "
        "A sentence combining domains needs a citation for each domain. "
        "Domain strength is not review priority. Never copy placeholder tags.\n"
    )
