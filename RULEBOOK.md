# PsyMAS Evidence Rulebook

This document explains how PsyMAS converts detector output into review evidence. It is the human-readable companion to the executable configuration in `config/`. Statistical flags are review triggers; they are not findings of intent, copying, tampering, or misconduct.

## Sources of truth

The executable files remain authoritative:

1. `config/rulebook_index.csv` — index registry, domain, role, family, and evidence use.
2. `config/b3_index_mapping.yaml` — family aggregation, domain strength, and priority policy.
3. `config/default_thresholds.yaml` — runtime defaults.
4. `config/index_thresholds.yaml` — threshold definitions, calibration status, and sources.

This document must be updated whenever those files change. If prose and configuration disagree, the configuration controls runtime behavior and the disagreement is a documentation defect.

## Evidence roles

| Role | Enters B3 | Changes priority | Intended use |
|---|---:|---:|---|
| Evidence Flag | Yes | According to domain policy | Governed case evidence |
| Calibration Required | No | No | Visible until an approved cutoff is supplied |
| Support Only | No | No | Context and pair-level review |
| Display Only | No | No | Audit, visualization, or localization |

Raw statistics do not enter B3 merely because they are numerically extreme. A valid package flag or approved calibrated rule is required. Correction variants are retained for audit but count once at the configured family level.

## Domain policy

| Domain | Role | Current policy |
|---|---|---|
| RT | Primary scenario | Only the calibrated NT person-level signal is eligible |
| PK | Primary scenario | Eligible package-returned preknowledge families may count |
| TP | Primary scenario | Eligible answer-change families count once per family |
| MF | Supporting only | May strengthen a concern but cannot create a scenario alone |
| SIM | Context only | Pair candidates do not enter B3 or priority |
| CP | Localization only | Locates item ranges; does not enter priority |

## Rapid Guessing calibration

The former implementation treated any item-level rapid-response flag as a person-level flag and replaced missing response times with `0.01` seconds. Both behaviors inflated false positives.

The current person-level rule is:

- method: `NT_2` response-time effort;
- flag when `RTE <= 0.90`;
- require at least `90%` valid response-time observations;
- impute remaining eligible missing values with item medians for package execution, never with an artificially rapid time;
- retain CT and CUMP outputs for display and audit only.

On the bundled 500-examinee validation data, retrospective application of this rule detected 21 of 25 embedded rapid-guessing cases (84%) with 0 of 390 normal examinees flagged. This is a screening starting point, not universal calibration. Operational programs should re-estimate performance on representative data and document any threshold change.

This calibration also changes review-priority composition. The archived 2026-09-22 release snapshot used uncalibrated RT evidence and produced 115 moderate RT profiles and 26 `Critical / Expedited` cases. The current snapshot produces 21 moderate RT profiles and 3 `Critical / Expedited` cases. The `B6-04b` priority rule did not change; fewer cases now satisfy its cross-scenario requirement because RT evidence is no longer broadly activated.

## Similarity and copying pairs

AC and AS outputs remain `SIM` context because pairwise searches have a severe base-rate and multiple-comparison problem. They create a candidate queue for human review, not a copying determination.

Current safeguards:

- AC evaluates both source-copier directions; examinee ID order does not assign behavioral roles.
- A detector review candidate requires its package pair flag and exact-response agreement of at least `0.875`.
- A stricter audit tier combines method p-values within pair using Simes, applies Benjamini-Hochberg adjustment across pairs, and requires at least two method families.
- A confirmed pair must independently satisfy both AC and AS candidate rules (`AC AND AS`).
- Candidate pairs remain available for audit; only confirmed-pair participants appear as SIM reviewer context.
- Published confirmed pairs are canonical undirected pairs unless external evidence supports direction.
- Source/copier roles require independent context such as seating, session timing, access, or proctoring records.
- Pair flags do not enter B3, review priority, or misconduct conclusions.
- A production calibration should report pair-level recall, precision, and false-discovery behavior on an independent validation set.

The bundled snapshot uses the revised fixed-seed 80% copying implant. Under the current AC/AS agreement gate and cross-detector confirmation rule, it recovers 9 of 10 planted pairs with one false-positive pair. SIM remains context-only until multi-seed and operational validation demonstrate acceptable pair recall, precision, and false-discovery control.

Recommended operational confirmation requires at least two independent layers, for example response similarity plus timing similarity, followed by external contextual review. Narrowing the candidate set using legitimate administration metadata is preferable to simply relaxing p-value thresholds.

## B3 domain strength

PsyMAS counts distinct eligible families, not raw columns or correction variants:

| Rule | Condition | Strength |
|---|---|---|
| B3-00 | Required data unavailable | unavailable |
| B3-01 | No eligible family signal | none |
| B3-02 | One supporting family, no primary | weak |
| B3-03 | At least one primary family, no supporting | moderate |
| B3-04 | At least two supporting families, no primary | moderate |
| B3-05 | Primary and supporting families | strong |

Priority is derived from eligible primary domains (`RT`, `PK`, `TP`). `MF` is strengthening-only; `SIM` and `CP` do not contribute.

## AI reporting boundary

The LLM receives a case-specific packet after deterministic governance. It cannot compute flags, alter thresholds, create evidence, infer intent, determine misconduct, or recommend sanctions.

Every generated evidence claim must carry a machine-checkable citation to an active domain and eligible family. Priority statements must cite the exact governed priority. Before publication, PsyMAS verifies:

- active domain and governed strength;
- named family against the case trace;
- review priority;
- exposed-item counts when supplied;
- prohibited behavioral or misconduct conclusions.

Valid internal citations are removed before display. If any check fails, the model draft is discarded and replaced with a deterministic evidence summary. The same boundary applies to UI display and PDF export.

## Change control

Any rule or threshold change requires all of the following:

1. update the applicable files in `config/`;
2. update this document and the README summary;
3. add or update focused tests;
4. rerun `reproducibility/demo_validation/analyze_demo_validation.py`;
5. review detection, false-positive, traceability, and AI-report audit outputs;
6. regenerate the SQLite database and bundled evaluated Demo snapshot;
7. rebuild and recreate the Docker services.

Snapshots contain evaluated results, not just raw input. A snapshot generated under an older rulebook must not be presented as if it were computed under the current configuration.

## Validation artifacts

- Main report: `reproducibility/demo_validation/outputs/demo_validation_report.md`
- Detection metrics: `reproducibility/demo_validation/outputs/detection_metrics.csv`
- Pair metrics: `reproducibility/demo_validation/outputs/copying_pair_metrics.csv`
- Traceability metrics: `reproducibility/demo_validation/outputs/traceability_metrics.csv`
- AI-report audit: `reproducibility/demo_validation/outputs/ai_report_audit.csv`
- Exact rerun summary: `reproducibility/demo_validation/outputs/rerun_validation.json`
- Per-run hashes: `reproducibility/demo_validation/outputs/rerun_hashes.csv`
- Reproduction script: `reproducibility/demo_validation/analyze_demo_validation.py`
- R reproduction script: `reproducibility/demo_validation/analyze_demo_validation.R`
- Rerun validator: `reproducibility/demo_validation/validate_rerun_consistency.py`

Run the R analysis from the repository root with:

```bash
python reproducibility/demo_validation/analyze_demo_validation.py
docker run --rm -v "${PWD}:/workspace" -w /workspace psych-mas:latest \
  python reproducibility/demo_validation/validate_rerun_consistency.py --repetitions 1
Rscript reproducibility/demo_validation/analyze_demo_validation.R
```

The R outputs are written to `reproducibility/demo_validation/outputs_r/`. If `DBI` and `RSQLite` are unavailable, the script automatically uses Python's standard SQLite library only to export snapshot tables; all validation metrics are computed in R.
