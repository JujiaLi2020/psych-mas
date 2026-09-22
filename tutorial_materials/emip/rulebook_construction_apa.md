# Construction of the PsyMAS Evidence Rulebook

## Purpose and design principles

The PsyMAS rulebook defines how deterministic forensic outputs become reviewable evidence. It is not a misconduct classifier and it does not assign legal or disciplinary conclusions. Its purpose is to preserve the link between an observed response pattern, the method that produced it, the evidence domain to which it belongs, and the human-review action that follows.

The rulebook follows four principles. First, an evidence domain is defined by the substantive behavior being examined, not by the data type used by an index. Thus, a timing-based person-fit statistic remains misfit evidence when its inferential target is parametric person fit; it is not automatically rapid-guessing evidence. In contrast, rapid-guessing methods are assigned to the response-time domain because they directly operationalize unusually rapid or low-effort responding (Wise, 2017). Second, package-returned flags are preserved as package outputs. PsyMAS does not replace them with a second p-value calculation. Third, correction variants of one statistic are not treated as independent evidence. They are retained for sensitivity analysis and audit, while the rulebook counts at most one family-level signal. Fourth, missing or uncalibrated evidence is reported as unavailable or calibration-required rather than treated as a negative finding.

These principles are consistent with the measurement literature, but the rulebook itself is an operational specification. Published studies motivate the interpretation of rapid responding, answer-change irregularity, preknowledge statistics, and person-fit statistics; they do not, by themselves, establish a universal PsyMAS threshold, domain-strength scale, or review-priority policy. Those latter components are declared explicitly in the versioned configuration and should be locally calibrated before operational use.

## Rulebook construction

### 1. Index registry

The registry is the authoritative mapping from package output to evidence use. Each row records the source function, method or index family, data basis, evidence domain, evidence role, required inputs, package output type, flag source, threshold source, aggregation family, correction variant, and treatment in synthesis. The registry separates four evidence-use states:

* **Evidence Flag.** A package-returned or already approved rule flag may enter B3 after family aggregation.
* **Calibration Required.** The output remains visible, but cannot enter B3 until an external cutoff or calibration record is supplied.
* **Support Only.** The output can support interpretation or pair-level/contextual review but does not contribute to current B3 or priority.
* **Display Only.** The output is retained for localization, visualization, or audit and is not evidence strength by itself.

The distinction is important because a statistic can be scientifically informative without being ready for deterministic evidence synthesis. For example, change-point locations can help an analyst identify where behavior may have shifted, but a location estimate is not itself a validated person-level flag. Similarly, similarity and copying outputs require a prespecified pairwise decision rule and calibration before they can be used as priority-generating evidence.

### 2. Function-to-domain mapping

The mapping is based on the substantive target of each function:

| Function family | Primary domain | Role in PsyMAS | Operational interpretation |
|---|---|---|---|
| `detect_rg` CT, CUMP, NT | Response-Time (RT) | Primary scenario evidence | Unusually rapid or low-effort responding under the selected package threshold procedure |
| `detect_pk` | Preknowledge (PK) | Primary scenario evidence | Response or timing patterns associated with exposed or compromised items |
| `detect_tt` | Tampering / Answer Change (TP) | Primary scenario evidence | Unusual initial-to-final response or distractor-change patterns |
| `detect_ac`, `detect_as` | Similarity (SIM) | Context-only in the current rulebook | Pairwise similarity or copying evidence retained for interpretation and future calibration |
| `detect_pm` | Misfit (MF) | Supporting evidence | Parametric person-fit evidence, including score, response, time, and joint score-time statistics |
| `detect_nm` | Misfit (MF) | Supporting, calibration-required evidence | Nonparametric person-fit statistics retained until an approved cutoff is available |
| `detect_cp` | Change-Pattern (CP) | Localization/display-only evidence | Estimated locations of possible score or timing shifts; no current priority contribution |

This arrangement prevents data-basis overlap from becoming domain duplication. For example, `detect_pm` methods using response times remain MF because they evaluate person-fit under a model. They do not become a second RT scenario merely because time is an input. The `data_basis` field records whether the method uses scores, responses, time, or joint information without changing its substantive domain.

### 3. Family aggregation and correction variants

The raw output layer preserves every package column. The evidence layer then applies an aggregation key. A family is defined by the underlying statistic and substantive method, not by a correction suffix. For example, `EDI_SD_NO`, `EDI_SD_CO`, and `EDI_SD_TS` are correction variants of the same `tt_EDI_SD` family. They cannot generate three independent TP signals.

For each family, the registry specifies one of three treatments:

1. **Family-level signal.** The prespecified eligible variant represents the family in B3.
2. **Sensitivity-only.** The variant remains visible for robustness checks and audit but does not add another signal.
3. **Display-only or no-until-calibrated.** The output is available for interpretation but is excluded from B3 until the stated requirement is met.

In the current rulebook, `EDI_SD_CO` is the designated eligible variant for the `tt_EDI_SD` family, while `EDI_SD_NO` and `EDI_SD_TS` are sensitivity-only. `GBT_SD` is a separate eligible family. Therefore, TP strength can reflect two distinct eligible families (`tt_EDI_SD` and `tt_GBT_SD`), but it cannot be inflated by counting several EDI correction variants. The same family-level principle applies to the correction variants produced by `detect_pm`.

### 4. Evidence Input Table

After registry mapping, PsyMAS writes one traceable evidence row for each examinee-index or pair-index unit. The row includes the identifier, function, method, raw index, aggregation family, correction variant, domain, role, evidence-use state, flag status, threshold source, rule identifier, and source columns. A family-level key is used for B3. Raw index rows remain available for audit and detailed case review.

An output with no flag is not counted. A missing input is not coded as normal. A package statistic without a package flag is not converted into a flag by PsyMAS. An external-threshold output is counted only after the relevant threshold has been explicitly approved and recorded. This policy keeps statistical computation separate from evidence governance.

### 5. Domain evidence strength

For each examinee and domain, PsyMAS first removes ineligible rows, collapses correction variants, and counts distinct eligible family signals. It then applies the following deterministic B3 rules:

| Rule | Condition | Strength |
|---|---|---|
| B3-00 | Required data are unavailable | Unavailable |
| B3-01 | No eligible primary or supporting family signal is flagged | None |
| B3-02 | One eligible supporting family signal and no primary signal | Weak |
| B3-03 | One or more eligible primary family signals and no supporting signal | Moderate |
| B3-04 | At least two eligible supporting family signals and no primary signal | Moderate |
| B3-05 | At least one eligible primary and one eligible supporting family signal | Strong |

The scale describes governed evidence status within a domain. It does not describe the probability that misconduct occurred. Supporting MF evidence may strengthen interpretation of an active primary scenario, but MF alone does not create a scenario. SIM and CP are visible context domains under the current configuration and do not contribute to priority.

### 6. Case synthesis and review priority

Case synthesis is performed after domain profiles are complete. The current priority-generating scenario domains are RT, PK, and TP. MF is supporting-only; SIM and CP are context-only. The case profile therefore describes the pattern of eligible primary domains rather than the number of raw flags. A single active primary domain produces a single-scenario signal. Multiple active primary domains produce a cross-scenario pattern. Supporting MF evidence may increase the strength of an already active scenario but cannot create a scenario by itself.

Review priority is a workflow recommendation based on the governed domain profile, evidence completeness, and the program's declared configuration. It is not a ranking of examinee culpability. The system may therefore recommend expedited review for convergent eligible evidence while still requiring a human analyst to examine context, item-level behavior, and alternative explanations.

### 7. Reporting, LLM constraints, and audit

The report layer receives governed evidence records and selected raw-data summaries. Raw data are used to explain the practical meaning of a governed signal, for example, whether flagged response-time indices coincide with item-level timing below the item mean, or whether a preknowledge signal is concentrated on exposed items. Raw data do not bypass the registry or create an ungoverned flag.

The LLM may summarize the evidence path, explain important index values, identify the response behavior requiring inspection, and draft cautious reviewer-facing language. It may not compute indices, create or modify flags, change thresholds, reinterpret a display-only output as evidence, infer intent, determine misconduct, or recommend sanctions. Every substantive report claim must be traceable to a governed domain, a family-level signal or explicitly labeled contextual observation, and the relevant source record. Missing evidence and calibration requirements must be disclosed.

## Implementation and maintenance

The CSV registry is the runtime mapping used by the application. `b3_index_mapping.yaml` is the human-readable policy specification for domain roles, family aggregation, B3 strength rules, priority contribution, and reporting constraints. The two files should be versioned together. Any change to a function, index family, correction variant, threshold source, or eligibility state requires a rulebook review, a configuration change, and a rerun or explicit revalidation of the worked example. Thresholds should be supported by package documentation, published methodological evidence, simulation calibration, historical baselines, or an approved program rule. In the absence of such support, the index remains calibration-required or display-only.

The rulebook therefore provides a conservative and auditable bridge between forensic computation and human review. It makes clear which outputs are counted, why they are counted, how repeated variants are controlled, what evidence is missing, and where human judgment remains necessary.

## References

Belov, D. I. (2017). On the optimality of the detection of examinees with aberrant answer changes. *Applied Psychological Measurement, 41*(5), 338–352. https://doi.org/10.1177/0146621617692077

Chalmers, R. P. (2012). *mirt: A multidimensional item response theory package for the R environment*. *Journal of Statistical Software, 48*(6), 1–29. https://doi.org/10.18637/jss.v048.i06

Gorney, K., Wollack, J. A., Sinharay, S., & Eckerly, C. (2023). Using item scores and distractors to detect item compromise and preknowledge. *Journal of Educational and Behavioral Statistics, 48*(5), 636–660. https://doi.org/10.3102/10769986231159923

Sinharay, S. (2017). Detection of item preknowledge using likelihood ratio test and score test. *Journal of Educational and Behavioral Statistics, 42*(1), 46–68. https://doi.org/10.3102/1076998616673872

Wise, S. L. (2017). Rapid-guessing behavior: Its identification, interpretation, and implications. *Educational Measurement: Issues and Practice, 36*(4), 52–61. https://doi.org/10.1111/emip.12165

Gorney, K., & Deng, Y. (2024). *aberrance: Detection of aberrant behavior in educational testing* [R package]. https://kyliegorney.r-universe.dev/aberrance/
