# PsyMAS Tutorial Materials

## Suggested tutorial framing

PsyMAS is presented as a human-in-the-loop workbench for organizing unusual test-taking
patterns. The system does not turn a statistical flag into a misconduct finding. It
preserves the path from a detector output to an evidence domain, a case-level review
priority, an AI-assisted explanation, and a human decision.

## Six-step walkthrough

### Step 1. Confirm the assessment inputs

The user first checks the response matrix and optional response-time, item-parameter,
exposure, and answer-change files. In the saved Demo, the response and response-time
matrices contain 500 examinees and 40 items. Forty item-parameter rows are also stored
in the snapshot and were supplied to the saved run. The interface therefore does not
need to estimate item parameters for this demonstration.

### Step 2. Inspect deterministic outputs

The deterministic layer runs the selected `aberrance` functions when their required
inputs are available. It retains statistics, p-values, package-returned flags, and
diagnostic outputs. A flag is an input to review, not a conclusion. The detailed index
table is useful for audit, but the main teaching view should begin with index families
and evidence roles rather than a very wide table.

### Step 3. Translate outputs into governed evidence

The Evidence Input Table records the source function, method, index, domain, role, flag
source, threshold source, data basis, aggregation family, and provenance columns. The
rulebook prevents correction variants from being counted as independent evidence. In
the focal case, the three EDI_SD correction variants are collapsed into one
`tt_EDI_SD` family-level signal.

### Step 4. Read the focal case profile

For Examinee 332, `CUMP` and `NT` support moderate Response-Time evidence. The
`tt_EDI_SD` family is the primary Tampering signal, and `tt_GBT_SD` is a distinct
supporting family; together they produce strong Tampering evidence. Misfit,
Preknowledge, and Similarity have no eligible flagged indicators, while Change Pattern
is unavailable in the saved Evidence Input Table. The resulting stored profile is
`Convergent (Traceable)` with `Critical / Expedited` priority. This means that multiple
governed domains require expedited human review; it does not establish intent.

### Step 5. Use AI assistance carefully

The AI layer receives governed index evidence and selected raw-data summaries. The raw
summaries clarify the flagged pattern, for example item-level accuracy, response-time
comparisons, exposed-item performance, answer-change concentration, or change-point
locations. The prompt requires the model to explain why the case needs review, interpret
important indices in plain language, connect multiple indicators, disclose unavailable
evidence, and state what a reviewer should inspect next. It does not allow the model to
create flags, change thresholds, infer intent, determine misconduct, or recommend
sanctions.

### Step 6. Record human adjudication

The reviewer records the final decision and note separately from detector outputs. The
saved Demo has no human decision for Examinee 332, so any decision shown during a live
tutorial should be described as a new human action, not as part of the archived Demo
result.

## Suggested contrast cases

Case 1 shows an RT-only, medium-priority signal. Case 17 shows RT and PK evidence with
critical/expedited priority. Case 332 shows RT and TP evidence with critical/expedited
priority. These cases illustrate how domain combinations change the review queue without
claiming that the system has estimated operational classification accuracy.

## Figure captions

**Figure 1. Examinee 332 performance context.** Item-level response accuracy and
response time are shown against cohort/item means. The figure is contextual evidence
for human review; it is not itself a flagging rule.

**Figure 2. Examinee 332 governed domain profile.** The saved case has moderate RT,
strong TP, no eligible MF/PK/SIM evidence, and unavailable CP evidence. The profile
summarizes governed evidence rather than raw detector output counts.

**Figure 3. From evidence families to review priority.** The focal path begins with
`rg_CUMP`, `rg_NT`, the `tt_EDI_SD` family, and `tt_GBT_SD`. These family-level inputs
feed RT and TP domain summaries and then the saved Critical/Expedited review priority.
The EDI correction variants remain traceable but are not counted as separate families.

## Tutorial check questions

1. Which input files were actually available in the saved run?
2. Which four evidence families entered the focal case path?
3. Why do `EDI_SD_NO`, `EDI_SD_CO`, and `EDI_SD_TS` not count as three TP families?
4. Which domains affected the saved priority, and which domains were context or unavailable?
5. What does Critical/Expedited mean operationally, and what does it not mean?
6. Which claims can the LLM make, and which decisions remain human?

## Claims the tutorial should avoid

Do not call the saved snapshot a freshly recomputed v0.7.7 analysis. Do not report
detector sensitivity, specificity, false-positive rates, or LLM compliance rates from
this worked example. Do not describe a blank human decision as a negative adjudication.
