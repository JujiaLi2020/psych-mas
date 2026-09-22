# EM:IP Tutorial Material Collection Report

## 1. Collection scope

本次材料采集面向 Educational Measurement: Issues and Practice tutorial，目标是把
PsyMAS 的实际流程整理成可教学、可核查、可复现的材料包。采集过程没有重新运行
detectors，没有修改 production code、rulebook 或 Demo snapshot，也没有调用付费或外部
LLM。

## 2. 版本与运行边界

| Item | Verified value |
|---|---|
| Current distribution | PsyMAS v0.7.7 |
| Snapshot generation version | v0.7.6 |
| Snapshot status | Restored evaluated run; not recomputed at Demo startup |
| Run ID | `9458ac0d-ff96-4659-8cc9-43ce86f036b8` |
| Scenario | B |
| Seed in manifest | 2026 |
| Design | 500 examinees, 40 dichotomous items, 2PL, log-time RT |
| Main case | Examinee 332 |
| New detector run during collection | No |

v0.7.7 是当前软件发行版本，但 Demo 中的统计输出仍然是 v0.7.6 生成并被
v0.7.7 恢复的保存结果。教程应同时报告这两个事实。

## 3. Parameter provenance

保存的 `session_inputs.json` 中包含 40 行 `item_params`、500 行 person-parameter
结果、responses 和 response times。manifest 明确写明：item parameters were supplied
to the saved demonstration run; item-parameter re-estimation was not used. `mirt` 是在
没有有效 `item_params.csv` 时的 fallback，而不是本次保存运行中实际使用的路径。

仍有一个需要谨慎说明的 provenance gap：manifest 没有记录 person parameters 是否在
生成阶段估计，只有“stored in snapshot; no independent re-estimation provenance was
recorded”。因此教程不应声称完整重现了 person-parameter estimation。

## 4. Examinee 332: observed evidence path

332 号案例的保存结果如下：

| Layer | Saved result |
|---|---|
| Raw detector families entering evidence | `rg_CUMP`, `rg_NT`, `tt_EDI_SD`, `tt_GBT_SD` |
| RT domain | Moderate, rule B3-04; two supporting flags (`CUMP`, `NT`) |
| TP domain | Strong, rule B3-05; primary `tt_EDI_SD` plus supporting `tt_GBT_SD` |
| MF domain | None; no eligible flagged PM indicator |
| PK domain | None; no eligible flagged PK indicator |
| SIM domain | None; no eligible flagged similarity indicator |
| CP domain | Unavailable; no Evidence Input rows |
| Case profile in saved store | `Convergent (Traceable)` |
| Priority | `Critical / Expedited`, rule `RP-04b` |
| Primary concern | RT, TP |
| Human decision | Blank in saved run |
| LLM explanation | Blank in saved SQLite row; generated on demand by the UI |

The important aggregation check passes: `EDI_SD_NO / CO / TS` is represented as one
`tt_EDI_SD` family-level signal. The `EDI_SD_CO` flag and p-value flag are retained in
the raw/evidence trace, but do not become independent TP families. The lineage should
therefore describe four families, not five raw flag columns.

## 5. Contrast cases

- Examinee 1: RT-only, moderate RT evidence, medium priority. This is a useful
  single-domain comparison.
- Examinee 17: RT moderate plus PK strong, critical/expedited priority. This is a
  useful cross-domain comparison.
- Examinee 332: RT moderate plus TP strong, critical/expedited priority. This is the
  focal detailed case.

The saved run did not contain a case with `RT_Strength = unavailable`; therefore a
missing-RT contrast should not be presented as an observed case unless a separate,
clearly labelled rerun is later created.

## 6. Important discrepancy for teaching

The SQLite store contains legacy-compatible labels such as `Convergent (Traceable)` and
rule identifiers such as `B6-04b`. The tutorial should explain the current conceptual
logic in plain language, but should quote the exact stored value when showing a database
record. Do not silently relabel the saved run as a newly recomputed run.

## 7. Files to cite in the tutorial

Use the case packet for claims about the observed case, the manifest for provenance,
and the collection metadata for the fact that no new detector run was performed.
The generated figures are derived from the saved snapshot and are not screenshots of a
new run.
