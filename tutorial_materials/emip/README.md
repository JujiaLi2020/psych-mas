# EM:IP Tutorial Materials for PsyMAS

This folder is a read-only teaching and evidence-collection package for the
PsyMAS tutorial. It was collected from the current v0.7.7 project without
running a new detector job or changing production code.

## Source boundary

- Current distribution: PsyMAS `v0.7.7`.
- Saved demonstration results: generated under PsyMAS `v0.7.6`, restored by
  the v0.7.7 Demo workflow without recomputation.
- Preserved run ID: `9458ac0d-ff96-4659-8cc9-43ce86f036b8`.
- Main worked example: Examinee `332`.
- Item parameters: 40 supplied parameter rows are present in the saved
  snapshot; the saved run did not use the `mirt` fallback.
- No paid or external LLM call was made while collecting these materials.

## Contents

- `collection_report.md`: Chinese fact-checking report and discrepancies.
- `manuscript_ready_materials.md`: English tutorial-ready explanation,
  walkthrough, case descriptions, captions, and check questions.
- `rulebook_construction_apa.md`: publication-ready explanation of the
  evidence-domain mapping, family aggregation, B3 synthesis, priority policy,
  and LLM/reporting boundaries.
- `claim_evidence_matrix.csv`: claim-to-source mapping for the tutorial.
- `case_332/`: extractable evidence packet for Examinee 332.
- `contrast_cases/`: two saved-run comparison cases (1, RT only; 17, RT + PK).
- `figures/`: data-derived PNG figures suitable for tutorial editing.
- `reproducibility/`: collection script, source hashes, SQLite summary, and
  collection metadata.
- `remaining_gaps.md`: items that should not be implied as validated.

## Recreate this package

From the project root:

```powershell
python tutorial_materials/emip/reproducibility/collect_materials.py
python tutorial_materials/emip/reproducibility/generate_figures.py
```

Both commands read the saved SQLite database and Demo snapshot. They do not
modify `data/`, the snapshot, the rulebook, or the production application.

## Recommended tutorial use

Use the six-step walkthrough in `manuscript_ready_materials.md` as the main
teaching sequence. Use the full CSV files only as supporting evidence. Present
the v0.7.6 generation/version distinction explicitly; do not describe the
saved Demo output as a freshly recomputed v0.7.7 run.
