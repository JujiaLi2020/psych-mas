# PsyMAS Reproducibility Capsule

This repository retains the bundled Demo snapshot and its provenance. Research validation code, source archives, and regeneration tools are local-only and are not distributed with the application.

This directory records the worked-example generation and saved demonstration run.

The bundled snapshot was generated for PsyMAS `v0.7.7` and refreshed from the
fixed-seed 80% copying simulation on 2026-10-01. Its generation version, run ID, seed,
input checksums, and parameter provenance are recorded in `demo_manifest.json`.
Demo startup restores that evaluated snapshot as-is and does not recompute it.

- `generate_psymas_data.R`: R generation script; fixed seed is `2026`.
- Local-only `demo_validation/generated_80/psymas_tutorial_data/`: archived source inputs and truth tables used for the bundled snapshot.
- Local-only `../tools/generate_demo_snapshot.py`: reproducibly runs the current detector rules and packages the evaluated snapshot.
- Local-only `demo_validation/validate_rerun_consistency.py`: reruns the saved inputs/configuration and compares detector/report hashes with the snapshot baseline.
- `demo_manifest.json`: records generation version, package version, run identifier, design, parameter provenance, and SHA-256 values.
- `../data/psymas_demo_evaluated_snapshot_v0.7.7.zip`: processed session inputs and SQLite outputs for the saved run.

The saved demonstration uses item parameters derived from simulation metadata. PsyMAS uses `mirt` to estimate item parameters only when a new run does not provide `item_params.csv`. The original `response_long.csv` and auxiliary truth tables must be archived alongside this capsule for full regeneration from the simulation source.
