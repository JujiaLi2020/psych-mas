# PsyMAS Reproducibility Capsule

This directory records the worked-example generation and saved demonstration run.

- `generate_psymas_data.R`: R generation script; fixed seed is `2026`.
- `demo_manifest.json`: run identifier, design, parameter provenance, and SHA-256 values.
- `../data/psymas_demo_evaluated_snapshot.zip`: processed session inputs and SQLite outputs for the saved run.

The saved demonstration uses item parameters derived from simulation metadata. PsyMAS uses `mirt` to estimate item parameters only when a new run does not provide `item_params.csv`. The original `response_long.csv` and auxiliary truth tables must be archived alongside this capsule for full regeneration from the simulation source.
