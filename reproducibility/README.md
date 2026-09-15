# PsyMAS Reproducibility Capsule

This directory records the worked-example generation and saved demonstration run.

The bundled snapshot is an evaluated run generated with PsyMAS v0.7.6 and
packaged with the v0.7.8 release. Its original run ID is preserved; the
snapshot is restored as-is and is not recomputed during Demo startup.

- `generate_psymas_data.R`: R generation script; fixed seed is `2026`.
- `demo_manifest.json`: generation version, package version, run identifier, design, parameter provenance, and SHA-256 values.
- `../data/psymas_demo_evaluated_snapshot.zip`: processed session inputs and SQLite outputs for the saved run.

The saved demonstration uses item parameters derived from simulation metadata;
the saved run therefore supplied `item_params.csv` and did not re-estimate
item parameters. PsyMAS uses `mirt` only when a new run does not provide
`item_params.csv`. The original `response_long.csv` and auxiliary truth tables
must be archived alongside this capsule for full regeneration from the
simulation source.
