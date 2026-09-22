# Collection commands

Run from the project root with the same Python environment used by PsyMAS:

```powershell
python tutorial_materials/emip/reproducibility/collect_materials.py
python tutorial_materials/emip/reproducibility/generate_figures.py
```

The collection script opens `data/output/psymas_run.sqlite` in read-only mode,
reads `data/psymas_demo_evaluated_snapshot.zip`, and writes only to
`tutorial_materials/emip/`. It does not call the detector backend, write to the
SQLite database, alter the Demo snapshot, or call an LLM.

The expected source run is recorded in `reproducibility/collection_metadata.json`.
Source hashes are recorded in `reproducibility/source_sha256.json`.
