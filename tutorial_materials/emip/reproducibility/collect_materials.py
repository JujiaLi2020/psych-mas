"""Collect read-only EM:IP tutorial materials from the PsyMAS v0.7.7 demo.

This script does not run detectors or modify project data. It extracts a small,
auditable teaching packet from the saved evaluated snapshot and SQLite store.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sqlite3
import zipfile
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rows(con: sqlite3.Connection, table: str, where: str = "", params=()):
    columns = [r[1] for r in con.execute(f"pragma table_info({table})")]
    query = f"select * from {table} {where}"
    return columns, [dict(zip(columns, row)) for row in con.execute(query, params)]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def write_csv(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8", newline="")
        return
    fields = list(records[0])
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    root = args.project_root.resolve()
    out = (args.out or root / "tutorial_materials" / "emip").resolve()
    out.mkdir(parents=True, exist_ok=True)

    source_db = root / "data" / "output" / "psymas_run.sqlite"
    source_zip = root / "data" / "psymas_demo_evaluated_snapshot.zip"
    manifest_path = root / "reproducibility" / "demo_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    con = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)
    try:
        case = out / "case_332"
        for table in ("b3_input", "domain_evidence", "governed_review", "final_flags", "master_results", "review_queue"):
            _, records = rows(con, table, "where Examinee_ID = ?", (332,))
            write_csv(case / f"{table}.csv", records)

        _, indices = rows(con, "indices_table", "where Examinee_ID = ?", (332,))
        write_json(case / "indices_table.json", indices[0] if indices else {})

        selected = {"1_rt_only": 1, "17_rt_pk": 17}
        old_focal_contrast = out / "contrast_cases" / "332_rt_tp"
        if old_focal_contrast.exists():
            shutil.rmtree(old_focal_contrast)
        contrast_records = []
        for label, examinee_id in selected.items():
            _, records = rows(con, "governed_review", "where Examinee_ID = ?", (examinee_id,))
            for record in records:
                record["case_label"] = label
                contrast_records.append(record)
        write_csv(out / "contrast_cases" / "candidate_overview.csv", contrast_records)
        for label, examinee_id in selected.items():
            target = out / "contrast_cases" / label
            target.mkdir(parents=True, exist_ok=True)
            for table in ("domain_evidence", "governed_review", "final_flags"):
                _, records = rows(con, table, "where Examinee_ID = ?", (examinee_id,))
                write_csv(target / f"{table}.csv", records)

        # A compact, reproducible summary of all saved run IDs and profile counts.
        run_counts = con.execute("select Run_ID, count(*) as n from master_results group by Run_ID").fetchall()
        profile_counts = con.execute("select Evidence_Status, Review_Priority, count(*) as n from governed_review group by Evidence_Status, Review_Priority").fetchall()
        write_json(out / "reproducibility" / "sqlite_summary.json", {
            "source_database": str(source_db),
            "run_counts": [dict(run_id=r[0], n=r[1]) for r in run_counts],
            "profile_counts": [dict(evidence_status=r[0], review_priority=r[1], n=r[2]) for r in profile_counts],
            "tables": [r[0] for r in con.execute("select name from sqlite_master where type='table' order by name")],
        })
    finally:
        con.close()

    # Extract only the relevant 332 rows from session_inputs.json; do not copy the full snapshot.
    with zipfile.ZipFile(source_zip) as z:
        session = json.loads(z.read("session_inputs.json"))
    write_json(out / "case_332" / "session_input_metadata.json", {
        "snapshot_manifest": manifest,
        "response_rows": session.get("forensic_responses", [])[331:332],
        "response_time_rows": session.get("forensic_rt_data", [])[331:332],
        "person_parameter_rows": session.get("forensic_person_params", [])[331:332],
        "compromised_items": session.get("prep_compromised_items", []),
        "answer_change_rows": [r for r in session.get("prep_answer_changes", []) if str(r.get("examinee_id", "")).upper() in {"E332", "332"}],
        "item_parameter_count": len(session.get("item_params", [])),
        "item_parameter_preview": session.get("item_params", [])[:5],
    })

    hashes = {
        "demo_manifest.json": sha256(manifest_path),
        "psymas_demo_evaluated_snapshot.zip": sha256(source_zip),
        "psymas_run.sqlite": sha256(source_db),
    }
    write_json(out / "reproducibility" / "source_sha256.json", hashes)
    write_json(out / "reproducibility" / "collection_metadata.json", {
        "collection_script": "tutorial_materials/emip/reproducibility/collect_materials.py",
        "collection_mode": "read_only_saved_run_extraction",
        "current_distribution": manifest.get("package_software_version"),
        "snapshot_generation_version": manifest.get("generation_software_version"),
        "run_id": manifest.get("run_id"),
        "examinee_case": 332,
        "new_detector_run_performed": False,
    })


if __name__ == "__main__":
    main()
