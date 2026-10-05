"""SQLite-backed integrated store for one PsyMAS detect run."""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Mapping
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

SCHEMA_VERSION = "1.0"
DEFAULT_DB_PATH = Path("data") / "output" / "psymas_run.sqlite"

DATA_TABLES = (
    "indices_export",
    "indices_table",
    "column_legend",
    "b3_input",
    "governed_review",
    "domain_evidence",
    "review_queue",
    "master_results",
    "final_flags",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    return value


def _json_text(value: Any) -> str:
    return json.dumps(_json_safe(value), ensure_ascii=False, default=str, allow_nan=False)


class RunStore:
    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._table_cache: dict[str, pd.DataFrame] = {}
        self._review_decisions_cache: dict[str, dict[str, dict[str, str]]] = {}
        self.init_schema()

    def clear_cache(self) -> None:
        """Invalidate materialized tables after an external database update."""
        self.init_schema()
        self._table_cache.clear()
        self._review_decisions_cache.clear()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    metadata_json TEXT,
                    forensic_result_json TEXT,
                    pair_visual_index_json TEXT,
                    is_active INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS review_decisions (
                    run_id TEXT NOT NULL,
                    examinee_id TEXT NOT NULL,
                    final_decision TEXT,
                    reviewer_note TEXT,
                    llm_explanation TEXT,
                    updated_at TEXT,
                    PRIMARY KEY (run_id, examinee_id)
                );
                CREATE TABLE IF NOT EXISTS llm_report_audits (
                    audit_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    examinee_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS llm_audit_case
                    ON llm_report_audits(run_id, examinee_id, created_at);
                """
            )
            conn.commit()

    def save_llm_audit(self, record: dict) -> None:
        """Append an immutable generation record, never overwrite a prior draft."""
        self.init_schema()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO llm_report_audits VALUES (?, ?, ?, ?, ?)",
                (record['audit_id'], record['run_id'], record['case_id'], record['created_at'], _json_text(record)),
            )

    def load_llm_audits(self, examinee_id: str, run_id: str | None = None) -> list[dict]:
        # A frozen snapshot may replace the file behind a cached RunStore.
        # Apply additive migrations before querying the replacement database.
        self.init_schema()
        run_id = str(run_id or self.active_run_id() or '')
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT record_json FROM llm_report_audits WHERE run_id=? AND examinee_id=? ORDER BY created_at DESC",
                (run_id, str(examinee_id)),
            ).fetchall()
        return [json.loads(row['record_json']) for row in rows]

    def active_run_id(self) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT run_id FROM runs WHERE is_active = 1 ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        return str(row["run_id"]) if row else None

    def list_runs(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    r.run_id,
                    r.created_at,
                    r.schema_version,
                    r.metadata_json,
                    r.is_active,
                    COUNT(d.examinee_id) AS review_decisions
                FROM runs r
                LEFT JOIN review_decisions d ON d.run_id = r.run_id
                GROUP BY r.run_id, r.created_at, r.schema_version, r.metadata_json, r.is_active
                ORDER BY r.is_active DESC, r.created_at DESC
                """
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            metadata: dict[str, Any] = {}
            if row["metadata_json"]:
                try:
                    parsed = json.loads(row["metadata_json"])
                    if isinstance(parsed, dict):
                        metadata = parsed
                except json.JSONDecodeError:
                    metadata = {}
            out.append(
                {
                    "run_id": row["run_id"],
                    "created_at": row["created_at"],
                    "schema_version": row["schema_version"],
                    "metadata": metadata,
                    "is_active": bool(row["is_active"]),
                    "review_decisions": int(row["review_decisions"] or 0),
                }
            )
        return out

    def set_active_run(self, run_id: str) -> bool:
        run_id = str(run_id)
        if not run_id:
            return False
        with self._connect() as conn:
            row = conn.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            if row is None:
                return False
            conn.execute("UPDATE runs SET is_active = 0")
            conn.execute("UPDATE runs SET is_active = 1 WHERE run_id = ?", (run_id,))
            conn.commit()
        return True

    def has_run(self, run_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT 1 FROM runs WHERE run_id = ?", (str(run_id),)).fetchone()
            return row is not None

    def _drop_data_tables(self, conn: sqlite3.Connection) -> None:
        for name in DATA_TABLES:
            conn.execute(f'DROP TABLE IF EXISTS "{name}"')

    def save_run(
        self,
        *,
        run_id: str,
        tables: dict[str, pd.DataFrame],
        forensic_result: dict | None = None,
        pair_visual_index: dict[str, list[dict]] | None = None,
        metadata: dict | None = None,
        review_decisions: dict[str, dict] | None = None,
    ) -> None:
        run_id = str(run_id)
        created_at = datetime.now(timezone.utc).isoformat()
        # A saved run replaces the materialized tables. Any in-process table
        # cache must be invalidated before the next page rerun.
        self._table_cache.clear()
        self._review_decisions_cache.clear()
        with self._connect() as conn:
            conn.execute("UPDATE runs SET is_active = 0")
            self._drop_data_tables(conn)
            for table_name, df in tables.items():
                if table_name not in DATA_TABLES or not isinstance(df, pd.DataFrame):
                    continue
                if df.empty:
                    continue
                df.copy().to_sql(table_name, conn, if_exists="replace", index=False)
            conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
            conn.execute(
                """
                INSERT INTO runs (
                    run_id, created_at, schema_version, metadata_json,
                    forensic_result_json, pair_visual_index_json, is_active
                ) VALUES (?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    run_id,
                    created_at,
                    SCHEMA_VERSION,
                    _json_text(metadata or {}),
                    _json_text(forensic_result or {}),
                    _json_text(pair_visual_index or {}),
                ),
            )
            conn.execute("DELETE FROM review_decisions WHERE run_id = ?", (run_id,))
            for examinee_id, payload in (review_decisions or {}).items():
                if not isinstance(payload, dict):
                    continue
                conn.execute(
                    """
                    INSERT INTO review_decisions (
                        run_id, examinee_id, final_decision, reviewer_note, llm_explanation, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        str(examinee_id),
                        str(payload.get("final_decision", "") or ""),
                        str(payload.get("reviewer_note", "") or ""),
                        str(payload.get("llm_explanation", "") or ""),
                        created_at,
                    ),
                )
            conn.commit()

    def load_table(self, name: str) -> pd.DataFrame:
        if name not in DATA_TABLES:
            return pd.DataFrame()
        cached = self._table_cache.get(name)
        if isinstance(cached, pd.DataFrame):
            return cached.copy()
        with self._connect() as conn:
            try:
                table = pd.read_sql_query(f'SELECT * FROM "{name}"', conn)
            except Exception:
                table = pd.DataFrame()
        self._table_cache[name] = table.copy()
        return table

    def load_forensic_result(self, run_id: str | None = None) -> dict:
        run_id = str(run_id or self.active_run_id() or "")
        if not run_id:
            return {}
        with self._connect() as conn:
            row = conn.execute(
                "SELECT forensic_result_json FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if not row or not row["forensic_result_json"]:
            return {}
        try:
            payload = json.loads(row["forensic_result_json"])
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def load_pair_visual_index(self, run_id: str | None = None) -> dict[str, list[dict]]:
        run_id = str(run_id or self.active_run_id() or "")
        if not run_id:
            return {}
        with self._connect() as conn:
            row = conn.execute(
                "SELECT pair_visual_index_json FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if not row or not row["pair_visual_index_json"]:
            return {}
        try:
            payload = json.loads(row["pair_visual_index_json"])
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def load_review_decisions(self, run_id: str | None = None) -> dict[str, dict]:
        run_id = str(run_id or self.active_run_id() or "")
        if not run_id:
            return {}
        cached = self._review_decisions_cache.get(run_id)
        if isinstance(cached, dict):
            return {key: value.copy() for key, value in cached.items()}
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT examinee_id, final_decision, reviewer_note, llm_explanation
                FROM review_decisions WHERE run_id = ?
                """,
                (run_id,),
            ).fetchall()
        out: dict[str, dict] = {}
        for row in rows:
            out[str(row["examinee_id"])] = {
                "final_decision": row["final_decision"] or "",
                "reviewer_note": row["reviewer_note"] or "",
                "llm_explanation": row["llm_explanation"] or "",
            }
        self._review_decisions_cache[run_id] = {key: value.copy() for key, value in out.items()}
        return out

    def save_review_decision(
        self,
        examinee_id: str,
        *,
        final_decision: str = "",
        reviewer_note: str = "",
        llm_explanation: str = "",
        run_id: str | None = None,
    ) -> None:
        run_id = str(run_id or self.active_run_id() or "")
        if not run_id:
            return
        updated_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO review_decisions (
                    run_id, examinee_id, final_decision, reviewer_note, llm_explanation, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, examinee_id) DO UPDATE SET
                    final_decision = excluded.final_decision,
                    reviewer_note = excluded.reviewer_note,
                    llm_explanation = excluded.llm_explanation,
                    updated_at = excluded.updated_at
                """,
                (
                    run_id,
                    str(examinee_id),
                    str(final_decision or ""),
                    str(reviewer_note or ""),
                    str(llm_explanation or ""),
                    updated_at,
                ),
            )
            conn.commit()
        self._review_decisions_cache.pop(run_id, None)


@lru_cache(maxsize=4)
def get_run_store(db_path: Path | str = DEFAULT_DB_PATH) -> RunStore:
    """Return one store instance per database path during the app process.

    Streamlit reruns the script for widget interactions. Reusing the store
    instance avoids reopening the database and recreating the schema on every
    rerun; materialized table reads are cached by ``RunStore.load_table``.
    """
    return RunStore(db_path)
