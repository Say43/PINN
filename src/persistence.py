from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from src.config import ExperimentConfig


SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit() -> str:
    supplied = os.environ.get("PINN_SOURCE_COMMIT", "").strip()
    if supplied:
        return supplied
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def trial_key(config: ExperimentConfig) -> str:
    payload = f"{config.persistence.study_id}\n{config.canonical_json()}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class ResultStore:
    def __init__(self, database: str | Path, backup: str | Path) -> None:
        self.path = Path(database)
        self.backup_path = Path(backup)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.backup_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=60.0, isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self._create_schema()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "ResultStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield self.connection
        except Exception:
            self.connection.execute("ROLLBACK")
            raise
        else:
            self.connection.execute("COMMIT")

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS study_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS trials (
                trial_key TEXT PRIMARY KEY,
                study_id TEXT NOT NULL,
                pde TEXT NOT NULL,
                backbone TEXT NOT NULL,
                precision TEXT NOT NULL,
                regularization TEXT NOT NULL,
                seed INTEGER NOT NULL,
                config_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                trial_key TEXT NOT NULL REFERENCES trials(trial_key),
                attempt_no INTEGER NOT NULL,
                status TEXT NOT NULL CHECK(status IN (
                    'running', 'completed', 'numerical_fail', 'infra_fail', 'interrupted'
                )),
                started_at TEXT NOT NULL,
                finished_at TEXT,
                wall_seconds REAL,
                iterations INTEGER,
                function_evaluations INTEGER,
                solver_nfe INTEGER,
                parameter_count INTEGER,
                relative_l2 REAL,
                relative_l2_censored INTEGER,
                success INTEGER,
                time_to_target_seconds REAL,
                final_metrics_json TEXT,
                history_json TEXT,
                error_type TEXT,
                error_message TEXT,
                runtime_json TEXT,
                git_commit TEXT NOT NULL,
                UNIQUE(trial_key, attempt_no)
            );
            CREATE INDEX IF NOT EXISTS idx_runs_trial_status ON runs(trial_key, status);
            """
        )
        self.connection.execute(
            "INSERT OR REPLACE INTO study_meta(key, value) VALUES('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        columns = {row[1] for row in self.connection.execute("PRAGMA table_info(runs)")}
        if "solver_nfe" not in columns:
            self.connection.execute("ALTER TABLE runs ADD COLUMN solver_nfe INTEGER")
        if "runtime_json" not in columns:
            self.connection.execute("ALTER TABLE runs ADD COLUMN runtime_json TEXT")

    def recover_stale_runs(self) -> int:
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE runs
                   SET status='interrupted', finished_at=?, error_type='stale_running',
                       error_message='Recovered as interrupted during resume'
                 WHERE status='running'
                """,
                (utc_now(),),
            )
        return cursor.rowcount

    def has_terminal_result(self, key: str) -> bool:
        row = self.connection.execute(
            """
            SELECT 1 FROM runs
             WHERE trial_key=? AND status IN ('completed', 'numerical_fail')
             LIMIT 1
            """,
            (key,),
        ).fetchone()
        return row is not None

    def start_attempt(self, config: ExperimentConfig) -> tuple[str, str, int]:
        key = trial_key(config)
        run_id = str(uuid.uuid4())
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO trials(
                    trial_key, study_id, pde, backbone, precision, regularization,
                    seed, config_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    config.persistence.study_id,
                    config.pde.name,
                    config.model.backbone,
                    config.train.precision,
                    config.train.regularization,
                    config.train.seed,
                    config.canonical_json(),
                    utc_now(),
                ),
            )
            row = connection.execute(
                "SELECT COALESCE(MAX(attempt_no), 0) + 1 AS next_attempt FROM runs WHERE trial_key=?",
                (key,),
            ).fetchone()
            attempt_no = int(row["next_attempt"])
            connection.execute(
                """
                INSERT INTO runs(run_id, trial_key, attempt_no, status, started_at, git_commit)
                VALUES (?, ?, ?, 'running', ?, ?)
                """,
                (run_id, key, attempt_no, utc_now(), git_commit()),
            )
        self.backup()
        return run_id, key, attempt_no

    def finish_attempt(self, run_id: str, *, status: str, values: dict[str, Any]) -> None:
        allowed = {"completed", "numerical_fail", "infra_fail", "interrupted"}
        if status not in allowed:
            raise ValueError(f"invalid terminal status: {status}")
        payload = {
            "finished_at": utc_now(),
            "wall_seconds": values.get("wall_seconds"),
            "iterations": values.get("iterations"),
            "function_evaluations": values.get("function_evaluations"),
            "solver_nfe": values.get("solver_nfe"),
            "parameter_count": values.get("parameter_count"),
            "relative_l2": values.get("relative_l2"),
            "relative_l2_censored": int(bool(values.get("relative_l2_censored", False))),
            "success": int(bool(values.get("success", False))),
            "time_to_target_seconds": values.get("time_to_target_seconds"),
            "final_metrics_json": json.dumps(values.get("final_metrics", {}), sort_keys=True),
            "history_json": json.dumps(values.get("history", []), sort_keys=True),
            "error_type": values.get("error_type"),
            "error_message": values.get("error_message"),
            "runtime_json": json.dumps(values.get("runtime", {}), sort_keys=True),
        }
        with self.transaction() as connection:
            connection.execute(
                """
                UPDATE runs SET
                    status=:status, finished_at=:finished_at, wall_seconds=:wall_seconds,
                    iterations=:iterations, function_evaluations=:function_evaluations,
                    solver_nfe=:solver_nfe,
                    parameter_count=:parameter_count, relative_l2=:relative_l2,
                    relative_l2_censored=:relative_l2_censored, success=:success,
                    time_to_target_seconds=:time_to_target_seconds,
                    final_metrics_json=:final_metrics_json, history_json=:history_json,
                    error_type=:error_type, error_message=:error_message,
                    runtime_json=:runtime_json
                WHERE run_id=:run_id
                """,
                {"run_id": run_id, "status": status, **payload},
            )
        self.backup()

    def backup(self) -> None:
        temporary = self.backup_path.with_suffix(self.backup_path.suffix + ".tmp")
        if temporary.exists():
            temporary.unlink()
        destination = sqlite3.connect(temporary)
        try:
            self.connection.backup(destination)
            result = destination.execute("PRAGMA integrity_check").fetchone()[0]
            if result != "ok":
                raise RuntimeError(f"SQLite backup integrity check failed: {result}")
        finally:
            destination.close()
        shutil.move(str(temporary), str(self.backup_path))
