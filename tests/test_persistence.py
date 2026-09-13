import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from src.config import ExperimentConfig, PersistenceConfig
from src.persistence import ResultStore, trial_key


class PersistenceTests(unittest.TestCase):
    def test_result_store_persists_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            config = ExperimentConfig(
                persistence=PersistenceConfig(
                    database=str(tmp_path / "results.sqlite"),
                    backup=str(tmp_path / "backup.sqlite"),
                    study_id="test",
                )
            )
            with ResultStore(config.persistence.database, config.persistence.backup) as store:
                run_id, key, attempt = store.start_attempt(config)
                self.assertEqual(key, trial_key(config))
                self.assertEqual(attempt, 1)
                store.finish_attempt(
                    run_id,
                    status="completed",
                    values={"relative_l2": 0.05, "success": True, "history": []},
                )
                self.assertTrue(store.has_terminal_result(key))
            connection = sqlite3.connect(config.persistence.backup)
            try:
                self.assertEqual(connection.execute("PRAGMA integrity_check").fetchone()[0], "ok")
                self.assertEqual(connection.execute("SELECT status FROM runs").fetchone()[0], "completed")
            finally:
                connection.close()

    def test_source_commit_and_runtime_are_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            config = ExperimentConfig(
                persistence=PersistenceConfig(
                    database=str(tmp_path / "results.sqlite"),
                    backup=str(tmp_path / "backup.sqlite"),
                    study_id="test-provenance",
                )
            )
            with patch.dict("os.environ", {"PINN_SOURCE_COMMIT": "abc123"}):
                with ResultStore(config.persistence.database, config.persistence.backup) as store:
                    run_id, _, _ = store.start_attempt(config)
                    store.finish_attempt(
                        run_id,
                        status="completed",
                        values={
                            "relative_l2": 0.05,
                            "success": True,
                            "runtime": {"device": "cuda:0", "device_name": "Tesla T4"},
                        },
                    )
                    row = store.connection.execute(
                        "SELECT git_commit, runtime_json FROM runs"
                    ).fetchone()
            self.assertEqual(row[0], "abc123")
            self.assertEqual(
                row[1], '{"device": "cuda:0", "device_name": "Tesla T4"}'
            )

    def test_stale_running_attempt_is_retained(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            config = ExperimentConfig(
                persistence=PersistenceConfig(
                    database=str(tmp_path / "results.sqlite"),
                    backup=str(tmp_path / "backup.sqlite"),
                    study_id="test-stale",
                )
            )
            with ResultStore(config.persistence.database, config.persistence.backup) as store:
                store.start_attempt(config)
                self.assertEqual(store.recover_stale_runs(), 1)
                status = store.connection.execute("SELECT status FROM runs").fetchone()[0]
                self.assertEqual(status, "interrupted")
