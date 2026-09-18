import json
import multiprocessing as mp
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from bench.v5 import study_conditions
from kaggle.publish import KagglePublisher
from kaggle.v5_runner import V5QuotaState, _worker, baseline_gate, condition_batches
from src.config import ExperimentConfig, ModelConfig, PDEConfig, PersistenceConfig, TrainConfig
from src.persistence import ResultStore, trial_key


def wait_then_worker(event, incoming, outgoing):
    event.wait()
    _worker(0, incoming, outgoing)


class V5ResumeTests(unittest.TestCase):
    def test_real_process_interruption_retries_only_unfinished_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = ExperimentConfig(
                pde=PDEConfig(name="reaction", domain_points=4, boundary_points=2,
                              initial_points=2, evaluation_x=3, evaluation_t=3),
                model=ModelConfig(mlp_hidden_dim=4, num_layers=1),
                train=TrainConfig(max_iters=1, history_size=2, line_search_fn=None, device="cpu"),
                persistence=PersistenceConfig(database=str(root / "runs.sqlite"),
                                              backup=str(root / "backup.sqlite"),
                                              study_id="resume_NOT_STUDY_DATA"))
            completed = config.with_condition(seed=101)
            interrupted = config.with_condition(seed=102)
            context = mp.get_context("spawn")
            incoming, outgoing = context.Queue(), context.Queue()
            with ResultStore(config.persistence.database, config.persistence.backup) as store:
                first, _, _ = store.start_attempt(completed)
                worker = context.Process(target=_worker, args=(0, incoming, outgoing))
                worker.start()
                try:
                    incoming.put((first, completed.to_dict()))
                    run_id, status, values = outgoing.get(timeout=40)
                    self.assertEqual(status, "completed")
                    store.finish_attempt(run_id, status=status, values=values)
                finally:
                    incoming.put(None)
                    worker.join(timeout=10)
                    if worker.is_alive():
                        worker.terminate(); worker.join(timeout=5)
                second, key, _ = store.start_attempt(interrupted)
                # Kill a real spawned process after its attempt was scheduled,
                # before training; resumption restarts whole unfinished runs.
                event = context.Event()
                killed = context.Process(target=wait_then_worker, args=(event, incoming, outgoing))
                killed.start()
                killed.terminate(); killed.join(timeout=10)
                self.assertEqual(store.recover_stale_runs(), 1)
                self.assertTrue(store.has_terminal_result(trial_key(completed)))
                self.assertFalse(store.has_terminal_result(key))
                retry, retry_key, attempt = store.start_attempt(interrupted)
                self.assertEqual((retry_key, attempt), (key, 2))
                worker = context.Process(target=_worker, args=(0, incoming, outgoing))
                worker.start()
                try:
                    incoming.put((retry, interrupted.to_dict()))
                    run_id, status, values = outgoing.get(timeout=40)
                    self.assertEqual(status, "completed")
                    store.finish_attempt(run_id, status=status, values=values)
                finally:
                    incoming.put(None); worker.join(timeout=10)
                    if worker.is_alive():
                        worker.terminate(); worker.join(timeout=5)
                rows = store.connection.execute(
                    "SELECT status FROM runs WHERE trial_key=? ORDER BY attempt_no", (key,)).fetchall()
                self.assertEqual([r[0] for r in rows], ["interrupted", "completed"])
                self.assertEqual(store.connection.execute("PRAGMA integrity_check").fetchone()[0], "ok")

    def test_quota_roundtrip_keeps_v3_state_and_v5_consumption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mounted, published, restored = (root / name for name in ("mounted", "publish", "restored"))
            mounted.mkdir(); restored.mkdir()
            (mounted / "quota_state.json").write_text('{"legacy": true}', encoding="utf-8")
            with patch.object(KagglePublisher, "mounted_dir", new_callable=property,
                              fget=lambda _: mounted):
                publisher = KagglePublisher("test/private", published)
            db = root / "backup.sqlite"
            # File copying is tested without invoking any remote API.
            db.write_bytes(b"placeholder")
            state = root / "v5_quota_state.json"
            V5QuotaState(actual_lambda_hours=.17).save(state)
            with patch.dict("sys.modules", {"kagglehub": type("Hub", (), {"dataset_upload": staticmethod(lambda *a, **kw: None)})}):
                publisher.publish(database_backup=db, quota_state=state, note="local test")
            self.assertEqual(json.loads((published / "quota_state.json").read_text()), {"legacy": True})
            with patch.object(KagglePublisher, "mounted_dir", new_callable=property,
                              fget=lambda _: published):
                publisher.restore(restored / state.name)
            self.assertEqual(V5QuotaState.load(restored / state.name).actual_lambda_hours, .17)

    def test_baseline_barrier_and_failure(self):
        base = ExperimentConfig.from_json("configs/reaction_v5.json")
        conditions = list(study_conditions(base, lambda_r=1e-4, study_id="v5_test"))
        baseline, batches = condition_batches(conditions, "matrix", 2)
        self.assertEqual([len(b) for b in batches[:3]], [2, 2, 1])
        self.assertEqual([c for b in batches[:3] for c in b], baseline)
        with tempfile.TemporaryDirectory() as directory:
            with ResultStore(Path(directory) / "db", Path(directory) / "backup") as store:
                with self.assertRaises(RuntimeError):
                    baseline_gate(store, baseline)
                for config in baseline:
                    run_id, _, _ = store.start_attempt(config)
                    store.finish_attempt(run_id, status="completed", values={"success": True})
                with self.assertRaises(RuntimeError):
                    baseline_gate(store, baseline)
