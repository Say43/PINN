"""Two-GPU, resume-aware runner for the Reaction V5 study.

The parent process is the only SQLite writer and Kaggle Dataset publisher.  Each
worker owns one visible T4 and returns a completed training result to the parent.
This avoids concurrent database backups and concurrent Dataset uploads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import queue
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from bench.v5 import lambda_selection_conditions, study_conditions
from kaggle.publish import KagglePublisher
from src.config import ExperimentConfig
from src.persistence import ResultStore, trial_key


@dataclass
class V5QuotaState:
    total_quota_hours: float = 27.0
    prior_hours: float = 1.5
    reserve_hours: float = 1.0
    lambda_limit_hours: float = 0.3
    matrix_limit_hours: float = 7.4
    actual_lambda_hours: float = 0.0
    actual_matrix_hours: float = 0.0

    @classmethod
    def load(cls, path: Path) -> "V5QuotaState":
        if not path.exists():
            return cls()
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def can_start_batch(self, phase: str, estimated_hours: float) -> bool:
        actual = getattr(self, f"actual_{phase}_hours")
        limit = getattr(self, f"{phase}_limit_hours")
        total = self.prior_hours + self.actual_lambda_hours + self.actual_matrix_hours
        return actual + estimated_hours <= limit and total + estimated_hours <= (
            self.total_quota_hours - self.reserve_hours
        )

    def record(self, phase: str, hours: float) -> None:
        name = f"actual_{phase}_hours"
        setattr(self, name, getattr(self, name) + hours)


def _runtime(device: str) -> dict:
    import torch

    resolved = torch.device(device)
    data = {
        "device": str(resolved),
        "device_name": torch.cuda.get_device_name(resolved),
        "device_capability": list(torch.cuda.get_device_capability(resolved)),
        "torch_version": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
    }
    return data


def _worker(gpu_index: int, incoming, outgoing) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    import torch

    from src.train import LocalGpuTimeLimit, train_once

    while True:
        item = incoming.get()
        if item is None:
            return
        run_id, raw = item
        config = ExperimentConfig.from_dict(raw)
        runtime = {"device": config.train.device, "worker_gpu_index": gpu_index}
        try:
            runtime = {**_runtime(config.train.device), "worker_gpu_index": gpu_index}
            result = train_once(config)
            outgoing.put((run_id, result.status, {**result.values, "runtime": runtime}))
        except FloatingPointError as error:
            outgoing.put(
                (
                    run_id,
                    "numerical_fail",
                    {
                        "relative_l2": 10.0,
                        "relative_l2_censored": True,
                        "success": False,
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                        "runtime": runtime,
                    },
                )
            )
        except (torch.cuda.OutOfMemoryError, LocalGpuTimeLimit, KeyboardInterrupt) as error:
            outgoing.put(
                (
                    run_id,
                    "infra_fail",
                    {
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                        "runtime": runtime,
                    },
                )
            )
        except Exception as error:
            outgoing.put(
                (
                    run_id,
                    "infra_fail",
                    {
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                        "runtime": runtime,
                    },
                )
            )


def _verify_frozen_preregistration(path: Path, expected_sha256: str) -> None:
    payload = path.read_bytes()
    text = payload.decode("utf-8")
    if "Status: ENTWURF" in text or "NICHT eingefroren" in text:
        raise RuntimeError("V5 matrix is blocked while the preregistration is still a draft")
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise RuntimeError(f"V5 preregistration hash mismatch: expected {expected_sha256}, got {actual}")


def _condition_queue(args, base: ExperimentConfig) -> tuple[str, list[ExperimentConfig]]:
    if args.phase == "lambda":
        return "lambda", list(lambda_selection_conditions(base))
    if args.lambda_r is None or not args.study_id or not args.prereg_sha256:
        raise RuntimeError("matrix phase requires --lambda-r, --study-id, and --prereg-sha256")
    _verify_frozen_preregistration(args.preregistration, args.prereg_sha256)
    return "matrix", list(
        study_conditions(base, lambda_r=args.lambda_r, study_id=args.study_id)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("lambda", "matrix"), required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/reaction_v5.json"))
    parser.add_argument("--lambda-r", type=float)
    parser.add_argument("--study-id")
    parser.add_argument("--preregistration", type=Path, default=Path("PREREGISTRATION-V5.md"))
    parser.add_argument("--prereg-sha256")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--wall-budget-seconds", type=float, default=2200.0)
    parser.add_argument(
        "--quota-state", type=Path, default=Path("/kaggle/working/v5_quota_state.json")
    )
    args = parser.parse_args()

    handle = os.environ.get("PINN_RESULTS_DATASET")
    if not handle:
        raise RuntimeError("PINN_RESULTS_DATASET must name the durable private results Dataset")
    publisher = KagglePublisher(handle)
    database = Path("/kaggle/working/results/results.sqlite")
    backup = Path("/kaggle/working/results/results.backup.sqlite")
    publisher.restore(database, backup, args.quota_state)

    base = ExperimentConfig.from_json(args.config)
    phase, conditions = _condition_queue(args, base)
    quota_state = V5QuotaState.load(args.quota_state)
    batch_count = math.ceil(len(conditions) / args.workers)
    phase_limit = getattr(quota_state, f"{phase}_limit_hours")
    estimated_batch_hours = phase_limit / batch_count
    process_started = time.perf_counter()

    with ResultStore(database, backup) as store:
        store.recover_stale_runs()
        pending_conditions = [
            condition for condition in conditions if not store.has_terminal_result(trial_key(condition))
        ]
        if not pending_conditions:
            print(f"V5 {phase}: all conditions already terminal", flush=True)
            return

        context = mp.get_context("spawn")
        outgoing = context.Queue()
        incoming = [context.Queue() for _ in range(args.workers)]
        workers = [
            context.Process(target=_worker, args=(index, incoming[index], outgoing))
            for index in range(args.workers)
        ]
        for worker in workers:
            worker.start()
        try:
            for offset in range(0, len(pending_conditions), args.workers):
                elapsed = time.perf_counter() - process_started
                if elapsed + estimated_batch_hours * 3600.0 > args.wall_budget_seconds:
                    print("V5 wall budget guard stopped before the next batch", flush=True)
                    return
                if not quota_state.can_start_batch(phase, estimated_batch_hours):
                    raise RuntimeError(f"V5 quota guard stopped before the next {phase} batch")

                batch = pending_conditions[offset : offset + args.workers]
                checkpoint = time.perf_counter()
                run_ids = set()
                labels = {}
                for worker_index, condition in enumerate(batch):
                    run_id, _, _ = store.start_attempt(condition)
                    run_ids.add(run_id)
                    labels[run_id] = (
                        f"{condition.model.backbone}/{condition.train.precision}/"
                        f"{condition.train.regularization}/seed={condition.train.seed}/"
                        f"lambda={condition.train.lambda_r:g}"
                    )
                    incoming[worker_index].put((run_id, condition.to_dict()))

                failures = []
                while run_ids:
                    try:
                        run_id, status, values = outgoing.get(timeout=30.0)
                    except queue.Empty:
                        dead = [worker.pid for worker in workers if not worker.is_alive()]
                        if dead:
                            raise RuntimeError(f"V5 worker process died without a result: {dead}")
                        continue
                    if run_id not in run_ids:
                        raise RuntimeError(f"unexpected V5 result for run {run_id}")
                    now = time.perf_counter()
                    quota_state.record(phase, (now - checkpoint) / 3600.0)
                    checkpoint = now
                    quota_state.save(args.quota_state)
                    store.finish_attempt(run_id, status=status, values=values)
                    run_ids.remove(run_id)
                    note = f"V5 {phase}: {labels[run_id]}; status={status}"
                    publisher.publish(
                        database_backup=backup,
                        quota_state=args.quota_state,
                        note=note,
                        extra_files=(args.preregistration,),
                    )
                    if status == "infra_fail":
                        failures.append(labels[run_id])

                if failures:
                    raise RuntimeError(f"V5 infrastructure failure; retry unchanged: {failures}")
        finally:
            for worker_queue in incoming:
                worker_queue.put(None)
            for worker in workers:
                worker.join(timeout=30.0)
                if worker.is_alive():
                    worker.terminate()


if __name__ == "__main__":
    main()
