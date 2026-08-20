from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from bench.budget import QuotaState
from kaggle.publish import KagglePublisher
from src.config import ExperimentConfig
from src.train import run


def stage_conditions(stage: str, regularizations: tuple[str, ...] = ("none", "double_backprop")):
    seeds = (0,) if stage == "m2b" else ((1, 2, 3, 4) if stage == "stage_a" else (0, 1, 2, 3, 4))
    for backbone in ("mlp", "grand", "gread"):
        for precision in ("fp32", "fp64"):
            for regularization in regularizations:
                for seed in seeds:
                    yield backbone, precision, regularization, seed


def apply_plan(base: ExperimentConfig, plan: dict, stage: str) -> ExperimentConfig:
    raw = base.to_dict()
    selected = plan["selected"]
    raw["train"]["max_iters"] = selected["max_iters"]
    raw["train"]["lambda_r"] = 1.0
    raw["train"]["device"] = "cuda:0"
    if selected["point_scheme"] == "reduced":
        if stage in {"m2b", "stage_a"}:
            raw["pde"].update(domain_points=196, boundary_points=100, initial_points=100)
        else:
            raw["pde"].update(domain_points=2025, boundary_points=50, initial_points=50)
    raw["persistence"]["database"] = "/kaggle/working/results/results.sqlite"
    raw["persistence"]["backup"] = "/kaggle/working/results/results.backup.sqlite"
    return ExperimentConfig.from_dict(raw)


def estimated_hours(plan: dict, backbone: str, precision: str, regularization: str) -> float:
    key = f"{backbone}:{precision}:{regularization}"
    seconds_per_iteration = plan["selected"]["seconds_per_iteration_by_cell"][key]
    return plan["safety_factor"] * seconds_per_iteration * plan["selected"]["max_iters"] / 3600.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Durable, resume-aware Kaggle study runner")
    parser.add_argument("--stage", choices=("m2b", "stage_a", "stage_b"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--quota-state", type=Path, default=Path("/kaggle/working/quota_state.json"))
    args = parser.parse_args()
    handle = os.environ.get("PINN_RESULTS_DATASET")
    if not handle:
        raise RuntimeError("PINN_RESULTS_DATASET must name the private durable results Dataset")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    executable_actions = {
        "run_m2b",
        "run_m2b_reduced_points",
        "run_m2b_stage_b_dropped",
        "run_m2b_stage_b_dropped_no_regularization",
    }
    if plan["action"] not in executable_actions:
        raise RuntimeError(f"plan is not executable: action={plan['action']}")
    publisher = KagglePublisher(handle)
    database = Path("/kaggle/working/results/results.sqlite")
    backup = Path("/kaggle/working/results/results.backup.sqlite")
    publisher.restore(database, backup, args.quota_state)
    base = apply_plan(ExperimentConfig.from_json(args.config), plan, args.stage)
    quota = QuotaState.load(args.quota_state)
    selected = plan["selected"]
    quota.stage_a_limit_hours = max(
        quota.stage_a_limit_hours, float(selected.get("stage_a_budget_hours", 2.0))
    )
    stage_b_dropped = "stage_b_dropped" in plan["action"]
    budget_stage = "stage_a" if args.stage == "m2b" and stage_b_dropped else (
        "calibration" if args.stage == "m2b" else args.stage
    )
    regularizations = tuple(selected.get("regularizations", ("none", "double_backprop")))
    queue = list(stage_conditions(args.stage, regularizations))
    queue.sort(key=lambda item: (estimated_hours(plan, item[0], item[1], item[2]), item))
    for backbone, precision, regularization, seed in queue:
        estimate = estimated_hours(plan, backbone, precision, regularization)
        if not quota.can_start(budget_stage, estimate):
            raise RuntimeError(
                f"quota guard stopped before {backbone}/{precision}/{regularization}/seed={seed}; "
                f"conservative estimate {estimate:.4f} h does not fit"
            )
        config = base.with_condition(
            backbone=backbone,
            precision=precision,
            regularization=regularization,
            seed=seed,
        )
        started = time.perf_counter()
        result = run(config)
        actual_hours = (time.perf_counter() - started) / 3600.0
        if result["status"] != "skipped":
            quota.record(budget_stage, actual_hours)
            quota.save(args.quota_state)
            note = (
                f"{args.stage}: {backbone}/{precision}/{regularization}/seed={seed}; "
                f"status={result['status']}"
            )
            publisher.publish(
                database_backup=config.persistence.backup,
                quota_state=args.quota_state,
                note=note,
                extra_files=(args.plan,),
            )


if __name__ == "__main__":
    main()
