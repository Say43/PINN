"""Nine bounded CPU smoke runs on fresh seeds; no study or convergence claims."""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--one", nargs=2, metavar=("BACKBONE", "SEED"))
    parser.add_argument("--out", type=Path, default=Path("results/local_comparison.json"))
    args = parser.parse_args()
    if args.one:
        import torch
        from src.config import ExperimentConfig
        from src.train import train_once
        torch.set_num_threads(2)
        config = ExperimentConfig.from_json("configs/reaction_v5.json")
        config = replace(
            config,
            model=replace(config.model, backbone=args.one[0]),
            train=replace(config.train, precision="fp32", regularization="none",
                          seed=int(args.one[1]), device="cpu", max_iters=50, log_every=50),
            persistence=replace(config.persistence, study_id="local_validation_NOT_STUDY_DATA"),
        )
        result = train_once(config)
        payload = {"status": result.status, "relative_l2": result.values["relative_l2"],
                   "initial_relative_l2": result.values["history"][0]["relative_l2"],
                   "iterations": result.values["iterations"], "wall_seconds": result.values["wall_seconds"],
                   "final_metrics": result.values["final_metrics"],
                   "function_evaluations": result.values["function_evaluations"]}
        print(json.dumps(payload), flush=True)
        return
    gate = json.loads(Path("results/local_validation.json").read_text(encoding="utf-8"))
    if not gate["ready_for_training_comparison"]:
        raise SystemExit("Validity gate failed; no training comparison")
    started = time.perf_counter()
    payload = {"note": "LOCAL_CPU_NOT_STUDY_DATA; 50 iterations, not the 2000-iteration V5 matrix",
               "seeds": [101, 102, 103], "precision": "fp32", "regularization": "none",
               "max_iters": 50, "timeout_per_run_seconds": 50,
               "conclusions": "numerical smoke and cost check only; no ranking or success-rate inference",
               "rows": []}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for seed in payload["seeds"]:
        for backbone in ("mlp", "grand", "gread"):
            row = {"backbone": backbone, "seed": seed}
            timeout = min(50., 540. - (time.perf_counter() - started))
            if timeout <= 0:
                row["status"] = "global_budget_not_started"
            else:
                try:
                    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2")
                    child = subprocess.run([sys.executable, "-B", "-m", "analysis.compare_local",
                                            "--one", backbone, str(seed)], capture_output=True,
                                           text=True, encoding="utf-8", timeout=timeout, env=env)
                    if child.returncode:
                        row.update(status="infra_fail", error=child.stderr[-2000:])
                    else:
                        row.update(json.loads(child.stdout))
                except subprocess.TimeoutExpired:
                    row["status"] = "diagnostic_timeout"
            payload["rows"].append(row)
            payload["wall_seconds"] = time.perf_counter() - started
            args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
