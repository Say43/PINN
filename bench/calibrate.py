from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from bench.budget import QuotaState
from kaggle.publish import KagglePublisher
from src.config import ExperimentConfig
from src.train import train_once


CALIBRATION_BACKBONES = ("mlp", "gread")
CALIBRATION_PRECISIONS = ("fp32", "fp64")


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def calibration_config(
    base: ExperimentConfig,
    *,
    backbone: str,
    precision: str,
    repeat: int,
    device: str,
    point_scheme: str,
) -> ExperimentConfig:
    raw = base.to_dict()
    if point_scheme == "reduced":
        raw["pde"].update(domain_points=196, boundary_points=100, initial_points=100)
    raw["model"]["backbone"] = backbone
    raw["train"].update(
        precision=precision,
        regularization="none",
        seed=10_000 + repeat,
        max_iters=300,
        log_every=300,
        device=device,
    )
    raw["persistence"]["study_id"] = "m2a_throwaway"
    return ExperimentConfig.from_dict(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="M2a target-hardware timing probe")
    parser.add_argument("--config", type=Path, default=Path("configs/stage_a.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hardware-label", choices=("p100", "2xt4"), required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--point-scheme", choices=("full", "reduced"), default="full")
    parser.add_argument("--results-dataset", help="private owner/slug for per-repeat persistence")
    parser.add_argument(
        "--quota-state", type=Path, default=Path("/kaggle/working/quota_state.json")
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("M2a requires a Kaggle GPU")
    visible = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    if args.hardware_label == "p100" and not any("P100" in name for name in visible):
        raise RuntimeError(f"hardware label p100 does not match visible GPUs: {visible}")
    if args.hardware_label == "2xt4" and (
        len(visible) < 2 or not all("T4" in name for name in visible[:2])
    ):
        raise RuntimeError(f"hardware label 2xt4 does not match visible GPUs: {visible}")
    publisher = KagglePublisher(args.results_dataset) if args.results_dataset else None
    if publisher is not None:
        publisher.restore(args.output, args.quota_state)
    base = ExperimentConfig.from_json(args.config)
    quota = QuotaState.load(args.quota_state)
    observed_hours: list[float] = []
    completed = set()
    if args.output.exists():
        with args.output.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    completed.add((row["backbone"], row["precision"], row["repeat"]))
                    observed_hours.append(float(row["wall_seconds"]) / 3600.0)
    for backbone in CALIBRATION_BACKBONES:
        for precision in CALIBRATION_PRECISIONS:
            for repeat in range(3):
                if (backbone, precision, repeat) in completed:
                    continue
                config = calibration_config(
                    base,
                    backbone=backbone,
                    precision=precision,
                    repeat=repeat,
                    device=args.device,
                    point_scheme=args.point_scheme,
                )
                conservative_hours = max([0.05, *(1.25 * value for value in observed_hours)])
                if not quota.can_start("calibration", conservative_hours):
                    raise RuntimeError(
                        "calibration quota guard stopped before the next probe; "
                        f"estimate={conservative_hours:.4f} h"
                    )
                started = time.perf_counter()
                result = train_once(config)
                elapsed = time.perf_counter() - started
                elapsed_hours = elapsed / 3600.0
                observed_hours.append(elapsed_hours)
                row = {
                    "schema_version": 1,
                    "stage": "m2a",
                    "throwaway": True,
                    "hardware": args.hardware_label,
                    "workers": args.workers,
                    "point_scheme": args.point_scheme,
                    "backbone": backbone,
                    "precision": precision,
                    "regularization": "none",
                    "repeat": repeat,
                    "iterations": 300,
                    "wall_seconds": elapsed,
                    "seconds_per_iteration": elapsed / 300.0,
                    "function_evaluations": result.values["function_evaluations"],
                    "parameter_count": result.values["parameter_count"],
                    "visible_gpus": visible,
                }
                append_jsonl(args.output, row)
                quota.record("calibration", elapsed_hours)
                quota.save(args.quota_state)
                if publisher is not None:
                    publisher.publish_files(
                        files=(args.output, args.quota_state),
                        note=(
                            f"M2a {args.hardware_label}/{args.point_scheme}: "
                            f"{backbone}/{precision}/repeat={repeat}"
                        ),
                    )
                print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
