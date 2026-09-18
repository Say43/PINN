"""Bounded V5 GPU cost probe; never writes study data."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import torch

from src.config import ExperimentConfig
from src.train import train_once


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--out", type=Path, default=Path("/kaggle/working/v5_profile.json"))
    args = parser.parse_args()
    if not 1 <= args.iters <= 50:
        raise ValueError("profile iterations must be between 1 and 50")
    if not torch.cuda.is_available():
        raise RuntimeError("V5 GPU profile requires CUDA")
    torch.set_num_threads(2)
    base = ExperimentConfig.from_json("configs/reaction_v5.json")
    cases = (("mlp", "fp32", "double_backprop"),
             ("grand", "fp32", "none"),
             ("gread", "fp32", "none"),
             ("gread", "fp64", "double_backprop"))
    payload = {"note": "V5_GPU_PROFILE_NOT_STUDY_DATA", "iterations_per_case": args.iters,
               "rows": []}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for backbone, precision, regularization in cases:
        config = replace(
            base,
            model=replace(base.model, backbone=backbone),
            train=replace(base.train, precision=precision, regularization=regularization,
                          device="cuda:0", max_iters=args.iters, log_every=args.iters),
            persistence=replace(base.persistence, study_id="gpu_profile_NOT_STUDY_DATA"),
        )
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        try:
            result = train_once(config)
            row = {
                "backbone": backbone, "precision": precision,
                "regularization": regularization, "status": result.status,
                "wall_seconds": time.perf_counter() - started,
                "trainer_wall_seconds": result.values["wall_seconds"],
                "iterations": result.values["iterations"],
                "function_evaluations": result.values["function_evaluations"],
                "relative_l2": result.values["relative_l2"],
                "peak_memory_bytes": torch.cuda.max_memory_allocated(),
            }
        except Exception as error:
            row = {"backbone": backbone, "precision": precision,
                   "regularization": regularization, "status": "infra_fail",
                   "error": type(error).__name__, "message": str(error),
                   "wall_seconds": time.perf_counter() - started}
        finally:
            torch.cuda.empty_cache()
        payload["rows"].append(row)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
