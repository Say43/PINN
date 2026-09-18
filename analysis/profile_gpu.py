"""Short local GPU cost probe for V5; never writes study data."""
from __future__ import annotations

import json
import time
from dataclasses import replace

import torch

from src.config import ExperimentConfig
from src.train import train_once


def main() -> None:
    torch.set_num_threads(2)
    base = ExperimentConfig.from_json("configs/reaction_v5.json")
    for backbone in ("mlp", "grand", "gread"):
        config = replace(
            base,
            model=replace(base.model, backbone=backbone),
            train=replace(base.train, device="cuda:0", max_iters=3, log_every=3),
            persistence=replace(base.persistence, study_id="gpu_profile_NOT_STUDY_DATA"),
        )
        started = time.perf_counter()
        try:
            result = train_once(config)
            print(json.dumps({
                "backbone": backbone, "status": result.status,
                "wall_seconds": time.perf_counter() - started,
                "trainer_wall_seconds": result.values["wall_seconds"],
                "iterations": result.values["iterations"],
                "function_evaluations": result.values["function_evaluations"],
                "relative_l2": result.values["relative_l2"],
                "peak_memory_bytes": torch.cuda.max_memory_allocated(),
            }), flush=True)
        except Exception as error:
            print(json.dumps({"backbone": backbone, "error": type(error).__name__,
                              "message": str(error), "wall_seconds": time.perf_counter() - started}), flush=True)
        finally:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
