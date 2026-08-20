from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


BACKBONES = ("mlp", "grand", "gread")
PRECISIONS = ("fp32", "fp64")
REGULARIZATIONS = ("none", "double_backprop")
SEEDS = (0, 1, 2, 3, 4)
SAFETY_FACTOR = 1.25
DOUBLE_BACKPROP_FACTOR = 2.0
STAGE_A_SECONDS = 2.0 * 3600.0


def load_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def summarize(rows: list[dict]) -> dict[tuple[str, str, str, str], float]:
    grouped: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        key = (row["hardware"], row["point_scheme"], row["backbone"], row["precision"])
        grouped[key].append(float(row["seconds_per_iteration"]))
    medians = {}
    for key, values in grouped.items():
        if len(values) < 3:
            continue
        medians[key] = statistics.median(values)
    return medians


def projected_cell_times(
    medians: dict[tuple[str, str, str, str], float],
    *,
    hardware: str,
    point_scheme: str,
) -> dict[str, float] | None:
    required = [
        (hardware, point_scheme, backbone, precision)
        for backbone in ("mlp", "gread")
        for precision in PRECISIONS
    ]
    if any(key not in medians for key in required):
        return None
    cells = {}
    for backbone in BACKBONES:
        proxy = "mlp" if backbone == "mlp" else "gread"
        for precision in PRECISIONS:
            base = medians[(hardware, point_scheme, proxy, precision)]
            for regularization in REGULARIZATIONS:
                factor = DOUBLE_BACKPROP_FACTOR if regularization == "double_backprop" else 1.0
                cells[f"{backbone}:{precision}:{regularization}"] = base * factor
    return cells


def plan_hardware(
    medians: dict[tuple[str, str, str, str], float],
    *,
    hardware: str,
    point_scheme: str,
) -> dict | None:
    cells = projected_cell_times(medians, hardware=hardware, point_scheme=point_scheme)
    if cells is None:
        return None
    # One active run is a hard durability invariant: a session kill may destroy
    # at most the current run. Two concurrent T4 workers would violate it.
    workers = 1
    seconds_per_seed_layer = sum(cells.values()) / workers
    raw_iters = STAGE_A_SECONDS / (SAFETY_FACTOR * len(SEEDS) * seconds_per_seed_layer)
    max_iters = int(math.floor(raw_iters / 1000.0) * 1000)
    projected_hours = (
        SAFETY_FACTOR * len(SEEDS) * sum(cells.values()) * max(max_iters, 0) / workers / 3600.0
    )
    return {
        "hardware": hardware,
        "workers": workers,
        "point_scheme": point_scheme,
        "max_iters": max_iters,
        "projected_stage_a_hours": projected_hours,
        "seconds_per_iteration_by_cell": cells,
    }


def make_plan(rows: list[dict]) -> dict:
    medians = summarize(rows)
    candidates = []
    for scheme in ("full", "reduced"):
        for hardware in ("p100", "2xt4"):
            candidate = plan_hardware(medians, hardware=hardware, point_scheme=scheme)
            if candidate is not None:
                candidates.append(candidate)
    full = [item for item in candidates if item["point_scheme"] == "full"]
    viable_full = [item for item in full if item["max_iters"] >= 5000]
    if viable_full:
        selected = max(viable_full, key=lambda item: item["max_iters"])
        action = "run_m2b"
    else:
        reduced = [item for item in candidates if item["point_scheme"] == "reduced"]
        viable_reduced = [item for item in reduced if item["max_iters"] >= 5000]
        if viable_reduced:
            selected = max(viable_reduced, key=lambda item: item["max_iters"])
            action = "run_m2b_reduced_points"
        elif not reduced:
            selected = max(full, key=lambda item: item["max_iters"]) if full else None
            action = "rerun_m2a_with_reduced_points"
        else:
            selected = max(reduced, key=lambda item: item["max_iters"])
            action = "drop_stage_b_and_replan_stage_a"
    return {
        "schema_version": 1,
        "safety_factor": SAFETY_FACTOR,
        "double_backprop_projection_factor": DOUBLE_BACKPROP_FACTOR,
        "minimum_max_iters": 5000,
        "action": action,
        "selected": selected,
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Choose Kaggle hardware and max_iters from M2a")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/m2_plan.json"))
    args = parser.parse_args()
    plan = make_plan(load_rows(args.inputs))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
