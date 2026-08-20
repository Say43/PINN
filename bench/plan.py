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
SAFETY_FACTOR = 1.0
DOUBLE_BACKPROP_FACTOR = 2.0
STAGE_A_SECONDS = 2.0 * 3600.0
DROPPED_STAGE_B_SECONDS = 3.5 * 3600.0
MIN_TIMING_REPEATS = 2


def load_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def summarize(rows: list[dict]) -> dict[tuple[str, str, str, str], float]:
    grouped: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["hardware"], row["point_scheme"], row["backbone"], row["precision"])
        grouped[key].append(row)
    medians = {}
    for key, cell_rows in grouped.items():
        newest_schema = max(int(row.get("schema_version", 1)) for row in cell_rows)
        values = [
            float(row["seconds_per_iteration"])
            for row in cell_rows
            if int(row.get("schema_version", 1)) == newest_schema
        ]
        if len(values) < MIN_TIMING_REPEATS:
            continue
        medians[key] = statistics.median(values)
    return medians


def projected_cell_times(
    medians: dict[tuple[str, str, str, str], float],
    *,
    hardware: str,
    point_scheme: str,
    regularizations: tuple[str, ...] = REGULARIZATIONS,
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
            for regularization in regularizations:
                factor = DOUBLE_BACKPROP_FACTOR if regularization == "double_backprop" else 1.0
                cells[f"{backbone}:{precision}:{regularization}"] = base * factor
    return cells


def plan_hardware(
    medians: dict[tuple[str, str, str, str], float],
    *,
    hardware: str,
    point_scheme: str,
    stage_a_seconds: float = STAGE_A_SECONDS,
    regularizations: tuple[str, ...] = REGULARIZATIONS,
) -> dict | None:
    cells = projected_cell_times(
        medians,
        hardware=hardware,
        point_scheme=point_scheme,
        regularizations=regularizations,
    )
    if cells is None:
        return None
    # One active run is a hard durability invariant: a session kill may destroy
    # at most the current run. Two concurrent T4 workers would violate it.
    workers = 1
    seconds_per_seed_layer = sum(cells.values()) / workers
    raw_iters = stage_a_seconds / (SAFETY_FACTOR * len(SEEDS) * seconds_per_seed_layer)
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
        "stage_a_budget_hours": stage_a_seconds / 3600.0,
        "regularizations": list(regularizations),
        "seconds_per_iteration_by_cell": cells,
    }


def make_plan(rows: list[dict], *, stage_a_profile_hours: float = 0.0) -> dict:
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
            available_seconds = DROPPED_STAGE_B_SECONDS - stage_a_profile_hours * 3600.0
            if available_seconds <= 0:
                raise ValueError("stage-a profiling consumed the post-Stage-B budget")
            dropped_stage_b = []
            for hardware in ("p100", "2xt4"):
                candidate = plan_hardware(
                    medians,
                    hardware=hardware,
                    point_scheme="reduced",
                    stage_a_seconds=available_seconds,
                )
                if candidate is not None:
                    dropped_stage_b.append(candidate)
            candidates.extend(dropped_stage_b)
            viable_dropped_stage_b = [
                item for item in dropped_stage_b if item["max_iters"] >= 5000
            ]
            if viable_dropped_stage_b:
                selected = max(viable_dropped_stage_b, key=lambda item: item["max_iters"])
                action = "run_m2b_stage_b_dropped"
            else:
                no_regularization = []
                for hardware in ("p100", "2xt4"):
                    candidate = plan_hardware(
                        medians,
                        hardware=hardware,
                        point_scheme="reduced",
                        stage_a_seconds=available_seconds,
                        regularizations=("none",),
                    )
                    if candidate is not None:
                        no_regularization.append(candidate)
                candidates.extend(no_regularization)
                viable_no_regularization = [
                    item for item in no_regularization if item["max_iters"] >= 5000
                ]
                if viable_no_regularization:
                    selected = max(viable_no_regularization, key=lambda item: item["max_iters"])
                    action = "run_m2b_stage_b_dropped_no_regularization"
                else:
                    selected = (
                        max(no_regularization, key=lambda item: item["max_iters"])
                        if no_regularization else max(dropped_stage_b, key=lambda item: item["max_iters"])
                    )
                    action = "cannot_meet_minimum_iterations"
    return {
        "schema_version": 2,
        "safety_factor": SAFETY_FACTOR,
        "double_backprop_projection_factor": DOUBLE_BACKPROP_FACTOR,
        "minimum_max_iters": 5000,
        "minimum_timing_repeats": MIN_TIMING_REPEATS,
        "stage_a_profile_hours": stage_a_profile_hours,
        "action": action,
        "selected": selected,
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Choose Kaggle hardware and max_iters from M2a")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/m2_plan.json"))
    parser.add_argument("--stage-a-profile-hours", type=float, default=0.0)
    args = parser.parse_args()
    plan = make_plan(load_rows(args.inputs), stage_a_profile_hours=args.stage_a_profile_hours)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
