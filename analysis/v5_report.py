"""Wertet die V5-Matrix (Reaction, rho = 5.25) nach PREREGISTRATION-V5.md aus.

Geschrieben vor der Sichtung der Matrix-Kennzahlen; wendet Abschnitt 6 mechanisch
an: Erfolgsquote (rel. L2 < 0.10, Wilson-95-%-KI) und Median log10(rel. L2)
(Bootstrap-Perzentil-KI, 95 %, 10 000 Resamples) je Zelle, dazu die
Backbone-Kontraste aus Abschnitt 3. Es trifft keine Auswahl.

    python -m analysis.v5_report --database results/results.sqlite --study-id <id>
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from itertools import product
from pathlib import Path

import numpy as np

SUCCESS_THRESHOLD = 0.10  # Abschnitt 6, fixiert vor Daten
BOOTSTRAP_RESAMPLES = 10_000
TERMINAL = ("completed", "numerical_fail")
BACKBONES = ("mlp", "grand", "gread")
PRECISIONS = ("fp32", "fp64")
REGULARIZATIONS = ("none", "double_backprop")
SEEDS = (0, 1, 2, 3, 4)


def load_rows(database: Path, study_id: str) -> list[dict]:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    query = """
        select t.study_id, t.backbone, t.precision, t.regularization, t.seed,
               r.attempt_no, r.status, r.relative_l2, r.relative_l2_censored,
               r.success, r.wall_seconds, r.iterations, r.function_evaluations,
               r.solver_nfe, r.parameter_count, r.error_type, r.git_commit,
               r.runtime_json
        from runs r join trials t on t.trial_key = r.trial_key
        where t.study_id = ?
        order by t.backbone, t.precision, t.regularization, t.seed, r.attempt_no
    """
    rows = [dict(row) for row in connection.execute(query, (study_id,))]
    connection.close()
    return rows


def terminal_rows(rows: list[dict]) -> dict[tuple, dict]:
    """Letzter terminaler Attempt je Bedingung (Backbone, Praezision, Reg., Seed)."""
    latest: dict[tuple, dict] = {}
    for row in rows:
        if row["status"] not in TERMINAL:
            continue
        key = (row["backbone"], row["precision"], row["regularization"], row["seed"])
        previous = latest.get(key)
        if previous is None or row["attempt_no"] > previous["attempt_no"]:
            latest[key] = row
    return latest


def wilson(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_median_ci(values: np.ndarray, seed: int = 0) -> tuple[float, float]:
    if values.size == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(BOOTSTRAP_RESAMPLES, values.size), replace=True)
    medians = np.median(samples, axis=1)
    return (float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5)))


def summarise_cell(cell_rows: list[dict]) -> dict:
    l2 = np.array([r["relative_l2"] for r in cell_rows], dtype=float)
    log10 = np.log10(np.clip(l2, 1e-8, 10.0))
    successes = int(sum(1 for v in l2 if math.isfinite(v) and v < SUCCESS_THRESHOLD))
    n = len(cell_rows)
    return {
        "n": n,
        "seeds": [r["seed"] for r in cell_rows],
        "relative_l2": [float(v) for v in l2],
        "successes": successes,
        "success_rate": successes / n if n else math.nan,
        "success_rate_wilson95": wilson(successes, n),
        "median_log10_rel_l2": float(np.median(log10)) if n else math.nan,
        "median_log10_bootstrap95": bootstrap_median_ci(log10),
        "censored": int(sum(1 for r in cell_rows if r["relative_l2_censored"])),
        "numerical_fail": int(sum(1 for r in cell_rows if r["status"] == "numerical_fail")),
        "wall_seconds_median": float(np.median([r["wall_seconds"] or 0.0 for r in cell_rows])),
        "function_evaluations_median": float(
            np.median([r["function_evaluations"] or 0 for r in cell_rows])
        ),
    }


def summarise_backbone(terminal: dict[tuple, dict], backbone: str) -> dict:
    """Erfolgsquote ueber alle 20 Laeufe eines Backbones (4 Zellen x 5 Seeds)."""
    rows = [r for key, r in terminal.items() if key[0] == backbone]
    return summarise_cell(rows) if rows else {"n": 0}


def report(rows: list[dict]) -> dict:
    terminal = terminal_rows(rows)
    cells = {}
    for backbone, precision, regularization in product(BACKBONES, PRECISIONS, REGULARIZATIONS):
        cell_rows = [
            terminal[(backbone, precision, regularization, seed)]
            for seed in SEEDS
            if (backbone, precision, regularization, seed) in terminal
        ]
        cells[f"{backbone}/{precision}/{regularization}"] = summarise_cell(cell_rows)

    baseline = cells["mlp/fp32/none"]
    baseline_gate = {
        "rule": "Abschnitt 10: |Erfolge - 2| <= 2 gegenueber lokal 2/5",
        "successes": baseline["successes"],
        "n": baseline["n"],
        "passed": baseline["n"] == 5 and abs(baseline["successes"] - 2) <= 2,
    }

    per_backbone = {b: summarise_backbone(terminal, b) for b in BACKBONES}
    commits = sorted({r["git_commit"] for r in terminal.values()})
    gpus = sorted(
        {
            json.loads(r["runtime_json"] or "{}").get("gpu_name", "unknown")
            for r in terminal.values()
        }
    )
    non_terminal = [r for r in rows if r["status"] not in TERMINAL]
    return {
        "study_id": rows[0]["study_id"] if rows else None,
        "success_threshold": SUCCESS_THRESHOLD,
        "terminal_runs": len(terminal),
        "expected_runs": len(BACKBONES) * len(PRECISIONS) * len(REGULARIZATIONS) * len(SEEDS),
        "non_terminal_attempts": len(non_terminal),
        "infra_fail_attempts": sum(1 for r in rows if r["status"] == "infra_fail"),
        "source_commits": commits,
        "gpus": gpus,
        "baseline_gate": baseline_gate,
        "cells": cells,
        "per_backbone": per_backbone,
    }


def format_table(result: dict) -> str:
    lines = [
        f"Studie: {result['study_id']}  terminale Laeufe: {result['terminal_runs']}/"
        f"{result['expected_runs']}  infra_fail-Versuche: {result['infra_fail_attempts']}",
        f"Commits: {result['source_commits']}  GPUs: {result['gpus']}",
        "",
        f"{'Zelle':<28}{'n':>3}{'Erfolge':>9}{'Quote':>7}{'Wilson95':>18}"
        f"{'med log10':>11}{'Boot95':>20}{'Wall(s)':>9}",
    ]
    for name, c in result["cells"].items():
        if c["n"] == 0:
            lines.append(f"{name:<28}{0:>3}   (keine terminalen Laeufe)")
            continue
        lo, hi = c["success_rate_wilson95"]
        blo, bhi = c["median_log10_bootstrap95"]
        lines.append(
            f"{name:<28}{c['n']:>3}{c['successes']:>9}{c['success_rate']:>7.2f}"
            f"{f'[{lo:.2f}, {hi:.2f}]':>18}{c['median_log10_rel_l2']:>11.3f}"
            f"{f'[{blo:.2f}, {bhi:.2f}]':>20}{c['wall_seconds_median']:>9.0f}"
        )
    lines.append("")
    for b, s in result["per_backbone"].items():
        if s["n"]:
            lo, hi = s["success_rate_wilson95"]
            lines.append(
                f"{b:<8} gesamt: {s['successes']}/{s['n']} = {s['success_rate']:.2f} "
                f"[{lo:.2f}, {hi:.2f}]  med log10 {s['median_log10_rel_l2']:.3f}"
            )
    g = result["baseline_gate"]
    lines.append("")
    lines.append(
        f"[{'PASS' if g['passed'] else 'FAIL'}] Baseline-Gate: {g['successes']}/{g['n']} "
        f"Erfolge ({g['rule']})"
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("results/results.sqlite"))
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    rows = load_rows(args.database, args.study_id)
    if not rows:
        raise SystemExit(f"keine Zeilen fuer study_id {args.study_id!r} in {args.database}")
    result = report(rows)
    print(format_table(result))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\ngeschrieben: {args.json}")


if __name__ == "__main__":
    main()
