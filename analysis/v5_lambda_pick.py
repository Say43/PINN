"""Waehlt lambda_r nach PREREGISTRATION-V5.md Abschnitt 5.

Regel nach der Revision D-13: MLP/FP32/double_backprop auf den Seeds 3 und 4,
also denen, auf denen die unregularisierte Baseline gelingt. Gewaehlt wird der
Kandidat mit dem niedrigsten Median des relativen L2 ueber diese Seeds; er gilt
danach unveraendert fuer alle Bedingungen. Bei exakter Gleichheit entscheidet der
kleinere Wert von lambda_r; dieser Zusatz ist Determinismus, keine inhaltliche
Regel. Kein Graph-Backbone geht in die Auswahl ein.

    python -m analysis.v5_lambda_pick --database results/results.sqlite
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from statistics import median

from bench.v5 import LAMBDA_CANDIDATES, LAMBDA_SELECTION_SEEDS

SELECTION_STUDY_ID = "reaction_v5_lambda_selection_seeds34_NOT_STUDY_DATA"
TERMINAL = ("completed", "numerical_fail")


def load_candidates(database: Path, study_id: str) -> list[dict]:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    query = """
        select t.config_json, t.seed, r.attempt_no, r.status, r.relative_l2,
               r.relative_l2_censored, r.wall_seconds, r.iterations,
               r.function_evaluations, r.git_commit
        from runs r join trials t on t.trial_key = r.trial_key
        where t.study_id = ? and t.backbone = 'mlp' and t.precision = 'fp32'
          and t.regularization = 'double_backprop'
        order by r.attempt_no
    """
    rows = [dict(row) for row in connection.execute(query, (study_id,))]
    connection.close()

    latest: dict[tuple[float, int], dict] = {}
    for row in rows:
        if row["status"] not in TERMINAL:
            continue
        lambda_r = float(json.loads(row["config_json"])["train"]["lambda_r"])
        row["lambda_r"] = lambda_r
        key = (lambda_r, row["seed"])
        previous = latest.get(key)
        if previous is None or row["attempt_no"] > previous["attempt_no"]:
            latest[key] = row
    return [latest[key] for key in sorted(latest)]


def pick(candidates: list[dict]) -> dict:
    expected = {
        (value, seed) for value in LAMBDA_CANDIDATES for seed in LAMBDA_SELECTION_SEEDS
    }
    missing = sorted(expected - {(row["lambda_r"], row["seed"]) for row in candidates})
    if missing:
        raise SystemExit(f"lambda selection incomplete; missing (lambda_r, seed): {missing}")

    summary = []
    for value in LAMBDA_CANDIDATES:
        runs = [row for row in candidates if row["lambda_r"] == value]
        summary.append(
            {
                "lambda_r": value,
                "median_relative_l2": median(row["relative_l2"] for row in runs),
                "runs": [
                    {
                        "seed": row["seed"],
                        "relative_l2": row["relative_l2"],
                        "censored": bool(row["relative_l2_censored"]),
                        "status": row["status"],
                        "wall_seconds": row["wall_seconds"],
                        "iterations": row["iterations"],
                        "function_evaluations": row["function_evaluations"],
                        "git_commit": row["git_commit"],
                    }
                    for row in sorted(runs, key=lambda row: row["seed"])
                ],
            }
        )
    best = min(summary, key=lambda row: (row["median_relative_l2"], row["lambda_r"]))
    return {
        "rule": (
            "PREREGISTRATION-V5.md Abschnitt 5 in der Fassung D-13: niedrigster Median "
            f"des relativen L2 ueber die Seeds {list(LAMBDA_SELECTION_SEEDS)}"
        ),
        "study_id": SELECTION_STUDY_ID,
        "selection_seeds": list(LAMBDA_SELECTION_SEEDS),
        "selected_lambda_r": best["lambda_r"],
        "candidates": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("results/results.sqlite"))
    parser.add_argument("--study-id", default=SELECTION_STUDY_ID)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    candidates = load_candidates(args.database, args.study_id)
    result = pick(candidates)
    for row in result["candidates"]:
        mark = "  <-- gewaehlt" if row["lambda_r"] == result["selected_lambda_r"] else ""
        per_seed = "  ".join(
            f"Seed {run['seed']}: {run['relative_l2']:.6f} "
            f"({run['wall_seconds']:.0f} s, {run['function_evaluations']} Auswertungen)"
            for run in row["runs"]
        )
        print(
            f"lambda_r={row['lambda_r']:<8g} Median={row['median_relative_l2']:.6f}  "
            f"{per_seed}{mark}"
        )
    print(f"\ngewaehlt: lambda_r = {result['selected_lambda_r']:g}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"geschrieben: {args.json}")


if __name__ == "__main__":
    main()
