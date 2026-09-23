"""Trennt die gescheiterten V5-Laeufe nach Versagensart.

EXPLORATIV, nicht praeregistriert: geschrieben nach der Sichtung der Matrixdaten.
Die Einteilung nutzt nur den finalen Trainings-Loss ohne Regularisierungsterm
(`loss_base`) und die praeregistrierte Erfolgsschwelle:

- `success`: relativer L2 < 0.10.
- `stalled`: Optimierer friert ein; `loss_base` bleibt ueber 1e-2, die
  Anfangsbedingung wird nie gelernt.
- `residual_satisfied_wrong`: `loss_base` unter 1e-3, Residuum, Rand- und
  Anfangsbedingung an den Kollokationspunkten erfuellt, die Loesung auf dem
  Auswertungsgitter trotzdem falsch.

In V5 liegen die gemessenen Werte bei rund 2e-1 und hoechstens 7e-5; jede Schwelle
dazwischen ergibt dieselbe Einteilung.

    python -m analysis.v5_failure_modes --study-id reaction_v5_rho525_20260922
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

SUCCESS_THRESHOLD = 0.10
STALLED_LOSS = 1.0e-2
SATISFIED_LOSS = 1.0e-3
BACKBONES = ("mlp", "grand", "gread")
KINDS = ("success", "stalled", "residual_satisfied_wrong", "other")


def classify(relative_l2: float, final_metrics: dict) -> str:
    if relative_l2 < SUCCESS_THRESHOLD:
        return "success"
    loss = final_metrics["loss_base"]
    if loss > STALLED_LOSS:
        return "stalled"
    if loss < SATISFIED_LOSS:
        return "residual_satisfied_wrong"
    return "other"


def load(database: Path, study_id: str) -> list[dict]:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    rows = [
        dict(row)
        for row in connection.execute(
            """
            select t.backbone, t.precision, t.regularization, t.seed, r.attempt_no,
                   r.status, r.relative_l2, r.function_evaluations, r.final_metrics_json
            from runs r join trials t on t.trial_key = r.trial_key
            where t.study_id = ? and r.status in ('completed', 'numerical_fail')
            order by r.attempt_no
            """,
            (study_id,),
        )
    ]
    connection.close()
    latest: dict[tuple, dict] = {}
    for row in rows:
        latest[(row["backbone"], row["precision"], row["regularization"], row["seed"])] = row
    return list(latest.values())


def summarise(rows: list[dict]) -> dict:
    per_backbone = {backbone: Counter() for backbone in BACKBONES}
    runs = []
    for row in rows:
        metrics = json.loads(row["final_metrics_json"] or "{}")
        kind = classify(row["relative_l2"], metrics)
        per_backbone[row["backbone"]][kind] += 1
        runs.append(
            {
                "backbone": row["backbone"],
                "precision": row["precision"],
                "regularization": row["regularization"],
                "seed": row["seed"],
                "relative_l2": row["relative_l2"],
                "kind": kind,
                "loss_base": metrics.get("loss_base"),
                "loss_initial": metrics.get("loss_initial"),
                "loss_domain": metrics.get("loss_domain"),
                "function_evaluations": row["function_evaluations"],
            }
        )
    failures = [run for run in runs if run["kind"] != "success"]
    return {
        "note": "EXPLORATORY_NOT_PREREGISTERED; written after the matrix data were seen",
        "thresholds": {
            "success_relative_l2": SUCCESS_THRESHOLD,
            "stalled_loss_base_above": STALLED_LOSS,
            "residual_satisfied_loss_base_below": SATISFIED_LOSS,
        },
        "per_backbone": {b: {k: per_backbone[b][k] for k in KINDS} for b in BACKBONES},
        "failure_loss_base_range": {
            kind: [
                min(run["loss_base"] for run in failures if run["kind"] == kind),
                max(run["loss_base"] for run in failures if run["kind"] == kind),
            ]
            for kind in ("stalled", "residual_satisfied_wrong")
            if any(run["kind"] == kind for run in failures)
        },
        "runs": sorted(
            runs, key=lambda run: (run["backbone"], run["precision"], run["regularization"], run["seed"])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("results/results.sqlite"))
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    result = summarise(load(args.database, args.study_id))
    print(f"{'Backbone':<8}" + "".join(f"{kind:>28}" for kind in KINDS))
    for backbone, counts in result["per_backbone"].items():
        print(f"{backbone:<8}" + "".join(f"{counts[kind]:>28}" for kind in KINDS))
    print("\nloss_base je Versagensart (min, max):", result["failure_loss_base_range"])
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"geschrieben: {args.json}")


if __name__ == "__main__":
    main()
