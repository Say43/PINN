"""Prueft die vorab in docs/decision-gate-m2b.md fixierten Manipulation Checks.

Das Skript wird VOR der Sichtung der M2b-Kennzahlen geschrieben und wendet die
dort festgelegten Kriterien mechanisch an. Es trifft keine Auswahl und keine
nachtraegliche Interpretation.

    python -m analysis.gate_check --database results/results.sqlite
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

# Aus docs/decision-gate-m2b.md, Abschnitt 4. Nicht hier aendern.
FAILURE_MODE_FLOOR = 0.5  # PC-1: mlp/fp32/none muss darueber liegen
SUCCESS_THRESHOLD = 0.10  # PC-2: mindestens eine Zelle muss darunter liegen
EXPECTED_CELLS = 6

TERMINAL = ("completed", "numerical_fail")


def load_rows(database: Path) -> list[dict]:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    query = """
        select t.pde, t.backbone, t.precision, t.regularization, t.seed,
               r.attempt_no, r.status, r.relative_l2, r.relative_l2_censored,
               r.wall_seconds, r.iterations, r.function_evaluations,
               r.solver_nfe, r.parameter_count, r.error_type, r.error_message
        from runs r join trials t on t.trial_key = r.trial_key
        order by t.backbone, t.precision, t.regularization, t.seed, r.attempt_no
    """
    rows = [dict(row) for row in connection.execute(query)]
    connection.close()
    return rows


def terminal_rows(rows: list[dict]) -> list[dict]:
    """Letzter terminaler Attempt je Zelle. Nicht-terminale werden ignoriert."""
    latest: dict[tuple, dict] = {}
    for row in rows:
        if row["status"] not in TERMINAL:
            continue
        key = (row["backbone"], row["precision"], row["regularization"], row["seed"])
        previous = latest.get(key)
        if previous is None or row["attempt_no"] > previous["attempt_no"]:
            latest[key] = row
    return list(latest.values())


def check(rows: list[dict]) -> dict:
    terminal = terminal_rows(rows)
    non_terminal = [r for r in rows if r["status"] not in TERMINAL]

    baseline = [
        r for r in terminal
        if r["backbone"] == "mlp" and r["precision"] == "fp32"
        and r["regularization"] == "none"
    ]
    errors = [r for r in terminal if r["error_type"]]

    pc1_value = baseline[0]["relative_l2_censored"] if baseline else None
    pc1 = {
        "name": "PC-1 Failure Mode reproduziert",
        "rule": f"mlp/fp32/none relativer L2 > {FAILURE_MODE_FLOOR}",
        "value": pc1_value,
        "passed": pc1_value is not None and pc1_value > FAILURE_MODE_FLOOR,
    }

    values = [r["relative_l2_censored"] for r in terminal if r["relative_l2_censored"] is not None]
    best = min(values) if values else None
    pc2 = {
        "name": "PC-2 Aufloesungsvermoegen",
        "rule": f"mindestens eine Zelle mit relativem L2 < {SUCCESS_THRESHOLD}",
        "value": best,
        "passed": best is not None and best < SUCCESS_THRESHOLD,
    }

    pc3 = {
        "name": "PC-3 Numerische Integritaet",
        "rule": "keine unerklaerten NaN/Inf- oder Infrastrukturabbrueche",
        "value": [f"{r['backbone']}/{r['precision']}: {r['error_type']}" for r in errors],
        "passed": not errors,
    }

    complete = {
        "name": "Vollstaendigkeit",
        "rule": f"{EXPECTED_CELLS} terminale Zellen erforderlich",
        "value": len(terminal),
        "passed": len(terminal) >= EXPECTED_CELLS,
    }

    checks = [complete, pc1, pc2, pc3]
    return {
        "checks": checks,
        "all_passed": all(c["passed"] for c in checks),
        "terminal_cells": len(terminal),
        "non_terminal_cells": len(non_terminal),
        "decision": (
            "Zusatzbudget fuer Option B freigeben"
            if all(c["passed"] for c in checks)
            else "Kein Zusatzbudget. Grenzen berichten."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("results/results.sqlite"))
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    rows = load_rows(args.database)
    result = check(rows)

    print(f"Terminale Zellen: {result['terminal_cells']}"
          f"  nicht-terminal: {result['non_terminal_cells']}\n")
    for entry in result["checks"]:
        mark = "PASS" if entry["passed"] else "FAIL"
        print(f"[{mark}] {entry['name']}")
        print(f"       Regel:   {entry['rule']}")
        print(f"       Messung: {entry['value']}")
    print(f"\nErgebnis: {result['decision']}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"geschrieben: {args.json}")


if __name__ == "__main__":
    main()
