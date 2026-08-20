"""Budgetplanung fuer das korrigierte Setup, parametrisiert ueber die Iterationszahl.

Grundlage sind die gemessenen Wall-Clock-Zeiten des M2b-Slices (T4, ein Worker,
396 Kollokationspunkte, 6000 Iterationen). Diese Zeiten sind gueltig, auch wenn
der Slice als Architekturevidenz verworfen wurde — Aliasing verfaelscht die
Ergebnisse, nicht die Laufzeiten.

Kostenmodell, bewusst einfach gehalten:

    Zeit(N, P) = Trainingsanteil * (P / 396) * (N / 6000)
               + Auswertungsanteil * (N / 6000)

Der Trainingsanteil skaliert mit der Punktzahl (MLP linear, k-NN-Graph mit
O(N*k) ebenfalls linear). Der Auswertungsanteil ist die fixe 101x101-Auswertung
alle `log_every` Iterationen; sie haengt an der Iterationszahl, nicht an der
Kollokationspunktzahl. Die Trennung stammt aus dem gemessenen Faktor 1.263
zwischen M2a-Projektion (nur Optimizer-Schleife) und realer Laufzeit.

    python -m bench.replan --quota 27
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Gemessen im M2b-Slice bei 396 Punkten und 6000 Iterationen (Sekunden).
MEASURED_TOTAL = {
    "mlp:fp32": 325.9,
    "mlp:fp64": 253.5,
    "grand:fp32": 595.8,
    "grand:fp64": 530.0,
    "gread:fp32": 969.9,
    "gread:fp64": 589.7,
}

# M2a-Projektion derselben Zellen: reine Optimizer-Schleife, ohne Auswertungen.
PROJECTED_TRAINING = {
    "mlp:fp32": 215.2,
    "mlp:fp64": 220.4,
    "grand:fp32": 475.5,
    "grand:fp64": 466.8,
    "gread:fp32": 475.5,
    "gread:fp64": 466.8,
}

BASE_POINTS = 396
BASE_ITERS = 6000
DOUBLE_BACKPROP_FACTOR = 2.0
SEEDS = 5


def cell_seconds(key: str, *, points: int, iters: int, regularization: str) -> float:
    training = PROJECTED_TRAINING[key]
    evaluation = max(0.0, MEASURED_TOTAL[key] - training)
    scaled = training * (points / BASE_POINTS) * (iters / BASE_ITERS)
    scaled += evaluation * (iters / BASE_ITERS)
    if regularization == "double_backprop":
        scaled *= DOUBLE_BACKPROP_FACTOR
    return scaled


def design_hours(*, points: int, iters: int, regularizations: tuple[str, ...],
                 backbones: tuple[str, ...], seeds: int = SEEDS) -> float:
    total = 0.0
    for backbone in backbones:
        for precision in ("fp32", "fp64"):
            for regularization in regularizations:
                total += cell_seconds(
                    f"{backbone}:{precision}",
                    points=points, iters=iters, regularization=regularization,
                )
    return total * seeds / 3600.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quota", type=float, default=27.0, help="verfuegbare GPU-Stunden")
    parser.add_argument("--reserve", type=float, default=2.0, help="unantastbare Reserve")
    parser.add_argument("--points", type=int, default=1224,
                        help="Gesamtpunkte (1024 Domaene + 100 Rand + 100 IC)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    usable = args.quota - args.reserve
    all_backbones = ("mlp", "grand", "gread")
    designs = {
        "A  nur none, 3 Backbones": (("none",), all_backbones),
        "B  none + double_backprop, 3 Backbones": (("none", "double_backprop"), all_backbones),
        "B- none + double_backprop, ohne gread": (("none", "double_backprop"), ("mlp", "grand")),
    }

    print(f"Quota {args.quota:.1f} h, Reserve {args.reserve:.1f} h, "
          f"verplanbar {usable:.1f} h")
    print(f"Punktzahl {args.points} (entspricht 32x32-Gitter, 4.02 Abtastungen/Periode)")
    print(f"{SEEDS} Seeds, ein Worker\n")

    header = f"{'Iterationen':>12} | " + " | ".join(f"{name:>38}" for name in designs)
    print(header)
    print("-" * len(header))
    rows = {}
    for iters in (2000, 3000, 4000, 5000, 6000, 8000, 10000, 20000):
        cells = []
        rows[iters] = {}
        for name, (regs, backbones) in designs.items():
            hours = design_hours(points=args.points, iters=iters,
                                 regularizations=regs, backbones=backbones)
            rows[iters][name] = hours
            mark = "OK " if hours <= usable else "NEIN"
            cells.append(f"{hours:32.1f} h {mark}")
        print(f"{iters:>12} | " + " | ".join(cells))

    print("\nHinweis: Die Praeregistrierung setzt eine Untergrenze von 5000 Iterationen.")
    print("Werte darunter sind nur zulaessig, wenn der Machbarkeitsanker zeigt, dass")
    print("das Ziel frueher erreicht wird — dann ist die Untergrenze gegenstandslos,")
    print("weil sie die Konvergenz sichern sollte, nicht eine Mindestlaufzeit.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "quota_hours": args.quota,
            "reserve_hours": args.reserve,
            "usable_hours": usable,
            "points": args.points,
            "seeds": SEEDS,
            "hours_by_iters": rows,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\ngeschrieben: {args.out}")


if __name__ == "__main__":
    main()
