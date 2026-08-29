"""Machbarkeitsanker: kann dieser Code Convection bei beta=50 ueberhaupt loesen?

Kein Studienlauf. Das Ergebnis geht NICHT in results.sqlite und darf in keiner
Auswertung als Evidenz auftauchen. Zweck ist ausschliesslich die Frage, ob das
Ziel (relativer L2 < 0.10) mit dieser Implementierung und aufgeloestem Gitter
erreichbar ist — und nach wie vielen L-BFGS-Iterationen.

Hintergrund: In M2b lag der beste Wert bei 1.21, weil das Gitter unter dem
Nyquist-Limit lag (FINDINGS.md §2). Bevor erneut Kaggle-Budget fliesst, muss
lokal und kostenlos belegt sein, dass das Ziel erreichbar ist. Andernfalls
waere jeder weitere Kaggle-Lauf ein Glueckspiel.

Referenzpunkt aus der Literatur: Andersen & Matsubara §5.1 berichten fuer
Convection beta=50 unter FP32 Erfolg des double-PINN nach "a little over 5000
iterations of L-BFGS optimization" bei 400 Domaenenpunkten.

    python -m bench.feasibility_anchor --max-iters 8000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.config import ExperimentConfig, ModelConfig, PDEConfig, PersistenceConfig, TrainConfig
from src.pdes.resolution import convection_resolution
from src.train import train_once

TARGET = 0.10


def build_config(
    *,
    pde_name: str,
    reference_path: str | None,
    domain_points: int,
    backbone: str,
    precision: str,
    regularization: str,
    max_iters: int,
    seed: int,
    device: str,
    gpu_seconds: float,
    lambda_r: float,
    beta: float,
    rho: float,
) -> ExperimentConfig:
    return ExperimentConfig(
        pde=PDEConfig(
            name=pde_name,
            domain_points=domain_points,
            boundary_points=100,
            initial_points=100,
            evaluation_x=101,
            evaluation_t=101,
            beta=beta,
            reaction=rho,
            reference_path=reference_path,
        ),
        model=ModelConfig(
            backbone=backbone,
            mlp_hidden_dim=128,
            graph_hidden_dim=56,
            num_layers=4,
            graph_k=8,
            diffusion_step=0.25,
        ),
        train=TrainConfig(
            precision=precision,
            regularization=regularization,
            seed=seed,
            max_iters=max_iters,
            history_size=100,
            learning_rate=1.0,
            line_search_fn="strong_wolfe",
            tolerance_grad=0.0,
            tolerance_change=0.0,
            lambda_r=lambda_r,
            log_every=100,
            device=device,
            max_local_gpu_seconds=gpu_seconds,
        ),
        persistence=PersistenceConfig(
            database="",
            backup="",
            study_id="feasibility_anchor_NOT_STUDY_DATA",
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pde", default="convection", choices=("convection","allen_cahn","reaction"))
    parser.add_argument("--beta", type=float, default=50.0)
    parser.add_argument("--rho", type=float, default=5.0)
    parser.add_argument("--reference", default=None)
    parser.add_argument("--domain-points", type=int, default=1024)
    parser.add_argument("--backbone", default="mlp")
    parser.add_argument("--precision", default="fp32")
    parser.add_argument("--regularization", default="double_backprop")
    parser.add_argument("--max-iters", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lambda-r", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-seconds", type=float, default=14400.0)
    parser.add_argument("--out", type=Path, default=Path("results/feasibility_anchor.json"))
    args = parser.parse_args()

    if args.pde == "reaction":
        report = None
        print(f"Reaction, rho={args.rho:g}: glatte Loesung, keine Nyquist-Bedingung")
    elif args.pde == "convection":
        report = convection_resolution(args.domain_points, beta=args.beta)
        print(f"Aufloesung: {report.describe()}")
    else:
        report = None
        print("Aufloesung: fuer Allen-Cahn noch keine Regel definiert")
    print(f"Konfiguration: {args.backbone}/{args.precision}/{args.regularization}, "
          f"{args.domain_points} Domaenenpunkte, {args.max_iters} Iterationen, "
          f"lambda_r={args.lambda_r:g}, {args.device}")
    print("Ziel: relativer L2 < 0.10\n", flush=True)

    config = build_config(
        pde_name=args.pde,
        reference_path=args.reference,
        beta=args.beta,
        rho=args.rho,
        domain_points=args.domain_points,
        backbone=args.backbone,
        precision=args.precision,
        regularization=args.regularization,
        max_iters=args.max_iters,
        seed=args.seed,
        device=args.device,
        gpu_seconds=args.gpu_seconds,
        lambda_r=args.lambda_r,
    )

    started = time.perf_counter()
    result = train_once(config)
    elapsed = time.perf_counter() - started

    values = result.values
    history = values.get("history") or []
    if isinstance(history, str):
        history = json.loads(history)

    first_hit = next(
        (h for h in history if h.get("relative_l2") is not None and h["relative_l2"] < TARGET),
        None,
    )
    best = min((h["relative_l2"] for h in history if h.get("relative_l2") is not None),
               default=None)

    print(f"\nStatus: {result.status}")
    print(f"Wall-Clock: {elapsed:.1f} s")
    print(f"Bester relativer L2: {best}")
    print(f"Finaler relativer L2: {values.get('relative_l2')}")
    if first_hit:
        print(f"ZIEL ERREICHT bei Iteration {first_hit['iteration']} "
              f"nach {first_hit['wall_seconds']:.1f} s")
    else:
        print("ZIEL NICHT ERREICHT")

    payload = {
        "note": "Machbarkeitsanker, KEINE Studiendaten",
        "resolution": report.describe() if report else "n/a",
        "samples_per_period": report.samples_per_period if report else None,
        "config": {
            "pde": args.pde,
            "beta": args.beta,
            "domain_points": args.domain_points,
            "backbone": args.backbone,
            "precision": args.precision,
            "regularization": args.regularization,
            "max_iters": args.max_iters,
            "seed": args.seed,
            "lambda_r": args.lambda_r,
            "device": args.device,
        },
        "status": result.status,
        "wall_seconds": elapsed,
        "best_relative_l2": best,
        "final_relative_l2": values.get("relative_l2"),
        "target": TARGET,
        "target_reached": bool(first_hit),
        "iterations_to_target": first_hit["iteration"] if first_hit else None,
        "history": history,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"geschrieben: {args.out}")


if __name__ == "__main__":
    main()
