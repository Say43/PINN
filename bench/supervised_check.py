"""Ueberwachter Gegentest: kann das Netz die Referenzloesung ueberhaupt darstellen?

Kein Studienlauf, KEINE Studiendaten. Der Test umgeht den Physics-Loss vollstaendig
und passt das Netz direkt an die Referenzloesung auf dem Auswertungsgitter an.

Er trennt zwei sehr verschiedene Diagnosen:

- Erreicht der ueberwachte Fit einen kleinen Fehler, dann kann das Netz die Loesung
  darstellen und die Schwierigkeit liegt in der Optimierung des Physics-Loss.
- Bleibt auch der ueberwachte Fit schlecht, liegt der Fehler in der Referenz, im
  Auswertungsgitter oder in der Modellanbindung — also ein Bug, kein Befund.

Fuer Convection ist dieser Test bereits gelaufen (bester relativer L2 0.0113 bei
beta=50, siehe FINDINGS.md). Fuer Allen-Cahn fehlte er.

    python -m bench.supervised_check --pde allen_cahn --reference data/allen_cahn.mat
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from src.config import ModelConfig, PDEConfig
from src.models import create_model
from src.pdes import create_pde
from src.train import set_determinism


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pde", default="allen_cahn", choices=("convection", "allen_cahn"))
    parser.add_argument("--reference", default=None)
    parser.add_argument("--beta", type=float, default=50.0)
    parser.add_argument("--domain-points", type=int, default=4096)
    parser.add_argument("--max-iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results/supervised_check.json"))
    args = parser.parse_args()

    pde_config = PDEConfig(
        name=args.pde,
        domain_points=args.domain_points,
        boundary_points=100,
        initial_points=100,
        evaluation_x=101,
        evaluation_t=101,
        beta=args.beta,
        reference_path=args.reference,
        allow_underresolved=True,  # irrelevant: der Physics-Loss wird nicht benutzt
    )
    set_determinism(args.seed, use_cuda=False)
    pde = create_pde(pde_config)
    model = create_model(
        ModelConfig(backbone="mlp", mlp_hidden_dim=128, num_layers=4), pde.bounds
    ).to(dtype=torch.float64)

    coords, target = pde.evaluation_grid(device="cpu", dtype=torch.float64)
    print(f"{args.pde}: {coords.shape[0]} Auswertungspunkte")
    print(f"Referenz: min={float(target.min()):.4f} max={float(target.max()):.4f} "
          f"Norm={float(target.norm()):.4f}")
    print(f"Nullpraediktor erreicht relativen L2 = 1.0\n", flush=True)

    optimizer = torch.optim.LBFGS(
        model.parameters(), max_iter=1, max_eval=25, history_size=100, lr=1.0,
        line_search_fn="strong_wolfe", tolerance_grad=0.0, tolerance_change=0.0,
    )
    target_norm = target.norm()
    history = []
    started = time.perf_counter()
    for iteration in range(1, args.max_iters + 1):
        def closure():
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(coords), target)
            loss.backward()
            return loss
        optimizer.step(closure)
        if iteration == 1 or iteration % 100 == 0 or iteration == args.max_iters:
            with torch.no_grad():
                error = float((model(coords) - target).norm() / target_norm)
                loss = float(torch.nn.functional.mse_loss(model(coords), target))
            history.append({"iteration": iteration, "relative_l2": error, "mse": loss})
            print(f"{iteration:>6}  relL2={error:.6f}  mse={loss:.3e}", flush=True)

    best = min(h["relative_l2"] for h in history)
    elapsed = time.perf_counter() - started
    verdict = (
        "Netz kann die Loesung darstellen -> Schwierigkeit liegt in der Optimierung"
        if best < 0.10 else
        "Auch ueberwacht kein Erfolg -> Verdacht auf Fehler in Referenz oder Anbindung"
    )
    print(f"\nBester relativer L2: {best:.6f}  ({elapsed:.0f} s)")
    print(f"Befund: {verdict}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "note": "ueberwachter Gegentest, KEINE Studiendaten",
        "pde": args.pde, "beta": args.beta, "domain_points": args.domain_points,
        "max_iters": args.max_iters, "seed": args.seed,
        "best_relative_l2": best, "wall_seconds": elapsed,
        "verdict": verdict, "history": history,
    }, indent=2), encoding="utf-8")
    print(f"geschrieben: {args.out}")


if __name__ == "__main__":
    main()
