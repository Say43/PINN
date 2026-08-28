"""Anfangs-Loss-Skalierung fuer lambda_r.

UEBERHOLT: Die hier verwendete Regel (Penalty auf 10 % des Anfangs-Loss)
waehlte lambda_r=1.0 und trieb Double Backprop nachweislich in die
Trivialloesung — der Initial-Loss stagnierte bei 0.43, der Optimierer fror
nach 300 Iterationen ein. Der funktionierende Bereich liegt bei 1e-4.
Die Nachfolgeregel waehlt lambda_r ueber einen Sweep auf der MLP-Baseline.
Siehe FINDINGS.md und PREREGISTRATION-V4.md.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import torch

from src.config import ExperimentConfig, ModelConfig, PDEConfig, TrainConfig
from src.models import create_model
from src.pdes import create_pde
from src.train import set_determinism


CANDIDATES = (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0)
TARGET_RATIO = 0.1
SEEDS = (0, 1, 2, 3, 4)


def measure_unweighted_terms(seed: int, *, beta: float = 50.0,
                            domain_points: int = 4096) -> tuple[float, float]:
    config = ExperimentConfig(
        pde=PDEConfig(
            name="convection",
            domain_points=domain_points,
            boundary_points=100,
            initial_points=100,
            beta=beta,
            evaluation_x=5,
            evaluation_t=5,
        ),
        model=ModelConfig(backbone="mlp", mlp_hidden_dim=128, num_layers=4),
        train=TrainConfig(
            precision="fp32",
            regularization="double_backprop",
            seed=seed,
            max_iters=1,
            device="cpu",
        ),
    )
    set_determinism(seed, use_cuda=False)
    pde = create_pde(config.pde)
    batch = pde.collocation().to(device="cpu", dtype=torch.float32)
    model = create_model(config.model, pde.bounds).to(dtype=torch.float32)
    breakdown = pde.loss(model, batch, None, "double_backprop", lambda_r=1.0)
    return float(breakdown.base.detach()), float(breakdown.regularizer.detach())


def select_lambda() -> dict:
    terms = []
    for seed in SEEDS:
        base, penalty = measure_unweighted_terms(seed)
        if not math.isfinite(base) or not math.isfinite(penalty) or base <= 0 or penalty <= 0:
            raise RuntimeError(f"invalid initial loss scale for seed {seed}: L0={base}, P={penalty}")
        terms.append({"seed": seed, "base_loss": base, "raw_penalty": penalty})

    candidates = []
    for lambda_r in CANDIDATES:
        ratios = [0.5 * lambda_r * item["raw_penalty"] / item["base_loss"] for item in terms]
        median_ratio = statistics.median(ratios)
        score = abs(math.log10(median_ratio / TARGET_RATIO))
        candidates.append(
            {
                "lambda_r": lambda_r,
                "ratios": ratios,
                "median_ratio": median_ratio,
                "log_distance_to_target": score,
            }
        )
    selected = min(candidates, key=lambda item: (item["log_distance_to_target"], item["lambda_r"]))
    return {
        "method": "outcome_blind_initial_loss_scaling",
        "device": "cpu",
        "precision": "fp32",
        "pde": "convection",
        "backbone": "mlp",
        "points": {"domain": 4096, "boundary": 100, "initial": 100},
        "target_ratio": TARGET_RATIO,
        "seeds": list(SEEDS),
        "terms": terms,
        "candidates": candidates,
        "selected_lambda_r": selected["lambda_r"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Select lambda_r without using outcome data")
    parser.add_argument("--output", type=Path, default=Path("results/lambda_r_selection.json"))
    args = parser.parse_args()
    result = select_lambda()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
