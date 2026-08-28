"""Outcome-aware feasibility diagnostics; never writes study data.

This module separates three questions that were previously mixed together:

1. Can the preregistered MLP represent the analytic convection solution when
   trained directly on that solution (supervised capacity check)?
2. How do the PINN loss terms and their parameter gradients scale at
   initialization as beta increases?
3. Does cold-start or curriculum PINN training solve a beta ladder, and does
   its loss generalize from the training grid to a denser held-out grid?

Every JSON artifact is labelled ``NOT_STUDY_DATA`` and is intentionally kept
out of ``results.sqlite``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch

from src.config import ModelConfig, PDEConfig, torch_dtype
from src.metrics import relative_l2
from src.models import create_model
from src.pdes.convection import ConvectionPDE
from src.train import set_determinism


NON_STUDY_NOTE = "Feasibility diagnostic, NOT_STUDY_DATA"


@dataclass(frozen=True)
class PinnSettings:
    domain_points: int = 400
    boundary_points: int = 200
    initial_points: int = 200
    hidden_dim: int = 128
    num_layers: int = 4
    precision: str = "fp32"
    regularization: str = "none"
    lambda_r: float = 1.0e-4
    max_iters: int = 1000
    history_size: int = 100
    learning_rate: float = 1.0
    log_every: int = 100
    seed: int = 0
    device: str = "cpu"
    allow_underresolved: bool = True


def _pde(beta: float, settings: PinnSettings, *, held_out: bool = False) -> ConvectionPDE:
    domain_points = max(4096, settings.domain_points) if held_out else settings.domain_points
    boundary_points = max(512, settings.boundary_points) if held_out else settings.boundary_points
    initial_points = max(512, settings.initial_points) if held_out else settings.initial_points
    return ConvectionPDE(
        PDEConfig(
            name="convection",
            domain_points=domain_points,
            boundary_points=boundary_points,
            initial_points=initial_points,
            evaluation_x=101,
            evaluation_t=101,
            beta=beta,
            allow_underresolved=settings.allow_underresolved if not held_out else False,
        )
    )


def _model(settings: PinnSettings, pde: ConvectionPDE) -> torch.nn.Module:
    config = ModelConfig(
        backbone="mlp",
        mlp_hidden_dim=settings.hidden_dim,
        num_layers=settings.num_layers,
    )
    dtype = torch_dtype(settings.precision)
    return create_model(config, pde.bounds).to(device=settings.device, dtype=dtype)


def _flat_gradient(
    value: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    *,
    retain_graph: bool,
) -> torch.Tensor:
    if not value.requires_grad:
        return torch.cat([torch.zeros_like(parameter).reshape(-1) for parameter in parameters])
    gradients = torch.autograd.grad(
        value,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    parts = [
        torch.zeros_like(parameter).reshape(-1)
        if gradient is None
        else gradient.detach().reshape(-1)
        for parameter, gradient in zip(parameters, gradients)
    ]
    return torch.cat(parts)


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator) == 0.0:
        return None
    return float(torch.dot(left, right) / denominator)


def gradient_snapshot(beta: float, settings: PinnSettings) -> dict:
    """Return loss sizes, parameter-gradient sizes, and term alignment at init."""
    set_determinism(settings.seed, use_cuda=settings.device.startswith("cuda"))
    dtype = torch_dtype(settings.precision)
    pde = _pde(beta, settings)
    batch = pde.collocation().to(device=settings.device, dtype=dtype)
    model = _model(settings, pde)
    breakdown = pde.loss(
        model,
        batch,
        None,
        settings.regularization,
        settings.lambda_r,
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    weighted_regularizer = 0.5 * settings.lambda_r * breakdown.regularizer
    terms = {
        "domain": breakdown.domain,
        "boundary": breakdown.boundary,
        "initial": breakdown.initial,
        "base": breakdown.base,
        "weighted_regularizer": weighted_regularizer,
        "total": breakdown.total,
    }
    gradients: dict[str, torch.Tensor] = {}
    names = list(terms)
    for index, name in enumerate(names):
        gradients[name] = _flat_gradient(
            terms[name],
            parameters,
            retain_graph=index < len(names) - 1,
        )
    return {
        "beta": beta,
        "settings": asdict(settings),
        "losses": {name: float(value.detach().cpu()) for name, value in terms.items()},
        "gradient_norms": {
            name: float(torch.linalg.vector_norm(value).cpu())
            for name, value in gradients.items()
        },
        "gradient_cosines": {
            "domain_initial": _cosine(gradients["domain"], gradients["initial"]),
            "domain_boundary": _cosine(gradients["domain"], gradients["boundary"]),
            "base_regularizer": _cosine(
                gradients["base"], gradients["weighted_regularizer"]
            ),
        },
    }


def _evaluate(model: torch.nn.Module, pde: ConvectionPDE, settings: PinnSettings) -> float:
    dtype = torch_dtype(settings.precision)
    coords, exact = pde.evaluation_grid(device=settings.device, dtype=dtype)
    with torch.no_grad():
        return float(relative_l2(model(coords), exact).cpu())


def _loss_snapshot(
    model: torch.nn.Module,
    pde: ConvectionPDE,
    settings: PinnSettings,
) -> dict[str, float]:
    dtype = torch_dtype(settings.precision)
    batch = pde.collocation().to(device=settings.device, dtype=dtype)
    breakdown = pde.loss(
        model,
        batch,
        None,
        settings.regularization,
        settings.lambda_r,
    )
    return breakdown.scalars()


def train_pinn_stage(
    model: torch.nn.Module,
    beta: float,
    settings: PinnSettings,
) -> dict:
    """Train one diagnostic stage, retaining the model for curriculum runs."""
    dtype = torch_dtype(settings.precision)
    pde = _pde(beta, settings)
    batch = pde.collocation().to(device=settings.device, dtype=dtype)
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=settings.learning_rate,
        max_iter=1,
        max_eval=25,
        tolerance_grad=0.0,
        tolerance_change=0.0,
        history_size=settings.history_size,
        line_search_fn="strong_wolfe",
    )
    started = time.perf_counter()
    evaluations = 0
    history: list[dict] = []
    for iteration in range(1, settings.max_iters + 1):
        def closure() -> torch.Tensor:
            nonlocal evaluations
            optimizer.zero_grad(set_to_none=True)
            loss = pde.loss(
                model,
                batch,
                None,
                settings.regularization,
                settings.lambda_r,
            ).total
            loss.backward()
            evaluations += 1
            return loss

        optimizer.step(closure)
        if iteration == 1 or iteration % settings.log_every == 0 or iteration == settings.max_iters:
            error = _evaluate(model, pde, settings)
            history.append(
                {
                    "iteration": iteration,
                    "relative_l2": error,
                    "wall_seconds": time.perf_counter() - started,
                    "function_evaluations": evaluations,
                }
            )
    held_out = _pde(beta, settings, held_out=True)
    train_loss = _loss_snapshot(model, pde, settings)
    held_out_loss = _loss_snapshot(model, held_out, settings)
    best = min(item["relative_l2"] for item in history)
    return {
        "beta": beta,
        "best_relative_l2": best,
        "final_relative_l2": history[-1]["relative_l2"],
        "target_reached": best < 0.10,
        "wall_seconds": time.perf_counter() - started,
        "function_evaluations": evaluations,
        "train_loss": train_loss,
        "held_out_loss": held_out_loss,
        "domain_generalization_ratio": (
            held_out_loss["loss_domain"] / train_loss["loss_domain"]
            if train_loss["loss_domain"] > 0.0
            else math.inf
        ),
        "history": history,
    }


def run_ladder(
    betas: Iterable[float],
    settings: PinnSettings,
    *,
    warm_start: bool,
) -> dict:
    betas = list(betas)
    if not betas:
        raise ValueError("at least one beta is required")
    stages: list[dict] = []
    model = None
    for beta in betas:
        if model is None or not warm_start:
            set_determinism(settings.seed, use_cuda=settings.device.startswith("cuda"))
            model = _model(settings, _pde(beta, settings))
        stages.append(train_pinn_stage(model, beta, settings))
    return {
        "note": NON_STUDY_NOTE,
        "diagnostic": "curriculum_ladder" if warm_start else "cold_start_ladder",
        "settings": asdict(settings),
        "stages": stages,
    }


def supervised_capacity(
    beta: float,
    settings: PinnSettings,
    *,
    max_iters: int,
    learning_rate: float,
    grid_size: int,
) -> dict:
    """Fit the analytic solution directly; this is not a PINN training run."""
    set_determinism(settings.seed, use_cuda=settings.device.startswith("cuda"))
    dtype = torch_dtype(settings.precision)
    pde = _pde(beta, settings)
    model = _model(settings, pde)
    x = torch.linspace(0.0, 2.0 * math.pi, grid_size, device=settings.device, dtype=dtype)
    t = torch.linspace(0.0, 1.0, grid_size, device=settings.device, dtype=dtype)
    xx, tt = torch.meshgrid(x, t, indexing="ij")
    coords = torch.stack((xx.reshape(-1), tt.reshape(-1)), dim=1)
    exact = pde.exact(coords)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    started = time.perf_counter()
    history: list[dict] = []
    for iteration in range(1, max_iters + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(coords)
        loss = torch.mean((prediction - exact).square())
        loss.backward()
        optimizer.step()
        if iteration == 1 or iteration % 100 == 0 or iteration == max_iters:
            with torch.no_grad():
                error = float(relative_l2(model(coords), exact).cpu())
            history.append(
                {
                    "iteration": iteration,
                    "mse": float(loss.detach().cpu()),
                    "relative_l2": error,
                    "wall_seconds": time.perf_counter() - started,
                }
            )
    return {
        "note": NON_STUDY_NOTE,
        "diagnostic": "supervised_capacity",
        "beta": beta,
        "settings": asdict(settings),
        "optimizer": "adam",
        "learning_rate": learning_rate,
        "grid_size": grid_size,
        "max_iters": max_iters,
        "best_relative_l2": min(item["relative_l2"] for item in history),
        "final_relative_l2": history[-1]["relative_l2"],
        "target_reached": min(item["relative_l2"] for item in history) < 0.01,
        "history": history,
    }


def _write(payload: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"written: {destination}")


def _settings(args: argparse.Namespace) -> PinnSettings:
    return PinnSettings(
        domain_points=args.domain_points,
        boundary_points=args.boundary_points,
        initial_points=args.initial_points,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        precision=args.precision,
        regularization=args.regularization,
        lambda_r=args.lambda_r,
        max_iters=getattr(args, "max_iters", 1000),
        log_every=getattr(args, "log_every", 100),
        seed=args.seed,
        device=args.device,
        allow_underresolved=args.allow_underresolved,
    )


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--domain-points", type=int, default=400)
    parser.add_argument("--boundary-points", type=int, default=200)
    parser.add_argument("--initial-points", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--precision", choices=("fp32", "fp64"), default="fp32")
    parser.add_argument(
        "--regularization",
        choices=("none", "double_backprop"),
        default="none",
    )
    parser.add_argument("--lambda-r", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-underresolved", action="store_true")
    parser.add_argument("--out", required=True, type=Path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshot = subparsers.add_parser("snapshot")
    _common(snapshot)
    snapshot.add_argument("--betas", type=float, nargs="+", default=[1.0, 10.0, 50.0])

    ladder = subparsers.add_parser("ladder")
    _common(ladder)
    ladder.add_argument("--betas", type=float, nargs="+", default=[1.0, 10.0, 50.0])
    ladder.add_argument("--max-iters", type=int, default=1000)
    ladder.add_argument("--log-every", type=int, default=100)
    ladder.add_argument("--warm-start", action="store_true")

    capacity = subparsers.add_parser("capacity")
    _common(capacity)
    capacity.add_argument("--beta", type=float, default=50.0)
    capacity.add_argument("--max-iters", type=int, default=2000)
    capacity.add_argument("--learning-rate", type=float, default=1.0e-3)
    capacity.add_argument("--grid-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = _settings(args)
    if args.command == "snapshot":
        payload = {
            "note": NON_STUDY_NOTE,
            "diagnostic": "initial_gradient_snapshot",
            "snapshots": [gradient_snapshot(beta, settings) for beta in args.betas],
        }
    elif args.command == "ladder":
        payload = run_ladder(args.betas, settings, warm_start=args.warm_start)
    else:
        payload = supervised_capacity(
            args.beta,
            settings,
            max_iters=args.max_iters,
            learning_rate=args.learning_rate,
            grid_size=args.grid_size,
        )
    _write(payload, args.out)


if __name__ == "__main__":
    main()
