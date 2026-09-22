"""Build the preregistered V5 tuning and study condition sets.

This module only constructs configurations.  It does not train, write the study
database, or turn the V5 draft into a frozen preregistration.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

from src.config import ExperimentConfig


LAMBDA_CANDIDATES = (1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2)
# Seed 0 collapses in the unregularised baseline, so a selection on it cannot
# separate the candidates.  The revised rule uses the two seeds whose baseline
# succeeds (DEVIATIONS.md, D-13).
LAMBDA_SELECTION_SEEDS = (3, 4)
BACKBONES = ("mlp", "grand", "gread")
PRECISIONS = ("fp32", "fp64")
REGULARIZATIONS = ("none", "double_backprop")
SEEDS = (0, 1, 2, 3, 4)


def validate_v5_base(config: ExperimentConfig) -> None:
    expected = {
        "name": "reaction",
        "domain_points": 1600,
        "boundary_points": 100,
        "initial_points": 100,
        "reaction": 5.25,
    }
    mismatches = {
        key: (getattr(config.pde, key), value)
        for key, value in expected.items()
        if getattr(config.pde, key) != value
    }
    if mismatches:
        raise ValueError(f"configuration does not match V5 draft: {mismatches}")
    if config.train.max_iters != 2000 or config.train.log_every != 500:
        raise ValueError("V5 requires max_iters=2000 and log_every=500")
    if config.model.graph_context != "fixed_support":
        raise ValueError("V5 requires fixed support for consistent graph derivatives")
    if "DRAFT" not in config.persistence.study_id:
        raise ValueError("V5 base must retain an explicit draft study_id")


def lambda_selection_conditions(base: ExperimentConfig) -> Iterator[ExperimentConfig]:
    """The MLP/FP32 double-backprop grid from V5 section 5, on the baseline-success seeds."""
    validate_v5_base(base)
    for lambda_r in LAMBDA_CANDIDATES:
        for seed in LAMBDA_SELECTION_SEEDS:
            yield replace(
                base,
                model=replace(base.model, backbone="mlp"),
                train=replace(
                    base.train,
                    precision="fp32",
                    regularization="double_backprop",
                    seed=seed,
                    lambda_r=lambda_r,
                ),
                persistence=replace(
                    base.persistence,
                    study_id="reaction_v5_lambda_selection_seeds34_NOT_STUDY_DATA",
                ),
            )


def study_conditions(
    base: ExperimentConfig, *, lambda_r: float, study_id: str
) -> Iterator[ExperimentConfig]:
    """The complete 12-cell by 5-seed V5 matrix after lambda selection."""
    validate_v5_base(base)
    if lambda_r not in LAMBDA_CANDIDATES:
        raise ValueError(f"lambda_r must be one of {LAMBDA_CANDIDATES}")
    if not study_id or "DRAFT" in study_id or "NOT_STUDY_DATA" in study_id:
        raise ValueError("study_id must identify the frozen V5 study")
    for backbone in BACKBONES:
        for precision in PRECISIONS:
            for regularization in REGULARIZATIONS:
                for seed in SEEDS:
                    yield replace(
                        base,
                        model=replace(base.model, backbone=backbone),
                        train=replace(
                            base.train,
                            precision=precision,
                            regularization=regularization,
                            seed=seed,
                            lambda_r=lambda_r,
                        ),
                        persistence=replace(base.persistence, study_id=study_id),
                    )
