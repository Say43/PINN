from __future__ import annotations

from src.config import PDEConfig
from src.pdes.allen_cahn import AllenCahnPDE
from src.pdes.base import CollocationBatch, LossBreakdown, PDE
from src.pdes.convection import ConvectionPDE


def create_pde(config: PDEConfig) -> PDE:
    if config.name == "convection":
        return ConvectionPDE(config)
    if config.name == "allen_cahn":
        return AllenCahnPDE(config)
    raise ValueError(f"unknown PDE: {config.name}")


__all__ = [
    "AllenCahnPDE",
    "CollocationBatch",
    "ConvectionPDE",
    "LossBreakdown",
    "PDE",
    "create_pde",
]
