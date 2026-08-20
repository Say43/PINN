from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from src.config import PDEConfig, RegularizationName
from src.models.graph import GraphTopology


@dataclass(frozen=True)
class CollocationBatch:
    coords: torch.Tensor
    domain: slice
    boundary_left: slice
    boundary_right: slice
    initial: slice

    def to(self, *, device: torch.device | str, dtype: torch.dtype) -> "CollocationBatch":
        coords = self.coords.to(device=device, dtype=dtype).detach().requires_grad_(True)
        return CollocationBatch(
            coords=coords,
            domain=self.domain,
            boundary_left=self.boundary_left,
            boundary_right=self.boundary_right,
            initial=self.initial,
        )


@dataclass(frozen=True)
class LossBreakdown:
    total: torch.Tensor
    base: torch.Tensor
    domain: torch.Tensor
    boundary: torch.Tensor
    initial: torch.Tensor
    regularizer: torch.Tensor

    def scalars(self) -> dict[str, float]:
        return {
            "loss_total": float(self.total.detach().cpu()),
            "loss_base": float(self.base.detach().cpu()),
            "loss_domain": float(self.domain.detach().cpu()),
            "loss_boundary": float(self.boundary.detach().cpu()),
            "loss_initial": float(self.initial.detach().cpu()),
            "loss_regularizer": float(self.regularizer.detach().cpu()),
        }


class ModelProtocol(Protocol):
    def __call__(self, coords: torch.Tensor, topology: GraphTopology | None = None) -> torch.Tensor: ...


def pointwise_gradient(values: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    gradient = torch.autograd.grad(
        values,
        coords,
        grad_outputs=torch.ones_like(values),
        create_graph=True,
        retain_graph=True,
        allow_unused=False,
    )[0]
    return gradient


def mean_square(values: torch.Tensor) -> torch.Tensor:
    return values.square().mean()


class PDE:
    bounds: tuple[tuple[float, float], tuple[float, float]]

    def __init__(self, config: PDEConfig) -> None:
        self.config = config

    def collocation(self) -> CollocationBatch:
        raise NotImplementedError

    def normalize_coords(self, coords: torch.Tensor) -> torch.Tensor:
        low = coords.new_tensor([self.bounds[0][0], self.bounds[1][0]])
        high = coords.new_tensor([self.bounds[0][1], self.bounds[1][1]])
        return (coords - low) / (high - low)

    def loss(
        self,
        model: ModelProtocol,
        batch: CollocationBatch,
        topology: GraphTopology | None,
        regularization: RegularizationName,
        lambda_r: float,
    ) -> LossBreakdown:
        raise NotImplementedError

    def evaluation_grid(
        self,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
