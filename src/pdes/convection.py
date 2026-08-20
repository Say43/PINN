from __future__ import annotations

import math

import torch

from src.config import PDEConfig, RegularizationName
from src.models.graph import GraphTopology
from src.pdes.base import (
    CollocationBatch,
    LossBreakdown,
    PDE,
    mean_square,
    pointwise_gradient,
)
from src.pdes.resolution import convection_resolution, require_convection_resolution


def _rectangular_grid(
    count: int,
    *,
    x_min: float,
    x_max: float,
    t_min: float,
    t_max: float,
) -> torch.Tensor:
    nx = max(1, int(math.sqrt(count)))
    nt = math.ceil(count / nx)
    x = x_min + (x_max - x_min) * torch.arange(nx, dtype=torch.float64) / nx
    t = torch.linspace(t_min + (t_max - t_min) / (nt + 1), t_max, nt, dtype=torch.float64)
    xx, tt = torch.meshgrid(x, t, indexing="ij")
    return torch.stack((xx.reshape(-1), tt.reshape(-1)), dim=1)[:count]


class ConvectionPDE(PDE):
    bounds = ((0.0, 2.0 * math.pi), (0.0, 1.0))

    def __init__(self, config: PDEConfig) -> None:
        super().__init__(config)
        self.beta = config.beta
        # Harte Vorbedingung, keine Warnung: ein unteraufgeloestes Gitter
        # produziert technisch fehlerfreie Laeufe mit sinnlosen Ergebnissen.
        if config.allow_underresolved:
            self.resolution = convection_resolution(
                config.domain_points, beta=config.beta
            )
        else:
            self.resolution = require_convection_resolution(
                config.domain_points, beta=config.beta
            )

    def collocation(self) -> CollocationBatch:
        domain = _rectangular_grid(
            self.config.domain_points,
            x_min=self.bounds[0][0],
            x_max=self.bounds[0][1],
            t_min=self.bounds[1][0],
            t_max=self.bounds[1][1],
        )
        boundary_t = torch.linspace(0.0, 1.0, self.config.boundary_points, dtype=torch.float64)
        left = torch.stack((torch.zeros_like(boundary_t), boundary_t), dim=1)
        right = torch.stack((torch.full_like(boundary_t, 2.0 * math.pi), boundary_t), dim=1)
        initial_x = 2.0 * math.pi * torch.arange(self.config.initial_points, dtype=torch.float64)
        initial_x = initial_x / self.config.initial_points
        initial = torch.stack((initial_x, torch.zeros_like(initial_x)), dim=1)
        coords = torch.cat((domain, left, right, initial), dim=0)
        n_domain = len(domain)
        n_boundary = len(left)
        return CollocationBatch(
            coords=coords,
            domain=slice(0, n_domain),
            boundary_left=slice(n_domain, n_domain + n_boundary),
            boundary_right=slice(n_domain + n_boundary, n_domain + 2 * n_boundary),
            initial=slice(n_domain + 2 * n_boundary, len(coords)),
        )

    def exact(self, coords: torch.Tensor) -> torch.Tensor:
        return torch.sin(coords[:, 0] - self.beta * coords[:, 1])

    def loss(
        self,
        model,
        batch: CollocationBatch,
        topology: GraphTopology | None,
        regularization: RegularizationName,
        lambda_r: float,
    ) -> LossBreakdown:
        coords = batch.coords
        prediction = model(coords, topology)
        first = pointwise_gradient(prediction, coords)
        ux, ut = first[:, 0], first[:, 1]
        residual = ut[batch.domain] + self.beta * ux[batch.domain]
        boundary_residual = prediction[batch.boundary_left] - prediction[batch.boundary_right]
        initial_coords = coords[batch.initial]
        initial_residual = prediction[batch.initial] - torch.sin(initial_coords[:, 0])
        loss_domain = mean_square(residual)
        loss_boundary = mean_square(boundary_residual)
        loss_initial = mean_square(initial_residual)
        base = loss_domain + loss_boundary + loss_initial
        regularizer = base.new_zeros(())
        if regularization == "double_backprop":
            residual_gradient = pointwise_gradient(residual, coords)[batch.domain]
            boundary_t = ut[batch.boundary_left] - ut[batch.boundary_right]
            initial_x = ux[batch.initial] - torch.cos(initial_coords[:, 0])
            regularizer = (
                residual_gradient.square().sum(dim=1).mean()
                + mean_square(boundary_t)
                + mean_square(initial_x)
            )
        total = base + (0.5 * lambda_r * regularizer if regularization == "double_backprop" else 0.0)
        return LossBreakdown(total, base, loss_domain, loss_boundary, loss_initial, regularizer)

    def evaluation_grid(self, *, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.linspace(0.0, 2.0 * math.pi, self.config.evaluation_x, device=device, dtype=dtype)
        t = torch.linspace(0.0, 1.0, self.config.evaluation_t, device=device, dtype=dtype)
        xx, tt = torch.meshgrid(x, t, indexing="ij")
        coords = torch.stack((xx.reshape(-1), tt.reshape(-1)), dim=1)
        return coords, self.exact(coords)
