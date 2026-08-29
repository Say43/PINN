"""Reaction-Gleichung nach Krishnapriyan et al., Appendix A, Gl. 14.

    u_t - rho * u * (1 - u) = 0,   x in [0, 2*pi],  t in [0, 1]
    u(x, 0) = exp(-(x - pi)^2 / (2 * (pi/4)^2))       (Gl. 17)
    u(0, t) = u(2*pi, t)                              (periodisch)

Analytische Loesung (Gl. 15):

    u(x, t) = h(x) e^{rho t} / ( h(x) e^{rho t} + 1 - h(x) )

Warum diese Gleichung: Sie hat keinen Ortsableitungsterm und damit keine
oszillierende Loesung — die Nyquist-Falle aus FINDINGS.md §2 existiert hier nicht.
Die Loesung ist glatt und monoton in t. Als Kontrast zu Convection ist sie
strukturell aussagekraeftig: GRAND kann nur diffundieren und sollte hier nicht
helfen, GREAD hat einen Reaktionsterm und sollte es.

Berichtete Fehler bei Krishnapriyan: Tab. E.2 fuer rho in {5, ..., 10}.
"""

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


def _rectangular_grid(
    count: int, *, x_min: float, x_max: float, t_min: float, t_max: float
) -> torch.Tensor:
    nx = max(1, int(math.sqrt(count)))
    nt = math.ceil(count / nx)
    x = x_min + (x_max - x_min) * torch.arange(nx, dtype=torch.float64) / nx
    t = torch.linspace(t_min + (t_max - t_min) / (nt + 1), t_max, nt, dtype=torch.float64)
    xx, tt = torch.meshgrid(x, t, indexing="ij")
    return torch.stack((xx.reshape(-1), tt.reshape(-1)), dim=1)[:count]


def _initial_profile(x: torch.Tensor) -> torch.Tensor:
    """Gaussprofil h(x) aus Gl. 17."""
    width = math.pi / 4.0
    return torch.exp(-((x - math.pi) ** 2) / (2.0 * width**2))


def _initial_profile_derivative(x: torch.Tensor) -> torch.Tensor:
    width = math.pi / 4.0
    return _initial_profile(x) * (-(x - math.pi) / width**2)


class ReactionPDE(PDE):
    bounds = ((0.0, 2.0 * math.pi), (0.0, 1.0))

    def __init__(self, config: PDEConfig) -> None:
        super().__init__(config)
        self.rho = config.reaction

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
        h = _initial_profile(coords[:, 0])
        growth = torch.exp(self.rho * coords[:, 1])
        return h * growth / (h * growth + 1.0 - h)

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
        u_domain = prediction[batch.domain]
        residual = ut[batch.domain] - self.rho * u_domain * (1.0 - u_domain)
        boundary_residual = prediction[batch.boundary_left] - prediction[batch.boundary_right]
        initial_coords = coords[batch.initial]
        initial_residual = prediction[batch.initial] - _initial_profile(initial_coords[:, 0])
        loss_domain = mean_square(residual)
        loss_boundary = mean_square(boundary_residual)
        loss_initial = mean_square(initial_residual)
        base = loss_domain + loss_boundary + loss_initial
        regularizer = base.new_zeros(())
        if regularization == "double_backprop":
            residual_gradient = pointwise_gradient(residual, coords)[batch.domain]
            boundary_t = ut[batch.boundary_left] - ut[batch.boundary_right]
            initial_x = ux[batch.initial] - _initial_profile_derivative(initial_coords[:, 0])
            regularizer = (
                residual_gradient.square().sum(dim=1).mean()
                + mean_square(boundary_t)
                + mean_square(initial_x)
            )
        total = base + (
            0.5 * lambda_r * regularizer if regularization == "double_backprop" else 0.0
        )
        return LossBreakdown(total, base, loss_domain, loss_boundary, loss_initial, regularizer)

    def evaluation_grid(self, *, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.linspace(0.0, 2.0 * math.pi, self.config.evaluation_x, device=device, dtype=dtype)
        t = torch.linspace(0.0, 1.0, self.config.evaluation_t, device=device, dtype=dtype)
        xx, tt = torch.meshgrid(x, t, indexing="ij")
        coords = torch.stack((xx.reshape(-1), tt.reshape(-1)), dim=1)
        return coords, self.exact(coords)
