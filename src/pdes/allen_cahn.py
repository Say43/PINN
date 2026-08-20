from __future__ import annotations

import math
from pathlib import Path

import numpy as np
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


def _interior_grid(count: int) -> torch.Tensor:
    nx = max(1, int(math.sqrt(count)))
    nt = math.ceil(count / nx)
    x = torch.linspace(-1.0, 1.0, nx + 2, dtype=torch.float64)[1:-1]
    t = torch.linspace(0.0, 1.0, nt + 1, dtype=torch.float64)[1:]
    xx, tt = torch.meshgrid(x, t, indexing="ij")
    return torch.stack((xx.reshape(-1), tt.reshape(-1)), dim=1)[:count]


class AllenCahnPDE(PDE):
    bounds = ((-1.0, 1.0), (0.0, 1.0))

    def __init__(self, config: PDEConfig) -> None:
        super().__init__(config)
        self.diffusion = config.diffusion
        self.reaction = config.reaction

    def collocation(self) -> CollocationBatch:
        domain = _interior_grid(self.config.domain_points)
        boundary_t = torch.linspace(0.0, 1.0, self.config.boundary_points, dtype=torch.float64)
        left = torch.stack((torch.full_like(boundary_t, -1.0), boundary_t), dim=1)
        right = torch.stack((torch.full_like(boundary_t, 1.0), boundary_t), dim=1)
        initial_x = torch.linspace(-1.0, 1.0, self.config.initial_points, dtype=torch.float64)
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

    @staticmethod
    def initial_condition(x: torch.Tensor) -> torch.Tensor:
        return x.square() * torch.cos(torch.pi * x)

    @staticmethod
    def initial_condition_dx(x: torch.Tensor) -> torch.Tensor:
        return 2.0 * x * torch.cos(torch.pi * x) - torch.pi * x.square() * torch.sin(torch.pi * x)

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
        second_from_x = pointwise_gradient(ux, coords)
        uxx, uxt = second_from_x[:, 0], second_from_x[:, 1]
        domain_u = prediction[batch.domain]
        residual = (
            ut[batch.domain]
            - self.diffusion * uxx[batch.domain]
            + self.reaction * domain_u.pow(3)
            - self.reaction * domain_u
        )
        boundary_u = prediction[batch.boundary_left] - prediction[batch.boundary_right]
        boundary_ux = ux[batch.boundary_left] - ux[batch.boundary_right]
        initial_coords = coords[batch.initial]
        initial_residual = prediction[batch.initial] - self.initial_condition(initial_coords[:, 0])
        loss_domain = mean_square(residual)
        loss_boundary = mean_square(boundary_u) + mean_square(boundary_ux)
        loss_initial = mean_square(initial_residual)
        base = loss_domain + loss_boundary + loss_initial
        regularizer = base.new_zeros(())
        if regularization == "double_backprop":
            residual_gradient = pointwise_gradient(residual, coords)[batch.domain]
            boundary_u_t = ut[batch.boundary_left] - ut[batch.boundary_right]
            boundary_ux_t = uxt[batch.boundary_left] - uxt[batch.boundary_right]
            initial_x = ux[batch.initial] - self.initial_condition_dx(initial_coords[:, 0])
            regularizer = (
                residual_gradient.square().sum(dim=1).mean()
                + mean_square(boundary_u_t)
                + mean_square(boundary_ux_t)
                + mean_square(initial_x)
            )
        total = base + (0.5 * lambda_r * regularizer if regularization == "double_backprop" else 0.0)
        return LossBreakdown(total, base, loss_domain, loss_boundary, loss_initial, regularizer)

    def _load_reference(self, coords: torch.Tensor) -> torch.Tensor:
        if self.config.reference_path is None:
            raise FileNotFoundError("Allen-Cahn evaluation requires pde.reference_path")
        path = Path(self.config.reference_path)
        if not path.exists():
            raise FileNotFoundError(f"Allen-Cahn reference not found: {path}")
        if path.suffix.lower() == ".mat":
            from scipy.io import loadmat

            data = loadmat(path)
            x_values = np.asarray(data["x"]).reshape(-1)
            t_values = np.asarray(data["t"]).reshape(-1)
            # The canonical FP64 artifact stores usol as [time, space].
            solution = np.real(np.asarray(data["usol"])).T
        else:
            data = np.load(path)
            x_values = np.asarray(data["x"]).reshape(-1)
            t_values = np.asarray(data["t"]).reshape(-1)
            solution = np.asarray(data["u"])
        x_grid = torch.as_tensor(x_values, dtype=coords.dtype, device=coords.device)
        t_grid = torch.as_tensor(t_values, dtype=coords.dtype, device=coords.device)
        values = torch.as_tensor(solution, dtype=coords.dtype, device=coords.device)
        ix = torch.searchsorted(x_grid, coords[:, 0].contiguous()).clamp(1, len(x_grid) - 1)
        it = torch.searchsorted(t_grid, coords[:, 1].contiguous()).clamp(1, len(t_grid) - 1)
        x0, x1 = x_grid[ix - 1], x_grid[ix]
        t0, t1 = t_grid[it - 1], t_grid[it]
        wx = (coords[:, 0] - x0) / (x1 - x0)
        wt = (coords[:, 1] - t0) / (t1 - t0)
        u00 = values[ix - 1, it - 1]
        u10 = values[ix, it - 1]
        u01 = values[ix - 1, it]
        u11 = values[ix, it]
        return (
            (1 - wx) * (1 - wt) * u00
            + wx * (1 - wt) * u10
            + (1 - wx) * wt * u01
            + wx * wt * u11
        )

    def evaluation_grid(self, *, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.linspace(-1.0, 1.0, self.config.evaluation_x, device=device, dtype=dtype)
        t = torch.linspace(0.0, 1.0, self.config.evaluation_t, device=device, dtype=dtype)
        xx, tt = torch.meshgrid(x, t, indexing="ij")
        coords = torch.stack((xx.reshape(-1), tt.reshape(-1)), dim=1)
        return coords, self._load_reference(coords)
