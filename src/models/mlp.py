from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        num_layers: int,
        bounds: tuple[tuple[float, float], tuple[float, float]],
    ) -> None:
        super().__init__()
        low = torch.tensor([bounds[0][0], bounds[1][0]], dtype=torch.float64)
        high = torch.tensor([bounds[0][1], bounds[1][1]], dtype=torch.float64)
        self.register_buffer("coord_low", low)
        self.register_buffer("coord_high", high)
        layers: list[nn.Module] = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def normalize(self, coords: torch.Tensor) -> torch.Tensor:
        low = self.coord_low.to(dtype=coords.dtype, device=coords.device)
        high = self.coord_high.to(dtype=coords.dtype, device=coords.device)
        return 2.0 * (coords - low) / (high - low) - 1.0

    def forward(self, coords: torch.Tensor, topology=None) -> torch.Tensor:
        del topology
        return self.network(self.normalize(coords)).squeeze(-1)
