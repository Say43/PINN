from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class GraphTopology:
    neighbors: torch.Tensor
    mask: torch.Tensor

    def to(self, device: torch.device | str) -> "GraphTopology":
        return GraphTopology(self.neighbors.to(device), self.mask.to(device))


def build_knn_graph(
    normalized_coords: torch.Tensor,
    *,
    k: int,
    chunk_size: int = 512,
) -> GraphTopology:
    """Build a deterministic, symmetrized k-NN graph with self-loops.

    Distances are evaluated in chunks to avoid materializing an N x N matrix.
    The graph is constructed on CPU and can subsequently be moved to a device.
    """

    coords = normalized_coords.detach().to(device="cpu", dtype=torch.float64)
    n_nodes = coords.shape[0]
    if n_nodes < 1:
        raise ValueError("cannot build a graph without nodes")
    directed_k = min(k + 1, n_nodes)
    directed: list[torch.Tensor] = []
    for start in range(0, n_nodes, chunk_size):
        query = coords[start : start + chunk_size]
        distances = torch.cdist(query, coords)
        nearest = torch.topk(distances, k=directed_k, largest=False, sorted=True).indices
        directed.extend(row for row in nearest)

    adjacency = [set([index]) for index in range(n_nodes)]
    for index, row in enumerate(directed):
        for neighbor in row.tolist():
            adjacency[index].add(neighbor)
            adjacency[neighbor].add(index)

    ordered = [sorted(neighbors) for neighbors in adjacency]
    max_degree = max(len(neighbors) for neighbors in ordered)
    indices = torch.empty((n_nodes, max_degree), dtype=torch.long)
    mask = torch.zeros((n_nodes, max_degree), dtype=torch.bool)
    for index, neighbors in enumerate(ordered):
        degree = len(neighbors)
        indices[index, :degree] = torch.tensor(neighbors, dtype=torch.long)
        indices[index, degree:] = index
        mask[index, :degree] = True
    return GraphTopology(indices, mask)


class DiffusionMixingLayer(nn.Module):
    def __init__(self, hidden_dim: int, *, reaction: bool, step_size: float) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.log_diffusivity = nn.Parameter(torch.tensor(0.541324854612918))
        self.log_reaction = nn.Parameter(torch.tensor(-2.25216846104409)) if reaction else None
        self.reaction = reaction
        self.step_size = step_size

    @staticmethod
    def _fused_linear(
        values: torch.Tensor,
        modules: tuple[nn.Linear, ...],
    ) -> tuple[torch.Tensor, ...]:
        weight = torch.cat([module.weight for module in modules], dim=0)
        bias = torch.cat([module.bias for module in modules], dim=0)
        projected = F.linear(values, weight, bias)
        return projected.split(modules[0].out_features, dim=-1)

    def _diffusion_from_projections(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        neighbor_value: torch.Tensor,
        self_value: torch.Tensor,
        topology: GraphTopology,
    ) -> torch.Tensor:
        scores = (query.unsqueeze(1) * key).sum(dim=-1) / math.sqrt(query.shape[-1])
        scores = scores.masked_fill(~topology.mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=1)
        aggregate = (attention.unsqueeze(-1) * neighbor_value).sum(dim=1)
        return self.output(aggregate - self_value)

    def _advance_from_diffusion(
        self,
        live: torch.Tensor,
        diffusion: torch.Tensor,
    ) -> torch.Tensor:
        diffusivity = F.softplus(self.log_diffusivity)
        derivative = diffusivity * diffusion
        if self.reaction:
            bounded = torch.tanh(live)
            reaction_rate = F.softplus(self.log_reaction)
            derivative = derivative + reaction_rate * bounded * (1.0 - bounded.square())
        return torch.tanh(live + self.step_size * derivative)

    def forward_pair(
        self,
        live: torch.Tensor,
        context: torch.Tensor,
        topology: GraphTopology,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context_query, context_key, context_value = self._fused_linear(
            context, (self.query, self.key, self.value)
        )
        live_query, live_value = self._fused_linear(live, (self.query, self.value))
        context_key = context_key[topology.neighbors]
        neighbor_value = context_value[topology.neighbors]
        live_diffusion = self._diffusion_from_projections(
            live_query,
            context_key,
            neighbor_value,
            live_value,
            topology,
        )
        context_diffusion = self._diffusion_from_projections(
            context_query,
            context_key,
            neighbor_value,
            context_value,
            topology,
        )
        next_live = self._advance_from_diffusion(live, live_diffusion)
        next_context = self._advance_from_diffusion(context, context_diffusion)
        return next_live, next_context


class PDEGraphNet(nn.Module):
    """GRAND/GREAD-style sparse mixer over a fixed collocation graph.

    The context stream receives detached coordinates but remains differentiable with
    respect to model parameters. Consequently each output depends on its own live
    coordinate and on neighbor context values, while autograd sums still produce the
    diagonal spatial/temporal derivatives required by a PINN residual.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_layers: int,
        bounds: tuple[tuple[float, float], tuple[float, float]],
        diffusion_step: float,
        reaction: bool,
    ) -> None:
        super().__init__()
        low = torch.tensor([bounds[0][0], bounds[1][0]], dtype=torch.float64)
        high = torch.tensor([bounds[0][1], bounds[1][1]], dtype=torch.float64)
        self.register_buffer("coord_low", low)
        self.register_buffer("coord_high", high)
        self.encoder = nn.Linear(2, hidden_dim)
        self.layers = nn.ModuleList(
            DiffusionMixingLayer(hidden_dim, reaction=reaction, step_size=diffusion_step)
            for _ in range(num_layers)
        )
        self.decoder = nn.Linear(hidden_dim, 1)
        self.ode_function_evaluations = 0
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

    def forward(self, coords: torch.Tensor, topology: GraphTopology | None = None) -> torch.Tensor:
        if topology is None:
            raise ValueError("graph backbones require a fixed GraphTopology")
        live_coords = self.normalize(coords)
        context_coords = self.normalize(coords.detach())
        live = torch.tanh(self.encoder(live_coords))
        context = torch.tanh(self.encoder(context_coords))
        for layer in self.layers:
            live, context = layer.forward_pair(live, context, topology)
        self.ode_function_evaluations += len(self.layers)
        return self.decoder(live).squeeze(-1)
