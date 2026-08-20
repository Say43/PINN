from __future__ import annotations

from src.config import ModelConfig
from src.models.graph import GraphTopology, PDEGraphNet, build_knn_graph
from src.models.mlp import MLP


def create_model(config: ModelConfig, bounds: tuple[tuple[float, float], tuple[float, float]]):
    if config.backbone == "mlp":
        return MLP(
            hidden_dim=config.mlp_hidden_dim,
            num_layers=config.num_layers,
            bounds=bounds,
        )
    if config.backbone in {"grand", "gread"}:
        return PDEGraphNet(
            hidden_dim=config.graph_hidden_dim,
            num_layers=config.num_layers,
            bounds=bounds,
            diffusion_step=config.diffusion_step,
            reaction=config.backbone == "gread",
        )
    raise ValueError(f"unknown backbone: {config.backbone}")


def count_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


__all__ = [
    "GraphTopology",
    "MLP",
    "PDEGraphNet",
    "build_knn_graph",
    "count_parameters",
    "create_model",
]
