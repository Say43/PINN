import unittest

import torch

from src.config import ModelConfig
from src.models import build_knn_graph, count_parameters, create_model


BOUNDS = ((0.0, 6.283185307179586), (0.0, 1.0))


class ModelTests(unittest.TestCase):
    def test_preregistered_parameter_counts_are_within_ten_percent(self) -> None:
        counts = {}
        for backbone in ("mlp", "grand", "gread"):
            config = ModelConfig(backbone=backbone)
            counts[backbone] = count_parameters(create_model(config, BOUNDS))
        target = counts["mlp"]
        self.assertEqual(target, 50049)
        self.assertTrue(all(abs(value - target) / target <= 0.10 for value in counts.values()))

    def test_graph_autograd_returns_diagonal_coordinate_derivatives(self) -> None:
        for backbone in ("grand", "gread"):
            with self.subTest(backbone=backbone):
                torch.manual_seed(0)
                coords = torch.tensor(
                    [[0.1, 0.2], [1.0, 0.4], [2.0, 0.7], [4.0, 0.9]],
                    dtype=torch.float64,
                    requires_grad=True,
                )
                topology = build_knn_graph(coords.detach(), k=2)
                config = ModelConfig(backbone=backbone, graph_hidden_dim=8, num_layers=2, graph_k=2)
                model = create_model(config, BOUNDS).to(dtype=torch.float64)
                output = model(coords, topology)
                summed = torch.autograd.grad(output.sum(), coords, create_graph=True, retain_graph=True)[0]
                diagonal_rows = []
                for index in range(len(coords)):
                    full = torch.autograd.grad(output[index], coords, retain_graph=True)[0]
                    diagonal_rows.append(full[index])
                    off_diagonal = full.clone()
                    off_diagonal[index] = 0
                    self.assertTrue(
                        torch.allclose(off_diagonal, torch.zeros_like(off_diagonal), atol=1.0e-12)
                    )
                self.assertTrue(torch.allclose(summed, torch.stack(diagonal_rows), atol=1.0e-12))
