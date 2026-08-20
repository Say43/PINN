import unittest
import math

import torch
from torch.nn import functional as F

from src.config import ModelConfig
from src.models import build_knn_graph, count_parameters, create_model
from src.models.graph import DiffusionMixingLayer


BOUNDS = ((0.0, 6.283185307179586), (0.0, 1.0))


class ModelTests(unittest.TestCase):
    def test_optimized_graph_layer_matches_original_formula_and_gradients(self) -> None:
        topology = build_knn_graph(torch.tensor([[0.0], [1.0], [2.0], [3.0]]), k=2)
        for reaction in (False, True):
            with self.subTest(reaction=reaction):
                torch.manual_seed(7)
                layer = DiffusionMixingLayer(5, reaction=reaction, step_size=0.25).double()
                live = torch.randn(4, 5, dtype=torch.float64, requires_grad=True)
                context = torch.randn(4, 5, dtype=torch.float64, requires_grad=True)

                def original_diffusion(left, right):
                    query = layer.query(left).unsqueeze(1)
                    key = layer.key(right)[topology.neighbors]
                    value = layer.value(right)[topology.neighbors]
                    scores = (query * key).sum(dim=-1) / math.sqrt(left.shape[-1])
                    scores = scores.masked_fill(
                        ~topology.mask, torch.finfo(scores.dtype).min
                    )
                    attention = torch.softmax(scores, dim=1)
                    aggregate = (attention.unsqueeze(-1) * value).sum(dim=1)
                    return layer.output(aggregate - layer.value(left))

                def original_advance(left, right):
                    derivative = F.softplus(layer.log_diffusivity) * original_diffusion(left, right)
                    if reaction:
                        bounded = torch.tanh(left)
                        derivative = derivative + F.softplus(layer.log_reaction) * bounded * (
                            1.0 - bounded.square()
                        )
                    return torch.tanh(left + layer.step_size * derivative)

                expected = (original_advance(live, context), original_advance(context, context))
                actual = layer.forward_pair(live, context, topology)
                self.assertTrue(torch.allclose(actual[0], expected[0], atol=1.0e-12, rtol=1.0e-12))
                self.assertTrue(torch.allclose(actual[1], expected[1], atol=1.0e-12, rtol=1.0e-12))
                variables = [live, context, *layer.parameters()]
                expected_gradients = torch.autograd.grad(
                    expected[0].sum() + expected[1].sum(), variables, retain_graph=True
                )
                actual_gradients = torch.autograd.grad(
                    actual[0].sum() + actual[1].sum(), variables
                )
                for actual_gradient, expected_gradient in zip(
                    actual_gradients, expected_gradients, strict=True
                ):
                    self.assertTrue(
                        torch.allclose(actual_gradient, expected_gradient, atol=1.0e-12, rtol=1.0e-12)
                    )

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
