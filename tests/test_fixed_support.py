import unittest
import torch

from src.config import ModelConfig
from src.models import create_model


class FixedSupportTests(unittest.TestCase):
    def make_model(self, backbone):
        torch.manual_seed(101)
        model = create_model(
            ModelConfig(backbone=backbone, graph_context="fixed_support",
                        graph_hidden_dim=8, num_layers=2, graph_k=2),
            ((0., 1.), (0., 1.)),
        ).double()
        model.set_support(torch.tensor(
            [[0., 0.], [0., 1.], [1., 0.], [1., 1.], [.5, .5]], dtype=torch.float64))
        return model

    def test_first_and_second_derivatives_match_finite_differences(self):
        # Queries away from kNN changes, with support geometry frozen.
        for backbone in ("grand", "gread"):
            model = self.make_model(backbone)
            coords = torch.tensor([[.17, .23], [.79, .86]], dtype=torch.float64,
                                  requires_grad=True)
            values = model(coords)
            first = torch.autograd.grad(values.sum(), coords, create_graph=True)[0]
            second = torch.autograd.grad(first[:, 1].sum(), coords)[0][:, 1]
            for eps in (1e-3, 2e-4):
                plus, minus = coords.detach().clone(), coords.detach().clone()
                plus[:, 1] += eps
                minus[:, 1] -= eps
                with torch.no_grad():
                    upper, lower = model(plus), model(minus)
                torch.testing.assert_close((upper - lower) / (2 * eps), first[:, 1],
                                           atol=2e-4, rtol=2e-4)
                torch.testing.assert_close((upper - 2 * values.detach() + lower) / eps**2,
                                           second, atol=2e-3, rtol=2e-3)

    def test_prediction_independent_of_batch_order_and_unrelated_queries(self):
        for backbone in ("grand", "gread"):
            model = self.make_model(backbone)
            coords = torch.tensor([[.17, .23], [.79, .86]], dtype=torch.float64)
            with torch.no_grad():
                expected = model(coords)
                actual = torch.cat([model(coords[i:i+1]) for i in range(2)])
                torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
                torch.testing.assert_close(model(coords.flip(0)).flip(0), expected)
            # Geometry stays separate even if a caller changes its query tensor.
            saved = model.support_coords.clone()
            coords.add_(.01)
            torch.testing.assert_close(model.support_coords, saved)

    def test_parameter_gradients_still_flow_through_support(self):
        model = self.make_model("gread")
        model(torch.tensor([[.17, .23]], dtype=torch.float64)).sum().backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_neighbour_entry_has_no_prediction_jump(self):
        for backbone in ("grand", "gread"):
            model = self.make_model(backbone)
            model.set_support(torch.tensor([[0., .5], [.5, .5], [1., .5]], dtype=torch.float64))
            model.query_radius.fill_(1.)
            jumps = []
            for eps in (1e-5, 1e-7):
                coords = torch.tensor([[.5-eps, .5], [.5+eps, .5]], dtype=torch.float64)
                with torch.no_grad():
                    values = model(coords)
                jumps.append(float((values[1]-values[0]).abs()))
            self.assertLess(jumps[1], .02 * jumps[0] + 1e-12)
