import unittest

import torch

from src.config import PDEConfig
from src.pdes.allen_cahn import AllenCahnPDE
from src.pdes.convection import ConvectionPDE


class ExactConvection(torch.nn.Module):
    def __init__(self, beta: float) -> None:
        super().__init__()
        self.beta = beta

    def forward(self, coords, topology=None):
        del topology
        return torch.sin(coords[:, 0] - self.beta * coords[:, 1])


class PDETests(unittest.TestCase):
    def test_convection_exact_solution_has_near_zero_loss(self) -> None:
        config = PDEConfig(
            allow_underresolved=True,
            name="convection", domain_points=25, boundary_points=8, initial_points=8)
        pde = ConvectionPDE(config)
        batch = pde.collocation().to(device="cpu", dtype=torch.float64)
        loss = pde.loss(ExactConvection(config.beta), batch, None, "none", 0.0)
        self.assertLess(float(loss.total), 1.0e-20)

    def test_allen_cahn_point_accounting_and_initial_derivative(self) -> None:
        config = PDEConfig(allow_underresolved=True, name="allen_cahn", domain_points=4096, boundary_points=100, initial_points=100)
        pde = AllenCahnPDE(config)
        batch = pde.collocation()
        self.assertEqual(len(batch.coords), 4396)
        x = torch.linspace(-1.0, 1.0, 11, dtype=torch.float64, requires_grad=True)
        value = pde.initial_condition(x)
        derivative = torch.autograd.grad(value.sum(), x)[0]
        self.assertTrue(torch.allclose(derivative, pde.initial_condition_dx(x), atol=1.0e-12))

    def test_allen_cahn_reference_matches_initial_condition(self) -> None:
        config = PDEConfig(allow_underresolved=True, 
            name="allen_cahn",
            domain_points=4,
            boundary_points=2,
            initial_points=2,
            evaluation_x=33,
            evaluation_t=3,
            reference_path="data/allen_cahn.mat",
        )
        pde = AllenCahnPDE(config)
        coords, reference = pde.evaluation_grid(device="cpu", dtype=torch.float64)
        at_initial = coords[:, 1] == 0
        expected = pde.initial_condition(coords[at_initial, 0])
        self.assertLess(float(torch.max(torch.abs(reference[at_initial] - expected))), 2.0e-4)
