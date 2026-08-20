import unittest

from src.config import PDEConfig
from src.pdes.convection import ConvectionPDE
from src.pdes.resolution import (
    MIN_SAMPLES_PER_PERIOD,
    convection_resolution,
    minimum_domain_points,
    require_convection_resolution,
)


class ResolutionTests(unittest.TestCase):
    def test_m2b_grid_is_diagnosed_as_below_nyquist(self):
        """Das Gitter, das den M2b-Slice entwertet hat, muss erkannt werden."""
        report = convection_resolution(196, beta=50.0)
        self.assertEqual(report.time_samples, 14)
        self.assertAlmostEqual(report.periods_in_time, 7.96, places=2)
        self.assertLess(report.samples_per_period, 2.0)
        self.assertFalse(report.adequate)

    def test_am26_point_count_is_rejected_as_marginal(self):
        """400 Punkte liegen ueber Nyquist, aber unter unserer Forderung."""
        report = convection_resolution(400, beta=50.0)
        self.assertGreater(report.samples_per_period, 2.0)
        self.assertLess(report.samples_per_period, MIN_SAMPLES_PER_PERIOD)
        self.assertFalse(report.adequate)

    def test_minimum_domain_points_for_beta_50(self):
        self.assertEqual(minimum_domain_points(beta=50.0), 1024)
        report = convection_resolution(1024, beta=50.0)
        self.assertTrue(report.adequate)

    def test_minimum_scales_with_beta(self):
        low = minimum_domain_points(beta=10.0)
        high = minimum_domain_points(beta=50.0)
        self.assertLess(low, high)

    def test_guard_raises_and_names_the_requirement(self):
        with self.assertRaises(ValueError) as ctx:
            require_convection_resolution(196, beta=50.0)
        self.assertIn("1024", str(ctx.exception))

    def test_pde_refuses_underresolved_config_by_default(self):
        config = PDEConfig(name="convection", domain_points=196, beta=50.0)
        with self.assertRaises(ValueError):
            ConvectionPDE(config)

    def test_pde_accepts_underresolved_only_with_explicit_opt_in(self):
        config = PDEConfig(
            name="convection", domain_points=196, beta=50.0, allow_underresolved=True
        )
        pde = ConvectionPDE(config)
        self.assertFalse(pde.resolution.adequate)

    def test_adequate_config_passes_without_opt_in(self):
        config = PDEConfig(name="convection", domain_points=4096, beta=50.0)
        pde = ConvectionPDE(config)
        self.assertTrue(pde.resolution.adequate)


if __name__ == "__main__":
    unittest.main()
