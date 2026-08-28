import math
import unittest

from analysis.diagnose_feasibility import PinnSettings, gradient_snapshot, supervised_capacity


class FeasibilityDiagnosticTests(unittest.TestCase):
    def test_gradient_snapshot_is_finite_and_separates_terms(self) -> None:
        settings = PinnSettings(
            domain_points=16,
            boundary_points=4,
            initial_points=4,
            hidden_dim=8,
            num_layers=1,
            regularization="double_backprop",
            allow_underresolved=True,
        )
        snapshot = gradient_snapshot(10.0, settings)
        self.assertEqual(snapshot["beta"], 10.0)
        for value in snapshot["losses"].values():
            self.assertTrue(math.isfinite(value))
        for value in snapshot["gradient_norms"].values():
            self.assertTrue(math.isfinite(value))

    def test_supervised_capacity_artifact_is_explicitly_non_study(self) -> None:
        settings = PinnSettings(
            domain_points=16,
            boundary_points=4,
            initial_points=4,
            hidden_dim=8,
            num_layers=1,
            allow_underresolved=True,
        )
        result = supervised_capacity(
            1.0,
            settings,
            max_iters=2,
            learning_rate=1.0e-3,
            grid_size=4,
        )
        self.assertIn("NOT_STUDY_DATA", result["note"])
        self.assertEqual(result["diagnostic"], "supervised_capacity")
        self.assertEqual(len(result["history"]), 2)


if __name__ == "__main__":
    unittest.main()
