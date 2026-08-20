import unittest

from bench.select_lambda import CANDIDATES, TARGET_RATIO, select_lambda


class LambdaSelectionTests(unittest.TestCase):
    def test_selection_is_from_preregistered_grid_and_outcome_blind(self) -> None:
        result = select_lambda()
        self.assertIn(result["selected_lambda_r"], CANDIDATES)
        self.assertEqual(result["target_ratio"], TARGET_RATIO)
        self.assertNotIn("relative_l2", result)
        self.assertEqual(result["device"], "cpu")
