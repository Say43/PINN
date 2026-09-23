import unittest

from analysis.v5_failure_modes import classify


class FailureModeTests(unittest.TestCase):
    def test_success_is_decided_by_the_preregistered_threshold_alone(self) -> None:
        self.assertEqual(classify(0.099, {"loss_base": 0.5}), "success")

    def test_stalled_run_keeps_a_large_loss(self) -> None:
        self.assertEqual(classify(0.999, {"loss_base": 0.1994}), "stalled")

    def test_low_loss_but_wrong_solution_is_its_own_mode(self) -> None:
        self.assertEqual(classify(0.999, {"loss_base": 2.8e-6}), "residual_satisfied_wrong")
        self.assertEqual(classify(0.105, {"loss_base": 6.5e-5}), "residual_satisfied_wrong")

    def test_values_between_the_two_modes_are_not_forced_into_either(self) -> None:
        self.assertEqual(classify(0.9, {"loss_base": 5.0e-3}), "other")


if __name__ == "__main__":
    unittest.main()
