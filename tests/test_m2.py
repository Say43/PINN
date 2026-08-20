import unittest

from bench.budget import QuotaState
from bench.plan import make_plan
from kaggle.build_notebook import launcher_source
from kaggle.runner import stage_conditions


def synthetic_rows(seconds_per_iteration: float = 0.01) -> list[dict]:
    rows = []
    for hardware in ("p100", "2xt4"):
        for backbone in ("mlp", "gread"):
            for precision in ("fp32", "fp64"):
                multiplier = 1.0
                if hardware == "2xt4" and precision == "fp64":
                    multiplier = 10.0
                for repeat in range(3):
                    rows.append(
                        {
                            "hardware": hardware,
                            "point_scheme": "full",
                            "backbone": backbone,
                            "precision": precision,
                            "seconds_per_iteration": seconds_per_iteration * multiplier,
                            "repeat": repeat,
                        }
                    )
    return rows


class M2Tests(unittest.TestCase):
    def test_plan_prefers_p100_for_fp64_heavy_matrix(self) -> None:
        plan = make_plan(synthetic_rows())
        self.assertEqual(plan["action"], "run_m2b")
        self.assertEqual(plan["selected"]["hardware"], "p100")
        self.assertGreaterEqual(plan["selected"]["max_iters"], 5000)
        self.assertEqual(plan["selected"]["workers"], 1)

    def test_plan_requests_reduced_recalibration_when_full_is_too_slow(self) -> None:
        plan = make_plan(synthetic_rows(seconds_per_iteration=1.0))
        self.assertEqual(plan["action"], "rerun_m2a_with_reduced_points")

    def test_quota_guard_preserves_reserve_and_stage_limits(self) -> None:
        state = QuotaState(actual_stage_a_hours=1.9)
        self.assertFalse(state.can_start("stage_a", 0.2))
        self.assertTrue(state.can_start("stage_a", 0.09))
        state = QuotaState(actual_calibration_hours=0.5, actual_stage_a_hours=2.0, actual_stage_b_hours=1.5)
        self.assertFalse(state.can_start("stage_b", 0.01))

    def test_stage_queue_sizes(self) -> None:
        self.assertEqual(len(list(stage_conditions("m2b"))), 12)
        self.assertEqual(len(list(stage_conditions("stage_a"))), 48)
        self.assertEqual(len(list(stage_conditions("stage_b"))), 60)

    def test_notebook_launcher_contains_no_project_source(self) -> None:
        source = launcher_source(
            "m2b", "owner/pinn-code", "owner/pinn-results", "owner/pinn-m2-plan", "p100"
        )
        self.assertIn("/kaggle/input/pinn-code", source)
        self.assertIn("python', '-m', 'kaggle.runner", source)
        self.assertNotIn("class PDEGraphNet", source)
