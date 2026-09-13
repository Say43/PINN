import unittest
from pathlib import Path

from bench.budget import QuotaState
from bench.calibrate import calibration_config
from bench.plan import SAFETY_FACTOR, make_plan, summarize
from kaggle.build_notebook import launcher_source
from kaggle.runner import stage_conditions
from src.config import ExperimentConfig


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

    def test_plan_applies_full_pruning_hierarchy(self) -> None:
        rows = synthetic_rows(seconds_per_iteration=1.0)
        for hardware in ("p100", "2xt4"):
            for backbone in ("mlp", "gread"):
                for precision in ("fp32", "fp64"):
                    for repeat in range(3):
                        rows.append({
                            "hardware": hardware,
                            "point_scheme": "reduced",
                            "backbone": backbone,
                            "precision": precision,
                            "seconds_per_iteration": 0.08,
                            "repeat": repeat,
                        })
        plan = make_plan(rows)
        self.assertEqual(plan["action"], "run_m2b_stage_b_dropped_no_regularization")
        self.assertEqual(plan["selected"]["max_iters"], 5000)
        self.assertEqual(plan["selected"]["regularizations"], ["none"])

    def test_plan_uses_latest_schema_and_two_repeat_minimum(self) -> None:
        rows = [
            {
                "hardware": "2xt4", "point_scheme": "full", "backbone": "gread",
                "precision": "fp32", "seconds_per_iteration": value,
                "schema_version": schema,
            }
            for schema, value in ((1, 1.0), (1, 1.1), (1, 1.2), (2, 0.1), (2, 0.2))
        ]
        medians = summarize(rows)
        self.assertAlmostEqual(medians[("2xt4", "full", "gread", "fp32")], 0.15)
        self.assertEqual(SAFETY_FACTOR, 1.0)

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
            "m2b", "owner/pinn-code", "owner/pinn-results", "owner/pinn-m2-plan", "p100",
            source_commit="0123456789abcdef",
        )
        self.assertIn("/kaggle/input/pinn-code", source)
        self.assertIn("python', '-m', 'kaggle.runner", source)
        self.assertIn('inputs.rglob("m2_plan.json")', source)
        self.assertIn("torch==2.5.1", source)
        self.assertIn('PINN_SOURCE_COMMIT"] = "0123456789abcdef"', source)
        self.assertNotIn("class PDEGraphNet", source)

    def test_t4_launcher_does_not_replace_kaggle_torch(self) -> None:
        source = launcher_source(
            "m2a", "owner/pinn-code", "owner/pinn-results", None, "2xt4"
        )
        self.assertNotIn("download.pytorch.org", source)

    def test_v5_lambda_launcher_uses_two_gpu_parent_runner(self) -> None:
        source = launcher_source(
            "v5_lambda", "owner/pinn-code", "owner/pinn-results", None, "2xt4",
            source_commit="0123456789abcdef",
        )
        self.assertIn("kaggle.v5_runner", source)
        self.assertIn("'--phase', 'lambda'", source)
        self.assertIn("'--workers', '2'", source)

    def test_v5_matrix_launcher_requires_frozen_inputs(self) -> None:
        with self.assertRaises(ValueError):
            launcher_source(
                "v5_matrix", "owner/pinn-code", "owner/pinn-results", None, "2xt4"
            )

    def test_v5_launcher_rejects_single_gpu_hardware(self) -> None:
        with self.assertRaises(ValueError):
            launcher_source(
                "v5_lambda", "owner/pinn-code", "owner/pinn-results", None, "p100"
            )

    def test_calibration_uses_minimal_valid_evaluation_grid(self) -> None:
        base = ExperimentConfig.from_json(Path("configs/stage_a.json"))
        config = calibration_config(
            base,
            backbone="gread",
            precision="fp32",
            repeat=0,
            device="cpu",
            point_scheme="full",
        )
        self.assertEqual((config.pde.evaluation_x, config.pde.evaluation_t), (3, 3))
        self.assertGreaterEqual(config.pde.evaluation_x * config.pde.evaluation_t, config.model.graph_k + 1)
