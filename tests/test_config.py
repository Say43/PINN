import unittest
from pathlib import Path

from src.config import ExperimentConfig


class ConfigTests(unittest.TestCase):
    def test_smoke_config_has_exactly_100_loss_points(self) -> None:
        config = ExperimentConfig.from_json(Path("configs/smoke.json"))
        self.assertEqual(
            config.pde.domain_points + config.pde.boundary_points + config.pde.initial_points,
            100,
        )
        self.assertEqual(config.train.device, "cpu")
        self.assertEqual(config.train.max_iters, 200)

    def test_condition_override_is_immutable(self) -> None:
        base = ExperimentConfig()
        changed = base.with_condition(backbone="gread", precision="fp64", seed=4)
        self.assertEqual(base.model.backbone, "mlp")
        self.assertEqual(changed.model.backbone, "gread")
        self.assertEqual(changed.train.precision, "fp64")
        self.assertEqual(changed.train.seed, 4)
