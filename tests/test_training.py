import tempfile
import unittest
from pathlib import Path

from src.config import ExperimentConfig, ModelConfig, PDEConfig, PersistenceConfig, TrainConfig
from src.train import run, train_once


class TrainingTests(unittest.TestCase):
    def test_all_stage_a_condition_families_take_one_step(self) -> None:
        for pde_name in ("convection", "allen_cahn"):
            for backbone in ("mlp", "grand", "gread"):
                for precision in ("fp32", "fp64"):
                    for regularization in ("none", "double_backprop"):
                        with self.subTest(
                            pde=pde_name,
                            backbone=backbone,
                            precision=precision,
                            regularization=regularization,
                        ):
                            reference_path = "data/allen_cahn.mat" if pde_name == "allen_cahn" else None
                            config = ExperimentConfig(
                                pde=PDEConfig(allow_underresolved=True, 
                                    name=pde_name,
                                    domain_points=9,
                                    boundary_points=3,
                                    initial_points=3,
                                    evaluation_x=5,
                                    evaluation_t=5,
                                    reference_path=reference_path,
                                ),
                                model=ModelConfig(
                                    backbone=backbone,
                                    mlp_hidden_dim=8,
                                    graph_hidden_dim=6,
                                    num_layers=1,
                                    graph_k=2,
                                ),
                                train=TrainConfig(
                                    precision=precision,
                                    regularization=regularization,
                                    max_iters=1,
                                    history_size=2,
                                    line_search_fn=None,
                                    log_every=1,
                                    device="cpu",
                                ),
                            )
                            result = train_once(config)
                            self.assertEqual(result.status, "completed")
                            self.assertEqual(result.values["iterations"], 1)
                            self.assertGreaterEqual(result.values["function_evaluations"], 1)

    def test_run_skips_existing_terminal_trial(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            config = ExperimentConfig(
                pde=PDEConfig(allow_underresolved=True, 
                    domain_points=4,
                    boundary_points=2,
                    initial_points=2,
                    evaluation_x=3,
                    evaluation_t=3,
                ),
                model=ModelConfig(mlp_hidden_dim=4, num_layers=1),
                train=TrainConfig(max_iters=1, history_size=2, line_search_fn=None, log_every=1),
                persistence=PersistenceConfig(
                    database=str(tmp_path / "results.sqlite"),
                    backup=str(tmp_path / "backup.sqlite"),
                    study_id="resume-test",
                ),
            )
            first = run(config)
            second = run(config)
            self.assertEqual(first["status"], "completed")
            self.assertEqual(second["status"], "skipped")
