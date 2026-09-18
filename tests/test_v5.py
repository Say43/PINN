import unittest
import tempfile
from dataclasses import replace
from pathlib import Path

from bench.v5 import LAMBDA_CANDIDATES, lambda_selection_conditions, study_conditions, validate_v5_base
from src.config import ExperimentConfig
from kaggle.v5_runner import V5QuotaState, _verify_frozen_preregistration


class V5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base = ExperimentConfig.from_json(Path("configs/reaction_v5.json"))

    def test_lambda_selection_is_exactly_the_preregistered_baseline_grid(self) -> None:
        conditions = list(lambda_selection_conditions(self.base))
        self.assertEqual([item.train.lambda_r for item in conditions], list(LAMBDA_CANDIDATES))
        self.assertTrue(all(item.model.backbone == "mlp" for item in conditions))
        self.assertTrue(all(item.train.precision == "fp32" for item in conditions))
        self.assertTrue(all(item.train.regularization == "double_backprop" for item in conditions))
        self.assertTrue(all(item.train.seed == 0 for item in conditions))
        self.assertTrue(all("NOT_STUDY_DATA" in item.persistence.study_id for item in conditions))

    def test_study_matrix_has_60_unique_conditions(self) -> None:
        conditions = list(
            study_conditions(self.base, lambda_r=1.0e-4, study_id="reaction_v5_frozen")
        )
        keys = {
            (item.model.backbone, item.train.precision, item.train.regularization, item.train.seed)
            for item in conditions
        }
        self.assertEqual(len(conditions), 60)
        self.assertEqual(len(keys), 60)
        self.assertTrue(all(item.pde.name == "reaction" for item in conditions))
        self.assertTrue(all(item.pde.reaction == 5.25 for item in conditions))
        self.assertTrue(all(item.train.max_iters == 2000 for item in conditions))

    def test_draft_id_cannot_be_used_for_study_matrix(self) -> None:
        with self.assertRaises(ValueError):
            list(study_conditions(self.base, lambda_r=1.0e-4, study_id="reaction_v5_DRAFT"))

    def test_v5_rejects_legacy_graph_context(self) -> None:
        legacy = replace(self.base, model=replace(self.base.model, graph_context="legacy_detached"))
        with self.assertRaises(ValueError):
            validate_v5_base(legacy)

    def test_matrix_preflight_rejects_draft_preregistration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prereg.md"
            path.write_text("**Status: ENTWURF, NICHT eingefroren.**", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                _verify_frozen_preregistration(path, "irrelevant")

    def test_v5_quota_preserves_reserve_and_phase_limits(self) -> None:
        state = V5QuotaState(actual_matrix_hours=7.3)
        self.assertFalse(state.can_start_batch("matrix", 0.2))
        self.assertTrue(state.can_start_batch("matrix", 0.09))
