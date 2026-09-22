import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import bench.freeze_v5 as freeze_v5
from kaggle.v5_runner import _verify_frozen_preregistration


class FreezeV5Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.origin = Path.cwd()
        self.work = Path(tempfile.mkdtemp())
        shutil.copy(self.origin / "PREREGISTRATION-V5.md", self.work / "PREREGISTRATION-V5.md")
        os.chdir(self.work)

    def tearDown(self) -> None:
        os.chdir(self.origin)
        shutil.rmtree(self.work, ignore_errors=True)

    def test_freeze_removes_draft_markers_and_keeps_every_section(self) -> None:
        before = (self.work / "PREREGISTRATION-V5.md").read_text(encoding="utf-8")
        lock = freeze_v5.freeze(1.0e-4, "reaction_v5_frozen", "2026-09-22")
        after = (self.work / "PREREGISTRATION-V5.md").read_text(encoding="utf-8")

        for marker in freeze_v5.DRAFT_MARKERS:
            self.assertNotIn(marker, after)
        self.assertEqual(before.count("\n## "), after.count("\n## "))
        for heading in ("## 5. Wahl von λ_r", "## 6. Metriken", "## 10. Abbruchkriterien"):
            self.assertIn(heading, after)
        # The reason V4 was never run is part of the record, not of the draft status.
        self.assertIn("V4 wurde nicht gefahren", after)
        self.assertEqual(
            lock["sha256"],
            hashlib.sha256((self.work / "PREREGISTRATION-V5.md").read_bytes()).hexdigest(),
        )
        self.assertIsNone(lock["freeze_commit"])

    def test_hash_matches_the_bytes_that_reach_kaggle(self) -> None:
        """The payload carries the Git blob (LF), not the CRLF working copy."""
        document = self.work / "PREREGISTRATION-V5.md"
        document.write_text(
            document.read_text(encoding="utf-8").replace("\n", "\r\n"),
            encoding="utf-8",
            newline="",
        )
        self.assertIn(b"\r\n", document.read_bytes())

        lock = freeze_v5.freeze(1.0e-4, "reaction_v5_frozen", "2026-09-22")
        payload = document.read_bytes()
        self.assertNotIn(b"\r\n", payload)
        self.assertEqual(lock["sha256"], hashlib.sha256(payload).hexdigest())
        _verify_frozen_preregistration(document, lock["sha256"])

    def test_frozen_document_passes_the_matrix_preflight(self) -> None:
        lock = freeze_v5.freeze(1.0e-4, "reaction_v5_frozen", "2026-09-22")
        _verify_frozen_preregistration(self.work / "PREREGISTRATION-V5.md", lock["sha256"])
        with self.assertRaises(RuntimeError):
            _verify_frozen_preregistration(self.work / "PREREGISTRATION-V5.md", "0" * 64)

    def test_freezing_twice_is_refused(self) -> None:
        freeze_v5.freeze(1.0e-4, "reaction_v5_frozen", "2026-09-22")
        with self.assertRaises(SystemExit):
            freeze_v5.freeze(1.0e-4, "reaction_v5_frozen", "2026-09-22")

    def test_freeze_rejects_a_candidate_outside_the_grid_and_a_draft_study_id(self) -> None:
        with self.assertRaises(SystemExit):
            freeze_v5.freeze(3.0e-4, "reaction_v5_frozen", "2026-09-22")
        with self.assertRaises(SystemExit):
            freeze_v5.freeze(1.0e-4, "reaction_v5_DRAFT", "2026-09-22")

    def test_recording_a_commit_refuses_a_changed_document(self) -> None:
        lock = freeze_v5.freeze(1.0e-4, "reaction_v5_frozen", "2026-09-22")
        document = self.work / "PREREGISTRATION-V5.md"
        document.write_text(document.read_text(encoding="utf-8") + "\nnachtraeglich\n", encoding="utf-8")
        with self.assertRaises(SystemExit):
            freeze_v5.record_commit()
        self.assertIsNone(json.loads((self.work / "PREREGISTRATION-V5.lock.json")
                                     .read_text(encoding="utf-8"))["freeze_commit"])
        self.assertEqual(lock["git_tag"], "prereg-v5")


if __name__ == "__main__":
    unittest.main()
