import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import bench.freeze_v5 as freeze_v5
from kaggle.v5_runner import _verify_frozen_preregistration


DRAFT_HEADER = (
    "# Präregistrierung V5 — Reaction an der Kollapskante\n\n"
    "**Status: ENTWURF, NICHT eingefroren.** Neue Studie.\n"
)


def draft_from_live_document(origin: Path) -> str:
    """A draft built from the real document body.

    The live preregistration is frozen since 2026-09-22, and freezing it again is
    refused by design. The tests therefore put a draft header in front of the real
    sections, so they still exercise every heading the frozen study depends on.
    """
    text = (origin / "PREREGISTRATION-V5.md").read_text(encoding="utf-8").replace("\r\n", "\n")
    return DRAFT_HEADER + text[text.index("\n---\n"):]


class LiveLockTests(unittest.TestCase):
    def test_live_preregistration_is_frozen_and_matches_its_lock(self) -> None:
        """The study ran against this hash; any later edit to the document breaks it."""
        lock = json.loads(Path("PREREGISTRATION-V5.lock.json").read_text(encoding="utf-8"))
        text = Path("PREREGISTRATION-V5.md").read_text(encoding="utf-8")
        for marker in freeze_v5.DRAFT_MARKERS:
            self.assertNotIn(marker, text)
        digest = hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()
        self.assertEqual(digest, lock["sha256"])
        self.assertEqual(lock["lambda_r"], 1.0e-4)
        self.assertEqual(lock["study_id"], "reaction_v5_rho525_20260922")


class FreezeV5Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.origin = Path.cwd()
        self.work = Path(tempfile.mkdtemp())
        (self.work / "PREREGISTRATION-V5.md").write_text(
            draft_from_live_document(self.origin), encoding="utf-8", newline="\n"
        )
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
