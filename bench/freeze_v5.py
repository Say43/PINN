"""Friert PREREGISTRATION-V5.md ein und schreibt die Lock-Datei.

Der Freeze ist mechanisch: Statuszeile setzen, SHA-256 des Dokuments bilden,
`PREREGISTRATION-V5.lock.json` schreiben. Der Freeze-Commit wird danach mit
`--record-commit` nachgetragen, weil er erst existiert, wenn das Dokument
committet ist.

    python -m bench.freeze_v5 --lambda-r 1e-4 --study-id reaction_v5_frozen
    python -m bench.freeze_v5 --record-commit
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import date
from pathlib import Path

from bench.v5 import LAMBDA_CANDIDATES

DOCUMENT = Path("PREREGISTRATION-V5.md")
LOCK = Path("PREREGISTRATION-V5.lock.json")
DRAFT_MARKERS = ("Status: ENTWURF", "NICHT eingefroren")


def canonical_text(text: str) -> str:
    """The form Git stores: LF line endings.

    On Windows the working copy carries CRLF, but `kaggle/build_v5_payload.py`
    ships the blob that `git show` returns. Hashing the working copy would yield a
    digest the Kaggle preflight cannot reproduce, and the matrix would refuse to
    start.
    """
    return text.replace("\r\n", "\n")


def document_digest() -> str:
    return hashlib.sha256(
        canonical_text(DOCUMENT.read_text(encoding="utf-8")).encode("utf-8")
    ).hexdigest()


def freeze_header(lambda_r: float, study_id: str, frozen_at: str) -> str:
    return (
        f"# Präregistrierung V5 — Reaction an der Kollapskante\n\n"
        f"**Status: EINGEFROREN.** Eingefroren am **{frozen_at}**; ab jetzt keine\n"
        f"inhaltliche Änderung mehr an diesem Dokument. Jede Abweichung wandert nach\n"
        f"`DEVIATIONS.md` mit Datum, Begründung und der Angabe, ob die betroffenen\n"
        f"Ergebnisse zum Zeitpunkt der Änderung bereits gesichtet waren.\n\n"
        f"Studien-ID: `{study_id}` · λ_r = {lambda_r:g}, ausgewählt nach Abschnitt 5\n"
        f"vor dem Freeze. SHA-256 und Freeze-Commit: siehe `{LOCK.name}`.\n\n"
        f"V3 (eingefroren, Tag `prereg-v3`) und V4 (Entwurf, nie ausgeführt) bleiben als\n"
        f"Dokumente ihrer jeweiligen Studie bestehen. V4 wurde nicht gefahren, weil ihre\n"
        f"Auswahlregel ein Fehlerband voraussetzte, das bei Convection nicht existiert.\n"
        f"V5 ersetzt sie auf Basis von Messungen, nicht Annahmen.\n"
    )


def freeze(lambda_r: float, study_id: str, frozen_at: str) -> dict:
    if lambda_r not in LAMBDA_CANDIDATES:
        raise SystemExit(f"lambda_r must be one of {LAMBDA_CANDIDATES}")
    if not study_id or "DRAFT" in study_id or "NOT_STUDY_DATA" in study_id:
        raise SystemExit("study_id must identify the frozen study")
    text = DOCUMENT.read_text(encoding="utf-8")
    if not any(marker in text for marker in DRAFT_MARKERS):
        raise SystemExit("document does not carry a draft status; refusing to freeze twice")

    canonical = canonical_text(text)
    marker = "\n---\n"
    index = canonical.index(marker)
    frozen = freeze_header(lambda_r, study_id, frozen_at) + canonical[index:]
    DOCUMENT.write_text(frozen, encoding="utf-8", newline="\n")

    for draft_marker in DRAFT_MARKERS:
        if draft_marker in frozen:
            raise SystemExit(f"draft marker survived the freeze: {draft_marker!r}")
    digest = document_digest()
    lock = {
        "schema_version": 1,
        "document": DOCUMENT.name,
        "frozen_at": frozen_at,
        "freeze_commit": None,
        "sha256": digest,
        "git_tag": "prereg-v5",
        "lambda_r": lambda_r,
        "study_id": study_id,
    }
    LOCK.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return lock


def record_commit() -> dict:
    """Record the freeze commit and verify the lock against the committed blob."""
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    digest = document_digest()
    if digest != lock["sha256"]:
        raise SystemExit(
            f"document changed after the freeze: locked {lock['sha256']}, found {digest}"
        )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, encoding="utf-8"
    ).strip()
    blob_digest = hashlib.sha256(
        subprocess.check_output(["git", "show", f"{commit}:{DOCUMENT.name}"])
    ).hexdigest()
    if blob_digest != digest:
        raise SystemExit(
            f"committed blob differs from the locked document: blob {blob_digest}, "
            f"locked {digest}"
        )
    lock["freeze_commit"] = commit
    LOCK.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return lock


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambda-r", type=float)
    parser.add_argument("--study-id")
    parser.add_argument("--frozen-at", default=date.today().isoformat())
    parser.add_argument("--record-commit", action="store_true")
    args = parser.parse_args()

    if args.record_commit:
        lock = record_commit()
    else:
        if args.lambda_r is None or not args.study_id:
            raise SystemExit("freezing requires --lambda-r and --study-id")
        lock = freeze(args.lambda_r, args.study_id, args.frozen_at)
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
