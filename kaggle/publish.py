from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


class KagglePublisher:
    """Fail-closed publisher for durable per-run Kaggle Dataset versions."""

    def __init__(self, handle: str, publish_dir: str | Path = "/kaggle/working/publish") -> None:
        if "/" not in handle:
            raise ValueError("results Dataset handle must be '<owner>/<slug>'")
        self.handle = handle
        self.publish_dir = Path(publish_dir)
        self._seed_from_mounted_dataset()

    @property
    def mounted_dir(self) -> Path:
        return Path("/kaggle/input") / self.handle.split("/", 1)[1]

    def _seed_from_mounted_dataset(self) -> None:
        self.publish_dir.mkdir(parents=True, exist_ok=True)
        if not self.mounted_dir.exists():
            return
        for source in self.mounted_dir.iterdir():
            if source.is_file():
                shutil.copy2(source, self.publish_dir / source.name)

    def restore(self, *targets: str | Path) -> None:
        """Restore named durable artifacts into /kaggle/working when available."""
        for target in targets:
            target_path = Path(target)
            source = self.mounted_dir / target_path.name
            if source.exists() and not target_path.exists():
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target_path)

    def publish(
        self,
        *,
        database_backup: str | Path,
        quota_state: str | Path,
        note: str,
        extra_files: tuple[str | Path, ...] = (),
    ) -> None:
        try:
            import kagglehub
        except ImportError as error:
            raise RuntimeError("kagglehub is required for durable result publication") from error
        self.publish_dir.mkdir(parents=True, exist_ok=True)
        backup = Path(database_backup)
        if not backup.exists():
            raise FileNotFoundError(f"database backup missing: {backup}")
        shutil.copy2(backup, self.publish_dir / "results.sqlite")
        state = Path(quota_state)
        if state.exists():
            shutil.copy2(state, self.publish_dir / "quota_state.json")
        for source in extra_files:
            source_path = Path(source)
            if source_path.exists():
                shutil.copy2(source_path, self.publish_dir / source_path.name)
        manifest = {
            "published_at": datetime.now(timezone.utc).isoformat(),
            "note": note,
            "database": "results.sqlite",
        }
        (self.publish_dir / "publication.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        kagglehub.dataset_upload(self.handle, str(self.publish_dir), version_notes=note)

    def publish_files(self, *, files: tuple[str | Path, ...], note: str) -> None:
        try:
            import kagglehub
        except ImportError as error:
            raise RuntimeError("kagglehub is required for durable result publication") from error
        self.publish_dir.mkdir(parents=True, exist_ok=True)
        copied = []
        for source in files:
            source_path = Path(source)
            if not source_path.exists():
                raise FileNotFoundError(f"publication artifact missing: {source_path}")
            target = self.publish_dir / source_path.name
            shutil.copy2(source_path, target)
            copied.append(target.name)
        manifest = {
            "published_at": datetime.now(timezone.utc).isoformat(),
            "note": note,
            "files": copied,
        }
        (self.publish_dir / "publication.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        kagglehub.dataset_upload(self.handle, str(self.publish_dir), version_notes=note)
