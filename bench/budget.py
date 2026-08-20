from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class QuotaState:
    total_quota_hours: float = 5.0
    reserve_hours: float = 1.0
    calibration_limit_hours: float = 0.5
    stage_a_limit_hours: float = 2.0
    stage_b_limit_hours: float = 1.5
    actual_calibration_hours: float = 0.0
    actual_stage_a_hours: float = 0.0
    actual_stage_b_hours: float = 0.0

    @classmethod
    def load(cls, path: str | Path) -> "QuotaState":
        path = Path(path)
        if not path.exists():
            return cls()
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def can_start(self, stage: str, conservative_hours: float) -> bool:
        actual = getattr(self, f"actual_{stage}_hours")
        limit = getattr(self, f"{stage}_limit_hours")
        total_actual = self.actual_calibration_hours + self.actual_stage_a_hours + self.actual_stage_b_hours
        return actual + conservative_hours <= limit and total_actual + conservative_hours <= (
            self.total_quota_hours - self.reserve_hours
        )

    def record(self, stage: str, actual_hours: float) -> None:
        attribute = f"actual_{stage}_hours"
        setattr(self, attribute, getattr(self, attribute) + actual_hours)
