from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


BackboneName = Literal["mlp", "grand", "gread"]
PrecisionName = Literal["fp32", "fp64"]
RegularizationName = Literal["none", "double_backprop"]
PDEName = Literal["convection", "allen_cahn", "reaction"]


@dataclass(frozen=True)
class PDEConfig:
    name: PDEName = "convection"
    domain_points: int = 64
    boundary_points: int = 18
    initial_points: int = 18
    evaluation_x: int = 41
    evaluation_t: int = 41
    beta: float = 50.0
    diffusion: float = 1.0e-4
    reaction: float = 5.0
    reference_path: str | None = None
    # Nur fuer Tests und Smoke-Laeufe. Studienlaeufe muessen die
    # Aufloesungspruefung bestehen (src/pdes/resolution.py, FINDINGS.md §2).
    # Der Wert steht im config_json jedes Laufs, damit eine gesetzte Ausnahme
    # niemals unbemerkt als Studienevidenz durchgeht.
    allow_underresolved: bool = False

    def __post_init__(self) -> None:
        for key in ("domain_points", "boundary_points", "initial_points"):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if self.evaluation_x < 2 or self.evaluation_t < 2:
            raise ValueError("evaluation grids must have at least two points per axis")


@dataclass(frozen=True)
class ModelConfig:
    backbone: BackboneName = "mlp"
    graph_context: Literal["legacy_detached", "fixed_support"] = "legacy_detached"
    mlp_hidden_dim: int = 128
    graph_hidden_dim: int = 56
    num_layers: int = 4
    graph_k: int = 8
    diffusion_step: float = 0.25

    def __post_init__(self) -> None:
        if self.graph_context not in {"legacy_detached", "fixed_support"}:
            raise ValueError("invalid graph_context")
        if self.num_layers < 1:
            raise ValueError("num_layers must be positive")
        if self.graph_k < 1:
            raise ValueError("graph_k must be positive")


@dataclass(frozen=True)
class TrainConfig:
    precision: PrecisionName = "fp32"
    regularization: RegularizationName = "none"
    seed: int = 0
    max_iters: int = 200
    history_size: int = 100
    learning_rate: float = 1.0
    line_search_fn: str | None = "strong_wolfe"
    tolerance_grad: float = 0.0
    tolerance_change: float = 0.0
    lambda_r: float = 1.0
    log_every: int = 25
    device: str = "cpu"
    max_local_gpu_seconds: float = 175.0

    def __post_init__(self) -> None:
        if self.max_iters < 1:
            raise ValueError("max_iters must be positive")
        if self.device.startswith("cuda") and self.max_local_gpu_seconds >= 180:
            raise ValueError("local GPU safety limit must be below 180 seconds")
        if self.regularization == "double_backprop" and self.lambda_r <= 0:
            raise ValueError("lambda_r must be positive for double_backprop")


@dataclass(frozen=True)
class PersistenceConfig:
    database: str = "results/results.sqlite"
    backup: str = "results/results.backup.sqlite"
    study_id: str = "pde_attention_pilot_v3"


@dataclass(frozen=True)
class ExperimentConfig:
    pde: PDEConfig = field(default_factory=PDEConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ExperimentConfig":
        return cls(
            pde=PDEConfig(**raw.get("pde", {})),
            model=ModelConfig(**raw.get("model", {})),
            train=TrainConfig(**raw.get("train", {})),
            persistence=PersistenceConfig(**raw.get("persistence", {})),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def canonical_json(self) -> str:
        raw = self.to_dict()
        # Preserve pre-existing V3 trial keys for the unchanged legacy path.
        if self.model.graph_context == "legacy_detached":
            raw["model"].pop("graph_context")
        return json.dumps(raw, sort_keys=True, separators=(",", ":"))

    def with_condition(
        self,
        *,
        backbone: BackboneName | None = None,
        precision: PrecisionName | None = None,
        regularization: RegularizationName | None = None,
        seed: int | None = None,
    ) -> "ExperimentConfig":
        raw = self.to_dict()
        if backbone is not None:
            raw["model"]["backbone"] = backbone
        if precision is not None:
            raw["train"]["precision"] = precision
        if regularization is not None:
            raw["train"]["regularization"] = regularization
        if seed is not None:
            raw["train"]["seed"] = seed
        return ExperimentConfig.from_dict(raw)


def torch_dtype(precision: PrecisionName):
    import torch

    return torch.float32 if precision == "fp32" else torch.float64
