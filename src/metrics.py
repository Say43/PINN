from __future__ import annotations

import math

import torch


def relative_l2(prediction: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    denominator = torch.linalg.vector_norm(reference)
    if denominator == 0:
        raise ValueError("relative L2 is undefined for a zero reference")
    return torch.linalg.vector_norm(prediction - reference) / denominator


def censored_relative_l2(value: float) -> tuple[float, bool]:
    if not math.isfinite(value):
        return 10.0, True
    clipped = min(10.0, max(1.0e-8, value))
    return clipped, clipped != value


def is_success(value: float, threshold: float = 0.10) -> bool:
    return math.isfinite(value) and value < threshold
