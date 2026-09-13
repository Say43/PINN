from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.config import ExperimentConfig, torch_dtype
from src.metrics import censored_relative_l2, is_success, relative_l2
from src.models import build_knn_graph, count_parameters, create_model
from src.persistence import ResultStore, trial_key
from src.pdes import create_pde


class LocalGpuTimeLimit(RuntimeError):
    pass


@dataclass(frozen=True)
class TrainingResult:
    status: str
    values: dict


def set_determinism(seed: int, *, use_cuda: bool = False) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False


def _is_local_cuda(device: torch.device) -> bool:
    return device.type == "cuda" and "KAGGLE_KERNEL_RUN_TYPE" not in os.environ


def _runtime_metadata(device: torch.device) -> dict:
    metadata = {
        "device": str(device),
        "torch_version": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
    }
    if device.type == "cuda":
        metadata["device_name"] = torch.cuda.get_device_name(device)
        metadata["device_capability"] = list(torch.cuda.get_device_capability(device))
    else:
        metadata["device_name"] = "cpu"
    return metadata


def _evaluate(model, pde, config, device, dtype):
    evaluation_coords, reference = pde.evaluation_grid(device=device, dtype=dtype)
    topology = None
    if config.model.backbone != "mlp":
        topology = build_knn_graph(
            pde.normalize_coords(evaluation_coords), k=config.model.graph_k
        ).to(device)
    with torch.no_grad():
        prediction = model(evaluation_coords, topology)
        value = float(relative_l2(prediction, reference).cpu())
    return value


def train_once(config: ExperimentConfig) -> TrainingResult:
    device = torch.device(config.train.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    set_determinism(config.train.seed, use_cuda=device.type == "cuda")
    dtype = torch_dtype(config.train.precision)
    pde = create_pde(config.pde)
    batch = pde.collocation().to(device=device, dtype=dtype)
    topology = None
    if config.model.backbone != "mlp":
        topology = build_knn_graph(
            pde.normalize_coords(batch.coords), k=config.model.graph_k
        ).to(device)
    model = create_model(config.model, pde.bounds).to(device=device, dtype=dtype)
    parameter_count = count_parameters(model)
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=config.train.learning_rate,
        max_iter=1,
        max_eval=25,
        tolerance_grad=config.train.tolerance_grad,
        tolerance_change=config.train.tolerance_change,
        history_size=config.train.history_size,
        line_search_fn=config.train.line_search_fn,
    )
    started = time.perf_counter()
    function_evaluations = 0
    history: list[dict] = []
    last_breakdown = None
    time_to_target = None

    for iteration in range(1, config.train.max_iters + 1):
        def closure():
            nonlocal function_evaluations, last_breakdown
            optimizer.zero_grad(set_to_none=True)
            last_breakdown = pde.loss(
                model,
                batch,
                topology,
                config.train.regularization,
                config.train.lambda_r,
            )
            if not torch.isfinite(last_breakdown.total):
                raise FloatingPointError("non-finite training loss")
            last_breakdown.total.backward()
            function_evaluations += 1
            return last_breakdown.total

        optimizer.step(closure)
        elapsed = time.perf_counter() - started
        if _is_local_cuda(device) and elapsed >= config.train.max_local_gpu_seconds:
            raise LocalGpuTimeLimit(
                f"local GPU limit reached after {elapsed:.1f}s; limit is "
                f"{config.train.max_local_gpu_seconds:.1f}s"
            )
        should_log = iteration == 1 or iteration % config.train.log_every == 0
        should_log = should_log or iteration == config.train.max_iters
        if should_log:
            error = _evaluate(model, pde, config, device, dtype)
            if time_to_target is None and is_success(error):
                time_to_target = elapsed
            entry = {
                "iteration": iteration,
                "function_evaluations": function_evaluations,
                "wall_seconds": elapsed,
                "relative_l2": error,
            }
            if last_breakdown is not None:
                entry.update(last_breakdown.scalars())
            history.append(entry)

    wall_seconds = time.perf_counter() - started
    final_error = _evaluate(model, pde, config, device, dtype)
    clipped_error, censored = censored_relative_l2(final_error)
    final_metrics = last_breakdown.scalars() if last_breakdown is not None else {}
    final_metrics["relative_l2_raw"] = final_error
    final_metrics["relative_l2_log10"] = math.log10(clipped_error)
    values = {
        "wall_seconds": wall_seconds,
        "iterations": config.train.max_iters,
        "function_evaluations": function_evaluations,
        "solver_nfe": getattr(model, "ode_function_evaluations", None),
        "parameter_count": parameter_count,
        "relative_l2": clipped_error,
        "relative_l2_censored": censored,
        "success": is_success(final_error),
        "time_to_target_seconds": time_to_target,
        "final_metrics": final_metrics,
        "history": history,
        "runtime": _runtime_metadata(device),
    }
    status = "numerical_fail" if censored and not math.isfinite(final_error) else "completed"
    return TrainingResult(status, values)


def run(config: ExperimentConfig, *, force: bool = False) -> dict:
    with ResultStore(config.persistence.database, config.persistence.backup) as store:
        store.recover_stale_runs()
        key = trial_key(config)
        if not force and store.has_terminal_result(key):
            return {"status": "skipped", "trial_key": key, "reason": "terminal result exists"}
        run_id, key, attempt_no = store.start_attempt(config)
        runtime = {"device": config.train.device}
        try:
            runtime = _runtime_metadata(torch.device(config.train.device))
            result = train_once(config)
        except (torch.cuda.OutOfMemoryError, LocalGpuTimeLimit, KeyboardInterrupt) as error:
            store.finish_attempt(
                run_id,
                status="infra_fail",
                values={
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "runtime": runtime,
                },
            )
            raise
        except FloatingPointError as error:
            values = {
                "relative_l2": 10.0,
                "relative_l2_censored": True,
                "success": False,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "runtime": runtime,
            }
            store.finish_attempt(run_id, status="numerical_fail", values=values)
            return {"status": "numerical_fail", "trial_key": key, "attempt_no": attempt_no}
        except Exception as error:
            store.finish_attempt(
                run_id,
                status="infra_fail",
                values={
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "runtime": runtime,
                },
            )
            raise
        store.finish_attempt(run_id, status=result.status, values=result.values)
        return {
            "status": result.status,
            "trial_key": key,
            "attempt_no": attempt_no,
            **result.values,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one preregistered PINN condition")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_json(args.config)
    print(json.dumps(run(config, force=args.force), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
