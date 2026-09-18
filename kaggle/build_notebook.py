from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def launcher_source(
    mode: str,
    code_dataset: str,
    results_dataset: str,
    plan_dataset: str | None,
    hardware: str,
    point_scheme: str = "full",
    calibration_backbones: tuple[str, ...] = ("mlp", "gread"),
    calibration_precisions: tuple[str, ...] = ("fp32", "fp64"),
    calibration_repeats: int = 3,
    budget_stage: str = "calibration",
    cell_wall_budget_seconds: float = 2200.0,
    source_commit: str | None = None,
    v5_lambda_r: float | None = None,
    v5_study_id: str | None = None,
    v5_prereg_sha256: str | None = None,
    v5_batch_estimate_seconds: float | None = None,
) -> str:
    if mode in {"v5_lambda", "v5_matrix"} and hardware != "2xt4":
        raise ValueError("V5 requires Kaggle 2x T4 hardware")
    mount = code_dataset.split("/", 1)[1]
    bootstrap = ""
    if hardware == "p100":
        bootstrap = '''subprocess.run([
    sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
    "--no-cache-dir", "--force-reinstall", "torch==2.5.1",
    "--index-url", "https://download.pytorch.org/whl/cu121",
], check=True)'''
    config = "configs/stage_b.json" if mode == "stage_b" else "configs/stage_a.json"
    if mode == "m2a":
        command = [
            "python", "-m", "bench.calibrate", "--config", "configs/stage_a.json",
            "--output", f"/kaggle/working/m2a_v2_{hardware}_{point_scheme}.jsonl",
            "--hardware-label", hardware,
            "--workers", "1", "--device", "cuda:0", "--point-scheme", point_scheme,
            "--backbones", *calibration_backbones,
            "--precisions", *calibration_precisions,
            "--repeats", str(calibration_repeats), "--budget-stage", budget_stage,
            "--results-dataset", results_dataset,
        ]
    elif mode in {"v5_lambda", "v5_matrix"}:
        if v5_batch_estimate_seconds is None or v5_batch_estimate_seconds <= 0:
            raise ValueError("V5 requires a measured batch estimate before launcher generation")
        phase = "lambda" if mode == "v5_lambda" else "matrix"
        command = [
            "python", "-m", "kaggle.v5_runner", "--phase", phase,
            "--config", "configs/reaction_v5.json", "--workers", "2",
            "--wall-budget-seconds", str(cell_wall_budget_seconds),
            "--batch-estimate-seconds", str(v5_batch_estimate_seconds),
        ]
        if mode == "v5_matrix":
            if v5_lambda_r is None or not v5_study_id or not v5_prereg_sha256:
                raise ValueError("v5_matrix requires lambda_r, study_id, and preregistration hash")
            command.extend([
                "--lambda-r", str(v5_lambda_r), "--study-id", v5_study_id,
                "--prereg-sha256", v5_prereg_sha256,
            ])
    else:
        if plan_dataset is None:
            raise ValueError("non-M2a notebooks require --plan-dataset")
        command = [
            "python", "-m", "kaggle.runner", "--stage", mode, "--config", config,
            "--plan", "__M2_PLAN__",
            "--wall-budget-seconds", str(cell_wall_budget_seconds),
        ]
    verify_payload = mode in {"v5_lambda", "v5_matrix"}
    return f'''import hashlib, json, os, shutil, subprocess, sys, zipfile
from pathlib import Path

inputs = Path("/kaggle/input")
legacy_mounted = Path("/kaggle/input/{mount}")
roots = list(legacy_mounted.rglob("pyproject.toml"))
if not roots:
    roots = list(inputs.rglob("pyproject.toml"))
if len(roots) != 1:
    raise RuntimeError(f"Expected one project root, found {{len(roots)}}")
mounted = roots[0].parent
root = Path("/kaggle/working/project")
root.mkdir(parents=True, exist_ok=True)
shutil.copytree(mounted, root, dirs_exist_ok=True)
for archive in mounted.glob("*.zip"):
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(root)
if {verify_payload!r}:
    manifest = json.loads((root / "pinn_payload_manifest.json").read_text(encoding="utf-8"))
    if manifest["source_commit"] != {source_commit or 'unknown'!r}:
        raise RuntimeError("V5 mounted source commit differs from launcher")
    for item in manifest["files"]:
        payload_path = root / item["path"]
        if hashlib.sha256(payload_path.read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError(f"V5 mounted source hash mismatch: {{item['path']}}")
{bootstrap}
os.chdir(root)
sys.path.insert(0, str(root))
os.environ["PINN_RESULTS_DATASET"] = "{results_dataset}"
os.environ["PINN_SOURCE_COMMIT"] = "{source_commit or 'unknown'}"
command = {command!r}
if "__M2_PLAN__" in command:
    plan_paths = list(inputs.rglob("m2_plan.json"))
    if len(plan_paths) != 1:
        raise RuntimeError(f"Expected one M2 plan, found {{len(plan_paths)}}")
    command = [str(plan_paths[0]) if value == "__M2_PLAN__" else value for value in command]
print("launcher ready:", " ".join(command), flush=True)
'''


def runner_cell_source(index: int, total: int) -> str:
    """One execution cell.

    Kaggle executes committed notebooks through papermill/nbclient, which kills a
    single cell after 3000 s (observed in the first M2b run, which lost the sixth
    condition mid-flight). Every cell therefore re-enters the resume-aware runner,
    which skips terminal attempts and stops on its own wall budget well before the
    cell timeout can strike.
    """
    return f'''print("runner pass {index}/{total}", flush=True)
subprocess.run(command, check=True)
'''


def clean_source_commit() -> str:
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], text=True, encoding="utf-8"
    ).strip()
    if status:
        raise RuntimeError("refusing to build a provenance-bearing notebook from a dirty worktree")
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, encoding="utf-8"
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a thin Kaggle launcher notebook")
    parser.add_argument(
        "--mode",
        choices=("m2a", "m2b", "stage_a", "stage_b", "v5_lambda", "v5_matrix"),
        required=True,
    )
    parser.add_argument("--code-dataset", required=True, help="owner/slug")
    parser.add_argument("--results-dataset", required=True, help="owner/slug")
    parser.add_argument("--plan-dataset", help="owner/slug containing m2_plan.json")
    parser.add_argument("--hardware", choices=("p100", "2xt4"), default="p100")
    parser.add_argument("--point-scheme", choices=("full", "reduced"), default="full")
    parser.add_argument(
        "--calibration-backbones", nargs="+", choices=("mlp", "gread"),
        default=["mlp", "gread"],
    )
    parser.add_argument(
        "--calibration-precisions", nargs="+", choices=("fp32", "fp64"),
        default=["fp32", "fp64"],
    )
    parser.add_argument("--calibration-repeats", type=int, default=3)
    parser.add_argument(
        "--budget-stage", choices=("calibration", "stage_a"), default="calibration"
    )
    parser.add_argument(
        "--runner-cells", type=int, default=1,
        help="number of resume passes; each is its own notebook cell with its own 3000 s timeout",
    )
    parser.add_argument(
        "--cell-wall-budget", type=float, default=2200.0,
        help="wall budget per runner cell in seconds; must stay clearly below the 3000 s cell timeout",
    )
    parser.add_argument("--kernel-id", required=True, help="owner/kernel-slug")
    parser.add_argument("--v5-lambda-r", type=float)
    parser.add_argument("--v5-study-id")
    parser.add_argument("--v5-prereg-sha256")
    parser.add_argument("--v5-batch-estimate-seconds", type=float)
    parser.add_argument("--output-dir", type=Path, default=Path("kaggle/generated"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.runner_cells < 1:
        raise SystemExit("--runner-cells must be at least 1")
    if args.cell_wall_budget >= 3000.0:
        raise SystemExit("--cell-wall-budget must stay below the 3000 s Kaggle cell timeout")

    def code_cell(source: str) -> dict:
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.splitlines()],
        }

    source_commit = clean_source_commit()
    cells = [code_cell(launcher_source(
        args.mode, args.code_dataset, args.results_dataset,
        args.plan_dataset, args.hardware, args.point_scheme,
        tuple(args.calibration_backbones), tuple(args.calibration_precisions),
        args.calibration_repeats, args.budget_stage, args.cell_wall_budget,
        source_commit, args.v5_lambda_r, args.v5_study_id, args.v5_prereg_sha256,
        args.v5_batch_estimate_seconds,
    ))]
    cells.extend(
        code_cell(runner_cell_source(index + 1, args.runner_cells))
        for index in range(args.runner_cells)
    )
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path = args.output_dir / f"{args.mode}.ipynb"
    notebook_path.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
    title = (
        f"PINN PDE Attention {args.mode} {args.hardware} {args.point_scheme}"
        if args.mode == "m2a"
        else f"PINN PDE Attention {args.mode}"
    )
    metadata = {
        "id": args.kernel_id,
        "title": title,
        "code_file": notebook_path.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": list(dict.fromkeys(
            source for source in (args.code_dataset, args.results_dataset, args.plan_dataset)
            if source is not None
        )),
        "competition_sources": [],
        "kernel_sources": [],
    }
    (args.output_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(notebook_path)


if __name__ == "__main__":
    main()
