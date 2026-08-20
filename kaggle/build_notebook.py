from __future__ import annotations

import argparse
import json
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
) -> str:
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
    else:
        if plan_dataset is None:
            raise ValueError("non-M2a notebooks require --plan-dataset")
        command = [
            "python", "-m", "kaggle.runner", "--stage", mode, "--config", config,
            "--plan", "__M2_PLAN__",
        ]
    return f'''import os, shutil, subprocess, sys, zipfile
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
{bootstrap}
os.chdir(root)
sys.path.insert(0, str(root))
os.environ["PINN_RESULTS_DATASET"] = "{results_dataset}"
command = {command!r}
if "__M2_PLAN__" in command:
    plan_paths = list(inputs.rglob("m2_plan.json"))
    if len(plan_paths) != 1:
        raise RuntimeError(f"Expected one M2 plan, found {{len(plan_paths)}}")
    command = [str(plan_paths[0]) if value == "__M2_PLAN__" else value for value in command]
subprocess.run(command, check=True)
'''


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a thin Kaggle launcher notebook")
    parser.add_argument("--mode", choices=("m2a", "m2b", "stage_a", "stage_b"), required=True)
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
    parser.add_argument("--kernel-id", required=True, help="owner/kernel-slug")
    parser.add_argument("--output-dir", type=Path, default=Path("kaggle/generated"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in launcher_source(
                    args.mode, args.code_dataset, args.results_dataset,
                    args.plan_dataset, args.hardware, args.point_scheme,
                    tuple(args.calibration_backbones), tuple(args.calibration_precisions),
                    args.calibration_repeats, args.budget_stage,
                ).splitlines()],
            }
        ],
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
