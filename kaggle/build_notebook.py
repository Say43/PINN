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
) -> str:
    mount = code_dataset.split("/", 1)[1]
    config = "configs/stage_b.json" if mode == "stage_b" else "configs/stage_a.json"
    if mode == "m2a":
        command = [
            "python", "-m", "bench.calibrate", "--config", "configs/stage_a.json",
            "--output", f"/kaggle/working/m2a_{hardware}_full.jsonl",
            "--hardware-label", hardware,
            "--workers", "1", "--device", "cuda:0", "--point-scheme", "full",
            "--results-dataset", results_dataset,
        ]
    else:
        if plan_dataset is None:
            raise ValueError("non-M2a notebooks require --plan-dataset")
        plan_mount = plan_dataset.split("/", 1)[1]
        command = [
            "python", "-m", "kaggle.runner", "--stage", mode, "--config", config,
            "--plan", f"/kaggle/input/{plan_mount}/m2_plan.json",
        ]
    return f'''import os, subprocess, sys
from pathlib import Path

roots = list(Path("/kaggle/input/{mount}").rglob("pyproject.toml"))
if len(roots) != 1:
    raise RuntimeError(f"Expected one project root, found {{len(roots)}}")
root = roots[0].parent
os.chdir(root)
sys.path.insert(0, str(root))
os.environ["PINN_RESULTS_DATASET"] = "{results_dataset}"
subprocess.run({command!r}, check=True)
'''


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a thin Kaggle launcher notebook")
    parser.add_argument("--mode", choices=("m2a", "m2b", "stage_a", "stage_b"), required=True)
    parser.add_argument("--code-dataset", required=True, help="owner/slug")
    parser.add_argument("--results-dataset", required=True, help="owner/slug")
    parser.add_argument("--plan-dataset", help="owner/slug containing m2_plan.json")
    parser.add_argument("--hardware", choices=("p100", "2xt4"), default="p100")
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
                    args.plan_dataset, args.hardware
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
    metadata = {
        "id": args.kernel_id,
        "title": f"PINN PDE Attention {args.mode}",
        "code_file": notebook_path.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [
            source for source in (args.code_dataset, args.results_dataset, args.plan_dataset)
            if source is not None
        ],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (args.output_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(notebook_path)


if __name__ == "__main__":
    main()
