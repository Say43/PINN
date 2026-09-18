"""Build an exact, minimal V5 code Dataset payload from a clean Git commit."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


RUNTIME_FILES = (
    "pyproject.toml",
    "PREREGISTRATION-V5.md",
    "configs/reaction_v5.json",
    "bench/__init__.py",
    "bench/v5.py",
    "kaggle/__init__.py",
    "kaggle/publish.py",
    "kaggle/v5_runner.py",
)


def git(*args: str) -> bytes:
    return subprocess.check_output(("git", *args))


def build(output: Path) -> dict:
    if git("status", "--porcelain").strip():
        raise RuntimeError("V5 payload requires a clean source checkout")
    commit = git("rev-parse", "HEAD").decode("ascii").strip()
    tracked = git("ls-tree", "-r", "--name-only", "HEAD").decode("utf-8").splitlines()
    files = sorted({*RUNTIME_FILES, *(path for path in tracked if path.startswith("src/") and path.endswith(".py"))})
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"payload directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"source_commit": commit, "scope": "PINN V5 runtime source and draft protocol only", "files": []}
    for path in files:
        if path not in tracked:
            raise FileNotFoundError(f"required tracked runtime file absent: {path}")
        data = git("show", f"{commit}:{path}")
        destination = output / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        manifest["files"].append({"path": path, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    (output / "pinn_payload_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(args.output)
    print(json.dumps({"source_commit": manifest["source_commit"], "files": len(manifest["files"]),
                      "total_bytes": sum(row["bytes"] for row in manifest["files"])}, sort_keys=True))


if __name__ == "__main__":
    main()
