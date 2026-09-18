"""Bounded CPU validity gate; diagnostic data, never study results.

Run before spending time on training: derivatives must agree with independent
finite differences, and the represented function must be continuous when query
neighbours change. A failed validity gate makes a seed-performance comparison
uninterpretable, so it is explicitly deferred.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import torch

from src.config import ExperimentConfig
from src.models import create_model
from src.pdes import create_pde
from src.train import set_determinism


def probe(base, backbone, seed):
    set_determinism(seed)
    pde = create_pde(base.pde)
    model = create_model(replace(base.model, backbone=backbone), pde.bounds).double()
    support = pde.collocation().coords.double()
    if backbone != "mlp":
        model.set_support(support)
    coords = torch.tensor([[.713, .21731], [2.371, .63371], [5.237, .83413]],
                          dtype=torch.float64, requires_grad=True)
    values = model(coords)
    ad = torch.autograd.grad(values.sum(), coords)[0][:, 1]
    eps = 1e-6
    plus, minus = coords.detach().clone(), coords.detach().clone()
    plus[:, 1] += eps; minus[:, 1] -= eps
    with torch.no_grad():
        fd = (model(plus) - model(minus)) / (2 * eps)
        individual = torch.cat([model(coords[i:i+1]) for i in range(len(coords))])
    discrepancy = float(torch.linalg.vector_norm(ad - fd) / torch.linalg.vector_norm(fd))
    batching = float((individual - values.detach()).abs().max())
    row = {"backbone": backbone, "seed": seed,
           "temporal_derivative_relative_error": discrepancy,
           "query_batch_max_difference": batching}
    if backbone == "mlp":
        row["neighbour_switch"] = None
        return row

    # Find and bisect an actual neighbourhood switch in time, with fixed support.
    line = torch.stack((torch.full((401,), 2.371, dtype=torch.float64),
                        torch.linspace(.2, .8, 401, dtype=torch.float64)), dim=1)
    topology = model.query_topology(line)
    neighbors = [row[mask] for row, mask in zip(topology.neighbors, topology.mask)]
    changes = [i for i in range(len(neighbors)-1) if not torch.equal(neighbors[i], neighbors[i+1])]
    largest = None
    for index in changes[:8]:
        index = int(index)
        left, right = float(line[index, 1]), float(line[index+1, 1])
        initial = neighbors[index]
        for _ in range(35):
            mid = (left + right) / 2
            current_topology = model.query_topology(torch.tensor([[2.371, mid]], dtype=torch.float64))
            current = current_topology.neighbors[0][current_topology.mask[0]]
            if torch.equal(current, initial):
                left = mid
            else:
                right = mid
        boundary = (left + right) / 2
        jumps = []
        for delta in (1e-5, 1e-7, 1e-9):
            queries = torch.tensor([[2.371, boundary-delta], [2.371, boundary+delta]], dtype=torch.float64)
            with torch.no_grad():
                output = model(queries)
            jumps.append(float((output[1]-output[0]).abs()))
        candidate = {"x": 2.371, "t": boundary, "deltas": [1e-5, 1e-7, 1e-9], "absolute_jumps": jumps}
        if largest is None or jumps[-1] > largest["absolute_jumps"][-1]:
            largest = candidate
    row["neighbour_switch"] = largest
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/local_validation.json"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    base = ExperimentConfig.from_json("configs/reaction_v5.json")
    started = time.perf_counter()
    rows = []
    for seed in (101, 102, 103):
        for backbone in ("mlp", "grand", "gread"):
            row = probe(base, backbone, seed)
            rows.append(row)
            print(json.dumps(row), flush=True)
    derivatives = all(r["temporal_derivative_relative_error"] < 1e-5 for r in rows)
    batches = all(r["query_batch_max_difference"] < 1e-10 for r in rows)
    continuity = all(r["neighbour_switch"] is None or r["neighbour_switch"]["absolute_jumps"][-1] < 1e-6 for r in rows)
    payload = {
        "note": "LOCAL_CPU_NOT_STUDY_DATA; untrained validity probes, no performance ranking",
        "seeds": [101, 102, 103], "model": base.model.__dict__,
        "gates": {"derivatives": derivatives, "query_batch_invariance": batches, "continuity": continuity},
        "ready_for_training_comparison": derivatives and batches and continuity,
        "training_comparison": "not run; validity gate first",
        "sample_size": "deferred until a valid primary comparison and effect size are specified",
        "wall_seconds": time.perf_counter() - started, "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"gates": payload["gates"], "seconds": payload["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
