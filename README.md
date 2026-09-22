# Controlled Comparison of PDE-Structured Graph Backbones in Physics-Informed Neural Networks

A physics-informed neural network (PINN) framework for the benchmark equations of
the PINN failure-mode literature (convection, reaction, Allen–Cahn), built to compare
three backbones — a vanilla MLP and two graph neural ODE architectures with a
diffusion (GRAND) or diffusion-reaction (GREAD) mixing layer — under a single trainer,
with numerical precision and regularisation as controlled factors. The framework
consists of one training loop for every condition, reference solutions with a
collocation-resolution check, a fully resumable run database, and a preregistered,
hash-locked study protocol that was frozen before any GPU run.

The first study slice was stopped by its own decision gate: the collocation grid had
been reduced below the Nyquist limit of the target solution, so every model satisfied
the PDE residual while being wrong. This document describes the framework, what the
pilot measured, the diagnostics that separated two distinct failure modes of PINNs on
these benchmarks, and the fully specified follow-up study that is prepared but not yet
run.

**Summary of findings.**
On the convection equation at β = 50, a grid of 1.75 samples per period admits
residual-satisfying functions that are worse than the zero predictor; the failure is
aliasing, not optimisation, and was caught by the gate before further budget was
spent. On the reaction equation, the framework solves ρ = 5 — reported as a failure
case by Krishnapriyan et al. — to relative L2 0.058 with an adequately resolved grid,
and locates the collapse edge between ρ = 5 and ρ = 6, where a second failure mode
appears that more collocation points do not cure. FP64 was faster than FP32 on the
T4 for every backbone, because the workload is launch-bound and FP64 needs fewer
line-search evaluations. 1.48 of 27 available GPU-hours were spent; the follow-up
study at the collapse edge (60 runs) is specified, and its launcher refuses to start
until its preregistration is frozen.

Working documents (preregistration, deviations, budget, findings, handover) are in
German; this README is the English report.

---

## 1. Framework

### 1.1 Problems and reference solutions (`src/pdes/`)

| equation | form | reference | role |
|---|---|---|---|
| convection | u_t + β u_x = 0, periodic, u(x,0) = sin x | analytic | primary benchmark of the failure-mode literature, β = 50 |
| reaction | u_t = ρ u (1 − u), Gaussian initial profile | analytic, residual 1e-31 | stiffness sweep and follow-up study |
| Allen–Cahn | u_t = 1e-4 u_xx + 5 u − 5 u³ | in-repo pseudo-spectral ETDRK4 solver, float64 | gated second stage |

Every problem exposes a resolution check that compares the collocation grid against
the wavenumber content of its reference solution, a precondition introduced after the
pilot (§2.2). The Allen–Cahn reference is generated in the repository
(`data/make_allen_cahn_reference.py`) and agrees with the Chebfun reference of the
original PINNs release to a relative L2 of 1.7e-5, so the framework has no external
data dependency.

### 1.2 Backbones (`src/models/`)

All three backbones map (x, t) to u and are sized to ≈ 50 k parameters (±10 %).

**MLP.** Tanh multilayer perceptron, 4 × 128 in the reference configuration, the
standard PINN baseline.

**Graph neural ODEs (GRAND, GREAD).** Collocation points form the nodes of a fixed
k-nearest-neighbour graph (k = 8) over the normalised (x, t) domain. A linear encoder
lifts each point to a hidden state, which is then advanced through a stack of
explicit Euler steps of a learned PDE on the graph. Each step computes attention
scores between a node and its neighbours (query/key dot products, softmax over the
neighbourhood, optionally weighted by distance), aggregates the neighbour values, and
takes the difference to the node's own value — a graph Laplacian with learned,
attention-based edge weights — scaled by a learned diffusivity. That is GRAND: pure
diffusion on the graph. GREAD adds a learned reaction term, an Allen–Cahn-type
nonlinearity tanh(h)(1 − tanh(h)²) with a learned rate, so that the mixing layer can
sharpen as well as smooth. A linear decoder reads out u. The graph is fixed so that a
gain from PDE structure cannot be confounded with a gain from global mixing.

The fixed-support graph context is shared between training and inference: the
neighbourhood of a query point is found within the support set that was used to
build the graph, which keeps the automatic derivatives that the PDE residual needs
consistent between the two (validated in `docs/local-validation-v5.md`).

### 1.3 Trainer (`src/train.py`, `src/persistence.py`)

One trainer for every condition; separate training scripts per architecture are the
most common source of unfair comparisons in exactly this literature. The optimiser is
L-BFGS with strong Wolfe line search, `max_eval = 25`, `tolerance_change =
tolerance_grad = 0` and a fixed iteration count. This choice is mandatory rather than
conventional: the precision confound identified in "FP64 is All You Need" (arXiv
2505.10949, §5.2) is L-BFGS-specific — its default `tolerance_change = 1e-7` sits
below FP32 machine epsilon — and disabling the tolerances is the only way to compare
FP32 and FP64 at equal training length. The optional regulariser is double
backpropagation (Andersen & Matsubara, arXiv 2605.30910), a penalty of weight λ_r on
the gradient of the residual, re-implemented in PyTorch from a JAX reference that does
not state its λ_r.

Every run is written to SQLite the moment it terminates; divergence and NaN are stored
as censored results (relative L2 = 10), never excluded. Runs resume from the database,
are ordered by ascending cost so that a quota cut-off leaves an analysable core, and
record source commit, Torch, CUDA and GPU metadata. The local 6 GB GPU is forbidden
for any run of a study matrix, because that would confound precision with hardware.

### 1.4 Study protocol

The study design (`PREREGISTRATION.md`, hash-locked in `PREREGISTRATION.lock.json`,
tag `prereg-v3`) is frozen before any GPU run and carries a pre-defined decision gate.
The null hypothesis — every architecture gain disappears once precision and
regularisation are controlled — is taken as the primary outcome. The directed
alternative is a double dissociation: GRAND, being pure diffusion, should not help on
an equation without a diffusion term, while GREAD's explicit reaction term should; if
both graph backbones win equally, that argues for an unspecific capacity effect.
Baseline definitions were extracted from the full texts of seven papers, with each
number tagged as sourced or estimated and seven cross-paper contradictions listed
rather than silently resolved (`docs/baselines.md`).

| factor | levels |
|---|---|
| equation | convection β = 50; Allen–Cahn, gated on the first stage |
| backbone | MLP, GRAND, GREAD |
| precision | FP32, FP64 |
| regularisation | none, double backprop |
| seeds | 5 |

## 2. Pilot results

### 2.1 Cost and precision on the T4

A throw-away timing probe (300 iterations, 4 cells, 3 repetitions, on a Kaggle P100
and on 2× T4) fixed the hardware and the iteration count before any study data
existed; the P100 was slower than the T4 for this workload. On the study problem
size, the workload is kernel-launch-bound rather than compute-bound, so the FP64
penalty of the Turing architecture never appears: FP64 was faster than FP32 for every
backbone (MLP 253 s vs. 326 s, GRAND 530 s vs. 596 s, GREAD 590 s vs. 970 s), driven
by fewer line-search evaluations (16 720 vs. 27 569 for the MLP). Graph backbones
cost two to three times the MLP at matched parameter count. The calibration
underestimated real run time by a factor of 1.26 because it timed only the optimiser
loop.

### 2.2 The first slice: residual satisfied, solution wrong

Six seed-0 cells (3 backbones × 2 precisions, no regularisation, 6 000 L-BFGS
iterations, convection β = 50, 396 collocation points):

| cell | rel. L2 | total loss | wall-clock |
|---|---|---|---|
| MLP / FP32 | 1.772 | 1e-6 | 326 s |
| MLP / FP64 | 1.576 | 1e-6 | 254 s |
| GRAND / FP32 | 1.367 | 3.6e-4 | 596 s |
| GRAND / FP64 | 1.206 | 3.7e-4 | 530 s |
| GREAD / FP32 | 1.227 | 4.3e-4 | 970 s |
| GREAD / FP64 | 2.314 | 3.0e-4 | 590 s |

Every model is worse than the trivial zero predictor (rel. L2 = 1.0) while its
residual loss is tiny. That combination is the signature of aliasing, not of
optimisation failure: the target sin(x − 50 t) has 7.96 periods in t ∈ [0, 1], and the
budget-reduced 14 × 14 grid samples it at 1.75 points per period, below the Nyquist
limit, so infinitely many functions satisfy the residual on that grid.

The pre-defined gate (`docs/decision-gate-m2b.md`) failed on its resolution criterion
("at least one cell < 0.10"; best was 1.206) and, by rule, no further budget was
spent. The cause is a process failure: the outcome-blind budget hierarchy reduced the
collocation points from 400 to 196 without any check against the PDE parameter,
because neither the preregistration nor the execution contract required one. The gate
caught it before the remaining 24 runs of the stage were spent.

### 2.3 Feasibility diagnostics (local CPU, not study data)

With a resolved grid (1 024 and 4 096 points), no baseline reached the preregistered
rel. L2 < 0.10 on convection β = 50 within the short budgets tested; the best resolved
anchor was 0.664. The diagnostic suite (`analysis/diagnose_feasibility.py`) separates
the causes:

- **Capacity is not the bottleneck.** The same 4 × 128 tanh MLP fits the analytic
  β = 50 solution to rel. L2 0.0113 under direct supervision.
- **The physics loss is badly conditioned at β = 50.** At identical initialisation the
  gradient norms of the domain and initial-condition terms are 63.9 vs. 2.15 with
  cosine −0.993: the two terms demand nearly opposite updates.
- **Cold start reproduces the literature.** β = 1 and β = 10 solve in 600 iterations
  (0.0088, 0.0027); β = 50 stays at 0.971 with a held-out domain loss 31 844× the
  training domain loss.
- **Curriculum helps but does not suffice** at this budget (0.634; the transition
  fails between β = 25 and 30).
- **Double backprop shows a λ_r trade-off, no success:** small λ_r overfits (factor
  2 211), large λ_r freezes the optimiser at the wrong solution.

### 2.4 Two failure modes on the reaction equation

The reaction equation has an analytic solution that satisfies the residual to 1e-31,
so the target cannot be a source of error. Sweeping the stiffness ρ with the MLP
baseline:

| ρ | domain points | best rel. L2 | |
|---|---|---|---|
| 1 | 1 600 | 0.0033 | solved |
| 3 | 1 600 | 0.0087 | solved |
| 5 | 400 | 0.981 | failed |
| **5** | **1 600** | **0.0575** | **solved** |
| 5.5 | 1 600 | 0.981 / 0.117 / 0.990 (seeds 0/1/2) | bimodal |
| 6 | 1 600 | 0.989 | failed |
| 7 | 1 600 | 0.993 | failed |
| 7 | 3 600 | 0.9998 | failed |
| 10 | 1 600 | 0.996 | failed |

Krishnapriyan et al. report ρ = 5 as a failure; with five collocation points across
the Gaussian initial profile instead of 2.5 it is not. The ρ = 7 control at 3 600
points separates two mechanisms:

- **Aliasing**: loss small (1e-5 … 1e-6), solution wrong, cured by more points.
- **Optimisation collapse**: loss stays large (≈ 2e-1), the optimiser freezes after a
  few hundred iterations, not cured by more points (3 600 points made ρ = 7 worse).

The failure-mode literature is therefore not "just under-resolution"; it is, for part
of the cases. The transition between solved and collapsed is sharp (more than an order
of magnitude between ρ = 5 and ρ = 6). For such a bimodal outcome the median of
log10(rel. L2) measures little; the informative statistic is the success rate over
seeds, and the follow-up study reports both as co-primary.

### 2.5 Corrections to earlier statements

- The claim that L-BFGS was standing still in this project was wrong: a missing
  `max_eval` in a diagnostic script, not in the trainer.
- The first execution of `analysis/gate_check.py` read a censoring flag instead of the
  metric and reported 0 for both gate quantities; the values above are from the
  corrected script (`results/gate_m2b.json`).
- The four-samples-per-period rule is a conservative project rule, not a solvability
  law established by the reference paper, which itself uses a 20 × 20 grid.

### 2.6 Budget

1.48 of 27 available GPU-hours were used: 0.54 h calibration, 0.94 h for the pilot
slice. No budget was spent on runs that would have built on the faulty grid. A costed
re-plan (`BUDGET.md`) shows the original full design would need ≈ 62 GPU-hours,
about twelve times the granted budget; the cheapest scientifically meaningful
extension (double backprop on convection only) needs ≈ 10 h.

## 3. Follow-up study (specified, not run)

The follow-up protocol (`PREREGISTRATION-V5.md`, status *draft*) moves the study to
the reaction equation at ρ = 5.25, the measured collapse edge, where the MLP baseline
succeeds in 2 of 5 seeds (final rel. L2 0.996, 0.999, 0.109, 0.083, 0.070). The
working point was chosen from MLP runs only, so it cannot favour a graph backbone.
Same 3 × 2 × 2 × 5 matrix (60 runs), 1 600 domain points, 2 000 iterations,
co-primary metrics success rate (Wilson 95 % CI) and median log10(rel. L2) (bootstrap
CI), threshold 0.10 fixed before data. The prediction is a contrast between the two
graph architectures: GRAND no gain, GREAD a gain. λ_r for double backprop is selected
outcome-blind from {1e-5, 1e-4, 1e-3, 1e-2} on the MLP baseline first.

The launcher (`kaggle/v5_runner.py`, one worker process per T4) refuses to build the
matrix while the preregistration carries the status *draft* or while its SHA-256 does
not match the frozen value. Local CPU checks validate the revised fixed-support graph
derivatives and inference consistency; short graph training runs hit their diagnostic
time limit, so the earlier GPU cost estimate does not establish that the 60-run matrix
fits the budget, and a profile on 2× T4 is required before the matrix starts
(`docs/local-validation-v5.md`). The next external steps are that profile and the
four λ_r selection runs.

## 4. Limitations

- The pilot slice is a single-seed technical pilot and permits no inference about
  architectures; it is reported because the failure and its cause are the finding.
- The double-backprop reference uses JAX/Optax and does not state its λ_r; this is a
  PyTorch re-implementation, not an exact reproduction.
- All resolution and feasibility diagnostics ran locally on CPU and are marked
  `NOT_STUDY_DATA`; they inform the design but are not study results.
- The GPU cost of the revised graph backbones is unmeasured.
- Early runs stored `git_commit = unknown` and did not persist the hardware seen;
  both are fixed for the follow-up.

## 5. Reproduction

```bash
pip install -e .                             # numpy, scipy, torch >= 2.6; Python >= 3.11
python -m unittest discover -s tests -v      # 52 tests, CPU only
python -m src.train --config configs/smoke.json   # 200 L-BFGS iterations, 100 points, writes a real SQLite row
```

The smoke test (including the hardest path, GREAD + FP64 + double backprop, with
verified resume) must be green before any Kaggle run. Kaggle notebooks are generated,
not hand-written: `kaggle/build_notebook.py` bundles the source as a private dataset,
and the runners resume from `results/results.sqlite` and publish after every
terminal run. The local diagnostics are reproduced with
`analysis/diagnose_feasibility.py` and `bench/feasibility_anchor.py`.

## 6. Repository layout

| Path | Content |
|---|---|
| `src/pdes/` | convection, Allen–Cahn, reaction with reference solutions; resolution check |
| `src/models/` | MLP; GRAND and GREAD graph neural ODE backbones |
| `src/train.py`, `src/persistence.py` | the single trainer; SQLite attempts and resume |
| `bench/` | calibration, budget planning, λ_r selection, feasibility anchors, follow-up matrix |
| `kaggle/` | notebook generator, resumable runners, fail-closed publication |
| `analysis/` | gate check, feasibility diagnostics |
| `results/` | `results.sqlite` (one row per run, including failures) and JSON artefacts |
| `tests/` | unit tests for PDEs, models, trainer, persistence, resolution check, follow-up matrix |
| `PREREGISTRATION.md`, `PREREGISTRATION.lock.json` | pilot study protocol, frozen (hash-locked) |
| `PREREGISTRATION-V4.md`, `PREREGISTRATION-V5.md` | superseded draft (never run); current follow-up draft |
| `DEVIATIONS.md` | every deviation from the literature and from the frozen plan, with reasons |
| `BUDGET.md`, `FINDINGS.md`, `HANDOFF.md` | quota ledger, findings (null results first), handover state |
| `docs/` | baseline definitions from seven full texts, gate rules, execution contract, costed options, local validation |

## 7. Data, licences and provenance

Code and documentation: MIT. `data/allen_cahn.mat` is generated in-repository by
`data/make_allen_cahn_reference.py` (Fourier pseudo-spectral, ETDRK4, float64) and
has no external data dependency; it agrees with the Chebfun reference of the original
PINNs release (Raissi et al., MIT) to a relative L2 error of 1.7e-5. Until 2026-09-18
the file was a copy from `miniHuiHui/PINN_FP64` (Xu et al. 2025), a repository
without a licence, and an earlier version of this section misattributed it to Raissi
et al.; see DEVIATIONS.md D-10 and data/README.md. Sources read in full for the
baseline definitions: Krishnapriyan et al. 2021 (arXiv 2109.01050), PINNsformer
(2307.11833), GRAND (2106.10934), GREAD (2211.14208), "FP64 is All You Need"
(2505.10949), Andersen & Matsubara 2026 (2605.30910), Expert's Guide to PINNs
(2308.08468).

The project was run by one person with AI coding agents (Claude, OpenAI Codex) under
a written brief: preregistration frozen before any GPU run, outcome-blind budget
cuts, no cherry-picking, and a stop-and-report at every gate. The handover document
records the process history, including the two occasions on which the project lead
reversed their own earlier decisions.

## 8. Citation

```bibtex
@software{pinn_graph_backbones_2026,
  title  = {Controlled Comparison of PDE-Structured Graph Backbones in Physics-Informed Neural Networks},
  author = {Say43},
  year   = {2026},
  url    = {https://github.com/Say43/PINN}
}
```
