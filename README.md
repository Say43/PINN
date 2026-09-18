# PDE-structured attention in PINNs: a preregistered pilot that found the confound first

A controlled study of whether a PDE-structured mixing layer (GRAND / GREAD) as the
backbone of a physics-informed neural network beats a vanilla MLP *after* controlling
for numerical precision (FP32 vs. FP64) and regularisation (double backprop). The
contribution is the confound control, not the architecture race. The pilot was run
under a hard budget of 5 Kaggle GPU-hours with a frozen preregistration, a
pre-defined decision gate and a full ledger of every run, including the failed ones.

**Headline result: the study stopped itself.** The first data slice passed the
failure-mode check and failed the resolution check, and the reason turned out to be a
collocation grid below the Nyquist limit of the target solution. No architecture
claim survives from that slice. What does survive is a set of feasibility findings
about *when* PINNs on the standard failure-mode benchmarks fail and why, a corrected
methodology (a resolution check as a hard precondition), and a fully specified
follow-up study (V5) at the measured collapse edge of the reaction equation, prepared
but not yet frozen or run.

Working documents (preregistration, deviations, budget, findings, handoff) are in
German; this README is the English report.

## 1. Question and design

**Research question.** Does a PINN with a PDE-structured backbone escape the known
failure modes (Krishnapriyan et al., 2021) within a fixed, small L-BFGS iteration
budget better than a vanilla MLP PINN, and does any such advantage survive control for
precision and regularisation?

**H0 (taken seriously).** Every architecture gain disappears once precision and
regularisation are controlled. A null result is a full result and is reported first.

**H1 (directed double dissociation).** GRAND is pure diffusion and should *not* help
on a PDE without a diffusion term; GREAD adds an explicit reaction term and should. If
both graph backbones win equally, that argues for an unspecific capacity effect, not
for the mechanism.

**Why L-BFGS is mandatory.** The precision confound identified in "FP64 is All You
Need" (arXiv 2505.10949, §5.2) is L-BFGS-specific: `tolerance_change = 1e-7` sits
below FP32 machine epsilon (1.19e-7). With Adam the confound does not exist in this
form, so an Adam setup would compute past H0 rather than test it. Consequently all
runs use L-BFGS with `tolerance_change = tolerance_grad = 0` and a fixed iteration
count, so that precision cannot be confounded with training length.

**Matrix (V3, frozen as tag `prereg-v3`).**

| Factor | Levels |
|---|---|
| PDE | Convection β = 50 (Stage A), Allen–Cahn (Stage B, gated) |
| Backbone | MLP, GRAND, GREAD, each ≈ 50 k parameters (±10 %) |
| Precision | FP32, FP64 |
| Regularisation | none, double backprop (Andersen & Matsubara, arXiv 2605.30910) |
| Seeds | 5 |

One trainer for every condition (`src/train.py`); separate training scripts per
architecture are the most common source of unfair comparisons in exactly this
literature. Graph backbones use a fixed k-NN graph (k = 8) over collocation points so
that "PDE structure helps" is not confounded with "global mixing helps". Baseline
definitions were extracted from the full texts of seven papers, with every number
tagged as *sourced* or *estimated* and seven cross-paper contradictions listed rather
than silently resolved (`docs/baselines.md`).

**Execution contract.** Every run is persisted to SQLite as soon as it terminates;
divergence and NaN are recorded as censored results (rel. L2 = 10), never excluded.
Runs are ordered by ascending cost so that an early quota cutoff leaves an
analysable core. The local 6 GB GPU is forbidden for any arm of the matrix, because
that would confound precision with hardware, the very error the study looks for in
others.

## 2. Results

### 2.1 Calibration and cost: FP64 is not slower here

A throw-away timing probe (M2a: 300 iterations, 4 cells, 3 repetitions, on both a
Kaggle P100 and 2× T4) chose the hardware and the iteration count before any study
data existed; the P100 was slower than the T4 for this workload. The cost figures from
the subsequent M2b slice remain valid regardless of the aliasing problem described
below. On this problem size the workload is kernel-launch-bound, not compute-bound, so
Turing's FP64 penalty never appears: FP64 was *faster* than FP32 in every backbone
(MLP 253 s vs. 326 s, GRAND 530 s vs. 596 s, GREAD 590 s vs. 970 s), driven by fewer
line-search evaluations (16 720 vs. 27 569 for the MLP). The P100-vs-T4 hardware
debate was moot for this problem format. Graph backbones cost two to three times the
MLP at matched parameter count. The calibration underestimated real run time by a
factor of 1.26 because it timed only the optimiser loop.

### 2.2 The M2b slice: residual satisfied, solution wrong

Six seed-0 cells (3 backbones × 2 precisions, no regularisation, 6 000 L-BFGS
iterations, convection β = 50, 396 collocation points):

| Cell | rel. L2 | total loss | wall-clock |
|---|---|---|---|
| MLP / FP32 | 1.772 | 1e-6 | 326 s |
| MLP / FP64 | 1.576 | 1e-6 | 254 s |
| GRAND / FP32 | 1.367 | 3.6e-4 | 596 s |
| GRAND / FP64 | 1.206 | 3.7e-4 | 530 s |
| GREAD / FP32 | 1.227 | 4.3e-4 | 970 s |
| GREAD / FP64 | 2.314 | 3.0e-4 | 590 s |

Every model is worse than the trivial zero predictor (rel. L2 = 1.0) while its
residual loss is tiny. That combination is the signature of **aliasing**, not of
optimisation failure. The target is `sin(x − 50 t)`, which has 7.96 periods in
`t ∈ [0, 1]`; the budget-reduced 14 × 14 grid samples it at 1.75 points per period,
below Nyquist. On that grid infinitely many functions satisfy the residual.

The pre-defined gate (`docs/decision-gate-m2b.md`) failed on criterion PC-2 ("at
least one cell < 0.10"; best was 1.206) and, by rule, no additional budget was spent.
The mistake is documented as a process failure: the outcome-blind budget hierarchy
reduced collocation points from 400 to 196 without any check against the PDE
parameter, because neither the preregistration nor the execution contract required
one. The gate caught it before 24 further runs were spent.

### 2.3 Feasibility diagnostics (local CPU, explicitly `NOT_STUDY_DATA`)

Once the grid was resolved (1 024 and 4 096 points), no baseline reached the
preregistered rel. L2 < 0.10 on convection β = 50 within the short budgets tested;
the best resolved anchor was 0.664. The diagnostic suite
(`analysis/diagnose_feasibility.py`) separates the causes:

- **Capacity is not the bottleneck.** The same 4 × 128 tanh MLP fits the analytic
  β = 50 solution to rel. L2 0.0113 under direct supervision.
- **The physics loss is badly conditioned at β = 50.** At identical initialisation the
  gradient norms of the domain and initial-condition terms are 63.9 vs. 2.15 with
  cosine −0.993: the two terms demand nearly opposite updates.
- **Cold start reproduces the literature.** β = 1 and β = 10 solve in 600 iterations
  (0.0088, 0.0027); β = 50 stays at 0.971 with a held-out domain loss 31 844× the
  training domain loss.
- **Curriculum helps but does not suffice** at this budget (0.634; transition fails
  between β = 25 and 30).
- **Double backprop shows a λ_r trade-off, no success:** small λ_r overfits (factor
  2 211), large λ_r freezes the optimiser at the wrong solution.

### 2.4 Two distinguishable failure modes, and one benchmark equation solved

The reaction equation (Krishnapriyan Appendix A) has an analytic solution that
satisfies the residual to 1e-31, so the target itself cannot be a source of error.
Sweeping the stiffness ρ with the MLP baseline:

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
points separates the two failure modes cleanly:

- **Aliasing**: loss small (1e-5 … 1e-6), solution wrong, cured by more points.
- **Optimisation collapse**: loss stays large (≈ 2e-1), optimiser freezes after a few
  hundred iterations, *not* cured by more points (3 600 points made ρ = 7 worse).

This refutes the tempting generalisation that the failure-mode literature is "just
under-resolution". It is, for part of the cases.

The transition between solved and collapsed is sharp (more than an order of magnitude
between ρ = 5 and ρ = 6). For such a bimodal outcome the median of log10(rel. L2)
measures little; the informative statistic is the **success rate over seeds**. The V3
preregistration had moved from success rate to median for good reasons that do not
apply here; V5 reports both as co-primary and reports any disagreement between them
rather than resolving it.

### 2.5 Retractions

- The claim that L-BFGS was standing still in this project was wrong: a missing
  `max_eval` in a diagnostic script, not in the trainer (which has set `max_eval = 25`
  since M1).
- The first execution of `analysis/gate_check.py` read a censoring flag instead of the
  metric and reported 0 for both gate quantities; the values above are from the
  corrected script (`results/gate_m2b.json`).
- The four-samples-per-period rule is a conservative project rule, not a solvability
  law established by the reference paper, which itself uses a 20 × 20 grid.

### 2.6 Budget

About **1.48 of 27 available GPU-hours** were used: 0.54 h calibration, 0.94 h for the
M2b slice. No budget was spent on runs that would have built on the faulty setup. A
costed re-plan (`BUDGET.md`) shows the original full design would need ≈ 62 GPU-hours,
about twelve times the granted budget; the cheapest scientifically meaningful
extension (double backprop on convection only) needs ≈ 10 h.

## 3. The follow-up study, V5 (prepared, draft, not run)

V5 (`PREREGISTRATION-V5.md`) moves the study to **reaction at ρ = 5.25**, the measured
collapse edge, where the MLP baseline succeeds in 2 of 5 seeds (final rel. L2 0.996,
0.999, 0.109, 0.083, 0.070). The working point was chosen from MLP runs only, so it
cannot favour a graph backbone. Same 3 × 2 × 2 × 5 matrix (60 runs), 1 600 domain
points, 2 000 iterations, co-primary metrics success rate (Wilson 95 % CI) and median
log10(rel. L2) (bootstrap CI), threshold 0.10 fixed before data. The prediction is a
contrast *between* the two graph architectures: GRAND (diffusion only) no gain, GREAD
(explicit reaction term) a gain. λ_r for double backprop is selected outcome-blind
from {1e-5, 1e-4, 1e-3, 1e-2} on the MLP baseline first.

The launcher (`kaggle/v5_runner.py`, one worker process per T4) refuses to build the
matrix while the preregistration carries the status *draft* or while its SHA-256 does
not match the frozen value, and every run now records source commit, Torch, CUDA and
GPU metadata. The next external step is the four λ_r selection runs.

## 4. Limitations

- The V3 slice is a single-seed technical pilot and permits no inference about
  architectures; it is reported because the failure and its cause are the finding.
- The double-backprop reference (Andersen & Matsubara) uses JAX/Optax and does not
  state its λ_r; this is a PyTorch re-implementation, not an exact reproduction.
- All resolution and feasibility diagnostics ran locally on CPU and are marked
  `NOT_STUDY_DATA`; they inform the design but are not study results.
- Known gaps: early runs stored `git_commit = unknown`; the hardware actually seen
  was not persisted in `study_meta` for those runs. Both are fixed for V5.

## 5. Reproduction

```bash
pip install -e .                       # numpy, scipy, torch >= 2.6; Python >= 3.11
python -m unittest discover -s tests -v   # 43 tests, CPU only
python -m src.train --config configs/smoke.json   # 200 L-BFGS iterations, 100 points, writes a real SQLite row
```

The smoke test (including the hardest path, GREAD + FP64 + double backprop, 13.9 s
on CPU with verified resume) must be green before any Kaggle run. Kaggle notebooks are
generated, not hand-written: `kaggle/build_notebook.py` bundles the source as a private
dataset and `kaggle/runner.py` / `kaggle/v5_runner.py` resume from `results/results.sqlite`
and publish after every terminal run. The local diagnostics are reproduced with
`analysis/diagnose_feasibility.py` and `bench/feasibility_anchor.py`.

## 6. Repository layout

| Path | Content |
|---|---|
| `PREREGISTRATION.md`, `PREREGISTRATION.lock.json` | V3 study, frozen (hash-locked) |
| `PREREGISTRATION-V4.md`, `PREREGISTRATION-V5.md` | V4 (never run), V5 (current draft) |
| `DEVIATIONS.md` | every deviation from the literature and from the frozen plan, with reasons |
| `BUDGET.md`, `FINDINGS.md`, `HANDOFF.md` | quota ledger, findings (null results first), handover state |
| `docs/baselines.md` | baseline definitions from seven full texts, sourced vs. estimated |
| `docs/decision-gate-m2b.md`, `docs/execution-contract.md`, `docs/feasibility-options.md` | gate rules, Kaggle resume/persistence contract, costed continuation options |
| `src/pdes/` | convection, Allen–Cahn, reaction with reference solutions; resolution check |
| `src/models/` | MLP, GRAND, GREAD (graph ODE backbones) |
| `src/train.py`, `src/persistence.py` | the single trainer; SQLite attempts and resume |
| `bench/` | calibration, budget planning, λ_r selection, feasibility anchors, V5 matrix |
| `kaggle/` | notebook generator, resumable runners, fail-closed publication |
| `analysis/` | gate check, feasibility diagnostics |
| `results/` | `results.sqlite` (one row per run, including failures) and JSON artefacts |
| `tests/` | unit tests for PDEs, models, trainer, persistence, resolution check, V5 |

## 7. Data, licences and provenance

Code and documentation: MIT. `data/allen_cahn.mat` is the canonical Allen–Cahn
reference solution from the original PINNs release (Raissi et al., MIT). Sources
read in full for the baseline definitions: Krishnapriyan et al. 2021
(arXiv 2109.01050), PINNsformer (2307.11833), GRAND (2106.10934), GREAD (2211.14208),
"FP64 is All You Need" (2505.10949), Andersen & Matsubara 2026 (2605.30910),
Expert's Guide to PINNs (2308.08468).

The project was run by one person with AI coding agents (Claude, OpenAI Codex) under
a written brief: preregistration frozen before any GPU run, outcome-blind budget
cuts, no cherry-picking, and a stop-and-report at every gate. The handover document
records the process history, including the two occasions on which the project lead
reversed their own earlier decisions.

## 8. Citation

```bibtex
@software{pinn_pde_attention_2026,
  title  = {PDE-structured attention in PINNs: a preregistered pilot that found the confound first},
  author = {Say43},
  year   = {2026},
  url    = {https://github.com/Say43/PINN}
}
```
