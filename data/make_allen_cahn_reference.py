"""Generate the Allen-Cahn reference solution used for Stage-B evaluation.

Solves  u_t = d * u_xx - r * u^3 + r * u  on x in [-1, 1) (periodic), t in [0, 1],
with u(x, 0) = x^2 cos(pi x), d = 1e-4, r = 5 -- the same problem that
`configs/stage_b.json` defines and `src/pdes/allen_cahn.py` enforces.

Method: Fourier pseudo-spectral in x (N = 512 modes, float64) with the ETDRK4
time stepper of Kassam & Trefethen (2005), dt = 1e-4, 3/2-rule dealiasing on
the cubic term. Output matches the layout the loader expects:
`t` (1 x 201), `x` (1 x 513), `usol` (201 x 513, time x space). The 513th
column is the periodic image x = +1 so that bilinear interpolation on the closed
evaluation interval [-1, 1] never extrapolates.

    python data/make_allen_cahn_reference.py            # writes data/allen_cahn.mat
    python data/make_allen_cahn_reference.py --check    # also prints SHA-256 and
                                                        # a self-convergence estimate

This file replaces a reference solution that had been copied from a third-party
repository without a licence (see DEVIATIONS.md, D-10). It has no external
data dependency.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
from scipy.io import savemat

DIFFUSION = 1e-4
REACTION = 5.0
N_X = 512
N_T_OUT = 201
T_END = 1.0


def solve(n_x: int = N_X, dt: float = 1e-4, n_out: int = N_T_OUT) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(-1.0, 1.0, n_x, endpoint=False)
    length = 2.0
    k = 2.0 * np.pi * np.fft.fftfreq(n_x, d=length / n_x)
    linear = -DIFFUSION * k**2 + REACTION  # linear part in Fourier space

    # ETDRK4 coefficients via contour integral (Kassam & Trefethen 2005).
    e = np.exp(dt * linear)
    e2 = np.exp(dt * linear / 2.0)
    m = 32
    roots = np.exp(1j * np.pi * (np.arange(1, m + 1) - 0.5) / m)
    lr = dt * linear[:, None] + roots[None, :]
    q = dt * np.real(np.mean((np.exp(lr / 2.0) - 1.0) / lr, axis=1))
    f1 = dt * np.real(np.mean((-4.0 - lr + np.exp(lr) * (4.0 - 3.0 * lr + lr**2)) / lr**3, axis=1))
    f2 = dt * np.real(np.mean((2.0 + lr + np.exp(lr) * (-2.0 + lr)) / lr**3, axis=1))
    f3 = dt * np.real(np.mean((-4.0 - 3.0 * lr - lr**2 + np.exp(lr) * (4.0 - lr)) / lr**3, axis=1))

    # 3/2-rule dealiasing mask for the cubic nonlinearity.
    kmax = np.abs(k).max()
    mask = np.abs(k) <= (2.0 / 3.0) * kmax

    def nonlinear(v_hat: np.ndarray) -> np.ndarray:
        u = np.real(np.fft.ifft(v_hat))
        out = -REACTION * np.fft.fft(u**3)
        out[~mask] = 0.0
        return out

    u0 = x**2 * np.cos(np.pi * x)
    v = np.fft.fft(u0)
    v[~mask] = 0.0

    t_out = np.linspace(0.0, T_END, n_out)
    n_steps = int(round(T_END / dt))
    assert abs(n_steps * dt - T_END) < 1e-12, "dt must divide T_END"
    steps_per_out = n_steps // (n_out - 1)
    assert steps_per_out * (n_out - 1) == n_steps, "dt must resolve the output grid"

    usol = np.empty((n_out, n_x))
    usol[0] = u0
    for i in range(1, n_out):
        for _ in range(steps_per_out):
            nv = nonlinear(v)
            a = e2 * v + q * nv
            na = nonlinear(a)
            b = e2 * v + q * na
            nb = nonlinear(b)
            c = e2 * a + q * (2.0 * nb - nv)
            nc = nonlinear(c)
            v = e * v + nv * f1 + 2.0 * (na + nb) * f2 + nc * f3
        usol[i] = np.real(np.fft.ifft(v))
    return t_out, x, usol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=Path(__file__).with_name("allen_cahn.mat"))
    parser.add_argument("--dt", type=float, default=1e-4)
    parser.add_argument("--check", action="store_true", help="print SHA-256 and a self-convergence estimate")
    args = parser.parse_args()

    t, x, usol = solve(dt=args.dt)
    # Close the periodic domain: append x = +1 with u(+1, t) = u(-1, t).
    x = np.append(x, 1.0)
    usol = np.concatenate([usol, usol[:, :1]], axis=1)
    savemat(args.out, {"t": t[None, :], "x": x[None, :], "usol": usol}, do_compression=False)
    print(f"wrote {args.out}  t {t.shape}  x {x.shape}  usol {usol.shape}")
    if args.check:
        digest = hashlib.sha256(Path(args.out).read_bytes()).hexdigest()
        print(f"SHA-256 {digest}")
        _, _, coarse = solve(dt=args.dt * 2.0)
        print(f"self-convergence |u(dt) - u(2dt)|_max = {np.abs(usol[:, :-1] - coarse).max():.3e}")


if __name__ == "__main__":
    main()
