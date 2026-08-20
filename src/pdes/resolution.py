"""Harte Aufloesungspruefung des Kollokationsgitters gegen den PDE-Parameter.

Hintergrund: In M2b lieferten alle sechs Zellen Residual-Losses von 1e-6 bis 4e-4
bei relativen L2-Fehlern von 1.21 bis 2.31 — schlechter als der Nullpraediktor.
Ursache war ein 14x14-Gitter fuer eine Loesung mit 7.96 Perioden in der Zeit, also
1.75 Abtastungen pro Periode. Unterhalb von zwei Abtastungen pro Periode ist die
Zielfunktion auf dem Gitter nicht rekonstruierbar; das Residuum laesst sich dann
von beliebig falschen Funktionen erfuellen (Aliasing).

Siehe FINDINGS.md Abschnitt 2 und DEVIATIONS.md D-7.

Die Pruefung bricht ab, statt zu warnen. Eine Warnung haette den Fehlschlag nicht
verhindert — die Laeufe waren technisch fehlerfrei.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Nyquist verlangt > 2. Vier ist der bewusst konservative Sicherheitsabstand:
# AM26 arbeitet mit 2.51 und braucht dafuer 10^6 L-BFGS-Iterationen.
MIN_SAMPLES_PER_PERIOD = 4.0


@dataclass(frozen=True)
class ResolutionReport:
    periods_in_time: float
    time_samples: int
    time_step: float
    samples_per_period: float
    required: float

    @property
    def adequate(self) -> bool:
        return self.samples_per_period >= self.required

    def describe(self) -> str:
        verdict = "ausreichend" if self.adequate else "UNZUREICHEND"
        return (
            f"{self.time_samples} Zeitstuetzstellen, dt={self.time_step:.5f}, "
            f"{self.periods_in_time:.2f} Perioden in t, "
            f"{self.samples_per_period:.2f} Abtastungen pro Periode "
            f"(gefordert {self.required:.1f}) -> {verdict}"
        )


def grid_shape(count: int) -> tuple[int, int]:
    """Spiegelt die Gitterkonstruktion aus convection._rectangular_grid."""
    nx = max(1, int(math.sqrt(count)))
    nt = math.ceil(count / nx)
    return nx, nt


def convection_resolution(
    domain_points: int,
    *,
    beta: float,
    t_min: float = 0.0,
    t_max: float = 1.0,
    required: float = MIN_SAMPLES_PER_PERIOD,
) -> ResolutionReport:
    """Abtastung des Zeitgitters gegen die Periode von sin(x - beta*t)."""
    _, nt = grid_shape(domain_points)
    span = t_max - t_min
    if beta == 0.0:
        period = math.inf
        periods = 0.0
    else:
        period = 2.0 * math.pi / abs(beta)
        periods = span / period
    if nt > 1:
        step = (t_max - (t_min + span / (nt + 1))) / (nt - 1)
    else:
        step = span
    samples = math.inf if not math.isfinite(period) else period / step
    return ResolutionReport(
        periods_in_time=periods,
        time_samples=nt,
        time_step=step,
        samples_per_period=samples,
        required=required,
    )


def minimum_domain_points(*, beta: float, t_span: float = 1.0,
                          required: float = MIN_SAMPLES_PER_PERIOD) -> int:
    """Kleinste quadratische Punktzahl, die die Forderung erfuellt."""
    if beta == 0.0:
        return 1
    candidate = 4
    while True:
        report = convection_resolution(candidate * candidate, beta=beta,
                                       t_max=t_span, required=required)
        if report.adequate:
            return candidate * candidate
        candidate += 1


def require_convection_resolution(domain_points: int, *, beta: float) -> ResolutionReport:
    """Bricht ab, wenn das Gitter die Loesung nicht aufloesen kann."""
    report = convection_resolution(domain_points, beta=beta)
    if not report.adequate:
        needed = minimum_domain_points(beta=beta)
        raise ValueError(
            "Kollokationsgitter loest die Loesung nicht auf: "
            f"{report.describe()}. "
            f"Bei beta={beta:g} sind mindestens {needed} Domaenenpunkte noetig "
            f"(aktuell {domain_points}). Siehe FINDINGS.md Abschnitt 2."
        )
    return report
