# PDE-strukturierte Attention in PINNs

Kontrollierte Studie: Bringt eine PDE-strukturierte Mixing-Schicht (GRAND / GREAD)
als PINN-Backbone einen Vorteil gegenüber einem Vanilla-MLP und gegenüber
PINNsformer — **und überlebt dieser Vorteil die Kontrolle für numerische Präzision
und Regularisierung?**

Der eigentliche Beitrag ist die Confound-Kontrolle, nicht der Architekturvergleich.
Ein Null-Resultat ist ein Ergebnis und wird als solches berichtet.

Status: **M0 — Präregistrierung im Entwurf, noch keine Freigabe, kein GPU-Verbrauch.**

## Dokumente

- [PREREGISTRATION.md](PREREGISTRATION.md) — Hypothesen, Matrix, Metriken, Ausschlussregeln
- [DEVIATIONS.md](DEVIATIONS.md) — Abweichungen nach dem Einfrieren
- [BUDGET.md](BUDGET.md) — GPU-Quota, Ist/Soll
- [FINDINGS.md](FINDINGS.md) — Ergebnisse (leer bis M5)
- [docs/baselines.md](docs/baselines.md) — exakte Baseline-Definitionen aus der Literatur

## Struktur

```
src/pdes/     convection, reaction, wave, allen_cahn + Referenzlösungen
src/models/   mlp, pinnsformer, grand, gread
src/train.py  EIN Trainer für alle Bedingungen (nicht verhandelbar)
bench/        Kalibrierung und Budgetplanung
kaggle/       Notebook-Generator und resume-fähiger Runner
analysis/     Auswertung und Plots
results/      results.sqlite — eine Zeile pro Lauf, auch Fehlläufe
```

## Ausführung

Produktivläufe laufen ausschließlich auf Kaggle-GPU. Lokal läuft nur der
CPU-Smoke-Test (200 Iterationen, 100 Kollokationspunkte), der vor jedem
Kaggle-Run grün sein muss — er kostet kein Quota und ist die Absicherung dagegen,
GPU-Stunden in einen Crash zu schicken.
