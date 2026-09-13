# PDE-strukturierte Attention in PINNs

Kontrollierte Studie: Bringt eine PDE-strukturierte Mixing-Schicht (GRAND / GREAD)
als PINN-Backbone einen Vorteil gegenüber einem Vanilla-MLP und gegenüber
PINNsformer — **und überlebt dieser Vorteil die Kontrolle für numerische Präzision
und Regularisierung?**

Der eigentliche Beitrag ist die Confound-Kontrolle, nicht der Architekturvergleich.
Ein Null-Resultat ist ein Ergebnis und wird als solches berichtet.

Status: **V3 beendet, V5 technisch vorbereitet, aber noch nicht eingefroren.** V3
wurde nach dem vorab definierten M2b-Gate gestoppt, weil keine Bedingung die
Auflösungsprüfung bestand. Lokale, ausdrücklich nicht als Studiendaten geführte
Diagnosen trennten anschließend Aliasing von Optimierungskollaps. Der neue
V5-Entwurf untersucht Reaction bei `rho = 5.25`; die Hauptmatrix wurde noch nicht
gestartet. Insgesamt wurden rund 1.48 von 27 verfügbaren GPU-Stunden verbraucht.

## Dokumente

- [HANDOFF.md](HANDOFF.md) — Übergabestand für den nächsten Bearbeiter

- [PREREGISTRATION.md](PREREGISTRATION.md) — Hypothesen, Matrix, Metriken, Ausschlussregeln
- [PREREGISTRATION-V5.md](PREREGISTRATION-V5.md) — aktueller Reaction-Entwurf
- [DEVIATIONS.md](DEVIATIONS.md) — Abweichungen nach dem Einfrieren
- [BUDGET.md](BUDGET.md) — GPU-Quota, Ist/Soll
- [FINDINGS.md](FINDINGS.md) — Ergebnisse (leer bis M5)
- [docs/baselines.md](docs/baselines.md) — exakte Baseline-Definitionen aus der Literatur
- [docs/execution-contract.md](docs/execution-contract.md) — Kaggle-Resume und Persistenz

## Struktur

```
src/pdes/     convection, allen_cahn, reaction + Referenzlösungen
src/models/   mlp, grand, gread (pinnsformer nur bei Restbudget)
src/train.py  EIN Trainer für alle Bedingungen (nicht verhandelbar)
bench/        Kalibrierung und Budgetplanung
kaggle/       Notebook-Generator und resume-fähiger Runner
analysis/     Auswertung und Plots
results/      results.sqlite — eine Zeile pro Lauf, auch Fehlläufe
```

## Ausführung

Alle Läufe, die in die Auswertung eingehen, stammen von derselben Kaggle-Hardware.
Die lokale GPU darf ausschließlich für Korrektheitstests und die outcome-blinde
`lambda_r`-Skalierung mit harter Laufzeitgrenze unter drei Minuten dienen. Sie
rechnet **keinen** Arm der Matrix, weil das Präzision mit Hardware konfundieren würde.

Der Smoke-Test (200 Iterationen, 100 Kollokationspunkte) muss vor jedem Kaggle-Run
grün sein. Er kostet kein Quota.

## Lokale Prüfung (CPU)

```powershell
python -m unittest discover -s tests -v
python -m src.train --config configs/smoke.json
```

Der verifizierte Smoke-Lauf absolvierte 200 L-BFGS-Iterationen mit 100 Loss-Punkten,
schrieb eine echte SQLite-Zeile und wurde beim zweiten Aufruf korrekt übersprungen.
Der härteste Pfad (GREAD + FP64 + Double Backprop) bestand denselben 200×100-Test
auf CPU in 13.86 s einschließlich 854 Solver-NFE und ebenfalls geprüftem Resume.
Für V3 wurde `lambda_r = 1.0` outcome-blind aus Initial-Lossskalen gewählt;
Artefakt: `results/lambda_r_selection.json`. V5 verwendet eine neue, noch
auszuführende Auswahl aus `{1e-5, 1e-4, 1e-3, 1e-2}`.

## Nächster externer Schritt

Zuerst werden die vier V5-`lambda_r`-Auswahlläufe auf 2× T4 ausgeführt. Der neue
Runner nutzt je einen Prozess pro GPU, während nur der Elternprozess SQLite und das
private Results Dataset aktualisiert. Erst nach Auswahl von `lambda_r`, Abschluss
und Freeze der Präregistrierung darf die 60-Lauf-Matrix starten. Der Matrix-Runner
prüft Status und SHA-256 der Präregistrierung vor dem ersten Lauf.
