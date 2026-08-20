# PDE-strukturierte Attention in PINNs

Kontrollierte Studie: Bringt eine PDE-strukturierte Mixing-Schicht (GRAND / GREAD)
als PINN-Backbone einen Vorteil gegenüber einem Vanilla-MLP und gegenüber
PINNsformer — **und überlebt dieser Vorteil die Kontrolle für numerische Präzision
und Regularisierung?**

Der eigentliche Beitrag ist die Confound-Kontrolle, nicht der Architekturvergleich.
Ein Null-Resultat ist ein Ergebnis und wird als solches berichtet.

Status: **M1 abgeschlossen, M2 vorbereitet, kein GPU-Verbrauch.** Die
Präregistrierung v3 ist über `PREREGISTRATION.lock.json` und Tag `prereg-v3`
eingefroren. Budget: 5 GPU-Stunden. Design gestuft (Stage A Convection, Stage B
Allen–Cahn), nicht vollfaktoriell. Pilot ohne konfirmatorischen Anspruch.

## Dokumente

- [HANDOFF.md](HANDOFF.md) — Übergabestand für den nächsten Bearbeiter

- [PREREGISTRATION.md](PREREGISTRATION.md) — Hypothesen, Matrix, Metriken, Ausschlussregeln
- [DEVIATIONS.md](DEVIATIONS.md) — Abweichungen nach dem Einfrieren
- [BUDGET.md](BUDGET.md) — GPU-Quota, Ist/Soll
- [FINDINGS.md](FINDINGS.md) — Ergebnisse (leer bis M5)
- [docs/baselines.md](docs/baselines.md) — exakte Baseline-Definitionen aus der Literatur
- [docs/execution-contract.md](docs/execution-contract.md) — Kaggle-Resume und Persistenz

## Struktur

```
src/pdes/     convection, allen_cahn + Referenzlösungen
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
`lambda_r = 1.0` wurde outcome-blind aus Initial-Lossskalen gewählt; Artefakt:
`results/lambda_r_selection.json`.

## Nächster externer Schritt

M2a erzeugt getrennte Timing-Proben auf P100 und 2x T4. Es startet erst, wenn das
versionierte Code-Dataset und das private Results Dataset auf Kaggle bereitstehen.
Das Notebook wird mit `kaggle/build_notebook.py` als dünner Launcher generiert; der
Projektcode wird als Dataset eingebunden und nicht in das Notebook kopiert.
