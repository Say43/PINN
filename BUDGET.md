# Budget

## Rahmen (Stand 2026-08-20, nach Neuskalierung)

| Posten | Stunden |
|---|---|
| GPU-Quota gesamt | **5.0** |
| Kalibrierung (M2a Probe + M2b Seed-0-Slice) | 0.5 |
| Stage A — Convection, 48 Laeufe (Seeds 1-4) | 2.0 |
| Stage B — Allen-Cahn, 60 Laeufe | 1.5 |
| Reserve (nicht verplanbar) | 1.0 |
| Bereits verbraucht (Kaggle-Abrechnung, seit Projektstart) | 0.57 |

Die Reserve ist ausschliesslich fuer Wiederholungen nach Abstuerzen. Sie wird nicht
fuer eine Stage C verplant.

Session-Limit: 12 h pro Notebook-Run. Netzwerk in Kaggle-Notebooks standardmaessig aus.
`/kaggle/working` geht beim Session-Ende verloren, wenn nicht als Dataset committed.

## Hardware — offen, wird gemessen

Frueher auf 2x T4 festgelegt, zurueckgenommen. Entscheidung faellt datenbasiert in M2a.

Die Matrix ist zur Haelfte FP64, und FP64-Durchsatz trennt die Optionen um mehr als
eine Groessenordnung:

| | FP32 | FP64 | Rate |
|---|---|---|---|
| P100 (Pascal GP100) | ~9.3 TFLOPS | ~4.7 TFLOPS | 1/2 |
| T4 (Turing) | ~8.1 TFLOPS | ~0.25 TFLOPS | 1/32 |

Auf 2x T4 gewinnt man Faktor 2 durch Parallelitaet und verliert auf der Haelfte aller
Laeufe bis Faktor 32. Erwartung ist daher P100. Gemessen wird trotzdem: M2a faehrt
beide Optionen mit je einer FP32- und einer FP64-Zelle.

Nach dem Praeregistrierungs-Freeze wurde der urspruengliche 2-Worker-Plan verworfen:
Auch bei 2x T4 laeuft genau **ein** Worker. Nur so kann ein Session-Kill hoechstens
den einen aktuell laufenden Einzellauf vernichten. Die zweite T4 bleibt ungenutzt;
dieser Sicherheitsnachteil geht ehrlich in die Hardwarewahl von M2a ein. Siehe
`DEVIATIONS.md`, D-1.

Der Smoke-Test laeuft lokal auf CPU und belastet das GPU-Quota nicht. Die lokale
1660 Ti darf nur fuer kurze Korrektheitstests bzw. die outcome-blinde lambda_r-
Skalierung mit harter Laufzeitgrenze unter drei Minuten verwendet werden; sie
rechnet keinen Studienarm.

## Quota-Guard

Vor jedem Kaggle-Run wird die kumulierte Verbrauchsschaetzung geprueft. Ueberschreitet
sie das Stufenbudget, bricht der Guard ab, statt weiterzurechnen. Die Reserve ist fuer
den Guard nicht verfuegbar.

## Ist/Soll-Bilanz

Wird nach jedem Kaggle-Run fortgeschrieben.

## M2-Iststand und finales Ausfuehrungsbudget (2026-08-20)

- Kaggle-Account vor Projektstart: 0.38 h; nach M2/Profiling: 0.95 h.
  Damit sind konservativ **0.57 h** dem Projekt zugerechnet.
- Interne Zuordnung: 0.542154 h Kalibrierung/Notebook-Overhead und 0.027846 h
  outcome-freies Stage-A-Profiling. Die externe Abrechnung ist fuer das
  Gesamtbudget massgeblich.
- Die verbindliche Kuerzungshierarchie hat Stage B gestrichen und Stage A deren
  1.5 h zugeschlagen. Double Backprop wurde danach ebenfalls gestrichen.
- Verbleibendes Stage-A-Budget nach Profiling: **3.472154 h**.
- Finaler Plan: 6 Zellen x 5 Seeds, 6000 Iterationen, 396 Loss-Punkte, ein T4-Worker.
  Projektion fuer M2b plus restliche Stage A: **3.222531 h**.
- Projektion gesamt: 0.57 + 3.222531 = **3.792531 h**. Damit bleiben 1.207469 h,
  davon 1.0 h unantastbare Reserve und rund 0.207 h operative Marge.

| Datum | Run | Stufe | Laeufe | Geschaetzt (h) | Ist (h) | Kumuliert (h) | Rest (h) |
|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | 0.0 | 5.0 |

## Zusatzbudgetrechnung fuer die Wiederherstellung des Scopes (2026-08-20)

Erstellt **vor** jeder Sichtung von M2b-Kennzahlen. Grundlage sind ausschliesslich
die M2a-Zeitmessungen (T4, reduziert, ein Worker) aus `results/m2_plan.json`. Keine
Fehler-, Loss- oder Erfolgswerte sind eingeflossen.

Basis: 6 Zellen `none`, Summe 0.386704 s/Iteration, 6000 Iterationen, 5 Seeds.

| Baustein | Stunden | Grundlage |
|---|---|---|
| Stage A Convection `none`, 5 Seeds | 3.223 | gemessen |
| — davon M2b (Seed 0) | 0.645 | laeuft |
| — davon Seeds 1–4 | 2.578 | offen |
| Double Backprop Convection, 5 Seeds | 6.445 | gemessen x Planerfaktor 2.0 |
| Stage B Allen-Cahn `none`, 5 Seeds | 17.293 | **extrapoliert** |
| Stage B Allen-Cahn Double Backprop | 34.585 | **extrapoliert** |

Die Allen-Cahn-Zahlen sind **nicht gemessen**. Sie skalieren die Convection-Zeiten
linear mit der Punktzahl (2125 statt 396, Faktor 5.37). Linear ist fuer k-NN-Graphen
mit O(N·k) und fuer das MLP plausibel, aber unbestaetigt. Vor einer Freigabe von
Stage B muesste eine eigene M2a-Probe auf Allen-Cahn laufen.

### Gesamtbedarf je Ausbaustufe (zzgl. 1.0 h Reserve)

| Option | Umfang | Gesamt-Quota |
|---|---|---|
| **A** | schmaler Convection-Pilot wie geplant | **3.79 h** |
| **B** | A + Double Backprop auf Convection | **10.24 h** |
| **C** | B + Allen-Cahn `none` | **27.53 h** |
| **D** | vollstaendiges praeregistriertes Design | **62.12 h** |

**Das urspruengliche Forschungsziel kostet rund das Zwoelffache des bewilligten
Budgets.** Die 5-h-Grenze konnte es nie tragen; das ist jetzt beziffert statt
vermutet.

Guenstigster wissenschaftlich sinnvoller Ausbau ist **B**: Double Backprop stellt
die Regularisierungshaelfte von H0 innerhalb derselben PDE wieder her, und die
Confound-Kontrolle war von Anfang an als der eigentliche Beitrag benannt. Der
Allen-Cahn-Kontrast ist eine eigene Groessenordnung und eine eigene Entscheidung.

Nebenhebel, falls Allen-Cahn dennoch gewuenscht ist: dessen Kosten haengen fast
ausschliesslich an der Punktzahl. Bei 396 Punkten wie Convection kostete Stage B
`none` ebenfalls nur 3.22 h. Ob Allen-Cahn mit 0.0001 Diffusion und scharfen
Interfaces auf 396 Punkten ueberhaupt sinnvoll aufgeloest ist, ist damit aber
offen und waere vorab zu pruefen, nicht hinterher zu behaupten.

### Dokumentationsnotiz

`HANDOFF.md` nennt fuer das reduzierte Allen-Cahn-Schema 2175 Punkte;
`kaggle/runner.py` setzt 2025 + 50 + 50 = **2125**. Der Code ist massgeblich.
