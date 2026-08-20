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
