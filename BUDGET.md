# Budget

## Rahmen (Stand 2026-08-20, nach Neuskalierung)

| Posten | Stunden |
|---|---|
| GPU-Quota gesamt | **5.0** |
| Kalibrierung (M2a Probe + M2b Seed-0-Slice) | 0.5 |
| Stage A — Convection, 48 Laeufe (Seeds 1-4) | 2.0 |
| Stage B — Allen-Cahn, 60 Laeufe | 1.5 |
| Reserve (nicht verplanbar) | 1.0 |
| Bereits verbraucht | 0.0 |

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

Bei 2x T4 laeuft der Runner mit zwei Workern (`CUDA_VISIBLE_DEVICES=0` / `=1`), da
Kaggle Session-Wallclock zaehlt und nicht Device-Stunden. Bei P100 mit einem Worker.

Der Smoke-Test laeuft lokal auf CPU/1660 Ti und belastet das GPU-Quota nicht.

## Quota-Guard

Vor jedem Kaggle-Run wird die kumulierte Verbrauchsschaetzung geprueft. Ueberschreitet
sie das Stufenbudget, bricht der Guard ab, statt weiterzurechnen. Die Reserve ist fuer
den Guard nicht verfuegbar.

## Ist/Soll-Bilanz

Wird nach jedem Kaggle-Run fortgeschrieben.

| Datum | Run | Stufe | Laeufe | Geschaetzt (h) | Ist (h) | Kumuliert (h) | Rest (h) |
|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | 0.0 | 5.0 |
