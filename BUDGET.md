# Budget

## Rahmen

| Posten | Wert |
|---|---|
| GPU-Quota gesamt | 29.0 h |
| Reserve (20 %, unantastbar) | 5.8 h |
| Planbudget (80 %) | 23.2 h |
| Zielhardware Kaggle | 2× T4 (Entscheidung 2026-08-20) |
| Session-Limit | 12 h pro Notebook-Run |
| Bereits verbraucht | 0.0 h |

## Anmerkung zur Hardwarewahl (relevant für die Nullhypothese)

Die FP64-Bedingung ist der Kern der Nullhypothese. FP64-Durchsatz unterscheidet
sich zwischen den Kaggle-Optionen um mehr als eine Größenordnung:

- **P100:** FP64 mit 1/2 der FP32-Rate (~4.7 TFLOPS) — FP64-Läufe sind bezahlbar.
- **T4:** FP64 mit 1/32 der FP32-Rate (~0.25 TFLOPS) — FP64-Läufe kosten ein
  Vielfaches; die zweite GPU hilft nur bei parallelen Läufen, nicht pro Lauf.

Gewählt wurde **2× T4**. Konsequenzen, die der Plan berücksichtigt:

- FP64-Läufe sind pro Lauf deutlich teurer als FP32. Die Präzisionsstufe wird
  trotzdem nicht gekürzt — sie ist der Kern der Nullhypothese. Gekürzt wird nach
  der in §3 der Präregistrierung festgelegten Reihenfolge.
- Kaggle zählt **Session-Wallclock**, nicht Device-Stunden. Zwei Worker-Prozesse,
  einer je GPU (`CUDA_VISIBLE_DEVICES=0` / `=1`), verdoppeln den Durchsatz pro
  Quota-Stunde. Der Runner ist entsprechend als 2-Worker-Design gebaut; beide
  Worker schreiben in dieselbe `results.sqlite` (WAL-Modus, Claim-Zeile pro Lauf).
- Der Smoke-Test läuft als **Kaggle-CPU-Session ohne Accelerator** und belastet
  das GPU-Quota nicht.

Die Kalibrierung (M2) misst das real, statt sich auf Datenblätter zu verlassen.

## Ist/Soll-Bilanz

Wird nach jedem Kaggle-Run vom budget-agent fortgeschrieben.

| Datum | Run | Zellen | Läufe | Geschätzt (h) | Ist (h) | Kumuliert (h) | Rest (h) |
|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | 0.0 | — |
