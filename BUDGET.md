# Budget

## Rahmen

| Posten | Wert |
|---|---|
| GPU-Quota gesamt | _(offen — vom Projektleiter zu setzen)_ |
| Reserve (20 %, unantastbar) | _(offen)_ |
| Planbudget (80 %) | _(offen)_ |
| Zielhardware Kaggle | _(offen — P100 vs. 2× T4, siehe Anmerkung)_ |
| Session-Limit | 12 h pro Notebook-Run |
| Bereits verbraucht | 0.0 h |

## Anmerkung zur Hardwarewahl (relevant für die Nullhypothese)

Die FP64-Bedingung ist der Kern der Nullhypothese. FP64-Durchsatz unterscheidet
sich zwischen den Kaggle-Optionen um mehr als eine Größenordnung:

- **P100:** FP64 mit 1/2 der FP32-Rate (~4.7 TFLOPS) — FP64-Läufe sind bezahlbar.
- **T4:** FP64 mit 1/32 der FP32-Rate (~0.25 TFLOPS) — FP64-Läufe kosten ein
  Vielfaches; die zweite GPU hilft nur bei parallelen Läufen, nicht pro Lauf.

Die Kalibrierung (M2) misst das real, statt sich auf Datenblätter zu verlassen.

## Ist/Soll-Bilanz

Wird nach jedem Kaggle-Run vom budget-agent fortgeschrieben.

| Datum | Run | Zellen | Läufe | Geschätzt (h) | Ist (h) | Kumuliert (h) | Rest (h) |
|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | 0.0 | — |
