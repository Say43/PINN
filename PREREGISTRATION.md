# Präregistrierung — PDE-strukturierte Attention in PINNs

**Status: ENTWURF (M0).** Wird nach Freigabe durch den Projektleiter und nach
Fertigstellung von `docs/baselines.md` (lit-agent) eingefroren. Ab dem Einfrieren
gilt: keine Änderung mehr an diesem Dokument; jede Abweichung wird in
`DEVIATIONS.md` mit Datum und Begründung protokolliert.

Eingefroren am: _(offen)_
Git-Commit beim Einfrieren: _(offen)_

---

## 1. Forschungsfrage

Verbessert eine PDE-strukturierte Mixing-Schicht (GRAND / GREAD) als Backbone eines
PINNs dessen Konvergenz und Robustheit gegen bekannte Failure Modes, verglichen mit
einem Vanilla-MLP-PINN und mit PINNsformer (Standard-Attention über eine
Pseudo-Sequenz von Kollokationspunkten)?

Motivation: In allgemeinen Sprachmodellen ist ein diffusionsartiger induktiver Bias
unmotiviert. In einem PINN ist die Zielfunktion selbst eine PDE-Lösung — der Bias
passt zur Aufgabe.

## 2. Nullhypothese

**H0:** Jeder beobachtete Architekturgewinn verschwindet, sobald für numerische
Präzision (FP32 vs. FP64) und für Regularisierung (Weight Decay, Double Backprop
auf den Residuen) kontrolliert wird.

H0 wird ernst genommen und nicht wegargumentiert. Ein sauberes Null-Resultat ist ein
vollwertiges Projektergebnis und wird in `FINDINGS.md` an erster Stelle berichtet.

**Falsifikationsbedingung für H0:** Der Backbone-Haupteffekt (GRAND/GREAD vs. MLP)
bleibt über *alle* Confound-Stufen hinweg gerichtet gleich und in Success Rate
mindestens +15 Prozentpunkte, mit über Seeds bootstrapped 95-%-KI, das die Null
nicht einschließt, in mindestens der Hälfte der Benchmark-PDEs.

**Bestätigung von H0:** Der Backbone-Effekt ist unter FP64 und/oder unter
Regularisierung statistisch nicht mehr von null unterscheidbar, oder er kehrt das
Vorzeichen. Dann lautet das Ergebnis: der Gewinn war ein Präzisions- bzw.
Regularisierungsartefakt.

## 3. Bedingungsmatrix

| Faktor | Stufen | n |
|---|---|---|
| PDE | convection (Failure-Regime), reaction, wave, allen_cahn | 4 |
| Backbone | mlp, pinnsformer, grand, gread | 4 |
| Präzision | fp32, fp64 | 2 |
| Regularisierung | none, weight_decay, double_backprop | 3 |
| Seed | 0..9 (Minimum 5, Ziel 10) | 5–10 |

Volle Matrix = 96 Zellen × Seeds. Ob finanzierbar, entscheidet `bench/plan.py`
auf Basis der Kalibrierung (M2). **Kürzungsregel: Bei Budgetmangel wird die Matrix
gekürzt, niemals die Seed-Zahl unter 5 gesenkt.** Kürzungsreihenfolge, vorab
festgelegt:

1. PDEs von 4 auf 2 (behalten: convection im Failure-Regime, reaction)
2. Regularisierung von 3 auf 2 (behalten: none, double_backprop)
3. Backbone von 4 auf 3 (streichen: pinnsformer — es ist Vergleichsbaseline, nicht
   Gegenstand der Hypothese)

Präzision wird **nie** gekürzt. Sie ist der Kern der Nullhypothese.

### PDE-Parameter

Werden aus Krishnapriyan et al. (NeurIPS 2021) exakt übernommen und vor dem
Einfrieren dieses Dokuments durch den lit-agent in `docs/baselines.md` mit
Seitenangabe belegt. Arbeitsannahme bis dahin (NICHT als belegt behandeln):

- **Convection:** u_t + β u_x = 0, x ∈ [0, 2π], t ∈ [0, 1], β im Failure-Regime
  (β ≈ 30–40), IC u(x,0) = sin(x), periodische Randbedingung.
- **Reaction:** u_t − ρ u (1 − u) = 0, ρ im Failure-Regime (ρ ≈ 5–10),
  gaußförmige IC.
- **Wave:** u_tt − β² u_xx = 0 mit dem in der Quelle angegebenen β.
- **Allen–Cahn:** steht *nicht* in Krishnapriyan et al. Quelle für die Parameter
  wird der lit-agent bestimmen (Kandidat: Wang et al., *An Expert's Guide*).
  Widerspruch dokumentieren statt still eine Variante wählen.

### Fairness-Auflagen

- Parameterzahl über alle Backbones auf ±10 % angeglichen. Wird pro Konfiguration
  in die Ergebnis-DB geschrieben und in `FINDINGS.md` tabelliert.
- Identischer Trainer, identisches Optimierer-Schema, identisches Iterationsbudget,
  identische Kollokationspunkte für alle Bedingungen. Kein architekturspezifisches
  Tuning.
- Verletzt eine Bedingung diese Auflagen, wird der Vergleich als unfair markiert
  und nicht als Evidenz gewertet.

## 4. Metriken

**Primär: Success Rate über Seeds.** Ein Lauf gilt als Erfolg, wenn der relative
L2-Fehler gegen die Referenzlösung auf einem festen Auswertungsgitter unter der
Schwelle liegt:

| PDE | Erfolgsschwelle (rel. L2) |
|---|---|
| convection | 0.10 |
| reaction | 0.10 |
| wave | 0.10 |
| allen_cahn | 0.10 |

Die Schwelle 0.10 ist vor jedem Lauf fixiert und wird nicht nachträglich
angepasst. (Begründung: in der Failure-Mode-Literatur trennt sie klar zwischen
„Lösung gefunden" und „auf triviale/verschobene Lösung kollabiert".)

**Sekundär:**
- Median und IQR des relativen L2-Fehlers über Seeds
- Wall-Clock bis zum Erreichen der Erfolgsschwelle (zensiert, falls nie erreicht)
- Anzahl Funktionsauswertungen des ODE-Solvers (nur grand/gread)
- finaler Residual-Loss und Randbedingungs-Loss getrennt

## 5. Auswertung

- Konfidenzintervalle über Seeds: Bootstrap-Perzentil, 95 %, 10 000 Resamples.
- Success Rate: Wilson-Intervall.
- Keine p-Werte ohne Korrektur für die Zahl der Vergleiche (Holm–Bonferroni über
  die Zahl der Backbone-Paarvergleiche pro PDE).
- Zentrale Auswertung ist **nicht** „welcher Backbone gewinnt", sondern die
  Interaktion Backbone × Präzision und Backbone × Regularisierung. Ein Backbone,
  der nur bei fp32-ohne-Regularisierung vorne liegt, hat nichts gezeigt.
- Der beste Einzellauf wird nirgends als Ergebnis berichtet.

## 6. Ausschlussregeln (vorab, abschließend)

Ein Lauf wird aus der Auswertung ausgeschlossen **nur** wenn:

1. der Prozess mit einem Infrastrukturfehler abbricht (OOM, Session-Kill,
   CUDA-Fehler) — der Lauf wird mit `status='infra_fail'` in der DB behalten und
   mit gleichem Seed wiederholt; oder
2. die Referenzlösung für die Zelle nicht erzeugt werden konnte.

Divergenz, NaN im Loss oder ein hoher Endfehler sind **kein** Ausschlussgrund. Sie
sind das Messergebnis und zählen als Misserfolg. Alle Läufe, auch abgestürzte,
landen in `results/results.sqlite`.

## 7. Abbruchkriterien für das Projekt

- **Budget:** Der Quota-Guard bricht ab, sobald die kumulierte Verbrauchsschätzung
  80 % des Gesamtquotas erreicht. 20 % bleiben Reserve.
- **Inhaltlich:** Zeigt die Pilotmatrix (M3), dass der Backbone-Effekt bereits bei
  fp32/none unter 5 Prozentpunkten Success-Rate-Differenz liegt, wird die volle
  Matrix nicht gerechnet. Es wird als Null-Resultat mit der Pilotstatistik
  berichtet und das Restbudget in mehr Seeds auf der Pilotmatrix investiert.
- **Technisch:** Lässt sich die Parameterzahl nicht auf ±10 % angleichen, ohne eine
  Architektur zu verstümmeln, wird das dokumentiert und der betroffene Vergleich
  als nicht-belastbar gekennzeichnet.

## 8. Was das Projekt nicht ist

Kein Versuch, klassische Solver (FEM/FV) in Rechenzeit zu schlagen. Keine
Skalierung auf große Modelle oder 3D. Keine Sprachmodellierung.
