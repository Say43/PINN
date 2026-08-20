# Abweichungen von der Praeregistrierung

Nach dem Einfrieren wird hier jede Aenderung protokolliert: Datum, Abschnitt, vorher,
nachher, Begruendung, ob die betroffenen Ergebnisse bereits gesichtet waren, Commit.

**Die Praeregistrierung ist noch NICHT eingefroren.** Die folgenden Eintraege sind
daher keine Abweichungen von der Praeregistrierung, sondern **dokumentierte
Abweichungen von der Literatur**, die vor dem Einfrieren bewusst getroffen wurden.

---

## L-1 — Referenzloesung Convection: geschlossene Form statt Spektralverfahren
- **Gegen:** KP21 Gl. 6 (FFT-Rueckstransformation)
- **Gewaehlt:** `u(x,t) = sin(x - beta*t)` nach FP64 Gl. 6 / AM26 Gl. 9
- **Begruendung:** mathematisch aequivalent, numerisch nicht. Die FFT-Variante
  akkumuliert Rundungsfehler. Bei einer Studie mit Praezision als Nullhypothese
  waere eine rundungsfehlerbehaftete Referenz ein Eigentor.

## L-2 — Allen-Cahn-Parameter aus Expert's Guide statt Krishnapriyan
- **Gegen:** die Auftragsvorgabe "Parameter exakt aus Krishnapriyan"
- **Gewaehlt:** Expert's Guide §7.1 Gl. 7.1-7.4
- **Begruendung:** Allen-Cahn steht nicht in KP21 (Volltextsuche: null Treffer fuer
  "Allen", "Cahn", "wave", "Burgers"). Die Vorgabe ist fuer diese PDE nicht
  erfuellbar. Expert's Guide, FP64 §A.4 und AM26 §B.4 stimmen ueberein.

## L-3 — Allen-Cahn-Parameter unter L-BFGS statt Adam
- **Gegen:** Expert's Guide, das ausschliesslich Adam verwendet
- **Gewaehlt:** L-BFGS, wie AM26 §B.4 es ebenfalls tut
- **Begruendung:** der Praezisions-Confound ist L-BFGS-spezifisch (FP64 §5.2). Mit
  Adam waere H0 nicht testbar. Literaturgedeckt durch AM26.

## L-4 — Abbruchtoleranz deaktiviert
- **Gegen:** FP64 (feste Toleranz 1e-7) und AM26 (praezisionsabhaengig, sqrt(eps))
- **Gewaehlt:** `tolerance_change = tolerance_grad = 0`, feste Iterationszahl
- **Begruendung:** beide Literaturvarianten erzeugen je nach Praezision
  unterschiedlich lange Laeufe und konfundieren Praezision mit Trainingsdauer. Die
  Fairness-Auflage verlangt ein identisches Iterationsbudget. Der gemessene
  Praezisionseffekt ist damit ein reiner Arithmetikeffekt.
- **Konsequenz:** der von FP64 beschriebene Stopp-Confound wird bewusst NICHT
  reproduziert. Zeigt sich unter dieser Bedingung kein Praezisionseffekt, ist das
  ein Befund ueber die Reichweite der FP64-These, keine Widerlegung.

## L-5 — Aktivierung tanh statt ReLU
- **Gegen:** PINNsformer Tab. 4 (Baseline-PINN mit ReLU)
- **Gewaehlt:** tanh
- **Begruendung:** ReLU hat verschwindende zweite Ableitung und ist fuer Residuen
  mit u_xx untauglich. 3 von 4 Quellen verwenden tanh.

## L-6 — Parameterzahl-Zielgroesse ~50k (4x128)
- **Gegen:** keine einheitliche Vorgabe; Literatur streut 7.8k (KP21) bis 527k
  (PINNsformer)
- **Gewaehlt:** AM26-Konfiguration, 4 Hidden Layers x 128, tanh, Glorot-normal
- **Begruendung:** aus AM26 stammt der Double-Backprop-Arm, und die
  Confound-Kontrolle ist der Kern der Studie.

## L-7 — lambda_r wird selbst bestimmt
- **Gegen:** nichts; AM26 gibt den Wert schlicht nicht an (auch nicht im Appendix)
- **Gewaehlt:** Grid {1e-4, 1e-3, 1e-2, 1e-1, 1} einmalig auf
  Convection/MLP/FP32 mit 100 Domain-, 50 BC- und 50 IC-Punkten. Fuer Seeds 0-4
  wird vor dem Training das Verhaeltnis `(lambda_r/2)*P/L_0` berechnet; gewaehlt
  wird der Gridwert mit Median am naechsten zu 0.1, bei Gleichstand der kleinere.
  Referenzfehler und Trainingsverlauf bleiben ungenutzt. Danach gilt der Wert fuer
  alle Bedingungen identisch.
- **Begruendung:** ohne Wert nicht implementierbar. Die loss-skalierte,
  outcome-blinde Regel verhindert ein Tuning auf die Zielmetrik und ist lokal ohne
  Kaggle-Quota ausfuehrbar. Eine lokale GPU darf nur unter einer harten
  Drei-Minuten-Grenze verwendet werden.
- **Ergebnis (2026-08-20):** `lambda_r = 1.0`. Der Median von
  `(lambda_r/2)*P/L_0` war 0.1239904 und damit im praeregistrierten Grid am
  naechsten am Ziel 0.1. Berechnung ausschliesslich auf CPU in 6.5 s; keine
  Referenzfehler, Trainingskurven oder Studienergebnisse wurden gesichtet.
- **Artefakt:** `results/lambda_r_selection.json`.

## L-8 — Budgetkuerzung reduziert Punkte vor dem Streichen von Allen-Cahn
- **Gegen:** AM26-Punktzahlen 800 (Convection) und 4396 (Allen-Cahn)
- **Gewaehlt:** Falls M2a sonst weniger als 5000 Iterationen erlaubt, einmalige
  Reduktion auf 396 bzw. 2175 Loss-Punkte, identisch fuer alle Bedingungen je PDE.
- **Begruendung:** Das Streichen von Allen-Cahn beseitigt die praeregistrierte
  Doppeldissoziation und damit den Mechanismustest. Eine outcome-unabhaengige,
  symmetrische Punktreduktion bewahrt ihn und ist durch AM26s Overfitting-Befund
  inhaltlich plausibel. Erst wenn diese Stufe nicht reicht, wird Stage B gestrichen.

---

## Abweichungen nach dem Praeregistrierungs-Freeze

## D-1 — 2x T4 seriell statt mit zwei Workern
- **Datum:** 2026-08-20
- **Abschnitt:** Ausfuehrung/Budget; die Praeregistrierung selbst legt W nicht fest,
  der Uebergabestand und `BUDGET.md` sahen fuer 2x T4 jedoch W=2 vor.
- **Vorher:** zwei parallele Laeufe, einer je T4.
- **Nachher:** genau ein aktiver Studienlauf auf jeder Hardwareoption.
- **Begruendung:** Bei einem Session-Kill koennten sonst zwei unvollstaendige Laeufe
  verloren gehen. Das verletzt die haertere Persistenzregel "nie mehr als einen
  Einzellauf". Serieller Betrieb ist fail-safe und macht den Nachteil von T4 bei
  FP64 in der Hardwarewahl sichtbar, statt ihn durch riskante Parallelitaet zu
  kaschieren.
- **Ergebnisse gesichtet:** nein; keine Kaggle- oder Studienlaeufe gestartet,
  GPU-Verbrauch weiterhin 0.0 h.
- **Commit:** wird mit M1/M2-Vorbereitung eingetragen.
