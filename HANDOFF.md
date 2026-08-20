# Übergabe an GPT Sol

> **AKTUELLER NACHTRAG — ersetzt den darunter stehenden historischen M0-Stand.**
> M0 und M1 sind abgeschlossen; M2 ist implementiert, aber nicht auf Kaggle
> ausgeführt. GPU-Verbrauch bleibt **0.0 von 5.0 h**. Der frühere Text bleibt als
> Provenienz erhalten.

## Aktueller Stand nach der Übernahme

- Präregistrierung v3 eingefroren: Commit
  `80cba6050642e29121af44517d2928474b3cff37`, SHA-256
  `809e27249e424189a5e074293288f14d7bf1e77518b9eb5e2d6a143126832d40`,
  Tag `prereg-v3`, Lock in `PREREGISTRATION.lock.json`.
- M1 implementiert: ein Trainer für Convection/Allen–Cahn, MLP/GRAND/GREAD,
  FP32/FP64 und none/double_backprop; deterministische Ausführung, SQLite-Attempts,
  Integritätsbackup und Resume.
- 17/17 Unit-Tests und `compileall` grün; der kombinierte Trainer-Test umfasst 24
  PDE/Backbone/Präzisions/Regularisierungs-Subfälle. CPU-Smoke MLP: 200
  L-BFGS-Iterationen, exakt 100 Loss-Punkte, 2.43 s, 435 Closure-Auswertungen.
  Härtester CPU-Smoke GREAD+FP64+Double-Backprop: 200×100, 13.86 s, 854 Solver-NFE.
  Beide schrieben echte SQLite-Zeilen und wurden beim zweiten Aufruf übersprungen.
- `lambda_r = 1.0`, outcome-blind aus Initial-Lossskalen auf CPU gewählt. Artefakt:
  `results/lambda_r_selection.json`. Die lokale GPU wurde nicht verwendet.
- Kanonische Allen–Cahn-Referenz eingebunden: `data/allen_cahn.mat`, SHA-256
  `ce640f188e334520f636d3d650cae6056a92a486546d40889bff93610bdbfa71`.
- M2 vorbereitet: `bench/calibrate.py`, `bench/plan.py`, Quota-Guard,
  `kaggle/build_notebook.py`, serieller Runner und fail-closed Publikation nach
  jedem terminalen Lauf. Details: `docs/execution-contract.md`.

## Aufgelöste frühere offene Punkte

1. M2a/M2b bleibt unverändert und ist implementiert.
2. Bei Budgetdruck werden zuerst outcome-unabhängig die Punktzahlen auf 396 bzw.
   2175 reduziert; erst danach wird Stage B gestrichen.
3. Die AM26-Summe 4396 ist korrekt: 4096 Domain + 100 IC + je 100 Residuen für
   Periodizität von `u` und `u_x`.
4. Aus Sicherheitsgründen läuft auch auf 2x T4 nur ein Studienworker. Sonst könnte
   ein Session-Kill zwei Läufe vernichten; dokumentiert als D-1.

## Nächster externer Schritt

Benötigt werden Kaggle-Username sowie private Slugs für Code- und Results-Dataset.
Danach je ein M2a-Notebook für P100 und 2x T4 erzeugen und ausführen, beide JSONL
mit `bench/plan.py` auswerten, M2b fahren und **danach vor Stage A anhalten**. Keine
Tokens oder Zugangsdaten ins Repo schreiben.

---

Stand: 2026-08-20. Repo: `c:\Festplatte (D)\Dateien\AI\PINN`, Branch `master`,
sauberer working tree. **Kein GPU-Quota verbraucht (0.0 von 5.0 h).**

---

## 1. Wo das Projekt steht

**Meilenstein M0, kurz vor Abschluss.** Es existieren Dokumente, aber **noch kein
Code** — kein `src/`, kein `bench/`, kein `kaggle/`, keine Tests, keine
`results.sqlite`. Die Verzeichnisse sind leer angelegt.

| Datei | Zustand |
|---|---|
| `PREREGISTRATION.md` | **v3, NICHT eingefroren** — wartet auf Freigabe der Projektleitung |
| `docs/baselines.md` | fertig, 1208 Zeilen, alle 7 Quellen im Volltext gelesen |
| `DEVIATIONS.md` | 7 Einträge L-1…L-7 (Abweichungen von der Literatur, vor dem Freeze) |
| `BUDGET.md` | Rahmen steht, Ist/Soll-Tabelle leer |
| `FINDINGS.md` | leer bis M5 |
| `README.md` | steht |

**Nächster Schritt ist nicht Code schreiben.** Nächster Schritt ist die Freigabe der
Präregistrierung durch die Projektleitung, dann der Freeze-Commit. Erst danach M1.

---

## 2. Das Projekt in fünf Sätzen

Forschungsfrage: Entkommt ein PINN mit PDE-strukturiertem Backbone (GRAND / GREAD)
den bekannten Failure Modes innerhalb eines festen, kleinen Iterationsbudgets besser
als ein Vanilla-MLP-PINN? Der eigentliche Beitrag ist nicht der Architekturvergleich,
sondern die Confound-Kontrolle: **H0 sagt, jeder Architekturgewinn verschwindet,
sobald man für numerische Präzision (FP32/FP64) und Regularisierung (Double
Backprop) kontrolliert.** H0 wird ernst genommen; ein Null-Resultat ist ein
vollwertiges Ergebnis und wird zuerst berichtet. Das Ganze ist ein **Pilot ohne
konfirmatorischen Anspruch** — 5 GPU-Stunden geben nicht mehr her, und das steht so
im ersten Absatz von `FINDINGS.md`. Zusätzlich präregistriert ist eine gerichtete
**Doppeldissoziation** (H1): GRAND sollte auf Convection *schlechter* sein, GREAD auf
Allen–Cahn besser — gewinnt GRAND auf beiden, spricht das gegen die
Mechanismus-Erklärung und für einen unspezifischen Kapazitätseffekt.

---

## 3. Harte Constraints

```
GPU-Quota:       5.0 h gesamt  (0.5 Kalibrierung / 2.0 Stage A / 1.5 Stage B / 1.0 Reserve)
Hardware:        OFFEN — P100 vs. 2x T4, wird in M2a gemessen, nicht angenommen
Session-Limit:   12 h pro Kaggle-Notebook-Run
Persistenz:      /kaggle/working geht verloren, wenn nicht als Dataset committed
Netzwerk:        in Kaggle-Notebooks standardmäßig aus
```

Die **Reserve ist unantastbar** und ausschließlich für Wiederholungen nach Abstürzen.

**Lokale Hardware:** GTX 1660 Ti (6 GB), Python 3.13.5, Torch 2.6.0+cu124. Erlaubt
für Korrektheitstests, Debugging, grobe Kostenrangfolge und die λ_r-Bestimmung.
**Verboten:** damit einen Arm der Matrix rechnen. Präzision und Hardware dürfen nicht
konfundiert werden — das wäre exakt der Fehler, den die Studie anderen nachweist.

---

## 4. Design in Kurzform

**Stage A — Convection, β = 50** (60 Läufe)
3 Backbones (mlp, grand, gread) × 2 Präzisionen (fp32, fp64) × 2 Regularisierungen
(none, double_backprop) × 5 Seeds. Die 12 Seed-0-Läufe entstehen in der Kalibrierung
und zählen als Daten.

**Stage B — Allen–Cahn** (60 Läufe), identisches Design, Gate nach Stage A.

**PINNsformer ist nicht im Kern** — nur bei Restbudget über der Reserve. Andernfalls
in `FINDINGS.md` unter „was ein vollständiges Design zusätzlich bräuchte".

**Kürzungsreihenfolge:** erst Stage B streichen, dann Regularisierung auf `none`.
Präzision und Seed-Zahl bleiben unangetastet.

**Primärmetrik:** Median log10(rel. L2) pro Zelle, Bootstrap-KI 95 %, 10 000
Resamples über die Seeds. Success Rate (< 0.10) nur deskriptiv, es wird nicht darauf
getestet.

---

## 5. Was `lit-agent` geklärt hat — die vier folgenreichen Befunde

**(a) L-BFGS ist zwingend, nicht Geschmackssache.** Der Präzisions-Confound aus
„FP64 is All You Need" (arXiv 2505.10949, §5.2) ist ein **L-BFGS-spezifischer**
Mechanismus: `tolerance_change` = 1e-7 liegt unter dem Maschinenepsilon von FP32
(1.19e-7). Mit Adam existiert dieser Confound in dieser Form nicht. **Ein Adam-Setup
würde H0 nicht testen, sondern an ihr vorbei rechnen.**

**(b) Allen–Cahn und Wave stehen nicht in Krishnapriyan et al.** Volltextsuche nach
„Allen", „Cahn", „wave", „Burgers": null Treffer. KP21 enthält nur Convection,
Reaction-Diffusion und Reaction. Die ursprüngliche Auftragsvorgabe „Parameter exakt
aus Krishnapriyan" ist für Allen–Cahn nicht erfüllbar; kanonische Quelle ist
Expert's Guide §7.1 (wortgleich in FP64 §A.4 und AM26 §B.4, kein Widerspruch).

**(c) β = 50, nicht 30–40.** KP21 tabelliert 20/30/40, aber PINNsformer, FP64 und
AM26 verwenden übereinstimmend β = 50. Nur damit sind die Zahlen an deren Tabellen
anschlussfähig. Failure setzt laut KP21 schon ab β > 10 ein.

**(d) λ_r für Double Backprop ist in AM26 nirgends beziffert** — auch nicht im
Appendix. Muss selbst bestimmt werden (Plan: lokaler Grid, dann eingefroren).

Außerdem: **beide „verdächtigen" Quellen existieren.** „FP64 is All You Need" ist
arXiv 2505.10949 (NeurIPS 2025 bestätigt), und arXiv 2605.30910 (Andersen &
Matsubara, 29.05.2026) ist korrekt, nicht erfunden.

Sieben Widersprüche sind in `docs/baselines.md` als W-1…W-7 dokumentiert und
**nicht** stillschweigend aufgelöst.

---

## 6. Getroffene Festlegungen (alle in `DEVIATIONS.md` als L-1…L-7 begründet)

| Was | Wert | Warum |
|---|---|---|
| Optimierer | L-BFGS, `memory = 100` | sonst ist H0 nicht testbar |
| Abbruchtoleranz | `tolerance_change = tolerance_grad = 0`, feste Iterationszahl | verhindert, dass Präzision mit Trainingsdauer konfundiert |
| Convection-Referenz | `sin(x − βt)`, geschlossen | FFT-Variante akkumuliert Rundungsfehler |
| Architektur | 4 × 128, tanh, Glorot-normal, Inputs auf [−1,1] | AM26 — Quelle des Double-Backprop-Arms |
| Parameterziel | ≈ 50 k, alle Backbones ±10 % | Literatur streut 7.8 k…527 k, keine Vorgabe |
| Kollokationspunkte | Convection 400+200+200, Allen–Cahn 4096+100+100 | AM26-Schema, eine Größenordnung billiger als 101×101 |
| Graphtopologie | k-NN, k = 8, x und t einzeln auf [0,1] normiert, symmetrisiert, Self-Loops, fix | lokaler Operator → lokaler Graph; volle Vernetzung würde „PDE-Struktur hilft" mit „globales Mischen hilft" konfundieren |

---

## 7. Was noch offen ist

1. **Freigabe der Präregistrierung** durch die Projektleitung, dann Freeze-Commit.
   Datum und Commit-SHA in den Kopf von `PREREGISTRATION.md` eintragen.
2. **Kürzungshebel-Frage, gestellt, noch unbeantwortet:** Falls `max_iters` unter die
   5000er-Untergrenze fällt — soll „Kollokationspunkte reduzieren" vor „Stage B
   streichen" auf die Kürzungsliste? Empfehlung war ja, weil der Verlust der
   Doppeldissoziation schwerer wiegt als eine dokumentierte Abweichung von der
   Baseline-Punktzahl. Durch die Wahl des AM26-Schemas (400 statt 10 201 Punkte) ist
   der Druck aber deutlich gesunken.
3. **M2a/M2b-Aufteilung bestätigen lassen** (siehe §8) — sie ist meine Auflösung
   eines Widerspruchs in der Vorgabe, nicht die Vorgabe selbst.
4. **λ_r bestimmen** (lokal, kein Quota) und in `DEVIATIONS.md` L-7 eintragen.
5. **AM26-Punktzahl für Allen–Cahn prüfen:** Tabelle nennt Gesamt 4396, aber
   4096 + 100 + 100 = 4296. Offene Unstimmigkeit aus einer Layout-Rekonstruktion,
   vor dem ersten Kaggle-Run am Original-PDF gegenprüfen.
6. **Layout-Rekonstruktionen allgemein:** mehrere Tabellen in `docs/baselines.md`
   wurden aus `pdftotext -layout` rekonstruiert und sind dort als solche markiert.
   Die kritischen (KP21 Tab. 2, AM26 Tab. 1) sind unabhängig bestätigt, die übrigen
   nur plausibilisiert.

---

## 8. Die Zirkularität in der Kalibrierung — aufgelöst, bitte nicht wieder aufmachen

Die Projektleitung wollte, dass Kalibrierungsläufe als vollwertige Daten zählen
**und** dass die Kalibrierung `max_iters` festlegt. Beides zusammen ist zirkulär: ein
Lauf mit noch unbekanntem `max_iters` kann kein gültiger Datenpunkt sein, weil
`max_iters` über alle Bedingungen identisch sein muss. Auflösung:

- **M2a (~0.3 h):** Timing-Probe, 300 Iterationen, 4 Zellen, **beide**
  Hardware-Optionen, 3 Wiederholungen. Reine Wegwerf-Messung. Wählt Hardware und
  `max_iters`.
- **M2b (~0.2 h):** Seed-0-Slice über alle 12 Stage-A-Zellen mit dem final fixierten
  `max_iters`. Vollwertige Daten. Stage A braucht danach nur noch 48 Läufe.

`max_iters` ist der größte Wert, für den die geschätzte Gesamtzeit von Stage A unter
2.0 h bleibt, abgerundet auf ein Vielfaches von 1000. **Untergrenze 5000** —
literaturgestützt, weil AM26 §5.1 für Convection β = 50 unter FP32 berichtet, dass
das double-PINN „a little over 5000 iterations of L-BFGS optimization" bis zum Erfolg
braucht.

---

## 9. Arbeitsregeln, die die Projektleitung durchgesetzt hat

1. **Kein Kaggle-Run vor grünem Smoke-Test** (200 Iterationen, 100 Punkte, inklusive
   Ergebnis-Persistenz und Resume-Pfad).
2. **Ein einziger Trainer für alle Bedingungen.** Getrennte Trainingsskripte pro
   Architektur sind die häufigste Quelle unfairer Vergleiche in genau dieser
   Literatur.
3. **Jeder abgeschlossene Einzellauf wird sofort persistiert.** Ein Session-Timeout
   darf nie mehr als einen Lauf vernichten. Resume ermittelt beim Start die fehlenden
   (Bedingung, Seed)-Paare. Läufe in aufsteigender Kostenreihenfolge, damit bei
   vorzeitigem Quota-Ende ein auswertbarer Kern vorliegt.
4. **Kein Cherry-Picking.** Alle Läufe landen in der DB, auch abgestürzte. Divergenz
   und NaN sind Messergebnis (Zensur bei rel. L2 = 10.0), kein Ausschlussgrund.
   Ausschluss nur bei Infrastrukturfehlern, dann Wiederholung mit gleichem Seed.
5. **Fairness vor Ergebnis.** Ist ein Vergleich unfair, wird das gesagt, nicht
   berichtet.
6. **Bei Unsicherheit fragen**, statt GPU-Stunden in die falsche Richtung zu schicken.
7. **Commit nach jeder Phase.**

Die Projektleitung hat zweimal eigene frühere Vorgaben korrigiert (Statistik-Umbau
von Success Rate auf Median log-L2; Hardware von 2× T4 zurück auf offen) und erwartet
dieselbe Bereitschaft in der Gegenrichtung: begründeter Widerspruch ist ausdrücklich
erwünscht, stilles Abweichen nicht.

---

## 10. Nächste konkrete Schritte

1. Freigabe der Präregistrierung einholen → Freeze-Commit, Datum und SHA eintragen.
2. **M1:** `src/pdes/` (convection, allen_cahn + Referenzlösungen), `src/models/`
   (mlp, grand, gread), `src/train.py` (EIN Trainer, config-gesteuert),
   `src/metrics.py`, `src/config.py`, Determinismus-Kontrolle (torch, numpy,
   cudnn-deterministic), Tests, `results.sqlite`-Schema mit Resume-Logik.
   Winz-Konfiguration lokal grün: 200 Iterationen, 100 Kollokationspunkte, echte
   Ergebniszeile in der DB.
3. λ_r-Grid lokal bestimmen, Wert einfrieren.
4. **M2:** `bench/calibrate.py` und `bench/plan.py`, Kaggle-Notebook via
   `kaggle/build_notebook.py` (Code als Kaggle-Dataset einbinden, nicht ins Notebook
   pasten), M2a + M2b fahren, `BUDGET.md` fortschreiben.
5. **Stop, Bericht an die Projektleitung, Freigabe für Stage A abwarten.**
