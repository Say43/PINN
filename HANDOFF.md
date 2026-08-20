# Naechster Schritt (Stand 2026-08-20, 22:45 — alles gestoppt)

Nichts laeuft. Kein Kaggle-Kernel aktiv, keine lokalen Prozesse, working tree sauber.
Verbraucht: 1.48 von 27 verfuegbaren GPU-Stunden.

**Wo es weitergeht:** Der Machbarkeitsanker wurde vor dem Ende abgebrochen und muss
neu gestartet werden. Er ist die Vorbedingung fuer jeden weiteren Kaggle-Lauf.

    python -m bench.feasibility_anchor --domain-points 1024 --max-iters 20000         --precision fp32 --regularization double_backprop --device cpu         --out results/anchor_dbp.json

    python -m bench.feasibility_anchor --domain-points 1024 --max-iters 20000         --precision fp32 --regularization none --device cpu         --out results/anchor_none.json

Laufzeit lokal auf CPU: etwa 40 bzw. 20 Minuten, kein GPU-Quota. Die Anker sind
KEINE Studiendaten und schreiben nicht in results.sqlite.

**Woran die Entscheidung haengt:** Erreicht der Double-Backprop-Anker den relativen
L2 von 0.10 bei hoechstens 4000 Iterationen, passt die vollstaendige Option B
(3 Backbones x 2 Praezisionen x 2 Regularisierungen x 5 Seeds, 60 Laeufe) mit
22.5 h in die 27 h. Braucht er mehr, siehe Tabelle in `bench/replan.py` bzw.
`results/replan_27h.json`.

**Danach:** lambda_r bei korrekter Aufloesung neu bestimmen (lokal, der Wert 1.0
stammt von einem 10x10-Gitter), dann kurze Neukalibrierung auf Kaggle mit 1224
Punkten (~0.2 h), dann Matrix zur Freigabe vorlegen.

**Was heute passiert ist:** M2b vollstaendig erhoben und als Evidenz verworfen —
das Kollokationsgitter lag unter dem Nyquist-Limit. Details in FINDINGS.md und
DEVIATIONS.md D-7. Die Aufloesungspruefung ist jetzt eine harte Vorbedingung im
Code; sie lehnt auch die praeregistrierten 400 Punkte ab.

---

# Sofort-Uebergabe an Claude Code waehrend M2b (2026-08-20, ca. 20:43 CEST)

Dieser Abschnitt ist der aktuelle operative Stand und ersetzt auch den direkt
darunter stehenden Nachtrag nach M2a.

## Was gerade laeuft

- Privater Kaggle-Kernel: `says43/pinn-pde-attention-m2b`
  (`https://www.kaggle.com/code/says43/pinn-pde-attention-m2b`).
- Letzter kontrollierter Status: `RUNNING`, kein Fehlerstatus.
- M2b fuehrt genau sechs Convection-/Seed-0-Laeufe aus: MLP, GRAND und GREAD,
  jeweils FP32/FP64, `regularization=none`, reduziertes 396-Punkte-Schema,
  6000 L-BFGS-Iterationen, ein Worker auf 2x T4.
- Zwei von sechs Bedingungen sind bereits transaktional im privaten Dataset
  `says43/pinn-pde-attention-results` gesichert und in SQLite `completed`:
  MLP/FP32 und MLP/FP64. Die vier Graphbedingungen waren beim Handoff noch offen.
- Ergebniskennzahlen wurden absichtlich noch nicht gelesen. Bisher wurde nur
  Status/Vollstaendigkeit geprueft, damit keine partielle Ergebniskenntnis weitere
  Entscheidungen beeinflusst.
- Den laufenden Kernel nicht duplizieren oder neu starten. Er publiziert nach jedem
  terminalen Lauf. Bei Fehler erst Logs holen; ein Neustart setzt aus SQLite fort
  und ueberspringt terminale Versuche.
- **Nach M2b anhalten. Stage A nicht automatisch starten.**

Statuskontrolle:

```powershell
& '.\.venv\Scripts\kaggle.exe' kernels status says43/pinn-pde-attention-m2b
& '.\.venv\Scripts\kaggle.exe' datasets files says43/pinn-pde-attention-results
```

## Wichtige Korrektur zur Aussagekraft

Die budgetbedingt gekuerzte Matrix ist **nicht ausreichend, um das urspruengliche
Forschungsziel vollstaendig zu beantworten**. Das muss im weiteren Bericht klar
und an prominenter Stelle stehen:

- Der laufende M2b-Slice mit nur Seed 0 ist ein technischer Pilot und erlaubt
  keinerlei robuste Inferenz.
- Selbst die danach geplante gekuerzte Stage A (30 Laeufe, fuenf Seeds) untersucht
  nur Convection. Sie kann einen Architekturvergleich bedingt auf
  `regularization=none` und stratifiziert nach Praezision liefern.
- Weil Double Backprop gestrichen wurde, kann sie die urspruengliche H0
  (Architekturgewinn verschwindet nach Kontrolle fuer Praezision **und**
  Regularisierung) nicht vollstaendig testen.
- Weil Allen-Cahn/Stage B gestrichen wurde, kann sie die praeregistrierte
  Doppeldissoziation GRAND-Convection versus GREAD-Allen-Cahn nicht testen.
- PINNsFormer bleibt ebenfalls ausserhalb der Kernmatrix. Ein Ergebnis der
  gekuerzten Stage A darf daher weder als allgemeiner Attention-Vergleich noch als
  Bestaetigung/Widerlegung des urspruenglichen Gesamtmechanismus formuliert werden.

Die Kuerzung war kein stiller Scope-Wechsel, sondern die vorab festgelegte
outcome-blinde Budgethierarchie bei harter 5-h-Grenze. Sie erhaelt einen engeren,
deskriptiven Convection-Piloten, aber nicht den gesamten urspruenglichen Claim.

## Empfohlene Fortsetzung nach Abschluss von M2b

1. Alle sechs M2b-Zellen aus dem privaten Dataset laden; SQLite-Integritaet,
   Konfigurationen, Status und Quota pruefen. Erst bei 6/6 terminalen Zellen die
   Kennzahlen gemeinsam auswerten; Fehler/Timeouts zaehlen als Ergebnisse.
2. `BUDGET.md`, `FINDINGS.md` und diesen Handoff mit Ist-Quota, Nullbefund zuerst,
   Kennzahlen und klarer Reichweitenbegrenzung aktualisieren. Praeregistrierung
   nicht aendern.
3. Vor weiteren 24 Stage-A-Laeufen neu entscheiden: Unter unveraenderter 5-h-Grenze
   kann nur der engere Convection-Pilot abgeschlossen werden. Fuer das
   urspruengliche Ziel muss zuerst eine outcome-unabhaengige Zusatzbudgetrechnung
   fuer mindestens Double Backprop auf Convection und den Allen-Cahn-Kontrast
   erstellt und eine hoehere GPU-Quote freigegeben werden. Nicht versuchen, diese
   beiden Ziele sprachlich gleichzusetzen.
4. Die 1.0-h-Reserve bleibt bis zu dieser Entscheidung unantastbar und dient nur
   technischen Wiederholungen. Keine lokale GPU fuer Studienarme verwenden.

Letzter sauberer Commit vor M2b: `5c4494e`. Finaler Plan-Hash:
`8a9111bc3c83eb547c020b456881d412ee16e116ef402734dd68e07aa2407ce4`.
Vor M2b waren dem Projekt 0.57 von 5.0 Kaggle-GPU-Stunden zugerechnet. M2b wird
gegen das Stage-A-Budget gebucht; nach Abschluss mit der Kaggle-Kontoseite bzw.
CLI abgleichen.

---

# Uebergabe-Nachtrag nach M2a (2026-08-20)

Dieser Abschnitt ersetzt fuer den aktuellen Ausfuehrungsstand alle darunter
stehenden historischen Statusangaben.

- M2a und outcome-freies Zusatzprofiling sind abgeschlossen. Noch keine
  Studienergebnisse; M2b wurde nicht gestartet.
- Kaggle privat: `says43/pinn-pde-attention-code`,
  `says43/pinn-pde-attention-results`.
- Dem Projekt zugerechnete Kaggle-Abrechnung: 0.57 h von 5.0 h.
- Finaler Plan: 2x T4, ein Worker, reduziertes 396-Punkte-Schema, Stage B gestrichen,
  nur `regularization=none`, 6000 Iterationen, 6 Zellen x 5 Seeds.
- M2b umfasst damit 6 Seed-0-Laeufe; danach **vor Stage A anhalten und berichten**.
- P100 ist nur mit offiziellem PyTorch 2.5.1+cu121 (`sm_60`) lauffaehig, war aber
  langsamer als T4. Details und outcome-freie Entscheidungen: `DEVIATIONS.md` D-2
  bis D-6.

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
  Integritätsbackup und Resume. Commit: `0195568`.
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
