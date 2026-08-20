# Präregistrierung — PDE-strukturierte Attention in PINNs

**Status: ENTWURF v2 (M0), NICHT eingefroren.** Fassung nach der Budget-Neuskalierung
auf 5 GPU-h. Wird nach Freigabe durch die Projektleitung und nach Vorliegen von
`docs/baselines.md` eingefroren. Ab dem Einfrieren gilt: keine Änderung mehr an
diesem Dokument; jede Abweichung wandert nach `DEVIATIONS.md` mit Datum, Begründung
und der Angabe, ob die betroffenen Ergebnisse zum Zeitpunkt der Änderung bereits
gesichtet waren.

Eingefroren am: _(offen)_ · Git-Commit beim Einfrieren: _(offen)_

---

## 1. Forschungsfrage

**Entkommt ein PINN mit PDE-strukturiertem Backbone (GRAND / GREAD) den bekannten
Failure Modes innerhalb eines festen, kleinen Iterationsbudgets besser als ein
Vanilla-MLP-PINN — und überlebt ein etwaiger Vorteil die Kontrolle für numerische
Präzision und Regularisierung?**

Die Formulierung ist bewusst nicht „welcher Backbone erreicht den niedrigsten
Fehler". Das Iterationsbudget ist fix und klein (§4). Für eine Failure-Mode-Studie
ist „entkommt dem Failure Mode unter Budget" die relevantere Frage; für eine
Konvergenzstudie im Grenzwert wäre sie die falsche. Diese Studie ist ersteres.

Motivation: In allgemeinen Sprachmodellen ist ein diffusionsartiger induktiver Bias
unmotiviert. In einem PINN ist die Zielfunktion selbst eine PDE-Lösung — der Bias
passt zur Aufgabe.

## 2. Hypothesen

**H0 (die ernstzunehmende Nullhypothese):** Jeder beobachtete Architekturgewinn
verschwindet, sobald für numerische Präzision (FP32 vs. FP64) und für
Regularisierung (Double Backprop auf den Residuen) kontrolliert wird.

**H1 (gerichtete Doppeldissoziation).** Vorab festgelegt, weil sie den Test schärft:

| PDE | Charakter | Vorhersage |
|---|---|---|
| Convection (großes β) | reiner Transport, keine Diffusion | GRAND **schlechter oder gleich** MLP. Eine reine Diffusionsarchitektur ist hier der falsche Bias. |
| Allen–Cahn | Reaktions-Diffusion | GREAD **besser** als MLP und besser als GRAND. |

Ein Ergebnis, in dem GRAND auf *beiden* PDEs gewinnt, spricht **gegen** die
Mechanismus-Erklärung und für einen unspezifischen Kapazitäts- oder
Optimierungseffekt. Das wird als solches berichtet und nicht als Erfolg verkauft.

**Was H0 stützen würde:** Der Backbone-Effekt ist unter FP64 und/oder unter Double
Backprop nicht mehr von null unterscheidbar oder kehrt das Vorzeichen. Dann lautet
das Ergebnis: der Gewinn war ein Präzisions- bzw. Regularisierungsartefakt.

**Was H0 schwächen würde:** Die Doppeldissoziation aus H1 zeigt sich und bleibt
über beide Präzisionsstufen und beide Regularisierungsstufen gerichtet stabil.

Ein sauberes Null-Resultat ist ein vollwertiges Projektergebnis und wird in
`FINDINGS.md` an erster Stelle berichtet.

## 3. Design — gestuft, nicht vollfaktoriell

Bei 5 GPU-h ist ein vollfaktorielles Design nicht finanzierbar. Gestuft mit Gate
nach jeder Stufe.

### Stage A — Convection im Failure-Regime (60 Läufe)

| Faktor | Stufen | n |
|---|---|---|
| Backbone | mlp, grand, gread | 3 |
| Präzision | fp32, fp64 | 2 |
| Regularisierung | none, double_backprop | 2 |
| Seed | 0, 1, 2, 3, 4 | 5 |

12 Zellen × 5 Seeds = 60 Läufe. Die 12 Seed-0-Läufe entstehen in der Kalibrierung
(§4) und zählen als Daten.

### Stage B — Allen–Cahn (60 Läufe)

Identisches Design, andere PDE. Gate: Stage B startet nur, wenn Stage A im Budget
geblieben ist und der Quota-Guard grünes Licht gibt.

### Warum genau diese zwei PDEs

Inhaltlich, nicht willkürlich: Convection ist Transport ohne Diffusionsterm,
Allen–Cahn ist Reaktions-Diffusion. Die in §2 vorhergesagte Doppeldissoziation ist
der schärfste Test, den das Budget hergibt. Eine Architektur, die auf der Gleichung
gewinnt, zu der sie strukturell passt, und auf der anderen nicht, hat den
Mechanismus gezeigt. Eine, die überall gleich gewinnt, hat nur Kapazität gezeigt.

### PINNsformer

**Nicht im Kern.** PINNsformer ist die Kontrolle für „hilft irgendeine aufwendige
Architektur", nicht für die Kernfrage „hilft eine *passende* Architektur". Er kommt
nur zurück, wenn nach Stage B mehr als die 1.0 h Reserve übrig ist. Andernfalls
erscheint er in `FINDINGS.md` unter „was ein vollständiges Design zusätzlich
bräuchte", mit Stundenschätzung aus der Kalibrierung.

### Kürzungsreihenfolge bei Budgetmangel (vorab, verbindlich)

1. **Stage B streichen** (nur Convection, dafür vollständige Statistik)
2. Regularisierung auf nur `none` reduzieren

**Präzision und Seed-Zahl bleiben unangetastet.** Präzision ist der Kern von H0,
und unter 5 Seeds ist keine Streuungsaussage mehr möglich.

Hinweis zur Umkehrung gegenüber v1: In v1 stand die PDE-Achse ganz oben auf der
Kürzungsliste. Sie steht jetzt unten, weil die PDE-Achse hier den Mechanismus
trägt (Doppeldissoziation) und nicht bloß die Generalisierung.

### Fairness-Auflagen

- **Parameterzahl über alle Backbones auf ±10 % angeglichen**, Referenz ist die
  MLP-Konfiguration aus Krishnapriyan et al. (exakte Zahl aus `docs/baselines.md`).
  Die tatsächliche Parameterzahl jedes Modells wird pro Lauf in die Ergebnis-DB
  geschrieben und in `FINDINGS.md` tabelliert. Gelingt die Angleichung nicht, ohne
  eine Architektur zu verstümmeln, wird der betroffene Vergleich als
  nicht-belastbar gekennzeichnet.
- Ein einziger Trainer für alle Bedingungen. Identisches Optimierer-Schema,
  identische Lernrate, identisches `max_iters`, identische Kollokationspunkte.
- **Kein architekturspezifisches Tuning.** Es gibt kein Tuning-Budget für
  irgendeinen Backbone; alle laufen mit den Baseline-Hyperparametern.

### Graphtopologie für GRAND/GREAD (Forscherfreiheitsgrad — hier festgelegt)

- Kollokationspunkte liegen auf einem festen Raum-Zeit-Gitter (Anordnung und
  Anzahl aus `docs/baselines.md`).
- Koordinaten werden vor der Graphkonstruktion **einzeln auf [0,1] normiert**
  (x und t getrennt), damit die Nachbarschaft nicht von der willkürlichen
  Skalierung des Gebiets abhängt.
- **k-NN mit k = 8** im normierten Raum-Zeit, euklidisch, symmetrisiert
  (Kante, wenn i in kNN(j) **oder** j in kNN(i)), plus Self-Loops.
- Der Graph wird **einmal vorab** konstruiert und bleibt über das gesamte Training
  fix. Attention wird nur auf den vorhandenen Kanten berechnet (sparse), nicht
  vollvernetzt.

Begründung: Der Differentialoperator ist lokal. Ein lokaler Graph ist damit das
ehrliche strukturelle Analogon zur PDE. Volle Vernetzung würde GRAND zu einem
global mischenden Modell machen und den Befund „PDE-Struktur hilft" mit „globales
Mischen hilft" konfundieren — also genau die Verwechslung erzeugen, die diese
Studie vermeiden soll. k = 8 ist die kleinste Wahl, die auf einem 2D-Gitter beide
Achsen plus Diagonalen abdeckt. Nebeneffekt: O(N·k) statt O(N²), was bei diesem
Budget ohnehin nötig ist.

Nicht präregistriert und daher nicht Gegenstand von Behauptungen: die Sensitivität
gegenüber k. Falls Restbudget bleibt, wird k ∈ {4, 16} als *exploratorische*
Nebenanalyse gefahren und in `FINDINGS.md` explizit als exploratorisch markiert.

## 4. Iterationsbudget und Kalibrierung

**Umkehrung gegenüber v1:** Die Kalibrierung legt nicht fest, wie lange die Matrix
dauert. Sie legt fest, **wie viele Iterationen ein Lauf bekommt.**

### Ablauf

**M2a — Timing-Probe (~0.3 h, reine Messung, keine Daten).** Auf **beiden**
Hardware-Optionen (P100 und 2× T4) werden vier repräsentative Zellen
(mlp/gread × fp32/fp64) über je 300 Iterationen mit 3 Wiederholungen vermessen.
Ergebnis: Sekunden pro Iteration pro Zelle pro Hardware. Daraus wird die Hardware
**datenbasiert** gewählt. Erwartung der Projektleitung ist P100 (FP64 mit 1/2 statt
1/32 der FP32-Rate); die Messung entscheidet, nicht die Erwartung.

**Wahl von `max_iters`.** Sei T_i die gemessene Sekunden-pro-Iteration von Zelle i
auf der gewählten Hardware und W die Zahl paralleler Worker. `max_iters` ist der
größte Wert, für den gilt:

```
(Summe über alle 12 Stage-A-Zellen x 5 Seeds von T_i * max_iters) / W  <=  2.0 h
```

Das Ergebnis wird auf das nächstkleinere Vielfache von 1000 abgerundet und **vor**
M2b fixiert. Reicht die Rechnung für weniger als 5000 Iterationen, wird die
Kürzungsliste aus §3 gezogen, statt das Iterationsbudget weiter zu senken — unter
5000 Iterationen ist die Aussage „entkommt dem Failure Mode" nicht mehr sinnvoll
interpretierbar.

**M2b — Seed-0-Slice (~0.2 h, vollwertige Daten).** Auf der gewählten Hardware
laufen alle 12 Stage-A-Zellen mit Seed 0 und dem final fixierten `max_iters`. Diese
12 Läufe landen in `results.sqlite` und zählen in der Auswertung mit. Stage A
braucht danach nur noch die 48 Läufe mit Seeds 1–4.

**Warum diese Zweiteilung.** Kalibrierungsläufe können nur dann als Daten zählen,
wenn sie mit demselben `max_iters` gefahren wurden wie der Rest der Matrix. Ein Lauf
mit abweichendem Iterationsbudget wäre unter den Fairness-Auflagen kein gültiger
Datenpunkt. Die Probe (M2a) ist deshalb strikt Wegwerf-Messung, der Seed-0-Slice
(M2b) strikt Daten. Ohne diese Trennung wäre die Anforderung „Kalibrierung erzeugt
Daten" mit der Anforderung „identisches `max_iters` über alle Bedingungen"
zirkulär.

### `max_iters` ist über alle Backbones identisch

Nicht gleiche Wall-Clock pro Backbone. Dass GRAND/GREAD pro Iteration teurer sind,
ist ein **Befund**, kein zu kompensierender Nachteil, und wird als solcher berichtet:
Wall-Clock pro Lauf und Zahl der Solver-Funktionsauswertungen stehen in der DB und
in `FINDINGS.md` neben der Fehlermetrik. Wer die Architektur einsetzen will, muss
beides sehen.

### Optimierer

Falls die Baseline Adam **und** L-BFGS kombiniert, wird `max_iters` auf die
Adam-Phase bezogen und die L-BFGS-Phase mit fixer, für alle Bedingungen gleicher
Schrittzahl angehängt. Die konkreten Zahlen kommen aus `docs/baselines.md` und
werden vor dem Freeze hier eingetragen. _(offen bis lit-agent)_

## 5. Metriken und Statistik

**Diese Studie ist ein Pilot. Es werden keine konfirmatorischen Behauptungen
aufgestellt.** Berichtet werden Effektstärken mit Unsicherheit und Kostenmessungen.
`FINDINGS.md` sagt das im ersten Absatz.

**Primär:** Median von log10(relativer L2-Fehler) pro Zelle, mit
Bootstrap-Perzentil-KI (95 %, 10 000 Resamples über die 5 Seeds). Kontinuierlich,
deutlich mehr Power pro Seed als ein binärer Ausgang.

Begründung der Umstellung gegenüber v1: Success Rate als Primärmetrik bei n = 5 ist
unterdimensioniert — fünf binäre Ausgänge liefern ein KI, das fast das ganze
Einheitsintervall überdeckt; 3-von-5 ist von 1-von-5 nicht trennbar.

**Behandlung von Ausreißern und Divergenz (vorab):** Der relative L2-Fehler wird vor
der Logarithmierung auf [1e-8, 10.0] geklippt. NaN oder Inf im Endfehler werden als
10.0 gewertet (zensiert nach oben), **nicht** ausgeschlossen. Der Median ist gegen
diese Zensur robust; der Anteil zensierter Läufe wird pro Zelle mitberichtet.

**Sekundär, rein deskriptiv:** Success Rate bei relativem L2 < 0.10. Wird berichtet,
**es wird nicht darauf getestet.**

**Weitere Sekundärmetriken:** Wall-Clock pro Lauf; Iterationen bzw. Wall-Clock bis
zum ersten Unterschreiten von 0.10 (zensiert, falls nie erreicht); Zahl der
Funktionsauswertungen des ODE-Solvers (nur grand/gread); finaler Residual-Loss und
IC/BC-Loss getrennt; Parameterzahl.

**Tests:** Holm–Bonferroni über die Backbone-Paarvergleiche innerhalb einer PDE
bleibt drin, das Ergebnis wird aber explizit als Pilot gerahmt und nicht als
Hypothesenprüfung mit Anspruch berichtet. Der beste Einzellauf wird nirgends als
Ergebnis berichtet.

**Zentrale Auswertung** ist die Interaktion Backbone × Präzision und
Backbone × Regularisierung, plus die Doppeldissoziation Backbone × PDE aus §2 —
nicht der Backbone-Haupteffekt allein.

## 6. Ausschlussregeln (vorab, abschließend)

Ein Lauf wird aus der Auswertung ausgeschlossen **nur** wenn:

1. der Prozess mit einem Infrastrukturfehler abbricht (OOM, Session-Kill,
   CUDA-Fehler). Der Lauf bleibt mit `status='infra_fail'` in der DB und wird mit
   demselben Seed aus der Reserve wiederholt.
2. die Referenzlösung für die Zelle nicht erzeugt werden konnte.

Divergenz, NaN im Loss und hohe Endfehler sind **kein** Ausschlussgrund. Sie sind
das Messergebnis (§5, Zensur bei 10.0). Alle Läufe, auch abgestürzte, landen in
`results/results.sqlite`. Kein Cherry-Picking.

## 7. Budget und Abbruchkriterien

| Posten | Stunden |
|---|---|
| Kalibrierung (M2a Probe + M2b Seed-0-Slice) | 0.5 |
| Stage A (48 Läufe, Seeds 1–4) | 2.0 |
| Stage B (60 Läufe) | 1.5 |
| Reserve (nicht verplanbar) | 1.0 |
| **Gesamt** | **5.0** |

Die Reserve ist für Wiederholungen nach Abstürzen. Sie wird nicht für eine Stage C
verplant. Der Quota-Guard prüft vor jedem Kaggle-Run die kumulierte
Verbrauchsschätzung und bricht ab, statt weiterzurechnen.

**Inhaltliches Abbruchkriterium:** Zeigt Stage A, dass die Backbone-Unterschiede im
Median log10(rel. L2) kleiner sind als die Seed-Streuung innerhalb einer Zelle,
wird Stage B nicht gefahren. Es wird als Null-Resultat berichtet und das
Restbudget in zusätzliche Seeds auf Stage A investiert (Präzision der
Effektschätzung statt Breite).

**Hardware-Konfundierung — ausgeschlossen.** Die lokale GTX 1660 Ti wird für
Korrektheitstests, Debugging und grobe Kostenrangfolge der Backbones genutzt. Sie
wird **unter keinen Umständen** benutzt, um den FP32-Arm zu rechnen, während der
FP64-Arm auf Kaggle läuft. Präzision und Hardware dürfen nicht konfundiert werden —
das wäre exakt der Fehler, den diese Studie anderen nachweist. Alle in die
Auswertung eingehenden Läufe stammen von derselben Kaggle-Hardware.

## 8. Was das Projekt nicht ist

Kein Versuch, klassische Solver (FEM/FV) in Rechenzeit zu schlagen. Keine
Skalierung auf große Modelle oder 3D. Keine Sprachmodellierung. Keine
konfirmatorische Studie — siehe §5.
