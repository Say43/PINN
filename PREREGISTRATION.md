# Präregistrierung — PDE-strukturierte Attention in PINNs

**Status: ENTWURF v3 (M0), NICHT eingefroren.** Fassung nach der Budget-Neuskalierung
auf 5 GPU-h und nach Einarbeitung von `docs/baselines.md` (lit-agent, 2026-08-20).
`docs/baselines.md` liegt vor. Fehlt zum Einfrieren nur noch die Freigabe der
Projektleitung. Ab dem Einfrieren gilt: keine Änderung mehr an
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

### PDE-Definitionen (belegt in `docs/baselines.md`)

**Convection** — KP21 §3.1 Gl. 5, Failure-Punkt nach PINNsformer Gl. 8 / FP64 §A.1 /
AM26 §B.1:

```
u_t + β u_x = 0,   x ∈ [0, 2π],  t ∈ [0, 1],   β = 50
IC: u(x,0) = sin(x)        BC: periodisch
Referenz: u(x,t) = sin(x − βt)      (geschlossene Form)
```

β = 50 statt der in v2 genannten 30–40: KP21 sweept bis β = 70 und tabelliert 20/30/40,
aber PINNsformer, FP64 und AM26 verwenden übereinstimmend β = 50. Nur mit β = 50 sind
unsere Zahlen an deren Tabellen anschlussfähig. Der Failure setzt laut KP21 bereits
ab β > 10 ein, β = 50 liegt also klar im Regime.

**Referenzlösung geschlossen, nicht spektral.** KP21 Gl. 6 gibt die Lösung als
FFT-Rücktransformation an, FP64 Gl. 6 und AM26 Gl. 9 geschlossen als sin(x − βt).
Mathematisch äquivalent, numerisch nicht: die FFT-Variante akkumuliert Rundungsfehler.
Für eine Studie, deren Nullhypothese die numerische Präzision ist, wäre eine
rundungsfehlerbehaftete Referenz ein Eigentor. Daher geschlossene Form.

**Allen–Cahn** — Expert's Guide §7.1 Gl. 7.1–7.4, wortgleich in FP64 §A.4 und
AM26 §B.4 (kein Widerspruch zwischen den drei Quellen):

```
u_t − 0.0001 u_xx + 5u³ − 5u = 0,   x ∈ [−1, 1],  t ∈ [0, 1]
IC: u(x,0) = x² cos(πx)     BC: periodisch in u und u_x
Referenz: numerisch (spektral), kein geschlossener Ausdruck
```

Allen–Cahn steht **nicht** in Krishnapriyan et al. — Volltextsuche nach „Allen",
„Cahn", „wave", „Burgers" ergab dort null Treffer. Die Auflage „Parameter exakt aus
Krishnapriyan" gilt daher nur für Convection; für Allen–Cahn ist Expert's Guide die
kanonische Quelle. Das ist ein bewusster Quellenmix und in `DEVIATIONS.md` vermerkt.

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

- **Parameterzahl über alle Backbones auf ±10 % angeglichen**, Zielgröße
  **≈ 50 000 Parameter** (4 Hidden Layers × 128, tanh, Glorot-normal — die
  Konfiguration von AM26 §3/§A.1). Die Literatur gibt hier keine Vorgabe her: die
  Baselines reichen von 7,8 k (KP21, 4×50) bis 527 k (PINNsformer, 4×512), also über
  eine Größenordnung. Gewählt wurde AM26, weil von dort der Double-Backprop-Arm
  stammt und die Confound-Kontrolle der Kern der Studie ist. Die tatsächliche
  Parameterzahl jedes Modells wird pro Lauf in die Ergebnis-DB geschrieben und in
  `FINDINGS.md` tabelliert. Gelingt die Angleichung nicht, ohne eine Architektur zu
  verstümmeln, wird der betroffene Vergleich als nicht-belastbar gekennzeichnet.
- **Aktivierung tanh** für alle Backbones. PINNsformer Tab. 4 verwendet für sein
  Baseline-PINN ReLU; ReLU hat verschwindende zweite Ableitung und ist für Residuen
  mit u_xx untauglich. Drei von vier Quellen verwenden tanh.
- **Eingänge auf [−1, 1] normalisiert** (AM26 §3), einheitlich über alle Bedingungen.
- Ein einziger Trainer für alle Bedingungen. Identisches Optimierer-Schema,
  identisches Iterationsbudget, identische Kollokationspunkte.
- **Kein architekturspezifisches Tuning.** Es gibt kein Tuning-Budget für
  irgendeinen Backbone; alle laufen mit den Baseline-Hyperparametern.

### Kollokationspunkte (Forscherfreiheitsgrad — hier festgelegt)

Es gibt keine kanonische Punktzahl: KP21 sampelt zufällig und nennt 1 000 (Fußnote 4)
bzw. 10 000 (Tab. E.1), PINNsformer und FP64 verwenden ein 101×101-Gitter (10 201),
AM26 nur 400 Domänenpunkte bei Convection. AM26 zeigt zudem, dass unter
Regularisierung **weniger** Punkte besser sind.

Gewählt wird das **AM26-Schema**, Gitter, für beide PDEs:

| PDE | Domäne | Rand (je Seite) | IC | Gesamt |
|---|---|---|---|---|
| Convection | 400 | 200 | 200 | 800 |
| Allen–Cahn | 4 096 | 100 | 100 | 4 296 |

Begründung: es ist das Schema der Quelle, aus der der Double-Backprop-Arm stammt, und
es ist um eine Größenordnung billiger als 101×101 — bei 5 GPU-h der Unterschied
zwischen „Studie" und „keine Studie". Die AM26-Tabelle nennt für Allen–Cahn eine
Gesamtzahl von 4 396; 4 096 + 100 + 100 ergibt 4 296. Die Differenz von 100 ist eine
offene Unstimmigkeit der Layout-Rekonstruktion und wird vor dem ersten Kaggle-Run am
Original-PDF geprüft (`docs/baselines.md`, Abschnitt „Layout-Rekonstruktionen").

Identische Punkte für alle Bedingungen innerhalb einer PDE.

### Graphtopologie für GRAND/GREAD (Forscherfreiheitsgrad — hier festgelegt)

- Der Graph wird über die Kollokationspunkte oben gebildet.
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
Kürzungsliste aus §3 gezogen, statt das Iterationsbudget weiter zu senken.

Die 5000er-Untergrenze ist jetzt literaturgestützt und nicht mehr nur gesetzt: AM26
§5.1 berichtet für Convection β = 50 unter FP32, dass das double-PINN „a little over
5000 iterations of L-BFGS optimization" bis zum Erfolg braucht. Unterhalb dieser
Marke misst man systematisch den Zustand *vor* der Konvergenz und kann über
„entkommt dem Failure Mode" nichts aussagen.

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

### Optimierer: L-BFGS, und warum das nicht verhandelbar ist

**L-BFGS, `memory = 100`, kein Adam.** Die gesamte Failure-Mode-Linie (KP21,
PINNsformer, FP64, AM26) verwendet durchgehend L-BFGS; KP21 begründet das explizit
in Fußnote 3. Expert's Guide verwendet ausschließlich Adam über 200 k–300 k Schritte.

Entscheidend: **der Präzisions-Confound aus FP64 ist ein L-BFGS-spezifischer
Mechanismus.** Er entsteht daraus, dass das Abbruchkriterium `tolerance_change` in
der Größenordnung des Maschinenepsilons von FP32 liegt (1e-7 gegen ε_fp32 = 1.19e-7,
FP64 §5.2). Mit Adam existiert dieser Confound in dieser Form nicht. **Ein
Adam-Setup würde H0 nicht testen, sondern an ihr vorbei rechnen.** Damit ist die
Optimiererwahl keine Geschmacksfrage: L-BFGS ist Voraussetzung dafür, dass die
Studie ihre eigene Nullhypothese überhaupt prüfen kann.

Konsequenz für Allen–Cahn: dessen Parameter stammen aus dem Adam-Regime (Expert's
Guide), werden aber unter L-BFGS gefahren. AM26 §B.4 tut genau das und berichtet
dafür Zahlen, das Vorgehen ist also literaturgedeckt — die Kombination
Allen–Cahn-Parameter × L-BFGS ist trotzdem in `DEVIATIONS.md` vermerkt.

### Abbruchkriterium: fix und präzisionsunabhängig

`tolerance_change` und `tolerance_grad` werden auf **0** gesetzt (bzw. den kleinsten
darstellbaren Wert), sodass **kein Lauf toleranzbedingt vorzeitig stoppt**. Jeder
Lauf erhält exakt `max_iters` L-BFGS-Iterationen. Beide Toleranzen sowie die
tatsächliche Iterations- und Funktionsauswertungszahl werden pro Lauf protokolliert.

Begründung: Die Literatur bietet zwei Varianten — FP64 verwendet eine **feste**
Toleranz (1e-7) für beide Präzisionen, AM26 eine **präzisionsabhängige**
(√ε, also 3.45e-4 bei FP32 gegen 1.49e-8 bei FP64). Beide erzeugen unterschiedlich
lange Trainingsläufe je nach Präzision und konfundieren damit Präzision mit
Trainingsdauer. Da die Fairness-Auflage ein identisches Iterationsbudget über alle
Bedingungen verlangt, wird die Toleranz hier ganz deaktiviert. Der gemessene
Präzisionseffekt ist dann ein reiner Arithmetikeffekt und kein Artefakt der
Stopp-Heuristik.

**Das ist eine bewusste Abweichung vom Mechanismus, den FP64 beschreibt.** Wir
reproduzieren den dort dokumentierten Stopp-Confound nicht, sondern schalten ihn
aus, um den verbleibenden Präzisionseffekt isoliert zu messen. Sollte sich unter
dieser Bedingung **kein** Präzisionseffekt zeigen, ist das ein Befund über die
Reichweite der FP64-These und wird als solcher berichtet, nicht als Widerlegung.

### λ_r für Double Backprop

Die Definition steht in AM26 §3.2 Gl. 5 (Penalty auf den **Residuen**: ∇_{x,t}F auf
der Domäne, ∇_t B am Rand, ∇_x I auf der IC, Vorfaktor λ_r/2, Basisgewichte
λ_F = λ_B = λ_I = 1). **Der Wert von λ_r ist im Paper nirgends angegeben**, auch
nicht im Appendix.

Festlegung: λ_r wird durch einen kleinen Grid λ_r ∈ {1e-4, 1e-3, 1e-2, 1e-1, 1}
**auf der lokalen GPU** (Convection, FP32, reduzierte Punktzahl) bestimmt, auf den
besten Wert eingefroren und danach **für alle Bedingungen identisch** verwendet.
Kein Kaggle-Quota, kein bedingungsspezifisches Tuning. Der gewählte Wert und der
Grid werden in `DEVIATIONS.md` festgehalten.

Kostenhinweis: Double Backprop verdoppelt die Gradientenrechnung (AM26 §4.3). Das
ist in der Kalibrierung bereits enthalten, weil `double_backprop` eine eigene Zelle
ist.

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
Funktionsauswertungen des ODE-Solvers (nur grand/gread); Zahl der L-BFGS-Iterationen
und Funktionsauswertungen; finaler Residual-Loss und IC/BC-Loss getrennt;
Parameterzahl.

**Einschränkung beim Loss:** Zwischen `none` und `double_backprop` ist der Loss
**nicht vergleichbar**, weil der Penalty-Term mit eingeht (AM26 Tab. 1, Fußnote 3).
Der Loss wird daher nur innerhalb einer Regularisierungsstufe verglichen. Über
Stufen hinweg gilt allein der relative L2-Fehler gegen die Referenzlösung.

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
