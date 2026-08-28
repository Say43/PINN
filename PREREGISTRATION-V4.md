# Präregistrierung V4 — PDE-strukturierte Attention im messbaren β-Bereich

**Status: ENTWURF, NICHT eingefroren.** Neue Studie, nicht Fortsetzung von V3.

`PREREGISTRATION.md` (V3, eingefroren, Tag `prereg-v3`) bleibt unverändert gültig als
Dokument der ersten Studie. Deren Ergebnis steht in `FINDINGS.md`: bei β = 50 löst
keine Architektur die Gleichung, es gibt nichts zu vergleichen. V4 ist die Konsequenz
daraus und wird als eigenständige Studie geführt, damit die Grenze zwischen
präregistriert und nachträglich nicht verwischt.

---

## 1. Was aus V3 gelernt wurde und hier eingeht

Alles Folgende ist **belegt**, nicht vermutet, und stammt aus lokalen Diagnoseläufen,
die ausdrücklich als `NOT_STUDY_DATA` geführt werden:

1. **Kapazität ist nicht der Engpass.** Dasselbe 4×128-tanh-MLP erreicht beim
   direkt überwachten Fit der analytischen β = 50-Lösung einen relativen L2 von
   **0.0113**. Das Netz kann die Lösung darstellen.
2. **Der Physics-Loss ist bei β = 50 unauflösbar zerrissen.** Die Gradientnormen von
   Domänen- und Anfangsterm stehen bei 63.87 zu 2.15, ihr Kosinus bei **−0.993**.
   Beide Terme verlangen nahezu entgegengesetzte Updates.
3. **Bei kleinem β funktioniert es.** β = 1 erreicht 0.0088, β = 10 erreicht 0.0027.
   Der Übergang ins Versagen liegt zwischen β = 25 und β = 30.
4. **Bei β = 50 ist alles gesättigt.** MLP, GRAND und GREAD landen bei 0.66 bis 0.74,
   ununterscheidbar. Weder FP64 noch Double Backprop ändern daran etwas.
5. **Die Auflösungsanforderung ist empirisch bestimmt.** Vier Abtastungen pro Periode
   reichen nicht (Loss 1e-5 bei Fehler 1.41). Acht reichen (Fehler sinkt monoton mit
   dem Loss). Der Wert **8** ist im Code als harte Vorbedingung verankert.
6. **λ_r = 1.0 zerstört Double Backprop.** Der Anfangsterm stagniert bei 0.43, der
   Optimierer friert nach 300 Iterationen ein. Bei λ_r = 1e-4 fällt der Anfangsterm
   auf 1.03e-3. Die alte Auswahlregel („Penalty auf 10 % des Anfangs-Loss") ist
   damit widerlegt.

## 2. Forschungsfrage

**Verschafft ein PDE-strukturierter Backbone (GRAND / GREAD) einem PINN einen
Vorteil in dem β-Bereich, in dem die Aufgabe überhaupt lösbar ist — und überlebt
dieser Vorteil die Kontrolle für numerische Präzision und Regularisierung?**

Der Unterschied zu V3 ist die Wahl des Arbeitspunktes. V3 hat bei β = 50 gemessen,
wo alle Architekturen gesättigt scheitern und ein Vergleich bedeutungslos ist. V4
misst dort, wo die Baseline weder trivial gewinnt noch gesättigt verliert.

## 3. Hypothesen

**H0 (unverändert aus V3):** Jeder beobachtete Architekturgewinn verschwindet unter
Kontrolle für Präzision (FP32/FP64) und Regularisierung (none / Double Backprop).

**H0-Präzision, bereits vorläufig gestützt:** Nach Abschalten der Abbruchtoleranz
liegen FP32 und FP64 bei β = 50 gleichauf (0.69 gegen 0.66, n = 1). Der in der
FP64-Arbeit berichtete Präzisionsvorteil ist demnach ein Artefakt der
Stopp-Heuristik. V4 prüft das mit n = 5 im messbaren Bereich.

**H1:** GRAND ist auf Convection — reiner Transport ohne Diffusionsterm — **nicht
besser** als das MLP. Ein Gewinn von GRAND ausgerechnet hier spräche gegen die
Mechanismus-Erklärung und für einen unspezifischen Kapazitätseffekt.

## 4. Wahl des Arbeitspunktes β — Regel vor der Messung

β wird **nicht gewählt, sondern gemessen**. Phase 1 fährt ausschließlich die
MLP-Baseline (fp32, `none`, Seed 0) bei je der größten Iterationszahl, die für
dieses β ins Budget passt:

| β | Domänenpunkte | Iterationen | Matrixkosten |
|---|---|---|---|
| 15 | 400 | 6000 | 20.6 h |
| 20 | 676 | 4000 | 20.1 h |
| 25 | 1024 | 3000 | 21.0 h |
| 30 | 1521 | 2000 | 19.7 h |

**Auswahlregel, vorab und abschließend:** Gewählt wird das **größte** β, dessen
MLP-Baseline einen relativen L2 im Band **[0.15, 0.75]** erreicht.

Begründung der Bandgrenzen. Unter 0.15 hat die Baseline die Aufgabe im Wesentlichen
gelöst; es bleibt kein Raum, in dem eine bessere Architektur sich zeigen könnte
(Bodeneffekt). Über 0.75 ist das Versagen gesättigt, wie bei β = 50 belegt
(Deckeneffekt). Das größte β wird gewählt, weil die Aufgabe dort am schwersten ist
und ein Architekturvorteil am meisten Spielraum hat, sichtbar zu werden.

Liegt **kein** β im Band, wird die Studie **nicht** gefahren und als Null-Resultat
berichtet. Es wird nicht nach einem passenden β weitergesucht.

Die Entscheidung stützt sich ausschließlich auf die **MLP-Baseline**. Zu keinem
Zeitpunkt der β-Wahl wird ein Graph-Backbone gerechnet. Damit kann die Wahl den
Architekturvergleich nicht begünstigen.

## 5. Wahl von λ_r — Regel vor der Messung

Der Wert ist in AM26 nicht angegeben und muss selbst bestimmt werden. Die alte
Heuristik ist widerlegt (§1.6).

**Neue Regel:** Bei gewähltem β wird die **MLP-Baseline** mit
λ_r ∈ {1e-5, 1e-4, 1e-3, 1e-2} gefahren (fp32, Seed 0). Gewählt wird der Wert mit dem
niedrigsten relativen L2. Dieser Wert gilt danach **unverändert für alle Backbones,
Präzisionen und Seeds**.

Wieder gilt: nur die Baseline wird verwendet, kein Graph-Backbone. Es gibt kein
architekturspezifisches Tuning.

## 6. Bedingungsmatrix

| Faktor | Stufen | n |
|---|---|---|
| Backbone | mlp, grand, gread | 3 |
| Präzision | fp32, fp64 | 2 |
| Regularisierung | none, double_backprop | 2 |
| Seed | 0, 1, 2, 3, 4 | 5 |

12 Zellen × 5 Seeds = **60 Läufe**, alle auf Convection bei dem in Phase 1
bestimmten β, mit der zugehörigen Punktzahl und Iterationszahl aus der Tabelle in §4.

Parameterzahl über alle Backbones auf ±10 % angeglichen, Referenz 50 049 (MLP 4×128).
Gemessen in V3: GRAND 51 301, GREAD 51 305 — Auflage erfüllt.

## 7. Metriken

**Primär:** Median von log10(relativer L2) pro Zelle, Bootstrap-Perzentil-KI
(95 %, 10 000 Resamples über die 5 Seeds).

**Zentrale Auswertung:** die Interaktionen Backbone × Präzision und
Backbone × Regularisierung. Ein Backbone-Vorteil, der nur in einer Confound-Stufe
auftritt, zählt als widerlegt.

**Sekundär, deskriptiv:** Anteil der Läufe unter 0.10; Wall-Clock; L-BFGS-Iterationen
und Funktionsauswertungen; Solver-NFE bei den Graph-Backbones; Loss-Zerlegung.

**Der Loss ist zwischen Regularisierungsstufen nicht vergleichbar** (AM26 Tab. 1
Fußnote 3). Über Stufen hinweg gilt allein der relative L2.

**Zensur:** relativer L2 wird auf [1e-8, 10.0] geklippt; NaN und Inf zählen als 10.0
und werden **nicht** ausgeschlossen. Anteil zensierter Läufe wird berichtet.

## 8. Ausschlussregeln

Ausschluss nur bei Infrastrukturfehler (OOM, Session-Kill, CUDA-Fehler) — Lauf bleibt
mit `status='infra_fail'` in der DB und wird mit identischem Seed wiederholt.
Divergenz und hohe Fehler sind Messergebnisse, kein Ausschlussgrund.

## 9. Budget

| Posten | Stunden |
|---|---|
| Phase 1: β-Auswahl (4 Baseline-Läufe) | 0.6 |
| Phase 1: λ_r-Auswahl (4 Baseline-Läufe) | 0.9 |
| Phase 2: Hauptmatrix, 60 Läufe | ~20.1 |
| Reserve (unantastbar) | 2.0 |
| Bereits verbraucht (V3) | 1.5 |
| **Gesamt** | **~25.1 von 27** |

Ausführung auf Kaggle, eine GPU, ein Worker, Persistenz nach jedem Einzellauf.
Das 12-Stunden-Session-Limit erzwingt mindestens zwei Sessions für Phase 2; die
Resume-Logik ist dafür gebaut und hat sich in V3 beim Abbruch bewährt.

**Anmerkung zur Ausführungsplattform:** Für Modelle dieser Größe ist die lokale CPU
gemessen schneller als die Kaggle-T4 (1024 Punkte, 6000 Iterationen: 635 s lokal
gegen hochgerechnet rund 1000 s auf T4), weil die Modelle zu klein sind, um eine GPU
auszulasten. Die Ausführung auf Kaggle ist eine bewusste Vorgabe der Projektleitung.

## 10. Abbruchkriterien

- Kein β im Band [0.15, 0.75] → Studie wird nicht gefahren, Null-Resultat.
- Quota-Guard bricht ab, sobald die kumulierte Schätzung das Stufenbudget erreicht.
- Zeigt die Hauptmatrix Backbone-Unterschiede kleiner als die Seed-Streuung innerhalb
  einer Zelle, wird das als Null-Resultat berichtet.

## 11. Was diese Studie nicht kann

Keine Doppeldissoziation — Allen–Cahn ist nicht enthalten, damit bleibt die positive
Hälfte der Mechanismus-These ungeprüft. Kein PINNsformer-Vergleich. Keine Aussage
über β = 50, wo alle Architekturen gesättigt scheitern. Kein Vergleich gegen
klassische Solver. Die Ergebnisse gelten für eine Gleichung, ein Netzformat, eine
Graphtopologie (k = 8) und ein festes Iterationsbudget.
