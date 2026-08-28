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
dieses β eine Hauptmatrix von höchstens 6.0 h ergibt:

| β | Perioden in t | Domänenpunkte | Gitter | Iterationen | Matrixkosten |
|---|---|---|---|---|---|
| 10 | 1.59 | 169 | 13×13 | 7500 | 5.96 h |
| 12 | 1.91 | 256 | 16×16 | 6000 | 5.83 h |
| 15 | 2.39 | 400 | 20×20 | 4500 | 5.69 h |
| 20 | 3.18 | 676 | 26×26 | 3000 | 5.48 h |
| 25 | 3.98 | 1024 | 32×32 | 2000 | 5.07 h |
| 30 | 4.77 | 1521 | 39×39 | 1500 | 5.32 h |

Alle Punktzahlen erfüllen die Auflösungsforderung von acht Abtastungen pro Periode.
Die Iterationszahl ist an das Budget gekoppelt, nicht frei gewählt; je größer β,
desto teurer ein Schritt und desto weniger Schritte sind finanzierbar.

Der Kandidatenbereich reicht bewusst bis β = 10 hinunter, weil die kleinen β billig
sind und als Rückfallebene dienen, falls die großen bei ihrer Iterationszahl
gesättigt scheitern.

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

**Neue Regel:** Bei gewähltem β und dessen Iterationszahl wird die **MLP-Baseline**
mit λ_r ∈ {1e-5, 1e-4, 1e-3, 1e-2} gefahren (fp32, `double_backprop`, Seed 0).
Gewählt wird der Wert mit dem niedrigsten relativen L2. Dieser Wert gilt danach
**unverändert für alle Backbones, Präzisionen und Seeds**.

Sollten alle vier Werte schlechter abschneiden als die unregularisierte Baseline aus
Phase 1a, wird das so berichtet: Double Backprop hilft in diesem Setup nicht. Der
Arm bleibt dennoch in der Matrix, weil H0 ihn als Kontrollstufe braucht — er wird
dann mit dem besten der vier Werte gefahren.

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
| Bereits verbraucht (V3) | 1.5 |
| Phase 1a: β-Auswahl (6 Baseline-Läufe) | 0.2 |
| Phase 1b: λ_r-Auswahl (4 Baseline-Läufe) | 0.3 |
| Phase 2: Hauptmatrix, 60 Läufe | ~5.1 bis 6.0 |
| Reserve (unantastbar) | 1.0 |
| **Gesamt** | **~8.1 bis 9.0 von 27** |

Zwei Maßnahmen bringen die Matrix von 20 h auf unter 6 h:

**Zwei Worker auf 2× T4.** Kaggle rechnet Session-Wallclock ab, nicht Device-Stunden.
Zwei Prozesse, einer je GPU (`CUDA_VISIBLE_DEVICES=0` / `=1`), halbieren den
Quota-Verbrauch bei gleicher Arbeit. **Dies kehrt `DEVIATIONS.md` D-1 um**, wo aus
Sicherheitsgründen ein Worker festgelegt wurde. Begründung der Umkehr: Die
Persistenzgarantie hat sich beim Abbruch in V3 bewährt — von sechs Läufen ging genau
der eine gerade laufende verloren, fünf waren transaktional gesichert. Mit zwei
Workern wächst das Verlustfenster von einem auf zwei Läufe. Bei Laufzeiten von unter
zehn Minuten pro Lauf und einem Faktor zwei beim Budget ist dieser Tausch vertretbar.
Die Umkehr wird als eigener Eintrag protokolliert.

**`log_every` von 100 auf 500.** Die Zwischenauswertung auf dem 101×101-Gitter kostet
in V3 gemessen rund 20 % der Laufzeit. Fünfmal seltener auszuwerten spart davon vier
Fünftel. Die Endauswertung bleibt unverändert auf 101×101, die Primärmetrik ist davon
nicht berührt — nur die Trainingskurven werden gröber aufgelöst.

Ein zusätzlich geprüfter Hebel wurde **verworfen**: die Zwischenauswertung auf 51×51
zu verkleinern hätte nur 0.05 bis 0.25 h gebracht und einen Eingriff in den Trainer
erfordert. Das Verhältnis von Ersparnis zu Risiko rechtfertigt das nicht.

Ausführung auf Kaggle, 2× T4, Persistenz nach jedem Einzellauf. Die Hauptmatrix passt
mit unter 6 h in eine einzige Session, das 12-Stunden-Limit wird nicht mehr zum
Engpass. Die Resume-Logik bleibt trotzdem aktiv.

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
