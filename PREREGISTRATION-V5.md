# Präregistrierung V5 — Reaction an der Kollapskante

**Status: EINGEFROREN.** Eingefroren am **2026-09-22**; ab jetzt keine
inhaltliche Änderung mehr an diesem Dokument. Jede Abweichung wandert nach
`DEVIATIONS.md` mit Datum, Begründung und der Angabe, ob die betroffenen
Ergebnisse zum Zeitpunkt der Änderung bereits gesichtet waren.

Studien-ID: `reaction_v5_rho525_20260922` · λ_r = 0.0001, ausgewählt nach Abschnitt 5
vor dem Freeze. SHA-256 und Freeze-Commit: siehe `PREREGISTRATION-V5.lock.json`.

V3 (eingefroren, Tag `prereg-v3`) und V4 (Entwurf, nie ausgeführt) bleiben als
Dokumente ihrer jeweiligen Studie bestehen. V4 wurde nicht gefahren, weil ihre
Auswahlregel ein Fehlerband voraussetzte, das bei Convection nicht existiert.
V5 ersetzt sie auf Basis von Messungen, nicht Annahmen.

---

## 1. Der Arbeitspunkt ist gemessen, nicht gewählt

Alle folgenden Werte stammen aus lokalen `NOT_STUDY_DATA`-Läufen der **MLP-Baseline**.
Zu keinem Zeitpunkt wurde ein Graph-Backbone gerechnet; die Wahl des Arbeitspunktes
kann den Architekturvergleich daher nicht begünstigen.

**Gleichung:** Reaction nach Krishnapriyan et al., Appendix A, Gl. 14.

```
u_t - rho * u * (1 - u) = 0,   x in [0, 2pi],  t in [0, 1]
u(x, 0) = exp(-(x - pi)^2 / (2 (pi/4)^2))
u(0, t) = u(2pi, t)
u(x,t) = h(x) e^{rho t} / ( h(x) e^{rho t} + 1 - h(x) )     (analytisch, Gl. 15)
```

Die analytische Lösung erfüllt das Residuum auf **1e-31**. Es gibt keine
interpolierte Referenzdatei und damit keine Fehlerquelle zwischen Zielfunktion und
Messung — anders als bei Allen–Cahn.

**Kollokation:** 1600 Domänenpunkte (40×40), 100 Rand-, 100 IC-Punkte.
Bei 400 Punkten liegen nur 2.5 Stützstellen über der Gaußbreite π/4 und die Aufgabe
ist nicht lösbar (rel. L2 0.981); bei 1600 sind es fünf und sie ist lösbar (0.0575).

**Iterationen:** 2000, L-BFGS, `max_eval=25`, `strong_wolfe`, Toleranzen 0.

**rho = 5.25.** Gemessene Basisrate der MLP-Baseline über fünf Seeds:

| Seed | finaler rel. L2 | Erfolg (< 0.10) |
|---|---|---|
| 0 | 0.996 | nein |
| 1 | 0.999 | nein |
| 2 | 0.109 | nein |
| 3 | 0.083 | ja |
| 4 | 0.070 | ja |

**Erfolgsquote 2/5 = 40 %**, Median log10(rel. L2) = −0.964.

Die Tabelle verwendet den finalen Fehler nach 2000 Iterationen. Die zuvor an dieser
Stelle genannten Werte 0.980 und 0.993 waren die besten protokollierten Zwischenwerte
der Seeds 0 und 1; die Erfolgsquote und der Median bleiben durch die Korrektur
unverändert.

Die Wahl fiel auf rho = 5.25, weil dort beide Ausgänge vorkommen. Bei rho = 5.0
gelingt es (0.058), ab rho = 6 kollabiert es durchgehend (0.989 und schlechter). Eine
Basisrate um 40 % lässt Raum nach oben für eine bessere Architektur und nach unten
für eine schlechtere.

## 2. Forschungsfrage

**Erhöht ein PDE-strukturierter Backbone die Wahrscheinlichkeit, dem Kollaps zu
entkommen — und überlebt ein solcher Vorteil die Kontrolle für numerische Präzision
und Regularisierung?**

## 3. Hypothesen

**H0:** Jeder Architekturvorteil verschwindet unter Kontrolle für Präzision und
Regularisierung.

**H1 — mechanistische Vorhersage:** Reaction hat **keinen Diffusionsterm**.
Deshalb erwarten wir von GRAND keinen spezifischen Strukturvorteil; allgemeines
Graph-Mixing könnte trotzdem helfen. GREAD besitzt zusätzlich einen
Reaktionsterm, der jedoch nicht mit dem logistischen PDE-Term identisch ist.
Ein GREAD-Vorteil wäre ein Hinweis auf den vermuteten Mechanismus, allein aber
noch kein Beweis dafür.

Vorhergesagt wird daher:

| Backbone | Vorhersage |
|---|---|
| mlp | Basisrate 40 % |
| grand | **kein Vorteil** gegenüber mlp |
| gread | **Vorteil** gegenüber mlp und gegenüber grand |

Das ist ein Kontrast **zwischen den beiden Graph-Architekturen**, nicht nur gegen
das MLP. Beide haben nahezu dieselbe Parameterzahl (51 301 gegen 51 305), dieselbe
Graphtopologie und denselben Trainer. Der einzige Unterschied ist der Reaktionsterm.
Wenn GREAD gewinnt und GRAND nicht, ist das schwer als unspezifischer
Kapazitätseffekt zu erklären.

Gewinnen **beide** gleichermaßen, spricht das gegen die Mechanismus-These und für
einen allgemeinen Vorteil von Graph-Mixing. Gewinnt **GRAND** und GREAD nicht, ist
die These widerlegt.

## 4. Bedingungsmatrix

| Faktor | Stufen | n |
|---|---|---|
| Backbone | mlp, grand, gread | 3 |
| Präzision | fp32, fp64 | 2 |
| Regularisierung | none, double_backprop | 2 |
| Seed | 0, 1, 2, 3, 4 | 5 |

12 Zellen × 5 Seeds = **60 Läufe**. Die fünf `mlp/fp32/none`-Läufe liegen bereits
lokal vor; sie werden auf der Zielhardware wiederholt, damit alle ausgewerteten
Läufe von derselben Hardware stammen.

**Implementierungsnachtrag vor Freeze:** Die Graph-Backbones benutzen für V5
`fixed_support`. Alle 1900 Kollokationskoordinaten bilden einen eingefrorenen
Kontextgraphen (symmetrisiertes k-NN, k = 8). Jeder Auswertungspunkt fragt
denselben Kontext über einen festen geometrischen Radius ab; eine glatte,
kompakte Distanzgewichtung macht Vorhersage und automatische Ableitungen
konsistent. Diese Definition ersetzt den bisherigen, vom aktuellen Anfragebatch
abhängigen Kontext. Ältere Graph-Ergebnisse sind daher nicht direkt mit V5
vergleichbar. Die MLP-Baseline bleibt unverändert.

## 5. Wahl von λ_r

MLP-Baseline, fp32, `double_backprop`, λ_r ∈ {1e-5, 1e-4, 1e-3, 1e-2}, **Seeds 3
und 4**; der Kandidat mit dem niedrigsten Median des relativen L2 über diese beiden
Seeds gilt danach unverändert für alle Bedingungen. Kein Graph-Backbone, kein
architekturspezifisches Tuning.

**Warum nicht Seed 0.** Die erste Fassung dieses Abschnitts wählte auf Seed 0. Die
vier Läufe wurden am 2026-09-22 auf 2× T4 ausgeführt und kollabierten alle
(relativer L2 0.9910 bis 0.9993). Das ist die Eigenschaft des Seeds, nicht der
Regularisierung: Seed 0 gehört zu den drei von fünf Seeds, auf denen bereits die
unregularisierte Baseline scheitert (Abschnitt 1). Eine Auswahl unter vier
kollabierten Läufen entscheidet auf Rauschen. Die Seeds 3 und 4 sind genau die, auf
denen die Baseline gelingt; dort kann die Auswahl zeigen, ob ein λ_r den Erfolg
erhält oder zerstört. Die Änderung erfolgte vor dem Freeze und ausschließlich auf
MLP-Daten; kein Graph-Backbone wurde zu diesem Zeitpunkt gerechnet. Dokumentiert als
D-13. Die verworfenen Seed-0-Läufe bleiben als `NOT_STUDY_DATA` in der Datenbank und
werden in FINDINGS.md berichtet.

Begründung der Kandidatenliste: λ_r = 1.0 trieb Double Backprop bei Convection
nachweislich in die Trivialfalle (Anfangsterm stagniert bei 0.43, Optimierer friert
nach 300 Iterationen ein); bei λ_r = 1e-4 fiel der Anfangsterm auf 1.03e-3.

## 6. Metriken

**Ko-primär, beide werden berichtet:**

1. **Erfolgsquote** über die fünf Seeds, Erfolg bei relativem L2 < **0.10**.
   Wilson-95-%-Intervall.
2. **Median log10(relativer L2)** pro Zelle, Bootstrap-Perzentil-KI (95 %,
   10 000 Resamples).

Warum beide: Der Ausgang ist überwiegend bimodal — die Läufe landen entweder unter
0.11 oder über 0.97. Für einen solchen Ausgang ist die Erfolgsquote die natürliche
Größe. Aber die Zwischenwerte häufen sich ausgerechnet an der Schwelle (0.070,
0.083, 0.109, 0.117), sodass die Klassifikation an der dritten Nachkommastelle
hängt. Der Median nutzt die Information vollständig und ist gegen die Schwellenwahl
unempfindlich. Widersprechen sich die beiden Größen, wird der Widerspruch berichtet
und nicht aufgelöst.

**Die Schwelle 0.10 ist hiermit fixiert** und wird nach Sichtung der Daten nicht
mehr verändert.

**Sekundär:** Wall-Clock, L-BFGS-Iterationen und Funktionsauswertungen, Solver-NFE
bei den Graph-Backbones, Loss-Zerlegung, Parameterzahl.

**Zensur:** relativer L2 auf [1e-8, 10.0] geklippt; NaN und Inf zählen als 10.0 und
werden nicht ausgeschlossen.

## 7. Was diese Studie nachweisen kann — und was nicht

Bei fünf Seeds reicht das Wilson-Intervall einer Basisrate von 40 % von rund 12 %
bis 77 %. Die Matrix ist deshalb ein Pilot für große Effekte. Selbst 2/5 gegen
5/5 Erfolge ergeben im zweiseitigen exakten Fisher-Test p ≈ 0.167; ein
signifikanter Vorteil ist damit nicht zugesichert. Ein moderater Effekt von
40 % auf 60 % kann mit diesem Design nicht zuverlässig erkannt werden.

Das ist eine Eigenschaft des Budgets, keine der Fragestellung, und es steht hier,
damit es später nicht als nachträgliche Einschränkung erscheint.

Weitere Grenzen: eine Gleichung, ein Netzformat, eine Graphtopologie (k = 8), ein
festes Iterationsbudget, kein PINNsformer, keine Aussage über Convection oder
Allen–Cahn, wo alle Architekturen gesättigt scheitern.

## 8. Ausschlussregeln

Ausschluss nur bei Infrastrukturfehler (OOM, Session-Kill, CUDA-Fehler); der Lauf
bleibt mit `status='infra_fail'` in der DB und wird mit identischem Seed wiederholt.
Divergenz und Kollaps sind Messergebnisse, kein Ausschlussgrund.

## 9. Budget

| Posten | Stunden |
|---|---|
| Bereits verbraucht (V3) | 1.5 |
| λ_r-Auswahl (4 + 8 Baseline-Läufe) | 0.4 |
| Hauptmatrix, 60 Läufe | 10.0 |
| Reserve (unantastbar) | 1.0 |
| **Gesamt** | **12.9 von 27** |

Ausführung auf Kaggle, 2× T4, zwei Worker, `log_every = 500`, Persistenz nach jedem
Einzellauf.

Die Matrixzahl ist am 2026-09-22 auf der Zielhardware für `fixed_support` gemessen
worden und ersetzt die frühere Schätzung von 7.4 h, die aus der alten
Graph-Implementierung stammte. Gemessen wurden 300 Iterationen je Zelle:
GRAND/fp32 0.546 s und GREAD/fp64 mit Double Backprop 0.862 s je Iteration, bei
rund 2.1 Funktionsauswertungen je Iteration (`results/v5_gpu_profile_t4_300iters.json`).
Hochgerechnet auf 2000 Iterationen ergibt das rund 16 GPU-Worker-Stunden, bei zwei
Workern also etwa 8 h Session-Wallclock; 10.0 h sind die dafür freigegebene
Obergrenze. Die Hochrechnung setzt voraus, dass die Funktionsauswertungen je
Iteration nicht stark steigen. Beim MLP stiegen sie mit wachsendem λ_r von 2.1 auf
7.0; ob sich das auf die Graph-Backbones überträgt, ist nicht gemessen. Der
Quota-Guard bricht daher vor dem Erreichen der 10.0 h ab, statt weiterzurechnen.

## 10. Abbruchkriterien

- Weicht die Basisrate von `mlp/fp32/none` auf der Zielhardware um mehr als zwei von
  fünf Seeds von den lokal gemessenen 2/5 ab, wird angehalten und die Ursache
  geklärt, bevor die Matrix weiterläuft.
- Quota-Guard bricht ab, sobald die kumulierte Schätzung das Stufenbudget erreicht.
