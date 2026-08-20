# Findings

**Diese Studie ist ein Pilot ohne konfirmatorischen Anspruch.** Berichtet werden
Effektstärken mit Unsicherheit und Kostenmessungen; Null-Resultate stehen zuerst.

**Stand 2026-08-20: Es gibt keine verwertbare Aussage über Architekturen.** Der
M2b-Slice ist vollständig erhoben, aber durch einen Setup-Fehler entwertet. Die
Ursache ist identifiziert und unten dokumentiert.

---

## 1. Das zentrale Ergebnis ist ein Setup-Befund, kein Architekturbefund

Alle sechs Seed-0-Zellen liegen im relativen L2-Fehler zwischen **1.21 und 2.31**.
Ein Netz, das überall null vorhersagt, erreicht 1.0. Alle sechs Modelle sind also
**schlechter als der triviale Nullprädiktor** — und zwar bei einem Residual-Loss von
`1e-6` bis `4e-4`.

| Zelle | rel. L2 | loss_total | Wall-Clock |
|---|---|---|---|
| mlp / fp32 | 1.772 | 1e-6 | 325.9 s |
| mlp / fp64 | 1.576 | 1e-6 | 253.5 s |
| grand / fp32 | 1.367 | 3.59e-4 | 595.8 s |
| grand / fp64 | 1.206 | 3.67e-4 | 530.0 s |
| gread / fp32 | 1.227 | 4.27e-4 | 969.9 s |
| gread / fp64 | 2.314 | 3.00e-4 | 589.7 s |

Alle mit 6000 L-BFGS-Iterationen, Convection β = 50, `regularization=none`,
reduziertes 396-Punkte-Schema, 2× T4. Keine Zensur, keine Fehlerabbrüche, alle
Integritätsprüfungen grün.

Die Kombination „Residuum erfüllt, Lösung falsch" ist die Signatur von
**Aliasing**, nicht von Optimierungsversagen.

## 2. Die Ursache: das Kollokationsgitter liegt unter dem Nyquist-Limit

Die Lösung ist `sin(x − βt)` mit β = 50. Ihre Periode in der Zeit ist
2π/50 = 0.1257, das Zeitintervall [0,1] enthält also **7.96 Perioden**.

Das reduzierte Schema legt 196 Domänenpunkte an, und `_rectangular_grid`
(`src/pdes/convection.py:26`) macht daraus ein **14×14**-Gitter — also 14
Stützstellen für knapp acht Schwingungen.

| Schema | Punkte | Gitter | dt | Abtastungen pro Periode | |
|---|---|---|---|---|---|
| **reduziert (verwendet)** | 196 | 14×14 | 0.0718 | **1.75** | **unter Nyquist** |
| AM26 Convection | 400 | 20×20 | 0.0501 | 2.51 | grenzwertig |
| PINNsformer / FP64 | 10 201 | 101×101 | 0.0099 | 12.69 | ausreichend |

Unterhalb von zwei Abtastungen pro Periode ist die Zielfunktion auf dem
Kollokationsgitter **nicht rekonstruierbar**. Das Residuum lässt sich dort von
unendlich vielen Funktionen erfüllen, die zwischen den Stützstellen beliebig falsch
sind. Genau das zeigen die Zahlen: winziger Loss, Fehler über 1.

**Damit misst der M2b-Slice keine Architektureigenschaft.** Er misst, wie
verschiedene Backbones ein unterbestimmtes Problem unterschiedlich falsch lösen.
Ein Vergleich zwischen ihnen wäre bedeutungslos, und weder Architektur- noch
Präzisionsaussagen dürfen aus diesen Daten abgeleitet werden.

## 3. Wie der Fehler entstanden ist

Die Kürzungshierarchie wurde outcome-blind angewandt, wie vorgesehen. Der zuerst
gezogene Hebel war die Reduktion der Kollokationspunkte von 400 auf 196
(`DEVIATIONS.md` D-6). Dieser Hebel wurde vom Lead-Agent als bevorzugt empfohlen,
weil er den Allen–Cahn-Kontrast erhält, statt Stage B zu streichen.

**Bei dieser Empfehlung wurde die Abtastbedingung nicht geprüft.** 400 Punkte sind
bei β = 50 bereits grenzwertig (2.51 Abtastungen pro Periode); 196 Punkte fallen
darunter. Die Reduktion war damit kein neutraler Budgethebel, sondern eine
Änderung, die das Problem unlösbar macht. Weder die Präregistrierung noch der
Ausführungsvertrag enthielten eine Prüfung der Auflösung gegen den PDE-Parameter —
das ist die eigentliche Lücke.

Der Fehler wurde durch das vorab definierte Kriterium **PC-2** gefangen, bevor
weitere 24 Läufe oder Zusatzbudget verbraucht wurden.

## 4. Gate-Ergebnis

| Prüfung | Regel | Messung | Ergebnis |
|---|---|---|---|
| Vollständigkeit | 6 terminale Zellen | 6 | bestanden |
| PC-1 Failure Mode | mlp/fp32/none > 0.5 | 1.772 | bestanden |
| PC-2 Auflösungsvermögen | mindestens eine Zelle < 0.10 | bester Wert 1.206 | **gescheitert** |
| PC-3 Integrität | keine Fehler/Zensur | keine | bestanden |

Regelfolge laut `docs/decision-gate-m2b.md`: **kein Zusatzbudget**, Grenzen
berichten. Diese Regel wird eingehalten. Ein Ausbau auf Option B oder C wäre auf
Basis eines nachweislich unterbestimmten Setups nicht zu rechtfertigen.

Anmerkung zur Prüfung selbst: Die erste Ausführung von `analysis/gate_check.py` las
die Spalte `relative_l2_censored`, die ein Zensur-Flag ist und kein Wert. Das
Skript meldete dadurch `0` für beide Messgrößen. Korrigiert auf `relative_l2`; die
oben stehenden Werte stammen aus der korrigierten Fassung. Artefakt:
`results/gate_m2b.json`.

## 5. Was gültig bleibt

Die Kostenmessungen sind vom Aliasing nicht betroffen und bleiben verwertbar:

- **FP64 ist in dieser Konfiguration nicht teurer als FP32, sondern billiger.**
  MLP 253.5 s gegen 325.9 s, GRAND 530.0 s gegen 595.8 s, GREAD 589.7 s gegen
  969.9 s. Ursache sind die Funktionsauswertungen der L-BFGS-Liniensuche: 16 720
  gegen 27 569 beim MLP. Bei dieser Problemgröße ist die Last
  Kernel-Launch-gebunden, nicht rechengebunden; der erwartete FP64-Nachteil der
  Turing-Architektur tritt nicht auf. Die ursprüngliche Hardwaredebatte
  P100-gegen-T4 war für dieses Problemformat gegenstandslos.
- **Die Projektion aus M2a unterschätzt die reale Laufzeit um Faktor 1.26.** Die
  Kalibrierung misst nur die Optimizer-Schleife, nicht die fixen Auswertungen.
- **Graph-Backbones kosten das Zwei- bis Dreifache des MLP** bei angeglichener
  Parameterzahl (51 301 bzw. 51 305 gegen 50 049, innerhalb der ±10-%-Auflage).

## 6. Was ein tragfähiges Design bräuchte

1. **Auflösungsprüfung als harte Vorbedingung.** Mindestens vier Abtastungen pro
   Periode der analytischen Lösung, geprüft vor jedem Lauf, mit Abbruch statt
   Warnung. Für β = 50 heißt das mindestens 32 Zeitstützstellen, also ein Gitter ab
   etwa 32×32 = 1024 Domänenpunkten.
2. **Neubewertung des Budgets mit dieser Untergrenze.** Die Zusatzbudgetrechnung in
   `BUDGET.md` geht von 196 Punkten aus und ist damit hinfällig; die realistischen
   Kosten liegen um ein Vielfaches höher.
3. Erst danach ist die Frage nach Double Backprop und Allen–Cahn sinnvoll zu
   stellen.

## 7. Offene technische Lücken

- `git_commit` wird in der `runs`-Tabelle als `unknown` gespeichert. Die Provenienz
  einzelner Läufe ist damit nicht an den Code gebunden.
- `study_meta` enthält nur `schema_version`. Die tatsächlich sichtbare Hardware
  wird nicht persistiert, obwohl Invariante 1 des Ausführungsvertrags sie verlangt.
  Sie ließ sich nur indirekt über `device: cuda:0` und die Kernel-Metadaten
  bestätigen.
- Der beim Abbruch verwaiste `running`-Attempt ist nicht erhalten geblieben: der
  Resume stellt die Datenbank aus der publizierten Kopie wieder her, in der er nie
  enthalten war. Der Ausführungsvertrag sagt zu, solche Attempts als `interrupted`
  zu behalten.

## 8. Verbrauch

Dem Projekt zugerechnet: **rund 1.48 GPU-Stunden von 5.0**. Davon 0.542 h
Kalibrierung und 0.940 h für den M2b-Slice inklusive des nach dem Abbruch
nachgeholten sechsten Laufs. Es wurde kein Budget für Läufe verbraucht, die auf dem
fehlerhaften Setup aufgebaut hätten.
