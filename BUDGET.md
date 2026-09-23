# Budget

## Aktueller Rahmen fuer V5 (Stand 2026-09-22, gemessen)

| Posten | Stunden |
|---|---|
| Verfuegbare Gesamtquote | **27.0** |
| Bereits verbraucht (V3) | rund **1.48** |
| V5-Profil, zwei Kostenproben | **0.18** |
| V5-Lambda-Auswahl, 4 + 8 Laeufe | **0.4** (Limit 1.5) |
| V5-Hauptmatrix, 60 Laeufe | **10.0** (Obergrenze, Projektleitung 2026-09-22) |
| Reserve | **1.0** |

Der V5-Runner rechnet Kaggle-Session-Wallclock, nicht die Summe beider gleichzeitig
laufenden T4-Prozesse. Lambda-Auswahl und Hauptmatrix haben getrennte Limits. Die
Hauptmatrix darf erst nach dem Freeze der V5-Praeregistrierung starten.

### Iststand nach Abschluss von V5 (2026-09-23)

| Posten | Soll | Ist |
|---|---|---|
| V5-Profile und Notebook-Overhead | — | rund 0.19 |
| V5-Lambda-Auswahl, 4 + 8 Laeufe | 0.4 | 0.26 |
| V5-Hauptmatrix, 60 Laeufe | 10.0 (Obergrenze) | **8.29** |
| **V5 gesamt, Kaggle-Abrechnung** | — | **8.74** |
| Projekt gesamt | 27.0 verfuegbar | **10.2** |

Grundlage: Kaggle-Wochenquota 5.40 h vor dem ersten V5-Kernel, 14.14 h nach dem
Matrix-Kernel (2026-09-23), und der Quota-Zustand des Runners
(`actual_matrix_hours = 8.288`). Die gemessene Hochrechnung von rund 8 h hat
gehalten; der Guard musste nicht eingreifen. Die Reserve von 1.0 h ist unberuehrt.

### Gemessene Kosten auf 2x T4 (2026-09-22)

Zwei Proben, beide `NOT_STUDY_DATA`: 25 Iterationen je vier Zellen
(`results/v5_gpu_profile_t4_25iters.json`) und 300 Iterationen je zwei Zellen
(`results/v5_gpu_profile_t4_300iters.json`).

| Zelle | s/Iteration | Auswertungen/Iteration | 2000 Iterationen |
|---|---|---|---|
| mlp / fp32 / double_backprop, lambda 1e-5 | 0.055 | 2.15 | 110 s (gemessen) |
| grand / fp32 / none | 0.546 | 2.07 | rund 1090 s |
| gread / fp32 / none | 0.600 | 2.08 | rund 1200 s |
| gread / fp64 / double_backprop, lambda 1e-5 | 0.862 | 2.16 | rund 1725 s |

Hochrechnung der 60 Laeufe: rund 16 GPU-Worker-Stunden, bei zwei Workern also
etwa 8 h Session-Wallclock. Die frueheren 7.4 h stammten aus der alten
Graph-Implementierung und sind damit ersetzt.

**Unsicherheit, die im Guard bleibt:** Die Zahl der Funktionsauswertungen je
Iteration haengt am Verlauf der Linie-Suche. Beim MLP stieg sie mit wachsendem
lambda_r von 2.15 auf 7.04, die Laufzeit entsprechend von 110 s auf 242 s. Ob sich
das auf die Graph-Backbones uebertraegt, ist nicht gemessen. Der Quota-Guard bricht
deshalb vor dem naechsten Batch ab, sobald die Schaetzung die 10.0 h erreicht,
statt weiterzurechnen. Die Reserve bleibt fuer den Guard gesperrt.

**Wochenquota Kaggle:** 30 h je Woche, Reset 2026-09-26. Am 2026-09-22 vor dem
V5-Start waren 5.4 h verbraucht; Profil und erste Lambda-Auswahl haben rund 0.3 h
hinzugefuegt.

Die folgenden Tabellen dokumentieren den historischen V3-Rahmen und dessen
tatsaechlichen Verbrauch.

## Historischer V3-Rahmen (Stand 2026-08-20, nach Neuskalierung)

| Posten | Stunden |
|---|---|
| GPU-Quota gesamt | **5.0** |
| Kalibrierung (M2a Probe + M2b Seed-0-Slice) | 0.5 |
| Stage A — Convection, 48 Laeufe (Seeds 1-4) | 2.0 |
| Stage B — Allen-Cahn, 60 Laeufe | 1.5 |
| Reserve (nicht verplanbar) | 1.0 |
| Bereits verbraucht (Kaggle-Abrechnung, seit Projektstart) | 1.21 |

Die Reserve ist ausschliesslich fuer Wiederholungen nach Abstuerzen. Sie wird nicht
fuer eine Stage C verplant.

Session-Limit: 12 h pro Notebook-Run. Netzwerk in Kaggle-Notebooks standardmaessig aus.
`/kaggle/working` geht beim Session-Ende verloren, wenn nicht als Dataset committed.

## Hardware — offen, wird gemessen

Frueher auf 2x T4 festgelegt, zurueckgenommen. Entscheidung faellt datenbasiert in M2a.

Die Matrix ist zur Haelfte FP64, und FP64-Durchsatz trennt die Optionen um mehr als
eine Groessenordnung:

| | FP32 | FP64 | Rate |
|---|---|---|---|
| P100 (Pascal GP100) | ~9.3 TFLOPS | ~4.7 TFLOPS | 1/2 |
| T4 (Turing) | ~8.1 TFLOPS | ~0.25 TFLOPS | 1/32 |

Auf 2x T4 gewinnt man Faktor 2 durch Parallelitaet und verliert auf der Haelfte aller
Laeufe bis Faktor 32. Erwartung ist daher P100. Gemessen wird trotzdem: M2a faehrt
beide Optionen mit je einer FP32- und einer FP64-Zelle.

Nach dem Praeregistrierungs-Freeze wurde der urspruengliche 2-Worker-Plan verworfen:
Auch bei 2x T4 laeuft genau **ein** Worker. Nur so kann ein Session-Kill hoechstens
den einen aktuell laufenden Einzellauf vernichten. Die zweite T4 bleibt ungenutzt;
dieser Sicherheitsnachteil geht ehrlich in die Hardwarewahl von M2a ein. Siehe
`DEVIATIONS.md`, D-1.

Der Smoke-Test laeuft lokal auf CPU und belastet das GPU-Quota nicht. Die lokale
1660 Ti darf nur fuer kurze Korrektheitstests bzw. die outcome-blinde lambda_r-
Skalierung mit harter Laufzeitgrenze unter drei Minuten verwendet werden; sie
rechnet keinen Studienarm.

## Quota-Guard

Vor jedem Kaggle-Run wird die kumulierte Verbrauchsschaetzung geprueft. Ueberschreitet
sie das Stufenbudget, bricht der Guard ab, statt weiterzurechnen. Die Reserve ist fuer
den Guard nicht verfuegbar.

## Ist/Soll-Bilanz

Wird nach jedem Kaggle-Run fortgeschrieben.

## M2-Iststand und finales Ausfuehrungsbudget (2026-08-20)

- Kaggle-Account vor Projektstart: 0.38 h; nach M2/Profiling: 0.95 h.
  Damit sind konservativ **0.57 h** dem Projekt zugerechnet.
- Interne Zuordnung: 0.542154 h Kalibrierung/Notebook-Overhead und 0.027846 h
  outcome-freies Stage-A-Profiling. Die externe Abrechnung ist fuer das
  Gesamtbudget massgeblich.
- Die verbindliche Kuerzungshierarchie hat Stage B gestrichen und Stage A deren
  1.5 h zugeschlagen. Double Backprop wurde danach ebenfalls gestrichen.
- Verbleibendes Stage-A-Budget nach Profiling: **3.472154 h**.
- Finaler Plan: 6 Zellen x 5 Seeds, 6000 Iterationen, 396 Loss-Punkte, ein T4-Worker.
  Projektion fuer M2b plus restliche Stage A: **3.222531 h**.
- Projektion gesamt: 0.57 + 3.222531 = **3.792531 h**. Damit bleiben 1.207469 h,
  davon 1.0 h unantastbare Reserve und rund 0.207 h operative Marge.

| Datum | Run | Stufe | Laeufe | Geschaetzt (h) | Ist (h) | Kumuliert (h) | Rest (h) |
|---|---|---|---|---|---|---|---|
| 2026-08-20 | M2a + Profiling | Kalibrierung | — | 0.5 | 0.570 | 0.570 | 4.430 |
| 2026-08-20 | M2b (abgebrochen) | Stage A | 5 von 6 | 0.645 | 0.637 | 1.207 | 3.793 |

## M2b-Iststand nach Kernel-Abbruch (2026-08-20, 21:30)

Der Kernel `says43/pinn-pde-attention-m2b` endete mit
`KernelWorkerStatus.CANCEL_ACKNOWLEDGED`, also durch Abbruch, nicht durch regulaeren
Abschluss. Kernel-Log ist 0 Byte, die Ursache ist daraus **nicht** feststellbar.

Zustand in `results.sqlite` (nur Status und Kosten gelesen, keine Fehlerkennzahlen):

| Zelle | Status | Wall-Clock | Iterationen | Funktionsauswertungen |
|---|---|---|---|---|
| mlp / fp32 | completed | 325.9 s | 6000 | 27 569 |
| mlp / fp64 | completed | 253.5 s | 6000 | 16 720 |
| grand / fp32 | completed | 595.8 s | 6000 | 14 948 |
| grand / fp64 | completed | 530.0 s | 6000 | 12 605 |
| gread / fp64 | completed | 589.7 s | 6000 | 12 675 |
| gread / fp32 | **running** (abgeschnitten) | — | — | — |

Fuenf von sechs Zellen sind terminal und transaktional gesichert. Die
Persistenzgarantie hat gehalten: der Abbruch hat genau einen Einzellauf gekostet,
wie im Ausfuehrungsvertrag vorgesehen. Der verwaiste `running`-Attempt wird beim
Resume als `interrupted` behalten und als Infrastrukturfehler mit identischem Seed
neu versucht.

### Gemessener Mehraufwand gegenueber der Projektion

Die M2a-Zeiten erfassen nur die Optimizer-Schleife (D-3), nicht die fixen
101x101-Auswertungen. Der Unterschied ist jetzt gemessen:

| Zelle | projiziert | ist | Faktor |
|---|---|---|---|
| mlp / fp32 | 215.2 s | 325.9 s | 1.51 |
| mlp / fp64 | 220.4 s | 253.5 s | 1.15 |
| grand / fp32 | 475.5 s | 595.8 s | 1.25 |
| grand / fp64 | 466.8 s | 530.0 s | 1.14 |
| gread / fp64 | 466.8 s | 589.7 s | 1.26 |

**Mittlerer Faktor 1.263, Median 1.253.**

Konsequenz: Die Stage-A-Projektion steigt von 3.223 h auf **4.071 h** und
**ueberschreitet damit das Stage-A-Limit von 3.472 h**. Unter der selbstgesetzten
5-h-Grenze wuerde der Quota-Guard vor dem Ende von Stage A ausloesen.

Anmerkung zu D-4: Der dort als nicht praeregistriert entfernte Sicherheitsfaktor
1.25 entspricht fast exakt dem jetzt gemessenen 1.263. Die Entfernung war formal
richtig — der Faktor war nicht praeregistriert —, aber er hat empirisch genau das
abgedeckt, was nun fehlt. Kuenftige Projektionen verwenden den **gemessenen**
Faktor, nicht einen gesetzten.

## Zusatzbudgetrechnung fuer die Wiederherstellung des Scopes (2026-08-20)

Erstellt **vor** jeder Sichtung von M2b-Kennzahlen. Grundlage sind ausschliesslich
die M2a-Zeitmessungen (T4, reduziert, ein Worker) aus `results/m2_plan.json`. Keine
Fehler-, Loss- oder Erfolgswerte sind eingeflossen.

Basis: 6 Zellen `none`, Summe 0.386704 s/Iteration, 6000 Iterationen, 5 Seeds.

| Baustein | Stunden | Grundlage |
|---|---|---|
| Stage A Convection `none`, 5 Seeds | 3.223 | gemessen |
| — davon M2b (Seed 0) | 0.645 | laeuft |
| — davon Seeds 1–4 | 2.578 | offen |
| Double Backprop Convection, 5 Seeds | 6.445 | gemessen x Planerfaktor 2.0 |
| Stage B Allen-Cahn `none`, 5 Seeds | 17.293 | **extrapoliert** |
| Stage B Allen-Cahn Double Backprop | 34.585 | **extrapoliert** |

Die Allen-Cahn-Zahlen sind **nicht gemessen**. Sie skalieren die Convection-Zeiten
linear mit der Punktzahl (2125 statt 396, Faktor 5.37). Linear ist fuer k-NN-Graphen
mit O(N·k) und fuer das MLP plausibel, aber unbestaetigt. Vor einer Freigabe von
Stage B muesste eine eigene M2a-Probe auf Allen-Cahn laufen.

### Gesamtbedarf je Ausbaustufe (zzgl. 1.0 h Reserve)

| Option | Umfang | Gesamt-Quota |
|---|---|---|
| **A** | schmaler Convection-Pilot wie geplant | **3.79 h** |
| **B** | A + Double Backprop auf Convection | **10.24 h** |
| **C** | B + Allen-Cahn `none` | **27.53 h** |
| **D** | vollstaendiges praeregistriertes Design | **62.12 h** |

**Das urspruengliche Forschungsziel kostet rund das Zwoelffache des bewilligten
Budgets.** Die 5-h-Grenze konnte es nie tragen; das ist jetzt beziffert statt
vermutet.

Guenstigster wissenschaftlich sinnvoller Ausbau ist **B**: Double Backprop stellt
die Regularisierungshaelfte von H0 innerhalb derselben PDE wieder her, und die
Confound-Kontrolle war von Anfang an als der eigentliche Beitrag benannt. Der
Allen-Cahn-Kontrast ist eine eigene Groessenordnung und eine eigene Entscheidung.

Nebenhebel, falls Allen-Cahn dennoch gewuenscht ist: dessen Kosten haengen fast
ausschliesslich an der Punktzahl. Bei 396 Punkten wie Convection kostete Stage B
`none` ebenfalls nur 3.22 h. Ob Allen-Cahn mit 0.0001 Diffusion und scharfen
Interfaces auf 396 Punkten ueberhaupt sinnvoll aufgeloest ist, ist damit aber
offen und waere vorab zu pruefen, nicht hinterher zu behaupten.

### Dokumentationsnotiz

`HANDOFF.md` nennt fuer das reduzierte Allen-Cahn-Schema 2175 Punkte;
`kaggle/runner.py` setzt 2025 + 50 + 50 = **2125**. Der Code ist massgeblich.
