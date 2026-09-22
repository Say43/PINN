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
- **Commit:** `0195568` (`Complete M1 pipeline and prepare safe Kaggle M2`).

## D-2 — Pascal-kompatibles PyTorch fuer die P100-Probe
- **Datum:** 2026-08-20
- **Vorher:** Kaggle-Standardimage mit PyTorch 2.10.0+cu128.
- **Nachher:** Fuer P100 wird vor M2a das offizielle Wheel PyTorch 2.5.1+cu121
  installiert und in einem separaten Prozess geprueft.
- **Begruendung:** Das Standardimage enthaelt keine `sm_60`-Kernels. Der private,
  datenfreie Test bestaetigte P100, `sm_60` und eine echte CUDA-Matrixmultiplikation.
- **Ergebnisse gesichtet:** keine Studienergebnisse; nur Kompatibilitaets- und Zeiten.
- **Artefakt:** `p100_compatibility.json` im privaten Results-Dataset.

## D-3 — M2a-v2 trennt Trainingszeit von fixer Endauswertung
- **Datum:** 2026-08-20
- **Vorher:** Aeussere `train_once`-Zeit geteilt durch 300; darin lagen zwei
  Zwischen- und eine Endauswertung.
- **Nachher:** 3x3-Auswertungsgitter in M2a, Zeit aus dem internen Optimizer-Loop;
  aeussere Sessionzeit bleibt die Quota-Groesse. Neue Dateien tragen Schema 2.
- **Begruendung:** Fixe 101x101-Auswertungskosten duerfen nicht proportional auf
  5000+ Iterationen extrapoliert werden. M2a ist Wegwerf-Timing; M2b war noch nicht
  gestartet.
- **Wiederholungen:** Der Quota-Guard stoppte T4/full nach 11/12 Zeilen. Nur
  GREAD/FP64 hat zwei statt drei Wiederholungen (0.1053 und 0.1093 s/Iteration).
  Der Planer verlangt deshalb mindestens zwei Wiederholungen und kennzeichnet dies.
- **Ergebnisse gesichtet:** ausschliesslich Laufzeiten, keine Studienfehler.

## D-4 — Nicht praeregistrierten Faktor 1.25 aus der Iterationsformel entfernt
- **Datum:** 2026-08-20
- **Vorher:** `bench/plan.py` multiplizierte die eingefrorene Budgetformel mit 1.25.
- **Nachher:** Faktor 1.0, exakt wie in `PREREGISTRATION.md` Abschnitt 4 festgelegt.
- **Begruendung:** Der zusaetzliche Faktor war weder praeregistriert noch in
  `HANDOFF.md`, `BUDGET.md` oder dem Ausfuehrungsvertrag festgelegt. Die Reserve
  bleibt separat unantastbar.
- **Ergebnisse gesichtet:** nur M2a-Zeiten, keine Studienergebnisse.

## D-5 — Semantisch identische Graphprojektionen fusioniert
- **Datum:** 2026-08-20
- **Aenderung:** Identische `query/key/value`-Projektionen eines Forward-Passes
  werden einmal berechnet; Q/K/V werden mit unveraenderten Parametern als blockweise
  lineare Operation ausgefuehrt und danach geteilt.
- **Validierung:** FP64-Ausgaben, Koordinaten- und Parametergradienten stimmen mit
  der vorherigen Formel bis 1e-12 ueberein; alle Trainerfamilien-Tests sind gruen.
  CPU-Mikrobenchmark: etwa 1.20x.
- **Profiling:** Vier outcome-freie T4-Zellen (GREAD x FP32/FP64 x zwei Repeats)
  wurden transparent Stage A belastet, nicht der ausgeschoepften Kalibrierung.
- **Ergebnisse gesichtet:** keine Studienergebnisse.

## D-6 — Finale outcome-freie M2-Kuerzung
- **Datum:** 2026-08-20
- **Angewandte Reihenfolge:** 396 Punkte; danach Stage B streichen und 1.5 h Stage A
  zuschlagen; danach Regularisierung auf `none` reduzieren.
- **Final:** T4, ein Worker, 6000 Iterationen, 6 Zellen x 5 Seeds. Praezision und
  Seedzahl bleiben unangetastet.
- **Begruendung:** Exakte Anwendung der eingefrorenen Kuerzungshierarchie auf die
  Timingdaten; keine Fehler-, Loss- oder Erfolgswerte wurden verwendet.

## D-7 — M2b-Slice als Evidenz verworfen: Kollokationsgitter unter Nyquist
- **Datum:** 2026-08-20
- **Abschnitt:** Ausfuehrung, nicht Praeregistrierung
- **Befund:** Das reduzierte Schema mit 196 Domaenenpunkten ergibt ein 14x14-Gitter.
  Die Loesung sin(x - 50t) hat 7.96 Perioden in t; 14 Stuetzstellen sind 1.75
  Abtastungen pro Periode und damit unter dem Nyquist-Limit.
- **Folge:** Alle sechs Zellen erreichen Residual-Losses von 1e-6 bis 4e-4 bei
  relativen L2-Fehlern von 1.21 bis 2.31, also schlechter als der Nullpraediktor.
  Signatur von Aliasing. Der Slice wird nicht als Architekturevidenz gewertet.
- **Ergebnisse gesichtet:** ja. Der Befund wurde erst nach dem Gate-Lauf sichtbar.
  Die Verwerfung folgt aber der vorab in docs/decision-gate-m2b.md fixierten
  Regel PC-2, nicht einer nachtraeglichen Begruendung.
- **Verantwortlich:** Der Hebel "Punktzahl zuerst reduzieren" wurde vom Lead-Agent
  empfohlen, ohne die Abtastbedingung gegen beta zu pruefen.
- **Konsequenz:** Auflaesungspruefung als harte Vorbedingung vor jedem Lauf,
  Mindestens vier Abtastungen pro Periode. Details in FINDINGS.md Abschnitt 6.

## D-8 — Umkehr von D-1: zwei Worker auf 2x T4
- **Datum:** 2026-08-21
- **Gilt fuer:** PREREGISTRATION-V4 (neue Studie), nicht rueckwirkend fuer V3
- **Vorher (D-1):** genau ein Studienworker, damit ein Session-Kill hoechstens einen
  Lauf vernichtet.
- **Nachher:** zwei Worker, einer je T4.
- **Begruendung:** Kaggle rechnet Session-Wallclock ab, nicht Device-Stunden. Zwei
  Worker halbieren den Quota-Verbrauch. Die Persistenzgarantie hat sich in V3
  bewaehrt: beim Abbruch des M2b-Kernels ging von sechs Laeufen genau der eine
  gerade laufende verloren. Das Verlustfenster waechst von einem auf zwei Laeufe,
  bei Laufzeiten unter zehn Minuten je Lauf.
- **Ergebnisse gesichtet:** ja, aber die Aenderung betrifft ausschliesslich
  Ausfuehrung und Kosten, keine wissenschaftliche Groesse.

## D-9 — log_every von 100 auf 500
- **Datum:** 2026-08-21
- **Gilt fuer:** PREREGISTRATION-V4
- **Begruendung:** Die Zwischenauswertung auf dem 101x101-Gitter kostete in V3
  gemessen rund 20 Prozent der Laufzeit. Die Endauswertung und damit die
  Primaermetrik bleiben unveraendert; nur die Trainingskurven werden groeber.
- **Verworfen:** Zwischenauswertung auf 51x51 zu verkleinern. Ersparnis nur 0.05 bis
  0.25 h, dafuer ein Eingriff in den Trainer. Nicht gerechtfertigt.

## D-10 — Allen–Cahn-Referenzloesung ersetzt: Lizenz und falsche Quellenangabe
- **Datum:** 2026-09-18
- **Abschnitt:** Daten/Provenienz, nicht Praeregistrierung; betrifft Stage B (Allen–Cahn)
- **Befund:** `data/allen_cahn.mat` war eine Kopie aus `miniHuiHui/PINN_FP64` (Xu et al.
  2025). Dieses Repository traegt keine Lizenz; die Datei wurde dennoch im MIT-Repo und im
  Kaggle-Code-Dataset weiterverteilt. README §7 gab sie zusaetzlich falsch als Datei aus dem
  Original-PINN-Release (Raissi et al., MIT) aus. Geprueft durch Vergleich mit
  `maziarraissi/PINNs/main/Data/AC.mat`: anderer Hash, anderes x-Gitter.
- **Vorher:** Fremddatei, 512 Punkte endpunkt-inklusiv, SHA-256 `ce640f18…bfa71`.
- **Nachher:** Eigene Loesung aus `data/make_allen_cahn_reference.py` (Fourier-
  Pseudospektral, ETDRK4, dt = 1e-4, float64), 513 Punkte inkl. periodischem Bild x = +1,
  SHA-256 `369b7fec…7763`. Abweichung zur alten Datei auf deren Gitter: relativer L2
  2.0e-5, punktweise max. 1.1e-3 an den Fronten; zur Raissi-Referenz: relativer L2 1.7e-5.
- **Ergebnisse gesichtet:** ja. Die vorhandenen `results/ac_*`-Laeufe wurden gegen die
  alte Datei ausgewertet und **nicht** neu berechnet. Die Referenzaenderung liegt zwei
  bis drei Groessenordnungen unter den dort berichteten Fehlern, kann die Aussagen also
  nicht veraendern; ein Neulauf kostet GPU-Budget ohne Erkenntnisgewinn.
- **Verantwortlich:** Die Herkunftsangabe in README §7 wurde nicht gegen data/README.md
  abgeglichen; die Lizenzpruefung der Quelle fehlte beim Einbinden am 2026-08-20.
- **Konsequenz:** Externe Datendateien nur noch mit nachgewiesener Lizenz oder als
  im Repository reproduzierbare Erzeugung; Hash und Quelle an genau einer Stelle
  (data/README.md) fuehren, README §7 verweist dorthin.

## D-11 — Zellen-Wallbudget: 3000-s-Grenze durch das 12-h-Sessionlimit ersetzt
- **Datum:** 2026-09-22
- **Gilt fuer:** Ausfuehrung (kaggle/v5_runner.py, kaggle/build_notebook.py), nicht
  fuer eine wissenschaftliche Groesse.
- **Vorher:** Beide Werkzeuge erzwangen ein Wallbudget unter 3000 s je Notebookzelle.
- **Befund:** Die 3000-s-Grenze war eine Schlussfolgerung aus genau einer
  Beobachtung: Der M2b-Kernel endete mit `CANCEL_ACKNOWLEDGED`, das Kernel-Log war
  0 Byte, und die sechste Zelle brach nach rund 2900 s ab. Das 0-Byte-Log ist
  inzwischen als Zeichenkodierungsfehler der Kaggle-CLI bekannt und belegt keine
  Ursache. Im Schwesterprojekt nanoWM lief eine einzelne Notebookzelle 3.4 h ohne
  Abbruch, was die angenommene Grenze widerlegt. Dokumentiert ist bei Kaggle nur
  das 12-h-Limit je Session.
- **Nachher:** `MAX_WALL_BUDGET_SECONDS = 11 h` als einzige Obergrenze; der Runner
  stoppt weiterhin selbst vor dem naechsten Batch, und mehrere Runner-Zellen bleiben
  als Resume-Reserve moeglich. Die Persistenz nach jedem Einzellauf ist unveraendert,
  das Verlustfenster bleibt also hoechstens zwei laufende Einzellaeufe.
- **Zusaetzlich:** Die Phasenlimits der Quota (`--lambda-limit-hours`,
  `--matrix-limit-hours`) sind jetzt auf der Kommandozeile setzbar, weil die
  V5-Schaetzung 7.4 h durch das GPU-Profil widerlegt ist (siehe BUDGET.md). Die
  aufgelaufenen Ist-Stunden bleiben Zustand und werden nie ueberschrieben; die
  1.0-h-Reserve bleibt fuer den Guard gesperrt.
- **Ergebnisse gesichtet:** nein. Zum Zeitpunkt der Aenderung existierte keine
  einzige V5-Matrixzeile.

## D-12 — Profilierung auf bis zu 400 Iterationen erweitert
- **Datum:** 2026-09-22
- **Gilt fuer:** `analysis/profile_gpu.py`, reine Kostenmessung (`NOT_STUDY_DATA`).
- **Vorher:** hoechstens 50 Iterationen, feste Auswahl von vier Zellen.
- **Nachher:** bis 400 Iterationen (ein Fuenftel eines Studienlaufs) und waehlbare
  Zellen; jede Zeile enthaelt zusaetzlich die Loss-Historie.
- **Begruendung:** Die Kosten eines Laufs haengen an den Funktionsauswertungen je
  Iteration, und die steigen erst, wenn die Linie-Suche in den Kollaps laeuft. Bei
  25 Iterationen sind es rund zwei je Iteration, lokal auf CPU bei kollabierten
  Laeufen bis elf. Eine Budgetplanung aus einem 25-Iterationen-Profil unterschaetzt
  die Matrix daher systematisch.
- **Ergebnisse gesichtet:** nein; das Profil schreibt nicht in die Studien-SQLite
  und meldet nur Kosten.
