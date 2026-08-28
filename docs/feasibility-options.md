# Fortsetzungsoptionen nach dem Feasibility-Gate

Stand: 2026-08-21. Dieses Dokument bewertet Optionen; es aendert die eingefrorene
Praeregistrierung nicht. Alle hier genannten neuen Laeufe sind `NOT_STUDY_DATA`.

## Gesicherter Diagnosebefund

1. Das MLP besitzt genug Kapazitaet: Beim ueberwachten Fit von
   `sin(x - 50t)` erreicht es besten relativen L2 0.0113.
2. Der Physics-Loss erzeugt bei beta=50 einen massiven Gradientenkonflikt:
   Domain- gegen Initialterm haben Normen 63.87 gegen 2.15 und Kosinus -0.993.
3. Bei 400 Domainpunkten overfittet ein unregularisierter Cold Start extrem:
   Trainings-Domainloss 3.43e-4, held-out 10.94, Faktor 31 844.
4. Curriculum reduziert diesen Gap auf etwa 51-54 und den L2 auf 0.63-0.68,
   erreicht die Zielschwelle 0.10 aber nicht.
5. Double Backprop reduziert mit groesserem `lambda_r` den Generalisierungs-Gap,
   verschlechtert gleichzeitig die Optimierung. Kein Wert im Grid
   `{1e-4, 1e-3, 1e-2, 1e-1, 1}` war im einheitlichen 600-Iterationen-Check
   erfolgreich.
6. Auch die laengeren 1024-/4096-Punkte-Anker erreichen 0.10 nicht. Mehr Punkte
   allein sind daher keine belegte Loesung.

## Referenztreue und offene Replikationsluecke

Andersen & Matsubara (AM26, arXiv:2605.30910v1) verwenden:

- JAX mit Flax NNX und Optax L-BFGS;
- 4 Hidden Layers x 128, tanh, Glorot-normal, Eingaben auf [-1,1];
- fuer Convection 20x20 = 400 Domainpunkte sowie je 200 Boundary- und
  Initialpunkte;
- L-BFGS-History 100 und eine relative Loss-Stoppregel ueber 100 Iterationen;
- Double Backprop ueber Domain-, Boundary- und Initialresiduen.

Das Projekt stimmt bei Netz und Lossform ueberein, verwendet aber PyTorch-L-BFGS
mit festen aeusseren Iterationen und deaktivierten Toleranzen. Vor allem nennt
AM26 den Wert von `lambda_r` nicht. Ohne Autorenwert oder Referenzcode ist eine
exakte Reproduktion des zentralen Erfolgsclaims nicht moeglich.

Die Projektheuristik von mindestens vier Zeitabtastungen pro Periode ist
konservativ und faengt das unregularisierte 14x14-M2b-Setup korrekt ab. Sie ist
aber nicht als universelles Gesetz durch AM26 gedeckt: AM26 berichtet gerade mit
20x20 und Double Backprop Erfolg. Fuer kuenftige Designs muessen Aufloesung und
held-out Residual gemeinsam als getrennte Checks behandelt werden.

## Option A — Methodischen Feasibility-Befund abschliessen

**Empfehlung, wenn keine neue Praeregistrierung gewuenscht ist.**

Lieferumfang:

- entwerteten M2b-Slice transparent berichten;
- zeigen, dass ein kleiner Trainingsloss ohne held-out Residual irrefuehrend ist;
- Aufloesungsheuristik, Gradientenkonflikt und Lambda-Trade-off dokumentieren;
- keine Architektur-Rangfolge behaupten.

Vorteile: wissenschaftlich sauber, kein weiteres GPU-Risiko, vorhandene Arbeit
bleibt als Reproduzierbarkeits- und Methodenergebnis wertvoll. Nachteil: Die
urspruengliche Architekturfrage bleibt unbeantwortet.

## Option B — AM26 zuerst exakt reproduzieren, dann neu praeregistrieren

**Empfohlener Weg, wenn der Architekturvergleich erhalten werden soll.**

Harte Reihenfolge:

1. separate JAX/Flax/Optax-Umgebung aufbauen;
2. 400+200+200-Convection-Baseline mit dem Paper-Optimizer reproduzieren;
3. `lambda_r` durch Autorenkontakt oder freigegebenen Referenzcode klaeren;
4. Erfolg auf mindestens zwei Seeds und mit held-out Residual bestaetigen;
5. erst danach Kosten neu kalibrieren und eine v4-Praeregistrierung einfrieren;
6. PyTorch- und JAX-Ergebnisse nicht als identische Implementierung behandeln.

Go-Gate: mindestens zwei unabhaengige Baselinelaeufe unter L2 0.10, ohne grossen
Train/Test-Residual-Gap. Ohne Go-Gate endet Option B in Option A.

## Option C — Curriculum als neue Forschungsfrage

Der lokale Check zeigt einen echten Effekt: gleicher Rechenumfang verbessert
beta=50 von 0.971 auf 0.634 und reduziert extremes Overfitting. Die Literatur
berichtet ebenfalls starke Curriculum-Effekte bei hoher Konvektion.

Das ist dennoch **keine Reparatur innerhalb der bestehenden Studie**. Warm Start
veraendert Initialisierung und Optimierungspfad und damit die Nullhypothese. Eine
saubere neue Studie muesste Cold Start gegen Curriculum faktoriell vergleichen;
Backbone-, Praezisions- und Regularisierungsfaktoren waeren dann zu viel fuer das
vorhandene Budget. Sinnvoll waere ein kleiner MLP-Methodenpilot vor jeder
Architekturfrage.

## Option D — Robustheitsbenchmark statt Architekturbenchmark

Neue Zielvariable: Welche Kombination verhindert einen grossen held-out
Residual-Gap bei vorgegebenem Kostenbudget? Kandidaten waeren FP64, Double
Backprop, Curriculum und spaeter adaptive Stichproben. Backbones waeren erst eine
zweite Stufe.

Vorteil: passt direkt zum beobachteten Problem. Nachteil: groesserer Scope-Wechsel;
erfordert neue Hypothesen, neue Gates und neue Praeregistrierung.

## Nicht empfohlene Option — bestehende Matrix unveraendert starten

Das wuerde hauptsaechlich vergleichen, wie verschiedene Backbones an einem nicht
validierten Physics-Loss scheitern. Niedrige Trainingsverluste koennten dabei
erneut massives Residual-Overfitting verbergen. Die 27 GPU-Stunden loesen diesen
Identifikationsfehler nicht.

## Entscheidung

- Ohne Scope-Neustart: **Option A**.
- Mit fortbestehendem Ziel Architekturvergleich: **Option B**, strikt vor jeder
  neuen Matrix.
- Wenn der interessanteste neue Befund verfolgt werden soll: **Option C** als
  kleiner, neu praeregistrierter Methodenpilot.

Option D ist langfristig wissenschaftlich interessant, aber fuer den aktuellen
Stand breiter als noetig. Die unveraenderte Matrix bleibt gesperrt.
