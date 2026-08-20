# Entscheidungs-Gate vor der Freigabe von Zusatzbudget (Option B)

**Erstellt am 2026-08-20, bevor irgendeine M2b-Kennzahl gesichtet wurde.**
Zweck: festhalten, was Option B leisten kann und was nicht, und vorab mechanische
Prüfkriterien definieren, die entscheiden, ob das Zusatzbudget überhaupt sinnvoll
eingesetzt wäre.

---

## 1. Was Option B ist

Convection β = 50, 396 Punkte, 6000 L-BFGS-Iterationen:
3 Backbones (mlp, grand, gread) × 2 Präzisionen (fp32, fp64) × 2 Regularisierungen
(none, double_backprop) × 5 Seeds = **60 Läufe**, Gesamtquota ≈ 10.2 h.

## 2. Was Option B beantworten kann

Die Literatur sagt für Convection β = 50 ein bestimmtes Muster voraus:

- **fp32 / none:** das Vanilla-PINN scheitert (KP21 Tab. 1: rel. Fehler ~0.8–0.96;
  FP64 Fig. 8a: rMAE 0.69 bei β = 50).
- **fp64 / none:** das Vanilla-PINN gelingt (FP64 Fig. 8a: rMAE 0.0059).
- **fp32 / double_backprop:** das Vanilla-PINN gelingt (AM26 §5.1, Erfolg nach
  „a little over 5000 iterations").

Damit ist die zentrale Frage des Projekts in **einer** PDE vollständig testbar:

> Falls GRAND/GREAD in der Zelle fp32/none besser abschneiden als das MLP —
> überlebt dieser Vorsprung, wenn man ihm mit fp64 bzw. Double Backprop genau die
> zwei Krücken wegnimmt, die das MLP ohnehin retten?

Kollabiert der Architekturvorsprung in beiden Kontrollzellen, ist H0 bestätigt: der
vermeintliche Architekturgewinn war ein Präzisions- bzw. Regularisierungsartefakt.
Das ist ein sauberes, publizierbares Negativresultat und exakt das, wozu das Projekt
angetreten ist.

**Nur Option B kann dieses Muster zeigen.** Option A hat lediglich die Präzisionsachse
und sieht die halbe Kollapsstruktur nicht.

## 3. Was Option B nicht beantworten kann

- **Keine Doppeldissoziation.** H1 sagt für Convection GRAND ≤ MLP voraus — die
  *negative* Richtung. Die positive Vorhersage (GREAD gewinnt auf Allen–Cahn, weil
  der Bias dort passt) ist ohne Stage B nicht prüfbar. Ein Nullbefund auf Convection
  ist deshalb **nicht** von „GRAND hilft nirgends" unterscheidbar.
- **Keine Aussage über PDE-strukturierte Attention allgemein.** Ein Backbone, eine
  Gleichung, ein Failure-Regime, eine Graphtopologie (k = 8), eine Punktzahl.
- **Kein Attention-Vergleich.** PINNsformer bleibt draußen.

Die zulässige Schlussformulierung lautet daher höchstens: „Auf dem kanonischen
Convection-Failure-Mode bei β = 50 ist ein etwaiger Vorsprung PDE-strukturierter
Backbones gegenüber einem parametergleichen MLP (nicht) durch numerische Präzision
und Residuen-Regularisierung erklärbar." Nicht mehr.

## 4. Vorab definierte Prüfkriterien (Manipulation Checks)

Diese Kriterien prüfen, ob das **Setup** funktioniert, nicht ob eine Hypothese
stimmt. Sie sind hier vor der Sichtung festgehalten, damit ihre Anwendung keine
nachträgliche Rationalisierung ist.

**PC-1 — Failure Mode reproduziert.**
Die Zelle `mlp / fp32 / none` muss den Failure Mode zeigen: relativer L2 > 0.5.
Trifft das nicht zu, weicht das Setup so stark von der Literatur ab, dass die
Confound-Frage keinen Gegenstand hat. → **Kein Zusatzbudget.**

**PC-2 — Auflösungsvermögen vorhanden.**
Mindestens eine der sechs M2b-Zellen muss den relativen L2 unter 0.10 bringen.
Liegen alle sechs im Failure, ist bei 6000 Iterationen und 396 Punkten kein Kontrast
messbar; die zusätzlichen 60 Läufe würden eine Tabelle aus lauter Misserfolgen
erzeugen. → **Kein Zusatzbudget**, stattdessen Bericht über die Grenzen des Budgets.

**PC-3 — Numerische Integrität.**
Keine der sechs Zellen darf mit NaN/Inf oder Infrastrukturfehler terminieren, ohne
dass die Ursache verstanden ist.

Alle drei Kriterien sind gegen Literaturerwartungen formuliert und **nicht** gegen
eine Hypothesenrichtung. Sie können den Architekturvergleich weder in die eine noch
in die andere Richtung begünstigen: PC-1 und PC-2 betreffen ausschließlich die
MLP-Baseline bzw. die Existenz irgendeines Kontrasts.

## 5. Entscheidungsregel

1. M2b vollständig abwarten (6/6 terminal).
2. PC-1 bis PC-3 prüfen.
3. Alle erfüllt → Zusatzbudget für den Double-Backprop-Arm freigeben, Option B
   fahren, Ergebnis mit der Reichweitenbegrenzung aus §3 berichten.
4. Eines verletzt → kein Zusatzbudget. Schmalen Pilot abschließen oder abbrechen und
   in `FINDINGS.md` berichten, woran es lag.

Die Präregistrierung wird dadurch nicht geändert. Dieses Dokument erweitert sie
nicht, sondern regelt allein den Einsatz von Budget, das über die ursprünglich
bewilligten 5 h hinausgeht.
