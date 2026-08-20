# Findings

**Diese Studie ist ein Pilot ohne konfirmatorischen Anspruch.** Berichtet werden
Effektstärken mit Unsicherheit und Kostenmessungen; Null-Resultate stehen zuerst.

_(Keine Studienergebnisse vor M2b. M2a/Profiling ist outcome-frei. Dem Projekt
zugerechnete Kaggle-Abrechnung: 0.57 h.)_

## Technische M2-Befunde (keine Studienergebnisse)

- 2x T4 mit einem Worker war schneller als P100 fuer die implementierte Matrix;
  P100 lief reproduzierbar erst nach Installation von PyTorch 2.5.1+cu121 mit
  Pascal-Architektur `sm_60`.
- Das 50-%-Punktschema allein reduzierte die Graphlaufzeit nicht ausreichend.
- Eine semantisch identische Fusion redundanter Q/K/V-Projektionen bestand
  Ausgabe- und Gradiententests in FP64 mit Toleranz 1e-12 und erreichte lokal auf
  CPU etwa Faktor 1.20. Das anschliessende T4-Profiling war rein zeitbasiert.
- Die vorab festgelegte Kuerzungsfolge ergibt: Stage B entfaellt,
  `regularization=none`, 6000 Iterationen, 396 Loss-Punkte.
