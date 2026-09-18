# Referenzdaten

`allen_cahn.mat` ist die Allen–Cahn-Referenzloesung fuer Stage B. Sie wird **im
Repository selbst erzeugt** und hat keine externe Datenabhaengigkeit:

```bash
python data/make_allen_cahn_reference.py --check
```

- Problem: u_t = 1e-4 u_xx − 5u³ + 5u, x ∈ [−1, 1) periodisch, t ∈ [0, 1],
  u(x, 0) = x² cos(πx) — identisch mit `configs/stage_b.json` und `src/pdes/allen_cahn.py`
- Verfahren: Fourier-Pseudospektral (512 Moden, float64), ETDRK4 (Kassam & Trefethen
  2005), dt = 1e-4, 3/2-Dealiasing des kubischen Terms
- Layout: `t` 1×201, `x` 1×513, `usol` 201×513 (Zeit × Raum); die 513. Spalte ist das
  periodische Bild x = +1, damit die bilineare Interpolation auf [−1, 1] nie extrapoliert
- Selbstkonvergenz |u(dt) − u(2dt)|_max = 1.4e-11
- Uebereinstimmung mit der Chebfun-Referenz aus dem Original-PINN-Repository
  (Raissi et al., `main/Data/AC.mat`, MIT): relativer L2-Fehler 1.7e-5, punktweise
  max. 1.1e-3 an den scharfen Fronten (Interpolationsfehler auf das andere Gitter)
- SHA-256 der eingecheckten Datei: `369b7fec0df907ce630b5f41dd2987a2c9eec77c13fca3a4bc8e5cc99127e763`
  (scipy `savemat` schreibt einen Zeitstempel in den Header, daher ist der Hash bei
  Neuerzeugung nicht bytegleich; der Inhalt ist es)

**Historie:** Bis 2026-09-18 lag hier eine Datei (SHA-256 `ce640f18…bfa71`) aus dem
Repository `miniHuiHui/PINN_FP64` (Xu et al. 2025), das keine Lizenz traegt. Sie enthielt
dieselbe Loesung auf einem endpunkt-inklusiven 512er-Gitter (relativer L2-Abstand zur
neuen Datei auf ihrem Gitter: 2.0e-5). Ersetzt aus Lizenzgruenden, siehe DEVIATIONS.md D-10.
Die Datei wird in das Kaggle-Code-Dataset aufgenommen, weil das Notebook-Netzwerk
deaktiviert bleibt.
