# Baseline-Definitionen aus der Literatur

Erstellt vom `lit-agent`, Phase 1 (M0). Stand: 2026-08-20.

**Lesekonvention.** Jede Zahl ist markiert als `[belegt: <Fundstelle>]` oder
`[unbelegt/geschätzt]`. Alle Angaben wurden im Volltext (arXiv-PDF, lokal per
`pdftotext -layout` extrahiert, bzw. arXiv-HTML) gelesen, nicht aus Abstracts oder
Sekundärquellen. Wo die PDF-Textextraktion Tabellenspalten gegen die Zeilenlabels
verschoben hat, ist das als **Layout-Rekonstruktion** gekennzeichnet — dort ist vor
der Implementierung ein Blick ins Original-PDF nötig.

Verwendete Volltexte:

| Kürzel | arXiv | Version gelesen |
|---|---|---|
| KP21 | 2109.01050 | v2, 11 Nov 2021, PDF |
| PF24 | 2307.11833 | ar5iv-HTML |
| GRAND | 2106.10934 | PDF |
| GREAD | 2211.14208 | PDF |
| FP64 | 2505.10949 | v2, 28 Nov 2025, PDF |
| AM26 | 2605.30910 | v1, 29 Mai 2026, PDF + arXiv-HTML |
| EG23 | 2308.08468 | PDF |

---

## 1. Krishnapriyan, Gholami, Zhe, Kirby, Mahoney — "Characterizing possible failure modes in physics-informed neural networks", NeurIPS 2021 (arXiv 2109.01050v2)

### 1.1 Welche Gleichungen das Paper enthält

Das Paper behandelt **genau drei** Probleme:

1. **1D Convection** (§3.1, Gl. 5)
2. **1D Reaction-Diffusion** (§3.2, Gl. 10)
3. **1D Reaction** (Appendix A, Gl. 14) — als Ergänzung zu §3

Eine Volltextsuche über das gesamte PDF nach `Allen`, `Cahn`, `wave`, `Wave`,
`Burgers`, `Helmholtz` liefert **null Treffer**. [belegt: Volltext-Grep über
2109.01050v2, alle 1334 Zeilen]

### 1.2 Convection (§3.1)

PDE (Gl. 5) [belegt: §3.1, Gl. 5]:

```
∂u/∂t + β ∂u/∂x = 0,   x ∈ Ω,  t ∈ [0, T]
u(x, 0) = h(x),        x ∈ Ω
```

Konkrete IC/BC (Gl. 9) [belegt: §3.1, Gl. 9]:

```
u(x, 0) = sin(x)
u(0, t) = u(2π, t)          (periodisch)
```

Gebiet: Ω = [0, 2π) für die periodische Randbedingung [belegt: §3.1, Text vor Gl. 8:
„For periodic boundary conditions with Ω = [0, 2π)"]. Das Zeitintervall wird in §3.1
nur als `t ∈ [0, T]` geschrieben; T = 1 steht **nicht** explizit im Haupttext, wird
aber in Fußnote 4 als Beispiel verwendet („for T = [0, 1], if we use 1000 collocation
points…") [belegt: Fußnote 4, S. 9] — für Convection selbst also
**[unbelegt/geschätzt: T = 1]**, wenn auch stark nahegelegt.

Analytische Lösung (Gl. 6) [belegt: §3.1, Gl. 6]:

```
u_analytical(x, t) = F^{-1}( F(h(x)) · e^{-i β k t} )
```

mit F = Fourier-Transformation, k = Frequenz im Fourierraum. Die geschlossene Form
`sin(x − βt)`, die spätere Arbeiten verwenden, steht in KP21 **nicht** explizit da —
sie steht bei FP64 und AM26 (siehe §5, §6). [belegt: Vergleich Gl. 6 KP21 vs. Gl. 6
FP64 / Gl. 9 AM26]

Loss (Gl. 7 + Gl. 8) [belegt: §3.1, Gl. 7 und Gl. 8]:

```
L(θ) = (1/N_u) Σ_{i=1..N_u} |û − u_0^i|²
     + (1/N_f) Σ_{i=1..N_f} λ_i |∂û/∂t + β ∂û/∂x|²
     + L_B

L_B = (1/N_b) Σ_{i=1..N_b} |û(θ, 0, t) − û(θ, 2π, t)|²
```

Die periodische Randbedingung wird also **weich** als Extraterm erzwungen, nicht hart
eingebaut [belegt: §3 Experiment setup: „We enforce this through an extra term in the
loss function that takes the difference between the predicted NN solution at each
boundary"].

Loss-Gewichtung: Es gibt einen Gewichtsparameter λ auf dem Residualterm. Konkrete
Zahlenwerte stehen im Haupttext nicht; §4 sagt nur, dass λ variiert wurde und dass
Tuning das Problem nicht löst [belegt: §4, Absatz „Finally, we study the impact of
changing the weight/multiplier for the soft regularization term (i.e., the λ parameter
in Eq. 3) … While we find that tuning λ can help change the error, it cannot resolve
the problem, as shown in Fig. E.1"]. **[unbelegt: numerischer Default-Wert von λ]** —
plausibler Default λ = 1, aber im Paper nicht so geschrieben.

### 1.3 Reaction-Diffusion (§3.2)

PDE (Gl. 10) [belegt: §3.2, Gl. 10]:

```
∂u/∂t − ν ∂²u/∂x² − ρ u(1 − u) = 0,   x ∈ Ω,  t ∈ (0, T]
u(x, 0) = h(x)
```

ν > 0 ist der Diffusionskoeffizient [belegt: §3.2, Text nach Gl. 10]. IC/BC sind
dieselben wie im Reaction-Fall (Gaußprofil, periodisch); der Loss (Gl. 13) enthält
denselben L_B wie Gl. 8 [belegt: §3.2, Text nach Gl. 13: „periodic boundary conditions
can be enforced by including L_B from Eq. 8 as an extra term"].

Referenzlösung: **Strang-Splitting** in Reaktions- und Diffusionsschritt (Gl. 11); der
Reaktionsschritt wird über die geschlossene Form Gl. 15 gelöst, der Diffusionsschritt
spektral (Gl. 12) [belegt: §3.2, Gl. 11–12].

```
u_diffusion(x, t) = F^{-1}( F(u(x, t=t_n)) · e^{-ν k² Δt} )
```

Getestete Parameter: ρ = 5 fest, ν ∈ {2, 3, 4, 5, 6} [belegt: Tab. 2 und Fig. D.1;
Fig. D.1 zeigt ρ = 5 mit ν = 2, 3, 4].

Berichtete Fehler (Haupttext) [belegt: §3.2, Observations]:
- ν = 5, ρ = 5: relativer Fehler **93 %**
- ν = 2, ρ = 5: relativer Fehler **50 %**

### 1.4 Reaction (Appendix A)

PDE (Gl. 14) [belegt: §A, Gl. 14]:

```
∂u/∂t − ρ u(1 − u) = 0,   x ∈ Ω,  t ∈ (0, T]
```

IC/BC (Gl. 17) [belegt: §A, Gl. 17]:

```
u(x, 0) = exp( −(x − π)² / (2 (π/4)²) )
u(0, t) = u(2π, t)
```

Analytische Lösung (Gl. 15) [belegt: §A, Gl. 15]:

```
u_analytical(x, t) = h(x) e^{ρt} / ( h(x) e^{ρt} + 1 − h(x) )
```

Loss: Gl. 16, gleiche Struktur wie Gl. 7, mit L_B aus Gl. 8 [belegt: §A, Gl. 16].

Getestete ρ-Werte: ρ ∈ {5, 6, 7, 8, 9, 10} in Tab. E.2 [belegt: Tab. E.2, Zeilenlabels].
Fig. A.1 zeigt zusätzlich einen ρ-Sweep; die dort abgebildeten Einzelwerte sind aus dem
Text nicht zahlengenau ablesbar. **[unbelegt: vollständige ρ-Liste in Fig. A.1]**

### 1.5 Netzarchitektur, Optimierer, Kollokationspunkte

Alles aus einem einzigen Absatz, §3 „Experiment setup" [belegt: §3, S. 4]:

- **Architektur:** 4-layer fully-connected NN, **50 Neuronen pro Layer**,
  **tanh**-Aktivierung.
- **Parameterzahl:** im Paper **nicht angegeben**. **[unbelegt/geschätzt]** — bei
  Eingabe (x,t)→2, drei Hidden-zu-Hidden-Matrizen 50×50 und Ausgabe 1 ergibt sich
  ≈ 7,8 k Parameter; das ist eine eigene Rechnung, keine Paperangabe.
- **Optimierer:** **L-BFGS**, Learning-Rate-Sweep von **1e-4 bis 2.0**.
  **Kein Adam.** Fußnote 3 begründet das ausdrücklich: „For reasons that are only
  partially understood, L-BFGS methods tend to perform better for existing PINN
  problems… we found that they [SGD-Varianten] underperform in comparison to L-BFGS."
  Es gibt **keine** Adam+L-BFGS-Kombination in KP21.
- **Iterationszahl:** im Paper **nicht angegeben**. **[unbelegt/geschätzt]**
- **Kollokationspunkte:** „randomly sample collocation points (x, t) on the domain" —
  also **zufälliges Sampling, kein Gitter**. Zahl im Haupttext nicht systematisch
  angegeben; Fußnote 4 nennt beispielhaft **1000** Kollokationspunkte für das gesamte
  Zeitintervall [belegt: Fußnote 4]. Für Tab. E.1 (Convection seq2seq) wurden
  ausdrücklich **10 000** Punkte verwendet [belegt: Caption Tab. E.1: „For this
  example only, we use a higher number of collocation points (10000)"].
- **N_u (IC-Punkte) und N_b (BC-Punkte):** im Paper **nicht beziffert**.
  **[unbelegt/geschätzt]**
- **Seeds:** „we run models at least ten times with different preset random seeds, and
  we average the relative and absolute errors" [belegt: §3, S. 4].
- **Metrik:** rel. L2 = (1/N) Σ ||û − u||₂ / ||u||₂, abs. Fehler = (1/N) Σ ||û − u||₂
  [belegt: §3, S. 4].

### 1.6 Ergebnisse: Convection

**Fig. 1(a)** zeigt rel./abs. Fehler über β. Der Text dazu [belegt: §3.1, Observations]:

> „the PINN is only able to achieve good solutions for small values of convection
> coefficient, and it fails when β becomes larger, **reaching a relative error of
> almost 100 % for β > 10**."

Die Einzelwerte der Fig.-1(a)-Kurve sind aus dem Text nicht zahlengenau ablesbar.
**[unbelegt: Werte der Fehlerkurve Fig. 1(a) außerhalb der Tabellen]**

**Tab. 1** (Curriculum-Regularisierung vs. reguläres PINN) [belegt: Tab. 1, S. 7;
Layout-Rekonstruktion aus der pdftotext-Ausgabe]:

| Fall | Metrik | Regular PINN | Curriculum training |
|---|---|---|---|
| 1D convection β = 20 | rel. Fehler | 7.50 × 10⁻¹ | 9.84 × 10⁻³ |
| 1D convection β = 20 | abs. Fehler | 4.32 × 10⁻¹ | 5.42 × 10⁻³ |
| 1D convection β = 30 | rel. Fehler | 8.97 × 10⁻¹ | 2.02 × 10⁻² |
| 1D convection β = 30 | abs. Fehler | 5.42 × 10⁻¹ | 1.10 × 10⁻² |
| 1D convection β = 40 | rel. Fehler | 9.61 × 10⁻¹ | 5.33 × 10⁻² |
| 1D convection β = 40 | abs. Fehler | 5.82 × 10⁻¹ | 2.69 × 10⁻² |

**Tab. E.1** (Convection, seq2seq/Time-Marching, 10 000 Kollokationspunkte)
[belegt: Tab. E.1, S. 19; Layout-Rekonstruktion]:

| β | Metrik | Entire state space | Δt = 0.05 | Δt = 0.1 |
|---|---|---|---|---|
| 30 | rel. | 7.38 × 10⁻¹ | 2.13 × 10⁻¹ | 1.05 × 10⁻¹ |
| 30 | abs. | 5.57 × 10⁻¹ | 1.29 × 10⁻¹ | 5.95 × 10⁻² |
| 40 | rel. | 8.25 × 10⁻¹ | 4.58 × 10⁻¹ | 2.41 × 10⁻¹ |
| 40 | abs. | 6.06 × 10⁻¹ | 2.58 × 10⁻¹ | 1.35 × 10⁻¹ |

**Fig. C.1** zeigt Heatmaps für β = 30, 50, 70 (plus mindestens einen kleineren Wert,
dessen Label in der Textextraktion nicht lesbar war) [belegt: Captions Fig. C.1(d)–(l)].
Damit ist belegt, dass KP21 **bis β = 70** getestet hat.

### 1.7 Ergebnisse: Reaction (Tab. E.2)

[belegt: Tab. E.2, S. 20; Layout-Rekonstruktion — Zeilenlabels ρ = 5, 6, 7, 8, 9, 10,
sechs rel./abs.-Paare in dieser Reihenfolge]

| ρ | Metrik | Entire state space | Δt = 0.05 | Δt = 0.1 |
|---|---|---|---|---|
| 5 | rel. | 9.79 × 10⁻¹ | 7.06 × 10⁻² | 7.09 × 10⁻² |
| 5 | abs. | 5.40 × 10⁻¹ | 2.52 × 10⁻² | 2.39 × 10⁻² |
| 6 | rel. | 9.88 × 10⁻¹ | 8.25 × 10⁻² | 7.78 × 10⁻² |
| 6 | abs. | 5.88 × 10⁻¹ | 3.02 × 10⁻² | 2.65 × 10⁻² |
| 7 | rel. | 9.92 × 10⁻¹ | 8.16 × 10⁻² | 7.56 × 10⁻² |
| 7 | abs. | 6.31 × 10⁻¹ | 3.03 × 10⁻² | 2.69 × 10⁻² |
| 8 | rel. | 9.94 × 10⁻¹ | 8.19 × 10⁻² | 7.44 × 10⁻² |
| 8 | abs. | 6.69 × 10⁻¹ | 3.10 × 10⁻² | 2.73 × 10⁻² |
| 9 | rel. | 9.95 × 10⁻¹ | 7.02 × 10⁻² | 8.63 × 10⁻² |
| 9 | abs. | 7.02 × 10⁻¹ | 2.83 × 10⁻² | 3.21 × 10⁻² |
| 10 | rel. | 9.96 × 10⁻¹ | 6.88 × 10⁻² | 7.47 × 10⁻² |
| 10 | abs. | 7.31 × 10⁻¹ | 2.85 × 10⁻² | 2.85 × 10⁻² |

**Wichtig für das Projekt:** Das Vanilla-PINN ist bei Reaction schon ab **ρ = 5**
vollständig gescheitert (rel. Fehler 0.979). Die Erfolgsschwelle 0.10 aus der
Präregistrierung ist damit für Vanilla-MLP-Reaction bei ρ = 5 sicher nicht erreichbar.

### 1.8 Ergebnisse: Reaction-Diffusion (Tab. 2)

[belegt: Tab. 2, S. 9; Layout-Rekonstruktion — Zeilenlabels ν = 2, 3, 4, 5, 6 bei
ρ = 5, fünf rel./abs.-Paare]

| ν (ρ = 5) | Metrik | Entire state space | Δt = 0.05 | Δt = 0.1 |
|---|---|---|---|---|
| 2 | rel. | 5.07 × 10⁻¹ | 2.04 × 10⁻² | 1.18 × 10⁻² |
| 2 | abs. | 2.70 × 10⁻¹ | 1.06 × 10⁻² | 6.41 × 10⁻³ |
| 3 | rel. | 7.98 × 10⁻¹ | 1.92 × 10⁻² | 1.56 × 10⁻² |
| 3 | abs. | 4.79 × 10⁻¹ | 1.01 × 10⁻² | 8.17 × 10⁻³ |
| 4 | rel. | 8.84 × 10⁻¹ | 2.37 × 10⁻² | 1.59 × 10⁻² |
| 4 | abs. | 5.74 × 10⁻¹ | 1.15 × 10⁻² | 8.01 × 10⁻³ |
| 5 | rel. | 9.35 × 10⁻¹ | 2.36 × 10⁻² | 2.31 × 10⁻² |
| 5 | abs. | 6.46 × 10⁻¹ | 1.09 × 10⁻² | 1.15 × 10⁻² |
| 6 | rel. | 9.60 × 10⁻¹ | 2.81 × 10⁻² | 2.69 × 10⁻² |
| 6 | abs. | 6.84 × 10⁻¹ | 1.17 × 10⁻² | 1.28 × 10⁻² |

Die Zuordnung ν = 2 → 5.07 × 10⁻¹ und ν = 5 → 9.35 × 10⁻¹ ist durch den Haupttext
unabhängig bestätigt (50 % bzw. 93 %) [belegt: §3.2, Observations].

---

## 2. Zhao, Ding, Prakash — "PINNsFormer: A Transformer-Based Framework For Physics-Informed Neural Networks", ICLR 2024 (arXiv 2307.11833)

Gelesen über ar5iv-HTML.

### 2.1 PDEs

**Convection** (Gl. 8) [belegt: §5/Gl. 8]:

```
∂u/∂t + β ∂u/∂x = 0,  β = 50
x ∈ [0, 2π],  t ∈ [0, 1]
u(x, 0) = sin(x),  u(0, t) = u(2π, t)
```

**1D-Reaction** (Gl. 9) [belegt: Gl. 9]:

```
∂u/∂t − ρ u(1 − u) = 0,  ρ = 5
x ∈ [0, 2π],  t ∈ [0, 1]
u(x, 0) = exp( −(x − π)² / (2 (π/4)²) ),  periodisch
u_analytical = h(x) e^{ρt} / ( h(x) e^{ρt} + 1 − h(x) )
```

**1D-Wave** (Gl. 11) [belegt: Gl. 11]:

```
∂²u/∂t² − β ∂²u/∂x² = 0,  β = 3
x ∈ [0, 1],  t ∈ [0, 1]
u(x, 0) = sin(πx) + ½ sin(βπx),  ∂u/∂t(x, 0) = 0
u(0, t) = u(1, t) = 0
u(x, t) = sin(πx) cos(2πt) + ½ sin(βπx) cos(2βπt)
```

⚠️ Achtung: β steht hier an **zwei verschiedenen Stellen** — als Koeffizient vor u_xx
und als Oberwellenzahl in der IC. Die analytische Lösung mit `cos(2πt)` und
`cos(2βπt)` ist nur konsistent, wenn der Koeffizient vor u_xx den Wert **4** hat
(Wellengeschwindigkeit c = 2), nicht β = 3. FP64 und AM26 schreiben die Gleichung
deshalb explizit als `u_tt − 4 u_xx = 0` mit β = 3 **nur** in der IC. Siehe
§ Widersprüche, W-3.

**2D Navier-Stokes** (Gl. 13): λ₁ = 1, λ₂ = 0.01, Druckauswertung bei t = 20.0,
2500 Trainingspunkte [belegt: Gl. 13 / §5].

### 2.2 Architekturen (Tab. 4)

[belegt: Tab. 4]

| Modell | Layer | Hidden | Params | Aktivierung |
|---|---|---|---|---|
| PINN (Baseline) | 4 | 512 | 527 k | ReLU |
| QRes | 4 | 256 | 397 k | — |
| FLS (First-Layer Sine) | 4 | 512 | 527 k | — |
| PINNsFormer | 1 Enc + 1 Dec, Embed 32, 2 Heads | 512 | 454 k | **Wavelet** |

Wavelet-Aktivierung (Gl. 4) [belegt: Gl. 4]: `Wavelet(x) = ω₁ sin(x) + ω₂ cos(x)`
mit lernbaren ω₁, ω₂.

⚠️ Die **ReLU**-Angabe für das Baseline-PINN steht so in Tab. 4; sie widerspricht der
gesamten übrigen PINN-Literatur (tanh). Siehe § Widersprüche, W-4.

### 2.3 Optimierung und Punkte

[belegt: §5 Experimental setup / Tab. 6-7 Bereich]

- Optimierer: **L-BFGS mit Strong-Wolfe-Liniensuche**, **1000 Iterationen**.
- Loss-Gewichte: λ_res = λ_ic = λ_bc = **1** (Gl. 6).
- Kollokationspunkte Baselines: N_ic = N_bc = **101**, Residuum auf **101 × 101**-Gitter
  (N_res = 10 201).
- PINNsFormer: N_ic = N_bc = **51**, Residuum auf **51 × 51**-Gitter.
- Testgitter: **101 × 101**.
- Pseudo-Sequenz: k = **5**, Δt = **1e-3** bzw. **1e-4**; Sensitivität in Tab. 7 über
  k ∈ {3, 5, 7, 10} und Δt ∈ {1e-1 … 1e-5}.
- Overhead bei k = 5: **2.92×** Trainingszeit, **2.15×** GPU-Speicher (Tab. 5).

### 2.4 Berichtete Fehler

Metriken (Gl. 7): rMAE (relativer L1) und rRMSE (relativer L2).

Tab. 1 (Convection β = 50, 1D-Reaction ρ = 5) [belegt: Tab. 1]:

| Modell | Convection rRMSE | Reaction rRMSE |
|---|---|---|
| PINN | 0.840 | 0.981 |
| QRes | 0.816 | 0.977 |
| FLS | 0.771 | 0.985 |
| PINNsFormer | **0.027** | **0.030** |

Tab. 2 (1D-Wave, mit NTK-Gewichtung) [belegt: Tab. 2]:

| Modell | Loss | rRMSE |
|---|---|---|
| PINN | 0.0193 | 0.335 |
| PINN + NTK | 0.00634 | 0.149 |
| PINNsFormer + NTK | **0.00421** | **0.058** |

Tab. 3 (Navier-Stokes) [belegt: Tab. 3]: PINN 9.08, QRes 4.45, FLS 2.77,
PINNsFormer 0.280 (rRMSE).

---

## 3. Chamberlain, Rowbottom, Gorinova, Webb, Rossi, Bronstein — "GRAND: Graph Neural Diffusion", ICML 2021 (arXiv 2106.10934)

**Für dieses Projekt liefert GRAND keine PDE-Benchmark-Parameter.** GRAND ist eine
Arbeit zur Knotenklassifikation auf Graphen (Cora, Citeseer, Pubmed, CoauthorCS,
Computers, Photo, ogbn-arxiv), nicht zu PINNs. Übernommen wird ausschließlich die
**Backbone-Dynamik**. [belegt: §6 Experiments, Tab. 3/4]

Kern-Diffusionsgleichung (Gl. 1) [belegt: §3.1, Gl. 1]:

```
∂x(t)/∂t = div[ G(x(t), t) ∇x(t) ],   x(0) gegeben
```

mit G = diag(a(x_i(t), x_j(t), t)), einer e × e Diagonalmatrix. Eingesetzt (Gl. 2):

```
∂x(t)/∂t = ( A(x(t)) − I ) x(t) = Ā(x(t)) x(t)
```

A(x) = (a(x_i, x_j)) ist die n × n **Attention-Matrix** mit der Sparsity-Struktur der
Adjazenz (a_ij = 0 falls (i,j) ∉ E) [belegt: §3.1, Text nach Gl. 2]. Ist A zeitkonstant,
lautet die Lösung analytisch `x(t) = e^{Āt} x(0)` [belegt: §3.1].

Attention (Gl. 10) [belegt: Gl. 10]:

```
a(X_i, X_j) = softmax( (W_K X_i)^T W_Q X_j / d_k )
```

Explizites Euler-Schema (Gl. 6/7) [belegt: §3.3, Gl. 6–7]:

```
x^{(k+1)} = ( I + τ Ā(x^{(k)}) ) x^{(k)} = Q^{(k)} x^{(k)}
```

Varianten [belegt: §4]: **GRAND-l** (lineare, zeitkonstante Attention → gekoppeltes
lineares ODE-System), **GRAND-nl** (nichtlinear), **GRAND-nl-rw** (nichtlinear mit
Graph-Rewiring).

Solver: Method-of-Lines-Analogie; explizite (Euler, Runge-Kutta 4, Adams-Bashforth)
und implizite Schemata (Implicit Euler, Adams-Moulton) werden verglichen
[belegt: §3.3 und Fig. 3].

Hyperparameter: Fig. 2/§6 nennt eine Schrittweite **τ = 1.0** bei variierter
Integrationszeit T [belegt: §6, Text zu Fig. 2: „step size τ = 1.0, varying the
integration time T"]. Weitere Werte (Hidden-Dim je Datensatz) stehen im Appendix und
wurden hier **nicht** extrahiert. **[unbelegt: Hidden-Dim, T-Werte je Datensatz]**

Genauigkeiten (Tab. 3, Planetoid-Splits) [belegt: Tab. 3]: GRAND-l 84.7 ± 0.6 (Cora),
73.3 ± 0.4 (Citeseer), 80.4 ± 0.4 (Pubmed); GRAND-nl 83.6/70.8/79.7;
GRAND-nl-rw 82.9/73.6/81.0.

---

## 4. Choi, Hong, Park, Cho et al. — "GREAD: Graph Neural Reaction-Diffusion Networks", ICML 2023 (arXiv 2211.14208)

**Auch GREAD liefert keine PDE-Benchmark-Parameter für PINNs** — es ist eine
Knotenklassifikations-Arbeit über 9 Datensätze [belegt: Contribution 3: „We consider a
comprehensive set of 9 datasets"].

Kerndynamik [belegt: §3, Gl. 7 im Kontext von Gl. 6–8]:

```
f(H(t)) := dH(t)/dt = −α L H(t) + β r(H(t))
y = o(H(T)),   H(0) = e(X)
```

L = Laplacian, α und β trainierbar, e = Encoder (mehrere FC-Layer mit ReLU),
o = Output-Layer mit Softmax [belegt: §3].

Reaktionsterme r (Gl. 10) [belegt: §3, Gl. 10]:

| Variante | r(H(t)) |
|---|---|
| Fisher (F) | H(t) ⊙ (1 − H(t)) |
| Allen-Cahn (AC) | H(t) ⊙ (1 − H(t)²) |
| Zeldovich (Z) | H(t) ⊙ (H(t) − H(t)²) |
| Blurring-Sharpening (BS) | (Ã − Ã²) H(t) |
| Source Term (ST) | H(0) |
| Filter Bank (FB) | −L H(t) |
| Filter Bank* (FB*) | −L H(t) + H(t) |

ODE-Solver: Dormand–Prince (dopri5) [belegt: §3, Referenz auf Dormand & Prince 1980].

Ranking-Übersicht (Tab. 1) [belegt: Tab. 1]: GREAD-BS mittlerer Rang 1.56 / mittlere
Accuracy 76.64; GREAD-FB* 6.72 / 74.51; GREAD-F 7.50 / 74.13; GREAD-AC 8.50 / 73.71.

**Achtung für das Projekt:** Der Name „Allen-Cahn" in GREAD bezeichnet den
*Reaktionsterm der Backbone-Dynamik*, **nicht** die Allen-Cahn-**Benchmark-PDE**. Diese
beiden Dinge dürfen in `src/models/gread.py` bzw. `src/pdes/` nicht verwechselt werden.

---

## 5. Xu, Liu, Nassereldine, Xiong — "FP64 is All You Need: Rethinking Failure Modes in Physics-Informed Neural Networks", arXiv 2505.10949

**Existenz verifiziert.** v1: 16. Mai 2025, v2: 28. Nov. 2025. Autoren: Chenhui Xu,
Dancheng Liu, Amir Nassereldine, Jinjun Xiong (University at Buffalo, SUNY). NeurIPS-2025-
Poster-Eintrag vorhanden (neurips.cc/virtual/2025/poster/120125). Der im Projektauftrag
genannte Venue „NeurIPS 2025" ist damit plausibel bestätigt. [belegt: arXiv-Abs-Seite,
NeurIPS-Posterseite]

### 5.1 PDEs (Appendix A)

**A.1 Convection** (Gl. 5) [belegt: §A.1, Gl. 5]:

```
∂u/∂t + β ∂u/∂x = 0,   x ∈ [0, 2π],  t ∈ [0, 1]
u(x, 0) = sin(x)
u(0, t) = u(2π, t)
β = 50   („Following prevailing practice [36, 33], we set β = 50")
u_ana(x, t) = sin(x − βt)                                   (Gl. 6)
```

**A.2 Reaction** (Gl. 7) [belegt: §A.2, Gl. 7–8]:

```
∂u/∂t − ρ u(1 − u) = 0,   x ∈ [0, 2π],  t ∈ [0, 1]
u(x, 0) = exp( −(x − π)² / (2 (π/4)²) )
u(0, t) = u(2π, t)
ρ = 5   („we adopt the standard choice ρ = 5 in accordance with [36, 33]")
u_ana = h(x) e^{ρt} / ( h(x)(e^{ρt} − 1) + 1 )
```

**A.3 Wave** (Gl. 9) [belegt: §A.3, Gl. 9–10]:

```
∂²u/∂t² − 4 ∂²u/∂x² = 0,   x ∈ [0, 1],  t ∈ [0, 1]
u(x, 0) = sin(πx) + ½ sin(βπx)
∂u/∂t(x, 0) = 0
u(0, t) = u(1, t) = 0
β = 3
u_ana(x, t) = sin(πx) cos(2πt) + ½ sin(βπx) cos(2βπt)
```

Beachte: Der Koeffizient ist hier **explizit 4**, β = 3 steuert nur die zweite
Harmonische. Das ist die saubere, in sich konsistente Fassung.

**A.4 Allen-Cahn** (Gl. 11) [belegt: §A.4, Gl. 11]:

```
∂u/∂t − 0.0001 ∂²u/∂x² + 5u³ − 5u = 0,   x ∈ (−1, 1),  t ∈ (0, 1)
u(x, 0) = x² cos(πx)
u(−1, t) = u(1, t)
∂u/∂x(−1, t) = ∂u/∂x(1, t)
```

Keine geschlossene Lösung; Referenz ist eine hochauflösende **spektrale Approximation**
[belegt: §A.4].

### 5.2 Setup (§3.2)

[belegt: §3.2 Experiments Settings]

- Vanilla-PINN: MLP mit **3 Hidden Layers, 512 Neuronen** pro Layer.
- Aktivierung: im Setup-Absatz nicht genannt. **[unbelegt: Aktivierungsfunktion]**
- Optimierer: **L-BFGS** („following common practice").
- Hardware: NVIDIA H100, CUDA 12.8, PyTorch 2.1.1.
- Weitere Architekturen (PINNsFormer, KAN, PINNMamba) „following their original settings".
- Kollokationspunktzahl im Haupttext nicht genannt. **[unbelegt]** — AM26 gibt für die
  von FP64 übernommenen Zahlen ein **10 201-Gitter** an [belegt: AM26 §4.3].

### 5.3 Mechanismus (§5.2) — direkt relevant für die H0 des Projekts

[belegt: §5.2]

> „the trigger condition for convergence in PINN (`tolerance_change`) has a value of
> **1e-7** that is smaller than the machine unit ε for single precision floating point
> numbers."

Maschinenepsilon (Gl. 4-Kontext) [belegt: §5.2]: FP32 ε = **1.19e-7 > 1e-7**;
FP64 ε = **2.22e-16**. Zusätzlich wachsen die Gewichtsnormen auf Größenordnung 1e+1,
was die effektive Auflösung weiter vergröbert [belegt: §5.2 + Fig. 7(b)].

**Konsequenz für die Implementierung:** Der Confound sitzt konkret im
`torch.optim.LBFGS`-Parameter `tolerance_change` (Default 1e-9 in PyTorch, hier
offenbar 1e-7 verwendet) im Zusammenspiel mit dem Datentyp. Wer FP32 vs. FP64 sauber
trennen will, muss `tolerance_change` und `tolerance_grad` explizit protokollieren.

Drei-Phasen-Dynamik [belegt: §4.3]: (1) *Un-Converged* (Loss und Fehler hoch),
(2) *Failure* (Loss ≈ 0, Fehler hoch), (3) *Success* (beides ≈ 0). Failure Modes sind
demnach **Zwischenzustände**, keine unentrinnbaren lokalen Minima. Größere
PDE-Parameter verlängern die Failure-Phase [belegt: §4.3].

### 5.4 Ergebnisse

**Tab. 1** (rRMSE, Convection β = 50 / Reaction ρ = 5) [belegt: Tab. 1]:

| Modell | Convection rRMSE | Reaction rRMSE |
|---|---|---|
| PINN | 0.7640 ± 0.0694 | 0.9778 ± 0.0018 |
| QRes | 0.8184 ± 0.0382 | 0.9830 ± 0.0026 |
| PINNsFormer | 0.0435 ± 0.0073 | 0.0296 ± 0.0027 |
| KAN | 0.6985 ± 0.0701 | 0.0312 ± 0.0034 |
| PirateNet | 0.9740 ± 0.1894 | 0.0443 ± 0.0064 |
| RoPINN | 0.7204 ± 0.0941 | 0.0965 ± 0.0310 |
| PINNMamba | 0.0197 ± 0.0038 | 0.0213 ± 0.0036 |
| **PINN_FP64** | **0.0072 ± 0.0017** | **0.0502 ± 0.0111** |

**Tab. 1, Fortsetzung** (Wave / Allen-Cahn, rRMSE) [belegt: Tab. 1]:

| Modell | Wave rRMSE | Allen-Cahn rRMSE |
|---|---|---|
| PINN | 0.2837 ± 0.0571 | 0.9662 ± 0.0300 |
| QRes | 0.5273 ± 0.1172 | 0.9846 ± 0.0092 |
| PINNsFormer | 0.3571 ± 0.0872 | 0.9913 ± 0.0420 |
| KAN | 0.1489 ± 0.0357 | 0.5661 ± 0.0440 |
| PirateNet | 0.2637 ± 0.0480 | 0.1889 ± 0.0180 |
| RoPINN | 0.0642 ± 0.0238 | — |
| PINNMamba | 0.0195 ± 0.0033 | 0.2645 ± 0.0201 |
| **PINN_FP64** | **0.0081 ± 0.0031** | **0.0545 ± 0.0112** |

**Fig. 8(a) — Präzisionssweep über β (Convection, rMAE)** [belegt: Fig. 8(a)]:

| β | BF16 | FP16 | TF32 | FP32 | FP64 |
|---|---|---|---|---|---|
| 1 | 0.8894 | 0.4322 | 0.0649 | 0.0103 | 0.0021 |
| 5 | 1.2670 | 0.6794 | 0.2644 | 0.0237 | 0.0037 |
| 10 | NaN | 0.9870 | 0.3133 | 0.0531 | 0.0043 |
| 30 | NaN | 0.9928 | 0.5811 | 0.4923 | 0.0047 |
| 50 | NaN | NaN | 0.7910 | 0.6904 | 0.0059 |
| 100 | NaN | NaN | 0.9910 | 0.8933 | 0.0192 |

**Fig. 8(b) — Modelle × Präzision bei β = 50 (rMAE)** [belegt: Fig. 8(b)]:

| Modell | BF16 | FP16 | TF32 | FP32 | FP64 |
|---|---|---|---|---|---|
| PINN | NaN | NaN | 0.7910 | 0.6904 | 0.0059 |
| QRes | NaN | 0.7870 | 0.7235 | 0.7498 | 0.0179 |
| PINNsFormer | NaN | 0.9128 | 0.4233 | 0.0231 | 0.0087 |
| KAN | 1.2789 | 0.9901 | 0.5231 | 0.6723 | 0.0247 |
| PirateNet | 1.1514 | 1.1699 | 0.8210 | 0.5881 | 0.0759 |
| PINNMamba | NaN | NaN | 0.2410 | 0.0188 | 0.0042 |

**Das ist die direkteste empirische Stütze der Projekt-H0:** Bei FP64 liegen alle
Architekturen zwischen 0.0042 und 0.0759, das Vanilla-PINN (0.0059) schlägt
PINNsFormer (0.0087) und PirateNet (0.0759). Bei FP32 dagegen ist der
Architekturabstand über eine Größenordnung (PINN 0.6904 vs. PINNsFormer 0.0231).
Der Architekturvorteil ist in dieser Tabelle **präzisionsabhängig und kehrt bei FP64
teilweise das Vorzeichen um.**

Laufzeit-Overhead FP64 vs. FP32: **1.1–1.3×** pro Iteration [belegt: §5, Text zur
Laufzeittabelle]. Aus derselben Tabelle: FP32 1609/1629/2295/1975 vs. FP64
2441/2481/3845/3167 (Convection/Reaction/Wave/Allen-Cahn; **Einheit in der
Textextraktion nicht eindeutig — vermutlich Sekunden Gesamtlaufzeit** —
[unbelegt: Einheit]).

---

## 6. Andersen & Matsubara — "PINNs Failure Modes are Overfitting", arXiv 2605.30910

**Existenz verifiziert.** Die im Projektauftrag genannte ID **stimmt**: arXiv
2605.30910, eingereicht **Freitag, 29. Mai 2026**, Autoren Nigel T. Andersen und
Takashi Matsubara. (2605 = Mai 2026 ist beim heutigen Datum 2026-08-20 konsistent.)
[belegt: arXiv-Abs-Seite + PDF-Volltext]

### 6.1 Double Backprop — exakte Definition (Gl. 5)

Basis-Loss (Gl. 2) [belegt: §3.1, Gl. 2]:

```
min_θ L(u_θ) = λ_F L_F(u_θ) + Σ_j^{N_B} λ_{B_j} L_{B_j}(u_θ) + Σ_j^{N_I} λ_{I_j} L_{I_j}(u_θ)
```

mit (Gl. 3, 4) [belegt: §3.1, Gl. 3–4]:

```
L_F(u_θ) = (1/n_d) Σ_{k=1..n_d} | F(u_θ(x_k^d, t_k^d)) |²
L_B(u_θ) = (1/n_b) Σ_{k=1..n_b} | B(u_θ(x_k^b, t_k^b)) |²
L_I(u_θ) = (1/n_i) Σ_{k=1..n_i} | I(u_θ(x_k^i, t_k^i)) |²
```

und `λ_F = λ_B = λ_I = 1` [belegt: §3.1, Text nach Gl. 4: „We follow the original
PINNs formulation and set λ_F = λ_B = λ_I = 1"].

**„double PINN" (Gl. 5)** — die reproduzierbare Formel [belegt: §3.2, Gl. 5]:

```
L(u_θ) = L_F(u_θ) + (λ_r/2) ‖ ∇_{x,t} F ‖²
       + Σ_j^{N_B} [ L_{B_j}(u_θ) + (λ_r/2) ‖ ∇_t B_j ‖² ]
       + Σ_j^{N_I} [ L_{I_j}(u_θ) + (λ_r/2) ‖ ∇_x I_j ‖² ]
```

Präzise gelesen:

- Der Gradient-Penalty steht **auf den Residuen selbst**, nicht auf dem Loss und nicht
  auf dem Netzausgang.
- **Domänen-Residuum F:** Penalty auf `∇_{x,t} F` — Gradient **nach beiden**
  Eingangsvariablen x und t.
- **Randbedingungs-Residuum B_j:** Penalty auf `∇_t B_j` — **nur nach t**
  (x ist auf dem Rand fixiert, dort gibt es keine freie x-Richtung).
- **Anfangsbedingungs-Residuum I_j:** Penalty auf `∇_x I_j` — **nur nach x**
  (t = 0 ist fixiert).
- Vorfaktor jeweils **λ_r / 2**, derselbe λ_r für alle drei Terme.
- Erklärung des Autors [belegt: §3.2, Text nach Gl. 5]: „Double-backpropagation
  regularizes the gradient of the loss with respect to the network inputs, and double
  PINN extends this to each PINN loss. By flattening the residuals, double PINN
  penalizes the network for reducing the losses at the collocation points at the
  expense of the surrounding region."

⚠️ **Der numerische Wert von λ_r steht nirgends im Paper.** Volltextsuche über die
gesamte PDF nach `λ_r`, `lambda_r`, „regularization strength", „1e-", „0.01", „0.001"
liefert keinen Treffer, der λ_r beziffert. Auch Appendix A („Experimental Setup") nennt
ihn nicht. Der einzige mit einem Symbol versehene Regularisierungsparameter der
Vergleichsverfahren in §5.1 ist `r` (`L = L_0 + (r/2)‖w‖²₂` bzw. `L = L_0 + r‖w‖₁`) —
auch dort **ohne** Zahlenwert. **[unbelegt: λ_r-Wert und r-Wert]** → Für das Projekt
ist λ_r ein zu kalibrierender Hyperparameter; das muss in `DEVIATIONS.md` bzw. der
Präregistrierung als bewusste Abweichung vermerkt werden.

**Rechenkosten** [belegt: §4.3 Schlussabsatz]: Double Backprop verdoppelt theoretisch
die Kosten, weil zusätzliche Gradienten nach den Eingängen gebildet werden müssen.
Praktisch wird das durch schnellere Konvergenz und deutlich weniger Punkte
überkompensiert.

### 6.2 PDEs (Appendix B)

**B.1 Convection** (Gl. 6–9) [belegt: §B.1]:

```
∂u/∂t + β ∂u/∂x = 0,   x ∈ [0, 2π],  t ∈ [0, 1]
u(x, 0) = sin(x)
u(0, t) = u(2π, t)
u_exact = sin(x − βt)
β = 50  („a common choice in the failure mode literature is β = 50, which we use here")
```

**B.2 Wave** (Gl. 10–14) [belegt: §B.2]:

```
∂²u/∂t² − 4 ∂²u/∂x² = 0,   x ∈ [0, 1],  t ∈ [0, 1]
u(x, 0) = sin(πx) + ½ sin(βπx)
∂u/∂t(x, 0) = 0
u(0, t) = u(1, t) = 0
β = 3  („We use β = 3 here as is standard")
u_exact = sin(πx) cos(2πt) + ½ sin(βπx) cos(2βπt)
```

**B.3 Reaction** (Gl. 15–18) [belegt: §B.3]:

```
∂u/∂t − ρ u(1 − u) = 0,   x ∈ [0, 2π],  t ∈ [0, 1]
u(x, 0) = exp( −(x − π)² / (2 (π/4)²) )
u(0, t) = u(2π, t)
u_exact = h(x) e^{ρt} / ( h(x)(e^{ρt} − 1) + 1 )
ρ = 5  („a common choice for the ρ, the growth-rate coefficient, is ρ = 5")
```

**B.4 Allen-Cahn** (Gl. 19–22) [belegt: §B.4]:

```
∂u/∂t − 0.0001 ∂²u/∂x² + 5u³ − 5u = 0,   x ∈ [−1, 1],  t ∈ [0, 1]
u(x, 0) = x² cos(πx)
u(−1, t) = u(1, t)
∂u/∂x(−1, t) = ∂u/∂x(1, t)
```

„The Allen-Cahn equation above is the only PDE we consider that does not have an exact
solution. Instead, for comparison, we use a high-resolution solution from a spectral
solver, obtained from the source code of Ref. [20]" — Ref. [20] ist die FP64-Arbeit
[belegt: §B.4].

### 6.3 Setup (§3 + Appendix A)

[belegt: §3, S. 3 und §A.1]

- **Architektur:** 4 Hidden Layers × **128** Neuronen, **tanh**, **Glorot-normal**-Init.
  Eingänge normalisiert auf **[−1, 1]**.
- **Parameterzahl:** nicht angegeben. **[unbelegt/geschätzt]** (eigene Rechnung: ≈ 50 k)
- **Optimierer:** **L-BFGS** (Optax-Implementierung in JAX/Flax NNX), `memory = 100`
  (angeglichen an PyTorch-Default, statt Optax-Default 10), sonst Optax-Defaults.
- **Iterationen:** **10⁶** L-BFGS-Iterationen für die Benchmark-Läufe in Tab. 1
  [belegt: §4.3: „training for 10⁶ L-BFGS iterations"]. Fig. 6 zeigt Läufe über
  ~25 000 Iterationen für den Regularisierer-Vergleich.
- **Stoppkriterium:** relative Loss-Differenz über je 100 Iterationen, verglichen mit
  √(Maschinenepsilon) → **3.453 × 10⁻⁴ bei FP32**, **1.490 × 10⁻⁸ bei FP64**
  [belegt: §A.1].
- **Kollokationspunkte:** **Gitter** über die Domäne, linear verteilte Punkte auf Rand
  und Anfangsbedingung („we use a grid collocation method over the domain, and linearly
  spaced points on the boundaries and initial conditions") [belegt: §3].
- **Punktzahlen für Tab. 1** [belegt: §4.3]: Domänen-Gitterpunkte **400** (Convection),
  **256** (Reaction), **144** (Wave), **4096** (Allen-Cahn). Randbedingung und
  Anfangsbedingung: **200** je Bedingung bei Convection, **100** je Bedingung bei
  Reaction/Wave/Allen-Cahn. Gesamt N: **800 / 456 / 444 / 4396**.
- **Testset:** 10 000 zufällige Punkte in der Domäne + je 10 000 linear verteilte auf
  Rand und IC [belegt: §A.1].
- Hardware: FP32 auf Titan RTX, FP64 auf A100 [belegt: §3].

### 6.4 Ergebnisse (Tab. 1)

[belegt: Tab. 1, S. 7; **Layout-Rekonstruktion** — in der pdftotext-Ausgabe sind die
Zeilenlabels um eine Zeile gegen die Zahlenspalten verschoben. Die hier angegebene
Zuordnung ist die einzige, bei der „double PINN (Ours)" jeweils den besten Wert und
das kleinste N hat, und sie ist für die PINN_FP64-Zeilen unabhängig gegen Tab. 1 von
FP64 (§5.4 oben) geprüft und identisch.]

| PDE | Modell | Loss | rMAE | rRMSE | N |
|---|---|---|---|---|---|
| Convection | PINNsFormer | 9.0e-4 ± 1.0e-4 | 3.3e-2 ± 6.8e-3 | 4.4e-2 ± 7.3e-3 | 10403 |
| Convection | PINNMamba | 1.0e-4 ± 2.0e-5 | 1.8e-2 ± 3.7e-3 | 2.0e-2 ± 3.8e-3 | 10403 |
| Convection | PINN_FP64 | 5.0e-6 ± 1.0e-6 | 5.9e-3 ± 1.3e-3 | 7.2e-3 ± 1.7e-3 | 10403 |
| Convection | **double PINN** | 4.0e-11 ± 4.1e-11 | 4.9e-6 ± 4.4e-6 | **5.4e-6 ± 4.6e-6** | **800** |
| Reaction | PINNsFormer | 3.0e-6 ± 1.0e-6 | 1.5e-2 ± 1.3e-3 | 3.0e-2 ± 2.7e-3 | 10403 |
| Reaction | PINNMamba | 1.0e-6 ± 1.0e-6 | 9.2e-3 ± 1.7e-3 | 2.1e-2 ± 3.6e-3 | 10403 |
| Reaction | PINN_FP64 | 1.0e-5 ± 5.0e-6 | 2.7e-2 ± 6.3e-3 | 5.0e-2 ± 1.1e-2 | 10403 |
| Reaction | **double PINN** | 3.5e-12 ± 3.5e-13 | 1.1e-5 ± 2.3e-6 | **2.9e-5 ± 8.0e-6** | **456** |
| Wave | PINNsFormer | 2.3e-2 ± 1.7e-3 | 3.5e-1 ± 8.7e-2 | 3.6e-1 ± 8.7e-2 | 10504 |
| Wave | PINNMamba | 2.0e-4 ± 3.0e-5 | 1.9e-2 ± 3.3e-3 | 2.0e-2 ± 3.3e-3 | 10504 |
| Wave | PINN_FP64 | 4.2e-5 ± 1.6e-5 | 8.0e-3 ± 3.2e-3 | 8.1e-3 ± 3.1e-3 | 10504 |
| Wave | **double PINN** | 3.5e-7 ± 8.9e-8 | 3.8e-4 ± 1.0e-4 | **3.8e-4 ± 1.0e-4** | **444** |
| Allen-Cahn | PINNsFormer | 4.6e-1 ± 2.9e-1 | 9.9e-1 ± 4.0e-2 | 9.9e-1 ± 4.2e-2 | 10504 |
| Allen-Cahn | PINNMamba | 2.7e-3 ± 2.0e-4 | 1.4e-1 ± 1.2e-2 | 2.7e-1 ± 2.0e-2 | 10504 |
| Allen-Cahn | PINN_FP64 | 1.3e-5 ± 4.0e-6 | 1.6e-2 ± 3.6e-3 | 5.5e-2 ± 1.1e-2 | 10504 |
| Allen-Cahn | **double PINN** | 4.0e-5 ± 6.1e-10 | 9.6e-5 ± 2.0e-5 | **3.6e-4 ± 9.8e-5** | **4396** |

Fußnote 3 der Tabelle: „Loss is not directly comparable across the methods due to the
regularization penalty" [belegt: Tab. 1, Fußnote 3]. **Für das Projekt heißt das: der
finale Residual-Loss darf zwischen `none` und `double_backprop` nicht direkt verglichen
werden — nur der rel. L2 gegen die Referenzlösung.** Das betrifft die
Sekundärmetrik „finaler Residual-Loss" in §4 der Präregistrierung.

### 6.5 Regularisierer-Vergleich (§5.1, Fig. 6)

Verglichen auf Convection β = 50 bei **FP32** [belegt: §5.1, Fig. 6]:

- **unregularisiert:** Failure durch Overfitting.
- **double PINN:** Erfolg nach „a little over 5000 iterations of L-BFGS optimization",
  niedrigster Loss, schnellste Konvergenz.
- **L2 (Weight Decay):** Erfolg, aber mehr Iterationen nötig und höherer Test-Loss.
- **L1:** vermeidet zwar das Overfitting-Failure, ist aber schwer zu optimieren →
  großer Fehler; „the sparsity enforced by L1 reduces the expressiveness of the network
  too much".

Begründung des Autors für die Überlegenheit von double PINN [belegt: §5.1]: „it does
not limit the network weights directly, but rather acts only on the residual and so
does not limit the expressivity of the network."

**Direkt relevant:** Die Regularisierungsstufen `weight_decay` und `double_backprop`
der Präregistrierung entsprechen genau den zwei erfolgreichen Varianten aus Fig. 6 —
die Stufenwahl ist damit literaturgestützt.

---

## 7. Wang, Sankaran, Wang, Perdikaris — "An Expert's Guide to Training Physics-Informed Neural Networks", arXiv 2308.08468

### 7.1 Allen-Cahn (§7.1)

PDE (Gl. 7.1–7.4) [belegt: §7.1, Gl. 7.1–7.4]:

```
u_t − 0.0001 u_xx + 5u³ − 5u = 0,   t ∈ [0, 1],  x ∈ [−1, 1]
u(0, x) = x² cos(πx)
u(t, −1) = u(t, 1)
u_x(t, −1) = u_x(t, 1)
```

Identisch zu FP64 §A.4 und AM26 §B.4 — die drei Quellen stimmen hier **exakt** überein.

Bester berichteter Fehler: rel. L2 = **5.37 × 10⁻⁵** [belegt: Tab. 1 und Caption Fig. 5].

Hyperparameter (Tab. 6) [belegt: Tab. 6, S. 31; **Layout-Rekonstruktion** — die
Wertespalte ist gegen die Namensspalte um eine Zeile verschoben; die hier angegebene
Zuordnung ist die einzige typkonsistente]:

| Parameter | Wert |
|---|---|
| Architecture | Modified MLP |
| Number of layers | 4 |
| Layer size | 256 |
| Activation | Tanh |
| Fourier feature scale | 2.0 |
| RWF | μ = 0.5, σ = 0.1 |
| Learning rate | 0.001 |
| Decay steps | 5 000 |
| Training steps | 300 000 |
| Batch size | 8 192 |
| Weighting scheme | NTK |
| Causal tolerance | 1.0 |
| Number of chunks | 32 |

### 7.2 Advection / Convection (§7.2)

PDE (Gl. 7.5–7.6) [belegt: §7.2, Gl. 7.5–7.6]:

```
u_t + c u_x = 0,   t ∈ [0, 1],  x ∈ (0, 2π)
u(0, x) = g(x)
periodische Randbedingungen
```

Konkret [belegt: §7.2, Text]: „we consider the challenging setting of **c = 80** with
an initial condition **g(x) = sin(x)**".

Bester berichteter Fehler: rel. L2 = **6.88 × 10⁻⁴** [belegt: Tab. 1].

Hyperparameter (Tab. 7) [belegt: Tab. 7, S. 31; gleiche Layout-Rekonstruktion]:
Modified MLP, 4 Layer, 256 Neuronen, Tanh, Fourier feature scale **1.0**,
RWF μ = 1.0 / σ = 0.1, LR 0.001, Decay steps **2 000**, Training steps **200 000**,
Batch size 8 192, Weighting scheme **Grad Norm**, Causal tolerance 1.0, 32 Chunks.

### 7.3 Allgemeines Setup

[belegt: §7, Absatz S. 10]

- Backbone in allen Ablationen: **MLP mit 4 Hidden Layers, 256 Neuronen, tanh**,
  Glorot-Initialisierung.
- Optimierer: **Adam**, Start-LR **1e-3**, **exponentieller Decay**. (Der genaue
  Decay-Rate-Wert wurde in der Textextraktion abgeschnitten. **[unbelegt: decay rate]**)
- **Kein L-BFGS** in der Standard-Pipeline — das ist der auffälligste Unterschied zur
  Failure-Mode-Linie (KP21, PINNsformer, FP64, AM26), die durchweg L-BFGS verwendet.
- Weitere Benchmarks (Tab. 1) [belegt: Tab. 1]: Stokes flow 8.04e-5,
  Kuramoto–Sivashinsky 1.61e-1, Lid-driven cavity (Re = 3200) 1.58e-1,
  Navier-Stokes im Torus 2.45e-1.

Referenz [47] im Literaturverzeichnis ist **Wight & Zhao, „Solving Allen-Cahn and
Cahn-Hilliard equations using the adaptive physics informed neural networks"**
[belegt: Referenzliste, Eintrag 47] — d.h. EG23 selbst verweist für Allen-Cahn auf
Wight & Zhao als Vorläufer.

---

## Entscheidungsrelevante Befunde

### A) Welche Gleichungen stehen wirklich in Krishnapriyan et al.?

**Die Vermutung der Projektleitung ist bestätigt, mit einer Präzisierung.**

KP21 enthält:

| Gleichung | in KP21? | Fundstelle |
|---|---|---|
| Convection | **ja** | §3.1, Gl. 5 |
| Reaction | **ja** | Appendix A, Gl. 14 |
| Reaction-Diffusion | **ja** | §3.2, Gl. 10 |
| Wave | **nein** | Volltext-Grep: 0 Treffer für „wave"/„Wave" |
| Allen-Cahn | **nein** | Volltext-Grep: 0 Treffer für „Allen"/„Cahn" |
| Burgers, Helmholtz | **nein** | 0 Treffer |

Präzisierung gegenüber der Arbeitsannahme in der Präregistrierung: **Reaction ist bei
KP21 in Appendix A, nicht im Haupttext.** Der Haupttext behandelt Convection (§3.1) und
Reaction-Diffusion (§3.2). Das ändert an der Sache nichts, sollte aber bei der
Zitation korrekt sein („Krishnapriyan et al. 2021, §A, Gl. 14").

**Die Entscheidung vom 2026-08-20 (Allen-Cahn → Reaction-Diffusion) ist damit
literaturgestützt korrekt.** Aber: Die Präregistrierung §3 listet „convection, reaction,
reaction_diffusion, wave" und behauptet „Alle vier Benchmark-PDEs stammen jetzt aus
einer Quelle." **Das ist so nicht haltbar — Wave steht nicht in KP21.** Nur drei der
vier stammen aus KP21.

Kanonische Quelle für die **Wave**-Parameter (β = 3, Koeffizient 4, x ∈ [0,1],
t ∈ [0,1], u(x,0) = sin(πx) + ½sin(3πx), u_t(x,0) = 0, Dirichlet):
→ **PINNsformer (Zhao et al., ICLR 2024), Gl. 11**, wo die Gleichung in dieser
Benchmark-Form auftaucht; von FP64 (§A.3, Gl. 9) und AM26 (§B.2, Gl. 10) übernommen.
Die Präregistrierung sollte für Wave also **PINNsformer** zitieren, nicht KP21.
Zusätzlich: die Präregistrierung schreibt „u_tt − β² u_xx = 0 mit dem in der Quelle
angegebenen β" — **das ist falsch.** Die Literaturform ist `u_tt − 4 u_xx = 0` mit
β = 3 **in der Anfangsbedingung**, nicht als Koeffizient. β² = 9 ≠ 4.

Kanonische Quelle für **Allen-Cahn**-Parameter (falls das Projekt sie doch je braucht),
in Prioritätsreihenfolge der Belegdichte:

1. **Wang, Sankaran, Wang, Perdikaris, „An Expert's Guide…", arXiv 2308.08468, §7.1,
   Gl. 7.1–7.4** — vollständigste Angabe inkl. Hyperparametertabelle (Tab. 6).
   Parameter: `u_t − 0.0001 u_xx + 5u³ − 5u = 0`, x ∈ [−1,1], t ∈ [0,1],
   `u(0,x) = x² cos(πx)`, periodisch in u und u_x.
2. Identisch bei FP64 §A.4 (Gl. 11) und AM26 §B.4 (Gl. 19–22). **Alle drei Quellen
   stimmen bei Allen-Cahn exakt überein** — kein Widerspruch.
3. **Wight & Zhao** wird von EG23 als Ref. [47] als Vorläufer geführt, wurde hier
   aber **nicht im Volltext gelesen**. Wer die Urheberschaft der Parameter belegen
   will, muss dort nachschlagen.
4. **Raissi et al. (Original-PINN)** — dass Allen-Cahn dort in dieser Standardform
   steht, ist **hier nicht verifiziert**.

Die Referenzlösung für Allen-Cahn ist in **allen** Quellen eine spektrale
Hochauflösungslösung, keine analytische Formel [belegt: FP64 §A.4, AM26 §B.4].

### B) Convection im Failure-Regime: welches β genau?

**Kurzantwort:** KP21 testet β = 1 … 70; das Vanilla-PINN scheitert **ab β > 10**. Die
neuere Literatur (PINNsformer, FP64, AM26) hat sich auf **β = 50** als
Standard-Failure-Punkt geeinigt. EG23 verwendet **c = 80**.

**Aus KP21:**

- Getesteter Bereich: Fig. 1(a) zeigt einen β-Sweep; Fig. C.1 zeigt Heatmaps für
  β = 30, 50, 70 [belegt: Captions Fig. C.1(d)–(l)].
- Failure-Schwelle im Klartext [belegt: §3.1, Observations]: „it fails when β becomes
  larger, **reaching a relative error of almost 100 % for β > 10**."
- Tabellierte Fehler (Tab. 1, reguläres PINN, rel. Fehler):

  | β | rel. Fehler (Regular PINN) | rel. Fehler (Curriculum) |
  |---|---|---|
  | 20 | 7.50 × 10⁻¹ | 9.84 × 10⁻³ |
  | 30 | 8.97 × 10⁻¹ | 2.02 × 10⁻² |
  | 40 | 9.61 × 10⁻¹ | 5.33 × 10⁻² |

- Tab. E.1 (mit 10 000 Punkten statt 1000): β = 30 → 7.38 × 10⁻¹, β = 40 → 8.25 × 10⁻¹.
  Mehr Kollokationspunkte helfen also **nur marginal**.

**Aus FP64 (Fig. 8a), rMAE, quantitativ feiner aufgelöst:**

| β | FP32 | FP64 |
|---|---|---|
| 1 | 0.0103 | 0.0021 |
| 5 | 0.0237 | 0.0037 |
| 10 | 0.0531 | 0.0043 |
| 30 | 0.4923 | 0.0047 |
| 50 | 0.6904 | 0.0059 |
| 100 | 0.8933 | 0.0192 |

Der Bruch liegt bei FP32 **zwischen β = 10 und β = 30** (0.053 → 0.492) — konsistent
mit KP21s „β > 10". Bei **FP64 gibt es diesen Bruch nicht** (0.0043 → 0.0047).
**Das ist die zentrale Zahl für die Nullhypothese des Projekts.**

**Empfehlung für die Präregistrierung:** Die Arbeitsannahme „β ≈ 30–40" liegt im
Failure-Regime, aber **β = 50** ist der Wert, gegen den sich alle drei
Vergleichsarbeiten (PINNsformer, FP64, AM26) messen lassen. Mit β = 50 sind die eigenen
Zahlen direkt gegen PINNsformer Tab. 1, FP64 Tab. 1 und AM26 Tab. 1 vergleichbar; mit
β = 30 oder 40 nur gegen KP21 Tab. 1. **Wenn Anschlussfähigkeit an die
Baseline-Tabellen gewünscht ist: β = 50.**

Erwartungswert bei β = 50, Vanilla-PINN, FP32: rel. L2 ≈ 0.76–0.84
[belegt: FP64 Tab. 1 rRMSE 0.7640; PINNsformer Tab. 1 rRMSE 0.840]. Das liegt weit
über der Erfolgsschwelle 0.10 → **Vanilla-MLP-Convection-fp32-none wird planmäßig
0 % Success Rate haben.** Bei FP64 dagegen rRMSE 0.0072 [belegt: FP64 Tab. 1] →
**100 % Success Rate.** Die Präzisions-Hauptwirkung wird also mit hoher
Wahrscheinlichkeit den Backbone-Haupteffekt dominieren.

### C) Double Backprop: exakte Definition

Quelle: **Andersen & Matsubara, arXiv 2605.30910, §3.2, Gl. 5** [belegt].

```
L(u_θ) =  L_F(u_θ)  +  (λ_r/2) ‖ ∇_{x,t} F ‖²
        + Σ_j^{N_B} [ L_{B_j}(u_θ) + (λ_r/2) ‖ ∇_t   B_j ‖² ]
        + Σ_j^{N_I} [ L_{I_j}(u_θ) + (λ_r/2) ‖ ∇_x   I_j ‖² ]
```

Als Implementierungsvorschrift:

1. Berechne die Residuen **punktweise**, nicht als Mittelwert:
   `F_k = F(u_θ(x_k, t_k))` für alle n_d Domänenpunkte,
   `B_k = B(u_θ(x_k, t_k))` auf dem Rand,
   `I_k = I(u_θ(x_k, 0))` auf der IC.
2. Die Basis-Losses sind die Mittelwerte der Quadrate (Gl. 3, 4).
3. Der Penalty ist die **quadrierte Norm des Gradienten des Residuums nach den
   Netz-Eingängen**:
   - Domäne: `∇_{x,t} F` → beide Eingänge, also `autograd.grad(F, [x, t])`.
   - Rand: `∇_t B_j` → **nur t**.
   - IC: `∇_x I_j` → **nur x**.
4. Vorfaktor λ_r/2, **identisch für alle drei Terme**.
5. Die Basis-Gewichte bleiben λ_F = λ_B = λ_I = **1** [belegt: §3.1].

**Nicht** der Loss selbst wird differenziert, **nicht** der Netzausgang — sondern das
Residuum. Das ist der Unterschied zum klassischen Drucker-&-LeCun-Double-Backprop
(dort: Gradient des Loss nach den Inputs); AM26 nennt seine Version explizit eine
**Erweiterung** davon auf die volle Menge der PINN-Residuen [belegt: §3.2, Abstract
„we extend double backpropagation over the full set of residuals"].

**Der Wert von λ_r ist im Paper nicht angegeben** (Volltextsuche negativ, auch nicht
in Appendix A). **[unbelegt: λ_r]** → Für das Projekt: λ_r muss in M2 kalibriert
werden; die Wahl ist eine dokumentationspflichtige Abweichung. Empfehlung: einen
kleinen Grid (z.B. λ_r ∈ {1e-4, 1e-3, 1e-2, 1e-1}) auf **einer** Kalibrierzelle, dann
denselben Wert für alle Bedingungen einfrieren — sonst verletzt man die
Fairness-Auflage „kein architekturspezifisches Tuning".

**Kostenwarnung:** Double Backprop verdoppelt die Gradientenrechnung
[belegt: §4.3 AM26]. Bei drei Regularisierungsstufen schlägt das direkt auf das
GPU-Budget durch — für `bench/plan.py` einplanen.

**Metrikwarnung:** AM26 Tab. 1 Fußnote 3 — der Loss ist zwischen regularisierten und
unregularisierten Läufen **nicht vergleichbar**. Die Sekundärmetrik „finaler
Residual-Loss und Randbedingungs-Loss getrennt" aus §4 der Präregistrierung muss
deshalb entweder auf den **unregularisierten** Loss-Anteil eingeschränkt oder als
zwischen Regularisierungsstufen nicht vergleichbar gekennzeichnet werden.

---

## Widersprüche

### W-1: Convection-β — KP21 vs. spätere Literatur

| Quelle | β | Fundstelle |
|---|---|---|
| KP21 | Sweep 1…70; tabelliert 20, 30, 40; Heatmaps 30, 50, 70 | Tab. 1, Tab. E.1, Fig. C.1 |
| PINNsformer | **50** | Gl. 8 |
| FP64 | **50** („Following prevailing practice [36, 33]") | §A.1 |
| AM26 | **50** („a common choice in the failure mode literature") | §B.1 |
| EG23 | **c = 80** | §7.2 |

Das ist **kein** echter Widerspruch in der PDE, sondern eine unterschiedliche Wahl des
Failure-Punkts auf demselben Sweep. Trotzdem sind die Fehlerzahlen zwischen KP21 (β=40)
und PINNsformer/FP64/AM26 (β=50) **nicht direkt vergleichbar**. Wer β = 30 oder 40
wählt, kann sich nicht gegen die Tabellen von PINNsformer/FP64/AM26 messen.

### W-2: Analytische Lösung der Convection-Gleichung

- KP21 Gl. 6 gibt sie **spektral**: `u = F^{-1}( F(h(x)) e^{-iβkt} )`.
- FP64 Gl. 6 und AM26 Gl. 9 geben sie **geschlossen**: `u = sin(x − βt)`.

Mathematisch äquivalent für h(x) = sin(x) auf periodischer Domäne, aber
**numerisch nicht identisch**: die FFT-Version akkumuliert Rundungsfehler, die
geschlossene Form nicht. **Für ein Projekt, dessen Nullhypothese die numerische
Präzision ist, ist das relevant.** Empfehlung: `sin(x − βt)` als Referenz verwenden
und das in DEVIATIONS.md als bewusste Wahl gegen KP21 Gl. 6 dokumentieren.

### W-3: Wave-Gleichung — Koeffizient β vs. 4

- **PINNsformer Gl. 11:** `∂²u/∂t² − β ∂²u/∂x² = 0` mit **β = 3**, IC
  `sin(πx) + ½ sin(βπx)`, Lösung `sin(πx)cos(2πt) + ½ sin(βπx)cos(2βπt)`.
- **FP64 §A.3 Gl. 9** und **AM26 §B.2 Gl. 10:** `∂²u/∂t² − 4 ∂²u/∂x² = 0`,
  β = 3 **nur** in der IC, gleiche Lösung.

**Das ist ein echter Widerspruch in der Gleichung.** Die angegebene Lösung ist mit
Koeffizient 4 (Wellengeschwindigkeit c = 2) konsistent, **nicht** mit 3. PINNsformer
verwendet dasselbe Symbol β doppelt und schreibt es dadurch in die PDE hinein.
**Nicht stillschweigend auflösen** — aber festhalten: FP64 und AM26 sind die
konsistente Fassung, PINNsformer die inkonsistente. Beide Fundstellen sind oben
wörtlich wiedergegeben.

Die Präregistrierung §3 („u_tt − β² u_xx = 0 mit dem in der Quelle angegebenen β") ist
**mit keiner der beiden Fassungen** identisch (β² = 9). Muss korrigiert werden.

### W-4: Aktivierungsfunktion des Baseline-PINN

- **PINNsformer Tab. 4:** Baseline-PINN mit **ReLU**, 4 Layer × 512, 527 k Parameter.
- **KP21 §3:** **tanh**, 4 Layer × 50.
- **AM26 §3/§A.1:** **tanh**, 4 Layer × 128.
- **EG23 §7:** **tanh**, 4 Layer × 256.
- **FP64 §3.2:** 3 Layer × 512, **Aktivierung nicht genannt**.

ReLU hat verschwindende zweite Ableitung und ist für PINN-Residuen mit u_xx
grundsätzlich ungeeignet. Die PINNsformer-Baseline ist damit möglicherweise
**geschwächt**, was ihre Baseline-Fehlerzahlen (Convection 0.840) gegenüber FP64s
eigener Vanilla-PINN-Messung (0.7640) nach oben verzerren könnte.
**Widerspruch dokumentiert, nicht aufgelöst.** Für das Projekt: **tanh** wählen (3 von
4 Quellen, und die einzige physikalisch sinnvolle Wahl bei zweiter Ableitung) und die
Abweichung von PINNsformer Tab. 4 in DEVIATIONS.md vermerken.

### W-5: Netzbreite und Parameterzahl der Vanilla-Baseline

| Quelle | Layer × Breite | Params |
|---|---|---|
| KP21 | 4 × 50 | nicht angegeben |
| AM26 | 4 × 128 | nicht angegeben |
| EG23 | 4 × 256 | nicht angegeben |
| PINNsformer | 4 × 512 | 527 k |
| FP64 | 3 × 512 | nicht angegeben |

Eine Größenordnung Unterschied (≈ 7,8 k bis 527 k). Die Fairness-Auflage „Parameterzahl
±10 % über alle Backbones" ist damit **nicht** durch eine Literaturvorgabe bestimmt —
das Projekt muss selbst eine Zielgröße festlegen. PINNsformer (454–527 k) ist die
einzige Quelle, die Parameterzahlen überhaupt beziffert; wenn Anschlussfähigkeit an
PINNsformer gewünscht ist, wäre **≈ 500 k** die naheliegende Zielgröße.

### W-6: Optimierer-Regime

- Failure-Mode-Linie (KP21, PINNsformer, FP64, AM26): **durchgehend L-BFGS**, kein Adam.
  KP21 begründet das explizit in Fußnote 3.
- EG23: **ausschließlich Adam** mit exponentiellem Decay, 200 k–300 k Schritte,
  kein L-BFGS.

Kein Widerspruch in Fakten, aber eine **methodische Gabelung**. Die
Präregistrierung fordert „identisches Optimierer-Schema" für alle Bedingungen —
gut. Aber die Wahl entscheidet, gegen welche Literatur die Zahlen anschlussfähig
sind. **Wichtig:** Der Präzisions-Confound aus FP64 ist ein **L-BFGS-spezifischer**
Mechanismus (`tolerance_change` vs. Maschinenepsilon). **Mit Adam existiert dieser
Confound in dieser Form nicht.** Wer die Nullhypothese des Projekts testen will, muss
**L-BFGS** verwenden — sonst testet man H0 gar nicht.

### W-7: Kollokationspunkt-Anordnung

- **KP21 §3:** „randomly sample collocation points (x, t) on the domain" → **Sampling**.
- **PINNsformer:** 101 × 101 **Gitter**.
- **AM26 §3:** „a grid collocation method over the domain, and linearly spaced points on
  the boundaries" → **Gitter**.
- **EG23:** Batch size 8 192; Sampling-Schema in der Hyperparametertabelle nicht
  ausgewiesen.

Für β = 50 sind 101 × 101 (PINNsformer/FP64) und 400 + 400 (AM26) um mehr als eine
Größenordnung verschieden — und AM26 zeigt gerade, dass **weniger** Punkte bei
Regularisierung **besser** sind. Die Präregistrierungs-Auflage „identische
Kollokationspunkte für alle Bedingungen" ist deshalb richtig, aber die Wahl der
absoluten Zahl ist eine freie Entscheidung ohne Literaturvorgabe.
**[unbelegt: kanonische Punktzahl]**

---

## Nicht auffindbar / unbelegt

### Nicht auffindbare Quellen

**Keine.** Alle sieben im Auftrag genannten Quellen wurden gefunden und im Volltext
gelesen. Insbesondere:

- **„FP64 is All You Need"** — existiert als **arXiv 2505.10949** (v1 16.05.2025,
  v2 28.11.2025), Xu, Liu, Nassereldine, Xiong, University at Buffalo (SUNY).
  NeurIPS-2025-Posterseite vorhanden. Die Venue-Angabe des Projektauftrags ist
  bestätigt. Die im Auftrag nicht genannte arXiv-ID lautet 2505.10949.
- **„PINNs Failure Modes are Overfitting"** — die im Auftrag genannte ID
  **arXiv 2605.30910 ist korrekt**. Andersen & Matsubara, eingereicht 29. Mai 2026.
  Der Verdacht, die ID könnte erfunden sein, hat sich **nicht** bestätigt.

### Nicht im Volltext gelesen

- **Wight & Zhao**, „Solving Allen-Cahn and Cahn-Hilliard equations using the adaptive
  physics informed neural networks" — nur als Ref. [47] in EG23 gesichtet. Wer die
  Urheberschaft der Allen-Cahn-Standardparameter belegen will, muss dort nachschlagen.
- **Raissi, Perdikaris, Karniadakis** (Original-PINN) — nicht gelesen; die Behauptung,
  Allen-Cahn stünde dort in der Standardform, ist **hier nicht verifiziert**.
- **PINNMamba** (Ref. [22]/[35] in AM26/FP64) — als Vergleichsbaseline in beiden
  Tabellen enthalten, Originalarbeit nicht gelesen.
- GRAND- und GREAD-**Appendices** (Hyperparameter je Datensatz).

### Im jeweiligen Paper nicht angegebene Werte

| Größe | Quelle | Status |
|---|---|---|
| λ_r (Double-Backprop-Gewicht) | AM26 | **nicht im Paper** — Volltextsuche negativ, auch Appendix A |
| r (L2/L1-Regularisierungsstärke im Vergleich §5.1) | AM26 | **nicht im Paper** |
| Parameterzahl des Netzes | KP21, AM26, EG23, FP64 | nicht angegeben |
| Iterationszahl | KP21 | nicht angegeben |
| N_u, N_b (IC-/BC-Punktzahlen) | KP21 | nicht angegeben |
| N_f systematisch | KP21 | nur Fußnote 4 (1000) und Tab. E.1 (10 000) |
| numerischer λ-Wert des Residualgewichts | KP21 | nicht angegeben (nur qualitativ in §4) |
| Aktivierungsfunktion | FP64 | nicht angegeben |
| Kollokationspunktzahl | FP64 | nicht angegeben (10 201 nur via AM26 §4.3) |
| Exponential-Decay-Rate | EG23 | in der Textextraktion abgeschnitten |
| Einheit der Laufzeittabelle | FP64 | nicht eindeutig (vermutlich Sekunden) |
| Hidden-Dim, Integrationszeit T je Datensatz | GRAND | nicht extrahiert (Appendix) |
| T = 1 für Convection im Haupttext | KP21 | nur indirekt über Fußnote 4 |

### Layout-Rekonstruktionen (vor Implementierung im Original-PDF gegenprüfen)

Die folgenden Tabellen wurden aus `pdftotext -layout`-Ausgaben rekonstruiert, in denen
die Zeilenlabels gegen die Zahlenspalten verschoben waren. Die Zuordnung ist jeweils
begründet, aber nicht durch Bildansicht bestätigt:

- **KP21 Tab. 1** (β = 20/30/40) — Plausibilität: monoton steigender Fehler mit β.
- **KP21 Tab. 2** (ν = 2…6) — unabhängig bestätigt durch Haupttext (ν=2 → 50 %,
  ν=5 → 93 %). Hohe Sicherheit.
- **KP21 Tab. E.1, E.2** — Plausibilität: monoton steigender Fehler.
- **AM26 Tab. 1** — unabhängig bestätigt: die PINN_FP64-Zeilen sind mit FP64 Tab. 1
  wertidentisch. Hohe Sicherheit.
- **EG23 Tab. 6, 7** — Zuordnung typkonsistent (nur eine Zuordnung ergibt „Activation:
  Tanh" statt „Activation: 2.0").
