# V5: lokale technische Validierung (2026-09-18)

Status: `NOT_STUDY_DATA`. Die Präregistrierung ist noch ein Entwurf, und die
V5-Hauptmatrix wurde nicht ausgeführt.

Die frühere Graph-Implementierung ließ den Kontext vom jeweiligen
Anfragebatch abhängen. Damit stimmten automatische Ableitungen für den
PINN-Verlust nicht zuverlässig mit Differenzenquotienten der Vorhersage
überein. Die erste Variante mit hart wechselnden Nachbarn hatte trotz
festem Kontext sichtbare Vorhersagesprünge; die Rohmessung liegt in
`results/local_validation_hard_neighbors.json`. V5 verwendet jetzt einen
festen Support und eine glatte Distanzgewichtung. Auf drei neuen Seeds
(101–103) bestanden beide Graph-Backbones und der MLP alle drei lokalen
Prüfungen: Ableitungsabweichung, Unabhängigkeit von der Anfragebatchgröße
und Stetigkeit an Nachbarwechseln. Die größte relative Ableitungsabweichung
lag bei etwa 1.2e-9. Rohdaten: `results/local_validation.json`.

Die vollständige Testsuite bestand mit 51 Tests. Ein separater lokaler
CPU-Kostencheck mit 50 statt 2000 Iterationen schloss die drei MLP-Läufe
ab; alle sechs Graph-Läufe erreichten das 50-Sekunden-Diagnoselimit.
Diese neun Versuche messen keine Erfolgsquote und keine Rangfolge der
Architekturen. Rohdaten: `results/local_comparison.json`.

Die lokale Python-Umgebung enthält PyTorch 2.13.0 ohne CUDA-Unterstützung.
Die vorhandene 6-GB-GPU wurde deshalb für diesen Check nicht verwendet.
Für eine Aussage über die Machbarkeit der 60 Läufe muss die korrigierte
Graph-Implementierung auf der vorgesehenen Zielhardware profiliert werden.
Der V5-Budgetwert 7.4 GPU-Stunden ist derzeit nur eine historische Schätzung.
