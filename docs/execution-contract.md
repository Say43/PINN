# Ausführungs- und Persistenzvertrag

Dieser Vertrag konkretisiert die Präregistrierung, ohne wissenschaftliche
Entscheidungen zu verändern.

## Sicherheitsinvarianten

1. Studienläufe nutzen ausschließlich die in M2a gewählte Kaggle-Hardware.
2. Es läuft genau ein Einzellauf gleichzeitig. Damit kann ein Session-Kill höchstens
   den aktuellen Lauf vernichten.
3. Vor dem Training wird ein `running`-Attempt in SQLite angelegt und lokal per
   SQLite-Backup mit `integrity_check` gesichert.
4. Nach jedem terminalen Einzellauf werden Ergebnis, Historie, Wall-Clock,
   L-BFGS-Funktionsauswertungen, Solver-NFE, Parameterzahl und Fehlerstatus in einer
   Transaktion geschrieben.
5. Unmittelbar danach wird das geprüfte Backup als neue Version eines **privaten
   Kaggle Results Dataset** veröffentlicht. Schlägt die Veröffentlichung fehl,
   stoppt der Runner vor dem nächsten Lauf.
6. Beim Resume werden verwaiste `running`-Attempts als `interrupted` behalten;
   `completed` und `numerical_fail` werden nicht wiederholt. Nur Infrastrukturfehler
   dürfen mit identischem Seed einen neuen Attempt erhalten.

## Warum das Notebook Internetzugriff benötigt

Kaggle mountet Input-Datasets read-only; `/kaggle/working` wird erst bei einem
ordnungsgemäßen Notebook-Abschluss als Output gesichert. Die harte Vorgabe verlangt
jedoch eine dauerhafte Sicherung **nach jedem** Lauf. Daher wird Internet nur für
den authentifizierten Aufruf `kagglehub.dataset_upload(...)` aktiviert. Ohne diese
Verbindung kann die Persistenzgarantie nicht erfüllt werden und der Runner startet
keinen weiteren Lauf.

KaggleHub ist im Notebook standardmäßig authentifiziert. Das Results Dataset wird
privat angelegt und über `PINN_RESULTS_DATASET=<owner>/<slug>` adressiert. Es werden
nur das geprüfte `results.sqlite`, `quota_state.json`, der M2-Plan und ein kleines
Publikationsmanifest hochgeladen.

## Notebook-Topologie

Das generierte Notebook enthält ausschließlich einen Launcher. Der Projektcode wird
als versioniertes Kaggle Dataset unter `/kaggle/input` eingebunden und nicht in das
Notebook kopiert. M2a wird getrennt auf P100 und 2x T4 ausgeführt. Die GPU-Auswahl
erfolgt in den Kaggle-Kernel-Einstellungen; der Launcher prüft und protokolliert die
tatsächlich sichtbare Hardware.
