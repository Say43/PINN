# Referenzdaten

`allen_cahn.mat` ist die hochauflösende spektrale Allen–Cahn-Referenz aus dem
offiziellen Repository zu *FP64 is All You Need*:

- Quelle: https://github.com/miniHuiHui/PINN_FP64/blob/main/allen_cahn.mat
- Rohdatei: `t` 1×201, `x` 1×512, `usol` 201×512 (Zeit × Raum)
- SHA-256: `ce640f188e334520f636d3d650cae6056a92a486546d40889bff93610bdbfa71`
- Abgerufen: 2026-08-20

Die Trainingspipeline interpoliert diese unveränderte Quelldatei bilinear auf das
festgelegte Auswertungsgitter. Die Datei wird in das Kaggle-Code-Dataset aufgenommen,
weil das Notebook-Netzwerk deaktiviert bleibt.
