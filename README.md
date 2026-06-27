# Masked Autoencoder implémenté à la main

MAE réimplémenté entièrement à la main pour comprendre le modèle en détail : passage avant et
rétropropagation écrits à la main, sans autograd ni torch.nn (gradient check < 1e-6). torch sert
uniquement de bibliothèque de tableaux, pour utiliser le GPU. Pré-entraînement auto-supervisé,
puis classification de défauts de surface d'acier (NEU) par sonde linéaire. Étude de méthodologie
sur STL-10.

Projet du cours d'introduction à la computer vision.

## Fichiers

- `mae.py` : le modèle (couches, encodeur/décodeur, perte, Adam).
- `train.py` : données, pré-entraînement, sonde linéaire, reconstruction.
- `experiments.py` / `experiments_neu.py` : ablation STL-10 / cas NEU.
- `notebook.ipynb` : étapes 4 à 12, tests, entraînement, figures.
- `visualisation.ipynb` : démo, reconstruction et prédiction sur une image.
- `report/rapport.pdf` : le rapport.

## Lancer

```bash
python experiments_neu.py   # NEU, ~10 min CPU
python experiments.py       # ablation STL-10, ~1 h CPU
python make_figures.py      # figures du rapport
```

Données NEU dans `data/neu/` (sinon depuis figshare). STL-10 : `python scripts/datasetdwnld.py`.
