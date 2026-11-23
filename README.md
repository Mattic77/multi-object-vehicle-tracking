# Suivi multi-objets de véhicules

Détection de plusieurs objets dans une vidéo, conservation de leur identifiant, trajectoires, et comptage de ceux qui franchissent une ligne virtuelle. Le détecteur est YOLO11n, le suivi est ByteTrack.

## Installation

Depuis un clone propre, à la racine du dépôt :

```bash
pip install -r requirements.txt
```

Le premier lancement télécharge `yolo11n.pt` (environ 6 Mo).

## Vidéo

Le dépôt ne contient pas les vidéos. Un extrait public de circulation suffit :

```bash
# Linux / Colab
mkdir -p data
wget -O data/vehicles.mp4 https://media.roboflow.com/supervision/video-examples/vehicles.mp4
```

```powershell
# Windows
New-Item -ItemType Directory -Force -Path data
Invoke-WebRequest -Uri "https://media.roboflow.com/supervision/video-examples/vehicles.mp4" -OutFile "data/vehicles.mp4"
```

Cette vidéo dure 22 s en 3840×2160. Sur CPU, un extrait court est plus confortable.

## Lancer l'expérience

Les hyperparamètres sont dans `configs/default.yaml`. La ligne de commande les remplace.

```bash
python scripts/run_experiment.py --source data/vehicles.mp4 --weights yolo11n.pt --line 0.05,0.60,0.95,0.60 --device cpu --save-vid
```

| Argument | Rôle |
|---|---|
| `--source` | Vidéo d'entrée |
| `--weights` | Poids YOLO, par défaut `yolo11n.pt` |
| `--line` | Ligne virtuelle `x1,y1,x2,y2`. Des valeurs entre 0 et 1 sont des fractions de l'image |
| `--device` | `cpu`, `cuda`, ou omis pour le choix automatique |
| `--save-vid` / `--no-save-vid` | Écrire ou non la vidéo annotée |

Sorties :

- `outputs/annotated.mp4` — identifiants, trajectoires, ligne, compteur
- `outputs/benchmark.json` et `outputs/benchmark_breakdown.png` — vitesse
- `outputs/evaluation.json` et `outputs/errors/` — qualité du suivi

Avec une vérité terrain MOTChallenge :

```bash
python scripts/run_experiment.py --source data/sequence.mp4 --gt data/gt.txt --save-vid
```

Le rapport contient alors précision, rappel, MOTA, objets manqués et changements d'identité.

Vérifier que le projet tient debout :

```bash
python tests/test_smoke.py
```

## Google Colab

1. Ouvrir [le notebook](https://colab.research.google.com/github/Mattic77/multi-object-vehicle-tracking/blob/main/run_colab.ipynb).
2. **Exécution → Modifier le type d'exécution → T4 GPU**.
3. **Exécution → Tout exécuter**.

Le notebook clone ce dépôt, installe `requirements.txt`, télécharge la vidéo publique, puis appelle `scripts/run_experiment.py`. Le passage GPU écrit `/content/outputs/result.mp4`. Le passage suivant refait la même vidéo sur CPU pour comparer la vitesse, dans `result_cpu.mp4`. La cellule ffmpeg convertit la vidéo GPU en H.264 pour la lire dans la page.

`run_experiment.py` affiche `Colab: True | CUDA: True` quand le GPU est bien pris.

## Chaîne

`src/pipeline.py` enchaîne, pour chaque image :

1. **data_loader** — ouverture de la vidéo (`cv2.VideoCapture`)
2. **detector + tracker** — `ObjectTracker.track` (YOLO11n puis ByteTrack via Ultralytics `model.track`). Le module `ObjectDetector` est celui de la détection seule ; il n'est pas relancé à chaque image, sinon le réseau tournerait deux fois
3. **counter + overlay** — `LineCounter` et le dessin des boîtes, identifiants et trajectoires
4. **evaluator** — `TrackingEvaluator`
5. **benchmark** — `PerformanceMonitor`, JSON et graphique

## Contributions

| Partie | Fichiers | Auteur dans l'historique git |
|---|---|---|
| Structure, détection YOLO11n, suivi ByteTrack, comptage, premier notebook | `src/detector.py`, `src/tracker.py`, `src/counter.py`, base de `src/main.py`, `run_colab.ipynb` | Matiic77 |
| Mise à jour du notebook Colab | `run_colab.ipynb` | Younes HADLI |
| Benchmark et évaluation du suivi (membre 3) | `src/metrics.py`, `src/evaluator.py` | nkrika |
| Orchestration, CLI, config, test de bout en bout, README (membre 4) | `src/pipeline.py`, `scripts/run_experiment.py`, `configs/default.yaml`, `tests/test_smoke.py`, `README.md` | Membre 4 |

`src/main.py` reste un raccourci vers le pipeline pour les anciennes commandes (`--input`, `--model`).
