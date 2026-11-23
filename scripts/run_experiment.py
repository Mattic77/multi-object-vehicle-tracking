"""Point d'entrée du projet.

Charge configs/default.yaml, applique les arguments CLI, puis lance le pipeline.
Depuis la racine du dépôt :

    python scripts/run_experiment.py --source data/vehicles.mp4 --weights yolo11n.pt --device cpu --save-vid
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline import colab_status, load_config, run  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Détection YOLO11n, suivi ByteTrack, comptage et évaluation"
    )
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"), help="Fichier YAML des hyperparamètres")
    parser.add_argument("--source", default=None, help="Vidéo d'entrée (fichier ou flux)")
    parser.add_argument("--weights", default=None, help="Poids YOLO11n, par exemple yolo11n.pt")
    parser.add_argument("--line", default=None, help="Ligne virtuelle x1,y1,x2,y2 (pixels ou fractions 0-1)")
    parser.add_argument("--device", default=None, help="cpu, cuda, ou omis pour le choix automatique")
    parser.add_argument(
        "--save-vid",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Écrire la vidéo annotée (défaut : valeur de la config)",
    )
    parser.add_argument("--output", default=None, help="Chemin de la vidéo annotée")
    parser.add_argument("--classes", default=None, help="Classes COCO, ex. 2,3,5,7. 'all' pour tout détecter")
    parser.add_argument("--conf", type=float, default=None, help="Seuil de confiance")
    parser.add_argument("--iou", type=float, default=None, help="Seuil IoU du NMS")
    parser.add_argument("--gt", default=None, help="gt.txt MOTChallenge optionnel")
    parser.add_argument("--metrics-out", default=None, help="JSON de vitesse")
    parser.add_argument("--bench-plot", default=None, help="Graphique des temps par étape")
    parser.add_argument("--eval-out", default=None, help="JSON d'évaluation du suivi")
    parser.add_argument("--error-dir", default=None, help="Dossier des captures d'erreurs")
    parser.add_argument("--max-frames", type=int, default=None, help="Limiter le nombre d'images traitées")
    parser.add_argument("--show", action="store_true", help="Afficher la fenêtre OpenCV")
    return parser


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    mapping = {
        "source": args.source,
        "weights": args.weights,
        "line": args.line,
        "device": args.device,
        "save_vid": args.save_vid,
        "output": args.output,
        "classes": args.classes,
        "conf": args.conf,
        "iou": args.iou,
        "gt": args.gt,
        "metrics_out": args.metrics_out,
        "bench_plot": args.bench_plot,
        "eval_out": args.eval_out,
        "error_dir": args.error_dir,
        "max_frames": args.max_frames,
    }
    for key, value in mapping.items():
        if value is not None:
            cfg[key] = value
    if args.show:
        cfg["show"] = True
    return cfg


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    status = colab_status()
    print(f"Colab: {status['in_colab']} | CUDA: {status['cuda']}")
    cfg = apply_overrides(load_config(args.config), args)
    if not cfg.get("source"):
        raise SystemExit("Aucune source vidéo. Passe --source ou renseigne source dans configs/default.yaml.")
    run(cfg)


if __name__ == "__main__":
    main()
