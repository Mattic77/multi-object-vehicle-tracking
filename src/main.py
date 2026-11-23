"""Ancien point d'entrée, conservé pour les commandes déjà écrites.

Préférer :

    python scripts/run_experiment.py --source video.mp4 --weights yolo11n.pt --save-vid
"""

from __future__ import annotations

import argparse

from pipeline import run
from tracker import DEFAULT_CLASS_IDS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Détection, suivi et comptage multi-objets avec YOLO11 et ByteTrack")
    parser.add_argument("--input", required=True, help="Chemin vers la vidéo d'entrée")
    parser.add_argument("--output", default="outputs/annotated.mp4", help="Chemin vers la vidéo annotée")
    parser.add_argument("--metrics-out", default="outputs/benchmark.json", help="Rapport JSON de performance")
    parser.add_argument("--bench-plot", default="outputs/benchmark_breakdown.png", help="Graphique des temps par étape")
    parser.add_argument("--eval-out", default="outputs/evaluation.json", help="Rapport JSON d'évaluation du suivi")
    parser.add_argument("--gt", default=None, help="Chemin optionnel vers gt.txt (MOTChallenge/VisDrone)")
    parser.add_argument("--error-dir", default="outputs/errors", help="Dossier des captures d'erreurs annotées")
    parser.add_argument("--model", default="yolo11n.pt", help="Modèle YOLO (ex: yolo11n.pt)")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Configuration du tracker")
    parser.add_argument("--conf", type=float, default=0.25, help="Seuil de confiance")
    parser.add_argument("--iou", type=float, default=0.45, help="Seuil IoU NMS")
    parser.add_argument("--classes", default=",".join(str(i) for i in DEFAULT_CLASS_IDS), help="Classes cibles")
    parser.add_argument("--line", default="0.05,0.60,0.95,0.60", help="Ligne virtuelle 'x1,y1,x2,y2'")
    parser.add_argument("--trail", type=int, default=30, help="Longueur de trajectoire")
    parser.add_argument("--device", default=None, help="Device ('cuda', 'cpu', ou auto)")
    parser.add_argument("--show", action="store_true", help="Afficher la fenêtre OpenCV")
    parser.add_argument("--no-save", action="store_true", help="Ne pas enregistrer la vidéo")
    parser.add_argument("--max-frames", type=int, default=None, help="Limiter le nombre d'images")
    return parser


def config_from_args(args: argparse.Namespace) -> dict:
    return {
        "source": args.input,
        "weights": args.model,
        "tracker": args.tracker,
        "conf": args.conf,
        "iou": args.iou,
        "classes": args.classes,
        "line": args.line,
        "trail": args.trail,
        "device": args.device,
        "show": args.show,
        "save_vid": not args.no_save,
        "output": args.output,
        "metrics_out": args.metrics_out,
        "bench_plot": args.bench_plot,
        "eval_out": args.eval_out,
        "gt": args.gt,
        "error_dir": args.error_dir,
        "max_frames": args.max_frames,
    }


if __name__ == "__main__":
    run(config_from_args(build_parser().parse_args()))
