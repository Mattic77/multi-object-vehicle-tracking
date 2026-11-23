"""Orchestration du suivi multi-objets.

Chaîne exécutée pour chaque image :
data_loader -> détecteur YOLO11n + tracker ByteTrack -> comptage et overlay
-> évaluateur -> benchmark.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")

import cv2
import yaml

from counter import LineCounter
from detector import ObjectDetector
from evaluator import TrackingEvaluator
from metrics import PerformanceMonitor
from tracker import DEFAULT_CLASS_IDS, ObjectTracker

SAMPLE_VIDEO_URL = "https://media.roboflow.com/supervision/video-examples/vehicles.mp4"


def running_in_colab() -> bool:
    """Vrai dans un runtime Google Colab."""
    if os.environ.get("COLAB_RELEASE_TAG"):
        return True
    try:
        import google.colab  # noqa: F401
    except ImportError:
        return False
    return True


def colab_status() -> Dict[str, Any]:
    """État utile pour vérifier qu'un notebook Colab a un GPU."""
    cuda = False
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
    except ImportError:
        cuda = False
    return {"in_colab": running_in_colab(), "cuda": cuda}


def load_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration introuvable : {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError("La configuration YAML doit être un objet")
    return loaded


def parse_class_ids(raw: Any) -> Optional[List[int]]:
    if raw is None:
        return list(DEFAULT_CLASS_IDS)
    if isinstance(raw, (list, tuple)):
        return [int(value) for value in raw]
    text = str(raw).strip()
    if text.lower() in ("all", "none", ""):
        return None
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_line(raw: Any, width: int, height: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    if isinstance(raw, (list, tuple)):
        parts = [float(value) for value in raw]
    else:
        parts = [float(value.strip()) for value in str(raw).split(",")]
    if len(parts) != 4:
        raise ValueError("La ligne doit contenir 4 valeurs : x1,y1,x2,y2")

    x1, y1, x2, y2 = parts
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.0:
        return (int(x1 * width), int(y1 * height)), (int(x2 * width), int(y2 * height))
    return (int(x1), int(y1)), (int(x2), int(y2))


def render_counter_overlay(
    frame,
    tracked_objects,
    line_start: Tuple[int, int],
    line_end: Tuple[int, int],
    counts: dict,
    fps: float,
):
    """Dessine identifiants, trajectoires, ligne virtuelle et compteurs."""
    cv2.line(frame, line_start, line_end, (0, 255, 255), 3)

    for obj in tracked_objects:
        x1, y1, x2, y2 = [int(value) for value in obj["bbox"]]
        caption = f"ID {obj['track_id']} | {obj['label']} {obj['conf']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(
            frame,
            caption,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )
        trajectory = obj.get("trajectory", [])
        if len(trajectory) >= 2:
            points = [(int(px), int(py)) for px, py in trajectory]
            for index in range(1, len(points)):
                cv2.line(frame, points[index - 1], points[index], (255, 160, 0), 2)

    cv2.rectangle(frame, (10, 10), (360, 125), (20, 20, 20), -1)
    cv2.putText(
        frame,
        f"FPS: {fps:.1f} | Total: {counts['total']}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"In: {counts['in']} | Out: {counts['out']}",
        (20, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
    )
    classes_summary = ", ".join(f"{key}:{value}" for key, value in counts.get("by_label", {}).items()) or "none"
    cv2.putText(
        frame,
        classes_summary[:42],
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 220, 255),
        1,
    )
    return frame


def open_video(source: str):
    """data_loader : ouvre une vidéo ou un flux et échoue avec un message explicite."""
    path = Path(source)
    looks_like_file = not str(source).isdigit() and "://" not in str(source)
    if looks_like_file and not path.exists():
        raise FileNotFoundError(
            f"Vidéo introuvable : {source}\n"
            "Télécharge un extrait public, par exemple :\n"
            f"  {SAMPLE_VIDEO_URL}\n"
            "puis relance avec --source chemin/vers/la/video.mp4"
        )
    capture = cv2.VideoCapture(source if not looks_like_file else str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Impossible d'ouvrir la source vidéo : {source}")
    return capture


def run(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Enchaîne les modules et écrit la vidéo, le benchmark et l'évaluation."""
    status = colab_status()
    if status["in_colab"]:
        print(f"Runtime Colab détecté | CUDA: {status['cuda']}")
        if not status["cuda"] and not cfg.get("device"):
            print("Aucun GPU visible. Dans Colab : Exécution -> Modifier le type d'exécution -> T4 GPU.")

    capture = open_video(str(cfg["source"]))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = capture.get(cv2.CAP_PROP_FPS) or 25.0

    ok, first = capture.read()
    if not ok or first is None:
        capture.release()
        raise RuntimeError(f"La source ne contient aucune image : {cfg['source']}")
    if width <= 0 or height <= 0:
        height, width = first.shape[:2]

    class_ids = parse_class_ids(cfg.get("classes", DEFAULT_CLASS_IDS))
    line_start, line_end = parse_line(cfg.get("line", "0.05,0.60,0.95,0.60"), width, height)
    device = cfg.get("device") or None
    weights = str(cfg.get("weights", "yolo11n.pt"))

    # ObjectDetector est le module YOLO11n. ObjectTracker.track appelle
    # model.track : détection puis ByteTrack dans le même passage, sans second
    # appel à ObjectDetector.detect.
    print(f"Détecteur branché : {ObjectDetector.__name__} | poids : {weights}")
    tracker = ObjectTracker(
        model_path=weights,
        tracker_config=str(cfg.get("tracker", "bytetrack.yaml")),
        conf_threshold=float(cfg.get("conf", 0.25)),
        iou_threshold=float(cfg.get("iou", 0.45)),
        class_ids=class_ids,
        trail_length=int(cfg.get("trail", 30)),
        device=device,
    )
    counter = LineCounter(start_point=line_start, end_point=line_end)
    bench = PerformanceMonitor(device=device)
    evaluator = TrackingEvaluator(gt_path=cfg.get("gt") or None, error_dir=cfg.get("error_dir", "outputs/errors"))

    writer = None
    output_path = Path(cfg.get("output", "outputs/annotated.mp4"))
    if cfg.get("save_vid", True):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (width, height))

    frame_idx = 0
    latest_counts: Dict[str, Any] = {"total": 0, "in": 0, "out": 0, "by_label": {}}
    max_frames = cfg.get("max_frames")
    show = bool(cfg.get("show", False))
    pending = first

    print(
        "Pipeline: data_loader -> YOLO11n (ObjectDetector) + ByteTrack (ObjectTracker) "
        "-> LineCounter + overlay -> TrackingEvaluator -> PerformanceMonitor"
    )

    try:
        while True:
            if max_frames is not None and frame_idx >= int(max_frames):
                break
            if pending is not None:
                frame = pending
                pending = None
            else:
                ok, frame = capture.read()
                if not ok:
                    break

            bench.start_frame()
            with bench.measure_step("detector_tracker"):
                tracked_objects = tracker.track(frame, frame_idx=frame_idx)
            with bench.measure_step("line_counting"):
                latest_counts = counter.update(tracked_objects)
            evaluator.update(frame_idx=frame_idx, tracked_objects=tracked_objects, frame=frame)
            with bench.measure_step("counter_viz"):
                annotated = render_counter_overlay(
                    frame=frame,
                    tracked_objects=tracked_objects,
                    line_start=line_start,
                    line_end=line_end,
                    counts=latest_counts,
                    fps=(1.0 / bench.frame_times_sec[-1]) if bench.frame_times_sec else 0.0,
                )
            bench.end_frame(active_objects_count=len(tracked_objects))

            if writer is not None:
                writer.write(annotated)
            if show:
                cv2.imshow("YOLO11 + ByteTrack", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_idx += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    bench_report = bench.save_report(
        cfg.get("metrics_out", "outputs/benchmark.json"),
        extra={
            "source": str(cfg["source"]),
            "weights": weights,
            "line": str(cfg.get("line")),
            "classes": class_ids,
            "frame_width": width,
            "frame_height": height,
            "unique_tracked_ids": len(tracker.seen_ids),
            "reappeared_ids_events": tracker.reappeared_ids_count,
            "counts": latest_counts,
        },
    )
    bench.plot_breakdown(cfg.get("bench_plot", "outputs/benchmark_breakdown.png"))
    eval_report = evaluator.save_report(cfg.get("eval_out", "outputs/evaluation.json"))

    print("=== Résumé Benchmark ===")
    print(json.dumps(bench_report, indent=2, ensure_ascii=False))
    print("=== Résumé Évaluation ===")
    print(json.dumps(eval_report["metrics"], indent=2, ensure_ascii=False))
    return {"benchmark": bench_report, "evaluation": eval_report, "counts": latest_counts}
