import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple
import cv2

from metrics import PerformanceMonitor
from counter import LineCounter
from evaluator import TrackingEvaluator
from tracker import DEFAULT_CLASS_IDS, ObjectTracker


def parse_class_ids(raw: Optional[str]) -> Optional[List[int]]:
    if raw is None or raw.strip().lower() in ("all", "none", ""):
        return None
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_line(raw: str, width: int, height: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    parts = [float(v.strip()) for v in raw.split(",")]
    if len(parts) != 4:
        raise ValueError("La ligne doit contenir 4 valeurs : x1,y1,x2,y2")

    x1, y1, x2, y2 = parts
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.0:
        return (int(x1 * width), int(y1 * height)), (int(x2 * width), int(y2 * height))
    return (int(x1), int(y1)), (int(x2), int(y2))


def draw_overlay(
    frame,
    tracked_objects,
    line_start: Tuple[int, int],
    line_end: Tuple[int, int],
    counts: dict,
    fps: float,
):
    cv2.line(frame, line_start, line_end, (0, 255, 255), 3)

    for obj in tracked_objects:
        x1, y1, x2, y2 = [int(v) for v in obj["bbox"]]
        tid = obj["track_id"]
        label = obj["label"]
        conf = obj["conf"]
        traj = obj.get("trajectory", [])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        caption = f"ID {tid} | {label} {conf:.2f}"
        cv2.putText(
            frame, caption, (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
        )

        if len(traj) >= 2:
            pts = [(int(px), int(py)) for px, py in traj]
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i - 1], pts[i], (255, 160, 0), 2)

    cv2.rectangle(frame, (10, 10), (360, 125), (20, 20, 20), -1)
    cv2.putText(
        frame, f"FPS: {fps:.1f} | Total: {counts['total']}",
        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
    )
    cv2.putText(
        frame, f"In: {counts['in']} | Out: {counts['out']}",
        (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2,
    )
    classes_summary = ", ".join(f"{k}:{v}" for k, v in counts.get("by_label", {}).items()) or "none"
    cv2.putText(
        frame, classes_summary[:42],
        (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 255), 1,
    )

    return frame


def run_pipeline(args: argparse.Namespace) -> None:
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise FileNotFoundError(f"Impossible d'ouvrir la source vidéo : {args.input}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    if not args.no_save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps_in, (width, height))

    class_ids = parse_class_ids(args.classes)
    line_start, line_end = parse_line(args.line, width, height)

    # Initialisation des modules
    tracker = ObjectTracker(
        model_path=args.model,
        tracker_config=args.tracker,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        class_ids=class_ids,
        trail_length=args.trail,
        device=args.device,
    )
    counter = LineCounter(start_point=line_start, end_point=line_end)

    # Modules Membre 3 : Benchmark & Évaluation
    bench = PerformanceMonitor(device=args.device)
    evaluator = TrackingEvaluator(gt_path=args.gt, error_dir=args.error_dir)

    frame_idx = 0
    latest_counts = {"total": 0, "in": 0, "out": 0, "by_label": {}}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        bench.start_frame()

        with bench.measure_step("track_yolo_bytetrack"):
            tracked_objects = tracker.track(frame, frame_idx=frame_idx)

        with bench.measure_step("line_counting"):
            latest_counts = counter.update(tracked_objects)

        # Évaluation qualité & détection des cas d'échec
        evaluator.update(frame_idx=frame_idx, tracked_objects=tracked_objects, frame=frame)

        with bench.measure_step("opencv_render"):
            annotated = draw_overlay(
                frame=frame,
                tracked_objects=tracked_objects,
                line_start=line_start,
                line_end=line_end,
                counts=latest_counts,
                fps=bench.frame_times_sec[-1] ** -1 if bench.frame_times_sec else 0.0,
            )

        bench.end_frame(active_objects_count=len(tracked_objects))

        if writer is not None:
            writer.write(annotated)

        if args.show:
            cv2.imshow("YOLO11 + ByteTrack Object Counting", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    # Sauvegarde des rapports Benchmark & Évaluation
    bench_report = bench.save_report(
        args.metrics_out,
        extra={
            "unique_tracked_ids": len(tracker.seen_ids),
            "reappeared_ids_events": tracker.reappeared_ids_count,
            "counts": latest_counts,
        },
    )
    bench.plot_breakdown(args.bench_plot)
    eval_report = evaluator.save_report(args.eval_out)

    print("=== Résumé Benchmark ===")
    print(json.dumps(bench_report, indent=2, ensure_ascii=False))
    print("=== Résumé Évaluation ===")
    print(json.dumps(eval_report["metrics"], indent=2, ensure_ascii=False))


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
    return parser


if __name__ == "__main__":
    run_pipeline(build_parser().parse_args())