from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def compute_iou(box_a: List[float], box_b: List[float]) -> float:
    """Calcule l'Intersection over Union (IoU) entre deux boîtes [x1, y1, x2, y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    if union_area <= 1e-9:
        return 0.0
    return inter_area / union_area


def load_mot_ground_truth(gt_path: str | Path, one_indexed: bool = True) -> Dict[int, List[Dict[str, Any]]]:
    """
    Charge un fichier gt.txt au format MOTChallenge :
    frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility
    Retourne un dictionnaire {frame_idx: [{"gt_id": int, "bbox": [x1, y1, x2, y2]}, ...]}.
    """
    gt_by_frame: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    path = Path(gt_path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier vérité terrain introuvable : {gt_path}")

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [float(x) for x in line.split(",")]
            if len(parts) < 6:
                continue

            frame_idx = int(parts[0]) - (1 if one_indexed else 0)
            gt_id = int(parts[1])
            x, y, w, h = parts[2], parts[3], parts[4], parts[5]

            # Ignorer les annotations marquées comme non considérées (flag == 0)
            if len(parts) >= 7 and int(parts[6]) == 0:
                continue

            gt_by_frame[frame_idx].append(
                {
                    "gt_id": gt_id,
                    "bbox": [x, y, x + w, y + h],
                }
            )
    return gt_by_frame


class TrackingEvaluator:
    """
    Évalue la qualité du suivi multi-objets (IoU, FN, FP, ID switches, fragmentation)
    et extrait automatiquement des captures d'erreurs annotées.
    """

    def __init__(
        self,
        gt_path: Optional[str | Path] = None,
        iou_threshold: float = 0.5,
        error_dir: str | Path = "outputs/errors",
        max_error_snapshots: int = 15,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.gt_by_frame = load_mot_ground_truth(gt_path) if gt_path else None
        self.error_dir = Path(error_dir)
        self.max_error_snapshots = max_error_snapshots
        self.saved_snapshots_count = 0

        # Compteurs globaux (Mode Vérité Terrain)
        self.total_gt = 0
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        self.total_id_switches = 0
        self.total_fragmentations = 0
        self.iou_scores: List[float] = []

        # Historique par objet réel (gt_id) pour suivre les ID switches et fragmentations
        self._gt_last_pred_id: Dict[int, int] = {}
        self._gt_currently_tracked: Dict[int, bool] = {}

        # Historique sans vérité terrain (heuristiques d'échec)
        self._track_history: Dict[int, List[Tuple[int, List[float]]]] = defaultdict(list)
        self._active_prev_frame: Dict[int, List[float]] = {}

        # Journal détaillé des erreurs détectées
        self.error_events: List[Dict[str, Any]] = []

    def _match_boxes(
        self, gt_objs: List[Dict[str, Any]], pred_objs: List[Dict[str, Any]]
    ) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
        """Appariement glouton par meilleur IoU >= iou_threshold."""
        if not gt_objs or not pred_objs:
            return [], list(range(len(gt_objs))), list(range(len(pred_objs)))

        candidates: List[Tuple[float, int, int]] = []
        for gi, g in enumerate(gt_objs):
            for pi, p in enumerate(pred_objs):
                score = compute_iou(g["bbox"], p["bbox"])
                if score >= self.iou_threshold:
                    candidates.append((score, gi, pi))

        # Trier par IoU décroissant
        candidates.sort(key=lambda x: x[0], reverse=True)

        matched_gt: set[int] = set()
        matched_pred: set[int] = set()
        matches: List[Tuple[int, int, float]] = []

        for score, gi, pi in candidates:
            if gi not in matched_gt and pi not in matched_pred:
                matched_gt.add(gi)
                matched_pred.add(pi)
                matches.append((gi, pi, score))

        unmatched_gt = [i for i in range(len(gt_objs)) if i not in matched_gt]
        unmatched_pred = [i for i in range(len(pred_objs)) if i not in matched_pred]
        return matches, unmatched_gt, unmatched_pred

    def _save_error_snapshot(
        self,
        frame: Optional[np.ndarray],
        frame_idx: int,
        reason: str,
        pred_objs: List[Dict[str, Any]],
        gt_objs: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[str]:
        """Sauvegarde une image annotée illustrant un cas d'échec."""
        if frame is None or self.saved_snapshots_count >= self.max_error_snapshots:
            return None

        self.error_dir.mkdir(parents=True, exist_ok=True)
        canvas = frame.copy()

        # Dessiner la vérité terrain en vert si disponible
        if gt_objs:
            for g in gt_objs:
                x1, y1, x2, y2 = [int(v) for v in g["bbox"]]
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 0), 2)
                cv2.putText(
                    canvas, f"GT:{g['gt_id']}", (x1, max(y1 - 6, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 2
                )

        # Dessiner les prédictions en rouge/orange
        for p in pred_objs:
            x1, y1, x2, y2 = [int(v) for v in p["bbox"]]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 60, 255), 2)
            cv2.putText(
                canvas, f"Pred:{p['track_id']}", (x1, min(y2 + 16, canvas.shape[0] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 60, 255), 2
            )

        cv2.putText(
            canvas, f"Frame {frame_idx} | {reason}", (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2
        )

        out_path = self.error_dir / f"error_frame_{frame_idx:05d}.jpg"
        cv2.imwrite(str(out_path), canvas)
        self.saved_snapshots_count += 1
        return str(out_path)

    def update(
        self,
        frame_idx: int,
        tracked_objects: List[Dict[str, Any]],
        frame: Optional[np.ndarray] = None,
    ) -> None:
        """Met à jour l'évaluation sur une image donnée."""
        # --- CAS 1 : Évaluation exacte avec vérité terrain ---
        if self.gt_by_frame is not None:
            gt_objs = self.gt_by_frame.get(frame_idx, [])
            matches, unmatched_gt, unmatched_pred = self._match_boxes(gt_objs, tracked_objects)

            self.total_gt += len(gt_objs)
            self.total_tp += len(matches)
            self.total_fn += len(unmatched_gt)
            self.total_fp += len(unmatched_pred)

            frame_id_switches = 0
            matched_gt_ids = set()

            for gi, pi, iou_val in matches:
                self.iou_scores.append(iou_val)
                gt_id = gt_objs[gi]["gt_id"]
                pred_id = tracked_objects[pi]["track_id"]
                matched_gt_ids.add(gt_id)

                # Vérification d'un changement d'identité (ID switch)
                if gt_id in self._gt_last_pred_id and self._gt_last_pred_id[gt_id] != pred_id:
                    self.total_id_switches += 1
                    frame_id_switches += 1
                    # Si l'objet avait été perdu précédemment puis retrouvé -> fragmentation
                    if not self._gt_currently_tracked.get(gt_id, True):
                        self.total_fragmentations += 1

                self._gt_last_pred_id[gt_id] = pred_id
                self._gt_currently_tracked[gt_id] = True

            # Marquer les objets manqués comme momentanément perdus
            for gi in unmatched_gt:
                gt_id = gt_objs[gi]["gt_id"]
                self._gt_currently_tracked[gt_id] = False

            # Journaliser et capturer les erreurs majeures
            if frame_id_switches > 0 or len(unmatched_gt) > 0:
                reason = f"ID_Switches={frame_id_switches}, Missed(FN)={len(unmatched_gt)}, FP={len(unmatched_pred)}"
                snap = None
                if frame_id_switches > 0:
                    snap = self._save_error_snapshot(frame, frame_idx, reason, tracked_objects, gt_objs)
                self.error_events.append(
                    {
                        "frame_idx": frame_idx,
                        "type": "id_switch" if frame_id_switches > 0 else "missed_objects",
                        "id_switches": frame_id_switches,
                        "missed_objects": len(unmatched_gt),
                        "false_positives": len(unmatched_pred),
                        "snapshot": snap,
                    }
                )
            return

        # --- CAS 2 : Analyse heuristique sans vérité terrain ---
        current_boxes: Dict[int, List[float]] = {}
        for obj in tracked_objects:
            tid = obj["track_id"]
            bbox = obj["bbox"]
            current_boxes[tid] = bbox
            self._track_history[tid].append((frame_idx, bbox))

        # Détecter les disparitions brutales en zone centrale (probables occlusions)
        if frame is not None:
            h, w = frame.shape[:2]
            margin_x, margin_y = w * 0.1, h * 0.1
            for prev_tid, prev_box in self._active_prev_frame.items():
                if prev_tid not in current_boxes:
                    cx = (prev_box[0] + prev_box[2]) / 2.0
                    cy = (prev_box[1] + prev_box[3]) / 2.0
                    if margin_x < cx < (w - margin_x) and margin_y < cy < (h - margin_y):
                        reason = f"Perte piste ID {prev_tid} en zone centrale (occlusion probable)"
                        snap = self._save_error_snapshot(frame, frame_idx, reason, tracked_objects)
                        self.error_events.append(
                            {
                                "frame_idx": frame_idx,
                                "type": "sudden_track_loss",
                                "lost_track_id": prev_tid,
                                "snapshot": snap,
                            }
                        )

        self._active_prev_frame = current_boxes

    def summary(self) -> Dict[str, Any]:
        if self.gt_by_frame is not None:
            precision = self.total_tp / max(self.total_tp + self.total_fp, 1)
            recall = self.total_tp / max(self.total_gt, 1)
            mota = 1.0 - (self.total_fn + self.total_fp + self.total_id_switches) / max(self.total_gt, 1)
            frag_rate = self.total_fragmentations / max(len(self._gt_last_pred_id), 1)

            return {
                "evaluation_mode": "ground_truth",
                "iou_threshold": self.iou_threshold,
                "mean_iou": round(float(np.mean(self.iou_scores)), 4) if self.iou_scores else 0.0,
                "total_gt_objects": self.total_gt,
                "true_positives": self.total_tp,
                "false_negatives_missed": self.total_fn,
                "false_positives": self.total_fp,
                "id_switches": self.total_id_switches,
                "fragmentations": self.total_fragmentations,
                "fragmentation_rate": round(frag_rate, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "mota": round(mota, 4),
                "logged_error_frames": len(self.error_events),
            }

        # Résumé sans vérité terrain
        short_tracks = [tid for tid, hist in self._track_history.items() if len(hist) < 5]
        return {
            "evaluation_mode": "unsupervised_heuristics",
            "total_unique_tracks": len(self._track_history),
            "short_unstable_tracks_lt5_frames": len(short_tracks),
            "sudden_central_losses": len(self.error_events),
        }

    def save_report(self, output_json: str | Path = "outputs/evaluation_report.json") -> Dict[str, Any]:
        report = {
            "metrics": self.summary(),
            "errors": self.error_events[:100],  # Limite à 100 événements détaillés
        }
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return report


# --- Test autonome avec données simulées (sans attendre les autres membres) ---
if __name__ == "__main__":
    # 1. Création d'un faux fichier gt.txt (2 véhicules sur 4 frames)
    mock_gt = Path("outputs/mock_gt.txt")
    mock_gt.parent.mkdir(parents=True, exist_ok=True)
    mock_gt.write_text(
        "1,1,100,100,80,50,1,2,1.0\n"
        "1,2,300,200,90,60,1,2,1.0\n"
        "2,1,110,100,80,50,1,2,1.0\n"
        "2,2,310,200,90,60,1,2,1.0\n"
        "3,1,120,100,80,50,1,2,1.0\n"
        "3,2,320,200,90,60,1,2,1.0\n",
        encoding="utf-8",
    )

    evaluator = TrackingEvaluator(gt_path=mock_gt, error_dir="outputs/test_errors")
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Frame 0 : suivi parfait (ID 10 et ID 20)
    evaluator.update(0, [
        {"track_id": 10, "bbox": [102, 101, 182, 151]},
        {"track_id": 20, "bbox": [300, 200, 390, 260]},
    ], frame=dummy_frame)

    # Frame 1 : véhicule 2 manqué (FN = 1) !
    evaluator.update(1, [
        {"track_id": 10, "bbox": [111, 100, 191, 150]},
    ], frame=dummy_frame)

    # Frame 2 : véhicule 1 change d'ID (10 -> 99 => ID Switch = 1) + 1 faux positif (ID 55) !
    evaluator.update(2, [
        {"track_id": 99, "bbox": [120, 100, 200, 150]},
        {"track_id": 20, "bbox": [320, 200, 410, 260]},
        {"track_id": 55, "bbox": [500, 400, 550, 440]},
    ], frame=dummy_frame)

    res = evaluator.save_report("outputs/test_evaluation.json")
    print(json.dumps(res, indent=2, ensure_ascii=False))