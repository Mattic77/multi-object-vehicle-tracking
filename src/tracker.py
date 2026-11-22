from collections import defaultdict, deque
from typing import List, Optional, Dict, Any, Tuple, Deque
import numpy as np
from ultralytics import YOLO

DEFAULT_CLASS_IDS = [0, 1, 2, 3, 5, 7]


class ObjectTracker:
    """Suivi multi-objets en temps réel avec ByteTrack et gestion des trajectoires."""

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        tracker_config: str = "bytetrack.yaml",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        class_ids: Optional[List[int]] = DEFAULT_CLASS_IDS,
        trail_length: int = 30,
        device: Optional[str] = None,
    ) -> None:
        self.model = YOLO(model_path)
        self.tracker_config = tracker_config
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_ids = class_ids
        self.trail_length = trail_length
        self.device = device

        self.trajectories: Dict[int, Deque[Tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=self.trail_length)
        )
        self.last_seen_frame: Dict[int, int] = {}
        self.seen_ids: set[int] = set()
        self.reappeared_ids_count: int = 0

    @staticmethod
    def _centroid(bbox: List[float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def track(self, frame: np.ndarray, frame_idx: int) -> List[Dict[str, Any]]:
        """
        Met à jour le suivi sur une frame et retourne les objets suivis :
        {
            "track_id": int,
            "bbox": [x1, y1, x2, y2],
            "centroid": (cx, cy),
            "conf": float,
            "cls": int,
            "label": str,
            "trajectory": [(cx, cy), ...],
        }
        """
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_config,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.class_ids,
            device=self.device,
            verbose=False,
        )

        tracked_objects: List[Dict[str, Any]] = []
        if not results:
            return tracked_objects

        result = results[0]
        if result.boxes is None or result.boxes.id is None or len(result.boxes) == 0:
            return tracked_objects

        boxes = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy().astype(int)

        for box, tid, conf, cls_id in zip(boxes, track_ids, confs, clss):
            x1, y1, x2, y2 = [float(v) for v in box]
            bbox = [x1, y1, x2, y2]
            cx, cy = self._centroid(bbox)

            # Détection approximative d'une réapparition après absence (> 15 frames)
            if tid in self.last_seen_frame and (frame_idx - self.last_seen_frame[tid]) > 15:
                self.reappeared_ids_count += 1

            self.last_seen_frame[tid] = frame_idx
            self.seen_ids.add(int(tid))
            self.trajectories[int(tid)].append((cx, cy))

            tracked_objects.append(
                {
                    "track_id": int(tid),
                    "bbox": bbox,
                    "centroid": (cx, cy),
                    "conf": float(conf),
                    "cls": int(cls_id),
                    "label": self.model.names.get(int(cls_id), str(cls_id)),
                    "trajectory": list(self.trajectories[int(tid)]),
                }
            )

        return tracked_objects