from typing import List, Optional, Dict, Any
import numpy as np
from ultralytics import YOLO

# Classes COCO par défaut :
# 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
# Mettre class_ids=None pour détecter toutes les classes COCO.
DEFAULT_CLASS_IDS = [0, 1, 2, 3, 5, 7]


class ObjectDetector:
    """Détecteur d'objets basé sur Ultralytics YOLO11."""

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        class_ids: Optional[List[int]] = DEFAULT_CLASS_IDS,
        device: Optional[str] = None,
    ) -> None:
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_ids = class_ids
        self.device = device

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Extrait les détections d'une image (frame BGR).

        Retourne une liste de dictionnaires :
        {
            "bbox": [x1, y1, x2, y2],
            "conf": float,
            "cls": int,
            "label": str,
        }
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.class_ids,
            device=self.device,
            verbose=False,
        )

        detections: List[Dict[str, Any]] = []
        if not results:
            return detections

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls_id in zip(boxes, confs, clss):
            x1, y1, x2, y2 = [float(v) for v in box]
            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "conf": float(conf),
                    "cls": int(cls_id),
                    "label": self.model.names.get(int(cls_id), str(cls_id)),
                }
            )

        return detections