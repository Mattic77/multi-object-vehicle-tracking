import time
from typing import Dict, Any


class PerformanceMonitor:
    """Mesure le temps de traitement, le FPS et des indicateurs de suivi."""

    def __init__(self) -> None:
        self._frame_start: float = 0.0
        self.total_frames: int = 0
        self.total_elapsed_sec: float = 0.0
        self.last_fps: float = 0.0
        self.max_active_objects: int = 0
        self.total_object_instances: int = 0

    def start_frame(self) -> None:
        self._frame_start = time.perf_counter()

    def end_frame(self, active_objects_count: int) -> float:
        elapsed = max(time.perf_counter() - self._frame_start, 1e-6)
        self.total_frames += 1
        self.total_elapsed_sec += elapsed
        self.last_fps = 1.0 / elapsed
        self.max_active_objects = max(self.max_active_objects, active_objects_count)
        self.total_object_instances += active_objects_count
        return self.last_fps

    @property
    def avg_fps(self) -> float:
        if self.total_elapsed_sec <= 0:
            return 0.0
        return self.total_frames / self.total_elapsed_sec

    def summary(self, extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "total_frames": self.total_frames,
            "total_time_sec": round(self.total_elapsed_sec, 3),
            "avg_fps": round(self.avg_fps, 2),
            "max_active_objects": self.max_active_objects,
            "avg_objects_per_frame": round(
                self.total_object_instances / max(self.total_frames, 1), 2
            ),
        }
        if extra:
            data.update(extra)
        return data