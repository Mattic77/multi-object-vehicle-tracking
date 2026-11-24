from __future__ import annotations

import json
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    import psutil
except ImportError:
    psutil = None


class PerformanceMonitor:
    """
    Mesure les performances temporelles (par étape et globales) ainsi que
    l'empreinte mémoire (CPU RAM et GPU VRAM) du pipeline de suivi vidéo.
    """

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = self._resolve_device(device)
        self.use_cuda = self.device == "cuda"

        self._frame_start: float = 0.0
        self.frame_times_sec: List[float] = []
        self.step_times_sec: Dict[str, List[float]] = defaultdict(list)

        self.max_active_objects: int = 0
        self.total_object_instances: int = 0

        if self.use_cuda and torch is not None:
            torch.cuda.reset_peak_memory_stats()

    @staticmethod
    def _resolve_device(device: Optional[str]) -> str:
        if device is not None:
            return "cuda" if "cuda" in device.lower() else "cpu"
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _sync(self) -> None:
        """Synchronise le GPU pour éviter des mesures de temps biaisées."""
        if self.use_cuda and torch is not None:
            torch.cuda.synchronize()

    def start_frame(self) -> None:
        """Démarre le chronométrage d'une nouvelle image/frame."""
        self._sync()
        self._frame_start = time.perf_counter()

    @contextmanager
    def measure_step(self, step_name: str) -> Generator[None, None, None]:
        """Context manager pour chronométrer une étape précise """
        self._sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            elapsed = max(time.perf_counter() - t0, 1e-9)
            self.step_times_sec[step_name].append(elapsed)

    def end_frame(self, active_objects_count: int = 0) -> float:
        """Termine le chronométrage de l'image/frame et retourne le FPS instantané."""
        self._sync()
        elapsed = max(time.perf_counter() - self._frame_start, 1e-6)
        self.frame_times_sec.append(elapsed)
        self.max_active_objects = max(self.max_active_objects, active_objects_count)
        self.total_object_instances += active_objects_count
        return 1.0 / elapsed

    def _get_memory_stats_mb(self) -> Dict[str, Optional[float]]:
        ram_mb: Optional[float] = None
        vram_peak_mb: Optional[float] = None

        if psutil is not None:
            process = psutil.Process()
            ram_mb = round(process.memory_info().rss / (1024 ** 2), 2)

        if self.use_cuda and torch is not None:
            vram_peak_mb = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)

        return {
            "cpu_ram_mb": ram_mb,
            "gpu_vram_peak_mb": vram_peak_mb,
        }

    def summary(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        total_frames = len(self.frame_times_sec)
        total_time = float(np.sum(self.frame_times_sec)) if total_frames > 0 else 0.0

        fps_series = [1.0 / t for t in self.frame_times_sec] if total_frames > 0 else [0.0]
        steps_ms: Dict[str, Dict[str, float]] = {}

        for step, times in self.step_times_sec.items():
            arr_ms = np.array(times) * 1000.0
            steps_ms[step] = {
                "mean_ms": round(float(np.mean(arr_ms)), 3),
                "std_ms": round(float(np.std(arr_ms)), 3),
                "max_ms": round(float(np.max(arr_ms)), 3),
            }

        report: Dict[str, Any] = {
            "device": self.device,
            "total_frames": total_frames,
            "total_time_sec": round(total_time, 3),
            "avg_fps": round(total_frames / total_time, 2) if total_time > 0 else 0.0,
            "min_fps": round(float(np.min(fps_series)), 2),
            "max_fps": round(float(np.max(fps_series)), 2),
            "avg_frame_time_ms": round(float(np.mean(self.frame_times_sec)) * 1000.0, 2)
            if total_frames > 0
            else 0.0,
            "steps_breakdown_ms": steps_ms,
            "memory": self._get_memory_stats_mb(),
            "max_active_objects": self.max_active_objects,
            "avg_objects_per_frame": round(
                self.total_object_instances / max(total_frames, 1), 2
            ),
        }
        if extra:
            report.update(extra)
        return report

    def save_report(self, output_json: str | Path, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        report = self.summary(extra=extra)
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return report

    def plot_breakdown(self, output_png: str | Path) -> None:
        """Génère un graphique des temps moyens par étape (en ms/frame)."""
        report = self.summary()
        steps = report["steps_breakdown_ms"]
        if not steps:
            return

        labels = list(steps.keys())
        means = [steps[k]["mean_ms"] for k in labels]
        stds = [steps[k]["std_ms"] for k in labels]

        plt.figure(figsize=(8, 4.5))
        bars = plt.bar(labels, means, yerr=stds, capsize=5, color=["#2b5c8f", "#388e3c", "#f57c00"][: len(labels)])
        plt.ylabel("Temps moyen par image (ms)")
        plt.title(f"Temps de traitement par étape ({self.device.upper()} | FPS moyen : {report['avg_fps']})")
        plt.grid(axis="y", linestyle="--", alpha=0.5)

        for bar, val in zip(bars, means):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{val:.2f} ms",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        out = Path(output_png)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()


# --- Test autonome (sans avoir besoin des fichiers des autres membres) ---
if __name__ == "__main__":
    bench = PerformanceMonitor(device="cpu")

    for frame_i in range(50):
        bench.start_frame()

        with bench.measure_step("track_yolo_bytetrack"):
            time.sleep(0.018)  # Simule ~18 ms d'inférence

        with bench.measure_step("line_counting"):
            time.sleep(0.001)  # Simule ~1 ms de calcul géométrique

        with bench.measure_step("opencv_render"):
            time.sleep(0.004)  # Simule ~4 ms de dessin OpenCV

        bench.end_frame(active_objects_count=7)

    res = bench.save_report("outputs/test_benchmark.json")
    bench.plot_breakdown("outputs/test_benchmark.png")
    print(json.dumps(res, indent=2, ensure_ascii=False))