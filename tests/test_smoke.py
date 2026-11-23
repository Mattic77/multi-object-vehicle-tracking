"""Vérifie qu'un clone peut charger la config, le CLI et finir un passage court.

Lancer depuis la racine du dépôt, après pip install -r requirements.txt :

    python tests/test_smoke.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from pipeline import load_config, parse_class_ids, parse_line, run  # noqa: E402
import run_experiment  # noqa: E402


def test_config_has_required_keys() -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    for key in ("source", "weights", "line", "device", "save_vid"):
        assert key in cfg, key
    assert cfg["weights"] == "yolo11n.pt"
    assert parse_class_ids(cfg["classes"]) == [0, 1, 2, 3, 5, 7]
    assert parse_line(cfg["line"], 1000, 500) == ((50, 300), (950, 300))


def test_cli_help_exposes_required_flags() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_experiment.py"), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    for flag in ("--source", "--weights", "--line", "--device", "--save-vid"):
        assert flag in completed.stdout, flag
    args = run_experiment.build_parser().parse_args(
        ["--source", "video.mp4", "--weights", "yolo11n.pt", "--line", "0,1,2,3", "--device", "cpu", "--save-vid"]
    )
    cfg = run_experiment.apply_overrides({"source": "old.mp4", "save_vid": False, "device": None}, args)
    assert cfg["source"] == "video.mp4"
    assert cfg["weights"] == "yolo11n.pt"
    assert cfg["device"] == "cpu"
    assert cfg["save_vid"] is True


def test_pipeline_finishes_on_synthetic_video() -> None:
    import cv2
    import numpy as np

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        folder = Path(tmp)
        video = folder / "tiny.mp4"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 240))
        try:
            for _ in range(3):
                frame = np.zeros((240, 320, 3), dtype=np.uint8)
                cv2.rectangle(frame, (40, 80), (140, 180), (0, 0, 180), -1)
                writer.write(frame)
        finally:
            writer.release()

        result = run(
            {
                "source": str(video),
                "weights": "yolo11n.pt",
                "tracker": "bytetrack.yaml",
                "device": "cpu",
                "conf": 0.25,
                "iou": 0.45,
                "classes": [2, 3, 5, 7],
                "line": "0.1,0.5,0.9,0.5",
                "trail": 10,
                "save_vid": True,
                "show": False,
                "output": str(folder / "out.mp4"),
                "metrics_out": str(folder / "benchmark.json"),
                "bench_plot": str(folder / "benchmark.png"),
                "eval_out": str(folder / "evaluation.json"),
                "gt": None,
                "error_dir": str(folder / "errors"),
                "max_frames": 3,
            }
        )
        assert result["benchmark"]["total_frames"] == 3
        assert (folder / "benchmark.json").exists()
        assert (folder / "evaluation.json").exists()
        assert (folder / "out.mp4").stat().st_size > 0
        assert (folder / "benchmark.png").exists()


def main() -> None:
    test_config_has_required_keys()
    test_cli_help_exposes_required_flags()
    print("Config et CLI OK. Passage court du pipeline (télécharge yolo11n.pt si besoin)...")
    test_pipeline_finishes_on_synthetic_video()
    print("Smoke test OK")


if __name__ == "__main__":
    main()
