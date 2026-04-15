"""Smoke test: verify Boxmot wrapper builds the right CLI argv.

Does NOT call the real BoxMOT engine — it monkey-patches subprocess so the
test runs offline with no model downloads. Verifies that snake_case kwargs
turn into the correct kebab-case CLI flags.

Run:
    python tests/smoke_test.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import Boxmot  # noqa: E402


def _capture_cmd(*_a, **_kw):
    _capture_cmd.cmd = _a[0]
    return subprocess.CompletedProcess(args=_a[0], returncode=0, stdout="", stderr="")


def test_track_argv():
    with patch("src.boxmot_api.subprocess.run", side_effect=_capture_cmd):
        Boxmot(detector="yolov8n", reid="osnet_x0_25_msmt17", tracker="botsort").track(
            source="video.mp4",
            save=True, save_txt=True, show_trajectories=True,
            classes=[0, 1], per_class=True, target_id=7,
            conf=0.3, iou=0.5, imgsz=640,
            project="outputs/track", name="exp1", exist_ok=True,
        )
    cmd = _capture_cmd.cmd
    assert "track" in cmd
    assert ["--detector", "yolov8n"] == cmd[cmd.index("--detector"):cmd.index("--detector") + 2]
    assert "--save" in cmd
    assert "--save-txt" in cmd
    assert "--show-trajectories" in cmd
    assert "--per-class" in cmd
    assert "--exist-ok" in cmd
    assert ["--classes", "0,1"] == cmd[cmd.index("--classes"):cmd.index("--classes") + 2]
    assert ["--target-id", "7"] == cmd[cmd.index("--target-id"):cmd.index("--target-id") + 2]
    assert ["--source", "video.mp4"] == cmd[cmd.index("--source"):cmd.index("--source") + 2]
    print("OK  track argv")


def test_val_argv():
    with patch("src.boxmot_api.subprocess.run", side_effect=_capture_cmd):
        Boxmot(detector="yolov8n", reid="lmbn_n_duke", tracker="boosttrack").val(
            benchmark="mot17-ablation", postprocessing="gbrc", verbose=True,
        )
    cmd = _capture_cmd.cmd
    assert "eval" in cmd
    assert ["--benchmark", "mot17-ablation"] == cmd[cmd.index("--benchmark"):cmd.index("--benchmark") + 2]
    assert ["--postprocessing", "gbrc"] == cmd[cmd.index("--postprocessing"):cmd.index("--postprocessing") + 2]
    assert "--verbose" in cmd
    print("OK  val argv")


def test_generate_argv():
    with patch("src.boxmot_api.subprocess.run", side_effect=_capture_cmd):
        Boxmot(detector="yolov8n", reid="osnet_x0_25_msmt17").generate(
            source="path/to/dataset", project="outputs/cache",
        )
    cmd = _capture_cmd.cmd
    assert "generate" in cmd
    assert "--tracker" not in cmd, "generate must not pass --tracker"
    assert ["--source", "path/to/dataset"] == cmd[cmd.index("--source"):cmd.index("--source") + 2]
    print("OK  generate argv")


def test_extra_passthrough():
    with patch("src.boxmot_api.subprocess.run", side_effect=_capture_cmd):
        Boxmot(detector="yolov8n", tracker="ocsort").track(
            source="0", show=True,
            extra={"--n-threads": 4, "--agnostic-nms": True},
        )
    cmd = _capture_cmd.cmd
    assert "--show" in cmd
    assert ["--n-threads", "4"] == cmd[cmd.index("--n-threads"):cmd.index("--n-threads") + 2]
    assert "--agnostic-nms" in cmd
    print("OK  extras passthrough")


if __name__ == "__main__":
    test_track_argv()
    test_val_argv()
    test_generate_argv()
    test_extra_passthrough()
    print("\nAll smoke tests PASSED.")
