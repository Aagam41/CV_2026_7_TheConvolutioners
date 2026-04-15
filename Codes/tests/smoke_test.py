"""Smoke test — verifies the BoxMOT 17 API surface we depend on.

Runs offline (does NOT call the real BoxMOT engine). Just confirms:
    - boxmot.Boxmot exists with the expected method signatures
    - src package imports cleanly
    - tracker factory lists known trackers
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_boxmot_import():
    import boxmot
    assert hasattr(boxmot, "Boxmot"), "boxmot.Boxmot missing"
    assert hasattr(boxmot, "track"), "boxmot.track missing"
    assert hasattr(boxmot, "evaluate"), "boxmot.evaluate missing"
    print("OK  boxmot.Boxmot exists")


def test_boxmot_signature():
    from boxmot import Boxmot
    sig = inspect.signature(Boxmot.__init__)
    params = set(sig.parameters)
    expected = {"self", "detector", "reid", "tracker", "classes", "project"}
    missing = expected - params
    assert not missing, f"Boxmot.__init__ missing params: {missing}"
    print("OK  Boxmot.__init__ signature matches v17")


def test_src_imports():
    from src import (Boxmot, TrackingPipeline, build_detector,
                     build_tracker, list_trackers)
    print("OK  src package imports")


def test_tracker_factory():
    from src import list_trackers
    trks = list_trackers()
    assert "botsort" in trks, f"botsort missing from {trks}"
    print(f"OK  tracker factory lists: {trks}")


def test_visdrone_dataset():
    from src.datasets import VisDroneMOT, SPLIT_DIRS
    assert "train" in SPLIT_DIRS and "val" in SPLIT_DIRS and "test-dev" in SPLIT_DIRS
    print(f"OK  VisDroneMOT splits: {sorted(SPLIT_DIRS)}")


if __name__ == "__main__":
    test_boxmot_import()
    test_boxmot_signature()
    test_src_imports()
    test_tracker_factory()
    test_visdrone_dataset()
    print("\nAll smoke tests PASSED.")
