"""Top-level public API.

For BoxMOT-consistent metrics + standard pipelines, use ``Boxmot``.
For custom detectors (e.g. SAHI) feeding a raw BoxMOT tracker, use
``TrackingPipeline`` together with ``build_detector`` / ``build_tracker``.
"""
from .boxmot_api import Boxmot, TrackRun, GenerateCache, EvalResult
from .pipeline import TrackingPipeline, RunResult
from .detectors import build_detector, register_detector, BaseDetector
from .trackers import build_tracker, register_tracker, list_trackers

__all__ = [
    # Unified BoxMOT API (recommended)
    "Boxmot", "TrackRun", "GenerateCache", "EvalResult",
    # Custom-detector pipeline
    "TrackingPipeline", "RunResult",
    "BaseDetector", "build_detector", "register_detector",
    "build_tracker", "register_tracker", "list_trackers",
]
