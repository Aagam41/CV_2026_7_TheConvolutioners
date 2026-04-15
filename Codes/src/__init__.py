"""Top-level public API.

For BoxMOT-consistent metrics + standard pipelines, use ``Boxmot``.
For custom detectors (e.g. SAHI) feeding a raw BoxMOT tracker, use
``TrackingPipeline`` together with ``build_detector`` / ``build_tracker``.
"""
from .boxmot_api import (
    Boxmot, TrackRunResult, GenerateResult, ValidationResult,
    TuneResult, ResearchResult, ExportResult,
    TrackRun, GenerateCache, EvalResult,
)
from .pipeline import TrackingPipeline, RunResult
from .detectors import build_detector, register_detector, BaseDetector
from .trackers import build_tracker, register_tracker, list_trackers

__all__ = [
    "Boxmot",
    "TrackRunResult", "GenerateResult", "ValidationResult",
    "TuneResult", "ResearchResult", "ExportResult",
    "TrackRun", "GenerateCache", "EvalResult",
    "TrackingPipeline", "RunResult",
    "BaseDetector", "build_detector", "register_detector",
    "build_tracker", "register_tracker", "list_trackers",
]
