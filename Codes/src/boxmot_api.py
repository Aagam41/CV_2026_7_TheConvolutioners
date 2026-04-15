"""Re-export BoxMOT 17's native Python API.

In BoxMOT 17 the official `Boxmot` facade exposes track / generate / val /
tune / research / export as Python methods. We just forward the symbols so
the rest of the project imports from one place.
"""
from boxmot import (
    Boxmot,
    track,
    evaluate,
    TrackRunResult,
    GenerateResult,
    ValidationResult,
    TuneResult,
    ResearchResult,
    ExportResult,
)

# Back-compat aliases for code that imported the old shim's result classes.
TrackRun = TrackRunResult
GenerateCache = GenerateResult
EvalResult = ValidationResult

__all__ = [
    "Boxmot",
    "track", "evaluate",
    "TrackRunResult", "GenerateResult", "ValidationResult",
    "TuneResult", "ResearchResult", "ExportResult",
    # legacy
    "TrackRun", "GenerateCache", "EvalResult",
]
