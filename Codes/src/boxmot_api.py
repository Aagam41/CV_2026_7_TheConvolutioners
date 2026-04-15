"""Re-export BoxMOT 17's native Python API.

Defensive imports — older/newer BoxMOT versions may rename or omit some
result classes, but `Boxmot`, `track`, `evaluate` are the stable core.

Important caveat: the Python facade is intentionally minimal — flags like
`--show-trajectories`, `--show-labels`, `--save-crop`, `--target-id` are
ONLY exposed via the BoxMOT CLI. For those we shell out to
``python -m boxmot.engine.cli track ...`` (see scripts/visdrone_run.py).
"""
from __future__ import annotations

import boxmot

# Stable symbols (BoxMOT 17+)
Boxmot = boxmot.Boxmot
track = getattr(boxmot, "track", None)
evaluate = getattr(boxmot, "evaluate", None)

# Result classes — defensive (names may shift between versions)
TrackRunResult = getattr(boxmot, "TrackRunResult", None)
GenerateResult = getattr(boxmot, "GenerateResult", None)
ValidationResult = getattr(boxmot, "ValidationResult", None)
TuneResult = getattr(boxmot, "TuneResult", None)
ResearchResult = getattr(boxmot, "ResearchResult", None)
ExportResult = getattr(boxmot, "ExportResult", None)

# Back-compat aliases for code that imported the old shim's result classes.
TrackRun = TrackRunResult
GenerateCache = GenerateResult
EvalResult = ValidationResult

__all__ = [
    "Boxmot", "track", "evaluate",
    "TrackRunResult", "GenerateResult", "ValidationResult",
    "TuneResult", "ResearchResult", "ExportResult",
    "TrackRun", "GenerateCache", "EvalResult",
]
