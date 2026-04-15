"""Boxmot — a unified Python API around BoxMOT's official CLI engine.

The class mirrors the four BoxMOT subcommands as Python methods:

    Boxmot(detector, reid, tracker).track(source=..., save=True, ...)
    Boxmot(detector, reid, tracker).val(benchmark=..., postprocessing="gbrc", ...)
    Boxmot(detector, reid, tracker).generate(benchmark=...)            # cache
    Boxmot(detector, reid, tracker).generate(source="path/to/data")    # direct
    Boxmot(...).tune(benchmark=..., n_trials=10)
    Boxmot.export(weights="osnet_x0_25_msmt17.pt", include=["onnx"])

All methods shell out to ``python -m boxmot.engine.cli <mode> ...`` so the
results — including HOTA / MOTA / IDF1 — are exactly what BoxMOT's own
benchmark pipeline produces. No parallel metric implementation here.

Common kwargs that map 1:1 to CLI flags (snake_case ↔ kebab-case):
    save, save_txt, save_crop, save_trajectories, show, show_trajectories,
    show_lost, show_kf_preds, show_labels, hide_labels, show_conf, hide_conf,
    classes, per_class, target_id, project, name, exist_ok, device, half,
    imgsz, conf, iou, vid_stride, batch_size, agnostic_nms, n_threads,
    line_width, verbose, postprocessing, gsi, eval_existing, split,
    benchmark, n_trials, objective, include, dynamic, weights.

Anything not listed is forwarded verbatim — pass `extra={"--foo": "bar"}` to
inject any flag BoxMOT's CLI accepts in the future.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

log = logging.getLogger(__name__)

# Flags that have NO value (presence-only booleans). If the kwarg is True,
# we emit just `--flag`; if False, we omit it entirely.
_BOOL_FLAGS = {
    "save", "save_txt", "save_crop", "save_trajectories",
    "show", "show_trajectories", "show_lost", "show_kf_preds",
    "show_labels", "hide_labels", "show_conf", "hide_conf",
    "per_class", "agnostic_nms", "half", "verbose", "exist_ok",
    "gsi", "eval_existing", "dynamic",
}


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------
@dataclass
class TrackRun:
    save_dir: Path
    video: Optional[Path]
    txt: Optional[Path]
    crops_dir: Optional[Path]
    elapsed: float
    args: Dict[str, Any]
    stdout: str = ""

    def __repr__(self) -> str:
        return (f"TrackRun(save_dir='{self.save_dir}', video='{self.video}', "
                f"txt='{self.txt}', elapsed={self.elapsed:.2f}s)")


@dataclass
class GenerateCache:
    cache_dir: Path
    detections_dir: Optional[Path]
    embeddings_dir: Optional[Path]
    timings: Dict[str, Any] = field(default_factory=dict)
    args: Dict[str, Any] = field(default_factory=dict)
    stdout: str = ""

    def __repr__(self) -> str:
        return f"GenerateCache(cache_dir='{self.cache_dir}', timings={self.timings})"


@dataclass
class EvalResult:
    save_dir: Path
    metrics: Dict[str, float]
    args: Dict[str, Any]
    stdout: str = ""

    def __repr__(self) -> str:
        keys = ("HOTA", "MOTA", "IDF1", "IDs", "FP", "FN")
        head = ", ".join(f"{k}={self.metrics[k]:.3f}" if k in self.metrics else ""
                         for k in keys if k in self.metrics)
        return f"EvalResult({head})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_cli_flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def _kwargs_to_argv(kwargs: Dict[str, Any]) -> List[str]:
    """Convert snake_case kwargs into BoxMOT CLI argv tokens."""
    argv: List[str] = []
    extra = kwargs.pop("extra", None) or {}

    for key, val in kwargs.items():
        if val is None:
            continue
        flag = _to_cli_flag(key)

        if key in _BOOL_FLAGS:
            if val:
                argv.append(flag)
            continue

        if isinstance(val, bool):
            if val:
                argv.append(flag)
            continue

        if isinstance(val, (list, tuple)):
            if key == "classes":
                argv += [flag, ",".join(str(v) for v in val)]
            elif key == "include":
                for v in val:
                    argv += [flag, str(v)]
            else:
                argv += [flag, ",".join(str(v) for v in val)]
            continue

        argv += [flag, str(val)]

    # Caller-supplied extras override / extend everything.
    for k, v in extra.items():
        if v is True:
            argv.append(k if k.startswith("--") else _to_cli_flag(k))
        elif v not in (False, None):
            argv += [k if k.startswith("--") else _to_cli_flag(k), str(v)]
    return argv


def _run_cli(mode: str, argv: Sequence[str], capture: bool = True) -> str:
    """Invoke `python -m boxmot.engine.cli <mode> <argv>` and return stdout."""
    cmd = [sys.executable, "-m", "boxmot.engine.cli", mode, *argv]
    log.info("Running: %s", " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.run(
        cmd,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        env=os.environ.copy(),
    )
    return proc.stdout or ""


def _resolve_save_dir(stdout: str, project: str, name: str) -> Path:
    """Best-effort: parse 'Results saved to <path>' from BoxMOT's stdout."""
    m = re.search(r"saved to[: ]+(\S+)", stdout, flags=re.IGNORECASE)
    if m:
        return Path(m.group(1).rstrip("."))
    # Fallback to the conventional location.
    return Path(project) / name


def _parse_metrics_table(stdout: str) -> Dict[str, float]:
    """Parse BoxMOT's printed metrics table into a flat dict.

    BoxMOT's eval output ends with a table whose header row contains
    HOTA / MOTA / IDF1 etc. We parse the COMBINED row.
    """
    metrics: Dict[str, float] = {}
    lines = stdout.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if "HOTA" in line and "MOTA" in line and "IDF1" in line:
            header_idx = i
            break
    if header_idx is None:
        return metrics

    headers = lines[header_idx].split()
    for row in lines[header_idx + 1:]:
        toks = row.split()
        if len(toks) < len(headers):
            continue
        if toks[0].lower() in {"combined", "overall", "all"}:
            for h, v in zip(headers, toks[1:]):
                try:
                    metrics[h] = float(v)
                except ValueError:
                    pass
            break
    return metrics


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
class Boxmot:
    """Unified Python interface to BoxMOT's track / generate / eval / tune."""

    def __init__(
        self,
        detector: Optional[str] = None,
        reid: Optional[str] = None,
        tracker: Optional[str] = None,
        device: Optional[str] = None,
        half: bool = False,
    ) -> None:
        self.detector = detector
        self.reid = reid
        self.tracker = tracker
        self.device = device
        self.half = half

    # ------------------------------------------------------------------ track
    def track(self, source: Union[str, Path], **kwargs) -> TrackRun:
        """Run detector + tracker on a source and (optionally) save outputs.

        See module docstring for the full kwarg list. Common ones:
            save, save_txt, save_crop, save_trajectories,
            show_trajectories, show_labels, hide_labels,
            classes=[0,1], per_class, target_id=7,
            project="runs/track", name="exp", imgsz=640, conf=0.3, iou=0.5.
        """
        argv = self._common_argv(kwargs)
        argv += ["--source", str(source)]
        argv += _kwargs_to_argv(kwargs)

        project = kwargs.get("project", "runs/track")
        name = kwargs.get("name", "exp")
        t0 = time.time()
        out = _run_cli("track", argv)
        elapsed = time.time() - t0

        save_dir = _resolve_save_dir(out, project, name)
        video = next(iter(save_dir.glob("*.mp4")), None) if save_dir.exists() else None
        txt = next(iter(save_dir.glob("*.txt")), None) if save_dir.exists() else None
        crops_dir = save_dir / "crops" if (save_dir / "crops").exists() else None

        return TrackRun(
            save_dir=save_dir, video=video, txt=txt, crops_dir=crops_dir,
            elapsed=elapsed, args=self._args_snapshot(kwargs, source=source),
            stdout=out,
        )

    # --------------------------------------------------------------- generate
    def generate(
        self,
        benchmark: Optional[str] = None,
        source: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> GenerateCache:
        """Pre-compute detections + ReID embeddings for fast re-evaluation.

        Use either ``benchmark`` (e.g. 'mot17-ablation') or ``source``
        (path / glob). Returns the cache directory holding dets and embs.
        """
        if not benchmark and not source:
            raise ValueError("generate(): pass benchmark=... or source=...")

        argv = self._common_argv(kwargs, include_tracker=False)
        if benchmark:
            argv += ["--benchmark", benchmark]
        if source:
            argv += ["--source", str(source)]
        argv += _kwargs_to_argv(kwargs)

        project = kwargs.get("project", "runs/generate")
        name = kwargs.get("name", "exp")
        t0 = time.time()
        out = _run_cli("generate", argv)
        elapsed = time.time() - t0

        save_dir = _resolve_save_dir(out, project, name)
        det_dir = save_dir / "dets"
        emb_dir = save_dir / "embs"
        return GenerateCache(
            cache_dir=save_dir,
            detections_dir=det_dir if det_dir.exists() else None,
            embeddings_dir=emb_dir if emb_dir.exists() else None,
            timings={"elapsed": elapsed},
            args=self._args_snapshot(kwargs, benchmark=benchmark, source=source),
            stdout=out,
        )

    # ----------------------------------------------------------------- val/eval
    def val(self, benchmark: Optional[str] = None, **kwargs) -> EvalResult:
        """Evaluate the (detector, reid, tracker) combo on a benchmark.

        Equivalent to ``boxmot eval``. Common kwargs:
            postprocessing="gsi" | "gbrc", verbose=True, split="train",
            eval_existing=True, project="runs/eval", name="exp".

        Returns an :class:`EvalResult` with HOTA / MOTA / IDF1 etc.
        """
        argv = self._common_argv(kwargs)
        if benchmark:
            argv += ["--benchmark", benchmark]
        argv += _kwargs_to_argv(kwargs)

        project = kwargs.get("project", "runs/eval")
        name = kwargs.get("name", "exp")
        out = _run_cli("eval", argv)

        save_dir = _resolve_save_dir(out, project, name)
        metrics = _parse_metrics_table(out)
        return EvalResult(
            save_dir=save_dir, metrics=metrics,
            args=self._args_snapshot(kwargs, benchmark=benchmark),
            stdout=out,
        )

    # `eval` is a builtin shadow but harmless as a method name.
    eval = val

    # ------------------------------------------------------------------- tune
    def tune(self, benchmark: Optional[str] = None, n_trials: int = 10,
             **kwargs) -> Path:
        """Tune tracker hyperparameters via BoxMOT's evolutionary search."""
        argv = self._common_argv(kwargs)
        if benchmark:
            argv += ["--benchmark", benchmark]
        argv += ["--n-trials", str(n_trials)]
        argv += _kwargs_to_argv(kwargs)
        out = _run_cli("tune", argv)
        return _resolve_save_dir(out,
                                 kwargs.get("project", "runs/tune"),
                                 kwargs.get("name", "exp"))

    # ----------------------------------------------------------------- export
    @staticmethod
    def export(weights: Union[str, Path], include: Iterable[str] = ("onnx",),
               dynamic: bool = False, device: str = "cpu", **kwargs) -> str:
        """Export a ReID model to ONNX / OpenVINO / TensorRT / TorchScript."""
        argv = ["--weights", str(weights), "--device", device]
        for fmt in include:
            argv += ["--include", fmt]
        if dynamic:
            argv.append("--dynamic")
        argv += _kwargs_to_argv(kwargs)
        return _run_cli("export", argv)

    # -------------------------------------------------------------- internals
    def _common_argv(self, kwargs: Dict[str, Any], include_tracker: bool = True) -> List[str]:
        """Inject the constructor-level defaults unless the call overrides them."""
        argv: List[str] = []
        det = kwargs.pop("detector", self.detector)
        rid = kwargs.pop("reid", self.reid)
        trk = kwargs.pop("tracker", self.tracker)
        dev = kwargs.pop("device", self.device)
        half = kwargs.pop("half", self.half)

        if det:
            argv += ["--detector", det]
        if rid:
            argv += ["--reid", rid]
        if include_tracker and trk:
            argv += ["--tracker", trk]
        if dev:
            argv += ["--device", str(dev)]
        if half:
            argv += ["--half"]
        return argv

    def _args_snapshot(self, kwargs: Dict[str, Any], **extra) -> Dict[str, Any]:
        snap = {
            "detector": self.detector, "reid": self.reid,
            "tracker": self.tracker, "device": self.device, "half": self.half,
        }
        snap.update({k: v for k, v in extra.items() if v is not None})
        snap.update(kwargs)
        return snap

    # ----------------------------------------------------- nice JSON repr
    def __repr__(self) -> str:
        return (f"Boxmot(detector={self.detector!r}, reid={self.reid!r}, "
                f"tracker={self.tracker!r}, device={self.device!r})")


__all__ = ["Boxmot", "TrackRun", "GenerateCache", "EvalResult"]
