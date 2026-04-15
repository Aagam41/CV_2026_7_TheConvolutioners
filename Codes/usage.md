# Usage Guide

This project gives you **two** ways to run multi-object tracking — pick the
one that matches your use case.

| Path | API | Detector | Tracker | Metrics |
|------|-----|----------|---------|---------|
| **A** | `Boxmot(...)` | Built-in (yolov8n, yolo11s, yolox_x, rf-detr-base, …) | Built-in (botsort, boosttrack, deepocsort, strongsort, ocsort, bytetrack, imprassoc, hybridsort, sfsort) | Official BoxMOT (HOTA / MOTA / IDF1 …) |
| **B** | `TrackingPipeline(...)` | Anything (SAHI+YOLO, your own subclass of `BaseDetector`, etc.) | Any BoxMOT tracker (incl. subclasses with custom Siamese ReID) | Use Path A's `val()` after Path B writes a MOT-format txt |

---

## Table of contents

- [Install](#install)
- [Path A — `Boxmot` API (consistent metrics)](#path-a--boxmot-api-consistent-metrics)
  - [`track`](#track)
  - [`generate`](#generate)
  - [`val` / `eval`](#val--eval)
  - [`tune`](#tune)
  - [`export`](#export)
  - [Full kwarg reference](#full-kwarg-reference)
- [Path B — `TrackingPipeline` (custom detectors & trackers)](#path-b--trackingpipeline-custom-detectors--trackers)
  - [Built-in custom detector: SAHI + YOLO](#built-in-custom-detector-sahi--yolo)
  - [Plug in your own detector](#plug-in-your-own-detector)
  - [Plug in your own tracker (BotSort + Siamese ReID)](#plug-in-your-own-tracker-botsort--siamese-reid)
  - [Mixing custom detector + custom tracker](#mixing-custom-detector--custom-tracker)
- [Recipes](#recipes)
  - [Benchmark several trackers on MOT17-ablation](#benchmark-several-trackers-on-mot17-ablation)
  - [Cache embeddings once, eval many trackers fast](#cache-embeddings-once-eval-many-trackers-fast)
  - [Save annotated video + crops + MOT txt + trajectories](#save-annotated-video--crops--mot-txt--trajectories)
  - [Track only a single ID](#track-only-a-single-id)
  - [Per-class tracking](#per-class-tracking)
  - [Tune `botsort` for 50 trials on MOT17-ablation](#tune-botsort-for-50-trials-on-mot17-ablation)
  - [Export ReID model to ONNX + TensorRT](#export-reid-model-to-onnx--tensorrt)
  - [Score a Path-B run with BoxMOT's metrics](#score-a-path-b-run-with-boxmots-metrics)

---

## Install

```bash
pip install -r requirements.txt
```

`requirements.txt` pulls `boxmot`, `ultralytics`, `sahi`, `opencv-python`,
`numpy`, `tqdm`, `PyYAML`. Model weights download automatically on first use.

---

## Path A — `Boxmot` API (consistent metrics)

This is the recommended path when you want results that match BoxMOT's
published numbers. Internally it shells out to BoxMOT's official engine
(`python -m boxmot.engine.cli`), so every flag is forwarded verbatim.

```python
from src import Boxmot

bm = Boxmot(
    detector="yolov8n",                  # see "Built-in detectors" below
    reid="osnet_x0_25_msmt17",           # any model from BoxMOT's REID zoo
    tracker="botsort",                   # see "Built-in trackers" below
    device="cuda:0",                     # 'cpu', 'cuda:0', '0', '0,1'
    half=False,                          # FP16 inference
)
```

**Built-in detectors** (just pass the bare name — `.pt` is appended automatically):
`yolov8n/s/m/l/x`, `yolov8n-seg`, `yolov8n-pose`, `yolo11n/s/m/l/x`, `yolo12n`,
`yolov9c`, `yolov10n`, `yolox_n/s/m/l/x`, `rf-detr-base`, plus the official
ablation detectors `yolox_x_MOT17_ablation`, `yolox_x_MOT20_ablation`,
`yolox_x_dancetrack_ablation`, `yolox_x_visdrone`.

**Built-in ReID** (lightweight to heavy):
`osnet_x0_25_msmt17`, `osnet_x0_25_market1501`, `mobilenetv2_x1_4_msmt17`,
`resnet50_msmt17`, `osnet_x1_0_msmt17`, `lmbn_n_cuhk03_d`, `lmbn_n_duke`,
`clip_market1501`, `clip_vehicleid`.

**Built-in trackers**:
`botsort`, `boosttrack`, `bytetrack`, `ocsort`, `deepocsort`, `strongsort`,
`imprassoc`, `hybridsort`, `sfsort`.

### `track`

Run detector + tracker on a video / image folder / stream / webcam.

```python
run = bm.track(
    source="video.mp4",        # or 0 (webcam), 'rtsp://...', 'https://youtu.be/...', 'path/*.jpg', 'path/'
    save=True,                 # write annotated mp4
    save_txt=True,             # write MOT-Challenge format predictions
    save_crop=True,            # save per-track crops
    save_trajectories=True,    # write trajectory file
    show=False,                # display window (set False for headless)
    show_trajectories=True,    # draw trail on the video
    show_lost=False,           # draw boxes for lost tracks (Kalman-only)
    show_kf_preds=False,       # draw Kalman-filter predictions
    show_labels=True,
    show_conf=True,
    line_width=2,
    classes=[0, 1],            # COCO class IDs to keep (omit = all classes)
    per_class=False,           # separate ID space per class
    target_id=None,            # int → highlight just this ID
    conf=0.3, iou=0.7, imgsz=640,
    vid_stride=1,
    project="outputs/track",
    name="exp1", exist_ok=True,
    verbose=True,
)
print(run)            # TrackRun(save_dir=..., video=..., txt=..., elapsed=...)
print(run.video)      # Path to annotated mp4
print(run.txt)        # Path to MOT-format predictions
```

### `generate`

Pre-compute detections + ReID embeddings so you can re-run several trackers
without paying the detection cost again.

```python
# (a) Cache for an official benchmark
cache = Boxmot(detector="yolov8n", reid="osnet_x0_25_msmt17").generate(
    benchmark="mot17-ablation",
    project="outputs/cache", name="mot17",
)
print(cache.cache_dir)               # outputs/cache/mot17
print(cache.detections_dir)          # outputs/cache/mot17/dets
print(cache.embeddings_dir)          # outputs/cache/mot17/embs
print(cache.timings)                 # {'elapsed': 123.4}

# (b) Cache for your own dataset (any folder of MOT-style sequences)
cache = Boxmot(detector="yolov8n", reid="osnet_x0_25_msmt17").generate(
    source="path/to/dataset",
    project="outputs/cache", name="mydata",
)
```

### `val` / `eval`

Evaluate `(detector, reid, tracker)` on a benchmark and parse the metrics.

```python
result = Boxmot(
    detector="yolox_x_MOT17_ablation",
    reid="lmbn_n_duke",
    tracker="boosttrack",
).val(
    benchmark="mot17-ablation",
    postprocessing="gbrc",      # 'gsi' | 'gbrc' | None
    verbose=True,
    project="outputs/eval", name="boost_mot17",
)

print(result.metrics)           # {'HOTA': 67.4, 'MOTA': 78.2, 'IDF1': 80.1, ...}
print(result.save_dir)          # outputs/eval/boost_mot17
```

Available benchmarks: `mot17-ablation`, `mot20-ablation`, `dancetrack-ablation`,
`visdrone-ablation`, `MOT17`, `MOT20`, `MOT17-mini`. Custom datasets work too —
just pass `source="path/to/dataset"` instead of `benchmark=...`, and structure
the folder MOT-Challenge style.

`bm.eval(...)` is an alias for `bm.val(...)`.

### `tune`

Evolutionary tracker hyperparameter search.

```python
out_dir = Boxmot(
    detector="yolov8n", reid="osnet_x0_25_msmt17", tracker="botsort"
).tune(
    benchmark="mot17-ablation",
    n_trials=50,
    objective="HOTA",          # 'HOTA' | 'MOTA' | 'IDF1'
    project="outputs/tune", name="botsort_mot17",
)
print(out_dir)                  # best params written to tracker yaml inside
```

### `export`

Export a ReID model to ONNX / OpenVINO / TensorRT / TorchScript.

```python
Boxmot.export(
    weights="osnet_x0_25_msmt17.pt",
    include=["onnx", "engine"],   # any of: onnx, openvino, engine, torchscript
    dynamic=True,
    device="cuda:0",
)
```

### Full kwarg reference

Anything below is a kwarg you can pass to `track / generate / val / tune /
export`. Snake-case maps automatically to BoxMOT's kebab-case CLI flag.

| kwarg | type | applies to | description |
|-------|------|-----------|-------------|
| `source` | str / Path / int | track, generate | video, folder, glob, URL, RTSP, webcam (`0`) |
| `benchmark` | str | generate, val, tune | `mot17-ablation`, `mot20-ablation`, `dancetrack-ablation`, `visdrone-ablation`, `MOT17`, `MOT20` |
| `detector` | str | all | overrides constructor value |
| `reid` | str | all | overrides constructor value |
| `tracker` | str | track, val, tune | overrides constructor value |
| `device` | str | all | `cpu`, `cuda:0`, `0`, `0,1` |
| `half` | bool | all | FP16 inference |
| `imgsz` | int | track, generate, val | detector input size |
| `conf` | float | track, generate, val | detection confidence threshold |
| `iou` | float | track, generate, val | NMS IoU threshold |
| `classes` | list[int] | track, generate, val | keep only these COCO class IDs |
| `agnostic_nms` | bool | track, generate, val | class-agnostic NMS |
| `vid_stride` | int | track | process every Nth frame |
| `batch_size` | int | generate, val | inference batch size |
| `n_threads` | int | generate, val | dataloader / eval threads |
| `save` | bool | track | write annotated mp4 |
| `save_txt` | bool | track | write MOT-format predictions |
| `save_crop` | bool | track | save per-track crops |
| `save_trajectories` | bool | track | write trajectory file |
| `show` | bool | track | display window |
| `show_trajectories` | bool | track | draw trails |
| `show_lost` | bool | track | draw boxes for lost tracks |
| `show_kf_preds` | bool | track | draw Kalman predictions |
| `show_labels` / `hide_labels` | bool | track | label visibility |
| `show_conf` / `hide_conf` | bool | track | confidence visibility |
| `line_width` | int | track | bbox line width |
| `per_class` | bool | track, val | per-class ID space |
| `target_id` | int | track | highlight a single ID |
| `postprocessing` | str | val | `gsi` or `gbrc` |
| `gsi` | bool | val | legacy alias for `postprocessing=gsi` |
| `eval_existing` | bool | val | evaluate an already-saved `runs/.../labels` folder |
| `split` | str | val | `train` or `test` |
| `n_trials` | int | tune | optuna trials |
| `objective` | str | tune | `HOTA`, `MOTA`, `IDF1` |
| `weights` | str / Path | export | ReID `.pt` |
| `include` | list[str] | export | `onnx`, `openvino`, `engine`, `torchscript` |
| `dynamic` | bool | export | dynamic input size |
| `project` | str / Path | all | output root (default `runs/<mode>`) |
| `name` | str | all | run subfolder (default `exp`) |
| `exist_ok` | bool | all | allow overwriting an existing run dir |
| `verbose` | bool | all | verbose logging |
| `extra` | dict | all | escape hatch — `{"--any-future-flag": value}` |

The `extra={...}` dict is forwarded verbatim, so any flag BoxMOT adds in the
future works with no code change:

```python
bm.track(source="0", show=True, extra={"--n-threads": 4, "--agnostic-nms": True})
```

---

## Path B — `TrackingPipeline` (custom detectors & trackers)

When you need **SAHI sliced inference**, **a Faster-RCNN detector**, **a
custom Siamese ReID network**, or anything else that BoxMOT's CLI doesn't
ship out of the box, use the Python pipeline.

The contract: a detector returns `(N, 6)` arrays `[x1,y1,x2,y2,conf,cls]`,
and a tracker is any BoxMOT class instance.

### Built-in custom detector: SAHI + YOLO

Already implemented in `src/detectors/sahi_yolo.py`.

```python
from src import TrackingPipeline

pipe = TrackingPipeline(
    detector_cfg={
        "type": "sahi_yolo",
        "model_path": "yolov8n.pt",
        "device": "cuda:0",
        "confidence_threshold": 0.3,
        "classes": [0],
        "sahi": {
            "slice_height": 640, "slice_width": 640,
            "overlap_height_ratio": 0.2, "overlap_width_ratio": 0.2,
            "model_type": "ultralytics",
        },
    },
    tracker_cfg={
        "type": "botsort",
        "reid_weights": "osnet_x0_25_msmt17.pt",
        "device": "cuda:0", "half": False,
    },
)

result = pipe.run(
    source="video.mp4",       # video file or image folder (auto-detected)
    output_dir="outputs/sahi_botsort",
    save_video=True,          # uses BoxMOT's built-in tracker.plot_results()
    save_mot=True,            # writes MOT-Challenge format txt
    show_trajectories=True,
    run_name="sahi_botsort",
)
print(result.video_path, result.mot_path)
```

### Plug in your own detector

```python
import numpy as np
from src import BaseDetector, register_detector, TrackingPipeline

class MyDetector(BaseDetector):
    def __init__(self, model_path, device="cuda:0"):
        # load your model here
        self.device = device

    def detect(self, frame: np.ndarray) -> np.ndarray:
        # ... your inference ...
        # MUST return shape (N, 6) = [x1, y1, x2, y2, conf, cls]
        return np.array([[100, 200, 300, 400, 0.9, 0]], dtype=np.float32)

register_detector("mydet", MyDetector)        # name now available in YAML

pipe = TrackingPipeline(
    detector_cfg={"type": "mydet", "model_path": "weights.pt"},
    tracker_cfg={"type": "ocsort"},
)
pipe.run(source="video.mp4", output_dir="outputs/custom")
```

### Plug in your own tracker (BotSort + Siamese ReID)

See `examples/custom_siamese_reid.py` for the full template. The pattern:

```python
from boxmot import BotSort
from src import register_tracker

class BotSortSiamese(BotSort):
    """BotSort with your custom appearance model."""
    def __init__(self, reid_weights, device="cuda:0", half=False, **kw):
        super().__init__(reid_weights=reid_weights, device=device, half=half, **kw)
        # swap out the appearance backbone with your own here
        # self.model = MySiameseReID(weights=reid_weights, device=device)

register_tracker("botsort_siamese", BotSortSiamese)
```

Then YAML:
```yaml
tracker:
  type: botsort_siamese
  reid_weights: weights/my_siamese.pt
  device: cuda:0
```

### Mixing custom detector + custom tracker

Just combine the two registrations and instantiate `TrackingPipeline` as
normal — same code path, both layers fully pluggable.

```python
register_detector("mydet", MyDetector)
register_tracker("botsort_siamese", BotSortSiamese)

pipe = TrackingPipeline(
    detector_cfg={"type": "mydet", "model_path": "det.pt"},
    tracker_cfg={"type": "botsort_siamese",
                 "reid_weights": "siamese.pt",
                 "device": "cuda:0"},
)
pipe.run(source="video.mp4", output_dir="outputs/full_custom")
```

---

## Recipes

### Benchmark several trackers on MOT17-ablation

```bash
python scripts/compare_trackers.py --config configs/compare.yaml
```

`configs/compare.yaml`:
```yaml
detector: yolox_x_MOT17_ablation
reid: lmbn_n_duke
benchmark: mot17-ablation
trackers: [botsort, boosttrack, strongsort, deepocsort, bytetrack, ocsort]
val:
  postprocessing: gbrc
  verbose: true
```

The script does what you already wrote:

```python
for tracker in cfg["trackers"]:
    bm = Boxmot(detector=detector, reid=reid, tracker=tracker,
                device=device, half=half)
    result = bm.val(benchmark=benchmark, **val_extras)
    rows.append({"tracker": tracker, **result.metrics})
    print(f"\n[{tracker}] {result}")
```

### Cache embeddings once, eval many trackers fast

```python
from src import Boxmot

# 1. Generate detections + embeddings ONCE
Boxmot(detector="yolov8n", reid="osnet_x0_25_msmt17").generate(
    benchmark="mot17-ablation", project="outputs/cache", name="mot17"
)

# 2. Reuse for every tracker — BoxMOT's val auto-detects the cache
for trk in ["botsort", "boosttrack", "deepocsort", "ocsort", "bytetrack"]:
    print(trk, Boxmot(detector="yolov8n", reid="osnet_x0_25_msmt17",
                      tracker=trk).val(benchmark="mot17-ablation").metrics)
```

### Save annotated video + crops + MOT txt + trajectories

```python
Boxmot(detector="yolov8n", reid="osnet_x0_25_msmt17", tracker="botsort").track(
    source="video.mp4",
    save=True, save_txt=True, save_crop=True, save_trajectories=True,
    show_trajectories=True, show_labels=True, show_conf=True,
    project="outputs/track", name="all_outputs", exist_ok=True,
)
```

### Track only a single ID

```python
Boxmot(detector="yolov8n", reid="osnet_x0_25_msmt17", tracker="deepocsort").track(
    source="video.mp4", target_id=7, save=True,
)
```

### Per-class tracking

Each class keeps its own ID counter:

```python
Boxmot(detector="yolov8n", tracker="botsort").track(
    source="video.mp4", per_class=True, save=True, save_txt=True,
)
```

### Tune `botsort` for 50 trials on MOT17-ablation

```python
Boxmot(detector="yolov8n", reid="osnet_x0_25_msmt17", tracker="botsort").tune(
    benchmark="mot17-ablation", n_trials=50, objective="HOTA",
    project="outputs/tune", name="botsort_hota",
)
```

### Export ReID model to ONNX + TensorRT

```python
Boxmot.export(
    weights="osnet_x0_25_msmt17.pt",
    include=["onnx", "engine"], dynamic=True, device="cuda:0",
)
```

### Score a Path-B run with BoxMOT's metrics

Path B writes MOT-Challenge format predictions. Drop them into a BoxMOT
`runs/.../labels` layout and use `eval_existing=True`:

```python
# Run custom pipeline first
from src import TrackingPipeline
pipe = TrackingPipeline(
    detector_cfg={"type": "sahi_yolo", "model_path": "yolov8n.pt", "classes": [0]},
    tracker_cfg={"type": "botsort", "reid_weights": "osnet_x0_25_msmt17.pt"},
)
pipe.run(source="MOT17/train/MOT17-04-FRCNN/img1",
         output_dir="outputs/runs/sahi_botsort/labels",
         run_name="MOT17-04-FRCNN")

# Then evaluate with BoxMOT's official metrics
from src import Boxmot
result = Boxmot().val(
    benchmark="MOT17",
    eval_existing=True,
    project="outputs/runs", name="sahi_botsort",
    split="train",
)
print(result.metrics)
```

---

## VisDrone-MOT (train / val / test-dev)

VisDrone-MOT ships sequences as folders of jpgs plus per-sequence
annotation files. Two things to know:

1. **The annotation format is *not* MOT-Challenge.** It has extra columns
   (object category, truncation, occlusion) and uses `score=0` to mark
   ignored regions. The framework converts it for you.
2. **BoxMOT bundles a `visdrone-ablation` benchmark** with the official
   `yolox_x_visdrone` detector + `lmbn_n_duke` ReID — use that when you
   want numbers comparable to the BoxMOT leaderboard.

### Expected layout on disk

```
/data/VisDrone/
├── VisDrone2019-MOT-train/
│   ├── sequences/uav0000013_00000_v/0000001.jpg ...
│   └── annotations/uav0000013_00000_v.txt
├── VisDrone2019-MOT-val/
│   ├── sequences/...
│   └── annotations/...
└── VisDrone2019-MOT-test-dev/
    ├── sequences/...
    └── annotations/...        # GT released for test-dev → local scoring
```

VisDrone categories (1-indexed; we drop `0=ignored-regions` and `11=others`):

```
1 pedestrian   2 people   3 bicycle   4 car   5 van
6 truck   7 tricycle   8 awning-tricycle   9 bus   10 motor
```

### A) Run BoxMOT's official VisDrone ablation benchmark

Easiest, most consistent metrics. Uses the bundled detector + ReID:

```python
from src import Boxmot

bm = Boxmot(detector="yolox_x_visdrone", reid="lmbn_n_duke", tracker="botsort")
print(bm.val(benchmark="visdrone-ablation", postprocessing="gbrc").metrics)
```

Or sweep several trackers via the existing `compare_trackers.py` with the
provided config:

```bash
python scripts/compare_trackers.py --config configs/visdrone_ablation.yaml
```

### B) Run on the FULL train / val / test-dev splits

The `visdrone_run.py` script iterates sequences and tracks each one. Use
`--mode boxmot` to invoke the BoxMOT engine per sequence, or
`--mode custom` to use SAHI+YOLO (good for small drone targets).

```bash
# Path A — official ablation detector, val split, with evaluation
python scripts/visdrone_run.py \
    --root /data/VisDrone --split val \
    --mode boxmot \
    --detector yolox_x_visdrone --reid lmbn_n_duke --tracker botsort \
    --classes 1 2 4 5 6 9 --per-class \
    --evaluate

# Path B — SAHI sliced inference, test-dev, with evaluation
python scripts/visdrone_run.py \
    --root /data/VisDrone --split test-dev \
    --mode custom \
    --detector yolov8n.pt --reid-weights osnet_x0_25_msmt17.pt \
    --tracker botsort --imgsz 1280 --slice-size 640 \
    --classes 1 2 4 5 6 9 --per-class \
    --evaluate
```

…or drive everything from a YAML:

```bash
python scripts/run_from_config.py --config configs/visdrone_train.yaml
python scripts/run_from_config.py --config configs/visdrone_val.yaml
python scripts/run_from_config.py --config configs/visdrone_testdev.yaml
```

What `--evaluate` does:
1. Calls `VisDroneMOT.export_motchallenge_gt()` to convert each sequence's
   VisDrone annotation into MOT-Challenge `gt.txt` (drops ignored regions,
   maps occlusion → visibility).
2. Builds the TrackEval directory layout (`<benchmark>-<split>/<seq>/gt/gt.txt`
   + `seqinfo.ini` + `seqmaps/`).
3. Calls `Boxmot.val(eval_existing=True)` so the metrics come from BoxMOT's
   own engine — no parallel scoring code.

### Just convert the GT (no tracking)

If you already have predictions and only need a TrackEval-ready GT folder:

```python
from src.datasets import VisDroneMOT
ds = VisDroneMOT("/data/VisDrone", split="val")
ds.export_motchallenge_gt("outputs/visdrone/gt", benchmark_name="VisDrone-MOT")
# -> outputs/visdrone/gt/VisDrone-MOT-val/<seq>/gt/gt.txt
#    outputs/visdrone/gt/seqmaps/VisDrone-MOT-val.txt
```

### Tune a tracker on VisDrone

```python
Boxmot(detector="yolox_x_visdrone", reid="lmbn_n_duke", tracker="botsort").tune(
    benchmark="visdrone-ablation", n_trials=30, objective="HOTA",
    project="outputs/visdrone/tune", name="botsort_visdrone",
)
```

### Generate (cache detections + ReID once)

```python
Boxmot(detector="yolox_x_visdrone", reid="lmbn_n_duke").generate(
    benchmark="visdrone-ablation",
    project="outputs/visdrone/cache", name="ablation",
)
# Subsequent val() runs reuse the cache → much faster sweeps.
```

### Train your own detector for VisDrone

If COCO YOLO isn't matching VisDrone's class set well enough, fine-tune
Ultralytics on VisDrone-DET first (the per-image detection split), then
point the framework at the resulting `.pt` file. Ultralytics ships a
ready-made `VisDrone.yaml` for that.

---

## CLI cheat-sheet (raw BoxMOT — same behavior, no Python)

If you'd rather skip the wrapper, every `Boxmot(...)` method maps 1:1 to a
BoxMOT CLI invocation:

```bash
# track
boxmot track --detector yolov8n --reid osnet_x0_25_msmt17 --tracker botsort \
             --source video.mp4 --save --save-txt --show-trajectories \
             --classes 0,1 --per-class --target-id 7 \
             --project outputs/track --name exp1 --exist-ok --verbose

# generate (cache dets + embs)
boxmot generate --detector yolov8n --reid osnet_x0_25_msmt17 \
                --benchmark mot17-ablation --project outputs/cache

# eval
boxmot eval --detector yolox_x_MOT17_ablation --reid lmbn_n_duke \
            --tracker boosttrack --benchmark mot17-ablation \
            --postprocessing gbrc --verbose

# tune
boxmot tune --detector yolov8n --reid osnet_x0_25_msmt17 --tracker botsort \
            --benchmark mot17-ablation --n-trials 50

# export
boxmot export --weights osnet_x0_25_msmt17.pt --include onnx --include engine --dynamic
```

The `Boxmot` Python API is just a thin shell around these — use whichever
suits your workflow.
