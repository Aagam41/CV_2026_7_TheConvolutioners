# MOT Framework — BoxMOT-first

A clean, plug-and-play multi-object tracking project built on top of
**BoxMOT**, **Ultralytics**, and **SAHI**.

Two ways to use it, both fully supported:

| Path | When | What you get |
|------|------|--------------|
| **A. `Boxmot` API** | Built-in detectors (yolov8/yolo11/yolox/rf-detr) + built-in trackers | BoxMOT-consistent metrics (HOTA / MOTA / IDF1) via the official engine |
| **B. `TrackingPipeline`** | Custom detectors (SAHI+YOLO, your own) and/or custom trackers (BotSort + your Siamese ReID) | Full Python control, BoxMOT trackers via direct class API |

For commands, flags and recipes for **track / generate / val / tune / export**
on both built-in and custom components — and a dedicated **VisDrone-MOT
(train / val / test-dev)** section — see **[usage.md](usage.md)**.

## Install

```bash
pip install -r requirements.txt
```

## Layout

```
mot_framework/
├── usage.md                       # full command + flag reference
├── configs/
│   ├── single.yaml                # one tracker (Path A)
│   ├── compare.yaml               # benchmark several trackers (Path A)
│   ├── visdrone_ablation.yaml     # BoxMOT visdrone-ablation benchmark
│   ├── visdrone_train.yaml        # full train split (Path B)
│   ├── visdrone_val.yaml          # full val split + eval
│   └── visdrone_testdev.yaml      # full test-dev split + eval
├── scripts/
│   ├── run_tracking.py            # uses Boxmot.track() / .val()
│   ├── compare_trackers.py        # uses Boxmot.generate() + .val()
│   ├── visdrone_run.py            # iterate every sequence in a VisDrone split
│   └── run_from_config.py         # YAML → visdrone_run argv driver
├── examples/
│   ├── custom_siamese_reid.py     # subclass BotSort with your Siamese net
│   └── sahi_botsort_pipeline.py   # SAHI+YOLO → BoxMOT tracker (Path B)
├── src/
│   ├── boxmot_api.py              # `Boxmot` wrapper (Path A)
│   ├── pipeline.py                # `TrackingPipeline` (Path B)
│   ├── detectors/                 # YoloDetector, SahiYoloDetector
│   ├── trackers/factory.py        # build_tracker() for BoxMOT classes
│   ├── sources/                   # video / image-folder source
│   ├── datasets/                  # VisDrone-MOT loader + GT converter
│   └── visualizer.py
└── tests/
    ├── smoke_test.py              # offline argv-construction tests
    └── _run_offline.py            # runs smoke test with stubbed deps
```

## 30-second example

```python
from src import Boxmot

# Path A — official engine, consistent metrics
bm = Boxmot(detector="yolov8n", reid="osnet_x0_25_msmt17", tracker="botsort")
run = bm.track(source="video.mp4", save=True, save_txt=True)
print(run.video, run.txt)

metrics = bm.val(benchmark="mot17-ablation", postprocessing="gbrc").metrics
print(metrics)            # {'HOTA': ..., 'MOTA': ..., 'IDF1': ..., ...}
```

```python
# Path B — SAHI+YOLO detector + any BoxMOT tracker
from src import TrackingPipeline

pipe = TrackingPipeline(
    detector_cfg={"type": "sahi_yolo", "model_path": "yolov8n.pt", "classes": [0]},
    tracker_cfg={"type": "botsort", "reid_weights": "osnet_x0_25_msmt17.pt"},
)
pipe.run(source="video.mp4", output_dir="outputs/sahi_botsort")
```

See **[usage.md](usage.md)** for everything else.
