# VisDrone MOT — YOLO + SAHI + BoxMOT

End-to-end Multi-Object Tracking pipeline for VisDrone UAV footage using:

| Layer | Library | Notes |
|---|---|---|
| **Detection** | Ultralytics YOLO | YOLOv8/9/10/11/12 (or YOLOX, RT-DETR) |
| **Sliced inference** | [SAHI](https://github.com/obss/sahi) | Dramatically improves small-object recall on high-res drone frames |
| **Tracking** | [BoxMOT](https://github.com/mikel-brostrom/boxmot) | BotSort (motion + appearance) and ByteTrack (motion-only) |

---

## Directory layout

```
Codes/                          ← YOU ARE HERE
├── detector.py                 # YOLOSAHIDetector wrapper
├── visdrone_utils.py           # Sequence reader, MOT writer, visualisation
├── track_visdrone.py           # Main CLI tracking script  ← start here
├── test_detector.py            # Quick detector smoke-test
├── requirements.txt
└── README.md

../dataset/VisDrone/            ← dataset (sibling of Codes/)
    VisDrone2019-MOT-train/
        annotations/
        sequences/
            uav0000013_00000_v/
                0000001.jpg …
    VisDrone2019-MOT-val/
        annotations/
        sequences/
            …
```

---

## Installation

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install CUDA-enabled PyTorch first (adjust for your CUDA version)
#    Skip this line to use CPU-only PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install all other dependencies
cd Codes/
pip install -r requirements.txt
```

---

## Quick start

### 1. Smoke-test the detector on a single frame

```bash
python test_detector.py \
    --seq-dir ../dataset/VisDrone/VisDrone2019-MOT-val/sequences/uav0000086_00000_v \
    --yolo-model yolov8n.pt \
    --device cpu \
    --show
```

This checks that YOLO + SAHI are installed correctly and that the
detector produces `(N, 6)` arrays before touching any tracker.

### 2. Track the full validation split with BotSort

```bash
python track_visdrone.py \
    --split val \
    --tracker botsort \
    --yolo-model yolov8n.pt \
    --reid-model osnet_x0_25_msmt17.pt \
    --device cuda:0
```

The ReID model is auto-downloaded from the BoxMOT model zoo on first run.

### 3. Track the full validation split with ByteTrack (motion-only, much faster)

```bash
python track_visdrone.py \
    --split val \
    --tracker bytetrack \
    --yolo-model yolov8n.pt \
    --device cuda:0
```

ByteTrack needs no ReID model — ideal when speed is critical.

### 4. Single sequence, both trackers

```bash
SEQ=uav0000013_00000_v

python track_visdrone.py --split train --seq $SEQ --tracker botsort \
    --yolo-model yolov8s.pt --reid-model osnet_x0_25_msmt17.pt --device 0

python track_visdrone.py --split train --seq $SEQ --tracker bytetrack \
    --yolo-model yolov8s.pt --device 0
```

---

## All command-line flags

```
usage: track_visdrone.py [-h]
  Dataset:
    --dataset-root DIR   Root of the VisDrone dataset (default: ../dataset/VisDrone)
    --split {train,val}  Which split to process (default: val)
    --seq SEQ            Single sequence name; if omitted all sequences are processed

  Detector (YOLO + SAHI):
    --yolo-model FILE    Path to YOLO weights (default: yolov8n.pt)
    --model-type STR     SAHI model type: yolov8|yolov5|yolox|rtdetr (default: yolov8)
    --conf FLOAT         Detector confidence threshold (default: 0.25)
    --iou FLOAT          NMS / postmerge IoU threshold (default: 0.45)
    --slice-h INT        SAHI slice height in pixels (default: 640)
    --slice-w INT        SAHI slice width in pixels (default: 640)
    --overlap-h FLOAT    Vertical overlap ratio (default: 0.2)
    --overlap-w FLOAT    Horizontal overlap ratio (default: 0.2)
    --classes INT [...]  Keep only these VisDrone class IDs (default: all)
                         0=pedestrian 1=people 2=bicycle 3=car 4=van
                         5=truck 6=tricycle 7=awning-tricycle 8=bus 9=motor

  Tracker:
    --tracker {botsort,bytetrack}   Tracker to use (default: botsort)
    --reid-model FILE               ReID weights for BotSort (auto-downloaded if absent)

  Hardware:
    --device STR         Torch device: cpu | cuda:0 | 0 (default: cpu)
    --half               Use FP16 (GPU only)

  Output:
    --output-dir DIR     Root output directory (default: runs/track)
    --no-save-video      Do not write annotated .mp4 files
    --no-save-txt        Do not write MOT-format .txt files
    --show               Display frames in an OpenCV window
    --fps FLOAT          Output video frame rate (default: 30.0)
    --no-draw-det        Skip drawing raw detections; only draw tracks
```

---

## Outputs

```
Codes/runs/track/
├── botsort/
│   ├── uav0000013_00000_v.mp4   ← annotated video (white=raw dets, coloured=tracks+IDs)
│   ├── uav0000013_00000_v.txt   ← MOT-format tracking result
│   └── …
└── bytetrack/
    ├── uav0000013_00000_v.mp4
    ├── uav0000013_00000_v.txt
    └── …
```

### MOT .txt format

Each line:
```
<frame>,<id>,<left>,<top>,<width>,<height>,<conf>,-1,-1,-1
```
This is standard MOT Challenge / VisDrone devkit format, compatible with
**TrackEval** and the official VisDrone evaluation toolkit.

---

## SAHI tuning guide for VisDrone

VisDrone frames are typically **1920×1080** or **2000×1500** with very small
objects. Recommended SAHI settings:

| Scenario | `--slice-h` | `--slice-w` | `--overlap-h/w` | Notes |
|---|---|---|---|---|
| Default / balanced | 640 | 640 | 0.2 | Good starting point |
| Densely packed small objects | 512 | 512 | 0.3 | More slices, higher recall |
| Speed-optimised | 960 | 960 | 0.1 | Fewer slices, ~2× faster |
| Full-image only (no SAHI) | equal to frame height/width | — | 0.0 | Disables slicing |

---

## Tracker selection guide

| | **BotSort** | **ByteTrack** |
|---|---|---|
| Type | Motion + Appearance | Motion-only |
| ReID model | Required | Not needed |
| Accuracy (HOTA on MOT17) | 69.4 | 67.7 |
| FPS | ~12 | ~720 |
| Best for | Crowded scenes, occlusions, identity preservation | High-throughput, reliable detections |

For UAV video with many small, dense objects, **ByteTrack** is often the
better starting point due to its speed. Upgrade to **BotSort** if you need
robust re-identification after occlusions.

---

## How it works

```
┌─────────────┐     BGR frame     ┌──────────────────────────┐
│  VisDrone   │ ──────────────▶   │    YOLOSAHIDetector       │
│  Sequence   │                   │  YOLO on 640×640 slices   │
│  (.jpg)     │                   │  NMM postmerge            │
└─────────────┘                   └──────────┬───────────────┘
                                             │ (N,6) [x1,y1,x2,y2,conf,cls]
                                  ┌──────────▼───────────────┐
                                  │   BoxMOT Tracker          │
                                  │  BotSort  or  ByteTrack   │
                                  │  tracker.update(dets,img) │
                                  └──────────┬───────────────┘
                                             │ (M,8) [x1,y1,x2,y2,id,conf,cls,ind]
                          ┌──────────────────┼──────────────────┐
                          ▼                  ▼                  ▼
                    MOT .txt file     annotated .mp4       console stats
```

---

## Using the boxmot CLI directly (plain YOLO, no SAHI)

If you want to quickly compare against plain YOLO (no SAHI slicing), you
can still use the standard BoxMOT CLI:

```bash
# BotSort
boxmot track \
    --yolo-model yolov8n.pt \
    --reid-model osnet_x0_25_msmt17.pt \
    --tracking-method botsort \
    --source ../dataset/VisDrone/VisDrone2019-MOT-val/sequences/ \
    --device 0

# ByteTrack
boxmot track \
    --yolo-model yolov8n.pt \
    --tracking-method bytetrack \
    --source ../dataset/VisDrone/VisDrone2019-MOT-val/sequences/ \
    --device 0
```

The custom `track_visdrone.py` adds SAHI on top of this same flow.
