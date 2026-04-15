# BoT-SORT with Random-Forest HSV ReID (VisDrone)

Files
- `rf_train.py`   train the Random-Forest same-identity classifier on HSV histograms
- `bot_sort_rf.py` the tracker (BoT-SORT core, FastReID replaced by RF)
- `track.py`      run detector + tracker on one sequence, write annotated MP4
- `eval.py`       run tracker over a split and compute MOT metrics

## 1. Install

```bash
pip install -r requirements.txt
```

## 2. Train the Random Forest (once)

```bash
python rf_train.py \
  --dataset /path/to/dataset \
  --split VisDrone2019-MOT-train \
  --out rf_hsv.pkl
```

The script crops every GT object, computes an 8×8×8 HSV histogram (512-D),
samples ~60k positive and ~60k negative pairs, and trains a
`RandomForestClassifier` on `|feat_a − feat_b|` with label {same_id, different_id}.
At tracking time, appearance distance = `1 − P(same_id)`.

## 3. Track one sequence

```bash
python track.py \
  --source /path/to/dataset/VisDrone2019-MOT-val/sequences/uav0000009_03358_v \
  --yolo yolo11_visdrone.pt \
  --rf rf_hsv.pkl \
  --output output.mp4 \
  --results output.txt
```

`--source` also accepts a video file.

## 4. Evaluate a full split

```bash
python eval.py \
  --dataset /path/to/dataset \
  --split VisDrone2019-MOT-val \
  --yolo yolo11_visdrone.pt \
  --rf rf_hsv.pkl \
  --output eval_out
```

This produces:
- `eval_out/videos/<seq>.mp4`     annotated videos
- `eval_out/results/<seq>.txt`    MOTChallenge-format tracks
- `eval_out/metrics.csv`          MOTA, IDF1, MOTP, MT/ML, IDs, Frag, ...

### HOTA
`motmetrics` ships HOTA only in some builds. If `hota` is not present in the
printed table, install [TrackEval](https://github.com/JonathonLuiten/TrackEval)
and point it at `eval_out/results/` + the VisDrone GT folder. The MOT-format
results written by `eval.py` are directly compatible with TrackEval's
`MotChallenge2DBox` dataset class.

## Notes on the RF-replaces-FastReID design

BoT-SORT's FastReID block outputs a 2048-D embedding; similarity is cosine.
Here the "embedding" is a normalized HSV histogram and the similarity comes
from the trained RF (pairwise). The tracker code keeps the standard
BoT-SORT machinery:

- Kalman predict/update on `(cx, cy, w, h, vx, vy, vw, vh)`
- Two-stage association (high-conf dets → low-conf dets)
- EMA-smoothed appearance feature per track (`alpha = 0.9`)
- IoU gate (`proximity_thresh`) and appearance gate (`appearance_thresh`)
- Track-buffer–based lost-track retirement

Camera Motion Compensation (GMC) from BoT-SORT is not included — VisDrone is
aerial and the KF + appearance cues usually suffice. It can be added as a
separate affine warp of track means between frames if needed.
