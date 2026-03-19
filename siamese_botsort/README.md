# Siamese BoT-SORT

Custom BoT-SORT multi-object tracker with a Siamese ResNet-18 ReID model, evaluated on VisDrone.

---

## Data

The `data/` subfolders are **not included** in this repository.  
Download the VisDrone dataset from the official source: https://github.com/VisDrone/VisDrone-Dataset

Expected layout after download:

```
data/
  VisDrone2019-MOT-train/
    sequences/
    annotations/
  VisDrone2019-VID-val/
    sequences/
    annotations/
  VisDrone2019-VID-test-dev/
    sequences/
    annotations/
```

---

## Order of Execution

### 1. Crop objects from VisDrone MOT training data

```
python crop.py
```

Reads `data/VisDrone2019-MOT-train` annotations, crops each annotated object from its frame, and saves per-identity image folders under `crops/train/`.

---

### 2. Train the Siamese ReID model

```
python train.py
```

Trains a Siamese ResNet-18 with contrastive loss on the crops.  
Outputs saved to `eval_siamese/`:

- `siamese_final.pth` — trained weights (saved to repo root)
- `metrics_log.csv` — per-epoch metrics
- `training_metrics.png` — loss and accuracy curves
- `distance_histogram.png` — embedding distance distribution

---

### 3. Run the BoT-SORT tracker (demo / visualise)

```
python botsort.py data/VisDrone2019-VID-val/sequences/<sequence_name>
```

Runs the tracker on a single image sequence using YOLOv8 detections and Siamese ReID.  
Saves annotated frames under `runs/<sequence_name>/` and an MP4 to `runs/<sequence_name>.mp4`.

---

### 4. Evaluate tracker on VisDrone VID

```
python eval_botsort.py --split val
```

Runs end-to-end MOT evaluation (YOLOv8 + BotSort) on the selected split.  
Results saved to `eval_botsort/` as `overall_metrics_XXX.json` and `per_sequence_metrics_XXX.json` (auto-incrementing run IDs).

Common options:

```
python eval_botsort.py --split both               # val + test-dev
python eval_botsort.py --split val --max-seqs 1   # quick smoke test
```
