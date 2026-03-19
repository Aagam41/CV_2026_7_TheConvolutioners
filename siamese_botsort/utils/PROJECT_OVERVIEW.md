# Siamese BoT-SORT Project Overview

## Purpose

This repository is a small computer-vision pipeline built around two ideas:

1. Train a Siamese re-identification model on cropped VisDrone object images.
2. Use those appearance embeddings as part of a BoT-SORT tracking workflow.

In its current state, both the Siamese training pipeline and the BoT-SORT integration are fully implemented. `botsort.py` loads the trained Siamese model directly and uses its 128-d embeddings as an appearance cost during data association.

## End-to-End Flow

1. `crop.py` reads VisDrone MOT annotations and crops object images from frames.
2. Crops are stored in per-object folders under `crops/train`.
3. `dataset.py` samples positive and negative image pairs from those folders.
4. `model.py` defines a Siamese ResNet-18 embedding model.
5. `train.py` trains the model with contrastive loss and evaluates it with a distance threshold.
6. Training produces:
   - `siamese_final.pth`
   - `metrics_log.csv`
   - `training_metrics.png`
   - `distance_histogram.png`
7. `botsort.py` loads `siamese_final.pth` and runs a custom BoT-SORT tracker (Kalman filter + IoU + cosine ReID cost).
8. `eval_botsort.py` drives YOLOv8 detection + `BotSort` over VisDrone VID sequences and writes MOT metrics to JSON.

## File-by-File Summary

### `crop.py`

- Reads from `data/VisDrone2019-MOT-train`.
- Iterates through VisDrone sequences and annotation files.
- Crops each annotated object box from its source frame.
- Saves crops into folders named like `sequence_obj_id` under `crops/train`.

This script is the preprocessing stage that creates the folder layout expected by the training dataset.

### `dataset.py`

- Defines `SiameseDataset`.
- Assumes one folder per object identity.
- Randomly returns either:
  - a positive pair from the same folder with label `1`
  - a negative pair from two different folders with label `0`
- Resizes images to `128 x 128` and converts them to tensors.
- Uses a fixed length of `10000`, so each epoch is based on random pair sampling rather than a strict dataset size.

### `model.py`

- Defines `SiameseNet`.
- Uses pretrained ResNet-18 as a backbone.
- Removes the classification head.
- Adds a linear projection from `512` features to a `128`-dimensional embedding.
- Exposes:
  - `forward_once(x)` for one image embedding
  - `forward(x1, x2)` for paired Siamese training

### `train.py`

- Defines `ContrastiveLoss` with margin `1.0`.
- Loads cropped object folders using `SiameseDataset`.
- Trains the Siamese model using Adam.
- Evaluates the model every epoch with a threshold-based distance rule.
- Logs metrics including:
  - accuracy
  - precision
  - recall
  - F1
  - TP, TN, FP, FN
  - mean same-pair distance
  - mean different-pair distance
- Saves metrics and visualization outputs.
- Saves trained weights to `siamese_final.pth`.

It supports a validation folder if `crops/val` exists; otherwise it evaluates on the training set.

### `eval.py`

- Computes threshold-based verification metrics from embedding distances.
- Prediction rule:
  - if distance `< threshold`, predict same object
  - else predict different object
- Returns classification metrics and average distance statistics.

Used by `train.py` for per-epoch evaluation; not involved in tracking.

### `botsort.py`

- Implements a custom BoT-SORT multi-object tracker.
- Uses a constant-velocity Kalman filter for motion prediction.
- Two-stage association: high-confidence detections use IoU + cosine appearance cost; low-confidence detections use IoU only.
- Loads `siamese_final.pth` at startup; extracts 128-d L2-normalised embeddings per detection crop.
- Per-track appearance is maintained as an exponential moving average of embeddings.
- Tracks are confirmed after `min_hits` consecutive frames and pruned after `max_age` frames without a match.

### `eval_botsort.py`

- CLI script that runs end-to-end MOT evaluation on VisDrone VID splits (`val` / `test-dev`).
- Uses YOLOv8 for detection and `BotSort` from `botsort.py` for tracking.
- Computes MOTA, MOTP, IDF1, HOTA (approximated), DETA, ASSA via `motmetrics`.
- Writes `overall_metrics.json` and `per_sequence_metrics.json` to a configurable output directory.

### `sequence_to_mp4.py`

- Standalone utility that converts a folder of image frames into an MP4 video.
- Not part of the training or evaluation pipeline.

## What Is Working

- VisDrone crop extraction
- Siamese pair dataset generation
- Siamese training with contrastive loss
- Threshold-based evaluation and logging
- Saving trained weights and plots
- Custom BoT-SORT tracking with Siamese ReID appearance cost
- Full MOT evaluation on VisDrone VID sequences

## Interpreting the Existing Results

The saved `metrics_log.csv` shows that training reduced the loss substantially and generally improved pair discrimination.

Observed trend across epochs:

- loss drops sharply from the first epoch onward
- mean same-object distance decreases
- mean different-object distance stays higher than mean same-object distance
- recall reaches `1.0` in later epochs
- accuracy reaches roughly the high `0.7` range

This suggests the model learned a useful separation in embedding space, although the gap is not yet strong enough to make the tracking integration obviously production-ready.

## Current Bottom Line

This project implements a complete end-to-end pipeline: Siamese ReID training on VisDrone crops followed by custom BoT-SORT tracking using those appearance embeddings, evaluated with standard MOT metrics on the VisDrone VID benchmark.

If this repository is for a course project, the cleanest high-level description is:

> Crop objects from VisDrone MOT, train a Siamese ResNet-18 to learn appearance embeddings, integrate those embeddings into a custom BoT-SORT tracker, and evaluate multi-object tracking performance on VisDrone VID.
