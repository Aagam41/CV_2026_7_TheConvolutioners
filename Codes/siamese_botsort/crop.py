# Usage: python crop.py
import os
import cv2

DATA_PATH = "data/VisDrone2019-MOT-train"
SAVE_PATH = "crops/train"

seq_path = os.path.join(DATA_PATH, "sequences")
ann_path = os.path.join(DATA_PATH, "annotations")

os.makedirs(SAVE_PATH, exist_ok=True)

for seq in os.listdir(seq_path):
    print(f"Processing {seq}")

    seq_img_dir = os.path.join(seq_path, seq)
    seq_ann_file = os.path.join(ann_path, seq + ".txt")

    if not os.path.exists(seq_ann_file):
        continue

    with open(seq_ann_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split(',')

        frame_id = int(parts[0])
        obj_id = int(parts[1])
        x, y, w, h = map(int, parts[2:6])
        if w <= 0 or h <= 0:
            continue

        img_name = f"{frame_id:07d}.jpg"
        img_path = os.path.join(seq_img_dir, img_name)

        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        img_h, img_w = img.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(img_w, x + w)
        y2 = min(img_h, y + h)

        if x2 <= x1 or y2 <= y1:
            continue

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        obj_dir = os.path.join(SAVE_PATH, f"{seq}_obj_{obj_id}")
        os.makedirs(obj_dir, exist_ok=True)

        save_name = f"{frame_id:07d}.jpg"
        cv2.imwrite(os.path.join(obj_dir, save_name), crop)