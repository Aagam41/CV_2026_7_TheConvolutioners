DATASET=data
SPLIT=VisDrone2019-MOT-val
OUT=det_eval

# --- baselines (no SAHI) ---
python detection_eval.py --dataset $DATASET --split $SPLIT \
  --weights yolov8n.pt --imgsz 1280 --output $OUT/yolov8n --save_video

python detection_eval.py --dataset $DATASET --split $SPLIT \
  --weights yolo11n.pt --imgsz 1280 --output $OUT/yolo11n --save_video

python detection_eval.py --dataset $DATASET --split $SPLIT \
  --weights yolo11_visdrone.pt --imgsz 1280 --no_class_map \
  --output $OUT/yolo11_visdrone --save_video

# --- SAHI hyperparam sweep on plain YOLO11n ---
for sh in 512 640 800; do
  for ov in 0.1 0.2 0.3; do
    python detection_eval.py --dataset $DATASET --split $SPLIT \
      --weights yolo11_visdrone.pt --imgsz 1280 \
      --sahi --slice_h $sh --slice_w $sh --overlap_h $ov --overlap_w $ov \
      --output $OUT/sahi_yolo11n_${sh}_${ov} --no_class_map --save_video
  done
done

