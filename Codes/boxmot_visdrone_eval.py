import os
import shutil

VISDRONE_ROOT = "/home/aagamsheth/Documents/Education/Ahmedabad University/MTECH CSE 2025 - 2027/SEM 2/CSE 641/Project/Codes/dataset/VisDrone"
OUTPUT_ROOT = "dataset"

SPLITS = {
    "train": "VisDrone2019-MOT-train",
    "val": "VisDrone2019-MOT-val",
    "test": "VisDrone2019-MOT-test-dev"
}

def convert_annotation(src_txt, dst_txt):
    with open(src_txt, 'r') as f, open(dst_txt, 'w') as out:
        for line in f.readlines():
            parts = line.strip().split(',')

            frame = int(parts[0])
            track_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            category = int(parts[7])

            # OPTIONAL: filter classes
            # if category not in [1, 4, 5, 6, 9]: continue

            conf = 1
            cls = 1
            vis = 1

            out.write(f"{frame},{track_id},{x},{y},{w},{h},{conf},{cls},{vis}\n")


def process_split(split_name, folder_name):
    split_path = os.path.join(VISDRONE_ROOT, folder_name)
    out_split = os.path.join(OUTPUT_ROOT, split_name)

    os.makedirs(out_split, exist_ok=True)

    seqs = os.listdir(os.path.join(split_path, "sequences"))

    for seq in seqs:
        print(f"Processing {split_name} - {seq}")

        seq_src = os.path.join(split_path, "sequences", seq)
        anno_src = os.path.join(split_path, "annotations", f"{seq}.txt")

        seq_dst = os.path.join(out_split, seq)
        img_dst = os.path.join(seq_dst, "img1")
        gt_dst = os.path.join(seq_dst, "gt")

        os.makedirs(img_dst, exist_ok=True)
        os.makedirs(gt_dst, exist_ok=True)

        # Copy images
        for img in sorted(os.listdir(seq_src)):
            shutil.copy(
                os.path.join(seq_src, img),
                os.path.join(img_dst, img)
            )

        # Convert annotations
        convert_annotation(
            anno_src,
            os.path.join(gt_dst, "gt.txt")
        )


if __name__ == "__main__":
    for split, folder in SPLITS.items():
        process_split(split, folder)

    print("✅ Conversion complete!")