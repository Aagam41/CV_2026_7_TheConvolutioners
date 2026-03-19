from __future__ import annotations

import argparse
from pathlib import Path

import cv2


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(sequence_dir: Path) -> list[Path]:
    images = [
        path for path in sorted(sequence_dir.iterdir())
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    ]
    if not images:
        raise ValueError(f"No image files found in {sequence_dir}")
    return images


def convert_sequence(sequence_dir: Path, output_path: Path, fps: float) -> None:
    images = list_images(sequence_dir)
    first_frame = cv2.imread(str(images[0]))
    if first_frame is None:
        raise ValueError(f"Could not read first frame: {images[0]}")

    height, width = first_frame.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_path}")

    try:
        for image_path in images:
            frame = cv2.imread(str(image_path))
            if frame is None:
                print(f"[Warn] Skipping unreadable frame: {image_path}")
                continue

            frame_height, frame_width = frame.shape[:2]
            if frame_width != width or frame_height != height:
                frame = cv2.resize(frame, (width, height))

            writer.write(frame)
    finally:
        writer.release()


def convert_all_sequences(runs_dir: Path, output_dir: Path, fps: float) -> None:
    subdirs = sorted(path for path in runs_dir.iterdir() if path.is_dir())
    if not subdirs:
        raise ValueError(f"No sequence folders found in {runs_dir}")

    for sequence_dir in subdirs:
        output_path = output_dir / f"{sequence_dir.name}.mp4"
        print(f"[Info] Converting {sequence_dir} -> {output_path}")
        convert_sequence(sequence_dir, output_path, fps)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert image-sequence folders into mp4 videos"
    )
    parser.add_argument(
        "--runs-dir",
        default="runs",
        help="Root folder containing sequence subfolders",
    )
    parser.add_argument(
        "--sequence-dir",
        default="",
        help="Optional single sequence folder to convert",
    )
    parser.add_argument(
        "--output-dir",
        default="runs_mp4",
        help="Folder where mp4 files will be written",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frames per second for the generated video",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    output_dir = Path(args.output_dir)

    if args.sequence_dir:
        sequence_dir = Path(args.sequence_dir)
        output_path = output_dir / f"{sequence_dir.name}.mp4"
        print(f"[Info] Converting {sequence_dir} -> {output_path}")
        convert_sequence(sequence_dir, output_path, args.fps)
        return

    convert_all_sequences(runs_dir, output_dir, args.fps)


if __name__ == "__main__":
    main()