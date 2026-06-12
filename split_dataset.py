"""
split_dataset.py
────────────────
Splits a flat Roboflow dataset into the train/val folder structure
that YOLOv8 expects.

Expected input layout:
    dataset/
    ├── images/          ← all .jpg / .png files flat in here
    └── labels/          ← all matching .txt files flat in here

Output layout after running:
    dataset/
    ├── images/
    │   ├── train/       ← 80% of images  (default)
    │   └── val/         ← 20% of images  (default)
    └── labels/
        ├── train/       ← matching .txt files for train images
        └── val/         ← matching .txt files for val images

Usage:
    python split_dataset.py --dataset /path/to/dataset
    python split_dataset.py --dataset /path/to/dataset --split 0.8

    # Example on Windows:
    python split_dataset.py --dataset C:/Users/you/dataset

    # Example on Mac/Linux:
    python split_dataset.py --dataset ~/Downloads/dataset
"""

import os
import shutil
import random
import argparse
from pathlib import Path


def split_dataset(dataset_dir: str, train_ratio: float = 0.8, seed: int = 42):
    dataset_path = Path(dataset_dir)
    images_dir   = dataset_path / "images"
    labels_dir   = dataset_path / "labels"

    # ── Validate input paths ──────────────────────────────────────────────────
    if not images_dir.exists():
        raise FileNotFoundError(f"images/ folder not found in: {dataset_path}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"labels/ folder not found in: {dataset_path}")

    # ── Collect all image files ───────────────────────────────────────────────
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = sorted([
        f for f in images_dir.iterdir()
        if f.suffix.lower() in image_extensions and f.is_file()
    ])

    if not image_files:
        raise ValueError(f"No image files found in {images_dir}")

    print(f"Found {len(image_files)} images total.")

    # ── Shuffle and split ─────────────────────────────────────────────────────
    random.seed(seed)
    random.shuffle(image_files)

    split_index  = int(len(image_files) * train_ratio)
    train_images = image_files[:split_index]
    val_images   = image_files[split_index:]

    print(f"  → Train : {len(train_images)} images ({train_ratio*100:.0f}%)")
    print(f"  → Val   : {len(val_images)} images ({(1-train_ratio)*100:.0f}%)")

    # ── Create output folders ─────────────────────────────────────────────────
    for split in ["train", "val"]:
        (images_dir / split).mkdir(parents=True, exist_ok=True)
        (labels_dir / split).mkdir(parents=True, exist_ok=True)

    # ── Move files ────────────────────────────────────────────────────────────
    missing_labels = []

    def move_pair(image_path: Path, split: str):
        label_path = labels_dir / (image_path.stem + ".txt")

        # Move image
        shutil.move(str(image_path), str(images_dir / split / image_path.name))

        # Move label if it exists
        if label_path.exists():
            shutil.move(str(label_path), str(labels_dir / split / label_path.name))
        else:
            missing_labels.append(image_path.name)

    for img in train_images:
        move_pair(img, "train")

    for img in val_images:
        move_pair(img, "val")

    # ── Report results ────────────────────────────────────────────────────────
    print("\nDone! Files moved successfully.")

    if missing_labels:
        print(f"\nWarning: {len(missing_labels)} images had no matching label file:")
        for name in missing_labels[:10]:
            print(f"  - {name}")
        if len(missing_labels) > 10:
            print(f"  ... and {len(missing_labels) - 10} more.")

    # ── Remind about data.yaml ────────────────────────────────────────────────
    yaml_path = dataset_path / "data.yaml"
    print("\n── data.yaml check ──────────────────────────────────────────────────")
    if yaml_path.exists():
        print(f"Found data.yaml at {yaml_path}")
        print("Make sure it contains:")
    else:
        print(f"No data.yaml found. Create one at {yaml_path} with:")

    print(f"""
path: {dataset_path.resolve()}
train: images/train
val: images/val

nc: <number of classes>
names: ["class1", "class2", ...]   # replace with your actual class names
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a flat YOLO dataset into train/val.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to your dataset/ folder — must contain images/ and labels/ subfolders"
    )
    parser.add_argument(
        "--split",
        type=float,
        default=0.8,
        help="Fraction of data to use for training (default: 0.8 = 80/20 split)"
    )
    args = parser.parse_args()

    split_dataset(dataset_dir=args.dataset, train_ratio=args.split)
