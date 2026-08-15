"""
Split dataset into train/test sets with configurable ratio.
Creates separate directories for train and test annotations.

Usage:
  python3 split_dataset.py --train-size 395  # 395 train, 100 test
  python3 split_dataset.py --train-size 412  # 412 train, 83 test
"""

import argparse
import json
import random
import shutil
from pathlib import Path

random.seed(42)

ROOT = Path(__file__).resolve().parent
DONE_IMG = ROOT / "annotations" / "seed" / "done" / "images"
DONE_LAB = ROOT / "annotations" / "seed" / "done" / "labels"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, required=True, help="Number of training samples")
    parser.add_argument("--output-dir", type=str, default="splits", help="Output directory name")
    args = parser.parse_args()

    # Get all label files
    all_labels = sorted(DONE_LAB.glob("*.json"))
    total = len(all_labels)
    test_size = total - args.train_size

    print(f"Total samples: {total}")
    print(f"Train size: {args.train_size}")
    print(f"Test size: {test_size}")

    if args.train_size >= total:
        print("ERROR: train_size must be less than total samples")
        return

    # Shuffle and split
    random.shuffle(all_labels)
    train_labels = all_labels[:args.train_size]
    test_labels = all_labels[args.train_size:]

    # Create output directories
    split_name = f"split_{args.train_size}_{test_size}"
    out_dir = ROOT / args.output_dir / split_name

    train_img_dir = out_dir / "train" / "images"
    train_lab_dir = out_dir / "train" / "labels"
    test_img_dir = out_dir / "test" / "images"
    test_lab_dir = out_dir / "test" / "labels"

    for d in [train_img_dir, train_lab_dir, test_img_dir, test_lab_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Copy train files
    print(f"\nCopying {len(train_labels)} train files...")
    for lab_path in train_labels:
        stem = lab_path.stem
        img_path = DONE_IMG / f"{stem}.png"

        shutil.copy2(lab_path, train_lab_dir / lab_path.name)
        if img_path.exists():
            shutil.copy2(img_path, train_img_dir / img_path.name)

    # Copy test files
    print(f"Copying {len(test_labels)} test files...")
    for lab_path in test_labels:
        stem = lab_path.stem
        img_path = DONE_IMG / f"{stem}.png"

        shutil.copy2(lab_path, test_lab_dir / lab_path.name)
        if img_path.exists():
            shutil.copy2(img_path, test_img_dir / img_path.name)

    # Save split info
    split_info = {
        "train_size": args.train_size,
        "test_size": test_size,
        "train_files": [p.stem for p in train_labels],
        "test_files": [p.stem for p in test_labels]
    }
    (out_dir / "split_info.json").write_text(json.dumps(split_info, indent=2))

    print(f"\n✓ Split created at: {out_dir}")
    print(f"  Train: {train_img_dir}")
    print(f"  Test: {test_img_dir}")

if __name__ == "__main__":
    main()
