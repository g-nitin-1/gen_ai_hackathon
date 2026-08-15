"""
Build YOLO dataset from a specific train/test split.

Usage:
  python3 build_yolo_from_split.py --split-dir splits/split_395_100
"""

import argparse
import json
import shutil
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent

def norm_bbox(x1, y1, x2, y2, w, h):
    cx = (x1 + x2) / 2 / w
    cy = (y1 + y2) / 2 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return cx, cy, bw, bh

def reset_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", required=True, help="Split directory path")
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    train_img_dir = split_dir / "train" / "images"
    train_lab_dir = split_dir / "train" / "labels"

    out_dir = split_dir / "yolo_data"

    # Process train data (80% for YOLO train, 20% for YOLO val)
    entries = []
    for lab_path in train_lab_dir.glob("*.json"):
        data = json.loads(lab_path.read_text())
        fields = data.get("fields", {})
        sig = fields.get("signature", {})
        stamp = fields.get("stamp", {})

        boxes = []
        if sig.get("present") and sig.get("bbox"):
            boxes.append((0, sig["bbox"]))  # class 0: signature
        if stamp.get("present") and stamp.get("bbox"):
            boxes.append((1, stamp["bbox"]))  # class 1: stamp

        if not boxes:
            continue

        img_path = train_img_dir / f"{lab_path.stem}.png"
        if not img_path.exists():
            continue

        entries.append((img_path, boxes))

    # Split 80/20 for YOLO internal train/val
    import random
    random.seed(42)
    random.shuffle(entries)
    split_idx = int(len(entries) * 0.8)
    splits = {"train": entries[:split_idx], "val": entries[split_idx:]}

    print(f"Building YOLO dataset from {split_dir.name}")
    print(f"  Total entries with boxes: {len(entries)}")
    print(f"  YOLO train: {len(splits['train'])}")
    print(f"  YOLO val: {len(splits['val'])}")

    for split_name, subset in splits.items():
        img_dir = out_dir / split_name / "images"
        lab_dir = out_dir / split_name / "labels"
        reset_dir(img_dir)
        reset_dir(lab_dir)

        for img_path, boxes in subset:
            img = Image.open(img_path)
            w, h = img.size
            shutil.copy2(img_path, img_dir / img_path.name)

            lines = []
            for cls_id, (x1, y1, x2, y2) in boxes:
                cx, cy, bw, bh = norm_bbox(float(x1), float(y1), float(x2), float(y2), w, h)
                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            (lab_dir / f"{img_path.stem}.txt").write_text("\n".join(lines))

    # Create data.yaml
    yaml_content = (
        f"path: {out_dir.resolve()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "names: [signature, stamp]\n"
    )
    (out_dir / "data.yaml").write_text(yaml_content)

    print(f"\n✓ YOLO dataset created at: {out_dir}")
    print(f"  data.yaml: {out_dir / 'data.yaml'}")

if __name__ == "__main__":
    main()
