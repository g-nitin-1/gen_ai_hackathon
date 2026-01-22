"""
Rebuild YOLO detection dataset (signature=0, stamp=1) from done labels.

Creates yolo_data/{train,val}/{images,labels} and a data.yaml that points to them.
Splits 80/20 by document.

Usage:
  python3 build_yolo_dataset.py
"""

import json
import random
import shutil
from pathlib import Path

from PIL import Image


random.seed(42)

DONE_IMG = Path("annotations/seed/done/images")
DONE_LAB = Path("annotations/seed/done/labels")
OUT_ROOT = Path("yolo_data")
TRAIN_RATIO = 0.8


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
    entries = []
    for lab in DONE_LAB.glob("*.json"):
        data = json.loads(lab.read_text())
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
        img_path = DONE_IMG / f"{lab.stem}.png"
        if not img_path.exists():
            continue
        entries.append((img_path, boxes))

    random.shuffle(entries)
    split = int(len(entries) * TRAIN_RATIO)
    splits = {"train": entries[:split], "val": entries[split:]}

    for split_name, subset in splits.items():
        img_dir = OUT_ROOT / split_name / "images"
        lab_dir = OUT_ROOT / split_name / "labels"
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

    # data.yaml
    OUT_ROOT.mkdir(exist_ok=True)
    yaml = (
        f"path: {OUT_ROOT.resolve()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "names: [signature, stamp]\n"
    )
    (OUT_ROOT / "data.yaml").write_text(yaml)
    print(f"YOLO dataset built with {len(entries)} images -> {OUT_ROOT}")


if __name__ == "__main__":
    main()
