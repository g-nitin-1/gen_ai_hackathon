"""
Build a patch dataset for stamp/signature classification from the done labels.

Creates:
  patch_data/train/{signature,stamp,background}
  patch_data/val/{signature,stamp,background}

Positives: crops from stamp/signature boxes in annotations/seed/done/labels.
Negatives: random crops from the same images that do not overlap positives.
"""

import json
import random
from pathlib import Path

from PIL import Image


random.seed(42)

DONE_IMG = Path("annotations/seed/done/images")
DONE_LAB = Path("annotations/seed/done/labels")
OUT = Path("patch_data")

# background crops per image
NEG_PER_IMAGE = 3
# train/val split ratio
TRAIN_RATIO = 0.8


def crop_patch(img: Image.Image, bbox, pad_ratio=0.1):
    x1, y1, x2, y2 = bbox
    w, h = img.size
    pw = (x2 - x1) * pad_ratio
    ph = (y2 - y1) * pad_ratio
    x1 = max(0, x1 - pw)
    y1 = max(0, y1 - ph)
    x2 = min(w, x2 + pw)
    y2 = min(h, y2 + ph)
    return img.crop((x1, y1, x2, y2))


def main():
    entries = []
    for lab in DONE_LAB.glob("*.json"):
        data = json.load(lab.open())
        fields = data.get("fields", {})
        sig = fields.get("signature", {})
        stamp = fields.get("stamp", {})
        boxes = []
        if sig.get("present") and sig.get("bbox"):
            boxes.append(("signature", sig["bbox"]))
        if stamp.get("present") and stamp.get("bbox"):
            boxes.append(("stamp", stamp["bbox"]))
        if not boxes:
            continue
        img_path = DONE_IMG / f"{lab.stem}.png"
        if not img_path.exists():
            continue
        entries.append((img_path, boxes))

    print("images with boxes:", len(entries))
    random.shuffle(entries)
    split = int(len(entries) * TRAIN_RATIO)
    for split_name, subset in [("train", entries[:split]), ("val", entries[split:])]:
        for cls in ["signature", "stamp", "background"]:
            (OUT / split_name / cls).mkdir(parents=True, exist_ok=True)
        for img_path, boxes in subset:
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            pos_masks = []
            for cls, box in boxes:
                crop = crop_patch(img, box)
                fname = f"{img_path.stem}_{cls}.png"
                crop.save(OUT / split_name / cls / fname)
                # keep for negative avoidance
                x1, y1, x2, y2 = box
                pos_masks.append((x1, y1, x2, y2))
            # negatives
            for i in range(NEG_PER_IMAGE):
                for _ in range(10):  # retry to avoid overlap
                    bw = random.uniform(0.1, 0.3) * w
                    bh = random.uniform(0.05, 0.2) * h
                    cx = random.uniform(bw / 2, w - bw / 2)
                    cy = random.uniform(h * 0.4, h - bh / 2)
                    x1 = cx - bw / 2
                    y1 = cy - bh / 2
                    x2 = cx + bw / 2
                    y2 = cy + bh / 2
                    overlap = False
                    for px1, py1, px2, py2 in pos_masks:
                        if not (x2 < px1 or px2 < x1 or y2 < py1 or py2 < y1):
                            overlap = True
                            break
                    if not overlap:
                        crop = crop_patch(img, (x1, y1, x2, y2))
                        fname = f"{img_path.stem}_bg{i}.png"
                        crop.save(OUT / split_name / "background" / fname)
                        break
    print("patch dataset built at", OUT)


if __name__ == "__main__":
    main()
