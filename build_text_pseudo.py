"""
Generate pseudo-boxes for dealer_name, model_name, horse_power, asset_cost from the gold set
by aligning gold field values to OCR tokens. Also builds a YOLO dataset for these fields and
draws overlays for inspection.

Outputs:
  - annotations/seed/done/text_pseudo_labels/<doc>.json  (pseudo boxes)
  - annotations/seed/done/marked_images_text/<doc>_overlay.png  (pseudo boxes drawn)
  - yolo_data_text/{train,val}/{images,labels}
  - yolo_data_text/data.yaml

Usage:
  python3 build_text_pseudo.py --ocr both
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from paddleocr import PaddleOCR
import easyocr
from PIL import Image, ImageDraw
from tqdm import tqdm


random.seed(42)

ROOT = Path(__file__).resolve().parent
DONE_IMG = ROOT / "annotations" / "seed" / "done" / "images"
DONE_LAB = ROOT / "annotations" / "seed" / "done" / "labels"
PSEUDO_LAB = ROOT / "annotations" / "seed" / "done" / "text_pseudo_labels"
OVERLAY_DIR = ROOT / "annotations" / "seed" / "done" / "marked_images_text"
YOLO_ROOT = ROOT / "yolo_data_text"
TRAIN_RATIO = 0.8
CLASSES = ["dealer_name", "model_name", "horse_power", "asset_cost"]


def box_from_quad(quad):
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]


def fuzzy_match_tokens(value: str, tokens: List[Tuple[str, float, list]], min_ratio=0.55):
    import difflib

    best = None
    for text, conf, box in tokens:
        ratio = difflib.SequenceMatcher(None, value.lower(), text.lower()).ratio()
        score = ratio * conf
        if best is None or score > best[0]:
            best = (score, text, box)
    if best and best[0] >= min_ratio:
        return best[2]
    return None


def match_numeric(target: Optional[float], tokens: List[Tuple[str, float, list]], tol_frac=0.4):
    if target is None:
        return None
    best = None
    for text, conf, box in tokens:
        digits = re.sub(r"[^0-9.]", "", text)
        if not digits:
            continue
        try:
            val = float(digits)
        except Exception:
            continue
        if abs(val - target) / max(abs(target), 1.0) <= tol_frac:
            score = conf * (1.0 - abs(val - target) / max(abs(target), 1.0))
            if best is None or score > best[0]:
                best = (score, box)
    return best[1] if best else None


def run_ocr(img_path: Path, use_paddle: bool, use_easy: bool):
    tokens = []  # (text, conf, box_xyxy)
    if use_paddle:
        paddle = PaddleOCR(lang="en", use_angle_cls=True, show_log=False, use_gpu=True)
        out = paddle.ocr(str(img_path), cls=True)
        for page in out:
            for line in page:
                text = line[1][0]
                conf = float(line[1][1])
                box = box_from_quad(line[0])
                tokens.append((text, conf, box))
    if use_easy:
        reader = easyocr.Reader(["en"], gpu=True)
        out = reader.readtext(str(img_path), detail=1, paragraph=False, min_size=10, text_threshold=0.4)
        for quad, text, conf in out:
            box = box_from_quad(quad)
            tokens.append((text, float(conf), box))
    return tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr", type=str, default="both", choices=["paddle", "easy", "both"])
    args = parser.parse_args()
    use_paddle = args.ocr in ("paddle", "both")
    use_easy = args.ocr in ("easy", "both")

    PSEUDO_LAB.mkdir(parents=True, exist_ok=True)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

    entries = []
    for lab in tqdm(list(DONE_LAB.glob("*.json")), desc="Gold pseudo"):
        data = json.loads(lab.read_text())
        f = data.get("fields", {})
        img_path = DONE_IMG / f"{lab.stem}.png"
        if not img_path.exists():
            continue
        tokens = run_ocr(img_path, use_paddle, use_easy)
        dealer_box = fuzzy_match_tokens(f.get("dealer_name", "") or "", tokens, min_ratio=0.55)
        model_box = fuzzy_match_tokens(f.get("model_name", "") or "", tokens, min_ratio=0.5)
        hp_box = match_numeric(f.get("horse_power", None), tokens, tol_frac=0.3)
        cost_box = match_numeric(f.get("asset_cost", None), tokens, tol_frac=0.3)
        pseudo = {
            "doc_id": lab.stem,
            "dealer_name": dealer_box,
            "model_name": model_box,
            "horse_power": hp_box,
            "asset_cost": cost_box,
        }
        (PSEUDO_LAB / lab.name).write_text(json.dumps(pseudo, indent=2))
        # overlay
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        colors = {"dealer_name": "green", "model_name": "purple", "horse_power": "orange", "asset_cost": "cyan"}
        for k, box in pseudo.items():
            if k == "doc_id" or box is None:
                continue
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline=colors.get(k, "yellow"), width=4)
        img.save(OVERLAY_DIR / f"{lab.stem}_overlay.png")
        # collect for YOLO
        boxes = []
        for cls_name, box in [("dealer_name", dealer_box), ("model_name", model_box), ("horse_power", hp_box), ("asset_cost", cost_box)]:
            if box:
                boxes.append((CLASSES.index(cls_name), box))
        if boxes:
            entries.append((img_path, boxes))
    print("pseudo labels written:", len(list(PSEUDO_LAB.glob('*.json'))))

    # YOLO dataset
    random.shuffle(entries)
    split = int(len(entries) * TRAIN_RATIO)
    splits = {"train": entries[:split], "val": entries[split:]}
    for split_name, subset in splits.items():
        img_dir = YOLO_ROOT / split_name / "images"
        lab_dir = YOLO_ROOT / split_name / "labels"
        for d in [img_dir, lab_dir]:
            if d.exists():
                for f in d.glob("*"):
                    f.unlink()
            d.mkdir(parents=True, exist_ok=True)
        for img_path, boxes in subset:
            img = Image.open(img_path)
            w, h = img.size
            (img_dir / img_path.name).write_bytes(img_path.read_bytes())
            lines = []
            for cls_id, (x1, y1, x2, y2) in boxes:
                cx = (x1 + x2) / 2 / w
                cy = (y1 + y2) / 2 / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            (lab_dir / f"{img_path.stem}.txt").write_text("\n".join(lines))

    (YOLO_ROOT / "data.yaml").write_text(
        f"path: {YOLO_ROOT.resolve()}\ntrain: train/images\nval: val/images\nnames: {CLASSES}\n"
    )
    print(f"YOLO text dataset built with {len(entries)} images -> {YOLO_ROOT}")


if __name__ == "__main__":
    main()
