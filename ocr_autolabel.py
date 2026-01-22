"""
Heuristic auto-labeling using Tesseract OCR.

Reads templates from annotations/seed/todo/{images,labels}, runs OCR, fills basic
fields, and writes filled JSONs plus image symlinks into annotations/seed/in_progress.

Dependencies (user to install manually if network is blocked):
  sudo apt-get install -y tesseract-ocr
  python3 -m pip install --user pytesseract pillow
"""

import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, Optional

from PIL import Image
import pytesseract
import cv2
import numpy as np
import argparse


ROOT = Path(__file__).resolve().parent
TODO_IMAGES = ROOT / "annotations" / "seed" / "todo" / "images"
TODO_LABELS = ROOT / "annotations" / "seed" / "todo" / "labels"
OUT_IMAGES = ROOT / "annotations" / "seed" / "in_progress" / "images"
OUT_LABELS = ROOT / "annotations" / "seed" / "in_progress" / "labels"


def ensure_dirs() -> None:
    OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    OUT_LABELS.mkdir(parents=True, exist_ok=True)


def ocr_text(image_path: Path) -> Dict:
    """Run Tesseract OCR, returning both full text and per-word data."""
    # full text for regex
    text = pytesseract.image_to_string(Image.open(image_path))
    # word-level data for potential future use
    data = pytesseract.image_to_data(Image.open(image_path), output_type=pytesseract.Output.DICT)
    return {"text": text, "data": data}


def extract_hp(text: str) -> Optional[int]:
    """
    Heuristic: look for patterns like 50 HP, 50HP, 50-Hp, or mis-OCR variants (5O HP).
    """
    patterns = [
        r"([0-9O]{2,3})\\s*[-_]??\\s*H\\s*P",
        r"([0-9O]{2,3})\\s*[-_]??\\s*HP",
        r"([0-9O]{2,3})\\s*HP",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            raw = m.group(1).replace("O", "0")
            try:
                return int(raw)
            except ValueError:
                continue
    return None


def extract_asset_cost(text: str) -> Optional[int]:
    """
    Heuristic: grab all long digit sequences (allowing commas/periods) and pick the largest.
    """
    cleaned = text.replace(",", "").replace(".", "")
    nums = []
    for n in re.findall(r"(\\d{5,})", cleaned):
        try:
            nums.append(int(n))
        except ValueError:
            continue
    return max(nums) if nums else None


def extract_hp_from_data(data: Dict) -> Optional[int]:
    words = data.get("text", [])
    for idx, tok in enumerate(words):
        if not tok:
            continue
        num = re.fullmatch(r"[0-9O]{2,3}", tok)
        if num:
            nxt = words[idx + 1].lower() if idx + 1 < len(words) else ""
            nxt2 = words[idx + 2].lower() if idx + 2 < len(words) else ""
            if "hp" in nxt or "h.p" in nxt or "hp" in nxt2:
                try:
                    return int(tok.replace("O", "0"))
                except ValueError:
                    continue
    return None


def extract_cost_from_data(data: Dict) -> Optional[int]:
    words = data.get("text", [])
    nums = []
    for tok in words:
        if not tok:
            continue
        cleaned = re.sub(r"[^0-9]", "", tok)
        if len(cleaned) >= 5:
            try:
                nums.append(int(cleaned))
            except ValueError:
                continue
    return max(nums) if nums else None


def extract_line_value(text: str, keyword: str) -> str:
    """
    Return the substring to the right of the keyword on the same line if present.
    Very rough heuristic; caller should strip the result.
    """
    for line in text.splitlines():
        if keyword.lower() in line.lower():
            # split on keyword and punctuation separators
            parts = re.split(keyword, line, flags=re.IGNORECASE, maxsplit=1)
            if len(parts) == 2:
                val = parts[1]
                # trim common separators
                val = re.sub(r"^[\\s:|-]+", "", val).strip()
                return val
    return ""


def fill_fields(base_label: Dict, ocr: Dict, force: bool = False) -> Dict:
    text = ocr["text"]
    data = ocr["data"]
    filled = base_label.copy()
    fields = filled.get("fields", {}).copy()

    # Dealer/model from keyworded lines if possible.
    dealer = extract_line_value(text, "dealer")
    model = extract_line_value(text, "model")

    hp = extract_hp(text) or extract_hp_from_data(data)
    cost = extract_asset_cost(text) or extract_cost_from_data(data)

    # Sanity bounds
    if hp is not None and not (15 <= hp <= 120):
        hp = None
    if cost is not None and not (10000 <= cost <= 20000000):
        cost = None

    if not force:
        hp = hp if fields.get("horse_power") is None else fields.get("horse_power")
        cost = cost if fields.get("asset_cost") is None else fields.get("asset_cost")

    fields.update(
        {
            "dealer_name": dealer or fields.get("dealer_name", ""),
            "model_name": model or fields.get("model_name", ""),
            "horse_power": hp,
            "asset_cost": cost,
            # stamp/signature remain false/null; visual detection not attempted here.
            "signature": fields.get("signature", {"present": False, "bbox": None}),
            "stamp": fields.get("stamp", {"present": False, "bbox": None}),
        }
    )
    filled["fields"] = fields
    filled["source"] = "auto_heuristic_tesseract"
    filled.setdefault("notes", "")
    if not filled["notes"]:
        filled["notes"] = "auto-filled; please verify"
    return filled


def detect_signature_stamp(img_path: Path) -> Dict[str, Dict]:
    """
    Very rough CV heuristic:
    - Convert to grayscale, adaptive threshold to find ink blobs.
    - Filter contours by area relative to image, location (prefer bottom 40%).
    - Signature: elongated blobs (w/h > 2.5) near bottom.
    - Stamp: more square-ish blobs (0.5 < w/h < 1.8) near bottom.
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {
            "signature": {"present": False, "bbox": None},
            "stamp": {"present": False, "bbox": None},
        }
    h, w = img.shape
    img_area = h * w

    # Adaptive threshold to isolate ink
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 10)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        if area < 0.001 * img_area or area > 0.05 * img_area:
            continue
        cx = x + cw / 2
        cy = y + ch / 2
        candidates.append({"bbox": (x, y, x + cw, y + ch), "area": area, "cx": cx, "cy": cy, "ratio": cw / ch})

    signature_bbox = None
    stamp_bbox = None

    # Prefer blobs in bottom 40% of the image
    bottom_candidates = [c for c in candidates if c["cy"] > 0.6 * h]

    # Signature: elongated
    sig_cands = [c for c in bottom_candidates if c["ratio"] > 2.5]
    if sig_cands:
        signature_bbox = max(sig_cands, key=lambda c: c["area"])["bbox"]

    # Stamp: squarish
    stamp_cands = [c for c in bottom_candidates if 0.5 < c["ratio"] < 1.8]
    if stamp_cands:
        stamp_bbox = max(stamp_cands, key=lambda c: c["area"])["bbox"]

    return {
        "signature": {"present": signature_bbox is not None, "bbox": list(signature_bbox) if signature_bbox else None},
        "stamp": {"present": stamp_bbox is not None, "bbox": list(stamp_bbox) if stamp_bbox else None},
    }


def process_one(label_path: Path, force: bool) -> None:
    img_path = TODO_IMAGES / f"{label_path.stem}.png"
    if not img_path.exists():
        print(f"Image missing for {label_path.name}, skipping")
        return

    # Skip if already processed
    out_label = OUT_LABELS / label_path.name
    if out_label.exists() and not force:
        return

    with open(label_path, "r") as f:
        base_label = json.load(f)

    ocr = ocr_text(img_path)
    filled = fill_fields(base_label, ocr, force=force)

    # Add stamp/signature guesses
    sig_stamp = detect_signature_stamp(img_path)
    filled["fields"]["signature"] = sig_stamp["signature"]
    filled["fields"]["stamp"] = sig_stamp["stamp"]
    filled["notes"] = (filled.get("notes") or "") + " | signature/stamp via CV heuristic"

    with open(out_label, "w") as f:
        json.dump(filled, f, indent=2, ensure_ascii=False)

    # Create/refresh symlink in in_progress images
    OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    target_img = OUT_IMAGES / img_path.name
    if target_img.exists() or target_img.is_symlink():
        target_img.unlink()
    target_img.symlink_to(img_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-label with Tesseract + CV heuristics.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N files.")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N files before processing.")
    parser.add_argument("--refresh", action="store_true", help="Rewrite outputs even if they exist.")
    args = parser.parse_args()

    ensure_dirs()
    label_files = sorted(TODO_LABELS.glob("*.json"))
    if args.offset:
        label_files = label_files[args.offset:]
    if args.limit is not None:
        label_files = label_files[: args.limit]

    total = len(label_files)
    for i, label_file in enumerate(label_files, 1):
        process_one(label_file, force=args.refresh)
        if i % 25 == 0:
            print(f"Processed {i}/{total}")
    print(f"Done. Filled labels in {OUT_LABELS} (processed {total} files)")


if __name__ == "__main__":
    main()
