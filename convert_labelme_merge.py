"""
Convert labelme JSONs in annotations/seed/in_progress/labels into the target schema
by merging manual stamp/signature boxes with OCR-filled text fields.

Usage:
  python3 convert_labelme_merge.py [--refresh]

Default: only processes files that look like labelme (have "shapes"/"imageData").
Use --refresh to overwrite existing schema files as well.
"""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LABEL_DIR = ROOT / "annotations" / "seed" / "in_progress" / "labels"
IMAGE_DIR = ROOT / "annotations" / "seed" / "in_progress" / "images"
BACKUP_DIR = ROOT / "annotations" / "seed" / "backup" / "labels"
OVERLAY_DIR = ROOT / "overlays"


def is_labelme(data: dict) -> bool:
    return "shapes" in data and "imageData" in data


def is_schema(data: dict) -> bool:
    return "fields" in data and isinstance(data["fields"], dict)


def rect_bbox(shape: dict):
    pts = shape.get("points", [])
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)]


def process_file(label_path: Path, force: bool = False):
    try:
        data = json.load(label_path.open())
    except Exception as e:
        print(f"Failed to load {label_path.name}: {e}")
        return

    if not force and is_schema(data):
        return

    sig_bbox = None
    stamp_bbox = None
    if is_labelme(data):
        for sh in data.get("shapes", []):
            label = (sh.get("label") or "").strip().lower()
            if label == "signature":
                sig_bbox = rect_bbox(sh)
            elif label == "stamp":
                stamp_bbox = rect_bbox(sh)
    else:
        # schema-looking but forcing refresh; keep existing boxes if present
        fields = data.get("fields", {})
        sig = fields.get("signature", {})
        stamp = fields.get("stamp", {})
        sig_bbox = sig.get("bbox")
        stamp_bbox = stamp.get("bbox")

    img_path = IMAGE_DIR / f"{label_path.stem}.png"
    if not img_path.exists():
        print(f"Image missing for {label_path.name}, skipping")
        return

    # Start with backup schema if available; else empty template.
    backup_path = BACKUP_DIR / label_path.name
    base_fields = None
    notes = ""
    if backup_path.exists():
        try:
            backup = json.load(backup_path.open())
            if is_schema(backup):
                base_fields = backup.get("fields", {})
                notes = backup.get("notes", "")
        except Exception:
            pass
    if base_fields is None:
        base_fields = {
            "dealer_name": "",
            "model_name": "",
            "horse_power": None,
            "asset_cost": None,
            "signature": {"present": False, "bbox": None},
            "stamp": {"present": False, "bbox": None},
        }

    # Start with base/backup values; do NOT run OCR—leave blanks if not in backup.
    filled = {
        "doc_id": label_path.stem,
        "fields": base_fields.copy(),
        "source": "labelme_manual",
        "notes": notes,
    }

    filled["fields"]["signature"] = {"present": bool(sig_bbox), "bbox": sig_bbox if sig_bbox else None}
    filled["fields"]["stamp"] = {"present": bool(stamp_bbox), "bbox": stamp_bbox if stamp_bbox else None}
    filled["source"] = "labelme_manual"

    with label_path.open("w") as f:
        json.dump(filled, f, indent=2, ensure_ascii=False)
    return filled


def write_overlay(label_path: Path, data: dict):
    img_path = IMAGE_DIR / f"{label_path.stem}.png"
    if not img_path.exists():
        return False
    try:
        from PIL import Image, ImageDraw  # noqa: WPS433
    except Exception:
        return False
    OVERLAY_DIR.mkdir(exist_ok=True)
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    sig = data.get("fields", {}).get("signature", {})
    if sig.get("present") and sig.get("bbox"):
        x1, y1, x2, y2 = map(float, sig["bbox"])
        draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
    stamp = data.get("fields", {}).get("stamp", {})
    if stamp.get("present") and stamp.get("bbox"):
        x1, y1, x2, y2 = map(float, stamp["bbox"])
        draw.rectangle([x1, y1, x2, y2], outline="blue", width=4)
    img.save(OVERLAY_DIR / f"{label_path.stem}_overlay.png")
    return True


def main():
    parser = argparse.ArgumentParser(description="Merge labelme JSON into target schema with backup fields (no OCR).")
    parser.add_argument("--refresh", action="store_true", help="Force processing even if already in schema format.")
    parser.add_argument("--overlays", action="store_true", help="Also regenerate overlay PNGs (red sig, blue stamp).")
    args = parser.parse_args()

    count = 0
    overlay_count = 0
    for lp in LABEL_DIR.glob("*.json"):
        merged = process_file(lp, force=args.refresh)
        if args.overlays and merged:
            if write_overlay(lp, merged):
                overlay_count += 1
        count += 1
        if count % 25 == 0:
            print(f"Processed {count}")
    msg = f"Done. Processed {count} files."
    if args.overlays:
        msg += f" Overlays: {overlay_count} -> {OVERLAY_DIR}"
    print(msg)


if __name__ == "__main__":
    main()
