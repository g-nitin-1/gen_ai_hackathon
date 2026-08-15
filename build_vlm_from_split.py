"""
Build VLM training JSONL from a specific train/test split.

Usage:
  python3 build_vlm_from_split.py --split-dir splits/split_395_100
"""

import argparse
import base64
import json
from pathlib import Path
from PIL import Image
import io

ROOT = Path(__file__).resolve().parent

def image_to_base64(img_path, max_size=1024):
    """Convert image to base64 string, resizing if needed."""
    img = Image.open(img_path).convert("RGB")

    # Resize if too large
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", required=True, help="Split directory path")
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    train_img_dir = split_dir / "train" / "images"
    train_lab_dir = split_dir / "train" / "labels"

    out_file = split_dir / "vlm_train.jsonl"

    system_prompt = (
        "You are a document extraction model. Read the tractor invoice/quotation image and return JSON with:\n"
        "dealer_name (string), model_name (string), horse_power (ONLY the numeric value, e.g. 45 not '45 HP'), "
        "asset_cost (ONLY the numeric value, no commas or currency symbols). "
        "If a field is missing, use empty string or null. Do not include units or text in numeric fields."
    )

    records = []
    for lab_path in sorted(train_lab_dir.glob("*.json")):
        img_path = train_img_dir / f"{lab_path.stem}.png"
        if not img_path.exists():
            continue

        data = json.loads(lab_path.read_text())
        fields = data.get("fields", {})

        # Build target JSON (text fields only for VLM)
        target = {
            "dealer_name": fields.get("dealer_name", ""),
            "model_name": fields.get("model_name", ""),
            "horse_power": fields.get("horse_power"),
            "asset_cost": fields.get("asset_cost")
        }

        # Convert image to base64
        img_b64 = image_to_base64(img_path)

        record = {
            "image_b64": img_b64,
            "system": system_prompt,
            "query": "Extract fields as JSON.",
            "response": json.dumps(target)
        }
        records.append(record)

    # Write JSONL
    with open(out_file, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"✓ VLM training data created: {out_file}")
    print(f"  Total records: {len(records)}")

if __name__ == "__main__":
    main()
