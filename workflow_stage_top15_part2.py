"""
Part 2: Fuse VLM outputs with YOLO results and stage final top-15.

This script:
1. Loads YOLO results from tmp_yolo_results.json
2. Loads VLM outputs from annotations/seed/in_progress/labels_vlm/
3. Fuses text fields (VLM) with boxes (YOLO)
4. Ranks by YOLO confidence and selects final top-K
5. Stages to in_progress with overlays

Usage:
  python3 workflow_stage_top15_part2.py --final-top 15
"""

import argparse
import json
from pathlib import Path
import shutil

from PIL import Image, ImageDraw
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "train"
VLM_DIR = ROOT / "annotations" / "seed" / "in_progress" / "labels_vlm"
IN_PROGRESS_IMG = ROOT / "annotations" / "seed" / "in_progress" / "images"
IN_PROGRESS_LAB = ROOT / "annotations" / "seed" / "in_progress" / "labels"
OVERLAY_DIR = ROOT / "overlays"


def to_num(val):
    """Convert string to int/float, handling commas."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return val
    if isinstance(val, str):
        s = val.replace(",", "").strip()
        if not s:
            return None
        try:
            if "." in s:
                return float(s)
            return int(s)
        except Exception:
            return val
    return val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-top", type=int, default=15, help="Final top K to stage")
    parser.add_argument("--yolo-results", default="tmp_yolo_results.json")
    args = parser.parse_args()

    # Load YOLO results
    yolo_results_path = ROOT / args.yolo_results
    if not yolo_results_path.exists():
        print(f"ERROR: {yolo_results_path} not found. Run workflow_stage_top15.py first.")
        return

    print(f"Loading YOLO results from {yolo_results_path}")
    with open(yolo_results_path) as f:
        yolo_results = json.load(f)

    # Create lookup by stem
    yolo_lookup = {r["stem"]: r for r in yolo_results}

    # Load VLM outputs
    if not VLM_DIR.exists():
        print(f"ERROR: {VLM_DIR} not found. Run vlm_query.py first.")
        return

    vlm_files = list(VLM_DIR.glob("*.json"))
    print(f"Found {len(vlm_files)} VLM outputs")

    # Fuse VLM + YOLO
    print("\n=== Fusing VLM text fields with YOLO boxes ===")
    fused_results = []

    for vlm_file in tqdm(vlm_files, desc="Fusing"):
        stem = vlm_file.stem

        # Get YOLO result
        yolo_data = yolo_lookup.get(stem)
        if not yolo_data:
            continue

        # Load VLM output
        vlm_obj = json.loads(vlm_file.read_text())
        fields = vlm_obj.get("parsed") or {}

        # Extract and normalize fields
        dealer_name = fields.get("dealer_name") or ""
        model_name = fields.get("model_name") or ""
        horse_power = to_num(fields.get("horse_power"))
        asset_cost = to_num(fields.get("asset_cost"))

        # Combine with YOLO boxes
        fused = {
            "stem": stem,
            "score": yolo_data["score"],
            "doc_id": stem,
            "fields": {
                "dealer_name": dealer_name,
                "model_name": model_name,
                "horse_power": horse_power,
                "asset_cost": asset_cost,
                "signature": {
                    "present": bool(yolo_data["sig_box"]),
                    "bbox": yolo_data["sig_box"],
                    "conf": yolo_data["sig_conf"]
                },
                "stamp": {
                    "present": bool(yolo_data["stamp_box"]),
                    "bbox": yolo_data["stamp_box"],
                    "conf": yolo_data["stamp_conf"]
                },
            },
            "source": "vlm+yolo_workflow",
            "vlm_conf_logprob": vlm_obj.get("conf_logprob"),
            "yolo_score": yolo_data["score"],
        }
        fused_results.append(fused)

    # Rank by YOLO score
    fused_results.sort(key=lambda x: (-x["score"], x["stem"]))

    print(f"\nTop 10 after fusion:")
    for i, r in enumerate(fused_results[:10], 1):
        sig_present = "✓" if r["fields"]["signature"]["present"] else "✗"
        stamp_present = "✓" if r["fields"]["stamp"]["present"] else "✗"
        print(f"  {i}. {r['stem']}: score={r['score']:.3f} sig={sig_present} stamp={stamp_present}")

    # Stage final top-K
    final_top = fused_results[:args.final_top]
    print(f"\n=== Staging final top {len(final_top)} to in_progress ===")

    IN_PROGRESS_IMG.mkdir(parents=True, exist_ok=True)
    IN_PROGRESS_LAB.mkdir(parents=True, exist_ok=True)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

    for r in tqdm(final_top, desc="Staging"):
        stem = r["stem"]
        img_path = TRAIN_DIR / f"{stem}.png"

        if not img_path.exists():
            print(f"Warning: {img_path} not found, skipping")
            continue

        # Copy image
        shutil.copy2(img_path, IN_PROGRESS_IMG / img_path.name)

        # Write label JSON
        label_data = {
            "doc_id": r["doc_id"],
            "fields": r["fields"],
            "source": r["source"],
            "notes": f"YOLO score={r['score']:.3f}; VLM logprob={r['vlm_conf_logprob']:.3f}; please verify",
        }
        label_path = IN_PROGRESS_LAB / f"{stem}.json"
        label_path.write_text(json.dumps(label_data, indent=2))

        # Generate overlay
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        # Draw signature box (red)
        if r["fields"]["signature"]["bbox"]:
            x1, y1, x2, y2 = r["fields"]["signature"]["bbox"]
            draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
            # Add confidence label
            conf = r["fields"]["signature"]["conf"]
            draw.text((x1, y1-20), f"Sig: {conf:.2f}", fill="red")

        # Draw stamp box (blue)
        if r["fields"]["stamp"]["bbox"]:
            x1, y1, x2, y2 = r["fields"]["stamp"]["bbox"]
            draw.rectangle([x1, y1, x2, y2], outline="blue", width=4)
            # Add confidence label
            conf = r["fields"]["stamp"]["conf"]
            draw.text((x1, y1-20), f"Stamp: {conf:.2f}", fill="blue")

        overlay_path = OVERLAY_DIR / f"{stem}_overlay.png"
        img.save(overlay_path)

    print(f"\n{'='*70}")
    print("SUCCESS!")
    print(f"{'='*70}")
    print(f"Staged {len(final_top)} images to: {IN_PROGRESS_IMG}")
    print(f"Staged {len(final_top)} labels to: {IN_PROGRESS_LAB}")
    print(f"Generated {len(final_top)} overlays to: {OVERLAY_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
