"""
Stage next top 15 candidates from remaining YOLO results.

Reads tmp_yolo_results.json, excludes already done images,
stages next top 15, runs VLM, fuses, and generates overlays.

Usage:
  python3 stage_next_top15.py [--skip-vlm] [--device 0]
"""

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "train"
DONE_DIR = ROOT / "annotations" / "seed" / "done" / "images"
IN_PROGRESS_IMG = ROOT / "annotations" / "seed" / "in_progress" / "images"
IN_PROGRESS_LAB = ROOT / "annotations" / "seed" / "in_progress" / "labels"
VLM_DIR = ROOT / "annotations" / "seed" / "in_progress" / "labels_vlm"
OVERLAY_DIR = ROOT / "overlays"
YOLO_RESULTS = ROOT / "tmp_yolo_results.json"


def to_num(val):
    """Convert string to int/float."""
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
    parser.add_argument("--skip-vlm", action="store_true", help="Skip VLM inference (use only YOLO boxes)")
    parser.add_argument("--device", default="0", help="Device for VLM")
    parser.add_argument("--top-n", type=int, default=15, help="Number to stage")
    args = parser.parse_args()

    # Load YOLO results
    if not YOLO_RESULTS.exists():
        print(f"ERROR: {YOLO_RESULTS} not found. Run workflow_stage_top15.py first.")
        return

    print(f"Loading YOLO results from {YOLO_RESULTS}")
    with open(YOLO_RESULTS) as f:
        yolo_results = json.load(f)

    # Get done set
    done = set(p.stem for p in DONE_DIR.glob("*.png"))
    print(f"Already done: {len(done)} images")

    # Filter out done images
    remaining = [r for r in yolo_results if r["stem"] not in done]
    print(f"Remaining candidates: {len(remaining)}")

    if len(remaining) < args.top_n:
        print(f"Warning: Only {len(remaining)} candidates available")
        next_batch = remaining
    else:
        next_batch = remaining[:args.top_n]

    print(f"\nNext top {len(next_batch)} candidates:")
    for i, r in enumerate(next_batch, 1):
        print(f"  {i}. {r['stem']}: score={r['score']:.3f}")

    # Create directories
    IN_PROGRESS_IMG.mkdir(parents=True, exist_ok=True)
    IN_PROGRESS_LAB.mkdir(parents=True, exist_ok=True)
    VLM_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_vlm:
        # Run VLM on next batch
        print(f"\n=== Running VLM on {len(next_batch)} images ===")
        import subprocess
        import sys

        # Save candidate stems to temp file
        candidate_stems = [r["stem"] for r in next_batch]
        temp_list = ROOT / "tmp_next_candidates.txt"
        temp_list.write_text("\n".join(candidate_stems))

        # Run VLM query
        cmd = [
            sys.executable, "vlm_query.py",
            "--model-path", "models/qwen2.5-vl-7b-instruct",
            "--device", args.device,
            "--limit", str(len(next_batch)),
            "--load-4bit"
        ]
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            print("VLM inference failed!")
            return

    # Fuse VLM + YOLO results
    print("\n=== Fusing VLM + YOLO results ===")
    yolo_lookup = {r["stem"]: r for r in next_batch}

    for r in tqdm(next_batch, desc="Staging"):
        stem = r["stem"]
        img_path = TRAIN_DIR / f"{stem}.png"

        if not img_path.exists():
            print(f"Warning: {img_path} not found, skipping")
            continue

        # Copy image
        shutil.copy2(img_path, IN_PROGRESS_IMG / img_path.name)

        # Load VLM output if available
        vlm_file = VLM_DIR / f"{stem}.json"
        dealer_name = ""
        model_name = ""
        horse_power = None
        asset_cost = None
        vlm_conf = 0.0

        if vlm_file.exists() and not args.skip_vlm:
            vlm_obj = json.loads(vlm_file.read_text())
            fields = vlm_obj.get("parsed") or {}
            dealer_name = fields.get("dealer_name") or ""
            model_name = fields.get("model_name") or ""
            horse_power = to_num(fields.get("horse_power"))
            asset_cost = to_num(fields.get("asset_cost"))
            vlm_conf = vlm_obj.get("conf_logprob", 0.0)

        # Create label JSON
        label_data = {
            "doc_id": stem,
            "fields": {
                "dealer_name": dealer_name,
                "model_name": model_name,
                "horse_power": horse_power,
                "asset_cost": asset_cost,
                "signature": {
                    "present": bool(r["sig_box"]),
                    "bbox": r["sig_box"],
                    "conf": r["sig_conf"]
                },
                "stamp": {
                    "present": bool(r["stamp_box"]),
                    "bbox": r["stamp_box"],
                    "conf": r["stamp_conf"]
                },
            },
            "source": "vlm+yolo_workflow" if not args.skip_vlm else "yolo_only",
            "notes": f"YOLO score={r['score']:.3f}; VLM logprob={vlm_conf:.3f}; please verify",
        }

        # Write label
        label_path = IN_PROGRESS_LAB / f"{stem}.json"
        label_path.write_text(json.dumps(label_data, indent=2))

        # Generate overlay
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        if r["sig_box"]:
            x1, y1, x2, y2 = r["sig_box"]
            draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
            draw.text((x1, y1-20), f"Sig: {r['sig_conf']:.2f}", fill="red")

        if r["stamp_box"]:
            x1, y1, x2, y2 = r["stamp_box"]
            draw.rectangle([x1, y1, x2, y2], outline="blue", width=4)
            draw.text((x1, y1-20), f"Stamp: {r['stamp_conf']:.2f}", fill="blue")

        overlay_path = OVERLAY_DIR / f"{stem}_overlay.png"
        img.save(overlay_path)

    print(f"\n{'='*70}")
    print("SUCCESS!")
    print(f"{'='*70}")
    print(f"Staged {len(next_batch)} images to: {IN_PROGRESS_IMG}")
    print(f"Staged {len(next_batch)} labels to: {IN_PROGRESS_LAB}")
    print(f"Generated {len(next_batch)} overlays to: {OVERLAY_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
