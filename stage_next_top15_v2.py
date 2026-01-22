"""
Stage next top 15 using LATEST trained models:
- YOLO v5 for detection
- Qwen VLM with optional LoRA adapter for text extraction
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
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
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return val
    if isinstance(val, str):
        s = val.replace(",", "").strip()
        if not s:
            return None
        try:
            return float(s) if "." in s else int(s)
        except:
            return val
    return val

def clean_hp(val):
    """Extract numeric horse power: '45 HP' -> 45"""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return int(val) if val == int(val) else val
    if isinstance(val, str):
        match = re.search(r'(\d+(?:\.\d+)?)', val)
        if match:
            num = float(match.group(1))
            return int(num) if num == int(num) else num
    return None

def fix_labels_with_vlm():
    """Merge VLM text fields into labels and fix HP format."""
    print("\n=== Merging VLM text fields into labels ===")
    fixed = 0
    for label_path in IN_PROGRESS_LAB.glob("*.json"):
        stem = label_path.stem
        vlm_file = VLM_DIR / f"{stem}.json"
        if not vlm_file.exists():
            continue
        label_data = json.loads(label_path.read_text())
        vlm_obj = json.loads(vlm_file.read_text())
        fields = vlm_obj.get("parsed") or {}
        label_data["fields"]["dealer_name"] = fields.get("dealer_name") or ""
        label_data["fields"]["model_name"] = fields.get("model_name") or ""
        label_data["fields"]["horse_power"] = clean_hp(fields.get("horse_power"))
        label_data["fields"]["asset_cost"] = to_num(fields.get("asset_cost"))
        label_data["vlm_conf_logprob"] = vlm_obj.get("conf_logprob", 0.0)
        label_path.write_text(json.dumps(label_data, indent=2))
        fixed += 1
        print(f"✓ {stem}: HP={label_data['fields']['horse_power']}, Cost={label_data['fields']['asset_cost']}")
    print(f"\nMerged VLM fields into {fixed} labels")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0")
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--use-lora", action="store_true", help="Use LoRA-trained VLM adapter")
    args = parser.parse_args()

    # Load YOLO results
    if not YOLO_RESULTS.exists():
        print(f"ERROR: {YOLO_RESULTS} not found. Run workflow_stage_top15.py first.")
        return

    print(f"Loading YOLO results from {YOLO_RESULTS}")
    with open(YOLO_RESULTS) as f:
        yolo_results = json.load(f)

    done = set(p.stem for p in DONE_DIR.glob("*.png"))
    remaining = [r for r in yolo_results if r["stem"] not in done]

    print(f"Already done: {len(done)} images")
    print(f"Remaining: {len(remaining)} candidates")

    next_batch = remaining[:args.top_n] if len(remaining) >= args.top_n else remaining

    print(f"\n=== Next top {len(next_batch)} candidates ===")
    for i, r in enumerate(next_batch, 1):
        print(f"  {i}. {r['stem']}: score={r['score']:.3f}")

    IN_PROGRESS_IMG.mkdir(parents=True, exist_ok=True)
    IN_PROGRESS_LAB.mkdir(parents=True, exist_ok=True)
    VLM_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

    # Clear old files from in_progress directories
    print("\n=== Clearing old in_progress files ===")
    for f in IN_PROGRESS_IMG.glob("*.png"):
        f.unlink()
    for f in IN_PROGRESS_LAB.glob("*.json"):
        f.unlink()
    for f in VLM_DIR.glob("*.json"):
        f.unlink()
    print("  Cleared old images, labels, and VLM outputs")

    # Step 1: Copy images to in_progress FIRST (before running VLM)
    print(f"\n=== Copying {len(next_batch)} images to in_progress ===")
    for r in next_batch:
        stem = r["stem"]
        img_path = TRAIN_DIR / f"{stem}.png"
        if img_path.exists():
            shutil.copy2(img_path, IN_PROGRESS_IMG / img_path.name)
            print(f"  Copied {stem}.png")

    # Step 2: Run VLM with optional LoRA adapter
    print(f"\n=== Running VLM on {len(next_batch)} images ===")
    cmd = [
        sys.executable, "vlm_query.py",
        "--model-path", "models/qwen2.5-vl-7b-instruct",
        "--device", args.device,
        "--images-dir", "annotations/seed/in_progress/images",
        "--limit", str(len(next_batch)),
        "--load-4bit"
    ]

    if args.use_lora:
        lora_path = ROOT / "vlm_lora_out"
        if lora_path.exists():
            print(f"Using LoRA adapter from {lora_path}")
            cmd.extend(["--adapter", str(lora_path)])
        else:
            print(f"Warning: LoRA adapter not found at {lora_path}, using base model")

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print("VLM inference failed!")
        return

    # Step 3: Fuse results and create labels
    print("\n=== Fusing VLM + YOLO results ===")
    yolo_lookup = {r["stem"]: r for r in next_batch}

    for r in tqdm(next_batch, desc="Creating labels"):
        stem = r["stem"]
        img_path = TRAIN_DIR / f"{stem}.png"

        if not img_path.exists():
            continue

        # Image already copied in Step 1

        vlm_file = VLM_DIR / f"{stem}.json"
        dealer_name = ""
        model_name = ""
        horse_power = None
        asset_cost = None
        vlm_conf = 0.0

        if vlm_file.exists():
            vlm_obj = json.loads(vlm_file.read_text())
            fields = vlm_obj.get("parsed") or {}
            dealer_name = fields.get("dealer_name") or ""
            model_name = fields.get("model_name") or ""
            horse_power = to_num(fields.get("horse_power"))
            asset_cost = to_num(fields.get("asset_cost"))
            vlm_conf = vlm_obj.get("conf_logprob", 0.0)

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
            "source": "vlm_lora+yolo_v5" if args.use_lora else "vlm+yolo_v5",
            "notes": f"YOLO score={r['score']:.3f}; VLM logprob={vlm_conf:.3f}; please verify",
        }

        label_path = IN_PROGRESS_LAB / f"{stem}.json"
        label_path.write_text(json.dumps(label_data, indent=2))

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

        img.save(OVERLAY_DIR / f"{stem}_overlay.png")

    # Auto-fix: merge VLM text fields into labels
    fix_labels_with_vlm()

    print(f"\n{'='*70}")
    print("SUCCESS!")
    print(f"Staged {len(next_batch)} to in_progress with VLM text fields merged")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
