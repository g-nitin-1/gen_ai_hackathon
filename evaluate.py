"""
Evaluate model predictions against ground truth labels.
Computes Document-Level Accuracy (DLA) and field-level metrics.

Usage:
  python3 evaluate.py --split-dir splits/split_395_100 --yolo-model runs/detect/stamp_sig_final/weights/best.pt --use-lora
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from difflib import SequenceMatcher
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent

def fuzzy_match(s1, s2, threshold=0.9):
    """Check if two strings match with ≥threshold similarity."""
    if not s1 or not s2:
        return s1 == s2
    s1 = str(s1).lower().strip()
    s2 = str(s2).lower().strip()
    ratio = SequenceMatcher(None, s1, s2).ratio()
    return ratio >= threshold

def numeric_match(v1, v2, tolerance=0.05):
    """Check if two numbers match within ±tolerance."""
    if v1 is None or v2 is None:
        return v1 == v2
    try:
        v1, v2 = float(v1), float(v2)
        if v2 == 0:
            return v1 == 0
        return abs(v1 - v2) / max(abs(v2), 1) <= tolerance
    except:
        return False

def iou(box1, box2):
    """Compute IoU between two bounding boxes [x1, y1, x2, y2]."""
    if not box1 or not box2:
        return 0.0

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0

def bbox_match(pred, gt, iou_threshold=0.5):
    """Check if bbox prediction matches ground truth."""
    pred_present = pred.get("present", False) if pred else False
    gt_present = gt.get("present", False) if gt else False

    if not gt_present and not pred_present:
        return True  # Both absent - correct
    if gt_present != pred_present:
        return False  # Presence mismatch

    pred_box = pred.get("bbox")
    gt_box = gt.get("bbox")

    if not pred_box or not gt_box:
        return False

    return iou(pred_box, gt_box) >= iou_threshold

def clean_hp(val):
    """Extract numeric horse power."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        match = re.search(r'(\d+(?:\.\d+)?)', val)
        if match:
            return float(match.group(1))
    return None

def clean_cost(val):
    """Extract numeric cost."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        s = re.sub(r'[^\d.]', '', val)
        try:
            return float(s)
        except:
            return None
    return None

def run_inference(test_img_dir, yolo_model, use_lora, device="0", split_dir=None):
    """Run VLM and YOLO inference on test images."""
    predictions = {}

    # Run YOLO
    print("\n=== Running YOLO inference ===")
    try:
        from ultralytics import YOLO
        model = YOLO(yolo_model)

        for img_path in tqdm(list(test_img_dir.glob("*.png")), desc="YOLO"):
            stem = img_path.stem
            results = model.predict(str(img_path), conf=0.25, verbose=False)

            sig_box, sig_conf = None, 0.0
            stamp_box, stamp_conf = None, 0.0

            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist()

                    if cls == 0 and conf > sig_conf:  # signature
                        sig_box, sig_conf = xyxy, conf
                    elif cls == 1 and conf > stamp_conf:  # stamp
                        stamp_box, stamp_conf = xyxy, conf

            predictions[stem] = {
                "signature": {"present": sig_box is not None, "bbox": sig_box, "conf": sig_conf},
                "stamp": {"present": stamp_box is not None, "bbox": stamp_box, "conf": stamp_conf}
            }
    except Exception as e:
        print(f"YOLO error: {e}")

    # Run VLM
    print("\n=== Running VLM inference ===")
    vlm_out_dir = ROOT / "tmp_eval_vlm"
    vlm_out_dir.mkdir(exist_ok=True)

    # Clear old VLM outputs
    for f in vlm_out_dir.glob("*.json"):
        f.unlink()

    cmd = [
        sys.executable, "vlm_query.py",
        "--model-path", "models/qwen2.5-vl-7b-instruct",
        "--device", device,
        "--images-dir", str(test_img_dir),
        "--out-dir", str(vlm_out_dir),
        "--done-dir", "/tmp/empty_done_dir",  # Don't skip any test images
        "--limit", "999999",  # Process all test images
        "--load-4bit"
    ]

    if use_lora:
        # First try split-specific LoRA, then fall back to global
        split_lora = Path(split_dir) / "vlm_lora" if split_dir else None
        global_lora = ROOT / "vlm_lora_out"

        lora_path = None
        if split_lora and split_lora.exists():
            lora_path = split_lora
            print(f"Using split-specific LoRA: {lora_path}")
        elif global_lora.exists():
            lora_path = global_lora
            print(f"Using global LoRA: {lora_path}")

        if lora_path:
            cmd.extend(["--adapter", str(lora_path)])

    subprocess.run(cmd, cwd=ROOT)

    # Merge VLM outputs
    for vlm_file in vlm_out_dir.glob("*.json"):
        stem = vlm_file.stem
        vlm_data = json.loads(vlm_file.read_text())
        fields = vlm_data.get("parsed", {})

        if stem not in predictions:
            predictions[stem] = {}

        predictions[stem]["dealer_name"] = fields.get("dealer_name", "")
        predictions[stem]["model_name"] = fields.get("model_name", "")
        predictions[stem]["horse_power"] = clean_hp(fields.get("horse_power"))
        predictions[stem]["asset_cost"] = clean_cost(fields.get("asset_cost"))

    return predictions

def evaluate(predictions, test_lab_dir):
    """Evaluate predictions against ground truth."""
    results = {
        "total": 0,
        "dla_correct": 0,  # Document-Level Accuracy
        "field_scores": {
            "dealer_name": {"correct": 0, "total": 0},
            "model_name": {"correct": 0, "total": 0},
            "horse_power": {"correct": 0, "total": 0},
            "asset_cost": {"correct": 0, "total": 0},
            "signature": {"correct": 0, "total": 0, "iou_sum": 0.0},
            "stamp": {"correct": 0, "total": 0, "iou_sum": 0.0},
        },
        "errors": []
    }

    for lab_path in test_lab_dir.glob("*.json"):
        stem = lab_path.stem
        gt = json.loads(lab_path.read_text())
        gt_fields = gt.get("fields", {})

        pred = predictions.get(stem, {})
        results["total"] += 1

        field_correct = {
            "dealer_name": False,
            "model_name": False,
            "horse_power": False,
            "asset_cost": False,
            "signature": False,
            "stamp": False
        }

        # Dealer name (fuzzy match ≥90%)
        results["field_scores"]["dealer_name"]["total"] += 1
        if fuzzy_match(pred.get("dealer_name"), gt_fields.get("dealer_name")):
            results["field_scores"]["dealer_name"]["correct"] += 1
            field_correct["dealer_name"] = True

        # Model name (exact match - but use fuzzy for robustness)
        results["field_scores"]["model_name"]["total"] += 1
        if fuzzy_match(pred.get("model_name"), gt_fields.get("model_name"), threshold=0.95):
            results["field_scores"]["model_name"]["correct"] += 1
            field_correct["model_name"] = True

        # Horse power (numeric ±5%)
        results["field_scores"]["horse_power"]["total"] += 1
        if numeric_match(pred.get("horse_power"), gt_fields.get("horse_power")):
            results["field_scores"]["horse_power"]["correct"] += 1
            field_correct["horse_power"] = True

        # Asset cost (numeric ±5%)
        results["field_scores"]["asset_cost"]["total"] += 1
        if numeric_match(pred.get("asset_cost"), gt_fields.get("asset_cost")):
            results["field_scores"]["asset_cost"]["correct"] += 1
            field_correct["asset_cost"] = True

        # Signature (presence + IoU ≥0.5)
        results["field_scores"]["signature"]["total"] += 1
        gt_sig = gt_fields.get("signature", {})
        pred_sig = pred.get("signature", {})
        if bbox_match(pred_sig, gt_sig):
            results["field_scores"]["signature"]["correct"] += 1
            field_correct["signature"] = True
        if gt_sig.get("bbox") and pred_sig.get("bbox"):
            results["field_scores"]["signature"]["iou_sum"] += iou(pred_sig["bbox"], gt_sig["bbox"])

        # Stamp (presence + IoU ≥0.5)
        results["field_scores"]["stamp"]["total"] += 1
        gt_stamp = gt_fields.get("stamp", {})
        pred_stamp = pred.get("stamp", {})
        if bbox_match(pred_stamp, gt_stamp):
            results["field_scores"]["stamp"]["correct"] += 1
            field_correct["stamp"] = True
        if gt_stamp.get("bbox") and pred_stamp.get("bbox"):
            results["field_scores"]["stamp"]["iou_sum"] += iou(pred_stamp["bbox"], gt_stamp["bbox"])

        # Document-Level Accuracy (all 6 fields correct)
        if all(field_correct.values()):
            results["dla_correct"] += 1
        else:
            # Record error
            results["errors"].append({
                "doc_id": stem,
                "failed_fields": [k for k, v in field_correct.items() if not v],
                "gt": gt_fields,
                "pred": pred
            })

    return results

def print_results(results):
    """Print evaluation results."""
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    total = results["total"]
    dla = results["dla_correct"] / total * 100 if total > 0 else 0

    print(f"\n📊 Document-Level Accuracy (DLA): {results['dla_correct']}/{total} = {dla:.1f}%")
    print(f"   Target: ≥95%  {'✅ PASS' if dla >= 95 else '❌ FAIL'}")

    print("\n📋 Field-Level Accuracy:")
    print("-" * 50)

    for field, scores in results["field_scores"].items():
        acc = scores["correct"] / scores["total"] * 100 if scores["total"] > 0 else 0
        print(f"  {field:15} {scores['correct']:3}/{scores['total']:3} = {acc:5.1f}%")

        if field in ["signature", "stamp"] and scores["total"] > 0:
            avg_iou = scores["iou_sum"] / scores["total"]
            print(f"  {' ':15} Avg IoU: {avg_iou:.3f}")

    print("\n❌ Error Analysis:")
    print("-" * 50)
    error_counts = {}
    for err in results["errors"]:
        for f in err["failed_fields"]:
            error_counts[f] = error_counts.get(f, 0) + 1

    for field, count in sorted(error_counts.items(), key=lambda x: -x[1]):
        print(f"  {field:15} {count:3} errors")

    print("\n" + "=" * 70)

    return dla

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", required=True, help="Split directory path")
    parser.add_argument("--yolo-model", default="runs/detect/stamp_sig_v62/weights/best.pt")
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--device", default="0")
    parser.add_argument("--skip-inference", action="store_true", help="Use cached predictions")
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    test_img_dir = split_dir / "test" / "images"
    test_lab_dir = split_dir / "test" / "labels"

    if not test_img_dir.exists():
        print(f"ERROR: Test images not found at {test_img_dir}")
        return

    print(f"Evaluating on: {split_dir.name}")
    print(f"Test images: {len(list(test_img_dir.glob('*.png')))}")
    print(f"YOLO model: {args.yolo_model}")
    print(f"Using LoRA: {args.use_lora}")

    # Run inference
    cache_file = split_dir / "predictions_cache.json"
    if args.skip_inference and cache_file.exists():
        print("\nUsing cached predictions...")
        predictions = json.loads(cache_file.read_text())
    else:
        predictions = run_inference(test_img_dir, args.yolo_model, args.use_lora, args.device, split_dir=split_dir)
        cache_file.write_text(json.dumps(predictions, indent=2))

    # Evaluate
    results = evaluate(predictions, test_lab_dir)

    # Print results
    dla = print_results(results)

    # Save results
    results_file = split_dir / "evaluation_results.json"
    results["dla_percentage"] = dla
    results_file.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to: {results_file}")

if __name__ == "__main__":
    main()
