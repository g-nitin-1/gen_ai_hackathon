"""
Fast workflow to stage top-15 candidates based on YOLO confidence.

Steps:
1. Run YOLO detection on all remaining images (excluding done)
2. Rank by YOLO confidence (sig_conf + stamp_conf)
3. Take top N candidates
4. Run VLM on those top N
5. Fuse VLM text + YOLO boxes
6. Select final top 15 and stage to in_progress with overlays

Usage:
  python3 workflow_stage_top15.py --device 0 --top-n 50
"""

import argparse
import json
from pathlib import Path
import shutil

from PIL import Image, ImageDraw
from ultralytics import YOLO
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "train"
DONE_DIR = ROOT / "annotations" / "seed" / "done" / "images"
IN_PROGRESS_DIR = ROOT / "annotations" / "seed" / "in_progress"
OVERLAY_DIR = ROOT / "overlays"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="0", help="YOLO device (0 or cpu)")
    parser.add_argument("--det-weights", type=str, default="yolo_runs/stamp_sig/weights/best.pt")
    parser.add_argument("--top-n", type=int, default=50, help="Top N candidates for VLM processing")
    parser.add_argument("--final-top", type=int, default=15, help="Final top K to stage")
    parser.add_argument("--scales", nargs="+", type=int, default=[640, 960], help="Multi-scale detection")
    parser.add_argument("--min-size", type=float, default=16.0, help="Min box dimension")
    parser.add_argument("--min-area", type=float, default=256.0, help="Min box area")
    parser.add_argument("--max-aspect", type=float, default=6.0, help="Max aspect ratio")
    args = parser.parse_args()

    device = args.device if args.device == "cpu" else f"cuda:{args.device}"

    # Load YOLO detector
    print(f"Loading YOLO detector from {args.det_weights}")
    det = YOLO(args.det_weights)

    # Get remaining images (exclude done set)
    done = set(p.stem for p in DONE_DIR.glob("*.png"))
    candidates = [p for p in TRAIN_DIR.glob("*.png") if p.stem not in done]
    print(f"Found {len(candidates)} remaining images to process")

    # Step 1: Run YOLO on all remaining images
    print("\n=== Step 1: Running YOLO detection on all remaining images ===")
    yolo_results = []

    for img_path in tqdm(candidates, desc="YOLO detection"):
        best_sig_conf = 0.0
        best_sig_box = None
        best_stamp_conf = 0.0
        best_stamp_box = None

        # Multi-scale detection
        for scale in args.scales:
            preds = det.predict(source=str(img_path), imgsz=scale, verbose=False, device=device)[0]

            for b in preds.boxes:
                cls = int(b.cls.item())
                conf = float(b.conf.item())
                x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())

                # Filter by size and aspect ratio
                w, h = x2 - x1, y2 - y1
                area = w * h
                aspect = max(w, h) / (min(w, h) + 1e-6)

                if w < args.min_size or h < args.min_size or area < args.min_area or aspect > args.max_aspect:
                    continue

                # Keep best per class
                if cls == 0 and conf > best_sig_conf:  # signature
                    best_sig_conf = conf
                    best_sig_box = [x1, y1, x2, y2]
                elif cls == 1 and conf > best_stamp_conf:  # stamp
                    best_stamp_conf = conf
                    best_stamp_box = [x1, y1, x2, y2]

        # Score = sum of confidences
        score = best_sig_conf + best_stamp_conf

        yolo_results.append({
            "path": img_path,
            "name": img_path.name,
            "stem": img_path.stem,
            "score": score,
            "sig_conf": best_sig_conf,
            "sig_box": best_sig_box,
            "stamp_conf": best_stamp_conf,
            "stamp_box": best_stamp_box,
        })

    # Step 2: Rank by YOLO confidence
    print("\n=== Step 2: Ranking by YOLO confidence ===")
    yolo_results.sort(key=lambda x: (-x["score"], x["name"]))

    print(f"\nTop 10 candidates by YOLO confidence:")
    for i, r in enumerate(yolo_results[:10], 1):
        print(f"  {i}. {r['name']}: score={r['score']:.3f} (sig={r['sig_conf']:.3f}, stamp={r['stamp_conf']:.3f})")

    # Step 3: Save top-N candidates for VLM processing
    top_n_candidates = yolo_results[:args.top_n]
    print(f"\n=== Step 3: Selected top {len(top_n_candidates)} candidates for VLM processing ===")

    # Save candidate list for VLM script
    candidate_list_path = ROOT / "tmp_top_candidates.json"
    with open(candidate_list_path, "w") as f:
        json.dump([r["stem"] for r in top_n_candidates], f, indent=2)
    print(f"Saved candidate list to {candidate_list_path}")

    # Also save detailed YOLO results
    yolo_results_path = ROOT / "tmp_yolo_results.json"
    with open(yolo_results_path, "w") as f:
        json.dump([{
            "stem": r["stem"],
            "score": r["score"],
            "sig_conf": r["sig_conf"],
            "sig_box": r["sig_box"],
            "stamp_conf": r["stamp_conf"],
            "stamp_box": r["stamp_box"],
        } for r in yolo_results], f, indent=2)
    print(f"Saved YOLO results to {yolo_results_path}")

    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Run VLM on top candidates:")
    print(f"   python3 vlm_query.py --model-path models/qwen2.5-vl-7b-instruct \\")
    print(f"     --device {args.device} --limit {args.top_n} --load-4bit")
    print()
    print("2. Then run the fusion and staging script:")
    print(f"   python3 workflow_stage_top15_part2.py --final-top {args.final_top}")
    print("="*70)


if __name__ == "__main__":
    main()
