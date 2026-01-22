"""
Fuse VLM text outputs with signature/stamp boxes from the detector into a single JSON.

Inputs:
  - annotations/seed/in_progress/labels_vlm/*.json (from vlm_query.py)
  - train/*.png images
  - YOLO signature/stamp weights (runs/detect/stamp_sig_v3/weights/best.pt by default)

Outputs:
  - annotations/seed/in_progress/labels_fused/<stem>.json with fields + sig/stamp bboxes
  - copies images to annotations/seed/in_progress/images for convenience
"""

import argparse
import json
from pathlib import Path
import shutil

from PIL import Image, ImageDraw
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm-dir", default="annotations/seed/in_progress/labels_vlm")
    ap.add_argument("--images-dir", default="train")
    ap.add_argument("--out-dir", default="annotations/seed/in_progress/labels_fused")
    ap.add_argument("--out-img-dir", default="annotations/seed/in_progress/images")
    ap.add_argument("--det-weights", default="runs/detect/stamp_sig_v3/weights/best.pt")
    ap.add_argument("--device", default="0")
    ap.add_argument("--overlay-dir", default=None, help="optional: save overlays here")
    ap.add_argument("--clear", action="store_true", help="clear out_dir before writing")
    ap.add_argument("--scales", nargs="+", type=int, default=[640, 960], help="detection scales for TTA")
    ap.add_argument("--min-size", type=float, default=16.0, help="min box width/height in pixels")
    ap.add_argument("--min-area", type=float, default=256.0, help="min box area in pixels^2")
    ap.add_argument("--max-aspect", type=float, default=6.0, help="max aspect ratio (long/short side)")
    args = ap.parse_args()

    vlm_dir = Path(args.vlm_dir)
    img_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)
    out_img_dir = Path(args.out_img_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_img_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = Path(args.overlay_dir) if args.overlay_dir else None
    if overlay_dir:
        overlay_dir.mkdir(parents=True, exist_ok=True)
    if args.clear:
        for p in out_dir.glob("*.json"):
            p.unlink()
        if overlay_dir:
            for p in overlay_dir.glob("*.png"):
                p.unlink()

    det = YOLO(args.det_weights)
    device = f"cuda:{args.device}" if args.device != "cpu" else "cpu"

    files = list(vlm_dir.glob("*.json"))
    print(f"fusing {len(files)} VLM jsons")

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
                if "." in s:
                    return float(s)
                return int(s)
            except Exception:
                return val
        return val

    for f in files:
        stem = f.stem
        img_path = img_dir / f"{stem}.png"
        if not img_path.exists():
            continue
        vlm_obj = json.loads(f.read_text())
        fields = vlm_obj.get("parsed") or {}
        hp = to_num(fields.get("horse_power"))
        cost = to_num(fields.get("asset_cost"))

        # run detector for sig/stamp with multi-scale TTA
        sig_box = None
        stamp_box = None
        best_sig = best_stamp = 0.0
        for scale in args.scales:
            preds = det.predict(source=str(img_path), imgsz=scale, verbose=False, device=device)[0]
            for b in preds.boxes:
                cls = int(b.cls.item())
                conf = float(b.conf.item())
                x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
                w, h = x2 - x1, y2 - y1
                area = w * h
                aspect = max(w, h) / (min(w, h) + 1e-6)
                if w < args.min_size or h < args.min_size or area < args.min_area or aspect > args.max_aspect:
                    continue
                if cls == 0 and conf > best_sig:
                    best_sig = conf
                    sig_box = [x1, y1, x2, y2]
                elif cls == 1 and conf > best_stamp:
                    best_stamp = conf
                    stamp_box = [x1, y1, x2, y2]

        out_data = {
            "doc_id": stem,
            "fields": {
                "dealer_name": fields.get("dealer_name") or "",
                "model_name": fields.get("model_name") or "",
                "horse_power": hp,
                "asset_cost": cost,
                "signature": {"present": bool(sig_box), "bbox": sig_box, "conf": best_sig},
                "stamp": {"present": bool(stamp_box), "bbox": stamp_box, "conf": best_stamp},
            },
            "source": "vlm+sigstamp_detector",
            "vlm_conf_logprob": vlm_obj.get("conf_logprob"),
        }
        out_dir.joinpath(f"{stem}.json").write_text(json.dumps(out_data, indent=2))
        # copy image
        shutil.copy2(img_path, out_img_dir / img_path.name)

        if overlay_dir:
            img = Image.open(img_path).convert("RGB")
            d = ImageDraw.Draw(img)
            if sig_box:
                d.rectangle(sig_box, outline="red", width=4)
            if stamp_box:
                d.rectangle(stamp_box, outline="blue", width=4)
            img.save(overlay_dir / f"{stem}_overlay.png")

    print(f"wrote fused JSONs to {out_dir}")


if __name__ == "__main__":
    main()
