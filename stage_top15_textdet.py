"""
Stage top-15 candidates using a text-field detector (dealer/model/hp/cost) plus OCR in the detected boxes.
Also predicts stamp/signature boxes with a separate detector.

Classes expected in text detector: [dealer_name, model_name, horse_power, asset_cost]

Usage:
  python3 stage_top15_textdet.py \
    --device 0 \
    --text-det-weights runs/detect/text_fields_v1/weights/best.pt \
    --sig-det-weights runs/detect/stamp_sig_v3/weights/best.pt \
    --clear
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from paddleocr import PaddleOCR
import easyocr
from PIL import Image, ImageDraw
from ultralytics import YOLO
from tqdm import tqdm


CLASSES = ["dealer_name", "model_name", "horse_power", "asset_cost"]


def run_ocr_crop(img_crop: Image.Image, use_paddle=True, use_easy=True):
    if img_crop.width < 5 or img_crop.height < 5:
        return "", 0.0, "", []
    texts = []
    img_arr = np.array(img_crop)
    if use_paddle:
        paddle = PaddleOCR(lang="en", use_angle_cls=True, show_log=False, use_gpu=True)
        out = paddle.ocr(img_arr, cls=True)
        if out:
            for page in out:
                if page is None:
                    continue
                for line in page:
                    texts.append((line[1][0], float(line[1][1])))
    if use_easy:
        reader = easyocr.Reader(["en"], gpu=True)
        try:
            e_out = reader.readtext(img_arr, detail=1, paragraph=False, min_size=5, text_threshold=0.4)
            if e_out:
                for _, text, conf in e_out:
                    texts.append((text, float(conf)))
        except Exception:
            pass
    if not texts:
        return "", 0.0, "", []
    texts.sort(key=lambda x: x[1], reverse=True)
    joined = " ".join(t for t, _ in texts[:3])
    raw = " | ".join(t for t, _ in texts)
    return joined, texts[0][1], raw, texts


def parse_hp(text: str):
    import re

    m = re.search(r"(\d{2,3}(?:\.\d)?)\s*(?:hp|h\.p)", text, re.IGNORECASE)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def parse_cost(text_pairs):
    import re

    best = None
    best_conf = -1.0
    for t, conf in text_pairs:
        # Split into tokens to avoid mixing two numbers in one string
        for tok in re.split(r"[\\s,;:|]+", t):
            digits = re.sub(r"\\D", "", tok)
            if len(digits) < 5 or len(digits) > 9:
                continue
            if conf > best_conf:
                best_conf = conf
                best = digits
    if best is None:
        return None
    try:
        return int(best)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="0", help="0 or cpu")
    parser.add_argument("--text-det-weights", type=str, required=True, help="YOLO weights for text fields")
    parser.add_argument("--sig-det-weights", type=str, required=True, help="YOLO weights for signature/stamp")
    parser.add_argument("--limit", type=int, default=None, help="limit candidates for speed")
    parser.add_argument("--clear", action="store_true", help="clear in_progress/overlays before staging")
    parser.add_argument(
        "--pad",
        type=float,
        default=0.1,
        help="padding fraction around detected boxes before OCR (0.1 = 10%)",
    )
    parser.add_argument(
        "--crop-scale",
        type=float,
        default=1.5,
        help="scale factor to upsample the crop before OCR (e.g., 2.0 doubles size)",
    )
    args = parser.parse_args()

    device = f"cuda:{args.device}" if args.device != "cpu" else "cpu"
    det_text = YOLO(args.text_det_weights)
    det_sig = YOLO(args.sig_det_weights)

    ROOT = Path(".")
    TRAIN = ROOT / "train"
    DONE = ROOT / "annotations" / "seed" / "done" / "images"
    OUT_IMG = ROOT / "annotations" / "seed" / "in_progress" / "images"
    OUT_LAB = ROOT / "annotations" / "seed" / "in_progress" / "labels"
    OVER = ROOT / "overlays"

    done = set(p.stem for p in DONE.glob("*.png"))
    candidates = [p for p in TRAIN.glob("*.png") if p.stem not in done]
    if args.limit:
        candidates = candidates[: args.limit]
    print("candidates", len(candidates))

    if args.clear:
        for f in OUT_IMG.glob("*"):
            f.unlink()
        for f in OUT_LAB.glob("*"):
            f.unlink()
        for f in OVER.glob("*"):
            f.unlink()

    results = []
    for p in tqdm(candidates, desc="Scoring"):
        img = Image.open(p).convert("RGB")
        w, h = img.size
        text_preds = det_text.predict(source=str(p), imgsz=1280, verbose=False, device=device)[0]
        best_boxes = {cls: (0.0, None) for cls in CLASSES}
        for b in text_preds.boxes:
            cls_id = int(b.cls.item())
            conf = float(b.conf.item())
            cls_name = CLASSES[cls_id] if cls_id < len(CLASSES) else None
            if cls_name is None:
                continue
            x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
            if conf > best_boxes[cls_name][0]:
                best_boxes[cls_name] = (conf, [x1, y1, x2, y2])

        fields = {}
        for cls_name, (conf, box) in best_boxes.items():
            if box is None:
                fields[cls_name] = (None, conf, None, "", [])
                continue
            cx1, cy1, cx2, cy2 = box
            pad_w = (cx2 - cx1) * args.pad
            pad_h = (cy2 - cy1) * args.pad
            px1 = max(0, cx1 - pad_w)
            py1 = max(0, cy1 - pad_h)
            px2 = min(w, cx2 + pad_w)
            py2 = min(h, cy2 + pad_h)
            crop = img.crop((px1, py1, px2, py2))
            if args.crop_scale != 1.0:
                cw, ch = crop.size
                crop = crop.resize(
                    (max(1, int(cw * args.crop_scale)), max(1, int(ch * args.crop_scale))),
                    resample=Image.BICUBIC,
                )
            text, ocr_conf, raw_text, text_pairs = run_ocr_crop(crop, use_paddle=True, use_easy=True)
            if cls_name == "horse_power":
                val = parse_hp(text)
            elif cls_name == "asset_cost":
                val = parse_cost(text_pairs)
            else:
                val = text.strip()
            fields[cls_name] = (val, conf, box, raw_text, text_pairs)

        # sig/stamp
        sig_pred = det_sig.predict(source=str(p), imgsz=640, verbose=False, device=device)[0]
        sig_box = None
        stamp_box = None
        best_sig = best_stamp = 0.0
        for b in sig_pred.boxes:
            cls = int(b.cls.item())
            conf = float(b.conf.item())
            x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
            if cls == 0 and conf > best_sig:
                best_sig = conf
                sig_box = [x1, y1, x2, y2]
            elif cls == 1 and conf > best_stamp:
                best_stamp = conf
                stamp_box = [x1, y1, x2, y2]

        score = best_boxes["asset_cost"][0]
        results.append((score, p.name, fields, sig_box, stamp_box))

    results.sort(key=lambda x: (-x[0], x[1]))
    print("top5", [(r[1], round(r[0], 3)) for r in results[:5]])

    OUT_IMG.mkdir(parents=True, exist_ok=True)
    OUT_LAB.mkdir(parents=True, exist_ok=True)
    OVER.mkdir(exist_ok=True)

    for r in results[:15]:
        stem = Path(r[1]).stem
        src = TRAIN / r[1]
        OUT_IMG.joinpath(src.name).write_bytes(src.read_bytes())
        fields = r[2]
        data = {
            "doc_id": stem,
            "fields": {
                "dealer_name": fields["dealer_name"][0] or "",
                "model_name": fields["model_name"][0] or "",
                "horse_power": fields["horse_power"][0],
                "asset_cost": fields["asset_cost"][0],
                "signature": {"present": bool(r[3]), "bbox": r[3]},
                "stamp": {"present": bool(r[4]), "bbox": r[4]},
            },
            "ocr_raw": {
                k: {
                    "text": fields[k][3] or "",
                    "candidates": [{"text": t, "conf": c} for t, c in fields[k][4]],
                }
                for k in CLASSES
            },
            "source": "text_field_detector",
            "notes": f"score_cost_box={r[0]:.3f}; please verify",
        }
        OUT_LAB.joinpath(f"{stem}.json").write_text(json.dumps(data, indent=2))
        img = Image.open(src).convert("RGB")
        d = ImageDraw.Draw(img)
        colors = {"dealer_name": "green", "model_name": "purple", "horse_power": "orange", "asset_cost": "cyan"}
        for k, (_, _, box, _, _) in fields.items():
            if box:
                d.rectangle(box, outline=colors.get(k, "yellow"), width=4)
        if r[3]:
            d.rectangle(r[3], outline="red", width=4)
        if r[4]:
            d.rectangle(r[4], outline="blue", width=4)
        img.save(OVER / f"{stem}_overlay.png")
    print("wrote top15 to in_progress and overlays (text-field detector)")


if __name__ == "__main__":
    main()
