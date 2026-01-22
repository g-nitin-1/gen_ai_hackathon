"""
Run YOLO stamp/signature detector + OCR heuristics to select top-15 high-confidence samples.
Writes images/labels to annotations/seed/in_progress/ and overlays to overlays/.

Usage:
  python3 stage_top15.py [--limit N] [--device 0]
"""

import argparse
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw
import pytesseract
from ultralytics import YOLO
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "train"
DONE_DIR = ROOT / "annotations" / "seed" / "done" / "images"
OUT_IMG = ROOT / "annotations" / "seed" / "in_progress" / "images"
OUT_LAB = ROOT / "annotations" / "seed" / "in_progress" / "labels"
OVERLAY_DIR = ROOT / "overlays"


SIG_RE = re.compile(r"sign", re.IGNORECASE)
STAMP_RE = re.compile(r"stamp|seal", re.IGNORECASE)


def parse_text(txt: str):
    dealer = ""
    model_name = ""
    for line in txt.splitlines():
        if "dealer" in line.lower():
            dealer = line.split(":")[-1].strip()
            break
    for line in txt.splitlines():
        if "model" in line.lower():
            model_name = line.split(":")[-1].strip()
            break
    m = re.search(r"([0-9O]{2,3})\\s*H\\s*P", txt, re.IGNORECASE)
    hp = int(m.group(1).replace("O", "0")) if m else None
    cleaned = txt.replace(",", "").replace(".", "")
    nums = [int(n) for n in re.findall(r"(\\d{5,})", cleaned)]
    cost = max(nums) if nums else None
    sig_flag = bool(SIG_RE.search(txt))
    stamp_flag = bool(STAMP_RE.search(txt))
    return dealer, model_name, hp, cost, sig_flag, stamp_flag


def score_candidate(path: Path, model: YOLO, device: str):
    try:
        txt = pytesseract.image_to_string(Image.open(path))
    except Exception:
        txt = ""
    dealer, model_name, hp, cost, sig_flag, stamp_flag = parse_text(txt)
    det = model.predict(source=str(path), imgsz=640, verbose=False, device=device)[0]
    sig_conf = 0
    sig_box = None
    stamp_conf = 0
    stamp_box = None
    for b in det.boxes:
        cls = int(b.cls.item())
        conf = float(b.conf.item())
        x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
        if cls == 0 and conf > sig_conf:
            sig_conf = conf
            sig_box = [x1, y1, x2, y2]
        if cls == 1 and conf > stamp_conf:
            stamp_conf = conf
            stamp_box = [x1, y1, x2, y2]
    score = sig_conf + stamp_conf
    score += 0.5 if hp else 0
    score += 0.5 if cost else 0
    score += 0.3 if dealer else 0
    score += 0.3 if model_name else 0
    return (score, path.name, dealer, model_name, hp, cost, sig_box, stamp_box)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of candidates processed")
    parser.add_argument("--device", type=str, default="0", help="YOLO device id, e.g., 0 or cpu")
    args = parser.parse_args()

    done = set(p.stem for p in DONE_DIR.glob("*.png"))
    candidates = [p for p in TRAIN_DIR.glob("*.png") if p.stem not in done]
    if args.limit:
        candidates = candidates[: args.limit]
    print(f"Candidates: {len(candidates)}")

    model = YOLO("yolo_runs/stamp_sig/weights/best.pt")

    results = []
    for p in tqdm(candidates, desc="Scoring", unit="img"):
        results.append(score_candidate(p, model, args.device))

    results.sort(key=lambda x: (-x[0], x[1]))
    print("top5 preview:", [(r[1], round(r[0], 3)) for r in results[:5]])

    OUT_IMG.mkdir(parents=True, exist_ok=True)
    OUT_LAB.mkdir(parents=True, exist_ok=True)
    OVERLAY_DIR.mkdir(exist_ok=True)

    from PIL import ImageDraw

    for r in results[:15]:
        name = r[1]
        stem = Path(name).stem
        src = TRAIN_DIR / name
        OUT_IMG.joinpath(src.name).write_bytes(src.read_bytes())
        data = {
            "doc_id": stem,
            "fields": {
                "dealer_name": r[2],
                "model_name": r[3],
                "horse_power": r[4],
                "asset_cost": r[5],
                "signature": {"present": bool(r[6]), "bbox": r[6]},
                "stamp": {"present": bool(r[7]), "bbox": r[7]},
            },
            "source": "yolo+ocr_heuristic",
            "notes": "auto-selected top15 by confidence; please verify",
        }
        OUT_LAB.joinpath(f"{stem}.json").write_text(json.dumps(data, indent=2))
        img = Image.open(src).convert("RGB")
        draw = ImageDraw.Draw(img)
        if r[6]:
            x1, y1, x2, y2 = r[6]
            draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
        if r[7]:
            x1, y1, x2, y2 = r[7]
            draw.rectangle([x1, y1, x2, y2], outline="blue", width=4)
        img.save(OVERLAY_DIR / f"{stem}_overlay.png")
    print("wrote top15 predictions to in_progress and overlays")


if __name__ == "__main__":
    main()
