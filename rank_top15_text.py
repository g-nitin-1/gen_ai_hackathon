"""
Rank top-15 candidates based on OCR confidence for horse_power and asset_cost (not signature/stamp).

Usage:
  python3 rank_top15_text.py --device 0 --ocr both --limit 300 --clear

Outputs:
  - Top 15 images/labels in annotations/seed/in_progress/{images,labels}
  - Overlays in overlays/ with hp/cost text drawn
"""

import argparse
import difflib
import os
import re
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from ultralytics import YOLO
from paddleocr import PaddleOCR
import easyocr
from tqdm import tqdm


# ensure writable tmp for OCR libs
ROOT = Path(__file__).resolve().parent
TMP_DIR = ROOT / "tmp_runtime"
TMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TMPDIR", str(TMP_DIR))

TRAIN_DIR = ROOT / "train"
DONE_DIR = ROOT / "annotations" / "seed" / "done" / "images"
OUT_IMG = ROOT / "annotations" / "seed" / "in_progress" / "images"
OUT_LAB = ROOT / "annotations" / "seed" / "in_progress" / "labels"
OVERLAY_DIR = ROOT / "overlays"


def run_paddle(paddle_ocr: PaddleOCR, img_path: Path):
    res = paddle_ocr.ocr(str(img_path), cls=True)
    lines = []
    if res:
        for page in res:
            for line in page:
                text = line[1][0]
                conf = float(line[1][1])
                lines.append((text, conf))
    full_text = "\n".join(t for t, _ in lines)
    return lines, full_text


def run_easyocr(reader: easyocr.Reader, img_path: Path):
    out = reader.readtext(str(img_path), detail=1, paragraph=False, min_size=10, text_threshold=0.4)
    lines = []
    for _, text, conf in out:
        lines.append((text, float(conf)))
    full_text = "\n".join(t for t, _ in lines)
    return lines, full_text


def best_ref_match(lines, refs, min_ratio=0.55):
    if not refs:
        return ""
    best_ref = ""
    best_score = 0.0
    for ref in refs:
        ref_low = ref.lower()
        for text, conf in lines:
            ratio = difflib.SequenceMatcher(None, ref_low, text.lower()).ratio()
            score = ratio * conf
            if score > best_score:
                best_ref = ref
                best_score = score
    return best_ref if best_score >= min_ratio else ""


def extract_hp(lines, full_text, ref_hps):
    hp_candidates = []
    hp_re = re.compile(r"(\d{2,3}(?:\.\d)?)\s*(?:hp|h\.p)", re.IGNORECASE)
    for text, conf in lines:
        for m in hp_re.finditer(text):
            hp_candidates.append((float(m.group(1)), conf))
    for m in hp_re.finditer(full_text):
        hp_candidates.append((float(m.group(1)), 0.4))
    if not hp_candidates:
        return None, 0.0
    # choose candidate closest to reference set if available, else highest conf
    if ref_hps:
        def closeness(c):
            val, conf = c
            return min(abs(val - r) for r in ref_hps), -conf
        hp_val, hp_conf = sorted(hp_candidates, key=closeness)[0]
    else:
        hp_val, hp_conf = max(hp_candidates, key=lambda x: x[1])
    return hp_val, hp_conf


def extract_cost(lines, ref_costs):
    """
    Extract asset_cost with context filters to avoid mobiles/account numbers.
    Strategy:
      - Accept 5-8 digit numbers (after stripping commas/symbols) within 1e5..3e6 range.
      - Bonus if line has cost-like keywords.
      - Rank by conf + keyword bonus + digit length; snap to closest ref if within 30%.
    """
    candidates = []
    num_re = re.compile(r"(?:₹|\$)?\s*[\d,\.]{5,}")
    keywords = ["rs", "amount", "total", "cost", "price", "value", "invoice", "ex", "ex-showroom", "onroad", "on-road"]
    for text, conf in lines:
        kw_bonus = 0.2 if any(k in text.lower() for k in keywords) else 0.0
        for m in num_re.finditer(text):
            raw = m.group()
            digits = re.sub(r"\D", "", raw)
            if len(digits) < 5 or len(digits) > 8:
                continue
            try:
                val = int(digits)
            except Exception:
                continue
            if not (100000 <= val <= 3000000):
                continue
            score = conf + kw_bonus + 0.01 * len(digits)
            candidates.append((val, conf, score))
    if not candidates:
        return None, 0.0
    candidates.sort(key=lambda x: (x[2], x[0]), reverse=True)
    val, conf, _ = candidates[0]
    if ref_costs:
        closest = min(ref_costs, key=lambda r: abs(r - val))
        if abs(closest - val) / max(closest, 1) < 0.3:
            val = closest
    return val, conf


def draw_overlay(img_path: Path, hp, cost):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 36)
    except Exception:
        font = None
    text = f"hp: {hp if hp is not None else '-'} | cost: {cost if cost is not None else '-'}"
    draw.rectangle([10, 10, img.width - 10, 70], outline="orange", width=3)
    draw.text((20, 20), text, fill="orange", font=font)
    return img


def load_classifier(path: Path, device):
    import torchvision.models as models
    ckpt = torch.load(path, map_location=device)
    classes = ckpt["classes"]
    state = ckpt["state_dict"]

    def build_model(arch: str):
        if arch == "resnet18":
            model = models.resnet18(weights=None)
        elif arch == "resnet34":
            model = models.resnet34(weights=None)
        else:
            model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
        return model

    model = None
    for arch in ["resnet34", "resnet18", "resnet50"]:
        try:
            m = build_model(arch)
            m.load_state_dict(state, strict=True)
            model = m
            break
        except Exception:
            continue
    if model is None:
        raise RuntimeError("Failed to load patch classifier weights")

    model.to(device)
    model.eval()
    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return model, tfm, classes


def classify_crop(model, tfm, device, img: Image.Image):
    with torch.no_grad():
        x = tfm(img).unsqueeze(0).to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="0", help="0 or cpu")
    parser.add_argument("--limit", type=int, default=None, help="limit candidates for speed")
    parser.add_argument("--clear", action="store_true", help="Clear in_progress and overlays before staging")
    parser.add_argument("--ocr", type=str, default="both", choices=["paddle", "easy", "both"], help="OCR backend")
    parser.add_argument("--det-weights", type=str, default="runs/detect/stamp_sig_v2/weights/best.pt", help="YOLO detector weights")
    parser.add_argument("--clf-weights", type=str, default="patch_runs/patch_classifier.pt", help="Patch classifier weights")
    args = parser.parse_args()

    device = f"cuda:{args.device}" if args.device != "cpu" else "cpu"
    paddle_ocr = PaddleOCR(lang="en", use_angle_cls=True, show_log=False, use_gpu=args.device != "cpu") if args.ocr in ("paddle", "both") else None
    easy_reader = easyocr.Reader(["en"], gpu=args.device != "cpu") if args.ocr in ("easy", "both") else None
    det = YOLO(args.det_weights)
    clf, tfm, classes = load_classifier(Path(args.clf_weights), device=device)
    cls_idx = {c: i for i, c in enumerate(classes)}

    # references
    ref_dealers, ref_models, ref_hps, ref_costs = [], [], [], []
    for lp in Path("annotations/seed/done/labels").glob("*.json"):
        d = lp.read_text()
        try:
            data = __import__("json").loads(d)
        except Exception:
            continue
        f = data.get("fields", {})
        if f.get("dealer_name"):
            ref_dealers.append(f["dealer_name"])
        if f.get("model_name"):
            ref_models.append(f["model_name"])
        if f.get("horse_power") is not None:
            ref_hps.append(f["horse_power"])
        if f.get("asset_cost") is not None:
            ref_costs.append(f["asset_cost"])

    done = set(p.stem for p in DONE_DIR.glob("*.png"))
    candidates = [p for p in TRAIN_DIR.glob("*.png") if p.stem not in done]
    if args.limit:
        candidates = candidates[: args.limit]
    print("candidates", len(candidates))

    if args.clear:
        for f in OUT_IMG.glob("*"):
            f.unlink()
        for f in OUT_LAB.glob("*"):
            f.unlink()
        for f in OVERLAY_DIR.glob("*"):
            f.unlink()

    results = []
    for p in tqdm(candidates, desc="Scoring"):
        lines = []
        full_texts = []
        if paddle_ocr:
            l, ft = run_paddle(paddle_ocr, p)
            lines.extend(l)
            full_texts.append(ft)
        if easy_reader:
            l, ft = run_easyocr(easy_reader, p)
            lines.extend(l)
            full_texts.append(ft)
        full_text = "\n".join(full_texts)
        dealer = best_ref_match(lines, ref_dealers)
        model_name = best_ref_match(lines, ref_models, min_ratio=0.5)
        hp_val, hp_conf = extract_hp(lines, full_text, ref_hps)
        cost_val, cost_conf = extract_cost(lines, ref_costs)
        # signature/stamp detection for completeness
        sig_box = None
        stamp_box = None
        best_sig = 0.0
        best_stamp = 0.0
        det_out = det.predict(source=str(p), imgsz=640, verbose=False, device=device)[0]
        img = Image.open(p).convert("RGB")
        for b in det_out.boxes:
            cls = int(b.cls.item())
            x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
            # pad crop
            pw = (x2 - x1) * 0.1
            ph = (y2 - y1) * 0.1
            w, h = img.size
            cx1 = max(0, x1 - pw)
            cy1 = max(0, y1 - ph)
            cx2 = min(w, x2 + pw)
            cy2 = min(h, y2 + ph)
            crop = img.crop((cx1, cy1, cx2, cy2))
            probs = classify_crop(clf, tfm, device, crop)
            if cls == 0:  # signature
                p_sig = probs[cls_idx.get("signature", 0)]
                if p_sig > best_sig:
                    best_sig = p_sig
                    sig_box = [x1, y1, x2, y2]
            elif cls == 1:  # stamp
                p_stamp = probs[cls_idx.get("stamp", 1)]
                if p_stamp > best_stamp:
                    best_stamp = p_stamp
                    stamp_box = [x1, y1, x2, y2]
        score = cost_conf
        results.append((score, p.name, hp_val, cost_val, dealer, model_name, sig_box, stamp_box))

    results.sort(key=lambda x: (-x[0], x[1]))
    print("top5", [(r[1], round(r[0], 3)) for r in results[:5]])

    OUT_IMG.mkdir(parents=True, exist_ok=True)
    OUT_LAB.mkdir(parents=True, exist_ok=True)
    OVERLAY_DIR.mkdir(exist_ok=True)

    for r in results[:15]:
        name = r[1]
        stem = Path(name).stem
        src = TRAIN_DIR / name
        OUT_IMG.joinpath(src.name).write_bytes(src.read_bytes())
        data = {
            "doc_id": stem,
            "fields": {
                "dealer_name": r[4] or "",
                "model_name": r[5] or "",
                "horse_power": r[2],
                "asset_cost": r[3],
                "signature": {"present": bool(r[6]), "bbox": r[6]},
                "stamp": {"present": bool(r[7]), "bbox": r[7]},
            },
            "source": f"ocr_text_{args.ocr}",
            "notes": f"text_score={r[0]:.3f}; please verify hp/cost",
        }
        OUT_LAB.joinpath(f"{stem}.json").write_text(__import__("json").dumps(data, indent=2))
        overlay = draw_overlay(src, r[2], r[3])
        draw = ImageDraw.Draw(overlay)
        if r[6]:
            x1, y1, x2, y2 = r[6]
            draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
        if r[7]:
            x1, y1, x2, y2 = r[7]
            draw.rectangle([x1, y1, x2, y2], outline="blue", width=4)
        overlay.save(OVERLAY_DIR / f"{stem}_overlay.png")
    print("wrote top15 to in_progress and overlays (text-focused)")


if __name__ == "__main__":
    main()
