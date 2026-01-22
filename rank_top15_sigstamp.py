"""
Rank top-15 candidates for signature/stamp using detector + patch classifier + PaddleOCR fields.

Usage:
  python3 rank_top15_sigstamp.py --device 0 --limit 300 --clear

Steps:
  - Load YOLO detector from yolo_runs/stamp_sig/weights/best.pt
  - Load patch classifier from patch_runs/patch_classifier.pt
  - Exclude done set, run detector to propose boxes, classify crops, keep best per class
  - Score = sig_prob + stamp_prob, pick top-15 (ranking still only uses sig+stamp)
  - PaddleOCR reads text; dealer/model/hp/cost are fuzzy-matched against gold references
  - Write images/labels to annotations/seed/in_progress/{images,labels} and overlays to overlays/
"""

import os
from pathlib import Path

# Ensure a writable temp directory before importing torch/torchvision
ROOT = Path(__file__).resolve().parent
TMP_DIR = ROOT / "tmp_runtime"
TMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TMPDIR", str(TMP_DIR))

import argparse
import json
import re
import difflib

import torch
from PIL import Image, ImageDraw
from torchvision import transforms
from ultralytics import YOLO
from paddleocr import PaddleOCR
import easyocr
from tqdm import tqdm
TRAIN_DIR = ROOT / "train"
DONE_DIR = ROOT / "annotations" / "seed" / "done" / "images"
OUT_IMG = ROOT / "annotations" / "seed" / "in_progress" / "images"
OUT_LAB = ROOT / "annotations" / "seed" / "in_progress" / "labels"
OVERLAY_DIR = ROOT / "overlays"
NUM_SAMPLES = 5  # fallback samples per class
SIG_RE = re.compile(r"sign", re.IGNORECASE)
STAMP_RE = re.compile(r"stamp|seal", re.IGNORECASE)


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

    arch_order = ["resnet34", "resnet18", "resnet50"]
    last_err = None
    model = None
    for arch in arch_order:
        try:
            m = build_model(arch)
            m.load_state_dict(state, strict=True)
            model = m
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    if model is None:
        raise RuntimeError(f"Failed to load classifier weights: {last_err}")

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


def fallback_samples(img: Image.Image):
    """Sample a few bottom-region windows as fallback proposals."""
    w, h = img.size
    samples = []
    bottom = h * 0.5
    widths = [0.2, 0.3]
    heights = [0.1, 0.15]
    for _ in range(NUM_SAMPLES):
        bw = w * widths[torch.randint(0, len(widths), (1,)).item()]
        bh = h * heights[torch.randint(0, len(heights), (1,)).item()]
        cx = torch.rand(1).item() * (w - bw) + bw / 2
        cy = torch.rand(1).item() * (h - bottom - bh) + bottom + bh / 2
        x1 = max(0, cx - bw / 2)
        y1 = max(0, cy - bh / 2)
        x2 = min(w, cx + bw / 2)
        y2 = min(h, cy + bh / 2)
        samples.append([x1, y1, x2, y2])
    return samples


def parse_text(txt: str):
    # deprecated: kept for backward compatibility if needed elsewhere
    dealer = ""
    model_name = ""
    hp = None
    cost = None
    sig_flag = bool(SIG_RE.search(txt))
    stamp_flag = bool(STAMP_RE.search(txt))
    return dealer, model_name, hp, cost, sig_flag, stamp_flag


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
    """Pick closest reference string to any OCR line using fuzzy ratio * confidence."""
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
    hp_re = re.compile(r"(\\d{2,3}(?:\\.\\d)?)\\s*(?:hp|h\\.p)", re.IGNORECASE)
    for text, _ in lines:
        for m in hp_re.finditer(text):
            hp_candidates.append(float(m.group(1)))
    for m in hp_re.finditer(full_text):
        hp_candidates.append(float(m.group(1)))
    if not hp_candidates:
        return None
    if ref_hps:
        return min(hp_candidates, key=lambda v: min(abs(v - r) for r in ref_hps))
    return max(hp_candidates)


def extract_cost(lines, ref_costs):
    nums = []
    num_re = re.compile(r"[\\d,]{5,}")
    for text, conf in lines:
        for m in num_re.finditer(text):
            raw = m.group().replace(",", "")
            if not raw.isdigit():
                continue
            val = int(raw)
            if val >= 40000:
                nums.append((val, conf))
    if not nums:
        return None
    nums.sort(key=lambda x: (x[1], x[0]), reverse=True)
    val = nums[0][0]
    if ref_costs:
        closest = min(ref_costs, key=lambda r: abs(r - val))
        if abs(closest - val) / max(closest, 1) < 0.25:
            return val
    return val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="0", help="0 or cpu")
    parser.add_argument("--limit", type=int, default=None, help="limit candidates for speed")
    parser.add_argument("--clear", action="store_true", help="Clear in_progress and overlays before staging")
    parser.add_argument("--det-weights", type=str, default="yolo_runs/stamp_sig/weights/best.pt", help="YOLO detector weights")
    parser.add_argument("--clf-weights", type=str, default="patch_runs/patch_classifier.pt", help="Patch classifier weights")
    parser.add_argument("--ocr", type=str, default="paddle", choices=["paddle", "easy", "both"], help="OCR backend")
    args = parser.parse_args()

    device = f"cuda:{args.device}" if args.device != "cpu" else "cpu"
    det = YOLO(args.det_weights)
    clf, tfm, classes = load_classifier(Path(args.clf_weights), device=device)
    cls_idx = {c: i for i, c in enumerate(classes)}
    paddle_ocr = PaddleOCR(lang="en", use_angle_cls=True, show_log=False, use_gpu=args.device != "cpu") if args.ocr in ("paddle", "both") else None
    easy_reader = easyocr.Reader(["en"], gpu=args.device != "cpu") if args.ocr in ("easy", "both") else None

    # build reference lists from done labels
    ref_dealers = []
    ref_models = []
    ref_hps = []
    ref_costs = []
    for lp in Path("annotations/seed/done/labels").glob("*.json"):
        d = json.load(lp.open())
        f = d.get("fields", {})
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
        for f in OUT_IMG.glob("*"): f.unlink()
        for f in OUT_LAB.glob("*"): f.unlink()
        for f in OVERLAY_DIR.glob("*"): f.unlink()

    results = []
    for i, p in enumerate(tqdm(candidates, desc="Scoring"), 1):
        det_out = det.predict(source=str(p), imgsz=640, verbose=False, device=device)[0]
        best_sig = (0, None)
        best_stamp = (0, None)
        img = Image.open(p).convert("RGB")
        # OCR-driven text fields (combine paddle/easy if requested)
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
        hp = extract_hp(lines, full_text, ref_hps)
        cost = extract_cost(lines, ref_costs)
        for b in det_out.boxes:
            cls = int(b.cls.item())
            conf = float(b.conf.item())
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
                if p_sig > best_sig[0]:
                    best_sig = (p_sig, [x1, y1, x2, y2])
            elif cls == 1:  # stamp
                p_stamp = probs[cls_idx.get("stamp", 1)]
                if p_stamp > best_stamp[0]:
                    best_stamp = (p_stamp, [x1, y1, x2, y2])
        # fallback sampling if no boxes
        if best_sig[0] == 0:
            for box in fallback_samples(img):
                crop = img.crop(box)
                probs = classify_crop(clf, tfm, device, crop)
                p_sig = probs[cls_idx.get("signature", 0)]
                if p_sig > best_sig[0]:
                    best_sig = (p_sig, box)
        if best_stamp[0] == 0:
            for box in fallback_samples(img):
                crop = img.crop(box)
                probs = classify_crop(clf, tfm, device, crop)
                p_stamp = probs[cls_idx.get("stamp", 1)]
                if p_stamp > best_stamp[0]:
                    best_stamp = (p_stamp, box)

        score = best_sig[0] + best_stamp[0]
        results.append((score, p.name, best_sig, best_stamp, dealer, model_name, hp, cost))

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
        sig_p, sig_box = r[2]
        stamp_p, stamp_box = r[3]
        data = {
            "doc_id": stem,
            "fields": {
                "dealer_name": r[4] or "",
                "model_name": r[5] or "",
                "horse_power": r[6],
                "asset_cost": r[7],
                "signature": {"present": bool(sig_box), "bbox": sig_box},
                "stamp": {"present": bool(stamp_box), "bbox": stamp_box},
            },
            "source": "det+patch_classifier+ocr",
            "notes": f"score={r[0]:.3f}; please verify",
        }
        OUT_LAB.joinpath(f"{stem}.json").write_text(json.dumps(data, indent=2))
        img = Image.open(src).convert("RGB")
        draw = ImageDraw.Draw(img)
        if sig_box:
            x1, y1, x2, y2 = sig_box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
        if stamp_box:
            x1, y1, x2, y2 = stamp_box
            draw.rectangle([x1, y1, x2, y2], outline="blue", width=4)
        img.save(OVERLAY_DIR / f"{stem}_overlay.png")
    print("wrote top15 to in_progress and overlays")


if __name__ == "__main__":
    main()
