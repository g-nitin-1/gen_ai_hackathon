"""
Stage top-15 candidates ranked by asset_cost using doc-level text models (no heuristics).
Fills dealer_name, model_name, horse_power, asset_cost, and stamp/signature boxes.

Prereq:
  python3 train_doc_field_model.py --ocr both
  (Optional) train YOLO/patch models; otherwise point to existing weights.

Usage:
  python3 stage_top15_doc_fields.py --device 0 --det-weights runs/detect/stamp_sig_v3/weights/best.pt --clf-weights patch_runs/patch_classifier.pt --clear
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from paddleocr import PaddleOCR
import easyocr
from PIL import Image, ImageDraw
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms
from ultralytics import YOLO
from tqdm import tqdm


def load_patch_clf(path: Path, device):
    import torchvision.models as models

    ckpt = torch.load(path, map_location=device)
    classes = ckpt["classes"]
    state = ckpt["state_dict"]
    for arch in ["resnet34", "resnet18", "resnet50"]:
        try:
            if arch == "resnet18":
                m = models.resnet18(weights=None)
            elif arch == "resnet34":
                m = models.resnet34(weights=None)
            else:
                m = models.resnet50(weights=None)
            m.fc = torch.nn.Linear(m.fc.in_features, len(classes))
            m.load_state_dict(state, strict=True)
            model = m
            break
        except Exception:
            model = None
            continue
    if model is None:
        raise RuntimeError("Failed to load patch classifier")
    model.to(device).eval()
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


def ocr_doc(img_path: Path, use_paddle=True, use_easy=True):
    lines = []
    if use_paddle:
        paddle = PaddleOCR(lang="en", use_angle_cls=True, show_log=False, use_gpu=True)
        p_out = paddle.ocr(str(img_path), cls=True)
        for page in p_out:
            for line in page:
                lines.append(line[1][0])
    if use_easy:
        reader = easyocr.Reader(["en"], gpu=True)
        for _, text, _ in reader.readtext(str(img_path), detail=1, paragraph=False, min_size=10, text_threshold=0.4):
            lines.append(text)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="0", help="0 or cpu")
    parser.add_argument("--det-weights", type=str, required=True, help="YOLO detector weights for stamp/signature")
    parser.add_argument("--clf-weights", type=str, required=True, help="Patch classifier weights for stamp/signature")
    parser.add_argument("--limit", type=int, default=None, help="limit candidates for speed")
    parser.add_argument("--clear", action="store_true", help="clear in_progress/overlays before staging")
    args = parser.parse_args()

    device = f"cuda:{args.device}" if args.device != "cpu" else "cpu"
    det = YOLO(args.det_weights)
    patch_clf, tfm, classes = load_patch_clf(Path(args.clf_weights), device=device)
    cls_idx = {c: i for i, c in enumerate(classes)}

    text_model = joblib.load("text_models/doc_field_models.joblib")
    vec = text_model["vec"]
    clf_dealer = text_model["dealer"]
    clf_model = text_model["model"]
    hp_reg = text_model["hp_reg"]
    cost_reg = text_model["cost_reg"]
    train_X = text_model["train_X"]

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
        text = ocr_doc(p, use_paddle=True, use_easy=True)
        if not text.strip():
            continue
        X = vec.transform([text])
        dealer_pred = clf_dealer.predict(X)[0]
        model_pred = clf_model.predict(X)[0]
        hp_pred = float(hp_reg.predict(X)[0]) if hp_reg is not None else None
        cost_pred = float(cost_reg.predict(X)[0]) if cost_reg is not None else None
        # similarity confidence
        sim = float(cosine_similarity(X, train_X).max())

        # sig/stamp
        sig_box = None
        stamp_box = None
        best_sig = best_stamp = 0.0
        det_out = det.predict(source=str(p), imgsz=640, verbose=False, device=device)[0]
        img = Image.open(p).convert("RGB")
        w, h = img.size
        for b in det_out.boxes:
            cls = int(b.cls.item())
            x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
            pw = (x2 - x1) * 0.1
            ph = (y2 - y1) * 0.1
            cx1 = max(0, x1 - pw)
            cy1 = max(0, y1 - ph)
            cx2 = min(w, x2 + pw)
            cy2 = min(h, y2 + ph)
            crop = img.crop((cx1, cy1, cx2, cy2))
            probs = classify_crop(patch_clf, tfm, device, crop)
            if cls == 0:
                p_sig = probs[cls_idx.get("signature", 0)]
                if p_sig > best_sig:
                    best_sig = p_sig
                    sig_box = [x1, y1, x2, y2]
            elif cls == 1:
                p_stamp = probs[cls_idx.get("stamp", 1)]
                if p_stamp > best_stamp:
                    best_stamp = p_stamp
                    stamp_box = [x1, y1, x2, y2]

        results.append((sim, p.name, dealer_pred, model_pred, hp_pred, cost_pred, sig_box, stamp_box))

    results.sort(key=lambda x: (-x[0], x[1]))
    print("top5", [(r[1], round(r[0], 3)) for r in results[:5]])

    OUT_IMG.mkdir(parents=True, exist_ok=True)
    OUT_LAB.mkdir(parents=True, exist_ok=True)
    OVER.mkdir(exist_ok=True)

    for r in results[:15]:
        stem = Path(r[1]).stem
        src = TRAIN / r[1]
        OUT_IMG.joinpath(src.name).write_bytes(src.read_bytes())
        data = {
            "doc_id": stem,
            "fields": {
                "dealer_name": r[2],
                "model_name": r[3],
                "horse_power": round(r[4], 1) if r[4] is not None else None,
                "asset_cost": int(r[5]) if r[5] is not None else None,
                "signature": {"present": bool(r[6]), "bbox": r[6]},
                "stamp": {"present": bool(r[7]), "bbox": r[7]},
            },
            "source": "doc_field_model",
            "notes": f"sim={r[0]:.3f}; please verify",
        }
        OUT_LAB.joinpath(f"{stem}.json").write_text(json.dumps(data, indent=2))
        img = Image.open(src).convert("RGB")
        d = ImageDraw.Draw(img)
        if r[6]:
            d.rectangle(r[6], outline="red", width=4)
        if r[7]:
            d.rectangle(r[7], outline="blue", width=4)
        img.save(OVER / f"{stem}_overlay.png")
    print("wrote top15 to in_progress and overlays (doc-level model, cost-ranked)")


if __name__ == "__main__":
    main()
