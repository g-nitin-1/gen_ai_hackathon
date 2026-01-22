"""
Stage top-15 images for annotation using OWLv2 scores (signature + stamp).

Loads todo images, excludes those already in done/ in_progress, runs OWLv2
detector, ranks by best signature + stamp score, and copies the top 15 into
annotations/seed/in_progress/{images,labels} with overlays.
"""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from tqdm.auto import tqdm
from transformers import Owlv2Processor, Owlv2ForObjectDetection

ROOT = Path(__file__).resolve().parent
TODO_DIR = ROOT / "annotations" / "seed" / "todo" / "images"
DONE_DIR = ROOT / "annotations" / "seed" / "done" / "images"
INPROG_IMG = ROOT / "annotations" / "seed" / "in_progress" / "images"
INPROG_LAB = ROOT / "annotations" / "seed" / "in_progress" / "labels"
OVERLAYS = ROOT / "overlays"


def score_with_owlv2(img_path: Path, proc, model, device: str, image_size: int, threshold: float):
    image = Image.open(img_path).convert("RGB")
    inputs = proc(text=["signature", "stamp"], images=image, size=image_size, return_tensors="pt").to(device)
    target_sizes = torch.tensor([(image.height, image.width)], device=device)
    with torch.no_grad():
        out = model(**inputs, interpolate_pos_encoding=True)
    res = proc.post_process_grounded_object_detection(out, target_sizes=target_sizes, threshold=threshold)[0]
    best_sig = best_stamp = 0.0
    sig_box = stamp_box = None
    for score, label, box in zip(res["scores"], res["labels"], res["boxes"]):
        conf = float(score)
        x1, y1, x2, y2 = [float(x) for x in box.tolist()]
        if label.item() == 0 and conf > best_sig:
            best_sig, sig_box = conf, [x1, y1, x2, y2]
        if label.item() == 1 and conf > best_stamp:
            best_stamp, stamp_box = conf, [x1, y1, x2, y2]
    score = best_sig + best_stamp
    return score, best_sig, best_stamp, sig_box, stamp_box


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="owlv2_custom_out")
    ap.add_argument("--processor-path", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--image-size", type=int, default=640)
    ap.add_argument("--threshold", type=float, default=0.1)
    ap.add_argument("--topk", type=int, default=15)
    ap.add_argument("--limit", type=int, default=None, help="Optional limit on candidates for quick runs")
    args = ap.parse_args()

    device = args.device
    proc = Owlv2Processor.from_pretrained(args.processor_path or args.model_path)
    model = Owlv2ForObjectDetection.from_pretrained(args.model_path).to(device).eval()

    done = set(p.stem for p in DONE_DIR.glob("*.png"))
    inprog = set(p.stem for p in INPROG_IMG.glob("*.png"))
    candidates = [p for p in TODO_DIR.glob("*.png") if p.stem not in done and p.stem not in inprog]
    if args.limit:
        candidates = candidates[: args.limit]
    if not candidates:
        print("No candidates found.")
        return
    print(f"Candidates: {len(candidates)}")

    results = []
    for p in tqdm(candidates, desc="Scoring", unit="img"):
        try:
            score, sig_conf, stamp_conf, sig_box, stamp_box = score_with_owlv2(
                p, proc, model, device, args.image_size, args.threshold
            )
            results.append((score, p.name, sig_conf, stamp_conf, sig_box, stamp_box))
        except Exception as e:
            print(f"skip {p}: {e}")
            continue

    results.sort(key=lambda x: (-x[0], x[1]))
    top = results[: args.topk]
    print("top5 preview:", [(r[1], round(r[0], 3)) for r in top[:5]])

    INPROG_IMG.mkdir(parents=True, exist_ok=True)
    INPROG_LAB.mkdir(parents=True, exist_ok=True)
    OVERLAYS.mkdir(exist_ok=True)

    for r in top:
        name = r[1]
        stem = Path(name).stem
        src = TODO_DIR / name
        dst_img = INPROG_IMG / src.name
        dst_img.write_bytes(src.read_bytes())
        data = {
            "doc_id": stem,
            "fields": {
                "dealer_name": "",
                "model_name": "",
                "horse_power": None,
                "asset_cost": None,
                "signature": {"present": bool(r[4]), "bbox": r[4]},
                "stamp": {"present": bool(r[5]), "bbox": r[5]},
            },
            "source": f"owlv2@{args.model_path}",
            "notes": "auto-selected top15 by OWLv2; please verify",
        }
        (INPROG_LAB / f"{stem}.json").write_text(json.dumps(data, indent=2))
        img = Image.open(src).convert("RGB")
        draw = ImageDraw.Draw(img)
        if r[4]:
            draw.rectangle(r[4], outline="red", width=4)
        if r[5]:
            draw.rectangle(r[5], outline="blue", width=4)
        img.save(OVERLAYS / f"{stem}_overlay.png")
    print("wrote top15 to in_progress and overlays")


if __name__ == "__main__":
    main()
