"""
Fuse VLM text with OWLv2 (open-vocab) signature/stamp boxes into unified JSONs + overlays.
Assumes top-15 images/labels are already in annotations/seed/in_progress/{images,labels_vlm}.
"""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from tqdm.auto import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-vlm", default="annotations/seed/in_progress/labels_vlm", help="Dir with VLM JSONs")
    ap.add_argument("--images-dir", default="annotations/seed/in_progress/images", help="Dir with page PNGs")
    ap.add_argument("--out-dir", default="annotations/seed/in_progress/labels_fused", help="Output fused JSON dir")
    ap.add_argument("--overlay-dir", default="overlays", help="Output overlays dir")
    ap.add_argument("--coco-file", default=None, help="Optional COCO-style JSON (uses its images, ignores VLM fields)")
    ap.add_argument(
        "--model-path",
        default="google/owlv2-base-patch16-ensemble",
        help="OWLv2 model path (base or fine-tuned).",
    )
    ap.add_argument(
        "--processor-path",
        default=None,
        help="Processor path (defaults to model-path if not set).",
    )
    ap.add_argument("--device", default="cuda:0", help="Device for OWLv2 (e.g., cuda:0 or cpu)")
    ap.add_argument("--threshold", type=float, default=0.1, help="Detection score threshold")
    ap.add_argument("--image-size", type=int, default=640, help="Resize shorter side (match training)")
    args = ap.parse_args()

    ROOT = Path(".")
    # If a COCO file is provided, override labels_vlm to point to that file.
    labels_vlm = ROOT / (args.coco_file or args.labels_vlm)
    img_dir = ROOT / args.images_dir
    out_fused = ROOT / args.out_dir
    overlays = ROOT / args.overlay_dir
    out_fused.mkdir(parents=True, exist_ok=True)
    overlays.mkdir(parents=True, exist_ok=True)

    # clear fused/overlays
    for p in out_fused.glob("*.json"):
        p.unlink()
    for p in overlays.glob("*.png"):
        p.unlink()

    proc_path = args.processor_path or args.model_path
    proc = Owlv2Processor.from_pretrained(proc_path)
    model = Owlv2ForObjectDetection.from_pretrained(args.model_path).to(args.device)
    prompts = ["signature", "stamp"]

    # If labels_vlm is a file, treat it as COCO and iterate images only.
    if labels_vlm.is_file():
        coco = json.loads(labels_vlm.read_text())
        files = coco["images"]
        def get_item(item):
            stem = Path(item["file_name"]).stem
            img_path = img_dir / item["file_name"]
            parsed = {}
            return stem, img_path, parsed
    else:
        files = list(labels_vlm.glob("*.json"))
        def get_item(path):
            stem = path.stem
            img_path = img_dir / f"{stem}.png"
            parsed = (json.loads(path.read_text()).get("parsed")) or {}
            return stem, img_path, parsed

    print(f"fusing {len(files)} items with OWLv2 boxes (model={args.model_path}, device={args.device})")

    for item in tqdm(files, desc="fusing", leave=False):
        stem, img_path, parsed = get_item(item)
        if not img_path.exists():
            continue
        image = Image.open(img_path).convert("RGB")
        inputs = proc(text=prompts, images=image, size=args.image_size, return_tensors="pt").to(args.device)
        target_sizes = torch.tensor([(image.height, image.width)], device=args.device)
        with torch.no_grad():
            out = model(**inputs, interpolate_pos_encoding=True)
        # Use grounded post-process (keeps alignment when resizing inside processor).
        res = proc.post_process_grounded_object_detection(
            out, target_sizes=target_sizes, threshold=args.threshold
        )[0]
        H, W = image.height, image.width
        sig_box = stamp_box = None
        best_sig = best_stamp = 0.0
        for score, label, box in zip(res["scores"], res["labels"], res["boxes"]):
            x1, y1, x2, y2 = box.tolist()
            # Clamp to image bounds to avoid boxes outside the page.
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            box = [x1, y1, x2, y2]
            conf = float(score)
            if label.item() == 0 and conf > best_sig:
                best_sig, sig_box = conf, box
            if label.item() == 1 and conf > best_stamp:
                best_stamp, stamp_box = conf, box
        data = {
            "doc_id": stem,
            "fields": {
                "dealer_name": parsed.get("dealer_name") or "",
                "model_name": parsed.get("model_name") or "",
                "horse_power": parsed.get("horse_power"),
                "asset_cost": parsed.get("asset_cost"),
                "signature": {"present": bool(sig_box), "bbox": sig_box, "conf": best_sig},
                "stamp": {"present": bool(stamp_box), "bbox": stamp_box, "conf": best_stamp},
            },
            "source": f"vlm+owlv2@{args.model_path}",
        }
        out_fused.joinpath(f"{stem}.json").write_text(json.dumps(data, indent=2))
        d = ImageDraw.Draw(image)
        def draw_box(box, color, label):
            if not box:
                return
            d.rectangle(box, outline=color, width=6)
            x1, y1, x2, y2 = box
            txt = f"{label}"
            pad = 2
            try:
                # Pillow >=8: use textbbox for accurate sizing
                bx1, by1, bx2, by2 = d.textbbox((x1, y1), txt)
                tw, th = bx2 - bx1, by2 - by1
            except Exception:
                # Fallback
                tw, th = 40, 12
            d.rectangle([x1, max(0, y1 - th - 2 * pad), x1 + tw + 2 * pad, y1], fill=color)
            d.text((x1 + pad, max(0, y1 - th - pad)), txt, fill="white")

        draw_box(sig_box, "red", f"sig {best_sig:.2f}")
        draw_box(stamp_box, "blue", f"stamp {best_stamp:.2f}")
        image.save(overlays / f"{stem}_overlay.png")
    print("done.")


if __name__ == "__main__":
    main()
