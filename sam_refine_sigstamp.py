"""
Refine signature/stamp boxes into masks using a pretrained SAM model.
Takes fused JSONs (with sig/stamp bboxes) and writes masks alongside.

Inputs:
  - annotations/seed/in_progress/labels_fused/*.json (from fuse_vlm_sigstamp.py)
  - annotations/seed/in_progress/images/*.png
  - SAM checkpoint (e.g., sam_vit_h_4b8939.pth)

Outputs:
  - annotations/seed/in_progress/labels_fused/*.json updated with "signature.mask" / "stamp.mask" as RLE
  - optional overlays with masks drawn
"""

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from segment_anything import sam_model_registry, SamPredictor


def mask_to_rle(mask: np.ndarray) -> str:
    # simple RLE encoding (flattened row-major)
    pixels = mask.flatten()
    rle = []
    prev = pixels[0]
    count = 1
    for p in pixels[1:]:
        if p == prev:
            count += 1
        else:
            rle.append(count)
            count = 1
            prev = p
    rle.append(count)
    return base64.b64encode(json.dumps(rle).encode()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fused-dir", default="annotations/seed/in_progress/labels_fused")
    ap.add_argument("--images-dir", default="annotations/seed/in_progress/images")
    ap.add_argument("--sam-checkpoint", required=True, help="path to SAM checkpoint (e.g., sam_vit_h_4b8939.pth)")
    ap.add_argument("--sam-model", default="vit_h")
    ap.add_argument("--device", default="0")
    ap.add_argument("--overlay-dir", default=None, help="optional: save overlays with masks")
    ap.add_argument("--clear-overlays", action="store_true", help="clear overlay dir before writing")
    args = ap.parse_args()

    device = f"cuda:{args.device}" if args.device != "cpu" else "cpu"
    sam = sam_model_registry[args.sam_model](checkpoint=args.sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    fused_dir = Path(args.fused_dir)
    img_dir = Path(args.images_dir)
    overlay_dir = Path(args.overlay_dir) if args.overlay_dir else None
    if overlay_dir:
        overlay_dir.mkdir(parents=True, exist_ok=True)
        if args.clear_overlays:
            for p in overlay_dir.glob("*"):
                if p.is_file():
                    p.unlink()

    files = list(fused_dir.glob("*.json"))
    print(f"refining {len(files)} fused jsons")

    for f in files:
        data = json.loads(f.read_text())
        stem = f.stem
        img_path = img_dir / f"{stem}.png"
        if not img_path.exists():
            continue
        image = np.array(Image.open(img_path).convert("RGB"))
        predictor.set_image(image)

        for key in ["signature", "stamp"]:
            entry = data["fields"].get(key) or {}
            box = entry.get("bbox")
            if not box:
                continue
            input_box = np.array(box, dtype=np.float32)
            masks, scores, _ = predictor.predict(
                box=input_box, multimask_output=True
            )
            # take best mask
            best_idx = int(np.argmax(scores))
            best_mask = masks[best_idx].astype(np.uint8)
            entry["mask"] = mask_to_rle(best_mask)
            data["fields"][key] = entry

            if overlay_dir:
                img_pil = Image.fromarray(image.copy())
                d = ImageDraw.Draw(img_pil, "RGBA")
                # tint mask
                mask_img = Image.fromarray((best_mask * 120).astype(np.uint8))
                color = (255, 0, 0, 80) if key == "signature" else (0, 0, 255, 80)
                d.bitmap((0, 0), mask_img, fill=color)
                img_pil.save(overlay_dir / f"{stem}_{key}_mask.png")

        f.write_text(json.dumps(data, indent=2))

    print("done.")


if __name__ == "__main__":
    main()
