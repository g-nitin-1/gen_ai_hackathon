"""
Draw a single training image with GT boxes (green) and OWLv2 preds (red/blue).
Usage:
python3 debug_overlay_one.py --index 0 --threshold 0.3 --image-size 640 --model-path owlv2_custom_out --processor-path owlv2_custom_out
"""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import Owlv2Processor, Owlv2ForObjectDetection


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", default="owlv2_data/annotations.json")
    ap.add_argument("--images-dir", default="owlv2_data/images")
    ap.add_argument("--index", type=int, default=0, help="Index of image in the COCO file")
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--image-size", type=int, default=640)
    ap.add_argument("--model-path", default="owlv2_custom_out")
    ap.add_argument("--processor-path", default=None)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    coco = json.loads(Path(args.annotations).read_text())
    imgs = coco["images"]
    if not imgs:
        raise SystemExit("No images in annotations.")
    idx = max(0, min(args.index, len(imgs) - 1))
    im_info = imgs[idx]
    anns = [a for a in coco["annotations"] if a["image_id"] == im_info["id"]]

    img_path = Path(args.images_dir) / im_info["file_name"]
    image = Image.open(img_path).convert("RGB")

    proc = Owlv2Processor.from_pretrained(args.processor_path or args.model_path)
    model = Owlv2ForObjectDetection.from_pretrained(args.model_path).to(args.device).eval()
    inputs = proc(text=["signature", "stamp"], images=image, size=args.image_size, return_tensors="pt").to(args.device)
    target_sizes = torch.tensor([(image.height, image.width)], device=args.device)
    with torch.no_grad():
        out = model(**inputs, interpolate_pos_encoding=True)
    res = proc.post_process_grounded_object_detection(out, target_sizes=target_sizes, threshold=args.threshold)[0]

    draw = ImageDraw.Draw(image)
    # GT in green
    for a in anns:
        x, y, w, h = a["bbox"]
        box = [x, y, x + w, y + h]
        draw.rectangle(box, outline="lime", width=4)
    # preds: draw top-1 per class
    best_sig = best_stamp = None
    for score, label, box in zip(res["scores"], res["labels"], res["boxes"]):
        if label.item() == 0:
            if best_sig is None or score > best_sig[0]:
                best_sig = (score, box)
        else:
            if best_stamp is None or score > best_stamp[0]:
                best_stamp = (score, box)
    for best, color, name in [(best_sig, "red", "sig"), (best_stamp, "blue", "stamp")]:
        if best is None:
            continue
        score, box = best
        x1, y1, x2, y2 = box.tolist()
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        draw.text((x1, y1), f"{name} {float(score):.2f}", fill="white", stroke_fill=color, stroke_width=2)

    out_path = Path("overlays_train") / f"debug_{Path(im_info['file_name']).stem}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
