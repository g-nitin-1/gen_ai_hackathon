"""
Quick sanity check: run OWLv2 on the COCO-style training set and report IoU.
Usage example:
python3 sanity_owlv2_iou.py --model-path owlv2_custom_out --processor-path owlv2_custom_out --device cuda:0 --limit 50 --threshold 0.1
"""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from tqdm.auto import tqdm


def iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="owlv2_custom_out")
    ap.add_argument("--processor-path", default=None)
    ap.add_argument("--annotations", default="owlv2_data/annotations.json")
    ap.add_argument("--images-dir", default="owlv2_data/images")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--threshold", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=None, help="Max images to evaluate")
    ap.add_argument("--image-size", type=int, default=640, help="Resize shorter side (match training)")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    proc = Owlv2Processor.from_pretrained(args.processor_path or args.model_path)
    model = Owlv2ForObjectDetection.from_pretrained(args.model_path).to(device).eval()

    data = json.loads(Path(args.annotations).read_text())
    imgs = {im["id"]: im for im in data["images"]}
    anns = {}
    for a in data["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)

    prompts = ["signature", "stamp"]
    sig_ious, stamp_ious = [], []

    for idx, (img_id, im) in enumerate(tqdm(imgs.items(), desc="eval", total=len(imgs))):
        if args.limit and idx >= args.limit:
            break
        img_path = Path(args.images_dir) / im["file_name"]
        image = Image.open(img_path).convert("RGB")
        inputs = proc(text=prompts, images=image, size=args.image_size, return_tensors="pt").to(device)
        target_sizes = torch.tensor([(image.height, image.width)], device=device)
        with torch.no_grad():
            out = model(**inputs)
        res = proc.post_process_grounded_object_detection(out, target_sizes=target_sizes, threshold=args.threshold)[0]

        H, W = image.height, image.width
        preds_sig, preds_stamp = [], []
        for score, label, box in zip(res["scores"], res["labels"], res["boxes"]):
            x1, y1, x2, y2 = box.tolist()
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            if label.item() == 0:
                preds_sig.append([x1, y1, x2, y2])
            else:
                preds_stamp.append([x1, y1, x2, y2])

        gt_sig, gt_stamp = [], []
        for a in anns.get(img_id, []):
            x, y, w, h = a["bbox"]
            box = [x, y, x + w, y + h]
            if a["category_id"] == 1:
                gt_sig.append(box)
            elif a["category_id"] == 2:
                gt_stamp.append(box)

        if preds_sig and gt_sig:
            sig_ious.append(max(iou(p, g) for p in preds_sig for g in gt_sig))
        if preds_stamp and gt_stamp:
            stamp_ious.append(max(iou(p, g) for p in preds_stamp for g in gt_stamp))

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    print(f"Evaluated {min(len(imgs), args.limit or len(imgs))} images")
    print(f"Mean IoU signature: {mean(sig_ious):.4f} over {len(sig_ious)} images with GT+pred")
    print(f"Mean IoU stamp:     {mean(stamp_ious):.4f} over {len(stamp_ious)} images with GT+pred")


if __name__ == "__main__":
    main()
