"""
Custom OWLv2 fine-tune on your COCO sig/stamp data (owlv2_data/annotations.json, images/).
No LoRA/quant here; trains full model. Prompts are ["signature", "stamp"].
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection, get_linear_schedule_with_warmup
from tqdm.auto import tqdm


PROMPTS = ["signature", "stamp"]


class CocoSigStamp(Dataset):
    def __init__(self, ann_path: Path, img_dir: Path):
        coco = json.loads(ann_path.read_text())
        self.imgs = {im["id"]: im for im in coco["images"]}
        self.anns_by_img = {}
        for ann in coco["annotations"]:
            self.anns_by_img.setdefault(ann["image_id"], []).append(ann)
        self.img_dir = img_dir
        self.items = []
        for img_id, im in self.imgs.items():
            if (img_dir / im["file_name"]).exists():
                self.items.append(img_id)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_id = self.items[idx]
        im_info = self.imgs[img_id]
        img_path = self.img_dir / im_info["file_name"]
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        anns = []
        for ann in self.anns_by_img.get(img_id, []):
            x, y, w, h = ann["bbox"]
            x1, y1, x2, y2 = x, y, x + w, y + h
            # normalize to [0,1]
            anns.append(
                {
                    "class": ann["category_id"] - 1,  # 0 or 1
                    "box": [x1 / W, y1 / H, x2 / W, y2 / H],
                }
            )
        return img, anns


def collate_fn(batch, processor, device, image_size=640):
    images, ann_list = zip(*batch)
    # sizes per image
    sizes = [im.size for im in images]  # (W, H)
    inputs = processor(
        text=PROMPTS,  # flat list of queries
        images=list(images),
        return_tensors="pt",
        padding=True,
        size=image_size,  # downsample to save memory
    )
    # build labels list
    labels = []
    for anns, (W, H) in zip(ann_list, sizes):
        if len(anns) == 0:
            labels.append(
                {
                    "class_labels": torch.zeros((0,), dtype=torch.long, device=device),
                    "boxes": torch.zeros((0, 4), dtype=torch.float, device=device),
                }
            )
        else:
            # map boxes to the resized+letterboxed square (image_size x image_size)
            scale = min(image_size / W, image_size / H)
            new_w, new_h = W * scale, H * scale
            pad_w, pad_h = image_size - new_w, image_size - new_h
            off_x, off_y = pad_w * 0.5, pad_h * 0.5
            abs_boxes = []
            for a in anns:
                x1, y1, x2, y2 = a["box"]
                # denormalize to original, apply scale+pad, renormalize to square
                x1 *= W
                y1 *= H
                x2 *= W
                y2 *= H
                x1 = (x1 * scale + off_x) / image_size
                x2 = (x2 * scale + off_x) / image_size
                y1 = (y1 * scale + off_y) / image_size
                y2 = (y2 * scale + off_y) / image_size
                cx = (x1 + x2) * 0.5
                cy = (y1 + y2) * 0.5
                w = x2 - x1
                h = y2 - y1
                abs_boxes.append([cx, cy, w, h])  # cxcywh in [0,1] over the resized square
            cls = torch.tensor([a["class"] for a in anns], dtype=torch.long, device=device)
            boxes = torch.tensor(abs_boxes, dtype=torch.float, device=device)
            labels.append({"class_labels": cls, "boxes": boxes})
    batch_out = {
        "pixel_values": inputs.pixel_values.to(device),
        "input_ids": inputs.input_ids.to(device),
        "attention_mask": inputs.attention_mask.to(device),
        "labels": labels,
    }
    return batch_out


def xyxy_from_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert center-format boxes (cx, cy, w, h) to xyxy, clamped to [0,1]."""
    cx, cy, w, h = boxes.unbind(-1)
    x1 = (cx - 0.5 * w).clamp(0, 1)
    y1 = (cy - 0.5 * h).clamp(0, 1)
    x2 = (cx + 0.5 * w).clamp(0, 1)
    y2 = (cy + 0.5 * h).clamp(0, 1)
    return torch.stack([x1, y1, x2, y2], dim=-1)


def compute_loss(outputs, labels, lambda_box=5.0, lambda_obj=1.0, lambda_cls=1.0, neg_obj_weight=0.1):
    """
    Lightweight training loss for OWLv2 outputs.
    Matches each GT to the patch with the highest class score for that query, then
    applies box L1 + objectness + class BCE. Unselected patches get a small negative
    objectness penalty.
    """
    pred_logits = outputs.logits  # [B, num_patches, num_queries]
    obj_logits = outputs.objectness_logits
    if obj_logits.ndim == 3 and obj_logits.shape[-1] == 1:
        obj_logits = obj_logits.squeeze(-1)
    pred_boxes_xyxy = xyxy_from_cxcywh(outputs.pred_boxes)  # [B, num_patches, 4]

    batch_size = pred_logits.shape[0]
    total_loss = 0.0
    for b in range(batch_size):
        gt = labels[b]
        gt_cls = gt["class_labels"]
        gt_boxes = gt["boxes"]
        num_gt = gt_cls.shape[0]

        if num_gt == 0:
            # No objects: push objectness down everywhere.
            obj_loss = F.binary_cross_entropy_with_logits(obj_logits[b], torch.zeros_like(obj_logits[b]))
            total_loss = total_loss + lambda_obj * obj_loss
            continue

        num_patches = pred_logits.shape[1]
        selected = torch.zeros(num_patches, dtype=torch.bool, device=pred_logits.device)
        cls_loss = torch.tensor(0.0, device=pred_logits.device)
        box_loss = torch.tensor(0.0, device=pred_logits.device)
        obj_loss = torch.tensor(0.0, device=pred_logits.device)

        for i in range(num_gt):
            c = int(gt_cls[i].item())
            scores = pred_logits[b, :, c]  # [num_patches]
            idx = torch.argmax(scores)
            selected[idx] = True

            cls_loss = cls_loss + F.binary_cross_entropy_with_logits(scores[idx], torch.tensor(1.0, device=scores.device))
            obj_loss = obj_loss + F.binary_cross_entropy_with_logits(obj_logits[b, idx], torch.tensor(1.0, device=scores.device))
            box_loss = box_loss + F.l1_loss(pred_boxes_xyxy[b, idx], gt_boxes[i])

        # Negative objectness for unselected patches (down-weighted).
        neg_mask = ~selected
        if neg_mask.any():
            obj_loss = obj_loss + neg_obj_weight * F.binary_cross_entropy_with_logits(
                obj_logits[b, neg_mask], torch.zeros_like(obj_logits[b, neg_mask])
            )

        cls_loss = cls_loss / num_gt
        box_loss = box_loss / num_gt
        obj_loss = obj_loss / num_gt
        total_loss = total_loss + lambda_cls * cls_loss + lambda_box * box_loss + lambda_obj * obj_loss

    return total_loss / batch_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", default="google/owlv2-base-patch16-ensemble")
    ap.add_argument("--train_file", default="owlv2_data/annotations.json")
    ap.add_argument("--image_dir", default="owlv2_data/images")
    ap.add_argument("--output_dir", default="owlv2_custom_out")
    ap.add_argument("--num_train_epochs", type=int, default=10)
    ap.add_argument("--per_device_train_batch_size", type=int, default=2)
    ap.add_argument("--learning_rate", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--image_size", type=int, default=448, help="shorter side resize")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--bf16", action="store_true", help="use bfloat16 autocast (disables scaler)")
    ap.add_argument("--no_grad_checkpoint", action="store_true", help="disable gradient checkpointing")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    processor = Owlv2Processor.from_pretrained(args.model_name_or_path)
    model = Owlv2ForObjectDetection.from_pretrained(args.model_name_or_path).to(device)
    if device.type == "cuda" and not args.no_grad_checkpoint:
        model.gradient_checkpointing_enable()

    ds = CocoSigStamp(Path(args.train_file), Path(args.image_dir))
    dl = DataLoader(
        ds,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        # Tensors are already moved to device inside collate_fn, so leave pin_memory off.
        pin_memory=False,
        collate_fn=lambda b: collate_fn(b, processor, device, image_size=args.image_size),
    )

    total_steps = args.num_train_epochs * max(1, len(dl) // args.gradient_accumulation_steps + 1)
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    use_amp = device.type == "cuda"
    use_scaler = use_amp and not args.bf16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    model.train()
    step = 0
    for epoch in range(args.num_train_epochs):
        for batch in tqdm(dl, desc=f"epoch {epoch+1}/{args.num_train_epochs}", leave=False):
            labels = batch.pop("labels")
            autocast_dtype = torch.bfloat16 if args.bf16 else None
            try:
                with torch.amp.autocast(device_type="cuda" if use_amp else "cpu", dtype=autocast_dtype, enabled=use_amp):
                    outputs = model(**batch, interpolate_pos_encoding=True)
                    loss = compute_loss(outputs, labels) / args.gradient_accumulation_steps
            except torch.cuda.OutOfMemoryError:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue

            if use_scaler:
                scaler.scale(loss).backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    scaler.step(optim)
                    scaler.update()
                    scheduler.step()
                    optim.zero_grad()
            else:
                loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optim.step()
                    scheduler.step()
                    optim.zero_grad()
            step += 1
        print(f"epoch {epoch+1}/{args.num_train_epochs} done, last_loss={(loss.item()*args.gradient_accumulation_steps):.4f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    print("saved to", out_dir)


if __name__ == "__main__":
    main()
