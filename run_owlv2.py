"""
Minimal fine-tune loop for OWLv2 on signature/stamp COCO data.
Requires:
  - COCO json at owlv2_data/annotations.json
  - images in owlv2_data/images
Writes model to owlv2_lora_out (or --output_dir).
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection, get_linear_schedule_with_warmup


class CocoSigStamp(Dataset):
    def __init__(self, ann_path: Path, img_dir: Path, processor):
        coco = json.loads(ann_path.read_text())
        self.imgs = {im["id"]: im for im in coco["images"]}
        self.anns_by_img = {}
        for ann in coco["annotations"]:
            self.anns_by_img.setdefault(ann["image_id"], []).append(ann)
        self.img_dir = img_dir
        self.processor = processor

        self.items = []
        for img_id, im in self.imgs.items():
            fname = im["file_name"]
            path = img_dir / fname
            if path.exists():
                self.items.append(img_id)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_id = self.items[idx]
        im_info = self.imgs[img_id]
        img = Image.open(self.img_dir / im_info["file_name"]).convert("RGB")
        anns = []
        for ann in self.anns_by_img.get(img_id, []):
            x, y, w, h = ann["bbox"]
            anns.append(
                {
                    "id": ann["id"],
                    "image_id": img_id,
                    "category_id": ann["category_id"],
                    "bbox": [x, y, x + w, y + h],
                    "area": ann["area"],
                    "iscrowd": 0,
                }
            )
        target = {"image_id": img_id, "annotations": anns}
        return img, target

def collate_fn(batch, processor, device, prompts):
    images, targets = zip(*batch)
    text = [prompts] * len(images)
    # Build labels expected by OWLv2: list of dicts with class_labels and boxes
    labels = []
    for t in targets:
        cls_labels = []
        boxes = []
        for ann in t["annotations"]:
            cls_labels.append(ann["category_id"] - 1)  # classes 0,1
            boxes.append(ann["bbox"])
        if len(cls_labels) == 0:
            cls_labels = torch.tensor([], dtype=torch.long)
            boxes = torch.zeros((0, 4), dtype=torch.float)
        labels.append({"class_labels": torch.tensor(cls_labels, dtype=torch.long),
                       "boxes": torch.tensor(boxes, dtype=torch.float)})
    inputs = processor(text=text, images=list(images), return_tensors="pt", padding=True)
    out = {
        "pixel_values": inputs.pixel_values.to(device),
        "input_ids": inputs.input_ids.to(device),
        "attention_mask": inputs.attention_mask.to(device),
        "labels": labels,
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", default="google/owlv2-base-patch16-ensemble")
    ap.add_argument("--train_file", default="owlv2_data/annotations.json")
    ap.add_argument("--image_dir", default="owlv2_data/images")
    ap.add_argument("--output_dir", default="owlv2_lora_out")
    ap.add_argument("--num_train_epochs", type=int, default=10)
    ap.add_argument("--per_device_train_batch_size", type=int, default=2)
    ap.add_argument("--learning_rate", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    processor = Owlv2Processor.from_pretrained(args.model_name_or_path)
    model = Owlv2ForObjectDetection.from_pretrained(args.model_name_or_path).to(device)

    prompts = ["signature", "stamp"]
    ds = CocoSigStamp(Path(args.train_file), Path(args.image_dir), processor)
    dl = DataLoader(ds, batch_size=args.per_device_train_batch_size, shuffle=True,
                    collate_fn=lambda b: collate_fn(b, processor, device, prompts))

    total_steps = args.num_train_epochs * (len(dl) // args.gradient_accumulation_steps + 1)
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    model.train()
    step = 0
    for epoch in range(args.num_train_epochs):
        for batch in dl:
            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optim.step()
                scheduler.step()
                optim.zero_grad()
            step += 1
        print(f"epoch {epoch+1}/{args.num_train_epochs} done, last_loss={loss.item()*args.gradient_accumulation_steps:.4f}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print("saved to", args.output_dir)

if __name__ == "__main__":
    main()
