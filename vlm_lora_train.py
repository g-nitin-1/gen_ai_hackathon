"""
Minimal LoRA fine-tune for a vision-language model (e.g., Qwen2.5-VL) on the gold set.

Input JSONL format (see the helper in the instructions):
{
  "system": "...system prompt...",
  "image_b64": "<base64 png>",
  "response": {... fields ...}   # will be serialized to JSON for the assistant turn
}

Usage example:
  python3 vlm_lora_train.py \
    --model-path /path/to/qwen2.5-vl-7b-instruct \
    --train-file data/vlm_gold.jsonl \
    --output-dir vlm_lora_out \
    --batch-size 1 --accum 8 --epochs 3 --lr 2e-4 --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 --device 0
"""

import argparse
import base64
import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm


class VlmJsonlDataset(Dataset):
    def __init__(self, path: Path, processor, user_prompt="Extract fields as JSON."):
        self.items: List[Dict[str, Any]] = []
        self.processor = processor
        self.user_prompt = user_prompt
        with path.open() as f:
            for line in f:
                rec = json.loads(line)
                self.items.append(rec)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rec = self.items[idx]
        system = rec["system"]
        resp_text = json.dumps(rec["response"])
        img = Image.open(BytesIO(base64.b64decode(rec["image_b64"]))).convert("RGB")
        # normalize to a fixed size to keep visual grid valid
        img = img.resize((1024, 1024), resample=Image.BICUBIC)
        def to_ids(x):
            if isinstance(x, torch.Tensor):
                return x[0] if x.dim() > 1 else x
            if isinstance(x, str):
                return self.processor.tokenizer(x, return_tensors="pt").input_ids[0]
            if isinstance(x, list):
                return self.processor.tokenizer("".join(x), return_tensors="pt").input_ids[0]
            raise TypeError(f"unsupported type for ids: {type(x)}")
        # messages with assistant response
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": self.user_prompt}],
            },
            {"role": "assistant", "content": [{"type": "text", "text": resp_text}]},
        ]
        # messages without assistant to locate label start
        pre_messages = messages[:-1]
        pre_raw = self.processor.apply_chat_template(
            pre_messages, add_generation_prompt=True, return_tensors="pt"
        )
        full_raw = self.processor.apply_chat_template(
            messages, add_generation_prompt=False, return_tensors="pt"
        )
        pre_ids = to_ids(pre_raw)
        full_ids = to_ids(full_raw)
        labels = full_ids.clone()
        labels[: pre_ids.shape[0]] = -100
        # Qwen2.5-VL expects an image placeholder; use the known special token if available
        img_token = getattr(self.processor, "image_token", "<image>")
        proc_out = self.processor(text=[img_token], images=[img], return_tensors="pt")
        pixel_values = proc_out.pixel_values[0]
        grid = proc_out.image_grid_thw[0] if proc_out.image_grid_thw is not None else None
        if grid is None or grid.numel() == 0 or (grid <= 0).any():
            image_grid_thw = torch.tensor([1, 1, 1], dtype=torch.long)
        else:
            image_grid_thw = torch.clamp(grid, min=1)
        return {
            "input_ids": full_ids,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }


@dataclass
class Collator:
    processor: Any

    def __call__(self, batch):
        input_ids = [b["input_ids"] for b in batch]
        labels = [b["labels"] for b in batch]
        pixel_values = torch.stack([b["pixel_values"] for b in batch])
        image_grid_thw = torch.clamp(torch.stack([b["image_grid_thw"] for b in batch]), min=1)
        pad_token_id = self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        attention_mask = input_ids.ne(pad_token_id).long()
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--train-file", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--device", default="0")
    ap.add_argument("--load-4bit", action="store_true", help="load base model in 4-bit (bitsandbytes) to save VRAM")
    args = ap.parse_args()

    device = f"cuda:{args.device}" if args.device != "cpu" else "cpu"
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    quant_cfg = None
    device_map = device
    if args.load_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        device_map = "auto"

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        device_map=device_map,
        trust_remote_code=True,
        quantization_config=quant_cfg,
    )
    if args.load_4bit:
        model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)

    ds = VlmJsonlDataset(Path(args.train_file), processor)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=Collator(processor))

    num_steps = args.epochs * len(dl) // args.accum + 1
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_steps)

    model.train()
    step = 0
    for epoch in range(args.epochs):
        pbar = tqdm(dl, desc=f"epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.accum
            loss.backward()
            if (step + 1) % args.accum == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            pbar.set_postfix(loss=float(loss) * args.accum)
            step += 1

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print("saved LoRA adapter to", args.output_dir)


if __name__ == "__main__":
    main()
