"""
Fixed LoRA fine-tune for Qwen2.5-VL on the gold set.
Properly handles image processing for training.

Usage:
  python3 vlm_lora_train_v2.py \
    --model-path models/qwen2.5-vl-7b-instruct \
    --train-file data/vlm_gold.jsonl \
    --output-dir vlm_lora_out \
    --batch-size 1 --accum 8 --epochs 3 --lr 2e-4 --device 0 --load-4bit
"""

import argparse
import base64
import json
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm


class VlmTrainDataset(Dataset):
    """Dataset that properly processes images for Qwen2.5-VL training."""

    def __init__(self, path: Path, processor, max_length=512):
        self.items: List[Dict[str, Any]] = []
        self.processor = processor
        self.max_length = max_length

        with path.open() as f:
            for line in f:
                rec = json.loads(line)
                self.items.append(rec)
        print(f"Loaded {len(self.items)} training samples")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rec = self.items[idx]
        system = rec["system"]
        resp_text = json.dumps(rec["response"])

        # Load and resize image
        img = Image.open(BytesIO(base64.b64decode(rec["image_b64"]))).convert("RGB")
        # Use smaller size to avoid memory issues
        img = img.resize((448, 448), resample=Image.BICUBIC)

        # Build the conversation for training
        conversation = [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "Extract fields as JSON."}
            ]},
            {"role": "assistant", "content": resp_text}
        ]

        # Apply chat template to get the full text
        text = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False
        )

        # Process with image
        inputs = self.processor(
            text=[text],
            images=[img],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        # Get input_ids and create labels
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        pixel_values = inputs["pixel_values"].squeeze(0) if "pixel_values" in inputs else None
        image_grid_thw = inputs.get("image_grid_thw", None)
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.squeeze(0)

        # Create labels - mask everything except the assistant response
        labels = input_ids.clone()

        # Find where assistant response starts (after "assistant" token)
        # For simplicity, we'll train on the full sequence
        # In practice, you'd mask the prompt tokens with -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }


def collate_fn(batch, pad_token_id):
    """Custom collate function to handle variable length sequences."""
    # Find max length
    max_len = max(b["input_ids"].size(0) for b in batch)

    input_ids = []
    attention_mask = []
    labels = []
    pixel_values = []
    image_grid_thw = []

    for b in batch:
        # Pad input_ids
        pad_len = max_len - b["input_ids"].size(0)
        input_ids.append(torch.cat([
            b["input_ids"],
            torch.full((pad_len,), pad_token_id, dtype=torch.long)
        ]))
        attention_mask.append(torch.cat([
            b["attention_mask"],
            torch.zeros(pad_len, dtype=torch.long)
        ]))
        labels.append(torch.cat([
            b["labels"],
            torch.full((pad_len,), -100, dtype=torch.long)
        ]))

        if b["pixel_values"] is not None:
            pixel_values.append(b["pixel_values"])
        if b["image_grid_thw"] is not None:
            image_grid_thw.append(b["image_grid_thw"])

    result = {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
    }

    if pixel_values:
        result["pixel_values"] = torch.stack(pixel_values)
    if image_grid_thw:
        result["image_grid_thw"] = torch.stack(image_grid_thw)

    return result


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
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--max-length", type=int, default=512)
    args = ap.parse_args()

    device = f"cuda:{args.device}" if args.device != "cpu" else "cpu"

    print(f"Loading processor from {args.model_path}")
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=False
    )

    # Ensure pad token
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    print(f"Loading model (4-bit: {args.load_4bit})")
    quant_cfg = None
    if args.load_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        device_map="auto" if args.load_4bit else device,
        trust_remote_code=True,
        quantization_config=quant_cfg,
        torch_dtype=torch.bfloat16,
    )

    if args.load_4bit:
        model = prepare_model_for_kbit_training(model)

    # Configure LoRA - only target language model layers
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Create dataset
    print("Loading dataset...")
    dataset = VlmTrainDataset(Path(args.train_file), processor, max_length=args.max_length)

    pad_token_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_token_id),
        num_workers=0,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Training loop
    model.train()
    global_step = 0

    for epoch in range(args.epochs):
        epoch_loss = 0
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for step, batch in enumerate(progress):
            # Move batch to device
            batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            try:
                outputs = model(**batch)
                loss = outputs.loss / args.accum
                loss.backward()

                epoch_loss += loss.item() * args.accum

                if (step + 1) % args.accum == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                progress.set_postfix({"loss": f"{loss.item() * args.accum:.4f}"})

            except Exception as e:
                print(f"Error at step {step}: {e}")
                optimizer.zero_grad()
                continue

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

    # Save model
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    print(f"Model saved to {out_dir}")


if __name__ == "__main__":
    main()
