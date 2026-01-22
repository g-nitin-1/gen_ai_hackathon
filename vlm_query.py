"""
Run a vision-language model (e.g., Qwen2.5-VL) to read fields directly from page images.
It prompts the model to return dealer_name, model_name, horse_power, asset_cost as JSON.

Usage example (model must be present locally; not downloaded here):
  python3 vlm_query.py --model-path /path/to/qwen2.5-vl-7b-instruct --device 0 --limit 15

Outputs:
  annotations/seed/in_progress/labels_vlm/<image_stem>.json  (raw VLM response + parsed fields)
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from tqdm import tqdm
from peft import PeftModel


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    # Grab first {...} block
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Local path to Qwen2.5-VL (e.g., qwen2.5-vl-7b-instruct)")
    parser.add_argument("--adapter", default=None, help="Optional LoRA adapter path to load on top of base model")
    parser.add_argument("--device", default="0", help="cuda device id or cpu")
    parser.add_argument("--images-dir", default="train", help="directory with page images")
    parser.add_argument("--done-dir", default="annotations/seed/done/images", help="skip images already done")
    parser.add_argument("--out-dir", default="annotations/seed/in_progress/labels_vlm", help="where to write VLM JSON outputs")
    parser.add_argument("--limit", type=int, default=15, help="max images to process")
    parser.add_argument("--load-4bit", action="store_true", help="load model in 4-bit to save VRAM")
    args = parser.parse_args()

    device = f"cuda:{args.device}" if args.device != "cpu" else "cpu"
    print(f"loading model from {args.model_path} on {device}")
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
        torch_dtype=torch.float16 if "cuda" in device else torch.float32,
        device_map=device_map,
        quantization_config=quant_cfg,
        trust_remote_code=True,
    )
    if args.adapter:
        print(f"loading LoRA adapter from {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)

    images_dir = Path(args.images_dir)
    done = set(p.stem for p in Path(args.done_dir).glob("*.png"))
    candidates = [p for p in images_dir.glob("*.png") if p.stem not in done]
    candidates = candidates[: args.limit]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"candidates: {len(candidates)}")

    system_prompt = (
        "You are a document extraction model. Read the tractor invoice/quotation image and return JSON with:\n"
        "dealer_name (string), model_name (string), horse_power (ONLY the numeric value, e.g. 45 not '45 HP'), "
        "asset_cost (ONLY the numeric value, no commas or currency symbols). "
        "If a field is missing, use empty string or null. Do not include units or text in numeric fields."
    )

    for img_path in tqdm(candidates, desc="VLM"):
        image = Image.open(img_path).convert("RGB")
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Return JSON only."},
                ],
            },
        ]
        chat_str = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        proc = processor(text=[chat_str], images=[image], return_tensors="pt")
        input_ids = proc.input_ids.to(device)
        attention_mask = proc.attention_mask.to(device)
        pixel_values = proc.pixel_values.to(device, dtype=model.dtype)
        grid = proc.image_grid_thw.to(device) if proc.image_grid_thw is not None else torch.tensor([[1, 1, 1]], device=device)

        with torch.no_grad():
            gen = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=grid,
                max_new_tokens=200,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )
        seq = gen.sequences[0]
        generated = processor.decode(seq, skip_special_tokens=True)
        # confidence: mean logprob of generated tokens
        prompt_len = input_ids.shape[1]
        gen_tokens = seq[prompt_len:]
        scores = gen.scores
        logprobs = []
        for i, token_id in enumerate(gen_tokens):
            if i >= len(scores):
                break
            logits = scores[i][0]
            lp = torch.log_softmax(logits, dim=-1)[token_id].item()
            logprobs.append(lp)
        conf = float(sum(logprobs) / len(logprobs)) if logprobs else 0.0
        parsed = extract_json(generated)

        data = {
            "doc_id": img_path.stem,
            "vlm_model": args.model_path,
            "raw_text": generated,
            "parsed": parsed or {},
            "conf_logprob": conf,
        }
        out_file = out_dir / f"{img_path.stem}.json"
        out_file.write_text(json.dumps(data, indent=2))

    print(f"wrote {len(candidates)} VLM outputs to {out_dir}")


if __name__ == "__main__":
    main()
