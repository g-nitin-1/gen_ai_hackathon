"""
Build a JSONL for VLM LoRA training from the gold set.

Each record:
{
  "system": "<system prompt>",
  "image_b64": "<base64 of PNG>",
  "response": {dealer_name, model_name, horse_power, asset_cost}
}

Usage:
  python3 prepare_vlm_jsonl.py --out data/vlm_gold.jsonl
"""

import argparse
import base64
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="annotations/seed/done/images")
    ap.add_argument("--labels", default="annotations/seed/done/labels")
    ap.add_argument("--out", default="data/vlm_gold.jsonl")
    args = ap.parse_args()

    img_dir = Path(args.images)
    lab_dir = Path(args.labels)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    sys_prompt = (
        "You are a document extraction model. Return JSON with dealer_name, model_name, "
        "horse_power (numeric), asset_cost (numeric, no commas). Use empty string/null if missing."
    )

    count = 0
    with out.open("w") as f:
        for lab in lab_dir.glob("*.json"):
            img = img_dir / f"{lab.stem}.png"
            if not img.exists():
                continue
            fields = json.loads(lab.read_text()).get("fields", {})
            b64 = base64.b64encode(img.read_bytes()).decode()
            rec = {
                "system": sys_prompt,
                "image_b64": b64,
                "response": {
                    "dealer_name": fields.get("dealer_name", ""),
                    "model_name": fields.get("model_name", ""),
                    "horse_power": fields.get("horse_power"),
                    "asset_cost": fields.get("asset_cost"),
                },
            }
            f.write(json.dumps(rec) + "\n")
            count += 1
    print(f"wrote {count} records to {out}")


if __name__ == "__main__":
    main()
