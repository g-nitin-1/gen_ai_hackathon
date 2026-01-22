"""
Fix labels: merge VLM text fields and clean horse_power format.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VLM_DIR = ROOT / "annotations" / "seed" / "in_progress" / "labels_vlm"
LABEL_DIR = ROOT / "annotations" / "seed" / "in_progress" / "labels"

def clean_hp(val):
    """Extract numeric horse power: '45 HP' -> 45"""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return int(val) if val == int(val) else val
    if isinstance(val, str):
        match = re.search(r'(\d+(?:\.\d+)?)', val)
        if match:
            num = float(match.group(1))
            return int(num) if num == int(num) else num
    return None

def to_num(val):
    """Convert to number, handle commas"""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return val
    if isinstance(val, str):
        s = val.replace(",", "").strip()
        if not s:
            return None
        try:
            return float(s) if "." in s else int(s)
        except:
            return val
    return val

def main():
    fixed = 0
    for label_path in LABEL_DIR.glob("*.json"):
        stem = label_path.stem
        vlm_file = VLM_DIR / f"{stem}.json"

        if not vlm_file.exists():
            print(f"No VLM for {stem}, skipping")
            continue

        # Load both files
        label_data = json.loads(label_path.read_text())
        vlm_obj = json.loads(vlm_file.read_text())
        fields = vlm_obj.get("parsed") or {}

        # Extract and clean
        dealer_name = fields.get("dealer_name") or ""
        model_name = fields.get("model_name") or ""
        horse_power = clean_hp(fields.get("horse_power"))
        asset_cost = to_num(fields.get("asset_cost"))

        # Merge (keep boxes, update text)
        label_data["fields"]["dealer_name"] = dealer_name
        label_data["fields"]["model_name"] = model_name
        label_data["fields"]["horse_power"] = horse_power
        label_data["fields"]["asset_cost"] = asset_cost
        label_data["vlm_conf_logprob"] = vlm_obj.get("conf_logprob", 0.0)

        # Write back
        label_path.write_text(json.dumps(label_data, indent=2))
        fixed += 1
        print(f"✓ {stem}: HP={horse_power}, Cost={asset_cost}")

    print(f"\nFixed {fixed} labels")

if __name__ == "__main__":
    main()
