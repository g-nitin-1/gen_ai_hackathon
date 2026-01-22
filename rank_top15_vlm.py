"""
Pick top-15 VLM predictions by confidence and copy to in_progress folders.
Uses outputs from vlm_query.py in annotations/seed/in_progress/labels_vlm.
Also prunes labels_vlm to only those top-15.
"""

import argparse
import json
from pathlib import Path
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clear", action="store_true", help="clear in_progress images/labels before writing")
    args = ap.parse_args()

    ROOT = Path(".")
    VLM_LABS = ROOT / "annotations" / "seed" / "in_progress" / "labels_vlm"
    TRAIN = ROOT / "train"
    OUT_IMG = ROOT / "annotations" / "seed" / "in_progress" / "images"
    OUT_LAB = ROOT / "annotations" / "seed" / "in_progress" / "labels"

    preds = []
    for f in VLM_LABS.glob("*.json"):
        obj = json.loads(f.read_text())
        conf = float(obj.get("conf_logprob") or 0.0)
        preds.append((conf, f))

    preds.sort(key=lambda x: (-x[0], x[1].name))
    top = preds[:15]
    print("taking top15 by VLM confidence")

    OUT_IMG.mkdir(parents=True, exist_ok=True)
    OUT_LAB.mkdir(parents=True, exist_ok=True)
    if args.clear:
        for p in OUT_IMG.glob("*.png"):
            p.unlink()
        for p in OUT_LAB.glob("*.json"):
            p.unlink()

    for conf, f in top:
        stem = f.stem
        img = TRAIN / f"{stem}.png"
        if not img.exists():
            continue
        shutil.copy2(img, OUT_IMG / img.name)
        obj = json.loads(f.read_text())
        parsed = obj.get("parsed") or {}
        data = {
            "doc_id": stem,
            "fields": {
                "dealer_name": parsed.get("dealer_name") or "",
                "model_name": parsed.get("model_name") or "",
                "horse_power": parsed.get("horse_power"),
                "asset_cost": parsed.get("asset_cost"),
                "signature": {"present": False, "bbox": None},
                "stamp": {"present": False, "bbox": None},
            },
            "source": "vlm",
            "notes": f"vlm_conf_logprob={conf:.3f}",
        }
        OUT_LAB.joinpath(f"{stem}.json").write_text(json.dumps(data, indent=2))

    # prune labels_vlm to only top stems
    keep = {Path(f).stem for _, f in top}
    for f in VLM_LABS.glob("*.json"):
        if f.stem not in keep:
            f.unlink()

    print("wrote top15 to in_progress")


if __name__ == "__main__":
    main()
