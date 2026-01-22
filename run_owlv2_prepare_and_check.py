"""
Helper script to rebuild the OWLv2 COCO dataset and print counts.
Runs build_owlv2_coco.py and then reports images/annotations.
"""
import json
from pathlib import Path
import subprocess


def main():
    # rebuild coco
    subprocess.run(["python3", "build_owlv2_coco.py"], check=True)
    ann_path = Path("owlv2_data/annotations.json")
    if not ann_path.exists():
        print("annotations.json not found; build step failed.")
        return
    data = json.loads(ann_path.read_text())
    print("images:", len(data.get("images", [])), "annotations:", len(data.get("annotations", [])))
    if len(data.get("images", [])) == 0 or len(data.get("annotations", [])) == 0:
        print("WARNING: no images or annotations found. Check done labels for signature/stamp bboxes and matching PNGs.")


if __name__ == "__main__":
    main()
