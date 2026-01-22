"""
Move completed annotations from in_progress to done.

Copies images, labels, and optionally marked_images overlays.

Usage:
  python3 move_to_done.py [--clear-in-progress]
"""

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
IN_PROGRESS_IMG = ROOT / "annotations" / "seed" / "in_progress" / "images"
IN_PROGRESS_LAB = ROOT / "annotations" / "seed" / "in_progress" / "labels"
DONE_IMG = ROOT / "annotations" / "seed" / "done" / "images"
DONE_LAB = ROOT / "annotations" / "seed" / "done" / "labels"
DONE_MARKED = ROOT / "annotations" / "seed" / "done" / "marked_images"
OVERLAY_DIR = ROOT / "overlays"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear-in-progress", action="store_true", help="Clear in_progress after moving")
    args = parser.parse_args()

    DONE_IMG.mkdir(parents=True, exist_ok=True)
    DONE_LAB.mkdir(parents=True, exist_ok=True)
    DONE_MARKED.mkdir(parents=True, exist_ok=True)

    moved = 0

    # Move images and labels
    for img_path in IN_PROGRESS_IMG.glob("*.png"):
        stem = img_path.stem
        lab_path = IN_PROGRESS_LAB / f"{stem}.json"

        if not lab_path.exists():
            print(f"Warning: No label for {img_path.name}, skipping")
            continue

        # Copy image to done/images
        dest_img = DONE_IMG / img_path.name
        shutil.copy2(img_path, dest_img)

        # Copy label to done/labels
        dest_lab = DONE_LAB / lab_path.name
        shutil.copy2(lab_path, dest_lab)

        # Copy overlay to done/marked_images if exists
        overlay_path = OVERLAY_DIR / f"{stem}_overlay.png"
        if overlay_path.exists():
            dest_marked = DONE_MARKED / f"{stem}_overlay.png"
            shutil.copy2(overlay_path, dest_marked)

        moved += 1
        print(f"Moved: {stem}")

    print(f"\n✓ Moved {moved} completed annotations to done/")
    print(f"  Images: {DONE_IMG}")
    print(f"  Labels: {DONE_LAB}")
    print(f"  Marked: {DONE_MARKED}")

    # Clear in_progress if requested
    if args.clear_in_progress:
        for f in IN_PROGRESS_IMG.glob("*"):
            f.unlink()
        for f in IN_PROGRESS_LAB.glob("*"):
            f.unlink()
        # Also clear backup if exists
        backup_dir = ROOT / "annotations" / "seed" / "backup" / "labels"
        if backup_dir.exists():
            for f in backup_dir.glob("*"):
                f.unlink()
        # Clear overlays
        for f in OVERLAY_DIR.glob("*"):
            f.unlink()
        print(f"\n✓ Cleared in_progress, backup, and overlays directories")


if __name__ == "__main__":
    main()
