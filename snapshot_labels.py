"""
Create a backup snapshot of current schema labels before editing with labelme.
Backups are stored in annotations/seed/backup/labels/<file>.json

Usage:
  python3 snapshot_labels.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "annotations" / "seed" / "in_progress" / "labels"
DEST = ROOT / "annotations" / "seed" / "backup" / "labels"


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    count = 0
    for p in SRC.glob("*.json"):
        try:
            data = json.load(p.open())
        except Exception:
            continue
        # Only snapshot files already in schema format
        if not ("fields" in data and isinstance(data["fields"], dict)):
            continue
        dest = DEST / p.name
        json.dump(data, dest.open("w"), indent=2, ensure_ascii=False)
        count += 1
    print(f"Snapshot saved for {count} files -> {DEST}")


if __name__ == "__main__":
    main()
