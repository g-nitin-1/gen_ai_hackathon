# Seed Annotation Guide

Purpose: create a small, high-quality seed set (≈15–25 pages) with ground-truth for the six required fields (dealer name, model name, horse power, asset cost, signature bbox, stamp bbox).

Note: this guide covers *ground-truth annotation*. The final inference output JSON (as per the problem statement) will also include fields like `confidence`, `processing_time_sec`, and `cost_estimate_usd`; those are produced by your pipeline at inference time and are not part of the manual labels.

## Directory convention
- `annotations/seed/images/`: copies or symlinks of selected PNGs from `train/`.
- `annotations/seed/labels/<doc_id>.json`: one JSON per page following the schema below.
- Keep a short tracker (e.g., `annotations/seed/status.csv`) with columns `file,status,checked_by,notes` if you want progress tracking.

## JSON schema (per page)
```json
{
  "doc_id": "90018465649_OTHERS_v1_pg1",
  "fields": {
    "dealer_name": "ABC Tractors Pvt Ltd",
    "model_name": "Mahindra 575 DI",
    "horse_power": 50,
    "asset_cost": 525000,
    "signature": { "present": true, "bbox": [100, 1450, 320, 1600] },
    "stamp": { "present": true, "bbox": [420, 1420, 680, 1605] }
  },
  "source": "seed_manual",
  "notes": ""
}
```
- Coordinate system: origin at top-left of the original image; `bbox` as `[x1, y1, x2, y2]` in pixels. If a field is absent, set `present: false` and `bbox: null`.
- If you annotate a resized image, record the scale factor in `notes` and later remap to original coordinates.
- For multi-page documents, treat each page independently (`doc_id` should match the page filename).

## Using labelme for stamp/signature boxes
1) Install: `pip install labelme` (use python3 environment).
2) Run: `labelme train/<file>.png`. Save the JSON as `annotations/seed/labels/<doc_id>_labelme.json`.
3) Draw two rectangles and label them `signature` and `stamp` (set `group_id` or `label` accordingly).
4) After saving, open the labelme JSON and copy the two corner points for each rectangle; convert to our schema as `[min_x, min_y, max_x, max_y]`. Paste into the final `<doc_id>.json` schema above.

## Capturing text fields
- Read the page (and optionally its OCR) to fill `dealer_name`, `model_name`, `horse_power` (numeric), and `asset_cost` (numeric, digits only). Use the exact strings visible; avoid adding extra punctuation or currency symbols.
- If uncertain, leave `notes` explaining the ambiguity for later review.

## Quality check
- Double-check coordinates align with the correct region after saving.
- Validate JSON format (run a quick `python -m json.tool annotations/seed/labels/<doc_id>.json`).
- Keep 4–5 pages aside as a validation subset you will not change later.
