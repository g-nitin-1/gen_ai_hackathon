"""
Train lightweight text models (TF-IDF + linear heads) for dealer_name, model_name,
horse_power, and asset_cost using OCR lines from gold labels.

Usage:
  python3 train_text_field_model.py --ocr both

Outputs:
  text_models/field_models.joblib  (vectorizer + heads)
"""

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
from paddleocr import PaddleOCR
import easyocr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from tqdm import tqdm


def fuzzy_match_true(line: str, true_val: str) -> bool:
    if not true_val:
        return False
    return true_val.lower() in line.lower()


def label_hp(line: str, true_hp):
    if true_hp is None:
        return False
    m = re.search(r"(\d{2,3}(?:\.\d)?)\s*(?:hp|h\.p)", line, re.IGNORECASE)
    if not m:
        return False
    try:
        val = float(m.group(1))
    except Exception:
        return False
    return abs(val - true_hp) <= 5


def label_cost(line: str, true_cost):
    if true_cost is None:
        return False
    digits = re.sub(r"\D", "", line)
    if len(digits) < 5:
        return False
    try:
        val = int(digits)
    except Exception:
        return False
    # accept within 40%
    return abs(val - true_cost) / max(true_cost, 1) <= 0.4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr", type=str, default="both", choices=["paddle", "easy", "both"])
    args = parser.parse_args()

    label_dir = Path("annotations/seed/done/labels")
    img_dir = Path("annotations/seed/done/images")
    records = []
    for lp in label_dir.glob("*.json"):
        data = json.loads(lp.read_text())
        f = data.get("fields", {})
        records.append(
            {
                "doc": lp.stem,
                "dealer": f.get("dealer_name", "") or "",
                "model": f.get("model_name", "") or "",
                "hp": f.get("horse_power", None),
                "cost": f.get("asset_cost", None),
            }
        )

    paddle = PaddleOCR(lang="en", use_angle_cls=True, show_log=False, use_gpu=True) if args.ocr in ("paddle", "both") else None
    easy = easyocr.Reader(["en"], gpu=True) if args.ocr in ("easy", "both") else None

    texts = []
    y_dealer = []
    y_model = []
    y_hp_cls = []
    y_hp_reg = []
    y_cost_cls = []
    y_cost_reg = []

    for rec in tqdm(records, desc="OCR"):
        img_path = img_dir / f"{rec['doc']}.png"
        if not img_path.exists():
            continue
        lines = []
        if paddle:
            p_out = paddle.ocr(str(img_path), cls=True)
            for page in p_out:
                for line in page:
                    lines.append(line[1][0])
        if easy:
            for _, text, _ in easy.readtext(str(img_path), detail=1, paragraph=False, min_size=10, text_threshold=0.4):
                lines.append(text)
        for ln in lines:
            texts.append(ln)
            y_dealer.append(1 if fuzzy_match_true(ln, rec["dealer"]) else 0)
            y_model.append(1 if fuzzy_match_true(ln, rec["model"]) else 0)
            hp_pos = label_hp(ln, rec["hp"])
            y_hp_cls.append(1 if hp_pos else 0)
            y_hp_reg.append(rec["hp"] if hp_pos else 0.0)
            cost_pos = label_cost(ln, rec["cost"])
            y_cost_cls.append(1 if cost_pos else 0)
            y_cost_reg.append(rec["cost"] if cost_pos else 0.0)

    vec = TfidfVectorizer(ngram_range=(1, 3), min_df=1, max_features=40000)
    X = vec.fit_transform(texts)

    clf_dealer = LogisticRegression(max_iter=300, class_weight="balanced")
    clf_model = LogisticRegression(max_iter=300, class_weight="balanced")
    clf_hp = LogisticRegression(max_iter=300, class_weight="balanced")
    clf_cost = LogisticRegression(max_iter=300, class_weight="balanced")
    reg_hp = Ridge(alpha=1.0)
    reg_cost = Ridge(alpha=1.0)

    clf_dealer.fit(X, y_dealer)
    clf_model.fit(X, y_model)
    clf_hp.fit(X, y_hp_cls)
    clf_cost.fit(X, y_cost_cls)
    reg_hp.fit(X, y_hp_reg)
    reg_cost.fit(X, y_cost_reg)

    Path("text_models").mkdir(exist_ok=True)
    joblib.dump(
        {
            "vec": vec,
            "dealer": clf_dealer,
            "model": clf_model,
            "hp_cls": clf_hp,
            "hp_reg": reg_hp,
            "cost_cls": clf_cost,
            "cost_reg": reg_cost,
        },
        "text_models/field_models.joblib",
    )
    print("saved text_models/field_models.joblib")


if __name__ == "__main__":
    main()
