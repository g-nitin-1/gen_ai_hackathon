"""
Train doc-level text models (no heuristics) for dealer_name, model_name,
horse_power, and asset_cost using OCR text from gold labels.

Models:
  - TF-IDF (1-3 grams) over full document text
  - dealer_name: multinomial logistic regression
  - model_name: multinomial logistic regression
  - horse_power: ridge regression (trained on docs with hp present)
  - asset_cost: ridge regression (trained on docs with cost present)
  - stores train document vectors for similarity-based confidence

Usage:
  python3 train_doc_field_model.py --ocr both

Outputs:
  text_models/doc_field_models.joblib
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from paddleocr import PaddleOCR
import easyocr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def build_text(img_path: Path, use_paddle: bool, use_easy: bool):
    lines = []
    if use_paddle:
        paddle = PaddleOCR(lang="en", use_angle_cls=True, show_log=False, use_gpu=True)
        p_out = paddle.ocr(str(img_path), cls=True)
        for page in p_out:
            for line in page:
                lines.append(line[1][0])
    if use_easy:
        reader = easyocr.Reader(["en"], gpu=True)
        for _, text, _ in reader.readtext(str(img_path), detail=1, paragraph=False, min_size=10, text_threshold=0.4):
            lines.append(text)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr", type=str, default="both", choices=["paddle", "easy", "both"])
    args = parser.parse_args()

    use_paddle = args.ocr in ("paddle", "both")
    use_easy = args.ocr in ("easy", "both")

    label_dir = Path("annotations/seed/done/labels")
    img_dir = Path("annotations/seed/done/images")

    docs = []
    dealers = []
    models = []
    hps = []
    costs = []
    doc_ids = []

    print("Collecting OCR text from gold set...")
    for lp in tqdm(list(label_dir.glob("*.json")), desc="Gold"):
        data = json.loads(lp.read_text())
        f = data.get("fields", {})
        img_path = img_dir / f"{lp.stem}.png"
        if not img_path.exists():
            continue
        text = build_text(img_path, use_paddle, use_easy)
        docs.append(text)
        dealers.append(f.get("dealer_name", "") or "")
        models.append(f.get("model_name", "") or "")
        hps.append(f.get("horse_power", None))
        costs.append(f.get("asset_cost", None))
        doc_ids.append(lp.stem)

    vec = TfidfVectorizer(ngram_range=(1, 3), max_features=60000, min_df=1)
    X = vec.fit_transform(docs)

    clf_dealer = LogisticRegression(max_iter=500, multi_class="auto")
    clf_model = LogisticRegression(max_iter=500, multi_class="auto")
    clf_dealer.fit(X, dealers)
    clf_model.fit(X, models)

    # Train regressors only on available targets
    hp_mask = [v is not None for v in hps]
    cost_mask = [v is not None for v in costs]
    hp_reg = Ridge(alpha=1.0)
    cost_reg = Ridge(alpha=1.0)
    if any(hp_mask):
        hp_reg.fit(X[hp_mask], np.array(hps)[hp_mask].astype(float))
    else:
        hp_reg = None
    if any(cost_mask):
        cost_reg.fit(X[cost_mask], np.array(costs)[cost_mask].astype(float))
    else:
        cost_reg = None

    Path("text_models").mkdir(exist_ok=True)
    joblib.dump(
        {
            "vec": vec,
            "dealer": clf_dealer,
            "model": clf_model,
            "hp_reg": hp_reg,
            "cost_reg": cost_reg,
            "train_X": X,
            "train_doc_ids": doc_ids,
        },
        "text_models/doc_field_models.joblib",
    )
    print("saved text_models/doc_field_models.joblib")


if __name__ == "__main__":
    main()
