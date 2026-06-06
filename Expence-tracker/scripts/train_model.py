#!/usr/bin/env python3
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib, pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from app.core.config import settings
from app.ml.models.classifier import build_tfidf_pipeline, build_label_encoder
from app.ml.utils.preprocessing import build_feature_text

SAMPLE_DATA = [
    ("Spotify monthly subscription", "Entertainment"),
    ("Netflix", "Entertainment"),
    ("Uber ride to airport", "Transport"),
    ("Grab taxi", "Transport"),
    ("Whole Foods groceries", "Groceries"),
    ("Supermarket weekly shop", "Groceries"),
    ("Electricity bill March", "Utilities"),
    ("Internet service April", "Utilities"),
    ("Restaurant dinner with client", "Dining"),
    ("McDonald's lunch", "Dining"),
    ("Gym membership", "Health"),
    ("Pharmacy prescription", "Health"),
    ("AWS monthly invoice", "Software"),
    ("GitHub Pro plan", "Software"),
    ("Hotel New York", "Travel"),
    ("Flight to London", "Travel"),
    ("Amazon purchase", "Shopping"),
    ("IKEA furniture", "Shopping"),
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/expenses.csv")
    args = parser.parse_args()

    p = Path(args.data)
    if p.exists():
        df = pd.read_csv(p)
        print(f"Loaded {len(df)} rows from {p}")
    else:
        print(f"No CSV found. Using built-in sample data ({len(SAMPLE_DATA)} rows).")
        df = pd.DataFrame(SAMPLE_DATA, columns=["description", "category"])

    X     = df["description"].apply(build_feature_text).tolist()
    y_raw = df["category"].tolist()
    le    = build_label_encoder(sorted(set(y_raw)))
    y     = le.transform(y_raw)

    pipeline = build_tfidf_pipeline()
    cv       = StratifiedKFold(n_splits=min(5, len(set(y_raw))), shuffle=True, random_state=42)
    scores   = cross_val_score(pipeline, X, y, cv=cv, scoring="f1_macro")
    print(f"CV macro-F1: {scores.mean():.3f} +/- {scores.std():.3f}")

    pipeline.fit(X, y)
    Path("data/models").mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, settings.MODEL_PATH)
    joblib.dump(le, settings.LABEL_ENCODER_PATH)
    print(f"Model saved. Categories: {list(le.classes_)}")

if __name__ == "__main__":
    main()
