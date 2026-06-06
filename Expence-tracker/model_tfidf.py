"""
model_tfidf.py — TF-IDF + Logistic Regression expense classification pipeline.

Classifies bank transaction descriptions into:
    Food | Transport | Housing | Entertainment | Healthcare
    Shopping | Education | Utilities

Usage (from command line)::

    python model_tfidf.py               # train on transactions.csv, save model.joblib
    python model_tfidf.py --csv path/to/file.csv --out my_model.joblib
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# ── Constants ────────────────────────────────────────────────────────────────

CATEGORIES: list[str] = [
    "Education",
    "Entertainment",
    "Food",
    "Healthcare",
    "Housing",
    "Shopping",
    "Transport",
    "Utilities",
]

_DIVIDER = "─" * 60


# ── Classifier ───────────────────────────────────────────────────────────────

class ExpenseClassifier:
    """
    Two-vectoriser TF-IDF + Logistic Regression classifier for expense
    transaction descriptions.

    The feature pipeline combines:

    * **char_wb n-grams** (2–4): character-level, word-boundary-aware.
      Captures merchant name fragments (``MCDON``, ``NETFL``) robustly.
    * **word n-grams** (1–2): full-token and bigram features.
      Captures phrases like ``PETROL STATION``, ``GAME PASS``.

    Both are merged via ``FeatureUnion`` before ``LogisticRegression``.
    """

    def __init__(self) -> None:
        self.pipeline: Pipeline | None = None
        self.label_encoder: LabelEncoder = LabelEncoder()
        self.classes_: list[str] = []

    # ── Pipeline factory ────────────────────────────────────────────────────

    @staticmethod
    def _build_pipeline() -> Pipeline:
        """Return an untrained sklearn Pipeline."""
        char_tfidf = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            min_df=1,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        word_tfidf = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            strip_accents="unicode",
            token_pattern=r"(?u)\b\w+\b",
        )
        feature_union = FeatureUnion(
            transformer_list=[
                ("char_ngrams", char_tfidf),
                ("word_ngrams", word_tfidf),
            ]
        )
        lr = LogisticRegression(
            C=5.0,
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=42,
        )
        return Pipeline(steps=[("features", feature_union), ("clf", lr)])

    # ── Public API ──────────────────────────────────────────────────────────

    def train(self, descriptions: list[str], labels: list[str]) -> None:
        """
        Fit the TF-IDF + LogisticRegression pipeline.

        Args:
            descriptions: Cleaned transaction description strings.
            labels:       Corresponding category labels.
        """
        if len(descriptions) != len(labels):
            raise ValueError(
                f"descriptions ({len(descriptions)}) and labels "
                f"({len(labels)}) must have the same length."
            )

        encoded = self.label_encoder.fit_transform(labels)
        self.classes_ = list(self.label_encoder.classes_)

        self.pipeline = self._build_pipeline()
        self.pipeline.fit(descriptions, encoded)

    def predict(self, description: str) -> dict[str, Any]:
        """
        Predict the expense category for a single description.

        Args:
            description: A cleaned transaction description string.

        Returns:
            Dict with keys:

            * ``category``          – top predicted category name
            * ``confidence``        – probability of the top category (0–1)
            * ``all_probabilities`` – dict mapping every category to its
              probability, sorted descending
        """
        self._check_fitted()
        proba: np.ndarray = self.pipeline.predict_proba([description])[0]
        top_idx: int = int(np.argmax(proba))
        all_probs = {
            cat: round(float(p), 6)
            for cat, p in sorted(
                zip(self.classes_, proba), key=lambda x: -x[1]
            )
        }
        return {
            "category":         self.classes_[top_idx],
            "confidence":       round(float(proba[top_idx]), 6),
            "all_probabilities": all_probs,
        }

    def predict_batch(self, descriptions: list[str]) -> list[dict[str, Any]]:
        """
        Predict categories for a list of descriptions efficiently.

        Args:
            descriptions: List of cleaned transaction description strings.

        Returns:
            List of dicts in the same format as :meth:`predict`.
        """
        self._check_fitted()
        probas: np.ndarray = self.pipeline.predict_proba(descriptions)
        results: list[dict[str, Any]] = []
        for proba in probas:
            top_idx = int(np.argmax(proba))
            all_probs = {
                cat: round(float(p), 6)
                for cat, p in sorted(
                    zip(self.classes_, proba), key=lambda x: -x[1]
                )
            }
            results.append({
                "category":          self.classes_[top_idx],
                "confidence":        round(float(proba[top_idx]), 6),
                "all_probabilities": all_probs,
            })
        return results

    def evaluate(
        self,
        descriptions: list[str],
        labels: list[str],
        *,
        print_report: bool = True,
    ) -> dict[str, Any]:
        """
        Evaluate the classifier on a held-out set.

        Prints classification report, per-class F1 bar, and confusion matrix.

        Args:
            descriptions: Test descriptions.
            labels:       True category labels.
            print_report: Whether to print metrics to stdout.

        Returns:
            Dict with ``accuracy``, ``macro_f1``, ``per_class_f1``.
        """
        self._check_fitted()
        encoded_true = self.label_encoder.transform(labels)
        encoded_pred = self.pipeline.predict(descriptions)
        pred_labels  = self.label_encoder.inverse_transform(encoded_pred)

        macro_f1   = f1_score(encoded_true, encoded_pred, average="macro")
        per_class  = f1_score(
            encoded_true, encoded_pred, average=None, labels=range(len(self.classes_))
        )
        accuracy   = float((encoded_pred == encoded_true).mean())

        if print_report:
            print(f"\n{_DIVIDER}")
            print("  Classification Report")
            print(_DIVIDER)
            print(
                classification_report(
                    labels, pred_labels, target_names=self.classes_, digits=3
                )
            )

            print(f"{_DIVIDER}")
            print("  Per-class F1 scores")
            print(_DIVIDER)
            for cat, f1 in sorted(
                zip(self.classes_, per_class), key=lambda x: -x[1]
            ):
                bar = "█" * int(f1 * 30)
                print(f"  {cat:<16} {f1:.3f}  {bar}")

            print(f"\n{_DIVIDER}")
            print("  Confusion Matrix")
            print(_DIVIDER)
            cm = confusion_matrix(encoded_true, encoded_pred)
            _print_confusion_matrix(cm, self.classes_)
            print(_DIVIDER)
            print(f"  Accuracy  : {accuracy:.4f}")
            print(f"  Macro F1  : {macro_f1:.4f}")
            print(_DIVIDER)

        return {
            "accuracy":     accuracy,
            "macro_f1":     macro_f1,
            "per_class_f1": dict(zip(self.classes_, per_class.tolist())),
        }

    def save(self, path: str | Path) -> None:
        """
        Serialise the fitted classifier to disk using joblib.

        Args:
            path: File path for the saved model (e.g. ``model.joblib``).
        """
        self._check_fitted()
        payload = {
            "pipeline":      self.pipeline,
            "label_encoder": self.label_encoder,
            "classes":       self.classes_,
        }
        joblib.dump(payload, path, compress=3)
        print(f"\n  Model saved → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "ExpenseClassifier":
        """
        Load a previously saved classifier from disk.

        Args:
            path: Path to the ``.joblib`` file produced by :meth:`save`.

        Returns:
            A ready-to-use ``ExpenseClassifier`` instance.
        """
        payload = joblib.load(path)
        obj = cls()
        obj.pipeline      = payload["pipeline"]
        obj.label_encoder = payload["label_encoder"]
        obj.classes_      = payload["classes"]
        return obj

    # ── Internal helpers ────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if self.pipeline is None:
            raise RuntimeError(
                "Classifier is not trained. Call train() or load() first."
            )


# ── Pretty confusion-matrix printer ─────────────────────────────────────────

def _print_confusion_matrix(cm: np.ndarray, labels: list[str]) -> None:
    """Print a compact ASCII confusion matrix."""
    abbr = [c[:4] for c in labels]
    header = "          " + "  ".join(f"{a:>4}" for a in abbr)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(
            f"\033[1m{v:>4}\033[0m" if j == i else f"{v:>4}"
            for j, v in enumerate(row)
        )
        print(f"  {labels[i]:<10}{row_str}   ← {labels[i]}")


# ── __main__ ─────────────────────────────────────────────────────────────────

def _main() -> None:
    parser = argparse.ArgumentParser(description="Train expense TF-IDF classifier")
    parser.add_argument(
        "--csv",  default="transactions.csv", help="Path to transactions CSV"
    )
    parser.add_argument(
        "--out",  default="model.joblib",     help="Output path for saved model"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Fraction used for evaluation"
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    # ── Load data ───────────────────────────────────────────────────────────
    print(f"\n{_DIVIDER}")
    print("  Loading dataset")
    print(_DIVIDER)
    df = pd.read_csv(csv_path)
    print(f"  Rows loaded  : {len(df)}")

    # Drop rows with missing description or category
    required = ["description", "category"]
    before = len(df)
    df = df.dropna(subset=required)
    # Drop the "Ambiguous" noise category so only labelled data is trained on
    df = df[df["category"].isin(CATEGORIES)]
    print(f"  Rows dropped : {before - len(df)}  (missing fields / ambiguous)")
    print(f"  Rows used    : {len(df)}")
    print(f"\n  Category distribution:")
    for cat, cnt in df["category"].value_counts().items():
        print(f"    {cat:<16} {cnt:>4}")

    descriptions: list[str] = df["description"].tolist()
    labels:       list[str] = df["category"].tolist()

    # ── Train / test split ──────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        descriptions,
        labels,
        test_size=args.test_size,
        random_state=42,
        stratify=labels,
    )
    print(f"\n  Train samples: {len(X_train)}")
    print(f"  Test  samples: {len(X_test)}")

    # ── Train ───────────────────────────────────────────────────────────────
    print(f"\n{_DIVIDER}")
    print("  Training pipeline  (char_wb 2-4 + word 1-2 TF-IDF → LogisticRegression)")
    print(_DIVIDER)
    clf = ExpenseClassifier()
    clf.train(X_train, y_train)
    print("  Training complete.")

    # ── Evaluate ────────────────────────────────────────────────────────────
    clf.evaluate(X_test, y_test)

    # ── Quick inference demo ─────────────────────────────────────────────────
    demo_descriptions = [
        "MCDONALDS PAYMENT",
        "NETFLIX.COM",
        "SHELL PETROL STN",
        "AMAZON MKTPLACE",
        "UNIVERSITY TUITION",
    ]
    print(f"\n{_DIVIDER}")
    print("  Sample predictions")
    print(_DIVIDER)
    results = clf.predict_batch(demo_descriptions)
    for desc, res in zip(demo_descriptions, results):
        print(
            f"  {desc:<28}  →  {res['category']:<16}  "
            f"({res['confidence']*100:.1f}%)"
        )

    # ── Save ────────────────────────────────────────────────────────────────
    clf.save(args.out)
    print(f"\n  Done.\n")


if __name__ == "__main__":
    _main()
