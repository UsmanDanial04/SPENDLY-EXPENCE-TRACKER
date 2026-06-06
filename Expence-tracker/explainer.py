"""
explainer.py — SHAP-based explainability for the expense classifier.

Wraps the saved TF-IDF + LogisticRegression pipeline and adds per-prediction
feature-contribution explanations via ``shap.LinearExplainer``.

Typical usage::

    from explainer import ExpenseExplainer

    ex = ExpenseExplainer("model.joblib")
    result = ex.explain_prediction("MCDONALDS #4829 PAYMENT")
    print(result["explanation_text"])
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import shap

# Suppress SHAP's FutureWarning about feature_perturbation deprecation
warnings.filterwarnings("ignore", category=FutureWarning, module="shap")

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_MODEL_PATH = "model.joblib"
TOP_N_FEATURES     = 5

_DIVIDER     = "─" * 60
_THIN_DIV    = "·" * 60


# ── Explainer class ──────────────────────────────────────────────────────────

class ExpenseExplainer:
    """
    SHAP-based explainer for the ``ExpenseClassifier`` pipeline.

    Loads a saved ``.joblib`` model file produced by ``model_tfidf.py`` and
    wraps it with a ``shap.LinearExplainer`` that operates directly on the
    TF-IDF transformed feature matrix.

    Attributes:
        model_path:    Path to the loaded ``.joblib`` file.
        pipeline:      The full sklearn ``Pipeline`` (features + clf).
        label_encoder: Fitted ``LabelEncoder`` mapping indices ↔ class names.
        classes:       Ordered list of category names.
        feature_names: Combined feature names from both TF-IDF vectorisers
                       (``char:<ngram>`` and ``word:<token>``).
        explainer:     Fitted ``shap.LinearExplainer`` instance.
    """

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH) -> None:
        """
        Load the model and initialise the SHAP explainer.

        Args:
            model_path: Path to the ``.joblib`` file saved by
                        ``ExpenseClassifier.save()``.

        Raises:
            FileNotFoundError: If ``model_path`` does not exist.
            KeyError:          If the joblib payload is missing expected keys.
        """
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found: '{path}'. "
                "Train and save a model first by running model_tfidf.py."
            )

        payload = joblib.load(path)
        self.model_path    = path
        self.pipeline      = payload["pipeline"]
        self.label_encoder = payload["label_encoder"]
        self.classes: list[str] = payload["classes"]

        # ── Decompose pipeline ──────────────────────────────────────────────
        self._feature_union  = self.pipeline.named_steps["features"]
        self._logistic_reg   = self.pipeline.named_steps["clf"]

        # ── Build combined feature name list ────────────────────────────────
        char_vec = self._feature_union.transformer_list[0][1]  # char_ngrams
        word_vec = self._feature_union.transformer_list[1][1]  # word_ngrams
        char_names = [
            f"char:{n}" for n in char_vec.get_feature_names_out()
        ]
        word_names = [
            f"word:{n}" for n in word_vec.get_feature_names_out()
        ]
        self.feature_names: list[str] = char_names + word_names

        # ── Initialise SHAP LinearExplainer ────────────────────────────────
        # Background = zero vector (mean-field baseline).
        # This means SHAP values represent each feature's contribution
        # relative to a "blank" input, which is interpretable for sparse
        # TF-IDF features where absence means the token isn't present.
        background = np.zeros((1, len(self.feature_names)))
        self.explainer = shap.LinearExplainer(self._logistic_reg, background)

    # ── Public API ───────────────────────────────────────────────────────────

    def explain_prediction(self, description: str) -> dict[str, Any]:
        """
        Predict the expense category and explain the top contributing features.

        The SHAP values are extracted for the predicted class only, so
        ``top_features`` reflects what pushed the model toward *that*
        category rather than an average over all classes.

        Args:
            description: A raw or pre-cleaned transaction description string.

        Returns:
            A dict with the following keys:

            ``predicted_category`` (str)
                Top predicted category name.

            ``confidence`` (float)
                Probability of the predicted category (0–1).

            ``top_features`` (list[dict])
                Up to ``TOP_N_FEATURES`` dicts, each with:

                * ``word``         – feature name (e.g. ``"mcdonalds"``)
                * ``contribution`` – absolute SHAP value (float)
                * ``direction``    – ``"positive"`` if the feature pushed
                  toward the predicted class, ``"negative"`` otherwise

            ``explanation_text`` (str)
                Human-readable summary sentence.
        """
        # ── Transform description through TF-IDF ───────────────────────────
        X_sparse = self._feature_union.transform([description])
        X_dense  = X_sparse.toarray()

        # ── Predict ────────────────────────────────────────────────────────
        proba: np.ndarray = self._logistic_reg.predict_proba(X_sparse)[0]
        pred_idx          = int(np.argmax(proba))
        predicted_cat     = self.classes[pred_idx]
        confidence        = float(proba[pred_idx])

        # ── SHAP values: shape (n_samples, n_features, n_classes) ──────────
        sv: np.ndarray = self.explainer.shap_values(X_dense)
        # Slice to (n_features,) for the predicted class
        sv_for_class: np.ndarray = sv[0, :, pred_idx]

        # ── Extract top-N by absolute contribution ─────────────────────────
        top_features = self._extract_top_features(sv_for_class, top_n=TOP_N_FEATURES)

        # ── Build explanation text ─────────────────────────────────────────
        # Use only word-level features for human-readable text (skip char n-grams)
        readable_words = [
            f["word"].replace("word:", "")
            for f in top_features
            if f["word"].startswith("word:") and f["direction"] == "positive"
        ]
        if not readable_words:
            # Fallback: use any positive feature, stripping prefix
            readable_words = [
                f["word"].split(":", 1)[-1]
                for f in top_features
                if f["direction"] == "positive"
            ]
        if readable_words:
            kw_str = ", ".join(readable_words[:3])
            explanation_text = (
                f"Classified as {predicted_cat} because of keywords: {kw_str}."
            )
        else:
            explanation_text = (
                f"Classified as {predicted_cat} "
                f"(confidence {confidence * 100:.1f}%)."
            )

        return {
            "predicted_category": predicted_cat,
            "confidence":         round(confidence, 6),
            "top_features":       top_features,
            "explanation_text":   explanation_text,
        }

    def explain_batch(
        self, descriptions: list[str]
    ) -> list[dict[str, Any]]:
        """
        Explain predictions for multiple descriptions efficiently.

        Args:
            descriptions: List of transaction description strings.

        Returns:
            List of dicts in the same format as :meth:`explain_prediction`.
        """
        return [self.explain_prediction(d) for d in descriptions]

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _extract_top_features(
        self, shap_values: np.ndarray, top_n: int = 5
    ) -> list[dict[str, Any]]:
        """
        Select the top-N features by absolute SHAP value.

        Args:
            shap_values: 1-D array of SHAP values for a single class.
            top_n:       Number of features to return.

        Returns:
            List of feature dicts sorted by descending |SHAP value|.
        """
        # Only consider features that are actually non-zero in the input
        nonzero_mask  = shap_values != 0.0
        nonzero_idx   = np.where(nonzero_mask)[0]

        if len(nonzero_idx) == 0:
            return []

        nonzero_vals  = shap_values[nonzero_idx]
        sorted_order  = np.argsort(np.abs(nonzero_vals))[::-1][:top_n]

        features: list[dict[str, Any]] = []
        for i in sorted_order:
            feat_idx     = int(nonzero_idx[i])
            shap_val     = float(nonzero_vals[i])
            feat_name    = self.feature_names[feat_idx]
            features.append({
                "word":         feat_name,
                "contribution": round(abs(shap_val), 6),
                "direction":    "positive" if shap_val > 0 else "negative",
            })
        return features


# ── Pretty printer ────────────────────────────────────────────────────────────

def _print_explanation(description: str, result: dict[str, Any]) -> None:
    """Print a formatted explanation block to stdout."""
    cat        = result["predicted_category"]
    conf       = result["confidence"]
    features   = result["top_features"]
    expl_text  = result["explanation_text"]

    bar_len    = int(conf * 28)
    conf_bar   = "█" * bar_len + "░" * (28 - bar_len)

    print(f"\n  Input      : \"{description}\"")
    print(f"  Prediction : {cat}  ({conf * 100:.1f}%)")
    print(f"  Confidence : [{conf_bar}]")
    print(f"  Explanation: {expl_text}")

    if features:
        print(f"\n  Top contributing features:")
        for f in features:
            arrow  = "▲" if f["direction"] == "positive" else "▼"
            colour = "\033[32m" if f["direction"] == "positive" else "\033[31m"
            reset  = "\033[0m"
            bar    = "█" * min(int(f["contribution"] * 20), 20)
            name   = f["word"].split(":", 1)[-1]   # strip char:/word: prefix
            prefix = f["word"].split(":")[0]         # char or word
            print(
                f"    {colour}{arrow}{reset}  {name:<22}  "
                f"[{prefix:>4}]  "
                f"{f['contribution']:>7.4f}  {colour}{bar}{reset}"
            )


# ── __main__ block ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    MODEL_PATH = "model.joblib"

    print(f"\n{'=' * 60}")
    print("  explainer.py — SHAP feature attribution demo")
    print(f"{'=' * 60}")

    # ── Load explainer ───────────────────────────────────────────────────────
    try:
        ex = ExpenseExplainer(MODEL_PATH)
        print(f"\n  Model loaded  : {ex.model_path}")
        print(f"  Categories    : {', '.join(ex.classes)}")
        print(f"  Total features: {len(ex.feature_names):,}")
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    # ── Test descriptions ────────────────────────────────────────────────────
    test_cases: list[str] = [
        "MCDONALDS PAYMENT",
        "NETFLIX.COM",
        "SHELL PETROL STN",
        "AMAZON MKTPLACE PMT",
        "UNIVERSITY TUITION",
    ]

    print(f"\n{'─' * 60}")
    print(f"  Running {len(test_cases)} explanations")
    print(f"{'─' * 60}")

    results = ex.explain_batch(test_cases)

    for description, result in zip(test_cases, results):
        print(f"\n{_THIN_DIV}")
        _print_explanation(description, result)

    print(f"\n{'=' * 60}")
    print("  Done.")
    print(f"{'=' * 60}\n")
