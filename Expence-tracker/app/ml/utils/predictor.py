from __future__ import annotations
import logging
from pathlib import Path
from typing import Any
import joblib
from app.core.config import settings

logger = logging.getLogger(__name__)
_pipeline: Any = None
_label_encoder: Any = None

def _load_model():
    global _pipeline, _label_encoder
    if _pipeline is not None:
        return _pipeline, _label_encoder
    model_path   = Path(settings.MODEL_PATH)
    encoder_path = Path(settings.LABEL_ENCODER_PATH)
    if not model_path.exists() or not encoder_path.exists():
        logger.warning("ML model not found – auto-classification disabled. Run scripts/train_model.py")
        return None, None
    _pipeline      = joblib.load(model_path)
    _label_encoder = joblib.load(encoder_path)
    return _pipeline, _label_encoder

def classify_expense(description: str) -> dict | None:
    pipeline, le = _load_model()
    if pipeline is None:
        return None
    proba      = pipeline.predict_proba([description])[0]
    top_idx    = proba.argmax()
    confidence = float(proba[top_idx])
    category   = le.inverse_transform([top_idx])[0]
    return {"category": category, "confidence": confidence}

def get_shap_values(description: str) -> dict[str, float] | None:
    try:
        import shap
        pipeline, le = _load_model()
        if pipeline is None:
            return None
        explainer   = shap.Explainer(pipeline.predict, shap.maskers.Text())
        shap_values = explainer([description])
        tokens      = shap_values.data[0]
        values      = shap_values.values[0].tolist()
        combined    = dict(zip(tokens, values))
        return dict(sorted(combined.items(), key=lambda x: abs(x[1]), reverse=True)[:10])
    except Exception as exc:
        logger.warning("SHAP failed: %s", exc)
        return None
