from fastapi import APIRouter, HTTPException
from app.schemas.expense import ClassifyRequest, ClassifyResponse
from app.ml.utils.predictor import classify_expense, get_shap_values

router = APIRouter()

@router.post("/classify", response_model=ClassifyResponse)
def classify(payload: ClassifyRequest):
    result = classify_expense(payload.description)
    if result is None:
        raise HTTPException(status_code=503, detail="ML model not available. Run scripts/train_model.py")
    return ClassifyResponse(**result)

@router.post("/explain", response_model=ClassifyResponse)
def explain(payload: ClassifyRequest):
    result = classify_expense(payload.description)
    if result is None:
        raise HTTPException(status_code=503, detail="ML model not available.")
    return ClassifyResponse(**result, shap_values=get_shap_values(payload.description))
