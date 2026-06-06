"""
routes.py — FastAPI router for all expense tracker endpoints.

Imports from the flat file structure:
    database.py       → SessionLocal, User, Transaction, ModelCorrection
    preprocessor.py   → parse_csv, clean_description
    model_tfidf.py    → ExpenseClassifier, CATEGORIES
    explainer.py      → ExpenseExplainer
    retrainer.py      → ActiveRetrainer, DEFAULT_* paths
    main.py           → get_current_user, get_db, _decode_token (at request time)

Endpoints
---------
POST   /upload                    Upload and classify a bank-statement CSV
GET    /transactions               List transactions for the current user
POST   /predict                    Predict category for a single description
PATCH  /correct/{transaction_id}   Submit a category correction
DELETE /transactions/{id}          Delete a transaction

All routes require a valid Bearer JWT resolved via ``get_current_user``
from main.py (imported lazily to avoid a circular-import cycle).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Annotated, Literal, Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from pydantic import BaseModel, Field
from sqlalchemy import extract
from sqlalchemy.orm import Session

# Flat-file local imports
from database import ModelCorrection, SessionLocal, Transaction, User
from explainer import ExpenseExplainer
from model_tfidf import CATEGORIES, ExpenseClassifier
from preprocessor import clean_description, parse_csv
from retrainer import (
    DEFAULT_CSV_PATH,
    DEFAULT_LOG_PATH,
    DEFAULT_MODEL_PATH,
    ActiveRetrainer,
)

log = logging.getLogger("routes")

router = APIRouter(tags=["expenses"])


# ── DB dependency (mirrors main.py's get_db, avoids circular import) ──────────

def get_db():
    """Yield a SQLAlchemy session scoped to the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


DbDep = Annotated[Session, Depends(get_db)]


# ── Auth: resolve current user from request ───────────────────────────────────

def get_current_user(
    request: Request,
    db: DbDep,
) -> User:
    """
    Validate the ``Authorization: Bearer <token>`` header using main.py's
    ``_decode_token``, then return the matching ``User`` row.

    Imported lazily from ``main`` to break the circular dependency
    (main imports routes; routes must not import main at module level).

    Raises:
        HTTPException 401: Missing, malformed, or invalid token.
        HTTPException 404: Token valid but user no longer exists.
    """
    from main import _decode_token  # lazy — avoids circular import at load time

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = auth_header.removeprefix("Bearer ").strip()
    payload = _decode_token(token)          # raises 401 on invalid/expired
    user_id = int(payload["sub"])
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


# ── App-state helpers ─────────────────────────────────────────────────────────

def _get_classifier(request: Request) -> ExpenseClassifier:
    """Pull the shared classifier from app.state; raise 503 if not loaded."""
    clf = getattr(request.app.state, "classifier", None)
    if clf is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML model is not loaded. Check server logs.",
        )
    return clf


def _get_explainer(request: Request) -> ExpenseExplainer:
    """
    Return a cached ExpenseExplainer from app.state.
    Built on first call so it shares the same model file as the classifier.
    """
    if not hasattr(request.app.state, "explainer") or request.app.state.explainer is None:
        _get_classifier(request)   # confirms model file exists first
        try:
            request.app.state.explainer = ExpenseExplainer(DEFAULT_MODEL_PATH)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            )
    return request.app.state.explainer


def _own_transaction_or_404(
    transaction_id: int, user: User, db: Session
) -> Transaction:
    """Fetch a transaction by PK that belongs to *user*, or raise 404."""
    txn = (
        db.query(Transaction)
        .filter(Transaction.id == transaction_id, Transaction.user_id == user.id)
        .first()
    )
    if txn is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transaction {transaction_id} not found.",
        )
    return txn


# ── Background retraining ─────────────────────────────────────────────────────

def _background_retrain() -> None:
    """
    Opens its own DB session, runs the active-learning threshold check,
    then closes the session.  Called via FastAPI BackgroundTasks so errors
    never surface to the HTTP client.
    """
    db = SessionLocal()
    try:
        retrainer = ActiveRetrainer(
            model_path=DEFAULT_MODEL_PATH,
            db_session=db,
            correction_threshold=20,
            csv_path=DEFAULT_CSV_PATH,
            log_path=DEFAULT_LOG_PATH,
        )
        triggered = retrainer.check_and_retrain()
        if triggered:
            log.info("Background retrain completed.")
    except Exception as exc:
        log.error("Background retrain failed: %s", exc)
    finally:
        db.close()


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class TransactionOut(BaseModel):
    id:                  int
    date:                datetime
    description:         str
    amount:              float
    predicted_category:  Optional[str]
    corrected_category:  Optional[str]
    confidence_score:    Optional[float]
    user_id:             int
    created_at:          datetime

    model_config = {"from_attributes": True}


class FailedRowOut(BaseModel):
    row_number: int
    error:      str
    raw_row:    dict


class UploadResponse(BaseModel):
    transactions_saved: int
    failed_rows:        int
    transactions:       list[TransactionOut]


class TransactionListResponse(BaseModel):
    total:        int
    transactions: list[TransactionOut]


class PredictRequest(BaseModel):
    description: str   = Field(..., min_length=1, max_length=500, examples=["MCDONALDS #442"])
    amount:      float = Field(..., gt=0,                          examples=[12.50])
    date:        datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc),
        examples=["2024-03-15T00:00:00Z"],
    )


class SHAPFeature(BaseModel):
    word:         str
    contribution: float
    direction:    Literal["positive", "negative"]


class PredictResponse(BaseModel):
    transaction_id:     int
    predicted_category: str
    confidence_score:   float
    all_probabilities:  dict[str, float]
    shap_explanation:   list[SHAPFeature]
    explanation_text:   str


class CorrectRequest(BaseModel):
    correct_category: str = Field(
        ...,
        description=f"One of: {', '.join(CATEGORIES)}",
        examples=["Healthcare"],
    )


class CorrectResponse(BaseModel):
    transaction_id:       int
    corrected_category:   str
    retraining_triggered: bool
    message:              str


class DeleteResponse(BaseModel):
    transaction_id: int
    message:        str


# ── POST /upload ──────────────────────────────────────────────────────────────

@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and classify a bank-statement CSV",
)
async def upload_transactions(
    request:      Request,
    file:         UploadFile = File(..., description="Bank-statement CSV"),
    db:           DbDep = None,
    current_user: CurrentUser = None,
) -> UploadResponse:
    """
    Accept a multipart CSV upload, parse and classify each row, and
    bulk-save transactions to the database.

    - Calls ``preprocessor.parse_csv()`` to normalise the CSV.
    - Batch-classifies descriptions via the ML classifier.
    - Saves valid rows with ``predicted_category`` and ``confidence_score``.
    - Returns a summary and the full list of saved transactions.

    Raises:
        HTTPException 400: Not a CSV, empty file, or no valid rows found.
        HTTPException 503: ML model not loaded.
    """
    if file.content_type not in (
        "text/csv", "application/csv", "application/octet-stream", "text/plain"
    ):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Expected a CSV file, got '{file.content_type}'.",
        )

    raw_bytes = await file.read()

    if not raw_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # Parse & clean
    try:
        clean_rows, failed_rows_raw = parse_csv(raw_bytes)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"CSV parsing error: {exc}",
        )

    if not clean_rows:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid rows found in the uploaded CSV.",
        )

    # Batch classify
    clf = _get_classifier(request)
    descriptions = [row["description"] for row in clean_rows]
    predictions  = clf.predict_batch(descriptions)

    # Persist
    saved: list[Transaction] = []
    for row, pred in zip(clean_rows, predictions):
        txn = Transaction(
            user_id            = current_user.id,
            date               = row["date"],
            description        = row["description"],
            amount             = row["amount"],
            predicted_category = pred["category"],
            corrected_category = None,
            confidence_score   = pred["confidence"],
        )
        db.add(txn)
        saved.append(txn)

    db.commit()
    for txn in saved:
        db.refresh(txn)

    log.info(
        "upload: user_id=%d saved=%d failed=%d file=%s",
        current_user.id, len(saved), len(failed_rows_raw), file.filename,
    )

    return UploadResponse(
        transactions_saved = len(saved),
        failed_rows        = len(failed_rows_raw),
        transactions       = [TransactionOut.model_validate(t) for t in saved],
    )


# ── GET /transactions ─────────────────────────────────────────────────────────

@router.get(
    "/transactions",
    response_model=TransactionListResponse,
    summary="List transactions for the current user",
)
def list_transactions(
    request:      Request,
    db:           DbDep = None,
    current_user: CurrentUser = None,
    category: Optional[str] = Query(
        None,
        description=f"Filter by category. One of: {', '.join(CATEGORIES)}",
    ),
    month: Optional[str] = Query(
        None,
        description="Filter by month — YYYY-MM format.",
        pattern=r"^\d{4}-(0[1-9]|1[0-2])$",
    ),
    sort_by: Literal["date", "amount"] = Query(
        "date",
        description="Sort results by date (desc) or amount (desc).",
    ),
) -> TransactionListResponse:
    """
    Return all transactions for the authenticated user.

    Filters:
    - ``category`` — matches ``corrected_category`` if set, else ``predicted_category``.
    - ``month`` — restricts to a calendar month (``YYYY-MM``).

    Sort: descending by ``date`` (default) or ``amount``.

    Raises:
        HTTPException 400: Unknown category or malformed month.
    """
    if category and category not in CATEGORIES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown category '{category}'. Valid values: {CATEGORIES}",
        )

    query = db.query(Transaction).filter(Transaction.user_id == current_user.id)

    if category:
        # Prefer corrected label; fall back to predicted when no correction exists
        query = query.filter(
            (Transaction.corrected_category == category)
            | (
                Transaction.corrected_category.is_(None)
                & (Transaction.predicted_category == category)
            )
        )

    if month:
        try:
            year, mon = int(month[:4]), int(month[5:])
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Could not parse month '{month}'. Use YYYY-MM.",
            )
        query = query.filter(
            extract("year",  Transaction.date) == year,
            extract("month", Transaction.date) == mon,
        )

    query = query.order_by(
        Transaction.amount.desc() if sort_by == "amount" else Transaction.date.desc()
    )

    transactions = query.all()
    log.info(
        "list_transactions: user_id=%d count=%d category=%s month=%s",
        current_user.id, len(transactions), category, month,
    )

    return TransactionListResponse(
        total=len(transactions),
        transactions=[TransactionOut.model_validate(t) for t in transactions],
    )


# ── POST /predict ─────────────────────────────────────────────────────────────

@router.post(
    "/predict",
    response_model=PredictResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Predict category for a single description and save the transaction",
)
def predict_transaction(
    body:         PredictRequest,
    request:      Request,
    db:           DbDep = None,
    current_user: CurrentUser = None,
) -> PredictResponse:
    """
    Classify a single transaction description and return a SHAP explanation.

    Workflow:
    1. Clean description via ``preprocessor.clean_description()``.
    2. Classify via the shared ML model.
    3. Fetch SHAP feature attributions from ``explainer.explain_prediction()``.
    4. Persist the transaction.
    5. Return prediction, confidence, all-class probabilities, and top features.

    Raises:
        HTTPException 400: Description empty after cleaning.
        HTTPException 503: ML model not loaded.
    """
    cleaned = clean_description(body.description)
    if not cleaned:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Description is empty after cleaning.",
        )

    # Classify
    try:
        clf  = _get_classifier(request)
        pred = clf.predict(cleaned)
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Classifier failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {exc}",
        )

    # SHAP explanation (non-fatal if it errors)
    shap_features: list[SHAPFeature] = []
    explanation_text = f"Classified as {pred['category']}."
    try:
        explainer    = _get_explainer(request)
        explanation  = explainer.explain_prediction(cleaned)
        shap_features = [SHAPFeature(**f) for f in explanation["top_features"]]
        explanation_text = explanation["explanation_text"]
    except Exception as exc:
        log.warning("SHAP explanation failed (non-fatal): %s", exc)

    # Persist
    txn = Transaction(
        user_id            = current_user.id,
        date               = body.date,
        description        = cleaned,
        amount             = body.amount,
        predicted_category = pred["category"],
        corrected_category = None,
        confidence_score   = pred["confidence"],
    )
    db.add(txn)
    db.commit()
    db.refresh(txn)

    log.info(
        "predict: user_id=%d txn_id=%d category=%s confidence=%.3f",
        current_user.id, txn.id, pred["category"], pred["confidence"],
    )

    return PredictResponse(
        transaction_id     = txn.id,
        predicted_category = pred["category"],
        confidence_score   = pred["confidence"],
        all_probabilities  = pred["all_probabilities"],
        shap_explanation   = shap_features,
        explanation_text   = explanation_text,
    )


# ── PATCH /correct/{transaction_id} ──────────────────────────────────────────

@router.patch(
    "/correct/{transaction_id}",
    response_model=CorrectResponse,
    summary="Correct the predicted category of a transaction",
)
def correct_transaction(
    transaction_id:   int,
    body:             CorrectRequest,
    background_tasks: BackgroundTasks,
    request:          Request,
    db:               DbDep = None,
    current_user:     CurrentUser = None,
) -> CorrectResponse:
    """
    Apply a human correction to a misclassified transaction.

    1. Validates ``correct_category`` against ``CATEGORIES``.
    2. Updates ``Transaction.corrected_category``.
    3. Inserts a ``ModelCorrection`` record (``original_prediction`` /
       ``corrected_label``) for active learning.
    4. Schedules ``ActiveRetrainer.check_and_retrain()`` as a background task.

    Raises:
        HTTPException 400: ``correct_category`` is not a valid category.
        HTTPException 404: Transaction not found or belongs to another user.
    """
    if body.correct_category not in CATEGORIES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"'{body.correct_category}' is not a valid category. "
                f"Choose from: {CATEGORIES}"
            ),
        )

    txn = _own_transaction_or_404(transaction_id, current_user, db)
    original_prediction = txn.predicted_category

    txn.corrected_category = body.correct_category
    db.add(txn)

    correction = ModelCorrection(
        transaction_id      = txn.id,
        original_prediction = original_prediction or "unknown",
        corrected_label     = body.correct_category,
    )
    db.add(correction)
    db.commit()
    db.refresh(correction)

    log.info(
        "correction: user_id=%d txn_id=%d '%s' → '%s'",
        current_user.id, txn.id, original_prediction, body.correct_category,
    )

    # Fire-and-forget retrain check
    background_tasks.add_task(_background_retrain)

    return CorrectResponse(
        transaction_id       = txn.id,
        corrected_category   = body.correct_category,
        retraining_triggered = True,
        message              = "Correction saved. Retraining check scheduled.",
    )


# ── DELETE /transactions/{transaction_id} ─────────────────────────────────────

@router.delete(
    "/transactions/{transaction_id}",
    response_model=DeleteResponse,
    summary="Delete a transaction",
)
def delete_transaction(
    transaction_id: int,
    request:        Request,
    db:             DbDep = None,
    current_user:   CurrentUser = None,
) -> DeleteResponse:
    """
    Permanently remove a transaction owned by the current user.

    Cascade deletes any associated ``ModelCorrection`` rows automatically
    (configured via the ORM relationship in ``database.py``).

    Raises:
        HTTPException 404: Transaction not found or belongs to another user.
    """
    txn = _own_transaction_or_404(transaction_id, current_user, db)
    db.delete(txn)
    db.commit()

    log.info("delete: user_id=%d txn_id=%d", current_user.id, transaction_id)

    return DeleteResponse(
        transaction_id = transaction_id,
        message        = f"Transaction {transaction_id} deleted successfully.",
    )