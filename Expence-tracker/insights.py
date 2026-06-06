"""
insights.py — Spending analytics router for the expense tracker.

All calculations are done with pandas on data fetched per-user from the DB.
Floats are rounded to 2 decimal places throughout.

Endpoints
---------
GET /insights/summary          Monthly spending summary with category breakdown
GET /insights/trends           Last-6-month totals with MoM change %
GET /insights/anomalies        Statistically unusual transactions (z-score > 2)
GET /insights/budget-status    Compare spending to per-category budget params
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Annotated, Literal, Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from database import SessionLocal, Transaction, User
from model_tfidf import CATEGORIES

log = logging.getLogger("insights")

router = APIRouter(prefix="/insights", tags=["insights"])


# ── DB + Auth dependencies ────────────────────────────────────────────────────

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


DbDep = Annotated[Session, Depends(get_db)]


def get_current_user(request: Request, db: DbDep) -> User:
    """Lazy-import _decode_token from main to avoid circular imports."""
    from main import _decode_token

    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = _decode_token(auth.removeprefix("Bearer ").strip())
    user = db.query(User).filter(User.id == int(payload["sub"])).first()
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


# ── Shared data loader ────────────────────────────────────────────────────────

def _load_user_transactions(user_id: int, db: Session) -> pd.DataFrame:
    """
    Fetch all transactions for *user_id* and return a cleaned DataFrame.

    The effective category is ``corrected_category`` when set, otherwise
    ``predicted_category``. Rows with no category at all are dropped.

    Columns: id, date, description, amount, category, user_id
    """
    rows = db.query(Transaction).filter(Transaction.user_id == user_id).all()
    if not rows:
        return pd.DataFrame(
            columns=["id", "date", "description", "amount", "category", "user_id"]
        )

    records = [
        {
            "id":          t.id,
            # Store as plain Python datetime — SQLite returns naive datetimes,
            # so we keep them naive throughout to avoid tz conversion errors.
            "date":        t.date if isinstance(t.date, __import__('datetime').datetime) else pd.Timestamp(t.date).to_pydatetime(),
            "description": t.description or "",
            "amount":      float(t.amount),
            "category":    t.corrected_category or t.predicted_category,
            "user_id":     t.user_id,
        }
        for t in rows
    ]
    df = pd.DataFrame(records)
    df = df.dropna(subset=["category"])
    # Use to_datetime without utc=True: dates from SQLite are naive,
    # so no timezone coercion is needed or safe.
    df["date"] = pd.to_datetime(df["date"])
    df["year_month"] = df["date"].dt.to_period("M")
    return df


def _parse_month(month: str) -> pd.Period:
    """Parse a YYYY-MM string into a pandas Period, raising HTTP 400 on failure."""
    try:
        return pd.Period(month, freq="M")
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid month format '{month}'. Use YYYY-MM.",
        )


def _current_month() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m")


def _r(value: float) -> float:
    """Round to 2 decimal places."""
    return round(value, 2)


# ── Response schemas ──────────────────────────────────────────────────────────

class CategorySummary(BaseModel):
    total:      float
    count:      int
    percentage: float   # % of total_spent


class SummaryResponse(BaseModel):
    month:           str
    total_spent:     float
    transaction_count: int
    avg_transaction: float
    top_merchant:    Optional[str]
    breakdown:       dict[str, CategorySummary]


class MonthTrend(BaseModel):
    month:       str
    total:       float
    count:       int
    mom_change:  Optional[float]  # month-over-month % change; None for first month
    by_category: dict[str, float]


class TrendsResponse(BaseModel):
    months: list[MonthTrend]


class TransactionBrief(BaseModel):
    id:          int
    date:        str
    description: str
    amount:      float
    category:    str


class AnomalyItem(BaseModel):
    transaction:    TransactionBrief
    zscore:         float
    category_mean:  float
    category_std:   float


class AnomaliesResponse(BaseModel):
    anomalies: list[AnomalyItem]
    checked_categories: int
    total_transactions:  int


class BudgetCategoryStatus(BaseModel):
    category:        str
    budget:          float
    spent:           float
    remaining:       float
    percentage_used: float
    status:          Literal["ok", "warning", "exceeded"]


class BudgetStatusResponse(BaseModel):
    month:      str
    categories: list[BudgetCategoryStatus]


# ── GET /insights/summary ─────────────────────────────────────────────────────

@router.get(
    "/summary",
    response_model=SummaryResponse,
    summary="Monthly spending summary with category breakdown",
)
def get_summary(
    request:      Request,
    db:           DbDep,
    current_user: CurrentUser,
    month: str = Query(
        default=None,
        description="Month to summarise — YYYY-MM. Defaults to the current month.",
        examples=["2024-03"],
    ),
) -> SummaryResponse:
    """
    Return total spending, per-category breakdown, top merchant, and average
    transaction size for a given calendar month.

    - ``breakdown[category].percentage`` is that category's share of
      ``total_spent`` (0–100).
    - ``top_merchant`` is the description with the highest single-month spend.
    - Returns zero-value totals (not 404) when there are no transactions.

    Args:
        month: YYYY-MM string; defaults to the current calendar month.

    Raises:
        HTTPException 400: Malformed month string.
    """
    target_month = _parse_month(month or _current_month())
    df = _load_user_transactions(current_user.id, db)

    # Filter to target month
    if df.empty or "year_month" not in df.columns:
        month_df = pd.DataFrame(columns=["id","date","description","amount","category"])
    else:
        month_df = df[df["year_month"] == target_month].copy()

    if month_df.empty:
        return SummaryResponse(
            month=str(target_month),
            total_spent=0.0,
            transaction_count=0,
            avg_transaction=0.0,
            top_merchant=None,
            breakdown={},
        )

    total_spent = month_df["amount"].sum()
    avg_txn     = month_df["amount"].mean()

    # Top merchant by total spend
    top_merchant = (
        month_df.groupby("description")["amount"]
        .sum()
        .idxmax()
    )

    # Category breakdown
    breakdown: dict[str, CategorySummary] = {}
    for cat, group in month_df.groupby("category"):
        cat_total = group["amount"].sum()
        breakdown[str(cat)] = CategorySummary(
            total      = _r(cat_total),
            count      = len(group),
            percentage = _r((cat_total / total_spent) * 100) if total_spent else 0.0,
        )

    log.info(
        "summary: user_id=%d month=%s total=%.2f categories=%d",
        current_user.id, target_month, total_spent, len(breakdown),
    )

    return SummaryResponse(
        month             = str(target_month),
        total_spent       = _r(total_spent),
        transaction_count = len(month_df),
        avg_transaction   = _r(avg_txn),
        top_merchant      = top_merchant,
        breakdown         = breakdown,
    )


# ── GET /insights/trends ──────────────────────────────────────────────────────

@router.get(
    "/trends",
    response_model=TrendsResponse,
    summary="Monthly spending totals for the last 6 months with MoM % change",
)
def get_trends(
    request:      Request,
    db:           DbDep,
    current_user: CurrentUser,
) -> TrendsResponse:
    """
    Return per-month spending totals for the trailing 6 calendar months
    (inclusive of the current month), along with a month-over-month
    percentage change.

    - ``mom_change`` is ``None`` for the earliest month in the window
      (no prior month to compare against).
    - ``mom_change`` is ``None`` when the previous month had zero spend
      (avoids division-by-zero).
    - ``by_category`` lists each category's total for that month; categories
      with no spend in a given month are omitted.
    """
    df = _load_user_transactions(current_user.id, db)

    # Build the 6-month window ending with the current month
    now          = pd.Timestamp.now()
    current_p    = pd.Period(now, freq="M")
    month_window = pd.period_range(end=current_p, periods=6, freq="M")

    monthly_records: list[MonthTrend] = []
    prev_total: Optional[float] = None

    for period in month_window:
        if df.empty or "year_month" not in df.columns:
            period_df = pd.DataFrame(columns=["amount", "category"])
        else:
            period_df = df[df["year_month"] == period]

        total = _r(float(period_df["amount"].sum())) if not period_df.empty else 0.0
        count = len(period_df)

        # Month-over-month change
        if prev_total is None:
            mom_change = None
        elif prev_total == 0.0:
            mom_change = None   # infinite / undefined
        else:
            mom_change = _r(((total - prev_total) / prev_total) * 100)

        # Per-category totals for this month
        by_category: dict[str, float] = {}
        if not period_df.empty:
            for cat, grp in period_df.groupby("category"):
                by_category[str(cat)] = _r(grp["amount"].sum())

        monthly_records.append(
            MonthTrend(
                month      = str(period),
                total      = total,
                count      = count,
                mom_change = mom_change,
                by_category= by_category,
            )
        )
        prev_total = total

    log.info("trends: user_id=%d months=%d", current_user.id, len(monthly_records))
    return TrendsResponse(months=monthly_records)


# ── GET /insights/anomalies ───────────────────────────────────────────────────

@router.get(
    "/anomalies",
    response_model=AnomaliesResponse,
    summary="Transactions with unusually high amounts (z-score > 2 within category)",
)
def get_anomalies(
    request:      Request,
    db:           DbDep,
    current_user: CurrentUser,
) -> AnomaliesResponse:
    """
    Detect statistically unusual transactions using per-category z-scores.

    A transaction is flagged when its amount is more than **2 standard
    deviations above** the category mean.  Returns up to 5 of the most
    recent flagged transactions.

    - Categories with fewer than 3 transactions are skipped (not enough
      data for a meaningful standard deviation).
    - ``zscore`` is always positive (we only flag high outliers).
    - ``category_mean`` and ``category_std`` are calculated across *all* of
      the user's transactions in that category, not just the current month.
    """
    df = _load_user_transactions(current_user.id, db)

    if df.empty:
        return AnomaliesResponse(anomalies=[], checked_categories=0, total_transactions=0)

    anomalies: list[AnomalyItem] = []
    checked_categories = 0

    for cat, group in df.groupby("category"):
        if len(group) < 3:
            # Need at least 3 data points for std to be meaningful
            continue
        checked_categories += 1

        cat_mean = group["amount"].mean()
        cat_std  = group["amount"].std(ddof=1)

        if cat_std == 0:
            continue

        zscores = (group["amount"] - cat_mean) / cat_std
        flagged = group[zscores > 2.0].copy()
        flagged["zscore"] = zscores[zscores > 2.0]

        for _, row in flagged.iterrows():
            anomalies.append(
                AnomalyItem(
                    transaction=TransactionBrief(
                        id          = int(row["id"]),
                        date        = row["date"].strftime("%Y-%m-%d"),
                        description = str(row["description"]),
                        amount      = _r(float(row["amount"])),
                        category    = str(cat),
                    ),
                    zscore        = _r(float(row["zscore"])),
                    category_mean = _r(cat_mean),
                    category_std  = _r(cat_std),
                )
            )

    # Sort by recency, return at most 5
    anomalies.sort(key=lambda a: a.transaction.date, reverse=True)
    anomalies = anomalies[:5]

    log.info(
        "anomalies: user_id=%d flagged=%d checked_categories=%d",
        current_user.id, len(anomalies), checked_categories,
    )

    return AnomaliesResponse(
        anomalies           = anomalies,
        checked_categories  = checked_categories,
        total_transactions  = len(df),
    )


# ── GET /insights/budget-status ───────────────────────────────────────────────

_WARNING_THRESHOLD = 0.80   # 80 % used → "warning"

@router.get(
    "/budget-status",
    response_model=BudgetStatusResponse,
    summary="Compare current-month spending against per-category budgets",
)
def get_budget_status(
    request:      Request,
    db:           DbDep,
    current_user: CurrentUser,
    month: str = Query(
        default=None,
        description="Month to evaluate — YYYY-MM. Defaults to the current month.",
    ),
    # One query param per category — all optional
    Food:          Optional[float] = Query(None, gt=0, description="Food budget"),
    Transport:     Optional[float] = Query(None, gt=0, description="Transport budget"),
    Housing:       Optional[float] = Query(None, gt=0, description="Housing budget"),
    Entertainment: Optional[float] = Query(None, gt=0, description="Entertainment budget"),
    Healthcare:    Optional[float] = Query(None, gt=0, description="Healthcare budget"),
    Shopping:      Optional[float] = Query(None, gt=0, description="Shopping budget"),
    Education:     Optional[float] = Query(None, gt=0, description="Education budget"),
    Utilities:     Optional[float] = Query(None, gt=0, description="Utilities budget"),
) -> BudgetStatusResponse:
    """
    Return how much of each provided budget has been used in the given month.

    Only categories for which a budget query parameter was supplied are
    included in the response.

    Status thresholds:

    - ``ok``       — spent < 80 % of budget
    - ``warning``  — 80 % ≤ spent < 100 % of budget
    - ``exceeded`` — spent ≥ 100 % of budget

    Args:
        month: YYYY-MM string; defaults to the current calendar month.
        Food, Transport, …: Budget amounts per category (all optional, must be > 0).

    Raises:
        HTTPException 400: No budgets provided or malformed month.
    """
    # Collect whichever category budgets were provided
    raw_budgets: dict[str, float] = {
        cat: val
        for cat, val in {
            "Food":          Food,
            "Transport":     Transport,
            "Housing":       Housing,
            "Entertainment": Entertainment,
            "Healthcare":    Healthcare,
            "Shopping":      Shopping,
            "Education":     Education,
            "Utilities":     Utilities,
        }.items()
        if val is not None
    }

    if not raw_budgets:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Provide at least one budget query parameter. "
                f"Available: {', '.join(CATEGORIES)}"
            ),
        )

    target_month = _parse_month(month or _current_month())
    df = _load_user_transactions(current_user.id, db)

    # Spending for target month
    if df.empty or "year_month" not in df.columns:
        month_df = pd.DataFrame(columns=["amount", "category"])
    else:
        month_df = df[df["year_month"] == target_month]

    # Aggregate spend per category for the month
    if month_df.empty:
        cat_totals: dict[str, float] = {}
    else:
        cat_totals = (
            month_df.groupby("category")["amount"]
            .sum()
            .to_dict()
        )

    statuses: list[BudgetCategoryStatus] = []
    for cat, budget in raw_budgets.items():
        spent     = _r(cat_totals.get(cat, 0.0))
        remaining = _r(budget - spent)
        pct_used  = _r((spent / budget) * 100) if budget else 0.0

        if spent >= budget:
            label: Literal["ok", "warning", "exceeded"] = "exceeded"
        elif spent >= budget * _WARNING_THRESHOLD:
            label = "warning"
        else:
            label = "ok"

        statuses.append(
            BudgetCategoryStatus(
                category       = cat,
                budget         = _r(budget),
                spent          = spent,
                remaining      = remaining,
                percentage_used= pct_used,
                status         = label,
            )
        )

    # Sort: exceeded → warning → ok, then alphabetically within each group
    _order = {"exceeded": 0, "warning": 1, "ok": 2}
    statuses.sort(key=lambda s: (_order[s.status], s.category))

    log.info(
        "budget-status: user_id=%d month=%s categories=%d",
        current_user.id, target_month, len(statuses),
    )

    return BudgetStatusResponse(
        month      = str(target_month),
        categories = statuses,
    )
