"""
Run this from the root of your expense-tracker folder:
    python setup_windows.py
"""
import os

FILES = {}

FILES["main.py"] = '''"""
Expense Tracker – application entry point.
Run with: uvicorn main:app --reload
"""
from app.core.app_factory import create_app

app = create_app()
'''

FILES["requirements.txt"] = '''fastapi==0.111.0
uvicorn[standard]==0.29.0
python-multipart==0.0.9
sqlalchemy==2.0.30
aiosqlite==0.20.0
pandas==2.2.2
scikit-learn==1.4.2
transformers==4.41.1
torch==2.3.0
shap==0.45.1
joblib==1.4.2
python-dotenv==1.0.1
pydantic==2.7.1
pydantic-settings==2.2.1
httpx==0.27.0
pytest==8.2.0
pytest-asyncio==0.23.6
'''

FILES[".env.example"] = '''APP_NAME="Expense Tracker"
APP_VERSION="1.0.0"
DATABASE_URL="sqlite+aiosqlite:///./data/expense_tracker.db"
ALLOWED_ORIGINS=["http://localhost:8000"]
MODEL_PATH="data/models/expense_classifier.joblib"
LABEL_ENCODER_PATH="data/models/label_encoder.joblib"
MIN_CONFIDENCE=0.6
'''

FILES[".gitignore"] = '''__pycache__/
*.py[cod]
.venv/
venv/
.env
*.db
*.sqlite
*.sqlite3
data/raw/*
data/processed/*
data/models/*
!data/raw/.gitkeep
!data/processed/.gitkeep
!data/models/.gitkeep
*.pkl
*.joblib
*.pt
.pytest_cache/
.coverage
.vscode/
.idea/
*.log
'''

FILES["pytest.ini"] = '''[pytest]
asyncio_mode = auto
testpaths = tests
python_files = test_*.py
python_functions = test_*
'''

FILES["app/__init__.py"] = ""
FILES["app/api/__init__.py"] = ""
FILES["app/api/routes/__init__.py"] = ""
FILES["app/core/__init__.py"] = ""
FILES["app/db/__init__.py"] = ""
FILES["app/ml/__init__.py"] = ""
FILES["app/ml/models/__init__.py"] = ""
FILES["app/ml/utils/__init__.py"] = ""
FILES["app/schemas/__init__.py"] = ""
FILES["app/services/__init__.py"] = ""
FILES["tests/__init__.py"] = ""
FILES["tests/unit/__init__.py"] = ""
FILES["tests/integration/__init__.py"] = ""
FILES["data/raw/.gitkeep"] = ""
FILES["data/processed/.gitkeep"] = ""
FILES["data/models/.gitkeep"] = ""

FILES["app/core/config.py"] = '''from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "Expense Tracker"
    APP_VERSION: str = "1.0.0"
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/expense_tracker.db"
    ALLOWED_ORIGINS: list[str] = ["http://localhost:8000", "http://127.0.0.1:8000"]
    MODEL_PATH: str = "data/models/expense_classifier.joblib"
    LABEL_ENCODER_PATH: str = "data/models/label_encoder.joblib"
    MIN_CONFIDENCE: float = 0.6

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
'''

FILES["app/core/app_factory.py"] = '''from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.core.config import settings
from app.db.session import init_db
from app.api.routes import expenses, categories, analytics, ml

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="AI-powered expense tracker with automatic category classification.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(expenses.router,   prefix="/api/v1/expenses",    tags=["Expenses"])
    app.include_router(categories.router, prefix="/api/v1/categories",  tags=["Categories"])
    app.include_router(analytics.router,  prefix="/api/v1/analytics",   tags=["Analytics"])
    app.include_router(ml.router,         prefix="/api/v1/ml",          tags=["ML"])
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

    @app.on_event("startup")
    async def startup():
        await init_db()

    return app
'''

FILES["app/db/session.py"] = '''from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from app.core.config import settings

engine = create_async_engine(settings.DATABASE_URL, echo=False, future=True)
AsyncSessionLocal = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

class Base(DeclarativeBase):
    pass

async def init_db() -> None:
    from app.db import models  # noqa
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
'''

FILES["app/db/models.py"] = '''from datetime import datetime
from sqlalchemy import String, Float, DateTime, ForeignKey, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.session import Base

class Category(Base):
    __tablename__ = "categories"
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    icon: Mapped[str] = mapped_column(String(10), default="💰")
    color: Mapped[str] = mapped_column(String(7), default="#6366f1")
    expenses: Mapped[list["Expense"]] = relationship(back_populates="category")

class Expense(Base):
    __tablename__ = "expenses"
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    description: Mapped[str] = mapped_column(String(255), nullable=False)
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    currency: Mapped[str] = mapped_column(String(3), default="USD")
    date: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    category_id: Mapped[int | None] = mapped_column(ForeignKey("categories.id"), nullable=True)
    ml_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_ml_classified: Mapped[bool] = mapped_column(default=False)
    category: Mapped["Category | None"] = relationship(back_populates="expenses")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())
'''

FILES["app/schemas/expense.py"] = '''from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict

class CategoryBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    icon: str = Field(default="💰", max_length=10)
    color: str = Field(default="#6366f1", pattern=r"^#[0-9A-Fa-f]{6}$")

class CategoryCreate(CategoryBase):
    pass

class CategoryOut(CategoryBase):
    id: int
    model_config = ConfigDict(from_attributes=True)

class ExpenseBase(BaseModel):
    description: str = Field(..., min_length=1, max_length=255)
    amount: float = Field(..., gt=0)
    currency: str = Field(default="USD", max_length=3)
    date: datetime = Field(default_factory=datetime.utcnow)
    notes: str | None = None
    category_id: int | None = None

class ExpenseCreate(ExpenseBase):
    pass

class ExpenseUpdate(BaseModel):
    description: str | None = None
    amount: float | None = Field(default=None, gt=0)
    currency: str | None = None
    date: datetime | None = None
    notes: str | None = None
    category_id: int | None = None

class ExpenseOut(ExpenseBase):
    id: int
    ml_confidence: float | None = None
    is_ml_classified: bool
    category: CategoryOut | None = None
    created_at: datetime
    updated_at: datetime
    model_config = ConfigDict(from_attributes=True)

class ClassifyRequest(BaseModel):
    description: str = Field(..., min_length=1)

class ClassifyResponse(BaseModel):
    category: str
    confidence: float
    shap_values: dict[str, float] | None = None

class SpendingSummary(BaseModel):
    total: float
    by_category: dict[str, float]
    by_month: dict[str, float]
    top_expense: ExpenseOut | None
'''

FILES["app/ml/utils/preprocessing.py"] = '''import re, string

_CURRENCY_RE  = re.compile(r"[\\$£€¥₹]\\s*\\d[\\d,\\.]*|\\d[\\d,\\.]*\\s*[\\$£€¥₹]")
_NUMBER_RE    = re.compile(r"\\b\\d+[\\d,\\.]*\\b")
_WHITESPACE_RE = re.compile(r"\\s+")

def clean_description(text: str) -> str:
    text = text.lower()
    text = _CURRENCY_RE.sub(" ", text)
    text = _NUMBER_RE.sub(" ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text

def build_feature_text(description: str, notes: str = "") -> str:
    parts = [clean_description(description)]
    if notes:
        parts.append(clean_description(notes))
    return " ".join(parts)
'''

FILES["app/ml/models/classifier.py"] = '''from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

def build_tfidf_pipeline(C: float = 5.0, max_features: int = 20_000) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=max_features, sublinear_tf=True, strip_accents="unicode")),
        ("clf",   LogisticRegression(C=C, max_iter=1000, class_weight="balanced", solver="lbfgs", multi_class="multinomial")),
    ])

def build_label_encoder(categories: list[str]) -> LabelEncoder:
    le = LabelEncoder()
    le.fit(categories)
    return le
'''

FILES["app/ml/utils/predictor.py"] = '''from __future__ import annotations
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
'''

FILES["app/services/expense_service.py"] = '''from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.db.models import Expense, Category
from app.schemas.expense import ExpenseCreate, ExpenseUpdate
from app.ml.utils.predictor import classify_expense
from app.core.config import settings

async def get_all_expenses(db: AsyncSession) -> list[Expense]:
    result = await db.execute(select(Expense).order_by(Expense.date.desc()))
    return list(result.scalars().all())

async def get_expense_by_id(db: AsyncSession, expense_id: int) -> Expense | None:
    return await db.get(Expense, expense_id)

async def create_expense(db: AsyncSession, payload: ExpenseCreate) -> Expense:
    expense = Expense(**payload.model_dump())
    if expense.category_id is None:
        prediction = classify_expense(expense.description)
        if prediction and prediction["confidence"] >= settings.MIN_CONFIDENCE:
            cat = await _get_or_create_category(db, prediction["category"])
            expense.category_id    = cat.id
            expense.ml_confidence  = prediction["confidence"]
            expense.is_ml_classified = True
    db.add(expense)
    await db.flush()
    await db.refresh(expense)
    return expense

async def update_expense(db: AsyncSession, expense_id: int, payload: ExpenseUpdate) -> Expense | None:
    expense = await get_expense_by_id(db, expense_id)
    if not expense:
        return None
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(expense, field, value)
    await db.flush()
    await db.refresh(expense)
    return expense

async def delete_expense(db: AsyncSession, expense_id: int) -> bool:
    expense = await get_expense_by_id(db, expense_id)
    if not expense:
        return False
    await db.delete(expense)
    return True

async def _get_or_create_category(db: AsyncSession, name: str) -> Category:
    result = await db.execute(select(Category).where(Category.name == name))
    cat = result.scalar_one_or_none()
    if not cat:
        cat = Category(name=name)
        db.add(cat)
        await db.flush()
    return cat
'''

FILES["app/services/analytics_service.py"] = '''from collections import defaultdict
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.db.models import Expense
from app.schemas.expense import SpendingSummary, ExpenseOut

async def get_spending_summary(db: AsyncSession) -> SpendingSummary:
    result   = await db.execute(select(Expense).order_by(Expense.amount.desc()))
    expenses = list(result.scalars().all())
    total        = sum(e.amount for e in expenses)
    by_category  = defaultdict(float)
    by_month     = defaultdict(float)
    for e in expenses:
        cat_name = e.category.name if e.category else "Uncategorised"
        by_category[cat_name] += e.amount
        by_month[e.date.strftime("%Y-%m")] += e.amount
    top = ExpenseOut.model_validate(expenses[0]) if expenses else None
    return SpendingSummary(
        total=round(total, 2),
        by_category={k: round(v, 2) for k, v in by_category.items()},
        by_month=dict(sorted(by_month.items())),
        top_expense=top,
    )
'''

FILES["app/api/routes/expenses.py"] = '''from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import get_db
from app.schemas.expense import ExpenseCreate, ExpenseUpdate, ExpenseOut
from app.services import expense_service

router = APIRouter()

@router.get("/", response_model=list[ExpenseOut])
async def list_expenses(db: AsyncSession = Depends(get_db)):
    return await expense_service.get_all_expenses(db)

@router.get("/{expense_id}", response_model=ExpenseOut)
async def get_expense(expense_id: int, db: AsyncSession = Depends(get_db)):
    expense = await expense_service.get_expense_by_id(db, expense_id)
    if not expense:
        raise HTTPException(status_code=404, detail="Expense not found")
    return expense

@router.post("/", response_model=ExpenseOut, status_code=status.HTTP_201_CREATED)
async def create_expense(payload: ExpenseCreate, db: AsyncSession = Depends(get_db)):
    return await expense_service.create_expense(db, payload)

@router.patch("/{expense_id}", response_model=ExpenseOut)
async def update_expense(expense_id: int, payload: ExpenseUpdate, db: AsyncSession = Depends(get_db)):
    expense = await expense_service.update_expense(db, expense_id, payload)
    if not expense:
        raise HTTPException(status_code=404, detail="Expense not found")
    return expense

@router.delete("/{expense_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_expense(expense_id: int, db: AsyncSession = Depends(get_db)):
    if not await expense_service.delete_expense(db, expense_id):
        raise HTTPException(status_code=404, detail="Expense not found")
'''

FILES["app/api/routes/categories.py"] = '''from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.db.session import get_db
from app.db.models import Category
from app.schemas.expense import CategoryCreate, CategoryOut

router = APIRouter()

@router.get("/", response_model=list[CategoryOut])
async def list_categories(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Category).order_by(Category.name))
    return list(result.scalars().all())

@router.post("/", response_model=CategoryOut, status_code=201)
async def create_category(payload: CategoryCreate, db: AsyncSession = Depends(get_db)):
    cat = Category(**payload.model_dump())
    db.add(cat)
    await db.flush()
    await db.refresh(cat)
    return cat

@router.delete("/{category_id}", status_code=204)
async def delete_category(category_id: int, db: AsyncSession = Depends(get_db)):
    cat = await db.get(Category, category_id)
    if not cat:
        raise HTTPException(status_code=404, detail="Category not found")
    await db.delete(cat)
'''

FILES["app/api/routes/analytics.py"] = '''from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import get_db
from app.schemas.expense import SpendingSummary
from app.services.analytics_service import get_spending_summary

router = APIRouter()

@router.get("/summary", response_model=SpendingSummary)
async def spending_summary(db: AsyncSession = Depends(get_db)):
    return await get_spending_summary(db)
'''

FILES["app/api/routes/ml.py"] = '''from fastapi import APIRouter, HTTPException
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
'''

FILES["scripts/train_model.py"] = '''#!/usr/bin/env python3
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
    ("McDonald\'s lunch", "Dining"),
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
'''

FILES["scripts/seed_db.py"] = '''#!/usr/bin/env python3
import asyncio, sys, random
from pathlib import Path
from datetime import datetime, timedelta
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.session import AsyncSessionLocal, init_db
from app.db.models import Category, Expense

CATEGORIES = [
    {"name": "Groceries",     "icon": "🛒", "color": "#22c55e"},
    {"name": "Dining",        "icon": "🍽️", "color": "#f97316"},
    {"name": "Transport",     "icon": "🚗", "color": "#3b82f6"},
    {"name": "Utilities",     "icon": "⚡", "color": "#eab308"},
    {"name": "Entertainment", "icon": "🎬", "color": "#a855f7"},
    {"name": "Health",        "icon": "💊", "color": "#ef4444"},
    {"name": "Software",      "icon": "💻", "color": "#06b6d4"},
    {"name": "Travel",        "icon": "✈️", "color": "#8b5cf6"},
    {"name": "Shopping",      "icon": "🛍️", "color": "#ec4899"},
]

SAMPLE_EXPENSES = [
    ("Whole Foods weekly shop", 87.50, "Groceries"),
    ("Spotify Premium",          9.99, "Entertainment"),
    ("Uber to downtown",        14.20, "Transport"),
    ("Electric bill",           65.00, "Utilities"),
    ("Dinner at restaurant",   120.00, "Dining"),
    ("AWS invoice",             43.12, "Software"),
    ("Flight booking",         310.00, "Travel"),
    ("Nike sneakers",           95.00, "Shopping"),
    ("Gym membership",          45.00, "Health"),
    ("Netflix",                 15.99, "Entertainment"),
]

async def seed():
    await init_db()
    async with AsyncSessionLocal() as db:
        cat_map = {}
        for c in CATEGORIES:
            cat = Category(**c)
            db.add(cat)
            await db.flush()
            cat_map[c["name"]] = cat.id
        for desc, amount, cat_name in SAMPLE_EXPENSES:
            db.add(Expense(
                description=desc, amount=amount,
                category_id=cat_map[cat_name],
                date=datetime.utcnow() - timedelta(days=random.randint(0, 60)),
                is_ml_classified=False,
            ))
        await db.commit()
        print(f"Seeded {len(CATEGORIES)} categories and {len(SAMPLE_EXPENSES)} expenses.")

if __name__ == "__main__":
    asyncio.run(seed())
'''

from pathlib import Path

FILES["frontend/index.html"] = open("frontend/index.html").read()
FILES["frontend/css/main.css"] = open("frontend/css/main.css").read()
FILES["frontend/js/api.js"] = open("frontend/js/api.js").read()
FILES["frontend/js/charts.js"] = open("frontend/js/charts.js").read()
FILES["frontend/js/app.js"] = open("frontend/js/app.js").read()

for path, content in FILES.items():
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    print(f"  created {path}")

print("\nDone! All files created.")
