from datetime import datetime
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
