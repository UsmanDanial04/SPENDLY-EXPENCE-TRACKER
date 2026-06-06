from sqlalchemy.ext.asyncio import AsyncSession
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
