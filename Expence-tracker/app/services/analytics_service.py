from collections import defaultdict
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
