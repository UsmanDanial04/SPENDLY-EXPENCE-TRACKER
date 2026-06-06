from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import get_db
from app.schemas.expense import SpendingSummary
from app.services.analytics_service import get_spending_summary

router = APIRouter()

@router.get("/summary", response_model=SpendingSummary)
async def spending_summary(db: AsyncSession = Depends(get_db)):
    return await get_spending_summary(db)
