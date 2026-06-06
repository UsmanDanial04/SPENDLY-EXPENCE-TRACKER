#!/usr/bin/env python3
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
