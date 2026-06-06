from fastapi import FastAPI
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
