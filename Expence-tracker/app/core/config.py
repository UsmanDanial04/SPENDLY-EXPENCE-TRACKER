from pydantic_settings import BaseSettings

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
